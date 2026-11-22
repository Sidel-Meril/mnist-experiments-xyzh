"""Progressive growing GAN, training, and the DVC checkpoint.

The generator and discriminator follow the progressive-growing schedule: both
networks start at 4x4 and gain a block each time the image doubles, up to the
CIFAR-10 size of 32x32. A new block is faded in with ``alpha``. At ``alpha`` 0
the output matches the previous resolution (nearest-upsampled on the generator,
average-pooled on the discriminator). At ``alpha`` 1 only the new block is
used. Pixel normalization, equalized learning-rate convolutions, and a
minibatch standard-deviation channel are the stability pieces from that
schedule. The loss stays ``BCELoss`` with one discriminator step and one
generator step, matching the other GAN scaffolds. Real images are average-pooled
down to the current resolution.

``ProgressiveGAN.load`` uses a local checkpoint when one is already on disk.
Otherwise the latest fit is read from the DVC remote in ``.dvc/config`` with
``dvc.api.open`` and cached by md5. An ``s3://`` remote needs
``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mnist_pg_gan.config import PREVIEW_COUNT, ROOT
from mnist_pg_gan.figures import save_grid, save_growth_rows, unit_interval

_MODEL_REPO = "https://github.com/Sidel-Meril/mnist-pg-gan-xyzh.git"
_MODEL_REMOTE = "demo"
_MODEL_RELATIVE = "models/mnist_pg_gan/mnist_pg_gan.pt"
LOCAL_MODEL = ROOT / "models" / "mnist_pg_gan" / "mnist_pg_gan.pt"
_session: tuple[str, str | None, Path] | None = None

_ARCH_KEYS = ("noise_dim", "width", "image_size", "channels")
_HPARAM_KEYS = (*_ARCH_KEYS, "lr_g", "lr_d", "beta1", "batch_size")
_GROWTH_PREVIEW = 8


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolutions_for(image_size: int) -> tuple[int, ...]:
    """Power-of-two resolutions from 4 up to ``image_size``."""
    if image_size < 4 or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two and at least 4")
    sizes: list[int] = []
    size = 4
    while size <= image_size:
        sizes.append(size)
        size *= 2
    return tuple(sizes)


def stage_channels(width: int, n_stages: int) -> list[int]:
    """Channel counts per resolution. Low resolutions keep the widest maps."""
    if width < 4:
        raise ValueError("width must be at least 4")
    if n_stages < 1:
        raise ValueError("n_stages must be at least 1")
    channels: list[int] = []
    current = width
    for index in range(n_stages):
        channels.append(current)
        if index >= 1:
            current = max(current // 2, 4)
    return channels


def _batch_std(images: torch.Tensor) -> float:
    """Mean per-pixel deviation across the batch. Identical images score near 0."""
    flat = images.detach().reshape(images.shape[0], -1).float()
    if flat.shape[0] < 2:
        return 0.0
    return float(flat.std(dim=0, unbiased=False).mean().cpu())


def _to_resolution(images: torch.Tensor, resolution: int) -> torch.Tensor:
    height = int(images.shape[-1])
    width = int(images.shape[-2])
    if height == resolution and width == resolution:
        return images
    divisible = height > resolution and width > resolution
    divisible = divisible and height % resolution == 0 and width % resolution == 0
    if divisible and height % 2 == 0 and width % 2 == 0:
        pooled = images
        while int(pooled.shape[-1]) > resolution:
            pooled = F.avg_pool2d(pooled, kernel_size=2, stride=2)
        return pooled
    return F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )


class PixelNorm(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(features.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        return features * scale


class EqualConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, padding: int = 0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = (2.0 / (in_channels * kernel * kernel)) ** 0.5
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.conv2d(features, self.weight * self.scale, self.bias, padding=self.padding)


class EqualLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (2.0 / in_dim) ** 0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weight * self.scale, self.bias)


class FirstBlock(nn.Module):
    def __init__(self, noise_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.project = EqualLinear(noise_dim, channels * 4 * 4)
        self.conv = EqualConv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.norm = PixelNorm()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        hidden = self.project(noise).view(noise.shape[0], self.channels, 4, 4)
        hidden = self.norm(self.act(hidden))
        return self.norm(self.act(self.conv(hidden)))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            EqualConv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            EqualConv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            EqualConv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            EqualConv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FromRGB(nn.Module):
    def __init__(self, channels: int, image_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            EqualConv2d(image_channels, channels, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class DiscHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = EqualConv2d(channels + 1, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.fc = EqualLinear(channels * 4 * 4, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        deviation = torch.sqrt(features.var(dim=0, unbiased=False) + 1e-8).mean()
        stats = deviation.view(1, 1, 1, 1).expand(features.shape[0], 1, *features.shape[-2:])
        hidden = self.act(self.conv(torch.cat([features, stats], dim=1)))
        logits = self.fc(hidden.flatten(1))
        return torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)


class GrowingGenerator(nn.Module):
    def __init__(self, noise_dim: int, width: int, image_size: int, image_channels: int) -> None:
        super().__init__()
        self.resolutions = resolutions_for(image_size)
        channels = stage_channels(width, len(self.resolutions))
        self.blocks = nn.ModuleList([FirstBlock(noise_dim, channels[0])])
        self.to_rgb = nn.ModuleList([EqualConv2d(channels[0], image_channels, 1)])
        for index in range(1, len(self.resolutions)):
            self.blocks.append(UpBlock(channels[index - 1], channels[index]))
            self.to_rgb.append(EqualConv2d(channels[index], image_channels, 1))

    def forward(self, noise: torch.Tensor, stage: int, alpha: float) -> torch.Tensor:
        if stage < 0 or stage >= len(self.resolutions):
            raise ValueError(f"stage {stage} is outside 0..{len(self.resolutions) - 1}")
        hidden = self.blocks[0](noise)
        rgb = torch.tanh(self.to_rgb[0](hidden))
        blend = min(1.0, max(0.0, float(alpha)))
        for index in range(1, stage + 1):
            hidden = self.blocks[index](hidden)
            new_rgb = torch.tanh(self.to_rgb[index](hidden))
            rgb = F.interpolate(rgb, scale_factor=2, mode="nearest")
            if index == stage:
                rgb = (1.0 - blend) * rgb + blend * new_rgb
            else:
                rgb = new_rgb
        return rgb


class GrowingDiscriminator(nn.Module):
    def __init__(self, width: int, image_size: int, image_channels: int) -> None:
        super().__init__()
        self.resolutions = resolutions_for(image_size)
        channels = stage_channels(width, len(self.resolutions))
        self.from_rgb = nn.ModuleList([FromRGB(count, image_channels) for count in channels])
        self.down = nn.ModuleList([nn.Identity()])
        for index in range(1, len(self.resolutions)):
            self.down.append(DownBlock(channels[index], channels[index - 1]))
        self.head = DiscHead(channels[0])

    def forward(self, images: torch.Tensor, stage: int, alpha: float) -> torch.Tensor:
        if stage < 0 or stage >= len(self.resolutions):
            raise ValueError(f"stage {stage} is outside 0..{len(self.resolutions) - 1}")
        expected = self.resolutions[stage]
        if int(images.shape[-1]) != expected or int(images.shape[-2]) != expected:
            raise ValueError(
                f"stage {stage} expects {expected}px images, got {tuple(images.shape[-2:])}"
            )
        if stage == 0:
            hidden = self.from_rgb[0](images)
            return self.head(hidden)
        blend = min(1.0, max(0.0, float(alpha)))
        hidden = self.down[stage](self.from_rgb[stage](images))
        if blend < 1.0:
            pooled = F.avg_pool2d(images, kernel_size=2, stride=2)
            previous = self.from_rgb[stage - 1](pooled)
            hidden = blend * hidden + (1.0 - blend) * previous
        for index in range(stage - 1, 0, -1):
            hidden = self.down[index](hidden)
        return self.head(hidden)


class ProgressiveGAN(nn.Module):
    """Generator plus discriminator that grow from 4px to ``image_size``.

    ``load`` rebuilds this class from a local checkpoint or the DVC remote.
    ``save`` writes ``state_dict``, ``columns``, and ``hparams``.
    """

    def __init__(self, noise_dim: int, width: int, image_size: int, channels: int) -> None:
        super().__init__()
        if noise_dim < 1:
            raise ValueError("noise_dim must be at least 1")
        if channels < 1:
            raise ValueError("channels must be at least 1")
        self.noise_dim = int(noise_dim)
        self.width = int(width)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.resolutions = resolutions_for(self.image_size)
        self.generator = GrowingGenerator(
            self.noise_dim, self.width, self.image_size, self.channels
        )
        self.discriminator = GrowingDiscriminator(self.width, self.image_size, self.channels)
        self.register_buffer("fixed_noise", torch.zeros(PREVIEW_COUNT, self.noise_dim))
        self._stage = len(self.resolutions) - 1
        self._alpha = 1.0

    @classmethod
    def from_hparams(cls, hparams: dict[str, float | int]) -> ProgressiveGAN:
        """Build the architecture stored in a tune result or a checkpoint."""
        return cls(
            noise_dim=int(hparams["noise_dim"]),
            width=int(hparams["width"]),
            image_size=int(hparams["image_size"]),
            channels=int(hparams["channels"]),
        )

    def _reset(self) -> None:
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None and module is not self:
                reset()

    def _use(self, stage: int, alpha: float) -> None:
        self._stage = int(stage)
        self._alpha = float(alpha)

    def _full_resolution(self) -> None:
        self._use(len(self.resolutions) - 1, 1.0)

    @staticmethod
    def data_loaders(
        train_x: np.ndarray,
        eval_x: np.ndarray,
        *,
        batch_size: int,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Shuffled train loader and an ordered eval loader of real images."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {"pin_memory": True} if torch.cuda.is_available() else {}
        train_loader = DataLoader(
            _image_set(train_x),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        )
        eval_loader = DataLoader(
            _image_set(eval_x),
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, eval_loader

    def fit_from_dataset(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        hparams: dict[str, float | int],
        *,
        epochs: int,
        seed: int,
        generations_dir: Path | None = None,
        snapshot_every: int = 1,
    ) -> dict[str, list[float]]:
        """Grow from 4px to full resolution. ``epochs`` is the length of each phase.

        The first resolution trains stable for ``epochs``. Each larger resolution
        fades the new block in for ``epochs``, then trains stable for ``epochs``.
        A fixed noise batch is drawn once so grids show the same latents growing.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be at least 1")
        if len(train_loader) == 0 or len(eval_loader) == 0:
            raise ValueError("train and eval splits must both be non-empty")
        seed_all(seed)
        self.cpu()
        self._reset()
        self.fixed_noise.copy_(torch.randn(self.fixed_noise.shape))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.BCELoss()
        generator_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=float(hparams["lr_g"]),
            betas=(float(hparams["beta1"]), 0.999),
        )
        discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(hparams["lr_d"]),
            betas=(float(hparams["beta1"]), 0.999),
        )
        history: dict[str, list[float]] = {
            "train_discriminator": [],
            "train_generator": [],
            "eval_discriminator": [],
            "eval_generator": [],
            "sample_std": [],
            "resolution": [],
            "alpha": [],
            "stage": [],
        }
        phases = _phases(len(self.resolutions), epochs)
        total = sum(int(phase["epochs"]) for phase in phases)
        global_epoch = 0
        rows: dict[tuple[int, str], np.ndarray] = {}
        row_order: list[tuple[int, str]] = []
        frames: list[dict[str, float | int | str]] = []
        for phase in phases:
            for epoch_in_phase in range(int(phase["epochs"])):
                alpha = _blend(str(phase["phase"]), epoch_in_phase, int(phase["epochs"]))
                self._use(int(phase["stage"]), alpha)
                train_d, train_g = self._epoch(
                    train_loader,
                    device,
                    criterion,
                    discriminator_opt,
                    generator_opt,
                )
                eval_d, eval_g = self._score_loader(eval_loader, device, criterion)
                global_epoch += 1
                sample_std = self._snapshot(
                    device,
                    generations_dir,
                    global_epoch,
                    snapshot_every,
                    int(phase["resolution"]),
                    str(phase["phase"]),
                    rows,
                    row_order,
                    frames,
                )
                history["train_discriminator"].append(train_d)
                history["train_generator"].append(train_g)
                history["eval_discriminator"].append(eval_d)
                history["eval_generator"].append(eval_g)
                history["sample_std"].append(sample_std)
                history["resolution"].append(float(phase["resolution"]))
                history["alpha"].append(alpha)
                history["stage"].append(float(phase["stage"]))
                print(
                    f"epoch {global_epoch}/{total} "
                    f"res={int(phase['resolution'])} {phase['phase']} alpha={alpha:.2f} "
                    f"d={train_d:.4f} g={train_g:.4f} sample_std={sample_std:.4f}"
                )
        if generations_dir is not None and row_order:
            captions = [
                f"{self.resolutions[stage]}px {phase_name}" for stage, phase_name in row_order
            ]
            save_growth_rows(
                [(rows[key], caption) for key, caption in zip(row_order, captions, strict=True)],
                generations_dir / "growth_timeline.png",
                display_size=self.image_size,
            )
            (generations_dir / "growth.json").write_text(
                json.dumps({"display_size": self.image_size, "frames": frames}, indent=2),
                encoding="utf-8",
            )
        self._full_resolution()
        self.eval()
        self.cpu()
        return history

    def _epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.BCELoss,
        discriminator_opt: torch.optim.Optimizer,
        generator_opt: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        self.train()
        d_total = 0.0
        g_total = 0.0
        seen = 0
        for (real,) in loader:
            real = real.to(device)
            batch_size = real.shape[0]
            d_loss = self._discriminator_step(
                real, batch_size, device, criterion, discriminator_opt
            )
            g_loss = self._generator_step(batch_size, device, criterion, generator_opt)
            d_total += float(d_loss.detach().cpu()) * batch_size
            g_total += float(g_loss.detach().cpu()) * batch_size
            seen += batch_size
        return d_total / seen, g_total / seen

    def _scores(self, images: torch.Tensor) -> torch.Tensor:
        view = _to_resolution(images, self.resolutions[self._stage])
        return self.discriminator(view, self._stage, self._alpha)

    def _sample(self, noise: torch.Tensor) -> torch.Tensor:
        return self.generator(noise, self._stage, self._alpha)

    def _discriminator_step(
        self,
        real: torch.Tensor,
        batch_size: int,
        device: torch.device,
        criterion: nn.BCELoss,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        real_loss = criterion(self._scores(real), real_labels)
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        fake_loss = criterion(self._scores(self._sample(noise).detach()), fake_labels)
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss

    def _generator_step(
        self,
        batch_size: int,
        device: torch.device,
        criterion: nn.BCELoss,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        real_labels = torch.ones(batch_size, 1, device=device)
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        loss = criterion(self._scores(self._sample(noise)), real_labels)
        loss.backward()
        optimizer.step()
        return loss

    def _score_loader(
        self,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.BCELoss,
    ) -> tuple[float, float]:
        self.eval()
        d_total = 0.0
        g_total = 0.0
        seen = 0
        with torch.inference_mode():
            for (real,) in loader:
                real = real.to(device)
                batch_size = real.shape[0]
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                noise = torch.randn(batch_size, self.noise_dim, device=device)
                fake = self._sample(noise)
                d_loss = criterion(self._scores(real), real_labels) + criterion(
                    self._scores(fake),
                    fake_labels,
                )
                g_loss = criterion(self._scores(fake), real_labels)
                d_total += float(d_loss.detach().cpu()) * batch_size
                g_total += float(g_loss.detach().cpu()) * batch_size
                seen += batch_size
        if seen == 0:
            raise ValueError("train and eval splits must both be non-empty")
        return d_total / seen, g_total / seen

    def _snapshot(
        self,
        device: torch.device,
        generations_dir: Path | None,
        epoch: int,
        snapshot_every: int,
        resolution: int,
        phase: str,
        rows: dict[tuple[int, str], np.ndarray],
        row_order: list[tuple[int, str]],
        frames: list[dict[str, float | int | str]],
    ) -> float:
        self.eval()
        with torch.inference_mode():
            fake = self._sample(self.fixed_noise.to(device))
        sample_std = _batch_std(fake)
        key = (self._stage, phase)
        if key not in rows:
            row_order.append(key)
        rows[key] = unit_interval(fake[:_GROWTH_PREVIEW])
        if generations_dir is not None and epoch % snapshot_every == 0:
            filename = f"epoch_{epoch:04d}_res{resolution:02d}_{phase}.png"
            save_grid(
                unit_interval(fake),
                generations_dir / filename,
                f"{resolution}px {phase} epoch {epoch}",
            )
            frames.append(
                {
                    "epoch": epoch,
                    "stage": self._stage,
                    "resolution": resolution,
                    "phase": phase,
                    "alpha": self._alpha,
                    "file": filename,
                }
            )
        return sample_std

    @torch.inference_mode()
    def generate(self, count: int, *, noise: torch.Tensor | None = None) -> np.ndarray:
        """Full-resolution samples in NCHW ``[-1, 1]``."""
        if count < 1:
            raise ValueError("generate needs at least one sample")
        self._full_resolution()
        device = next(self.parameters()).device
        self.eval()
        if noise is None:
            noise = torch.randn(count, self.noise_dim, device=device)
        else:
            noise = noise.to(device)
        return self._sample(noise).detach().cpu().numpy()

    @torch.inference_mode()
    def adversarial_losses(self, pixels: np.ndarray, batch_size: int) -> tuple[float, float]:
        """Mean discriminator and generator BCE at full resolution. Weights stay put."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(pixels) == 0:
            raise ValueError("adversarial_losses needs at least one image")
        self._full_resolution()
        device = next(self.parameters()).device
        loader = DataLoader(_image_set(pixels), batch_size=batch_size, shuffle=False)
        criterion = nn.BCELoss()
        return self._score_loader(loader, device, criterion)

    def save(
        self,
        path: Path,
        columns: list[str],
        hparams: dict[str, float | int],
    ) -> None:
        """Write a CPU checkpoint. Keys are ``state_dict``, ``columns``, and ``hparams``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        module = self.cpu()
        module.eval()
        torch.save(
            {
                "state_dict": module.state_dict(),
                "columns": list(columns),
                "hparams": {key: hparams[key] for key in _HPARAM_KEYS},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str | None = None) -> dict[str, object]:
        """Local checkpoint, else the DVC remote.

        Keys are ``model``, ``columns``, and ``hparams``.
        """
        blob = torch.load(ensure_model_file(path), map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "state_dict" not in blob or "columns" not in blob:
            raise TypeError(
                "progressive gan must be a torch checkpoint dict with state_dict and columns, "
                f"got {type(blob).__name__}"
            )
        hparams = blob.get("hparams")
        if not isinstance(hparams, dict) or any(key not in hparams for key in _ARCH_KEYS):
            raise TypeError("progressive gan checkpoint is missing architecture hparams")
        model = cls.from_hparams(hparams)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return {"model": model, "columns": list(blob["columns"]), "hparams": dict(hparams)}


def _phases(n_stages: int, epochs: int) -> list[dict[str, int | str]]:
    phases: list[dict[str, int | str]] = [
        {"stage": 0, "resolution": 4, "phase": "stable", "epochs": epochs}
    ]
    resolution = 4
    for stage in range(1, n_stages):
        resolution *= 2
        phases.append({"stage": stage, "resolution": resolution, "phase": "fade", "epochs": epochs})
        phases.append(
            {"stage": stage, "resolution": resolution, "phase": "stable", "epochs": epochs}
        )
    return phases


def _blend(phase: str, epoch_in_phase: int, epochs: int) -> float:
    if phase != "fade":
        return 1.0
    if epochs <= 1:
        return 0.5
    return (epoch_in_phase + 1) / epochs


def _image_set(features: np.ndarray) -> TensorDataset:
    pixels = torch.from_numpy(np.ascontiguousarray(features))
    return TensorDataset(pixels)


def _env(name: str) -> str:
    return os.getenv(f"MNIST_PG_GAN_{name}", "").strip()


def _cache_dir() -> Path:
    override = _env("CACHE")
    if override:
        return Path(override)
    local = os.getenv("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "mnist-pg-gan-model"
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "mnist-pg-gan-model"


def _repo_and_rev() -> tuple[str, str | None]:
    repo = _env("REPO") or _MODEL_REPO
    rev = _env("REV") or None
    return repo, rev


def _checksum(info: dict[str, object]) -> str:
    sources: list[dict[str, object]] = [info]
    nested = info.get("dvc_info")
    if isinstance(nested, dict):
        sources.append(nested)
    for source in sources:
        raw = source.get("md5")
        if not isinstance(raw, str) or not raw.strip():
            raw = source.get("checksum")
        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            if text.startswith("md5:"):
                text = text[4:]
            return text.split(".", 1)[0]
    raise RuntimeError("DVC model info has no md5 checksum")


def _clone_error(repo: str) -> RuntimeError:
    return RuntimeError(
        f"could not clone {repo}. Authenticate to GitHub so this process can read it "
        "(gh auth login, or a credential helper for https://github.com)."
    )


def _remote_url(name: str) -> str:
    """URL of one remote in ``.dvc/config``. A local demo path is not ``s3://``."""
    config = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    headers = {f"['remote \"{name}\"']", f'[remote "{name}"]'}
    in_section = False
    for line in config.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped in headers
            continue
        if not in_section:
            continue
        key, sep, raw = stripped.partition("=")
        if sep and key.strip() == "url":
            value = raw.strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError(f"DVC remote {name} has no url in .dvc/config")


def _remote_md5(repo: str, rev: str | None) -> str:
    from dvc.api import DVCFileSystem
    from dvc.scm import CloneError

    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    filesystem = DVCFileSystem(repo, rev=rev) if rev else DVCFileSystem(repo)
    try:
        info = filesystem.info(_MODEL_RELATIVE)
    except CloneError as exc:
        raise _clone_error(repo) from exc
    return _checksum(info)


def _read_remote_model(repo: str, rev: str | None) -> bytes:
    import dvc.api
    from dvc.scm import CloneError

    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    kwargs: dict[str, object] = {"repo": repo, "remote": _MODEL_REMOTE, "mode": "rb"}
    if rev:
        kwargs["rev"] = rev
    try:
        with dvc.api.open(_MODEL_RELATIVE, **kwargs) as source:
            data = source.read()
    except CloneError as exc:
        raise _clone_error(repo) from exc
    if not data:
        raise RuntimeError(f"DVC remote returned an empty {_MODEL_RELATIVE}")
    return data


def _require_object_storage_credentials() -> None:
    url = _remote_url(_MODEL_REMOTE)
    if not url.startswith("s3://"):
        return
    load_dotenv(ROOT / ".env")
    key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if key and secret:
        return
    raise RuntimeError(
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to download "
        f"the model from the {_MODEL_REMOTE} DVC remote."
    )


def fetch_cached_model() -> Path:
    """Md5-cached checkpoint. Download with ``dvc.api.open`` when the hash is new."""
    global _session
    repo, rev = _repo_and_rev()
    cached = _session
    if cached is not None and cached[0] == repo and cached[1] == rev and cached[2].is_file():
        return cached[2]
    digest = _remote_md5(repo, rev)
    dest = _cache_dir() / f"{digest}.pt"
    if not dest.is_file():
        _require_object_storage_credentials()
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(".pt.partial")
        try:
            partial.write_bytes(_read_remote_model(repo, rev))
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"could not download {_MODEL_RELATIVE} from the {_MODEL_REMOTE} DVC remote"
            ) from exc
        partial.replace(dest)
    _session = (repo, rev, dest)
    return dest


def ensure_model_file(path: Path | str | None = None) -> Path:
    """Named file, env file, local artifact, then the DVC md5 cache."""
    if path is not None and str(path).strip():
        chosen: Path | None = Path(path)
    else:
        override = _env("MODEL")
        chosen = Path(override) if override else None
    if chosen is None and LOCAL_MODEL.is_file():
        chosen = LOCAL_MODEL
    if chosen is not None:
        if not chosen.is_file():
            raise FileNotFoundError(f"progressive gan not found: {chosen}")
        return chosen
    return fetch_cached_model()
