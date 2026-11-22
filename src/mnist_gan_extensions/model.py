"""Class-conditional GAN, training, and the DVC checkpoint.

The generator follows ``GAN_Tutorial.ipynb``: three hidden linear layers and
ReLU, then ``Tanh`` so pixels stay in [-1, 1]. A learned class embedding is
concatenated with the noise, which is the cGAN conditioning (Mirza & Osindero).
The embedding is as wide as the noise so the label is not drowned by it.
The MLP discriminator concatenates an embedding of that same label.

``variant`` selects the discriminator experiment. ``cgan`` keeps the tutorial
MLP and binary cross-entropy. ``patchgan`` replaces that MLP with a
convolutional PatchGAN whose output is a spatial map of real/fake decisions.
``feature_matching`` keeps the MLP and adds an L1 match of discriminator
features to the generator loss.

``ConditionalGAN.load`` uses a local checkpoint when one is already on disk.
Otherwise the latest fit is read from the DVC remote in ``.dvc/config`` with
``dvc.api.open`` and cached by md5. An ``s3://`` remote needs
``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mnist_gan_extensions.config import (
    FM_WEIGHT,
    N_CLASSES,
    PREVIEW_COUNT,
    ROOT,
    VARIANTS,
)
from mnist_gan_extensions.figures import save_grid, unit_interval

_MODEL_REPO = "https://github.com/Sidel-Meril/mnist-gan-extensions-xyzh.git"
_MODEL_REMOTE = "demo"
_MODEL_RELATIVE = "models/mnist_gan_extensions/mnist_cgan.pt"
LOCAL_MODEL = ROOT / "models" / "mnist_gan_extensions" / "mnist_cgan.pt"
_session: tuple[str, str | None, Path] | None = None

_ARCH_KEYS = ("noise_dim", "hidden_dim", "in_features", "n_classes", "variant")
_HPARAM_KEYS = (*_ARCH_KEYS, "lr_g", "lr_d", "beta1", "batch_size", "fm_weight")
_VARIANT_ALIASES = {
    "cgan": "cgan",
    "patchgan": "patchgan",
    "feature_matching": "feature_matching",
    "feature-matching-loss": "feature_matching",
}


def canonical_variant(name: str) -> str:
    """Map a folder name or alias onto ``cgan``, ``patchgan``, or ``feature_matching``."""
    key = _VARIANT_ALIASES.get(name)
    if key is None:
        raise ValueError(f"variant must be one of {tuple(_VARIANT_ALIASES)}")
    return key


def complete_hparams(hparams: dict[str, float | int | str]) -> dict[str, float | int | str]:
    """Fill class count, variant, and feature-matching weight when a caller omits them."""
    full = dict(hparams)
    full.setdefault("n_classes", N_CLASSES)
    full.setdefault("fm_weight", FM_WEIGHT)
    full["variant"] = canonical_variant(str(full.get("variant", "cgan")))
    full["n_classes"] = int(full["n_classes"])
    full["fm_weight"] = float(full["fm_weight"])
    return full


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_images(flat: torch.Tensor) -> torch.Tensor:
    if flat.ndim == 2 and flat.shape[1] == 784:
        return flat.reshape(flat.shape[0], 1, 28, 28)
    return flat


def _batch_std(images: torch.Tensor) -> float:
    flat = images.detach().reshape(images.shape[0], -1).float()
    if flat.shape[0] < 2:
        return 0.0
    return float(flat.std(unbiased=False).cpu())


class ConditionalGenerator(nn.Module):
    """Maps noise and a class label to a flat sample in [-1, 1]."""

    def __init__(self, noise_dim: int, n_classes: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.embed = nn.Embedding(self.n_classes, noise_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_vec = self.embed(labels.long())
        return self.net(torch.cat([noise, class_vec], dim=1))


class ConditionalDiscriminator(nn.Module):
    """Tutorial MLP. A class embedding is concatenated with the flat image."""

    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.embed = nn.Embedding(self.n_classes, hidden_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = data.reshape(data.shape[0], -1)
        class_vec = self.embed(labels.long())
        hidden = self.features(torch.cat([flat, class_vec], dim=1))
        return self.head(hidden), hidden


class PatchDiscriminator(nn.Module):
    """PatchGAN on 28x28 images. The label is a stack of spatial one-hot maps.

    Four stride-2 layers would shrink 28x28 to a single cell. Two stride-2
    layers leave a 5x5 map, so each output is a patch decision rather than
    one score for the whole image. Channel width stays at 32 so a CPU run of
    the shared training loop finishes with the MLP variants.
    """

    def __init__(self, n_classes: int, _hidden_dim: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        width = 32
        self.features = nn.Sequential(
            nn.Conv2d(1 + self.n_classes, width, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, width * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width * 2, width * 4, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(width * 4, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = _as_images(data)
        if images.ndim != 4:
            raise ValueError("patchgan expects 28x28 images")
        one_hot = F.one_hot(labels.long(), self.n_classes).to(dtype=images.dtype)
        maps = one_hot[:, :, None, None].expand(-1, -1, images.shape[2], images.shape[3])
        hidden = self.features(torch.cat([images, maps], dim=1))
        return self.head(hidden), hidden


class ConditionalGAN(nn.Module):
    """cGAN trained with the tutorial's two-step update.

    ``load`` rebuilds this class from a local checkpoint or the DVC remote.
    ``save`` writes ``state_dict``, ``columns``, and ``hparams``.
    """

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_classes: int = N_CLASSES,
        variant: str = "cgan",
        fm_weight: float = FM_WEIGHT,
    ) -> None:
        super().__init__()
        if noise_dim < 1:
            raise ValueError("noise_dim must be at least 1")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if output_dim < 1:
            raise ValueError("output_dim must be at least 1")
        if n_classes < 1:
            raise ValueError("n_classes must be at least 1")
        if fm_weight < 0:
            raise ValueError("fm_weight must be non-negative")
        self.variant = canonical_variant(variant)
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.variant == "patchgan" and output_dim != 784:
            raise ValueError("patchgan expects 784 pixels (28x28)")
        self.noise_dim = int(noise_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.n_classes = int(n_classes)
        self.fm_weight = float(fm_weight)
        self.generator = ConditionalGenerator(
            self.noise_dim, self.n_classes, self.hidden_dim, self.output_dim
        )
        if self.variant == "patchgan":
            self.discriminator: nn.Module = PatchDiscriminator(self.n_classes, self.hidden_dim)
        else:
            self.discriminator = ConditionalDiscriminator(
                self.output_dim, self.n_classes, self.hidden_dim
            )
        self.register_buffer("fixed_noise", torch.zeros(PREVIEW_COUNT, self.noise_dim))
        self.register_buffer(
            "fixed_labels",
            torch.arange(PREVIEW_COUNT) % self.n_classes,
        )

    @classmethod
    def from_hparams(cls, hparams: dict[str, float | int | str]) -> ConditionalGAN:
        """Build the architecture stored in a tune result or a checkpoint."""
        full = complete_hparams(hparams)
        return cls(
            noise_dim=int(full["noise_dim"]),
            hidden_dim=int(full["hidden_dim"]),
            output_dim=int(full["in_features"]),
            n_classes=int(full["n_classes"]),
            variant=str(full["variant"]),
            fm_weight=float(full["fm_weight"]),
        )

    def _reset(self) -> None:
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None and module is not self:
                reset()
        if self.variant != "patchgan":
            return
        for module in self.discriminator.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def data_loaders(
        train_x: np.ndarray,
        train_y: np.ndarray,
        eval_x: np.ndarray,
        eval_y: np.ndarray,
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
            _labeled_set(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        )
        eval_loader = DataLoader(
            _labeled_set(eval_x, eval_y),
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, eval_loader

    def fit_from_dataset(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        hparams: dict[str, float | int | str],
        *,
        epochs: int,
        seed: int,
        generations_dir: Path | None = None,
        snapshot_every: int = 1,
    ) -> dict[str, list[float]]:
        """One discriminator step, then one generator step, per training batch.

        Eval losses are recorded and do not pick weights. Fixed noise and a
        repeating class cycle make epoch grids comparable across variants.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be at least 1")
        if len(train_loader) == 0 or len(eval_loader) == 0:
            raise ValueError("train and eval splits must both be non-empty")
        full = complete_hparams(hparams)
        if canonical_variant(str(full["variant"])) != self.variant:
            raise ValueError("hparams variant does not match this network")
        seed_all(seed)
        self.cpu()
        self._reset()
        self.fixed_noise.copy_(torch.randn(self.fixed_noise.shape))
        self.fixed_labels.copy_(torch.arange(self.fixed_labels.shape[0]) % self.n_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.BCELoss()
        generator_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=float(full["lr_g"]),
            betas=(float(full["beta1"]), 0.999),
        )
        discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(full["lr_d"]),
            betas=(float(full["beta1"]), 0.999),
        )
        history: dict[str, list[float]] = {
            "train_discriminator": [],
            "train_generator": [],
            "train_generator_bce": [],
            "eval_discriminator": [],
            "eval_generator": [],
            "eval_generator_bce": [],
            "sample_std": [],
        }
        for epoch in range(epochs):
            train_d, train_g, train_bce = self._epoch(
                train_loader,
                device,
                criterion,
                discriminator_opt,
                generator_opt,
            )
            eval_d, eval_g, eval_bce = self._score_loader(eval_loader, device, criterion)
            sample_std = self._snapshot(device, generations_dir, epoch, snapshot_every)
            history["train_discriminator"].append(train_d)
            history["train_generator"].append(train_g)
            history["train_generator_bce"].append(train_bce)
            history["eval_discriminator"].append(eval_d)
            history["eval_generator"].append(eval_g)
            history["eval_generator_bce"].append(eval_bce)
            history["sample_std"].append(sample_std)
            print(
                f"epoch {epoch + 1}/{epochs} variant={self.variant} "
                f"d={train_d:.4f} g={train_g:.4f} sample_std={sample_std:.4f}"
            )
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
    ) -> tuple[float, float, float]:
        self.train()
        d_total = 0.0
        g_total = 0.0
        bce_total = 0.0
        seen = 0
        for real, labels in loader:
            real = real.to(device)
            labels = labels.to(device)
            batch_size = real.shape[0]
            d_loss = self._discriminator_step(
                real, labels, batch_size, device, criterion, discriminator_opt
            )
            g_loss, g_bce = self._generator_step(
                real, labels, batch_size, device, criterion, generator_opt
            )
            d_total += float(d_loss.detach().cpu()) * batch_size
            g_total += float(g_loss.detach().cpu()) * batch_size
            bce_total += float(g_bce.detach().cpu()) * batch_size
            seen += batch_size
        return d_total / seen, g_total / seen, bce_total / seen

    def _discriminator_step(
        self,
        real: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        device: torch.device,
        criterion: nn.BCELoss,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        fake = self.generator(noise, labels).detach()
        loss = self._bce_pair(real, fake, labels, criterion)
        loss.backward()
        optimizer.step()
        return loss

    def _generator_step(
        self,
        real: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        device: torch.device,
        criterion: nn.BCELoss,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        fake = self.generator(noise, labels)
        loss, bce = self._generator_terms(real, fake, labels, criterion)
        loss.backward()
        optimizer.step()
        return loss, bce

    def _bce_pair(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.BCELoss,
    ) -> torch.Tensor:
        real_prob, _features = self.discriminator(real, labels)
        fake_prob, _fake_features = self.discriminator(fake, labels)
        real_loss = criterion(real_prob, torch.ones_like(real_prob))
        fake_loss = criterion(fake_prob, torch.zeros_like(fake_prob))
        return real_loss + fake_loss

    def _generator_terms(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.BCELoss,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generator objective and the adversarial BCE term inside it.

        Feature matching compares batch-mean discriminator features. The real
        features are a target, so they do not train the discriminator.
        """
        fake_prob, fake_feat = self.discriminator(fake, labels)
        bce = criterion(fake_prob, torch.ones_like(fake_prob))
        if self.variant != "feature_matching":
            return bce, bce
        with torch.no_grad():
            _real_prob, real_feat = self.discriminator(real, labels)
        match = F.l1_loss(fake_feat.mean(dim=0), real_feat.mean(dim=0))
        return bce + self.fm_weight * match, bce

    def _score_loader(
        self,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.BCELoss,
    ) -> tuple[float, float, float]:
        self.eval()
        d_total = 0.0
        g_total = 0.0
        bce_total = 0.0
        seen = 0
        with torch.inference_mode():
            for real, labels in loader:
                real = real.to(device)
                labels = labels.to(device)
                batch_size = real.shape[0]
                noise = torch.randn(batch_size, self.noise_dim, device=device)
                fake = self.generator(noise, labels)
                d_loss = self._bce_pair(real, fake, labels, criterion)
                g_loss, g_bce = self._generator_terms(real, fake, labels, criterion)
                d_total += float(d_loss.detach().cpu()) * batch_size
                g_total += float(g_loss.detach().cpu()) * batch_size
                bce_total += float(g_bce.detach().cpu()) * batch_size
                seen += batch_size
        if seen == 0:
            raise ValueError("train and eval splits must both be non-empty")
        return d_total / seen, g_total / seen, bce_total / seen

    def _snapshot(
        self,
        device: torch.device,
        generations_dir: Path | None,
        epoch: int,
        snapshot_every: int,
    ) -> float:
        self.eval()
        with torch.inference_mode():
            fake = _as_images(
                self.generator(self.fixed_noise.to(device), self.fixed_labels.to(device))
            )
        sample_std = _batch_std(fake)
        if generations_dir is not None and (epoch + 1) % snapshot_every == 0:
            save_grid(
                unit_interval(fake),
                generations_dir / f"epoch_{epoch + 1:04d}.png",
                f"{self.variant} epoch {epoch + 1}",
            )
        return sample_std

    @torch.inference_mode()
    def generate(
        self,
        count: int,
        *,
        labels: np.ndarray | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Samples in NCHW ``[-1, 1]`` when the output length is 784."""
        if count < 1:
            raise ValueError("generate needs at least one sample")
        device = next(self.parameters()).device
        self.eval()
        label_tensor = torch.as_tensor(labels, device=device, dtype=torch.long).reshape(-1)
        if int(label_tensor.numel()) != count:
            raise ValueError("generate needs one label per sample")
        if int(label_tensor.min()) < 0 or int(label_tensor.max()) >= self.n_classes:
            raise ValueError("label outside 0..n_classes-1")
        if noise is None:
            noise = torch.randn(count, self.noise_dim, device=device)
        else:
            noise = noise.to(device)
            if tuple(noise.shape) != (count, self.noise_dim):
                raise ValueError("noise shape does not match count and noise_dim")
        flat = self.generator(noise, label_tensor)
        return _as_images(flat).detach().cpu().numpy()

    @torch.inference_mode()
    def class_batch(self, per_class: int, *, noise: torch.Tensor | None = None) -> np.ndarray:
        """Class-major samples: ``per_class`` images of class 0, then class 1, and so on."""
        if per_class < 1:
            raise ValueError("per_class must be at least 1")
        labels = np.repeat(np.arange(self.n_classes, dtype=np.int64), per_class)
        return self.generate(len(labels), labels=labels, noise=noise)

    def score_split(
        self,
        pixels: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
    ) -> tuple[float, float, float]:
        """Mean discriminator loss, generator objective, and generator BCE."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(pixels) == 0:
            raise ValueError("score_split needs at least one image")
        device = next(self.parameters()).device
        loader = DataLoader(_labeled_set(pixels, labels), batch_size=batch_size, shuffle=False)
        criterion = nn.BCELoss()
        return self._score_loader(loader, device, criterion)

    def save(
        self,
        path: Path,
        columns: list[str],
        hparams: dict[str, float | int | str],
    ) -> None:
        """Write a CPU checkpoint. Keys are ``state_dict``, ``columns``, and ``hparams``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        full = complete_hparams(hparams)
        module = self.cpu()
        module.eval()
        torch.save(
            {
                "state_dict": module.state_dict(),
                "columns": list(columns),
                "hparams": {key: full[key] for key in _HPARAM_KEYS},
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
                "conditional gan must be a torch checkpoint dict with state_dict and columns, "
                f"got {type(blob).__name__}"
            )
        hparams = blob.get("hparams")
        if not isinstance(hparams, dict) or any(key not in hparams for key in _ARCH_KEYS):
            raise TypeError("conditional gan checkpoint is missing architecture hparams")
        model = cls.from_hparams(hparams)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return {"model": model, "columns": list(blob["columns"]), "hparams": dict(hparams)}


def _labeled_set(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    if len(features) != len(labels):
        raise ValueError("images and labels must have the same length")
    pixels = torch.from_numpy(np.ascontiguousarray(features))
    targets = torch.from_numpy(np.ascontiguousarray(np.asarray(labels, dtype=np.int64)))
    return TensorDataset(pixels, targets)


def _env(name: str) -> str:
    return os.getenv(f"MNIST_GAN_{name}", "").strip()


def _cache_dir() -> Path:
    override = _env("CACHE")
    if override:
        return Path(override)
    local = os.getenv("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "mnist-gan-extensions"
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "mnist-gan-extensions"


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
            raise FileNotFoundError(f"conditional gan not found: {chosen}")
        return chosen
    return fetch_cached_model()
