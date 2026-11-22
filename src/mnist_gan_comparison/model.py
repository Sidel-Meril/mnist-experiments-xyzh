"""Vanilla GAN, training, and the DVC checkpoint.

The generator and discriminator follow ``GAN_Tutorial.ipynb``: three hidden
linear layers, ReLU in the generator, LeakyReLU(0.2) in the discriminator,
and a sigmoid probability. The tutorial has no output activation because its
data are 2D coordinates. This generator ends in ``Tanh`` so samples match
pixels stored in [-1, 1]. The loss is ``BCELoss``. Each step updates the
discriminator on real labels of 1 and detached fakes labeled 0, then the
generator on labels of 1. Both optimizers are Adam with ``beta1`` 0.5.

``VanillaGAN.load`` uses a local checkpoint when one is already on disk.
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
from torch.utils.data import DataLoader, TensorDataset

from mnist_gan_comparison.config import PREVIEW_COUNT, ROOT
from mnist_gan_comparison.figures import save_grid, unit_interval

_MODEL_REPO = "https://github.com/Sidel-Meril/mnist-gan-comparison-xyzh.git"
_MODEL_REMOTE = "demo"
_MODEL_RELATIVE = "models/mnist_gan_comparison/mnist_gan.pt"
LOCAL_MODEL = ROOT / "models" / "mnist_gan_comparison" / "mnist_gan.pt"
_session: tuple[str, str | None, Path] | None = None

_ARCH_KEYS = ("noise_dim", "hidden_dim", "in_features")
_HPARAM_KEYS = (*_ARCH_KEYS, "lr_g", "lr_d", "beta1", "batch_size")


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
    """Mean per-pixel deviation across the batch. Identical images score near 0."""
    flat = images.detach().reshape(images.shape[0], -1).float()
    if flat.shape[0] < 2:
        return 0.0
    return float(flat.std(dim=0, unbiased=False).mean().cpu())


class Generator(nn.Module):
    """Maps noise to a flat sample. ``Tanh`` keeps image pixels in [-1, 1]."""

    def __init__(self, noise_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class Discriminator(nn.Module):
    """Probability that a flattened sample is real."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        flat = data.reshape(data.shape[0], -1)
        return self.net(flat)


class VanillaGAN(nn.Module):
    """Generator plus discriminator trained with the tutorial's two-step update.

    ``load`` rebuilds this class from a local checkpoint or the DVC remote.
    ``save`` writes ``state_dict``, ``columns``, and ``hparams``.
    """

    def __init__(self, noise_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if noise_dim < 1:
            raise ValueError("noise_dim must be at least 1")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if output_dim < 1:
            raise ValueError("output_dim must be at least 1")
        self.noise_dim = int(noise_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.generator = Generator(self.noise_dim, self.hidden_dim, self.output_dim)
        self.discriminator = Discriminator(self.output_dim, self.hidden_dim)
        self.register_buffer("fixed_noise", torch.zeros(PREVIEW_COUNT, self.noise_dim))

    @classmethod
    def from_hparams(cls, hparams: dict[str, float | int]) -> VanillaGAN:
        """Build the architecture stored in a tune result or a checkpoint."""
        return cls(
            noise_dim=int(hparams["noise_dim"]),
            hidden_dim=int(hparams["hidden_dim"]),
            output_dim=int(hparams["in_features"]),
        )

    def _reset(self) -> None:
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None and module is not self:
                reset()

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
        """One discriminator step, then one generator step, per training batch.

        Eval losses are recorded and do not pick weights. A fixed noise batch
        is drawn once so epoch grids show whether samples collapse.
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
        }
        for epoch in range(epochs):
            train_d, train_g = self._epoch(
                train_loader,
                device,
                criterion,
                discriminator_opt,
                generator_opt,
            )
            eval_d, eval_g = self._score_loader(eval_loader, device, criterion)
            sample_std = self._snapshot(device, generations_dir, epoch, snapshot_every)
            history["train_discriminator"].append(train_d)
            history["train_generator"].append(train_g)
            history["eval_discriminator"].append(eval_d)
            history["eval_generator"].append(eval_g)
            history["sample_std"].append(sample_std)
            print(
                f"epoch {epoch + 1}/{epochs} "
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
        real_loss = criterion(self.discriminator(real), real_labels)
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        fake_loss = criterion(self.discriminator(self.generator(noise).detach()), fake_labels)
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
        loss = criterion(self.discriminator(self.generator(noise)), real_labels)
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
                fake = self.generator(noise)
                d_loss = criterion(self.discriminator(real), real_labels) + criterion(
                    self.discriminator(fake),
                    fake_labels,
                )
                g_loss = criterion(self.discriminator(fake), real_labels)
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
    ) -> float:
        self.eval()
        with torch.inference_mode():
            fake = _as_images(self.generator(self.fixed_noise.to(device)))
        sample_std = _batch_std(fake)
        if generations_dir is not None and (epoch + 1) % snapshot_every == 0:
            save_grid(
                unit_interval(fake),
                generations_dir / f"epoch_{epoch + 1:04d}.png",
                f"epoch {epoch + 1}",
            )
        return sample_std

    @torch.inference_mode()
    def generate(self, count: int, *, noise: torch.Tensor | None = None) -> np.ndarray:
        """Samples in NCHW ``[-1, 1]`` when the output length is 784."""
        if count < 1:
            raise ValueError("generate needs at least one sample")
        device = next(self.parameters()).device
        self.eval()
        if noise is None:
            noise = torch.randn(count, self.noise_dim, device=device)
        else:
            noise = noise.to(device)
        flat = self.generator(noise)
        return _as_images(flat).detach().cpu().numpy()

    @torch.inference_mode()
    def adversarial_losses(self, pixels: np.ndarray, batch_size: int) -> tuple[float, float]:
        """Mean discriminator and generator BCE on one split. Weights stay put."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(pixels) == 0:
            raise ValueError("adversarial_losses needs at least one image")
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
                "vanilla gan must be a torch checkpoint dict with state_dict and columns, "
                f"got {type(blob).__name__}"
            )
        hparams = blob.get("hparams")
        if not isinstance(hparams, dict) or any(key not in hparams for key in _ARCH_KEYS):
            raise TypeError("vanilla gan checkpoint is missing architecture hparams")
        model = cls.from_hparams(hparams)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return {"model": model, "columns": list(blob["columns"]), "hparams": dict(hparams)}


def _image_set(features: np.ndarray) -> TensorDataset:
    pixels = torch.from_numpy(np.ascontiguousarray(features))
    return TensorDataset(pixels)


def _env(name: str) -> str:
    return os.getenv(f"MNIST_GAN_{name}", "").strip()


def _cache_dir() -> Path:
    override = _env("CACHE")
    if override:
        return Path(override)
    local = os.getenv("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "mnist-gan-model"
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "mnist-gan-model"


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
            raise FileNotFoundError(f"vanilla gan not found: {chosen}")
        return chosen
    return fetch_cached_model()
