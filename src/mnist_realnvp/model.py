"""Multi-scale RealNVP for grayscale MNIST, plus training and the DVC checkpoint.

The coupling layer follows the lecture notebook: a mask holds one part fixed,
and two heads predict scale ``s`` and shift ``t`` from that part.

    y = m * x + (1 - m) * (x * exp(s) + t)

``s`` is ``scale * tanh(.)``, the same bound as the notebook. The log-determinant
sums ``s`` only on the changed coordinates. The notebook sums the whole ``s``
vector; that count is wrong once ``s`` is dense, so the image model keeps the
masked sum.

Images use the convolutional form of those heads. Multi-scale RealNVP then
squeezes space into channels and factors half of the channels out as latent
variables at 14x14 and 7x7. The base density is a standard normal. Pixels in
``[0, 1]`` are uniformly dequantized and passed through a logit transform.

``RealNVP.load`` uses a local checkpoint when one is already on disk. Otherwise
the latest fit is read from the DVC remote in ``.dvc/config`` with ``dvc.api.open``
and cached by md5. An ``s3://`` remote needs ``AWS_ACCESS_KEY_ID`` and
``AWS_SECRET_ACCESS_KEY``.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_realnvp.config import IMAGE_CHW, ROOT

_MODEL_REPO = "https://github.com/Sidel-Meril/mnist-realnvp-xyzh.git"
_MODEL_REMOTE = "demo"
_MODEL_RELATIVE = "models/mnist_realnvp/mnist_realnvp.pt"
LOCAL_MODEL = ROOT / "models" / "mnist_realnvp" / "mnist_realnvp.pt"
_session: tuple[str, str | None, Path] | None = None

_ARCH_KEYS = ("hidden_channels", "n_layers", "scale")
_DIMS = int(math.prod(IMAGE_CHW))
_LOG_2 = math.log(2.0)
_GRAD_CLIP = 10.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bpd_from_nll(nll: float) -> float:
    """Bits per dimension from a mean per-image negative log-likelihood in nats."""
    return float(nll) / (_DIMS * _LOG_2)


def rows_for_dataset(
    codes: np.ndarray,
    indices: np.ndarray,
    dataset: str,
    categories: tuple[str, ...],
) -> np.ndarray:
    """Row positions for one source, or every position when ``dataset`` is ``both``."""
    chosen = np.asarray(indices, dtype=np.int64)
    if dataset == "both":
        return chosen
    if dataset not in categories:
        raise ValueError(f"dataset must be one of {categories + ('both',)}, got {dataset}")
    code = categories.index(dataset)
    selected = chosen[np.asarray(codes)[chosen] == code]
    if len(selected) == 0:
        raise ValueError(f"no rows for dataset {dataset}")
    return selected


def _channel_mask(channels: int, parity: int) -> torch.Tensor:
    mask = torch.zeros(1, channels, 1, 1)
    start = 0 if parity % 2 == 0 else 1
    mask[:, start::2] = 1
    return mask


class Squeeze(nn.Module):
    """Trade spatial size for channels: ``(C, H, W)`` becomes ``(4C, H/2, W/2)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if height % 2 or width % 2:
            raise ValueError(f"squeeze needs even spatial size, got {height}x{width}")
        reshaped = x.view(batch, channels, height // 2, 2, width // 2, 2)
        shuffled = reshaped.permute(0, 1, 3, 5, 2, 4).contiguous()
        return shuffled.view(batch, channels * 4, height // 2, width // 2)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels % 4:
            raise ValueError(f"unsqueeze needs a multiple of 4 channels, got {channels}")
        reshaped = x.view(batch, channels // 4, 2, 2, height, width)
        shuffled = reshaped.permute(0, 1, 4, 2, 5, 3).contiguous()
        return shuffled.view(batch, channels // 4, height * 2, width * 2)


class ScaleTranslate(nn.Module):
    """Three convolutions, the image analogue of the notebook's ``s`` and ``t`` MLPs."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden channels must be at least 1")
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_s, shift = self.net(x).chunk(2, dim=1)
        return raw_s, shift


class CouplingLayer(nn.Module):
    """Affine coupling. ``mask == 1`` is unchanged; ``mask == 0`` is scaled and shifted."""

    def __init__(self, channels: int, hidden: int, mask: torch.Tensor, scale: float) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)
        self.st = ScaleTranslate(channels, hidden)
        self.register_buffer("mask", mask.detach().float())

    def _factors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_s, shift = self.st(x * self.mask)
        return self.scale * torch.tanh(raw_s), shift

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_s, shift = self._factors(x)
        changed = 1.0 - self.mask
        y = self.mask * x + changed * (x * torch.exp(log_s) + shift)
        log_det = (changed * log_s).flatten(1).sum(1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        log_s, shift = self._factors(y)
        changed = 1.0 - self.mask
        return self.mask * y + changed * (y - shift) * torch.exp(-log_s)


class CouplingBlock(nn.Module):
    """Alternating channel masks so every channel is transformed."""

    def __init__(self, channels: int, hidden: int, n_layers: int, scale: float) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")
        layers = [
            CouplingLayer(channels, hidden, _channel_mask(channels, parity), scale)
            for parity in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = x.new_zeros(x.shape[0])
        for layer in self.layers:
            x, step = layer(x)
            log_det = log_det + step
        return x, log_det

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


class MultiScaleFlow(nn.Module):
    """Squeeze, couple, and factor out channels at two scales, then a final coupling."""

    def __init__(self, hidden: int, n_layers: int, scale: float) -> None:
        super().__init__()
        self.squeeze = Squeeze()
        self.block_14 = CouplingBlock(4, hidden, n_layers, scale)
        self.block_7 = CouplingBlock(8, hidden, n_layers, scale)
        self.block_out = CouplingBlock(4, hidden, n_layers, scale)
        self.latent_shapes = ((2, 14, 14), (4, 7, 7), (4, 7, 7))

    def forward(self, x: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        log_det = x.new_zeros(x.shape[0])
        x = self.squeeze(x)
        x, step = self.block_14(x)
        log_det = log_det + step
        z1, x = x.chunk(2, dim=1)
        x = self.squeeze(x)
        x, step = self.block_7(x)
        log_det = log_det + step
        z2, x = x.chunk(2, dim=1)
        z3, step = self.block_out(x)
        log_det = log_det + step
        return (z1, z2, z3), log_det

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        x = self.block_out.inverse(z3)
        x = torch.cat((z2, x), dim=1)
        x = self.block_7.inverse(x)
        x = self.squeeze.inverse(x)
        x = torch.cat((z1, x), dim=1)
        x = self.block_14.inverse(x)
        return self.squeeze.inverse(x)


def _standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (z.pow(2) + math.log(2.0 * math.pi))).flatten(1).sum(1)


class RealNVP(nn.Module):
    """Density model on ``1x28x28`` images. ``forward`` is per-image NLL in nats.

    ``load`` rebuilds this class from a local checkpoint or the DVC remote.
    ``save`` writes ``state_dict``, ``columns``, and ``hparams``.
    ``sample`` draws a standard normal and applies the inverse flow.
    """

    def __init__(
        self,
        hidden_channels: int,
        n_layers: int,
        scale: float,
        *,
        alpha: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 0.5:
            raise ValueError("logit alpha must lie in (0, 0.5)")
        self.alpha = float(alpha)
        self.hidden_channels = int(hidden_channels)
        self.n_layers = int(n_layers)
        self.flow = MultiScaleFlow(int(hidden_channels), int(n_layers), float(scale))

    @classmethod
    def from_hparams(cls, hparams: dict[str, float | int | str]) -> RealNVP:
        """Build the architecture stored in a tune result or a checkpoint."""
        return cls(
            hidden_channels=int(hparams["hidden_channels"]),
            n_layers=int(hparams["n_layers"]),
            scale=float(hparams["scale"]),
        )

    def _preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = x.clamp(0, 1)
        noise = torch.rand_like(pixels) if self.training else torch.full_like(pixels, 0.5)
        dequantized = (pixels * 255.0 + noise) / 256.0
        y = self.alpha + (1.0 - 2.0 * self.alpha) * dequantized
        latent = torch.log(y) - torch.log1p(-y)
        logit_terms = math.log(1.0 - 2.0 * self.alpha) - torch.log(y) - torch.log1p(-y)
        logit_det = logit_terms.flatten(1).sum(1)
        dequant_det = -math.log(256.0) * _DIMS
        return latent, logit_det + dequant_det

    def _from_logit(self, latent: torch.Tensor) -> torch.Tensor:
        y = torch.sigmoid(latent)
        pixels = (y - self.alpha) / (1.0 - 2.0 * self.alpha)
        return pixels.clamp(0, 1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Discrete-data lower bound, one nats value per image."""
        if x.ndim != 4 or tuple(x.shape[1:]) != IMAGE_CHW:
            raise ValueError(f"expected NCHW {IMAGE_CHW}, got {tuple(x.shape)}")
        latent, preprocess_det = self._preprocess(x.float())
        parts, flow_det = self.flow(latent)
        log_prior = parts[0].new_zeros(latent.shape[0])
        for part in parts:
            log_prior = log_prior + _standard_normal_log_prob(part)
        return log_prior + flow_det + preprocess_det

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x)

    @torch.inference_mode()
    def sample(self, n: int) -> torch.Tensor:
        """Images in ``[0, 1]`` from the inverse flow. Shape is ``(n, 1, 28, 28)``."""
        if n < 1:
            raise ValueError("n must be at least 1")
        device = next(self.parameters()).device
        draws = [torch.randn(n, *shape, device=device) for shape in self.flow.latent_shapes]
        return self._from_logit(self.flow.inverse(*draws))

    @torch.inference_mode()
    def reconstruct(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode and invert. Eval mode uses the bin-center dequantization."""
        was_training = self.training
        self.eval()
        latent, _ = self._preprocess(pixels.float())
        parts, _ = self.flow(latent)
        restored = self._from_logit(self.flow.inverse(*parts))
        if was_training:
            self.train()
        return restored

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
        """Shuffled train loader and an ordered eval loader."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {"pin_memory": True} if torch.cuda.is_available() else {}
        train_loader = DataLoader(
            _tensor_dataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        )
        eval_loader = DataLoader(
            _tensor_dataset(eval_x, eval_y),
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
        record_scores: bool = False,
        on_epoch: Callable[[int, dict[str, list[float]]], None] | None = None,
    ) -> dict[str, list[float]]:
        """Adam on mean NLL. The eval curve is recorded and does not pick weights.

        ``record_scores`` is accepted so the stage call matches the template.
        Likelihood and bits/dimension are always recorded.
        """
        del record_scores
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if len(train_loader) == 0 or len(eval_loader) == 0:
            raise ValueError("train and eval splits must both be non-empty")
        seed_all(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        history: dict[str, list[float]] = {
            "train": [],
            "eval": [],
            "bpd_train": [],
            "bpd_eval": [],
        }
        for epoch in range(epochs):
            self.train()
            total = 0.0
            seen = 0
            for batch_x, _batch_y in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad(set_to_none=True)
                nll = self(batch_x)
                loss = nll.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), _GRAD_CLIP)
                optimizer.step()
                total += float(nll.detach().sum().cpu())
                seen += len(batch_x)
            train_nll = total / seen
            eval_nll = self.mean_nll(eval_loader, device)
            history["train"].append(train_nll)
            history["eval"].append(eval_nll)
            history["bpd_train"].append(bpd_from_nll(train_nll))
            history["bpd_eval"].append(bpd_from_nll(eval_nll))
            schedule.step()
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"train_nll={train_nll:.2f} test_nll={eval_nll:.2f} "
                f"train_bpd={history['bpd_train'][-1]:.3f} "
                f"test_bpd={history['bpd_eval'][-1]:.3f}",
                flush=True,
            )
            if on_epoch is not None:
                on_epoch(epoch, history)
                self.to(device)
        self.eval()
        self.cpu()
        return history

    @torch.inference_mode()
    def mean_nll(self, loader: DataLoader, device: torch.device) -> float:
        """Eval-mode mean NLL. Dequantization uses the center of each bin."""
        self.eval()
        total = 0.0
        seen = 0
        for batch_x, _batch_y in loader:
            batch_x = batch_x.to(device)
            nll = self(batch_x)
            total += float(nll.detach().sum().cpu())
            seen += len(batch_x)
        if seen == 0:
            raise ValueError("loader is empty")
        return total / seen

    def save(
        self,
        path: Path,
        columns: list[str],
        hparams: dict[str, float | int | str],
    ) -> None:
        """Write a CPU checkpoint. Keys are ``state_dict``, ``columns``, and ``hparams``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        module = self.cpu()
        module.eval()
        stored = {
            key: hparams[key]
            for key in (*_ARCH_KEYS, "lr", "weight_decay", "batch_size", "dataset")
        }
        torch.save(
            {
                "state_dict": module.state_dict(),
                "columns": list(columns),
                "hparams": stored,
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
                "realnvp model must be a torch checkpoint dict with state_dict and columns, "
                f"got {type(blob).__name__}"
            )
        hparams = blob.get("hparams")
        if not isinstance(hparams, dict) or any(key not in hparams for key in _ARCH_KEYS):
            raise TypeError("realnvp model checkpoint is missing architecture hparams")
        model = cls.from_hparams(hparams)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return {"model": model, "columns": list(blob["columns"]), "hparams": dict(hparams)}


def _tensor_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    pixels = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    targets = torch.from_numpy(np.ascontiguousarray(np.asarray(labels, dtype=np.int64)))
    return TensorDataset(pixels, targets)


def _env(name: str) -> str:
    return os.getenv(f"REALNVP_{name}", "").strip()


def _cache_dir() -> Path:
    override = _env("CACHE")
    if override:
        return Path(override)
    local = os.getenv("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "realnvp-model"
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "realnvp-model"


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
            raise FileNotFoundError(f"realnvp model not found: {chosen}")
        return chosen
    return fetch_cached_model()
