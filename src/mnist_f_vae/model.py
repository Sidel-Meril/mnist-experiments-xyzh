"""f-VAE for grayscale MNIST: RealNVP inside the VAE latent space.

The encoder, reparameterization, and ``Tanh`` decoder follow the MNIST VAE.
Pixels stay in [-1, 1]. The loss sums squared error over pixels, then adds
``beta`` times a KL. The base posterior ``q0(z0|x)`` is diagonal normal. A
stack of RealNVP affine couplings, the same masked scale-and-shift as the
image RealNVP, pushes ``z0`` to ``zK``:

    y = m * x + (1 - m) * (x * exp(s) + t)
    s = scale * tanh(.)

The log-determinant sums ``s`` only on the changed coordinates. The KL is the
Monte Carlo estimate

    log q0(z0|x) - log N(zK; 0, I) - log|det dzK/dz0|

which is the KL of the flowed posterior ``qK`` against the unit normal prior.
At initialization the coupling heads are zero, so the flow is the identity and
this KL matches the usual Gaussian KL in expectation. The decoder reads ``zK``.
Generated samples draw that prior and decode it. Reported reconstructions push
the base posterior mean through the flow with no noise.

``FVAE.load`` uses a local checkpoint when one is already on disk. Otherwise
the latest fit is read from the DVC remote in ``.dvc/config`` with
``dvc.api.open`` and cached by md5. An ``s3://`` remote needs
``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_f_vae.config import ROOT

_MODEL_REPO = "https://github.com/Sidel-Meril/mnist-f-vae-xyzh.git"
_MODEL_REMOTE = "demo"
_MODEL_RELATIVE = "models/mnist_f_vae/mnist_f_vae.pt"
LOCAL_MODEL = ROOT / "models" / "mnist_f_vae" / "mnist_f_vae.pt"
_session: tuple[str, str | None, Path] | None = None

_ARCH_KEYS = (
    "latent_dim",
    "hidden_dim",
    "dropout",
    "in_features",
    "n_flows",
    "flow_hidden",
    "flow_scale",
)
_SAVED_KEYS = (*_ARCH_KEYS, "lr", "weight_decay", "batch_size", "beta")
_GRAD_CLIP = 1.0
_LOG_2PI = math.log(2.0 * math.pi)
_LOG_PI = math.log(math.pi)
_LOG_2 = math.log(2.0)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bpd_from_objective(mean_sse: float, mean_kl: float, n_pixels: int) -> float:
    """Bits per dimension of the Gaussian ELBO used by the training loss.

    Squared error is the negative log-density of ``N(x; dec(z), (1/2) I)``
    without the constant ``(D/2) log(pi)``. ``mean_kl`` is the flowed KL.
    """
    nll = float(mean_sse) + 0.5 * int(n_pixels) * _LOG_PI + float(mean_kl)
    return nll / (int(n_pixels) * _LOG_2)


def _feature_mask(dim: int, parity: int) -> torch.Tensor:
    mask = torch.zeros(dim)
    start = 0 if parity % 2 == 0 else 1
    mask[start::2] = 1
    return mask


def _gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    return -0.5 * (_LOG_2PI + logvar + (z - mu).pow(2) * inv_var).sum(dim=1)


def _standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * (_LOG_2PI + z.pow(2)).sum(dim=1)


def _flow_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z0: torch.Tensor,
    zk: torch.Tensor,
    log_det: torch.Tensor,
) -> torch.Tensor:
    """``log q0(z0|x) - log p(zK) - log|det|``, one Monte Carlo KL per row."""
    return _gaussian_log_prob(z0, mu, logvar) - _standard_normal_log_prob(zk) - log_det


def _fvae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z0: torch.Tensor,
    zk: torch.Tensor,
    log_det: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    per_image = (reconstructed - original).flatten(1).pow(2).sum(dim=1)
    kl = _flow_kl(mu, logvar, z0, zk, log_det)
    return per_image.mean() + float(beta) * kl.mean()


class ScaleTranslate(nn.Module):
    """MLP scale and shift, the vector form of the image coupling heads."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("flow hidden size must be at least 1")
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_s, shift = self.net(x).chunk(2, dim=-1)
        return raw_s, shift


class CouplingLayer(nn.Module):
    """Affine coupling. ``mask == 1`` is unchanged; ``mask == 0`` is scaled and shifted."""

    def __init__(self, dim: int, hidden: int, mask: torch.Tensor, scale: float) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        if int(mask.numel()) != int(dim):
            raise ValueError("mask length must equal the latent size")
        self.scale = float(scale)
        self.st = ScaleTranslate(dim, hidden)
        self.register_buffer("mask", mask.detach().float().reshape(dim))

    def _factors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_s, shift = self.st(x * self.mask)
        return self.scale * torch.tanh(raw_s), shift

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_s, shift = self._factors(x)
        changed = 1.0 - self.mask
        y = self.mask * x + changed * (x * torch.exp(log_s) + shift)
        log_det = (changed * log_s).sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        log_s, shift = self._factors(y)
        changed = 1.0 - self.mask
        return self.mask * y + changed * (y - shift) * torch.exp(-log_s)


class LatentFlow(nn.Module):
    """Alternating coordinate masks so every latent unit is transformed."""

    def __init__(self, dim: int, hidden: int, n_flows: int, scale: float) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("latent_dim must be at least 2")
        if n_flows < 2:
            raise ValueError("n_flows must be at least 2")
        layers = [
            CouplingLayer(dim, hidden, _feature_mask(dim, parity), scale)
            for parity in range(n_flows)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = z.new_zeros(z.shape[0])
        for layer in self.layers:
            z, step = layer(z)
            log_det = log_det + step
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


def _mean_loss(model: FVAE, loader: DataLoader, device: torch.device, beta: float) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            reconstructed, mu, logvar, z0, zk, log_det = model(batch_x)
            loss = _fvae_loss(reconstructed, batch_y, mu, logvar, z0, zk, log_det, beta)
            total += float(loss.detach().cpu()) * len(batch_y)
            seen += len(batch_y)
    if seen == 0:
        raise ValueError("train and eval splits must both be non-empty")
    return total / seen


def _tensor_pair(features: np.ndarray) -> TensorDataset:
    pixels = torch.from_numpy(np.ascontiguousarray(features))
    return TensorDataset(pixels, pixels.clone())


class FVAE(nn.Module):
    """VAE whose approximate posterior is a RealNVP on the encoder Gaussian.

    ``load`` rebuilds this class from a local checkpoint or the DVC remote.
    ``save`` writes ``state_dict``, ``columns``, and ``hparams``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        n_flows: int = 4,
        flow_hidden: int = 256,
        flow_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim < 2:
            raise ValueError("latent_dim must be at least 2")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_flows = int(n_flows)
        self.flow_hidden = int(flow_hidden)
        self.flow_scale = float(flow_scale)
        half = self.hidden_dim // 2
        drop = float(dropout)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.hidden_dim, half),
            nn.BatchNorm1d(half),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(half, self.latent_dim)
        self.fc_logvar = nn.Linear(half, self.latent_dim)
        self.flow = LatentFlow(self.latent_dim, self.flow_hidden, self.n_flows, self.flow_scale)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, half),
            nn.BatchNorm1d(half),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(half, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh(),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        for module in self.modules():
            if isinstance(module, ScaleTranslate):
                nn.init.zeros_(module.net[-1].weight)
                nn.init.zeros_(module.net[-1].bias)

    def _reset(self) -> None:
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None and module is not self:
                reset()
        self._initialize_weights()

    @classmethod
    def from_hparams(cls, hparams: dict[str, float | int]) -> FVAE:
        """Build the architecture stored in a tune result or a checkpoint."""
        return cls(
            input_dim=int(hparams["in_features"]),
            latent_dim=int(hparams["latent_dim"]),
            hidden_dim=int(hparams["hidden_dim"]),
            dropout=float(hparams["dropout"]),
            n_flows=int(hparams["n_flows"]),
            flow_hidden=int(hparams["flow_hidden"]),
            flow_scale=float(hparams["flow_scale"]),
        )

    def encode(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(flat)
        mu = self.fc_mu(hidden)
        logvar = torch.clamp(self.fc_logvar(hidden), min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        mu, logvar = self.encode(flat)
        z0 = self.reparameterize(mu, logvar)
        zk, log_det = self.flow(z0)
        reconstructed = self.decode(zk)
        return reconstructed.reshape_as(x), mu, logvar, z0, zk, log_det

    @staticmethod
    def data_loaders(
        train_x: np.ndarray,
        eval_x: np.ndarray,
        *,
        batch_size: int,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Shuffled train loader and an ordered eval loader.

        Each batch is ``(image, image)``. The reconstruction target is the image.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {"pin_memory": True} if torch.cuda.is_available() else {}
        drop_last = len(train_x) > batch_size and len(train_x) % batch_size == 1
        train_loader = DataLoader(
            _tensor_pair(train_x),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=drop_last,
            **loader_kwargs,
        )
        eval_loader = DataLoader(
            _tensor_pair(eval_x),
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, eval_loader

    def fit(
        self,
        train_x: np.ndarray,
        eval_x: np.ndarray,
        hparams: dict[str, float | int],
        *,
        epochs: int,
        seed: int,
    ) -> dict[str, list[float]]:
        """Adam on the ELBO. The eval curve is recorded and does not pick weights."""
        if len(train_x) == 0 or len(eval_x) == 0:
            raise ValueError("train and eval splits must both be non-empty")
        train_loader, eval_loader = self.data_loaders(
            train_x,
            eval_x,
            batch_size=int(hparams["batch_size"]),
            seed=seed,
        )
        return self.fit_from_dataset(
            train_loader,
            eval_loader,
            hparams,
            epochs=epochs,
            seed=seed,
        )

    def fit_from_dataset(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        hparams: dict[str, float | int],
        *,
        epochs: int,
        seed: int,
    ) -> dict[str, list[float]]:
        """Adam on the sampled ELBO, reading batches from data loaders.

        The eval loss is recorded and does not pick weights.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if len(train_loader) == 0 or len(eval_loader) == 0:
            raise ValueError("train and eval splits must both be non-empty")
        beta = float(hparams["beta"])
        seed_all(seed)
        self.cpu()
        self._reset()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
        history: dict[str, list[float]] = {"train": [], "eval": []}
        for epoch in range(epochs):
            self.train()
            total = 0.0
            det_total = 0.0
            seen = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstructed, mu, logvar, z0, zk, log_det = self(batch_x)
                loss = _fvae_loss(reconstructed, batch_y, mu, logvar, z0, zk, log_det, beta)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), _GRAD_CLIP)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(batch_y)
                det_total += float(log_det.detach().sum().cpu())
                seen += len(batch_y)
            history["train"].append(total / seen)
            history["eval"].append(_mean_loss(self, eval_loader, device, beta))
            print(
                f"epoch {epoch + 1}/{epochs} train={history['train'][-1]:.4f} "
                f"eval={history['eval'][-1]:.4f} log_det={det_total / seen:.4f}",
                flush=True,
            )
        self.eval()
        self.cpu()
        return history

    @torch.inference_mode()
    def reconstruct(self, pixels: np.ndarray, batch_size: int) -> np.ndarray:
        """Flow the base posterior mean, then decode. Layout matches ``pixels``."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        device = next(self.parameters()).device
        self.eval()
        chunks: list[np.ndarray] = []
        for start in range(0, len(pixels), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(pixels[start : start + batch_size])).to(
                device
            )
            flat = batch.reshape(batch.shape[0], -1)
            mu, _logvar = self.encode(flat)
            zk, _log_det = self.flow(mu)
            chunks.append(self.decode(zk).reshape_as(batch).detach().cpu().numpy())
        if not chunks:
            raise ValueError("reconstruct needs at least one image")
        return np.concatenate(chunks, axis=0)

    @torch.inference_mode()
    def score(
        self,
        pixels: np.ndarray,
        batch_size: int,
        *,
        beta: float,
        n_samples: int = 8,
    ) -> dict[str, float]:
        """Reconstruction of the flowed mean, plus a Monte Carlo ELBO.

        ``mse`` and ``mae`` are per pixel on ``decode(flow(mu))``. ``kl`` and
        ``log_det`` average ``n_samples`` draws. ``elbo`` is the negative of
        the training objective on those draws. ``bpd`` is that bound in bits
        per dimension for the matching Gaussian decoder.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        device = next(self.parameters()).device
        self.eval()
        totals = {"mse": 0.0, "mae": 0.0, "kl": 0.0, "sse": 0.0, "log_det": 0.0, "loss": 0.0}
        seen = 0
        for start in range(0, len(pixels), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(pixels[start : start + batch_size])).to(
                device
            )
            flat = batch.reshape(batch.shape[0], -1)
            mu, logvar = self.encode(flat)
            zk_mean, _det_mean = self.flow(mu)
            recon = self.decode(zk_mean).reshape_as(batch)
            per = (recon - batch).flatten(1)
            n = len(batch)
            width = flat.shape[1]
            totals["mse"] += float((per.pow(2).sum(dim=1) / width).sum())
            totals["mae"] += float(per.abs().mean(dim=1).sum())
            kl_acc = torch.zeros(n, device=device)
            sse_acc = torch.zeros(n, device=device)
            det_acc = torch.zeros(n, device=device)
            for _sample in range(n_samples):
                z0 = self.reparameterize(mu, logvar)
                zk, log_det = self.flow(z0)
                sampled = self.decode(zk).reshape_as(batch)
                sse_acc = sse_acc + (sampled - batch).flatten(1).pow(2).sum(dim=1)
                kl_acc = kl_acc + _flow_kl(mu, logvar, z0, zk, log_det)
                det_acc = det_acc + log_det
            sse_mean = sse_acc / n_samples
            kl_mean = kl_acc / n_samples
            totals["sse"] += float(sse_mean.sum())
            totals["kl"] += float(kl_mean.sum())
            totals["log_det"] += float((det_acc / n_samples).sum())
            totals["loss"] += float((sse_mean + float(beta) * kl_mean).sum())
            seen += n
        if seen == 0:
            raise ValueError("score needs at least one image")
        means = {key: value / seen for key, value in totals.items()}
        sse = means.pop("sse")
        means["elbo"] = -means.pop("loss")
        means["bpd"] = bpd_from_objective(sse, means["kl"], self.input_dim)
        return means

    @torch.inference_mode()
    def sample(self, n: int, *, height: int = 28, width: int = 28) -> np.ndarray:
        """Decode ``n`` draws from the standard normal prior. Layout is NCHW."""
        if n < 1:
            raise ValueError("sample needs at least one draw")
        if height * width != self.input_dim:
            raise ValueError("sample spatial size must match in_features")
        device = next(self.parameters()).device
        self.eval()
        flat = self.decode(torch.randn(n, self.latent_dim, device=device))
        return flat.reshape(n, 1, height, width).detach().cpu().numpy()

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
                "hparams": {key: hparams[key] for key in _SAVED_KEYS},
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
                "mnist f-vae must be a torch checkpoint dict with state_dict and columns, "
                f"got {type(blob).__name__}"
            )
        hparams = blob.get("hparams")
        if not isinstance(hparams, dict) or any(key not in hparams for key in _ARCH_KEYS):
            raise TypeError("mnist f-vae checkpoint is missing architecture hparams")
        model = cls.from_hparams(hparams)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return {"model": model, "columns": list(blob["columns"]), "hparams": dict(hparams)}


def _env(name: str) -> str:
    return os.getenv(f"MNIST_FVAE_{name}", "").strip()


def _cache_dir() -> Path:
    override = _env("CACHE")
    if override:
        return Path(override)
    local = os.getenv("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "mnist-f-vae"
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "mnist-f-vae"


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
            raise FileNotFoundError(f"mnist f-vae not found: {chosen}")
        return chosen
    return fetch_cached_model()
