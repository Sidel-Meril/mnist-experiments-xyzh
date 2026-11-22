"""MNIST variational autoencoder, training, and the DVC checkpoint.

The network is the ``VAE`` in ``VAE_Tutorial.ipynb``: encoder, ``fc_mu``,
``fc_logvar``, and the reparameterization trick. The decoder ends in ``Tanh``
because stage 02 stores pixels in [-1, 1]. The tutorial averages the ELBO over
every element. On a 784-d image that average collapses the posterior, so the
loss sums squared error over pixels and the KL terms over latent units, then
averages the batch. Beta stays 1. Reported reconstructions decode the
posterior mean. Generated samples decode draws from the unit normal prior.

``VAE.load`` uses a local checkpoint when one is already on disk. Otherwise
the latest fit is read from the DVC remote in ``.dvc/config`` with
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

from mnist_vae_comparison.config import ROOT

_MODEL_REPO = "https://github.com/Sidel-Meril/mnist-vae-comparison-xyzh.git"
_MODEL_REMOTE = "demo"
_MODEL_RELATIVE = "models/mnist_vae_comparison/mnist_vae.pt"
LOCAL_MODEL = ROOT / "models" / "mnist_vae_comparison" / "mnist_vae.pt"
_session: tuple[str, str | None, Path] | None = None

_ARCH_KEYS = ("latent_dim", "hidden_dim", "dropout", "in_features")
_SAVED_KEYS = (*_ARCH_KEYS, "lr", "weight_decay", "batch_size", "beta")
_GRAD_CLIP = 1.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    per_image = (reconstructed - original).flatten(1).pow(2).sum(dim=1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return per_image.mean() + float(beta) * kl.mean()


def _mean_loss(model: nn.Module, loader: DataLoader, device: torch.device, beta: float) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            reconstructed, mu, logvar = model(batch_x)
            loss = _vae_loss(reconstructed, batch_y, mu, logvar, beta)
            total += float(loss.detach().cpu()) * len(batch_y)
            seen += len(batch_y)
    if seen == 0:
        raise ValueError("train and eval splits must both be non-empty")
    return total / seen


def _tensor_pair(features: np.ndarray) -> TensorDataset:
    pixels = torch.from_numpy(np.ascontiguousarray(features))
    return TensorDataset(pixels, pixels.clone())


class VAE(nn.Module):
    """Variational autoencoder. ``fc_mu`` and ``fc_logvar`` parameterize q(z|x).

    ``load`` rebuilds this class from a local checkpoint or the DVC remote.
    ``save`` writes ``state_dict``, ``columns``, and ``hparams``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be at least 1")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
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

    def _reset(self) -> None:
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None and module is not self:
                reset()
        self._initialize_weights()

    @classmethod
    def from_hparams(cls, hparams: dict[str, float | int]) -> VAE:
        """Build the architecture stored in a tune result or a checkpoint."""
        return cls(
            input_dim=int(hparams["in_features"]),
            latent_dim=int(hparams["latent_dim"]),
            hidden_dim=int(hparams["hidden_dim"]),
            dropout=float(hparams["dropout"]),
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        mu, logvar = self.encode(flat)
        reconstructed = self.decode(self.reparameterize(mu, logvar))
        return reconstructed.reshape_as(x), mu, logvar

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
        for _epoch in range(epochs):
            self.train()
            total = 0.0
            seen = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstructed, mu, logvar = self(batch_x)
                loss = _vae_loss(reconstructed, batch_y, mu, logvar, beta)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), _GRAD_CLIP)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(batch_y)
                seen += len(batch_y)
            history["train"].append(total / seen)
            history["eval"].append(_mean_loss(self, eval_loader, device, beta))
        self.eval()
        self.cpu()
        return history

    @torch.inference_mode()
    def reconstruct(self, pixels: np.ndarray, batch_size: int) -> np.ndarray:
        """Posterior-mean reconstructions in the same layout and scale as ``pixels``."""
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
            chunks.append(self.decode(mu).reshape_as(batch).detach().cpu().numpy())
        if not chunks:
            raise ValueError("reconstruct needs at least one image")
        return np.concatenate(chunks, axis=0)

    @torch.inference_mode()
    def score(self, pixels: np.ndarray, batch_size: int, *, beta: float) -> dict[str, float]:
        """Per-pixel MSE and MAE, per-image KL, and ELBO, decoding the posterior mean."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        device = next(self.parameters()).device
        self.eval()
        totals = {"mse": 0.0, "mae": 0.0, "kl": 0.0, "loss": 0.0}
        seen = 0
        for start in range(0, len(pixels), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(pixels[start : start + batch_size])).to(
                device
            )
            flat = batch.reshape(batch.shape[0], -1)
            mu, logvar = self.encode(flat)
            recon = self.decode(mu).reshape_as(batch)
            per = (recon - batch).flatten(1)
            sse = per.pow(2).sum(dim=1)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            n = len(batch)
            totals["mse"] += float((sse / flat.shape[1]).sum())
            totals["mae"] += float(per.abs().mean(dim=1).sum())
            totals["kl"] += float(kl.sum())
            totals["loss"] += float((sse + float(beta) * kl).sum())
            seen += n
        if seen == 0:
            raise ValueError("score needs at least one image")
        means = {key: value / seen for key, value in totals.items()}
        means["elbo"] = -means.pop("loss")
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
                "mnist vae must be a torch checkpoint dict with state_dict and columns, "
                f"got {type(blob).__name__}"
            )
        hparams = blob.get("hparams")
        if not isinstance(hparams, dict) or any(key not in hparams for key in _ARCH_KEYS):
            raise TypeError("mnist vae checkpoint is missing architecture hparams")
        model = cls.from_hparams(hparams)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return {"model": model, "columns": list(blob["columns"]), "hparams": dict(hparams)}


def _env(name: str) -> str:
    return os.getenv(f"MNIST_VAE_{name}", "").strip()


def _cache_dir() -> Path:
    override = _env("CACHE")
    if override:
        return Path(override)
    local = os.getenv("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "mnist-vae-model"
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "mnist-vae-model"


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
            raise FileNotFoundError(f"mnist vae not found: {chosen}")
        return chosen
    return fetch_cached_model()
