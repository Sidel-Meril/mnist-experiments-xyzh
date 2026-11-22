"""Stage 07: train one β-VAE per KL weight and compare the three effects.

Reconstruction quality is per-pixel MSE on the test split, decoding the
posterior mean. Disentanglement is the count of latent axes whose posterior
mean still varies across those images. Generation diversity is the mean
pairwise per-pixel squared distance among prior samples. Each beta also
gets a latent traversal: one row per axis, walking that axis while the
other axes stay at 0.

The architecture comes from the tune result. Only beta changes. The test
split is scored after training. It does not pick the weights.

Usage::

    python src/mnist_betta_vae/07_explore.py \
        --data data/mnist_betta_vae/images.parquet --pixels data/mnist_betta_vae/pixels.npy \
        --split data/mnist_betta_vae/split.json --params reports/mnist_betta_vae/best_params.json \
        --out reports/mnist_betta_vae/beta_exploration.json --models models \
        --plots reports/mnist_betta_vae/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_betta_vae.config import (
    ACTIVE_UNIT_THRESHOLD,
    BETAS,
    DIVERSITY_SAMPLES,
    MODELING_CONTRACT_PATH,
    PIXEL_MEAN,
    PIXEL_STD,
    RANDOM_SEED,
    TRAVERSAL_SPAN,
    TRAVERSAL_STEPS,
)
from mnist_betta_vae.model import VAE
from mnist_contracts.model_contract import ModelingContract


def beta_stem(beta: float) -> str:
    """File stem for one KL weight, with the decimal point turned into an underscore."""
    return f"beta_{float(beta):g}".replace(".", "_")


def _gray(images: np.ndarray) -> np.ndarray:
    scale = np.float32(PIXEL_STD)
    shift = np.float32(PIXEL_MEAN)
    return np.clip(images[:, 0] * scale + shift, 0, 1)


def _save_traversal(images: np.ndarray, path: Path, *, beta: float) -> None:
    n_dim, steps = images.shape[0], images.shape[1]
    flat = _gray(images.reshape(-1, *images.shape[2:]))
    gray = flat.reshape(n_dim, steps, *flat.shape[-2:])
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(n_dim, steps, figsize=(1.15 * steps, 1.05 * n_dim), squeeze=False)
    for dim in range(n_dim):
        for step in range(steps):
            axes[dim, step].imshow(gray[dim, step], cmap="gray", vmin=0, vmax=1)
            axes[dim, step].axis("off")
        axes[dim, 0].set_ylabel(f"z{dim}", rotation=0, ha="right", va="center")
    fig.suptitle(f"latent traversal, beta={beta:g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _traversal_response(images: np.ndarray) -> list[float]:
    """Mean pixel variance along each latent walk. A flat row barely changes the image."""
    flat = images.reshape(images.shape[0], images.shape[1], -1)
    return [float(flat[dim].var(axis=0).mean()) for dim in range(flat.shape[0])]


def _save_metric_curves(rows: list[dict[str, object]], path: Path) -> None:
    betas = [float(row["beta"]) for row in rows]
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    series = (
        ("mse", "reconstruction MSE"),
        ("active_units", "active latent units"),
        ("diversity", "generation diversity"),
    )
    for axis, (key, title) in zip(axes, series, strict=True):
        axis.plot(betas, [float(row[key]) for row in rows], marker="o")
        axis.set_xlabel("beta")
        axis.set_title(title)
        axis.set_xticks(betas)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(
    data_path: Path,
    pixels_path: Path,
    split_path: Path,
    params_path: Path,
    out_path: Path,
    *,
    models_dir: Path,
    plots_dir: Path,
    epochs: int = 20,
    betas: tuple[float, ...] = BETAS,
) -> dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if len(betas) < 1:
        raise ValueError("betas must contain at least one weight")
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    _batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    base = dict(json.loads(params_path.read_text(encoding="utf-8"))["params"])
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    train_x = contract.feature_matrix(pixels, train_idx)
    test_x = contract.feature_matrix(pixels, test_idx)
    rows: list[dict[str, object]] = []
    for beta in betas:
        hparams = dict(base)
        hparams["beta"] = float(beta)
        stem = beta_stem(beta)
        model = VAE.from_hparams(hparams)
        train_loader, test_loader = model.data_loaders(
            train_x,
            test_x,
            batch_size=int(hparams["batch_size"]),
            seed=RANDOM_SEED,
        )
        model.fit_from_dataset(
            train_loader,
            test_loader,
            hparams,
            epochs=epochs,
            seed=RANDOM_SEED,
        )
        model_path = models_dir / f"{stem}.pt"
        model.save(model_path, list(contract.features), hparams)
        batch_size = int(hparams["batch_size"])
        scored = model.score(test_x, batch_size, beta=float(beta))
        report = model.latent_report(
            test_x,
            batch_size,
            active_threshold=ACTIVE_UNIT_THRESHOLD,
        )
        frames = model.traverse(steps=TRAVERSAL_STEPS, span=TRAVERSAL_SPAN)
        traversal_path = plots_dir / f"traversal_{stem}.png"
        _save_traversal(frames, traversal_path, beta=float(beta))
        row: dict[str, object] = {
            "beta": float(beta),
            "mse": scored["mse"],
            "mae": scored["mae"],
            "kl": scored["kl"],
            "elbo": scored["elbo"],
            "active_units": report["active_units"],
            "kl_per_dim": report["kl_per_dim"],
            "mu_variance": report["mu_variance"],
            "diversity": model.generation_diversity(DIVERSITY_SAMPLES, seed=RANDOM_SEED),
            "traversal_response": _traversal_response(frames),
            "model": model_path.name,
            "traversal": traversal_path.name,
        }
        rows.append(row)
        print(
            f"beta={float(beta):g} mse={scored['mse']:.4f} "
            f"active_units={report['active_units']} diversity={row['diversity']:.4f} "
            f"-> {model_path}"
        )
    payload: dict[str, object] = {
        "latent_dim": int(base["latent_dim"]),
        "epochs": epochs,
        "active_unit_threshold": ACTIVE_UNIT_THRESHOLD,
        "traversal_steps": TRAVERSAL_STEPS,
        "traversal_span": TRAVERSAL_SPAN,
        "diversity_samples": DIVERSITY_SAMPLES,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "runs": rows,
    }
    _save_metric_curves(rows, plots_dir / "metrics_vs_beta.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"beta exploration -> {out_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="07_explore", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--plots", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.params,
        args.out,
        models_dir=args.models,
        plots_dir=args.plots,
        epochs=args.epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
