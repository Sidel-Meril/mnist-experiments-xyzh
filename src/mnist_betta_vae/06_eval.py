"""Stage 06: score reconstruction quality and likelihood, and draw samples.

MSE and MAE are per pixel. KL is the per-image sum. ELBO is the negative of
the training objective. Reconstruction grids decode the posterior mean.
Generated samples decode draws from the unit normal prior. The latent
traversal walks each axis of the zero code from ``-span`` to ``span``.

Usage::

    python src/mnist_betta_vae/06_eval.py \
        --data data/mnist_betta_vae/images.parquet --pixels data/mnist_betta_vae/pixels.npy \
        --split data/mnist_betta_vae/split.json --model models/mnist_betta_vae/mnist_betta_vae.pt \
        --train-metrics reports/mnist_betta_vae/evaluation_train.json \
        --test-metrics reports/mnist_betta_vae/evaluation_test.json \
        --reconstructions-train reports/mnist_betta_vae/plots/reconstructions_train.png \
        --reconstructions-test reports/mnist_betta_vae/plots/reconstructions_test.png \
        --generated-samples reports/mnist_betta_vae/plots/generated_samples.png \
        --latent-traversal reports/mnist_betta_vae/plots/latent_traversals.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_betta_vae.config import (
    MODELING_CONTRACT_PATH,
    PIXEL_MEAN,
    PIXEL_STD,
    RANDOM_SEED,
    TRAVERSAL_SPAN,
    TRAVERSAL_STEPS,
)
from mnist_betta_vae.model import VAE, seed_all
from input_contracts.model_contract import ModelingContract


def _write(path: Path, metrics: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _gray(images: np.ndarray) -> np.ndarray:
    scale = np.float32(PIXEL_STD)
    shift = np.float32(PIXEL_MEAN)
    return np.clip(images[:, 0] * scale + shift, 0, 1)


def _class_rows(labels: np.ndarray, idx: np.ndarray) -> np.ndarray:
    chosen: list[int] = []
    split_labels = labels[idx]
    for label in range(10):
        hits = idx[split_labels == label]
        if len(hits):
            chosen.append(int(hits[0]))
    if not chosen:
        chosen = [int(row) for row in idx[:8]]
    if not chosen:
        raise ValueError("reconstruction grid needs at least one image")
    return np.asarray(chosen, dtype=np.int64)


def _save_reconstructions(
    original: np.ndarray,
    reconstructed: np.ndarray,
    path: Path,
) -> None:
    count = min(len(original), len(reconstructed))
    if count < 1:
        raise ValueError("reconstruction grid needs at least one image")
    original_gray = _gray(original[:count])
    restored_gray = _gray(reconstructed[:count])
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(2, count, figsize=(1.4 * count, 3.2), squeeze=False)
    for index in range(count):
        axes[0, index].imshow(original_gray[index], cmap="gray", vmin=0, vmax=1)
        axes[1, index].imshow(restored_gray[index], cmap="gray", vmin=0, vmax=1)
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_title("original")
    axes[1, 0].set_title("reconstructed")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_generated(images: np.ndarray, path: Path) -> None:
    gray = _gray(images)
    side = 8
    if len(gray) != side * side:
        raise ValueError("generated sample grid expects 64 images")
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(side, side, figsize=(8, 8))
    for axis, image in zip(axes.ravel(), gray, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    fig.suptitle("samples from N(0, I)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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


def _dataset_name(batch, contract: ModelingContract) -> str:
    column = next(item for item in contract.columns if item.name == "dataset")
    if not column.categories:
        raise ValueError("dataset column is missing categories")
    levels = np.array([str(item) for item in column.categories], dtype=object)
    codes = batch["dataset"].detach().cpu().numpy().astype(np.int64, copy=False)
    names = levels[codes]
    unique = set(names.tolist())
    if len(unique) != 1:
        raise ValueError("eval expects one dataset in the table")
    return str(next(iter(unique)))


def run(
    data_path: Path,
    pixels_path: Path,
    split_path: Path,
    model_path: Path,
    train_metrics_path: Path,
    test_metrics_path: Path,
    *,
    reconstructions_train_path: Path,
    reconstructions_test_path: Path,
    generated_samples_path: Path,
    latent_traversal_path: Path,
) -> dict[str, dict[str, float | int | str]]:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    bundle = VAE.load(model_path)
    if list(bundle["columns"]) != list(contract.features):
        raise ValueError("checkpoint columns do not match the model contract features")
    hparams = bundle["hparams"]
    if not isinstance(hparams, dict):
        raise TypeError("checkpoint hparams must be a dict")
    batch_size = int(hparams["batch_size"])
    latent_dim = int(hparams["latent_dim"])
    beta = float(hparams["beta"])
    n_train = len(split["train_idx"])
    n_test = len(split["test_idx"])
    dataset = _dataset_name(batch, contract)
    model = bundle["model"]
    if not isinstance(model, VAE):
        raise TypeError("checkpoint model must be a VAE")
    labels = batch[contract.target].detach().cpu().numpy()
    written: dict[str, dict[str, float | int | str]] = {}
    scored = (
        (
            "train",
            np.asarray(split["train_idx"], dtype=np.int64),
            train_metrics_path,
            reconstructions_train_path,
        ),
        (
            "test",
            np.asarray(split["test_idx"], dtype=np.int64),
            test_metrics_path,
            reconstructions_test_path,
        ),
    )
    for name, idx, metrics_path, grid_path in scored:
        rows = contract.feature_matrix(pixels, idx)
        metrics_body = model.score(rows, batch_size, beta=beta)
        metrics: dict[str, float | int | str] = {
            "mse": metrics_body["mse"],
            "mae": metrics_body["mae"],
            "kl": metrics_body["kl"],
            "elbo": metrics_body["elbo"],
            "beta": beta,
            "latent_dim": latent_dim,
            "dataset": dataset,
            "n_train": n_train,
            "n_test": n_test,
        }
        _write(metrics_path, metrics)
        picked = _class_rows(labels, idx)
        original = contract.feature_matrix(pixels, picked)
        restored = model.reconstruct(original, batch_size)
        _save_reconstructions(original, restored, grid_path)
        print(
            f"{name} mse={metrics['mse']:.4f} mae={metrics['mae']:.4f} "
            f"kl={metrics['kl']:.4f} elbo={metrics['elbo']:.4f} -> {metrics_path}"
        )
        print(f"{name} reconstructions -> {grid_path}")
        written[name] = metrics
    seed_all(RANDOM_SEED)
    _save_generated(model.sample(64), generated_samples_path)
    print(f"generated samples -> {generated_samples_path}")
    _save_traversal(
        model.traverse(steps=TRAVERSAL_STEPS, span=TRAVERSAL_SPAN),
        latent_traversal_path,
        beta=beta,
    )
    print(f"latent traversal -> {latent_traversal_path}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="06_eval", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-metrics", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path, required=True)
    parser.add_argument("--reconstructions-train", type=Path, required=True)
    parser.add_argument("--reconstructions-test", type=Path, required=True)
    parser.add_argument("--generated-samples", type=Path, required=True)
    parser.add_argument("--latent-traversal", type=Path, required=True)
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.model,
        args.train_metrics,
        args.test_metrics,
        reconstructions_train_path=args.reconstructions_train,
        reconstructions_test_path=args.reconstructions_test,
        generated_samples_path=args.generated_samples,
        latent_traversal_path=args.latent_traversal,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
