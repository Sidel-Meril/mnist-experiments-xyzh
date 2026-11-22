"""Stage 06: score a trained progressive GAN and write a generated batch.

Samples are drawn at the final 32x32 resolution. ``sample_std`` is the mean
per-pixel standard deviation across one fixed-noise batch. ``mode_collapse``
is true when that batch is much tighter than the real images.

Usage::

    python src/mnist_pg_gan/06_eval.py \
        --data data/mnist_pg_gan/images.parquet --pixels data/mnist_pg_gan/pixels.npy \
        --split data/mnist_pg_gan/split.json --model models/mnist_pg_gan/mnist_pg_gan.pt \
        --train-metrics reports/mnist_pg_gan/evaluation_train.json \
        --test-metrics reports/mnist_pg_gan/evaluation_test.json \
        --generated reports/mnist_pg_gan/plots/generated.png \
        --real reports/mnist_pg_gan/plots/real_test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mnist_pg_gan.config import COLLAPSE_STD_RATIO, MODELING_CONTRACT_PATH, PREVIEW_COUNT
from mnist_pg_gan.figures import save_grid, unit_interval
from mnist_pg_gan.model import ProgressiveGAN
from input_contracts.model_contract import ModelingContract


def _write(path: Path, metrics: dict[str, float | int | bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _metrics(
    model: ProgressiveGAN,
    pixels: np.ndarray,
    *,
    batch_size: int,
    n_train: int,
    n_test: int,
    noise_dim: int,
    image_size: int,
) -> dict[str, float | int | bool]:
    discriminator_loss, generator_loss = model.adversarial_losses(pixels, batch_size)
    real = pixels[: min(PREVIEW_COUNT, len(pixels))]
    generated = model.generate(len(real))
    real_std = _flat_std(real)
    sample_std = _flat_std(generated)
    collapsed = sample_std < COLLAPSE_STD_RATIO * real_std if real_std > 0 else sample_std < 1e-3
    return {
        "discriminator_loss": discriminator_loss,
        "generator_loss": generator_loss,
        "sample_std": sample_std,
        "real_std": real_std,
        "mode_collapse": bool(collapsed),
        "noise_dim": noise_dim,
        "image_size": image_size,
        "n_train": n_train,
        "n_test": n_test,
    }


def _flat_std(images: np.ndarray) -> float:
    flat = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    if flat.shape[0] < 2:
        return 0.0
    return float(flat.std(axis=0).mean())


def run(
    data_path: Path,
    pixels_path: Path,
    split_path: Path,
    model_path: Path,
    train_metrics_path: Path,
    test_metrics_path: Path,
    *,
    generated_path: Path,
    real_path: Path,
) -> dict[str, dict[str, float | int | bool]]:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    _batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    loaded = ProgressiveGAN.load(model_path)
    model = loaded["model"]
    hparams = loaded["hparams"]
    if not isinstance(model, ProgressiveGAN) or not isinstance(hparams, dict):
        raise TypeError("checkpoint did not restore a ProgressiveGAN")
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    batch_size = int(hparams["batch_size"])
    noise_dim = int(hparams["noise_dim"])
    image_size = int(hparams["image_size"])
    n_train = int(len(train_idx))
    n_test = int(len(test_idx))
    train_pixels = contract.feature_matrix(pixels, train_idx)
    test_pixels = contract.feature_matrix(pixels, test_idx)
    written = {
        "train": _metrics(
            model,
            train_pixels,
            batch_size=batch_size,
            n_train=n_train,
            n_test=n_test,
            noise_dim=noise_dim,
            image_size=image_size,
        ),
        "test": _metrics(
            model,
            test_pixels,
            batch_size=batch_size,
            n_train=n_train,
            n_test=n_test,
            noise_dim=noise_dim,
            image_size=image_size,
        ),
    }
    _write(train_metrics_path, written["train"])
    _write(test_metrics_path, written["test"])
    preview = min(PREVIEW_COUNT, len(test_pixels), int(model.fixed_noise.shape[0]))
    generated = model.generate(preview, noise=model.fixed_noise[:preview].cpu())
    save_grid(unit_interval(generated), generated_path, "generated 32px")
    save_grid(unit_interval(test_pixels[:preview]), real_path, "real test")
    print(
        f"eval sample_std={written['test']['sample_std']:.4f} "
        f"mode_collapse={written['test']['mode_collapse']} -> {generated_path}"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="06_eval", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-metrics", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.model,
        args.train_metrics,
        args.test_metrics,
        generated_path=args.generated,
        real_path=args.real,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
