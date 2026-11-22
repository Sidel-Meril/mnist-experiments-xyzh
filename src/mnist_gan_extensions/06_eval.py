"""Stage 06: score a trained conditional GAN and write sample grids.

``sample_std`` is the standard deviation of one batch drawn with the test
split's labels. ``real_std`` is the same statistic on those real images.
``mode_collapse`` is true when the generated batch is much tighter than the
real images. ``diversity`` is the mean within-class standard deviation of a
class-major grid. ``class_separation`` is the standard deviation of the
per-class means. ``sharpness`` is the mean absolute neighbor difference.

Usage::

    python src/mnist_gan_extensions/06_eval.py \
        --data data/mnist_gan_extensions/images.parquet --pixels data/mnist_gan_extensions/pixels.npy \
        --split data/mnist_gan_extensions/split.json --model models/mnist_gan_extensions/mnist_cgan.pt \
        --train-metrics reports/mnist_gan_extensions/evaluation_train.json \
        --test-metrics reports/mnist_gan_extensions/evaluation_test.json \
        --generated reports/mnist_gan_extensions/plots/generated.png \
        --real reports/mnist_gan_extensions/plots/real_test.png \
        --classes reports/mnist_gan_extensions/plots/classes.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mnist_gan_extensions.config import (
    CLASS_PREVIEW,
    COLLAPSE_STD_RATIO,
    MODELING_CONTRACT_PATH,
    N_CLASSES,
    PREVIEW_COUNT,
    RANDOM_SEED,
)
from mnist_gan_extensions.figures import save_class_grid, save_grid, unit_interval
from mnist_gan_extensions.model import ConditionalGAN
from input_contracts.model_contract import ModelingContract


def _write(path: Path, metrics: dict[str, float | int | bool | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _flat_std(images: np.ndarray) -> float:
    flat = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    if flat.shape[0] < 2:
        return 0.0
    return float(flat.std())


def _sharpness(images: np.ndarray) -> float:
    array = np.asarray(images, dtype=np.float64)
    if array.ndim == 4:
        array = array[:, 0]
    if array.shape[0] < 1 or array.shape[1] < 2 or array.shape[2] < 2:
        return 0.0
    dy = float(np.abs(np.diff(array, axis=1)).mean())
    dx = float(np.abs(np.diff(array, axis=2)).mean())
    return dx + dy


def _diversity(images: np.ndarray, n_classes: int) -> tuple[float, float]:
    per_class = images.shape[0] // n_classes
    chunks = np.asarray(images, dtype=np.float64).reshape(n_classes, per_class, -1)
    within = float(np.mean([chunk.std() for chunk in chunks]))
    separation = float(chunks.mean(axis=1).std())
    return within, separation


def _metrics(
    model: ConditionalGAN,
    pixels: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    n_train: int,
    n_test: int,
    noise_dim: int,
) -> tuple[dict[str, float | int | bool | str], np.ndarray]:
    discriminator_loss, generator_loss, generator_bce = model.score_split(
        pixels, labels, batch_size
    )
    count = min(PREVIEW_COUNT, len(pixels))
    generated = model.generate(count, labels=labels[:count])
    real_std = _flat_std(pixels[:count])
    sample_std = _flat_std(generated)
    collapsed = sample_std < COLLAPSE_STD_RATIO * real_std if real_std > 0 else sample_std < 1e-3
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    class_noise = torch.randn(N_CLASSES * CLASS_PREVIEW, noise_dim, generator=generator)
    class_images = model.class_batch(CLASS_PREVIEW, noise=class_noise)
    diversity, separation = _diversity(class_images, N_CLASSES)
    metrics: dict[str, float | int | bool | str] = {
        "discriminator_loss": discriminator_loss,
        "generator_loss": generator_loss,
        "generator_bce": generator_bce,
        "sample_std": sample_std,
        "real_std": real_std,
        "mode_collapse": bool(collapsed),
        "noise_dim": noise_dim,
        "n_train": n_train,
        "n_test": n_test,
        "variant": model.variant,
        "diversity": diversity,
        "class_separation": separation,
        "sharpness": _sharpness(class_images),
        "real_sharpness": _sharpness(pixels[:count]),
    }
    return metrics, class_images


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
    classes_path: Path,
) -> dict[str, dict[str, float | int | bool | str]]:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    loaded = ConditionalGAN.load(model_path)
    model = loaded["model"]
    hparams = loaded["hparams"]
    if not isinstance(model, ConditionalGAN) or not isinstance(hparams, dict):
        raise TypeError("checkpoint did not restore a ConditionalGAN")
    labels = batch[contract.target].detach().cpu().numpy().astype(np.int64)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    batch_size = int(hparams["batch_size"])
    noise_dim = int(hparams["noise_dim"])
    n_train = int(len(train_idx))
    n_test = int(len(test_idx))
    train_metrics, _train_classes = _metrics(
        model,
        contract.feature_matrix(pixels, train_idx),
        labels[train_idx],
        batch_size=batch_size,
        n_train=n_train,
        n_test=n_test,
        noise_dim=noise_dim,
    )
    test_pixels = contract.feature_matrix(pixels, test_idx)
    test_labels = labels[test_idx]
    test_metrics, class_images = _metrics(
        model,
        test_pixels,
        test_labels,
        batch_size=batch_size,
        n_train=n_train,
        n_test=n_test,
        noise_dim=noise_dim,
    )
    written = {"train": train_metrics, "test": test_metrics}
    _write(train_metrics_path, written["train"])
    _write(test_metrics_path, written["test"])
    preview = min(PREVIEW_COUNT, len(test_pixels), int(model.fixed_noise.shape[0]))
    generated = model.generate(
        preview,
        labels=test_labels[:preview],
        noise=model.fixed_noise[:preview].cpu(),
    )
    save_grid(unit_interval(generated), generated_path, f"generated {model.variant}")
    save_grid(unit_interval(test_pixels[:preview]), real_path, "real test")
    save_class_grid(
        unit_interval(class_images),
        classes_path,
        f"{model.variant} classes",
        n_classes=N_CLASSES,
    )
    print(
        f"eval variant={model.variant} sample_std={written['test']['sample_std']:.4f} "
        f"diversity={written['test']['diversity']:.4f} -> {generated_path}"
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
    parser.add_argument("--classes", type=Path, required=True)
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
        classes_path=args.classes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
