"""Write one MNIST dataset into data/mnist_gan_comparison/ so later stages stay dataset-agnostic.

Stage 01 can download either set. This script is the manual pull: it chooses
digits or Fashion-MNIST, optionally keeps a few images of each class, and
still writes ``data/mnist_gan_comparison/raw_images.parquet`` and ``data/mnist_gan_comparison/raw_images.npy``. Engineer,
split, train, and eval then run unchanged.

Usage::

    python scripts/mnist_gan_comparison/pull_sample_data.py --dataset digits
    python scripts/mnist_gan_comparison/pull_sample_data.py --dataset fashion --train-per-class 500
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import numpy as np

from mnist_gan_comparison.config import DATASETS, RANDOM_SEED, ROOT

SQL = ROOT / "src" / "mnist_gan_comparison" / "queries" / "images.sql"
N_CLASSES = 10


def run(
    sql_path: Path,
    out_path: Path,
    images_path: Path,
    *,
    dataset: str = "digits",
    train_per_class: int = 0,
    test_per_class: int = 0,
) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    get_data = runpy.run_path(str(ROOT / "src" / "mnist_gan_comparison" / "01_get_data.py"))
    if train_per_class <= 0 and test_per_class <= 0:
        get_data["load_data"](sql_path, out_path, images_path, dataset=dataset)
        return
    images, labels, splits = get_data["_download"](dataset)
    train_images, train_labels = _per_class(
        images[splits == "train"],
        labels[splits == "train"],
        train_per_class,
        RANDOM_SEED,
    )
    test_images, test_labels = _per_class(
        images[splits == "test"],
        labels[splits == "test"],
        test_per_class,
        RANDOM_SEED + 1,
    )
    sample = (
        np.concatenate([train_images, test_images], axis=0),
        np.concatenate([train_labels, test_labels], axis=0),
        np.array(["train"] * len(train_labels) + ["test"] * len(test_labels)),
    )
    get_data["load_data"](sql_path, out_path, images_path, dataset=dataset, sample=sample)


def _per_class(
    images: np.ndarray, labels: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("per-class counts must be at least 1 when a slice is requested")
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    for label in range(N_CLASSES):
        idx = np.flatnonzero(labels == label)
        if count > len(idx):
            raise ValueError(f"class {label} has {len(idx)} rows, need {count}")
        picked.append(rng.choice(idx, size=count, replace=False))
    chosen = np.concatenate(picked)
    rng.shuffle(chosen)
    return np.ascontiguousarray(images[chosen]), np.asarray(labels[chosen], dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pull_sample_data", description=__doc__)
    parser.add_argument("--sql", type=Path, default=SQL)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "mnist_gan_comparison" / "raw_images.parquet")
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "mnist_gan_comparison" / "raw_images.npy")
    parser.add_argument("--dataset", choices=DATASETS, default="digits")
    parser.add_argument("--train-per-class", type=int, default=0)
    parser.add_argument("--test-per-class", type=int, default=0)
    args = parser.parse_args(argv)
    run(
        args.sql,
        args.out,
        args.images,
        dataset=args.dataset,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
