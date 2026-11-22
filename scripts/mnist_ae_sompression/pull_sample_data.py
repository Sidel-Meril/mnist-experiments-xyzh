"""Write an MNIST slice to data/mnist_ae_sompression/ so later stages can run without the full get_data table.

Stage 01 downloads the full set. This script keeps a few images of each class
and still writes ``data/mnist_ae_sompression/raw_images.parquet`` and ``data/mnist_ae_sompression/raw_images.npy``, the
pair engineer reads.

Usage::

    python scripts/mnist_ae_sompression/pull_sample_data.py
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import numpy as np

from mnist_ae_sompression.config import RANDOM_SEED, ROOT

SQL = ROOT / "src" / "mnist_ae_sompression" / "queries" / "images.sql"
N_CLASSES = 10
PER_CLASS_TRAIN = 40
PER_CLASS_TEST = 10


def run(
    sql_path: Path,
    out_path: Path,
    images_path: Path,
    *,
    train_per_class: int = PER_CLASS_TRAIN,
    test_per_class: int = PER_CLASS_TEST,
) -> None:
    get_data = runpy.run_path(str(ROOT / "src" / "mnist_ae_sompression" / "01_get_data.py"))
    images, labels, splits = get_data["_download"]()
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
    get_data["load_data"](sql_path, out_path, images_path, sample=sample)


def _per_class(
    images: np.ndarray, labels: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
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
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "mnist_ae_sompression" / "raw_images.parquet")
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "mnist_ae_sompression" / "raw_images.npy")
    parser.add_argument("--train-per-class", type=int, default=PER_CLASS_TRAIN)
    parser.add_argument("--test-per-class", type=int, default=PER_CLASS_TEST)
    args = parser.parse_args(argv)
    run(
        args.sql,
        args.out,
        args.images,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
