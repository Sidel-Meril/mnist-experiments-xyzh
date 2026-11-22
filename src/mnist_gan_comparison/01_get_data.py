"""Stage 01: load MNIST digits or Fashion-MNIST.

``queries/images.sql`` is a mock of the extract. It is read and then ignored.
Rows come from ``torchvision.datasets.MNIST`` or ``FashionMNIST``. Both
datasets write the same columns. ``scripts/mnist_gan_comparison/pull_sample_data.py`` picks which
one lands in ``data/mnist_gan_comparison/`` before the shared train stages run.

Usage::

    python src/mnist_gan_comparison/01_get_data.py --sql src/mnist_gan_comparison/queries/images.sql \
        --out data/mnist_gan_comparison/raw_images.parquet --images data/mnist_gan_comparison/raw_images.npy --dataset digits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mnist_gan_comparison.config import CACHE, DATASETS, IMAGE_HWC


def load_data(
    sql_path: Path,
    out_path: Path,
    images_path: Path,
    *,
    dataset: str = "digits",
    sample: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Return the image index. The SQL file is not executed.

    ``sample`` is ``(uint8 NHWC images, labels, splits)``. The pipeline leaves it
    unset and downloads ``dataset``. Tests pass a small array instead.
    """
    statement = sql_path.read_text(encoding="utf-8")
    if "select" not in statement.lower():
        raise ValueError(f"mock sql has no SELECT: {sql_path}")
    images, labels, splits = _download(dataset) if sample is None else _as_arrays(sample)
    frame = _frame(images, labels, splits)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    np.save(images_path, np.ascontiguousarray(images))
    print(f"loaded {len(frame)} rows ({dataset}) -> {out_path}; images -> {images_path}")
    return frame


def _download(dataset: str = "digits") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST, FashionMNIST

    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    dataset_cls = MNIST if dataset == "digits" else FashionMNIST
    train = dataset_cls(root=str(CACHE), train=True, download=True)
    test = dataset_cls(root=str(CACHE), train=False, download=True)
    train_images = np.asarray(train.data)
    test_images = np.asarray(test.data)
    images = np.concatenate([train_images, test_images], axis=0)[..., None]
    labels = np.concatenate(
        [
            np.asarray(train.targets, dtype=np.int64),
            np.asarray(test.targets, dtype=np.int64),
        ]
    )
    splits = np.array(["train"] * len(train_images) + ["test"] * len(test_images))
    return np.ascontiguousarray(images), labels, splits


def _as_arrays(
    sample: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images, labels, splits = sample
    return np.asarray(images), np.asarray(labels), np.asarray(splits)


def _frame(images: np.ndarray, labels: np.ndarray, splits: np.ndarray) -> pd.DataFrame:
    if images.dtype != np.uint8 or images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_HWC:
        raise TypeError(f"raw MNIST images must be uint8 NHWC {IMAGE_HWC}, got {images.shape}")
    if len(images) != len(labels) or len(images) != len(splits):
        raise ValueError("images, labels, and splits must have the same length")
    return pd.DataFrame(
        {
            "image_id": np.arange(1, len(images) + 1),
            "label": labels,
            "mnist_split": pd.Series(splits, dtype="string"),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="01_get_data", description=__doc__)
    parser.add_argument("--sql", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="digits")
    args = parser.parse_args(argv)
    load_data(args.sql, args.out, args.images, dataset=args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
