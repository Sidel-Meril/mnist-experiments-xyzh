"""Stage 01: load MNIST digits and Fashion-MNIST.

``queries/images.sql`` is a mock of the extract. It is read and then ignored.
Rows come from ``torchvision.datasets.MNIST`` and ``FashionMNIST``.

Usage::

    python src/mnist_realnvp/01_get_data.py --sql src/mnist_realnvp/queries/images.sql \
        --out data/mnist_realnvp/raw_images.parquet --images data/mnist_realnvp/raw_images.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mnist_realnvp.config import CACHE, DIGIT_NAMES, FASHION_NAMES, IMAGE_HWC


def load_data(
    sql_path: Path,
    out_path: Path,
    images_path: Path,
    *,
    sample: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Return the image index. The SQL file is not executed.

    ``sample`` is ``(uint8 NHWC images, labels, splits, datasets)``. The pipeline
    leaves it unset and downloads both MNIST sets. Tests pass a small array instead.
    """
    statement = sql_path.read_text(encoding="utf-8")
    if "select" not in statement.lower():
        raise ValueError(f"mock sql has no SELECT: {sql_path}")
    images, labels, splits, datasets = _download() if sample is None else _as_arrays(sample)
    frame = _frame(images, labels, splits, datasets)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    np.save(images_path, np.ascontiguousarray(images))
    print(f"loaded {len(frame)} rows -> {out_path}; images -> {images_path}")
    return frame


def _download() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST, FashionMNIST

    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    datasets: list[np.ndarray] = []
    for name, dataset_cls in (("digits", MNIST), ("fashion", FashionMNIST)):
        train = dataset_cls(root=str(CACHE), train=True, download=True)
        test = dataset_cls(root=str(CACHE), train=False, download=True)
        train_images = _nhwc(train.data.numpy())
        test_images = _nhwc(test.data.numpy())
        images.extend((train_images, test_images))
        labels.extend(
            (
                np.asarray(train.targets, dtype=np.int64),
                np.asarray(test.targets, dtype=np.int64),
            )
        )
        splits.extend(
            (
                np.array(["train"] * len(train_images)),
                np.array(["test"] * len(test_images)),
            )
        )
        datasets.extend(
            (
                np.array([name] * len(train_images)),
                np.array([name] * len(test_images)),
            )
        )
    return (
        np.concatenate(images, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(splits, axis=0),
        np.concatenate(datasets, axis=0),
    )


def _nhwc(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim == 3:
        array = array[..., None]
    return np.ascontiguousarray(array)


def _as_arrays(
    sample: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images, labels, splits, datasets = sample
    return np.asarray(images), np.asarray(labels), np.asarray(splits), np.asarray(datasets)


def _frame(
    images: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    datasets: np.ndarray,
) -> pd.DataFrame:
    if images.dtype != np.uint8 or images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_HWC:
        raise TypeError(f"raw MNIST images must be uint8 NHWC {IMAGE_HWC}, got {images.shape}")
    if len(images) != len(labels) or len(images) != len(splits) or len(images) != len(datasets):
        raise ValueError("images, labels, splits, and datasets must have the same length")
    label_ids = np.asarray(labels, dtype=np.int64)
    source = np.asarray(datasets, dtype=str)
    names = np.empty(len(images), dtype=object)
    digit_rows = source == "digits"
    fashion_rows = source == "fashion"
    if not np.all(digit_rows | fashion_rows):
        raise ValueError("dataset must be digits or fashion")
    # Out-of-range labels stay in the table so the modeling contract can reject them.
    name_index = np.clip(label_ids, 0, len(DIGIT_NAMES) - 1)
    names[digit_rows] = np.asarray(DIGIT_NAMES, dtype=object)[name_index[digit_rows]]
    names[fashion_rows] = np.asarray(FASHION_NAMES, dtype=object)[name_index[fashion_rows]]
    return pd.DataFrame(
        {
            "image_id": np.arange(1, len(images) + 1),
            "label": label_ids,
            "mnist_split": pd.Series(np.asarray(splits, dtype=str), dtype="string"),
            "dataset": pd.Series(source, dtype="string"),
            "label_name": pd.Series(names, dtype="string"),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="01_get_data", description=__doc__)
    parser.add_argument("--sql", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args(argv)
    load_data(args.sql, args.out, args.images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
