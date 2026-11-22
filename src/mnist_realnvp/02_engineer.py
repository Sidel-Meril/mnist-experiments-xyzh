"""Stage 02: scale MNIST pixels into [0, 1] and build the modeling table.

RealNVP dequantizes these values inside the model. This stage only divides by 255
and stores NCHW float32.

Usage::

    python src/mnist_realnvp/02_engineer.py \
        --raw data/mnist_realnvp/raw_images.parquet --images data/mnist_realnvp/raw_images.npy \
        --out data/mnist_realnvp/images.parquet --pixels data/mnist_realnvp/pixels.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensordict import TensorDict

from mnist_realnvp.config import DATA_CONTRACT_PATH, MODELING_CONTRACT_PATH, PIXEL_MAX
from mnist_contracts.data_contract import DataContract
from mnist_contracts.model_contract import ModelingContract


def run(raw_path: Path, images_path: Path, out_path: Path, pixels_path: Path) -> TensorDict:
    """Apply the data contract, then write [0, 1] NCHW pixels and the modeling table."""
    raw = pd.read_parquet(raw_path)
    images = np.load(images_path)
    if len(images) != len(raw):
        raise ValueError(f"images rows {len(images)} do not match table rows {len(raw)}")
    data_contract = DataContract.load(DATA_CONTRACT_PATH)
    model_contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    cleaned = data_contract.validate(data_contract.apply(raw))
    pixels = _normalize(images)
    modeled = cleaned.copy()
    modeled["pixel_mean"] = pd.Series(pixels.mean(axis=(1, 2, 3)), index=cleaned.index)
    arrays = {name: np.asarray(modeled[name].to_numpy()) for name in modeled.columns}
    checked = model_contract.validate(model_contract.from_columns(arrays))
    n = int(checked.batch_size[0])
    model_contract.feature_matrix(pixels, np.arange(n))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixels_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pixels_path, pixels)
    names = [column.name for column in model_contract.columns if column.storage != "npy"]
    modeled.loc[:, names].to_parquet(out_path, index=False)
    print(f"engineered {n} rows -> {out_path}; pixels -> {pixels_path}")
    return checked


def _normalize(images: np.ndarray) -> np.ndarray:
    if images.dtype != np.uint8:
        raise TypeError("engineer expects uint8 images in 0..255")
    scaled = images.astype(np.float32) / np.float32(PIXEL_MAX)
    return np.ascontiguousarray(np.transpose(scaled, (0, 3, 1, 2)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="02_engineer", description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.raw, args.images, args.out, args.pixels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
