"""Stage 02: normalize pixels and build the modeling table.

``label_name`` is chosen from the dataset, because digits and fashion share
the codes 0-9 and one map cannot name both.

Usage::

    python src/mnist_f_vae/02_engineer.py \
        --raw data/mnist_f_vae/raw_images.parquet --images data/mnist_f_vae/raw_images.npy \
        --out data/mnist_f_vae/images.parquet --pixels data/mnist_f_vae/pixels.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensordict import TensorDict

from mnist_f_vae.config import (
    DATA_CONTRACT_PATH,
    LABEL_NAMES,
    MODELING_CONTRACT_PATH,
    PIXEL_MEAN,
    PIXEL_STD,
)
from input_contracts.data_contract import DataContract
from input_contracts.model_contract import ModelingContract


def run(raw_path: Path, images_path: Path, out_path: Path, pixels_path: Path) -> TensorDict:
    """Apply the data contract, then write normalized NCHW pixels and the modeling table."""
    raw = pd.read_parquet(raw_path)
    images = np.load(images_path)
    if len(images) != len(raw):
        raise ValueError(f"images rows {len(images)} do not match table rows {len(raw)}")
    data_contract = DataContract.load(DATA_CONTRACT_PATH)
    model_contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    cleaned = data_contract.validate(data_contract.apply(raw))
    pixels = _normalize(images)
    modeled = cleaned.copy()
    modeled["label_name"] = pd.Series(_label_names(cleaned), index=cleaned.index, dtype="string")
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


def _label_names(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=np.int64, copy=False)
    datasets = frame["dataset"].astype(str).to_numpy()
    safe = np.clip(labels, 0, 9)
    choices = [np.asarray(names, dtype=object)[safe] for names in LABEL_NAMES.values()]
    masks = [datasets == name for name in LABEL_NAMES]
    named = np.select(masks, choices, default="").astype(str)
    invalid = (labels < 0) | (labels > 9)
    return np.where(invalid, "", named)


def _normalize(images: np.ndarray) -> np.ndarray:
    if images.dtype != np.uint8:
        raise TypeError("engineer expects uint8 images in 0..255")
    scaled = images.astype(np.float32) / np.float32(255.0)
    normalized = (scaled - np.float32(PIXEL_MEAN)) / np.float32(PIXEL_STD)
    return np.ascontiguousarray(np.transpose(normalized, (0, 3, 1, 2)))


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
