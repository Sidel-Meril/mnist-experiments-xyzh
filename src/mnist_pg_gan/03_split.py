"""Stage 03: holdout indices.

When ``cifar_split`` contains both official sides, those sides are the split.
A table that is only one side, such as a test fixture, uses a stratified holdout.

Usage::

    python src/mnist_pg_gan/03_split.py --data data/mnist_pg_gan/images.parquet --out data/mnist_pg_gan/split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensordict import TensorDict

from mnist_pg_gan.config import MODELING_CONTRACT_PATH, RANDOM_SEED
from input_contracts.model_contract import ModelingContract


def run(
    data_path: Path,
    out_path: Path,
    *,
    test_size: float = 0.2,
) -> dict[str, list[int]]:
    frame = pd.read_parquet(data_path)
    official = _official_split(frame)
    if official is None:
        target_name = ModelingContract.load(MODELING_CONTRACT_PATH).target
        if isinstance(frame, TensorDict):
            target = frame[target_name].detach().cpu().numpy().astype(np.int64, copy=False)
            index = np.arange(int(frame.batch_size[0]))
            counts = np.unique(target, return_counts=True)[1]
        else:
            target = frame[target_name].astype("int64")
            index = frame.index.to_numpy()
            counts = target.value_counts()
        if counts.size == 0 or int(counts.min()) < 2:
            raise ValueError("stratified split needs at least two rows of each class")
        train_idx, test_idx = train_test_split(
            index,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=target,
        )
        official = (train_idx, test_idx)
    train_idx, test_idx = official
    indices = {
        "train_idx": [int(i) for i in train_idx],
        "test_idx": [int(i) for i in test_idx],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(indices), encoding="utf-8")
    print(f"split train={len(indices['train_idx'])} test={len(indices['test_idx'])} -> {out_path}")
    return indices


def _official_split(
    frame: pd.DataFrame | TensorDict,
) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(frame, TensorDict):
        return _official_batch_split(frame)
    if "cifar_split" not in frame.columns:
        return None
    sides = frame["cifar_split"].astype("string")
    train_idx = frame.index[sides.eq("train")].to_numpy()
    test_idx = frame.index[sides.eq("test")].to_numpy()
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    if len(train_idx) + len(test_idx) != len(frame):
        return None
    return train_idx, test_idx


def _official_batch_split(batch: TensorDict) -> tuple[np.ndarray, np.ndarray] | None:
    if "cifar_split" not in batch:
        return None
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    column = next(item for item in contract.columns if item.name == "cifar_split")
    if not column.categories:
        return None
    levels = np.array([str(item) for item in column.categories], dtype=object)
    codes = batch["cifar_split"].detach().cpu().numpy().astype(np.int64, copy=False)
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= len(levels)):
        return None
    sides = levels[codes]
    train_idx = np.flatnonzero(sides == "train")
    test_idx = np.flatnonzero(sides == "test")
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    if len(train_idx) + len(test_idx) != int(batch.batch_size[0]):
        return None
    return train_idx, test_idx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="03_split", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args(argv)
    run(args.data, args.out, test_size=args.test_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
