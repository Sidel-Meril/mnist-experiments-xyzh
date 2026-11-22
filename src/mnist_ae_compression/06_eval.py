"""Stage 06: score reconstructions on train and test.

``mse`` is the compression error on normalized pixels. ``compression_ratio``
is the input length divided by the latent size. The grids show the original
digit above its reconstruction.

Usage::

    python src/mnist_ae_compression/06_eval.py \
        --data data/mnist_ae_compression/images.parquet --pixels data/mnist_ae_compression/pixels.npy \
        --split data/mnist_ae_compression/split.json --model models/mnist_ae_compression/mnist_ae.pt \
        --train-metrics reports/mnist_ae_compression/evaluation_train.json \
        --test-metrics reports/mnist_ae_compression/evaluation_test.json \
        --reconstructions-train reports/mnist_ae_compression/plots/reconstructions_train.png \
        --reconstructions-test reports/mnist_ae_compression/plots/reconstructions_test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mnist_ae_compression.config import MODELING_CONTRACT_PATH
from mnist_ae_compression.model import Autoencoder, _save_reconstructions
from input_contracts.model_contract import ModelingContract


def _errors(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
    diff = reconstructed.astype(np.float64) - original.astype(np.float64)
    return float(np.mean(diff * diff)), float(np.mean(np.abs(diff)))


def _write(path: Path, metrics: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


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
) -> dict[str, dict[str, float | int]]:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    _batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    bundle = Autoencoder.load(model_path)
    if list(bundle["columns"]) != list(contract.features):
        raise ValueError("checkpoint columns do not match the model contract features")
    hparams = bundle["hparams"]
    if not isinstance(hparams, dict):
        raise TypeError("checkpoint hparams must be a dict")
    batch_size = int(hparams["batch_size"])
    latent_dim = int(hparams["latent_dim"])
    ratio = float(hparams["in_features"]) / float(latent_dim)
    n_train = len(split["train_idx"])
    n_test = len(split["test_idx"])
    model = bundle["model"]
    if not isinstance(model, Autoencoder):
        raise TypeError("checkpoint model must be an Autoencoder")
    written: dict[str, dict[str, float | int]] = {}
    scored = (
        ("train", split["train_idx"], train_metrics_path, reconstructions_train_path),
        ("test", split["test_idx"], test_metrics_path, reconstructions_test_path),
    )
    for name, idx, metrics_path, grid_path in scored:
        rows = contract.feature_matrix(pixels, idx)
        restored = model.reconstruct(rows, batch_size)
        mse, mae = _errors(rows, restored)
        metrics: dict[str, float | int] = {
            "mse": mse,
            "mae": mae,
            "compression_ratio": ratio,
            "latent_dim": latent_dim,
            "n_train": n_train,
            "n_test": n_test,
        }
        _write(metrics_path, metrics)
        _save_reconstructions(rows, restored, grid_path)
        print(
            f"{name} mse={metrics['mse']:.4f} mae={metrics['mae']:.4f} "
            f"compression_ratio={metrics['compression_ratio']:.3f} -> {metrics_path}"
        )
        print(f"{name} reconstructions -> {grid_path}")
        written[name] = metrics
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
