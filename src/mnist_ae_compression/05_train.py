"""Stage 05: fit the chosen autoencoder on the train split.

The test split is scored for the reconstruction-MSE curve. It does not stop
training and it was not used to choose the hyperparameters.

Usage::

    python src/mnist_ae_compression/05_train.py \
        --data data/mnist_ae_compression/images.parquet --pixels data/mnist_ae_compression/pixels.npy \
        --split data/mnist_ae_compression/split.json --params reports/mnist_ae_compression/best_params.json \
        --model models/mnist_ae_compression/mnist_ae.pt --plot reports/mnist_ae_compression/plots/loss.png \
        --reconstructions reports/mnist_ae_compression/reconstructions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from mnist_ae_compression.config import MODELING_CONTRACT_PATH, RANDOM_SEED
from mnist_ae_compression.model import Autoencoder
from mnist_contracts.model_contract import ModelingContract


def _save_loss(history: dict[str, list[float]], plot_path: Path) -> None:
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.plot(history["train"], label="train")
    ax.plot(history["eval"], label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction_mse")
    ax.legend()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(
    data_path: Path,
    pixels_path: Path,
    split_path: Path,
    params_path: Path,
    model_path: Path,
    *,
    epochs: int = 8,
    plot_path: Path,
    reconstructions_dir: Path | None = None,
) -> Path:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    _batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    hparams = json.loads(params_path.read_text(encoding="utf-8"))["params"]
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    model = Autoencoder.from_hparams(hparams)
    train_loader, test_loader = model.data_loaders(
        contract.feature_matrix(pixels, train_idx),
        contract.feature_matrix(pixels, test_idx),
        batch_size=int(hparams["batch_size"]),
        seed=RANDOM_SEED,
    )
    history = model.fit_from_dataset(
        train_loader,
        test_loader,
        hparams,
        epochs=epochs,
        seed=RANDOM_SEED,
        reconstructions_dir=reconstructions_dir,
    )
    model.save(model_path, list(contract.features), hparams)
    _save_loss(history, plot_path)
    print(f"trained {len(train_idx)} rows -> {model_path}; loss -> {plot_path}")
    if reconstructions_dir is not None:
        print(f"reconstructions -> {reconstructions_dir}")
    return model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="05_train", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--reconstructions", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.params,
        args.model,
        epochs=args.epochs,
        plot_path=args.plot,
        reconstructions_dir=args.reconstructions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
