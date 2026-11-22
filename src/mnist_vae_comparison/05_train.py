"""Stage 05: fit the chosen VAE on the train split.

The test split is scored for the loss curve. It does not stop training and it
was not used to choose the hyperparameters.

Usage::

    python src/mnist_vae_comparison/05_train.py \
        --data data/mnist_vae_comparison/images.parquet --pixels data/mnist_vae_comparison/pixels.npy \
        --split data/mnist_vae_comparison/split.json --params reports/mnist_vae_comparison/best_params.json \
        --model models/mnist_vae_comparison/mnist_vae.pt --plot reports/mnist_vae_comparison/plots/loss.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from mnist_vae_comparison.config import MODELING_CONTRACT_PATH, RANDOM_SEED
from mnist_vae_comparison.model import VAE
from mnist_contracts.model_contract import ModelingContract


def _save_loss(history: dict[str, list[float]], plot_path: Path) -> None:
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.plot(history["train"], label="train")
    ax.plot(history["eval"], label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("vae_loss")
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
    epochs: int = 20,
    plot_path: Path,
) -> Path:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    _batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    hparams = json.loads(params_path.read_text(encoding="utf-8"))["params"]
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    model = VAE.from_hparams(hparams)
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
    )
    model.save(model_path, list(contract.features), hparams)
    _save_loss(history, plot_path)
    print(f"trained {len(train_idx)} rows -> {model_path}; loss -> {plot_path}")
    return model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="05_train", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.params,
        args.model,
        epochs=args.epochs,
        plot_path=args.plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
