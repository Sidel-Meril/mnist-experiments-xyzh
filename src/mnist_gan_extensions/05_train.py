"""Stage 05: fit the class-conditional GAN on the train split.

The test split is scored so the loss curves show discriminator and generator
stability. It does not stop training and it was not used to choose
hyperparameters. The dataset name is not read here. Whichever images stage 01
wrote are the training set. ``params["variant"]`` selects cGAN, PatchGAN, or
feature matching. Fixed-noise batches written under ``--generations`` cycle
through class labels so a collapsed class shows up as a repeated digit.

Usage::

    python src/mnist_gan_extensions/05_train.py \
        --data data/mnist_gan_extensions/images.parquet --pixels data/mnist_gan_extensions/pixels.npy \
        --split data/mnist_gan_extensions/split.json --params reports/mnist_gan_extensions/best_params.json \
        --model models/mnist_gan_extensions/mnist_cgan.pt --plot reports/mnist_gan_extensions/plots/loss.png \
        --generations reports/mnist_gan_extensions/experiments/cgan/digits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_gan_extensions.config import MODELING_CONTRACT_PATH, RANDOM_SEED
from mnist_gan_extensions.model import ConditionalGAN, complete_hparams
from mnist_contracts.model_contract import ModelingContract


def _save_loss(history: dict[str, list[float]], plot_path: Path) -> None:
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.plot(history["train_discriminator"], label="train discriminator")
    ax.plot(history["train_generator"], label="train generator")
    ax.plot(history["eval_discriminator"], label="eval discriminator")
    ax.plot(history["eval_generator"], label="eval generator")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
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
    generations_dir: Path | None = None,
    snapshot_every: int = 1,
) -> dict[str, list[float]]:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    hparams = complete_hparams(json.loads(params_path.read_text(encoding="utf-8"))["params"])
    labels = batch[contract.target].detach().cpu().numpy().astype(np.int64)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    model = ConditionalGAN.from_hparams(hparams)
    train_loader, test_loader = model.data_loaders(
        contract.feature_matrix(pixels, train_idx),
        labels[train_idx],
        contract.feature_matrix(pixels, test_idx),
        labels[test_idx],
        batch_size=int(hparams["batch_size"]),
        seed=RANDOM_SEED,
    )
    history = model.fit_from_dataset(
        train_loader,
        test_loader,
        hparams,
        epochs=epochs,
        seed=RANDOM_SEED,
        generations_dir=generations_dir,
        snapshot_every=snapshot_every,
    )
    model.save(model_path, list(contract.features), hparams)
    _save_loss(history, plot_path)
    if generations_dir is not None:
        generations_dir.mkdir(parents=True, exist_ok=True)
        history_path = generations_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"trained {len(train_idx)} rows ({hparams['variant']}) -> {model_path}")
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="05_train", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--snapshot-every", type=int, default=1)
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.params,
        args.model,
        epochs=args.epochs,
        plot_path=args.plot,
        generations_dir=args.generations,
        snapshot_every=args.snapshot_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
