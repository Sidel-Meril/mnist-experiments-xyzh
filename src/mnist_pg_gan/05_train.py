"""Stage 05: grow the progressive GAN on the train split.

``--epochs`` is the number of epochs in each phase, not the total. Training
starts at 4x4, fades in 8, 16, and 32, and saves a fixed-noise grid at the
current resolution. ``growth_timeline.png`` stacks the last grid of each phase.

Usage::

    python src/mnist_pg_gan/05_train.py \
        --data data/mnist_pg_gan/images.parquet --pixels data/mnist_pg_gan/pixels.npy \
        --split data/mnist_pg_gan/split.json --params reports/mnist_pg_gan/best_params.json \
        --model models/mnist_pg_gan/mnist_pg_gan.pt --plot reports/mnist_pg_gan/plots/loss.png \
        --generations reports/mnist_pg_gan/generations/cifar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from mnist_pg_gan.config import MODELING_CONTRACT_PATH, RANDOM_SEED
from mnist_pg_gan.model import ProgressiveGAN
from mnist_contracts.model_contract import ModelingContract


def _save_loss(history: dict[str, list[float]], plot_path: Path) -> None:
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.plot(history["train_discriminator"], label="train discriminator")
    ax.plot(history["train_generator"], label="train generator")
    ax.plot(history["eval_discriminator"], label="eval discriminator")
    ax.plot(history["eval_generator"], label="eval generator")
    ax.set_xlabel("epoch")
    ax.set_ylabel("bce")
    handles, labels = ax.get_legend_handles_labels()
    if history.get("resolution"):
        twin = ax.twinx()
        twin.step(
            range(len(history["resolution"])),
            history["resolution"],
            where="post",
            color="0.35",
            label="resolution",
        )
        twin.set_ylabel("resolution")
        extra_handles, extra_labels = twin.get_legend_handles_labels()
        handles = handles + extra_handles
        labels = labels + extra_labels
    ax.legend(handles, labels, loc="upper right")
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
    _batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    hparams = json.loads(params_path.read_text(encoding="utf-8"))["params"]
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    model = ProgressiveGAN.from_hparams(hparams)
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
        generations_dir=generations_dir,
        snapshot_every=snapshot_every,
    )
    model.save(model_path, list(contract.features), hparams)
    _save_loss(history, plot_path)
    if generations_dir is not None:
        generations_dir.mkdir(parents=True, exist_ok=True)
        history_path = generations_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"trained {len(train_idx)} rows -> {model_path}; loss -> {plot_path}")
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
