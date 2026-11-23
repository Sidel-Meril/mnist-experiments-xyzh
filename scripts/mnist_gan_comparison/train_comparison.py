"""Pull each dataset, then train and evaluate with the same stage functions.

The only change between runs is the file written by ``pull_sample_data``.
Engineer, split, train, and eval are called with the same arguments aside
from the archive directory that keeps that run's plots and samples.

Usage::

    python scripts/mnist_gan_comparison/train_comparison.py --train-per-class 400 --epochs 40
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from mnist_gan_comparison.config import DATASETS, ROOT

sys.path.insert(0, str(ROOT / "scripts" / "mnist_gan_comparison"))

SHARED_PARAMS: dict[str, float | int] = {
    "noise_dim": 64,
    "hidden_dim": 256,
    "in_features": 784,
    "lr_g": 2e-4,
    # Quarter of the generator step. Equal rates let the discriminator win
    # and the fixed-noise grid collapse to one image.
    "lr_d": 5e-5,
    "beta1": 0.5,
    "batch_size": 128,
}


def run(
    *,
    datasets: tuple[str, ...] = DATASETS,
    train_per_class: int = 400,
    test_per_class: int = 80,
    epochs: int = 40,
    snapshot_every: int = 1,
) -> None:
    pull = importlib.import_module("pull_sample_data")
    engineer = importlib.import_module("mnist_gan_comparison.02_engineer")
    split = importlib.import_module("mnist_gan_comparison.03_split")
    train = importlib.import_module("mnist_gan_comparison.05_train")
    evaluate = importlib.import_module("mnist_gan_comparison.06_eval")
    raw = ROOT / "data" / "mnist_gan_comparison" / "raw_images.parquet"
    images = ROOT / "data" / "mnist_gan_comparison" / "raw_images.npy"
    table = ROOT / "data" / "mnist_gan_comparison" / "images.parquet"
    pixels = ROOT / "data" / "mnist_gan_comparison" / "pixels.npy"
    split_path = ROOT / "data" / "mnist_gan_comparison" / "split.json"
    params_path = ROOT / "reports" / "mnist_gan_comparison" / "best_params.json"
    model_path = ROOT / "models" / "mnist_gan_comparison" / "mnist_gan.pt"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps({"params": SHARED_PARAMS}, indent=2), encoding="utf-8")
    for dataset in datasets:
        archive = ROOT / "reports" / "mnist_gan_comparison" / "generations" / dataset
        archive.mkdir(parents=True, exist_ok=True)
        pull.run(
            pull.SQL,
            raw,
            images,
            dataset=dataset,
            train_per_class=train_per_class,
            test_per_class=test_per_class,
        )
        engineer.run(raw, images, table, pixels)
        split.run(table, split_path)
        train.run(
            table,
            pixels,
            split_path,
            params_path,
            model_path,
            epochs=epochs,
            plot_path=archive / "loss.png",
            generations_dir=archive,
            snapshot_every=snapshot_every,
        )
        evaluate.run(
            table,
            pixels,
            split_path,
            model_path,
            archive / "evaluation_train.json",
            archive / "evaluation_test.json",
            generated_path=archive / "final.png",
            real_path=archive / "real_test.png",
        )
        shutil.copy2(model_path, archive / "mnist_gan.pt")
        _mirror_reports(archive)
        (archive / "source.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "train_per_class": train_per_class,
                    "test_per_class": test_per_class,
                    "epochs": epochs,
                    "params": SHARED_PARAMS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"archived {dataset} -> {archive}")


def _mirror_reports(archive: Path) -> None:
    plots = ROOT / "reports" / "mnist_gan_comparison" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for name in ("loss.png", "final.png", "real_test.png"):
        source = archive / name
        target = plots / ("generated.png" if name == "final.png" else name)
        if name == "loss.png":
            target = plots / "loss.png"
        shutil.copy2(source, target)
    for name in ("evaluation_train.json", "evaluation_test.json"):
        shutil.copy2(archive / name, ROOT / "reports" / "mnist_gan_comparison" / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train_comparison", description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--test-per-class", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--snapshot-every", type=int, default=1)
    args = parser.parse_args(argv)
    chosen = tuple(args.dataset) if args.dataset else DATASETS
    run(
        datasets=chosen,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        epochs=args.epochs,
        snapshot_every=args.snapshot_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
