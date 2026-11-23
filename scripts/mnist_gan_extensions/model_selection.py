"""Pull each dataset, then train every GAN variant with the same stage functions.

The dataset swap is ``load_data`` (via ``pull_sample_data``). Engineer, split,
train, and eval are the same calls for MNIST and Fashion-MNIST, and for the
cGAN, PatchGAN, and feature-matching runs. Only ``params["variant"]`` and the
archive directory change between variants. ``best_params.json`` is rewritten
only for the digits cGAN, the archive copied onto the DVC eval paths.

Usage::

    python scripts/mnist_gan_extensions/model_selection.py --train-per-class 250 --epochs 40
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from mnist_gan_extensions.config import DATASETS, EXPERIMENT_DIRS, FM_WEIGHT, N_CLASSES, ROOT

sys.path.insert(0, str(ROOT / "scripts" / "mnist_gan_extensions"))

SHARED_PARAMS: dict[str, float | int | str] = {
    "noise_dim": 64,
    "hidden_dim": 256,
    "in_features": 784,
    "n_classes": N_CLASSES,
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "beta1": 0.5,
    "batch_size": 128,
    "fm_weight": FM_WEIGHT,
}


def run(
    *,
    datasets: tuple[str, ...] = DATASETS,
    variants: tuple[str, ...] = tuple(EXPERIMENT_DIRS),
    train_per_class: int = 250,
    test_per_class: int = 50,
    epochs: int = 40,
    snapshot_every: int = 10,
) -> None:
    pull = importlib.import_module("pull_sample_data")
    engineer = importlib.import_module("mnist_gan_extensions.02_engineer")
    split = importlib.import_module("mnist_gan_extensions.03_split")
    train = importlib.import_module("mnist_gan_extensions.05_train")
    evaluate = importlib.import_module("mnist_gan_extensions.06_eval")
    raw = ROOT / "data" / "mnist_gan_extensions" / "raw_images.parquet"
    images = ROOT / "data" / "mnist_gan_extensions" / "raw_images.npy"
    table = ROOT / "data" / "mnist_gan_extensions" / "images.parquet"
    pixels = ROOT / "data" / "mnist_gan_extensions" / "pixels.npy"
    split_path = ROOT / "data" / "mnist_gan_extensions" / "split.json"
    params_path = ROOT / "reports" / "mnist_gan_extensions" / "best_params.json"
    default_model = ROOT / "models" / "mnist_gan_extensions" / "mnist_cgan.pt"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
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
        for variant in variants:
            if variant not in EXPERIMENT_DIRS:
                raise ValueError(f"variant must be one of {tuple(EXPERIMENT_DIRS)}")
            folder = EXPERIMENT_DIRS[variant]
            archive = ROOT / "reports" / "mnist_gan_extensions" / "experiments" / folder / dataset
            archive.mkdir(parents=True, exist_ok=True)
            for stale in archive.glob("epoch_*.png"):
                stale.unlink()
            checkpoint = archive / "model.pt"
            params = dict(SHARED_PARAMS)
            params["variant"] = variant
            run_params = archive / "params.json"
            run_params.write_text(json.dumps({"params": params}, indent=2), encoding="utf-8")
            train.run(
                table,
                pixels,
                split_path,
                run_params,
                checkpoint,
                epochs=epochs,
                plot_path=archive / "loss.png",
                generations_dir=archive,
                snapshot_every=snapshot_every,
            )
            evaluate.run(
                table,
                pixels,
                split_path,
                checkpoint,
                archive / "evaluation_train.json",
                archive / "evaluation_test.json",
                generated_path=archive / "final.png",
                real_path=archive / "real_test.png",
                classes_path=archive / "classes.png",
            )
            (archive / "source.json").write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "experiment": folder,
                        "train_per_class": train_per_class,
                        "test_per_class": test_per_class,
                        "epochs": epochs,
                        "params": params,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if variant == "cgan" and dataset == "digits":
                default_model.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoint, default_model)
                params_path.write_text(
                    json.dumps({"params": params}, indent=2),
                    encoding="utf-8",
                )
                _mirror_reports(archive)
            print(f"archived {folder}/{dataset} -> {archive}")


def _mirror_reports(archive: Path) -> None:
    """Keep the DVC cGAN paths pointed at the digits run."""
    plots = ROOT / "reports" / "mnist_gan_extensions" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive / "loss.png", plots / "loss.png")
    shutil.copy2(archive / "final.png", plots / "generated.png")
    shutil.copy2(archive / "real_test.png", plots / "real_test.png")
    shutil.copy2(archive / "classes.png", plots / "classes.png")
    for name in ("evaluation_train.json", "evaluation_test.json"):
        shutil.copy2(archive / name, ROOT / "reports" / "mnist_gan_extensions" / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model_selection", description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--variant", action="append", choices=tuple(EXPERIMENT_DIRS))
    parser.add_argument("--train-per-class", type=int, default=250)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--snapshot-every", type=int, default=10)
    args = parser.parse_args(argv)
    chosen_data = tuple(args.dataset) if args.dataset else DATASETS
    chosen_variants = tuple(args.variant) if args.variant else tuple(EXPERIMENT_DIRS)
    run(
        datasets=chosen_data,
        variants=chosen_variants,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        epochs=args.epochs,
        snapshot_every=args.snapshot_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
