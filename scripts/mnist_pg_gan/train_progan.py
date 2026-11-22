"""Pull a CIFAR-10 slice, then grow the progressive GAN and archive the grids.

``--epochs`` is the number of epochs in each resolution phase. The archive
keeps the fixed-noise grids, the growth timeline, and the loss curve.

Usage::

    python scripts/mnist_pg_gan/train_progan.py --train-per-class 200 --test-per-class 40 --epochs 2
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from mnist_pg_gan.config import ROOT
from mnist_pg_gan.model import resolutions_for

sys.path.insert(0, str(ROOT / "scripts" / "mnist_pg_gan"))

SHARED_PARAMS: dict[str, float | int] = {
    "noise_dim": 64,
    "width": 64,
    "image_size": 32,
    "channels": 3,
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "beta1": 0.5,
    "batch_size": 32,
}


def run(
    *,
    train_per_class: int = 200,
    test_per_class: int = 40,
    epochs: int = 2,
    snapshot_every: int = 1,
) -> None:
    pull = importlib.import_module("pull_sample_data")
    engineer = importlib.import_module("mnist_pg_gan.02_engineer")
    split = importlib.import_module("mnist_pg_gan.03_split")
    train = importlib.import_module("mnist_pg_gan.05_train")
    evaluate = importlib.import_module("mnist_pg_gan.06_eval")
    raw = ROOT / "data" / "mnist_pg_gan" / "raw_images.parquet"
    images = ROOT / "data" / "mnist_pg_gan" / "raw_images.npy"
    table = ROOT / "data" / "mnist_pg_gan" / "images.parquet"
    pixels = ROOT / "data" / "mnist_pg_gan" / "pixels.npy"
    split_path = ROOT / "data" / "mnist_pg_gan" / "split.json"
    params_path = ROOT / "reports" / "mnist_pg_gan" / "best_params.json"
    model_path = ROOT / "models" / "mnist_pg_gan" / "mnist_pg_gan.pt"
    archive = ROOT / "reports" / "mnist_pg_gan" / "generations" / "cifar"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps({"params": SHARED_PARAMS}, indent=2), encoding="utf-8")
    pull.run(
        pull.SQL,
        raw,
        images,
        dataset="cifar",
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
    shutil.copy2(model_path, archive / "mnist_pg_gan.pt")
    _mirror_reports(archive)
    (archive / "source.json").write_text(
        json.dumps(
            {
                "dataset": "cifar",
                "train_per_class": train_per_class,
                "test_per_class": test_per_class,
                "epochs_per_phase": epochs,
                "resolutions": list(resolutions_for(int(SHARED_PARAMS["image_size"]))),
                "params": SHARED_PARAMS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"archived cifar -> {archive}")


def _mirror_reports(archive: Path) -> None:
    plots = ROOT / "reports" / "mnist_pg_gan" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for name in ("loss.png", "final.png", "real_test.png", "growth_timeline.png"):
        source = archive / name
        if not source.is_file():
            continue
        target_name = "generated.png" if name == "final.png" else name
        shutil.copy2(source, plots / target_name)
    for name in ("evaluation_train.json", "evaluation_test.json"):
        shutil.copy2(archive / name, ROOT / "reports" / "mnist_pg_gan" / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train_progan", description=__doc__)
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--snapshot-every", type=int, default=1)
    args = parser.parse_args(argv)
    run(
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        epochs=args.epochs,
        snapshot_every=args.snapshot_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
