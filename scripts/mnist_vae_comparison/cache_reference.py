"""Copy one finished run to reference/ so the next dataset can replace data/.

The active ``data/``, ``models/``, and ``reports/`` directories are what the
stages just wrote. This script copies those artifacts under ``reference/<name>``
before another dataset overwrites them. It does not train.

Usage::

    python scripts/mnist_vae_comparison/cache_reference.py mnist_digits
    python scripts/mnist_vae_comparison/cache_reference.py fashion_mnist
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mnist_vae_comparison.config import ROOT

_FILES = (
    "data/raw_images.parquet",
    "data/raw_images.npy",
    "data/images.parquet",
    "data/pixels.npy",
    "data/split.json",
    "models/mnist_vae.pt",
    "reports/best_params.json",
    "reports/mnist_vae.db",
    "reports/evaluation_train.json",
    "reports/evaluation_test.json",
    "reports/plots/optimization_history.html",
    "reports/plots/param_importances.html",
    "reports/plots/parallel_coordinate.html",
    "reports/plots/parameter_slice.html",
    "reports/plots/loss_vs_latent.png",
    "reports/plots/loss.png",
    "reports/plots/reconstructions_train.png",
    "reports/plots/reconstructions_test.png",
    "reports/plots/generated_samples.png",
)


def _source(rel: str) -> Path:
    top, _, rest = rel.partition("/")
    return ROOT / top / "mnist_vae_comparison" / rest


def run(name: str) -> Path:
    dest = ROOT / "reference" / name
    missing = [rel for rel in _FILES if not _source(rel).is_file()]
    if missing:
        raise FileNotFoundError("cache is missing: " + ", ".join(missing))
    if dest.exists():
        shutil.rmtree(dest)
    for rel in _FILES:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_source(rel), target)
    print(f"cached {name} -> {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cache_reference", description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args(argv)
    run(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
