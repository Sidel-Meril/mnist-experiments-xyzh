"""Stage 06: score train and test likelihood, and write sample grids.

``nll`` is the mean per-image negative log-likelihood in nats under bin-center
dequantization. ``bpd`` is that value in bits per dimension. Generated images
come from sampling the prior and inverting the flow.

Usage::

    python src/mnist_realnvp/06_eval.py \
        --data data/mnist_realnvp/images.parquet --pixels data/mnist_realnvp/pixels.npy \
        --split data/mnist_realnvp/split.json --model models/mnist_realnvp/mnist_realnvp.pt \
        --train-metrics reports/mnist_realnvp/evaluation_train.json \
        --test-metrics reports/mnist_realnvp/evaluation_test.json \
        --generated reports/mnist_realnvp/plots/generated.png \
        --reconstructions reports/mnist_realnvp/plots/reconstructions.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mnist_realnvp.config import MODELING_CONTRACT_PATH, RANDOM_SEED
from mnist_realnvp.figures import save_grid, save_reconstructions
from mnist_realnvp.model import RealNVP, bpd_from_nll, rows_for_dataset, seed_all
from mnist_contracts.model_contract import ModelingContract


def _write(path: Path, metrics: dict[str, float | int | str]) -> None:
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
    generated_path: Path,
    reconstructions_path: Path,
) -> dict[str, dict[str, float | int | str]]:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    bundle = RealNVP.load(model_path)
    if list(bundle["columns"]) != list(contract.features):
        raise ValueError("checkpoint columns do not match the model contract features")
    hparams = bundle["hparams"]
    if not isinstance(hparams, dict):
        raise TypeError("checkpoint hparams must be a dict")
    model = bundle["model"]
    if not isinstance(model, RealNVP):
        raise TypeError("checkpoint model is not a RealNVP")
    dataset_column = next(column for column in contract.columns if column.name == "dataset")
    categories = tuple(str(item) for item in dataset_column.categories or ())
    codes = batch["dataset"].detach().cpu().numpy()
    source = str(hparams.get("dataset", "digits"))
    train_idx = rows_for_dataset(codes, np.asarray(split["train_idx"]), source, categories)
    test_idx = rows_for_dataset(codes, np.asarray(split["test_idx"]), source, categories)
    batch_size = int(hparams["batch_size"])
    n_train = len(train_idx)
    n_test = len(test_idx)
    device = torch.device("cpu")
    model.to(device)
    written: dict[str, dict[str, float | int | str]] = {}
    scored = (
        ("train", train_idx, train_metrics_path),
        ("test", test_idx, test_metrics_path),
    )
    for name, idx, metrics_path in scored:
        features = np.ascontiguousarray(contract.feature_matrix(pixels, idx), dtype=np.float32)
        labels = torch.zeros(len(features), dtype=torch.int64)
        eval_loader = DataLoader(
            TensorDataset(torch.from_numpy(features), labels),
            batch_size=batch_size,
            shuffle=False,
        )
        nll = float(model.mean_nll(eval_loader, device))
        metrics: dict[str, float | int | str] = {
            "nll": nll,
            "bpd": bpd_from_nll(nll),
            "n_train": n_train,
            "n_test": n_test,
            "dataset": source,
        }
        _write(metrics_path, metrics)
        print(f"{name} nll={nll:.2f} bpd={metrics['bpd']:.3f} -> {metrics_path}")
        written[name] = metrics
    seed_all(RANDOM_SEED)
    model.eval()
    drawn = model.sample(64)
    save_grid(drawn, generated_path, f"RealNVP samples ({source}, inverse flow)")
    print(f"generated -> {generated_path}")
    take = min(8, n_test)
    originals = torch.from_numpy(
        np.ascontiguousarray(contract.feature_matrix(pixels, test_idx[:take]), dtype=np.float32)
    )
    restored = model.reconstruct(originals)
    save_reconstructions(
        originals,
        restored,
        reconstructions_path,
        f"RealNVP reconstructions ({source})",
    )
    print(f"reconstructions -> {reconstructions_path}")
    model.cpu()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="06_eval", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-metrics", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reconstructions", type=Path, required=True)
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.model,
        args.train_metrics,
        args.test_metrics,
        generated_path=args.generated,
        reconstructions_path=args.reconstructions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
