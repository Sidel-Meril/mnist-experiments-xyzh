"""Stage 05: fit the RealNVP on the train split.

The test split is scored for the NLL and bits-per-dimension curves. It does
not stop training and it was not used to choose the hyperparameters. After the
fit, samples are drawn by inverting the flow.

Usage::

    python src/mnist_realnvp/05_train.py \
        --data data/mnist_realnvp/images.parquet --pixels data/mnist_realnvp/pixels.npy \
        --split data/mnist_realnvp/split.json --params reports/mnist_realnvp/best_params.json \
        --model models/mnist_realnvp/mnist_realnvp.pt --plot reports/mnist_realnvp/plots/loss.png \
        --bpd-plot reports/mnist_realnvp/plots/bpd.png --samples reports/mnist_realnvp/plots/samples.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mnist_realnvp.config import MODELING_CONTRACT_PATH, RANDOM_SEED
from mnist_realnvp.figures import save_curve, save_grid
from mnist_realnvp.model import RealNVP, rows_for_dataset, seed_all
from mnist_contracts.model_contract import ModelingContract


def run(
    data_path: Path,
    pixels_path: Path,
    split_path: Path,
    params_path: Path,
    model_path: Path,
    *,
    epochs: int = 20,
    plot_path: Path,
    bpd_path: Path,
    samples_path: Path,
    dataset: str | None = None,
) -> Path:
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    hparams = dict(json.loads(params_path.read_text(encoding="utf-8"))["params"])
    if dataset is not None:
        hparams["dataset"] = dataset
    labels = batch[contract.target].detach().cpu().numpy()
    dataset_column = next(column for column in contract.columns if column.name == "dataset")
    categories = tuple(str(item) for item in dataset_column.categories or ())
    codes = batch["dataset"].detach().cpu().numpy()
    source = str(hparams["dataset"])
    train_idx = rows_for_dataset(codes, np.asarray(split["train_idx"]), source, categories)
    test_idx = rows_for_dataset(codes, np.asarray(split["test_idx"]), source, categories)
    model = RealNVP.from_hparams(hparams)
    train_loader, test_loader = model.data_loaders(
        contract.feature_matrix(pixels, train_idx),
        labels[train_idx],
        contract.feature_matrix(pixels, test_idx),
        labels[test_idx],
        batch_size=int(hparams["batch_size"]),
        seed=RANDOM_SEED,
    )
    def _snapshot(epoch: int, _history: dict[str, list[float]]) -> None:
        if (epoch + 1) % 2 != 0 and epoch + 1 != epochs:
            return
        model.save(model_path, list(contract.features), hparams)
        seed_all(RANDOM_SEED + epoch)
        save_grid(model.sample(64), samples_path, f"RealNVP samples ({source}, inverse flow)")
        print(f"checkpoint epoch {epoch + 1} -> {model_path}", flush=True)

    history = model.fit_from_dataset(
        train_loader,
        test_loader,
        hparams,
        epochs=epochs,
        seed=RANDOM_SEED,
        record_scores=True,
        on_epoch=_snapshot,
    )
    model.save(model_path, list(contract.features), hparams)
    save_curve(history["train"], history["eval"], plot_path, "negative_log_likelihood")
    save_curve(history["bpd_train"], history["bpd_eval"], bpd_path, "bits_per_dimension")
    seed_all(RANDOM_SEED)
    drawn = model.sample(64)
    save_grid(drawn, samples_path, f"RealNVP samples ({source}, inverse flow)")
    print(
        f"trained {len(train_idx)} {source} rows -> {model_path}; "
        f"loss -> {plot_path}; bpd -> {bpd_path}; samples -> {samples_path}"
    )
    return model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="05_train", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--bpd-plot", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", default=None, choices=["digits", "fashion", "both"])
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.params,
        args.model,
        epochs=args.epochs,
        plot_path=args.plot,
        bpd_path=args.bpd_plot,
        samples_path=args.samples,
        dataset=args.dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
