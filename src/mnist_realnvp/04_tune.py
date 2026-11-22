"""Stage 04: Optuna search for the MNIST RealNVP.

The study minimizes mean validation negative log-likelihood from stratified
cross-validation on the train split of one dataset. The test split is not read.
Trials vary width, the tanh scale, learning rate, weight decay, and batch size.
The coupling depth stays at 4 so each trial is a full multi-scale stack.
Trials are stored in a local SQLite database and resumed when that file already
holds the study. When the stored study already has the requested number of
completed trials, no new trials run. Tuning plots are written once the study
is finished.

Usage::

    python src/mnist_realnvp/04_tune.py \
        --data data/mnist_realnvp/images.parquet --pixels data/mnist_realnvp/pixels.npy \
        --split data/mnist_realnvp/split.json --out reports/mnist_realnvp/best_params.json \
        --storage reports/mnist_realnvp/mnist_realnvp.db \
        --history reports/mnist_realnvp/plots/optimization_history.html \
        --importances reports/mnist_realnvp/plots/param_importances.html \
        --parallel reports/mnist_realnvp/plots/parallel_coordinate.html \
        --slice reports/mnist_realnvp/plots/parameter_slice.html
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from sklearn.model_selection import StratifiedKFold

from mnist_realnvp.config import MODELING_CONTRACT_PATH, RANDOM_SEED
from mnist_realnvp.model import RealNVP, bpd_from_nll, rows_for_dataset
from input_contracts.model_contract import ModelingContract


def _sqlite_url(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.resolve().as_posix()


def _save_tuning_plots(
    study: optuna.Study,
    history_path: Path,
    importances_path: Path,
    parallel_path: Path,
    slice_path: Path,
) -> None:
    plots = (
        (plot_optimization_history, history_path),
        (plot_param_importances, importances_path),
        (plot_parallel_coordinate, parallel_path),
        (plot_slice, slice_path),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for plot, path in plots:
            path.parent.mkdir(parents=True, exist_ok=True)
            plot(study).write_html(path)


def run(
    data_path: Path,
    pixels_path: Path,
    split_path: Path,
    out_path: Path,
    *,
    storage_path: Path,
    history_path: Path,
    importances_path: Path,
    parallel_path: Path,
    slice_path: Path,
    n_trials: int = 2,
    epochs: int = 1,
    n_splits: int = 2,
    dataset: str = "digits",
) -> dict[str, object]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    labels = batch[contract.target].detach().cpu().numpy()
    dataset_column = next(column for column in contract.columns if column.name == "dataset")
    if not dataset_column.categories:
        raise ValueError("dataset column is missing categories")
    categories = tuple(str(item) for item in dataset_column.categories)
    codes = batch["dataset"].detach().cpu().numpy()
    train_idx = rows_for_dataset(
        codes,
        np.asarray(split["train_idx"], dtype=np.int64),
        dataset,
        categories,
    )
    counts = np.unique(labels[train_idx], return_counts=True)[1]
    if counts.size == 0 or int(counts.min()) < n_splits:
        raise ValueError(f"cross-validation needs at least {n_splits} train rows of each class")
    folds = _folds(contract, pixels, labels, train_idx, n_splits)

    def objective(trial: optuna.Trial) -> float:
        chosen = _hparams(trial)
        chosen["dataset"] = dataset
        scores: list[float] = []
        for fold_number, (fit_x, fit_y, val_x, val_y) in enumerate(folds):
            model = RealNVP.from_hparams(chosen)
            seed = RANDOM_SEED + trial.number * n_splits + fold_number
            train_loader, val_loader = model.data_loaders(
                fit_x,
                fit_y,
                val_x,
                val_y,
                batch_size=int(chosen["batch_size"]),
                seed=seed,
            )
            model.fit_from_dataset(
                train_loader,
                val_loader,
                chosen,
                epochs=epochs,
                seed=seed,
            )
            scores.append(float(model.mean_nll(val_loader, torch.device("cpu"))))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name="mnist_realnvp",
        storage=_sqlite_url(storage_path),
        load_if_exists=True,
    )
    queued = (
        {
            "hidden_channels": 64,
            "n_layers": 4,
            "scale": 2.0,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 128,
        },
        {
            "hidden_channels": 32,
            "n_layers": 4,
            "scale": 1.0,
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "batch_size": 64,
        },
    )
    if len(study.trials) == 0:
        for candidate in queued[:n_trials]:
            study.enqueue_trial(candidate)
    remaining = n_trials - sum(trial.state == TrialState.COMPLETE for trial in study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    if study.best_value is None:
        raise RuntimeError("optuna finished without a completed trial")
    params = _plain(dict(study.best_params))
    params["dataset"] = dataset
    _save_tuning_plots(study, history_path, importances_path, parallel_path, slice_path)
    payload: dict[str, object] = {
        "params": params,
        "val_nll": float(study.best_value),
        "val_bpd": bpd_from_nll(float(study.best_value)),
        "n_trials": n_trials,
        "epochs": epochs,
        "n_splits": n_splits,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"tuned val_nll={payload['val_nll']:.2f} val_bpd={payload['val_bpd']:.3f} "
        f"-> {out_path}; trials -> {storage_path}"
    )
    return payload


def _folds(
    contract: ModelingContract,
    pixels: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fit_pos, val_pos in splitter.split(train_idx, labels[train_idx]):
        fit_idx = train_idx[fit_pos]
        val_idx = train_idx[val_pos]
        folds.append(
            (
                contract.feature_matrix(pixels, fit_idx),
                labels[fit_idx],
                contract.feature_matrix(pixels, val_idx),
                labels[val_idx],
            )
        )
    return folds


def _hparams(trial: optuna.Trial) -> dict[str, float | int | str]:
    return {
        "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64]),
        "n_layers": trial.suggest_categorical("n_layers", [4]),
        "scale": trial.suggest_categorical("scale", [1.0, 2.0]),
        "lr": trial.suggest_categorical("lr", [1e-4, 1e-3]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
    }


def _plain(params: dict) -> dict[str, float | int | str]:
    plain: dict[str, float | int | str] = {}
    for key, value in params.items():
        if key in {"hidden_channels", "n_layers", "batch_size"}:
            plain[key] = int(value)
        elif key == "dataset":
            plain[key] = str(value)
        else:
            plain[key] = float(value)
    return plain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="04_tune", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--importances", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--dataset", default="digits", choices=["digits", "fashion", "both"])
    args = parser.parse_args(argv)
    run(
        args.data,
        args.pixels,
        args.split,
        args.out,
        storage_path=args.storage,
        history_path=args.history,
        importances_path=args.importances,
        parallel_path=args.parallel,
        slice_path=args.slice,
        n_trials=args.n_trials,
        epochs=args.epochs,
        n_splits=args.n_splits,
        dataset=args.dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
