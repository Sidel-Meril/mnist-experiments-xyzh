"""Stage 04: Optuna search over latent size for the MNIST autoencoder.

The study minimizes mean reconstruction MSE from stratified cross-validation
on the train split. The test split is not read. The first trials, one per
latent size, keep every other setting fixed so the error curve is a comparison
of the bottleneck. Further trials, when requested, vary width, dropout,
learning rate, weight decay, and batch size. Trials are stored in a local
SQLite database and resumed when that file already holds the study.

Usage::

    python src/mnist_ae_compression/04_tune.py \
        --data data/mnist_ae_compression/images.parquet --pixels data/mnist_ae_compression/pixels.npy \
        --split data/mnist_ae_compression/split.json --out reports/mnist_ae_compression/best_params.json \
        --storage reports/mnist_ae_compression/mnist_ae.db \
        --history reports/mnist_ae_compression/plots/optimization_history.html \
        --importances reports/mnist_ae_compression/plots/param_importances.html \
        --parallel reports/mnist_ae_compression/plots/parallel_coordinate.html \
        --slice reports/mnist_ae_compression/plots/parameter_slice.html \
        --latent-curve reports/mnist_ae_compression/plots/reconstruction_error_vs_latent.png
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import plotly.graph_objects as go
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

from mnist_ae_compression.config import LATENT_SIZES, MODELING_CONTRACT_PATH, RANDOM_SEED
from mnist_ae_compression.model import Autoencoder
from input_contracts.model_contract import ModelingContract

_FIXED = {
    "hidden_dim": 256,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "batch_size": 128,
}


def _sqlite_url(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.resolve().as_posix()


def _completed_trials_figure(study: optuna.Study) -> go.Figure:
    xs: list[int] = []
    ys: list[float] = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            continue
        xs.append(trial.number)
        ys.append(float(trial.value))
    figure = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    figure.update_layout(title="completed trials", xaxis_title="trial", yaxis_title="val_mse")
    return figure


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
            try:
                figure = plot(study)
            except (RuntimeError, ValueError):
                figure = _completed_trials_figure(study)
            figure.write_html(path)


def _save_latent_curve(study: optuna.Study, path: Path) -> None:
    grouped: dict[int, list[float]] = {}
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            continue
        if "latent_dim" not in trial.params:
            continue
        grouped.setdefault(int(trial.params["latent_dim"]), []).append(float(trial.value))
    if not grouped:
        raise RuntimeError("no completed trials to plot")
    sizes = sorted(grouped)
    means = [float(np.mean(grouped[size])) for size in sizes]
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.plot(sizes, means, marker="o")
    ax.set_xlabel("latent size")
    ax.set_ylabel("validation reconstruction MSE")
    ax.set_xticks(sizes)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
    latent_curve_path: Path,
    n_trials: int = len(LATENT_SIZES),
    epochs: int = 2,
    n_splits: int = 5,
) -> dict[str, object]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    batch, pixels = contract.read(data_path, pixels_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    labels = batch[contract.target].detach().cpu().numpy()
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    counts = np.unique(labels[train_idx], return_counts=True)[1]
    if counts.size == 0 or int(counts.min()) < n_splits:
        raise ValueError(f"cross-validation needs at least {n_splits} train rows of each class")
    folds = _folds(contract, pixels, labels, train_idx, n_splits)
    tensor = contract.tensor_feature()
    if tensor.shape is None:
        raise ValueError("pixels feature is missing a shape")
    in_features = math.prod(tensor.shape)

    def objective(trial: optuna.Trial) -> float:
        chosen = _hparams(trial)
        chosen["in_features"] = in_features
        scores: list[float] = []
        for fold_number, (fit_x, val_x) in enumerate(folds):
            model = Autoencoder.from_hparams(chosen)
            seed = RANDOM_SEED + trial.number * n_splits + fold_number
            train_loader, val_loader = model.data_loaders(
                fit_x,
                val_x,
                batch_size=int(chosen["batch_size"]),
                seed=seed,
            )
            history = model.fit_from_dataset(
                train_loader,
                val_loader,
                chosen,
                epochs=epochs,
                seed=seed,
            )
            scores.append(float(history["eval"][-1]))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name="mnist_ae",
        storage=_sqlite_url(storage_path),
        load_if_exists=True,
    )
    if not study.trials:
        for size in LATENT_SIZES[:n_trials]:
            study.enqueue_trial({**_FIXED, "latent_dim": int(size)})
    remaining = n_trials - sum(trial.state == TrialState.COMPLETE for trial in study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    if study.best_value is None:
        raise RuntimeError("optuna finished without a completed trial")
    params = dict(study.best_params)
    params["in_features"] = in_features
    params = _plain(params)
    _save_tuning_plots(study, history_path, importances_path, parallel_path, slice_path)
    _save_latent_curve(study, latent_curve_path)
    payload: dict[str, object] = {
        "params": params,
        "val_mse": float(study.best_value),
        "n_trials": n_trials,
        "epochs": epochs,
        "n_splits": n_splits,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"tuned val_mse={payload['val_mse']:.4f} -> {out_path}; trials -> {storage_path}")
    return payload


def _folds(
    contract: ModelingContract,
    pixels: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fit_pos, val_pos in splitter.split(train_idx, labels[train_idx]):
        fit_idx = train_idx[fit_pos]
        val_idx = train_idx[val_pos]
        folds.append(
            (
                contract.feature_matrix(pixels, fit_idx),
                contract.feature_matrix(pixels, val_idx),
            )
        )
    return folds


def _hparams(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "latent_dim": trial.suggest_categorical("latent_dim", list(LATENT_SIZES)),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


def _plain(params: dict) -> dict[str, float | int]:
    plain: dict[str, float | int] = {}
    integers = {"latent_dim", "hidden_dim", "batch_size", "in_features"}
    for key, value in params.items():
        if key in integers:
            plain[key] = int(value)
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
    parser.add_argument("--latent-curve", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=len(LATENT_SIZES))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-splits", type=int, default=5)
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
        latent_curve_path=args.latent_curve,
        n_trials=args.n_trials,
        epochs=args.epochs,
        n_splits=args.n_splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
