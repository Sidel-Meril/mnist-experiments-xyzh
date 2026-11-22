"""Demo coverage: contract checks, normalized pixels, and a small Optuna search."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pandera.errors
import pytest

from mnist_betta_vae.config import (
    BETAS,
    DATA_CONTRACT_PATH,
    LATENT_DIM,
    MODELING_CONTRACT_PATH,
    project_root,
)
from mnist_contracts.data_contract import DataContract
from mnist_betta_vae.model import VAE
from mnist_contracts.model_contract import ModelingContract

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "src" / "mnist_betta_vae" / "queries" / "images.sql"

_STAGES: dict[str, dict] = {}


def _stage(filename: str) -> dict:
    loaded = _STAGES.get(filename)
    if loaded is None:
        loaded = runpy.run_path(str(ROOT / "src" / "mnist_betta_vae" / filename))
        _STAGES[filename] = loaded
    return loaded


def _sample(n_per_class: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = n_per_class * 10
    images = np.zeros((count, 28, 28, 1), dtype=np.uint8)
    labels = np.repeat(np.arange(10, dtype=np.int64), n_per_class)
    images[0] = 255
    splits = np.array(["train"] * count)
    return images, labels, splits


def _write_raw(
    tmp_path: Path,
    sample: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    dataset: str = "mnist_digits",
) -> tuple[Path, Path]:
    raw_path = tmp_path / "raw_images.parquet"
    images_path = tmp_path / "raw_images.npy"
    _stage("01_get_data.py")["load_data"](
        SQL, raw_path, images_path, sample=sample, dataset=dataset
    )
    return raw_path, images_path


def test_normalized_pixels_match_contract(tmp_path: Path):
    images, labels, splits = _sample(n_per_class=1)
    images[1] = 0
    raw_path, images_path = _write_raw(tmp_path, (images, labels, splits))
    out_path = tmp_path / "images.parquet"
    pixels_path = tmp_path / "pixels.npy"
    batch = _stage("02_engineer.py")["run"](raw_path, images_path, out_path, pixels_path)
    pixels = np.load(pixels_path)
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    label_names = next(
        column.categories for column in contract.columns if column.name == "label_name"
    )
    dataset_names = next(
        column.categories for column in contract.columns if column.name == "dataset"
    )

    assert list(batch.keys()) == [
        "image_id",
        "label",
        "label_name",
        "image_split",
        "dataset",
        "pixel_mean",
    ]
    assert label_names is not None
    assert dataset_names is not None
    assert label_names[int(batch["label_name"][0])] == "zero"
    assert label_names[int(batch["label_name"][1])] == "one"
    assert dataset_names[int(batch["dataset"][0])] == "mnist_digits"
    assert int(batch["image_id"][0]) == 1
    assert pixels.shape == (10, 1, 28, 28)
    assert np.allclose(pixels[0], 1.0)
    assert np.allclose(pixels[1], -1.0)
    assert float(batch["pixel_mean"][0]) == pytest.approx(1.0)
    assert float(batch["pixel_mean"][1]) == pytest.approx(-1.0)
    contract.validate(batch)


def test_fashion_label_name(tmp_path: Path):
    raw_path, images_path = _write_raw(tmp_path, _sample(n_per_class=1), dataset="fashion_mnist")
    batch = _stage("02_engineer.py")["run"](
        raw_path,
        images_path,
        tmp_path / "images.parquet",
        tmp_path / "pixels.npy",
    )
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    label_names = next(
        column.categories for column in contract.columns if column.name == "label_name"
    )
    dataset_names = next(
        column.categories for column in contract.columns if column.name == "dataset"
    )
    assert label_names is not None
    assert dataset_names is not None
    assert label_names[int(batch["label_name"][3])] == "dress"
    assert dataset_names[int(batch["dataset"][0])] == "fashion_mnist"


def test_unknown_label_is_rejected(tmp_path: Path):
    images, labels, splits = _sample(n_per_class=1)
    labels = labels.copy()
    labels[0] = 11
    raw_path, images_path = _write_raw(tmp_path, (images, labels, splits))
    with pytest.raises(pandera.errors.SchemaError):
        _stage("02_engineer.py")["run"](
            raw_path,
            images_path,
            tmp_path / "images.parquet",
            tmp_path / "pixels.npy",
        )


def test_official_split_is_kept(tmp_path: Path):
    frame_path = tmp_path / "images.parquet"
    pd.DataFrame({"image_split": ["train", "train", "test", "test"]}).to_parquet(
        frame_path, index=False
    )
    payload = _stage("03_split.py")["run"](frame_path, tmp_path / "split.json", test_size=0.5)
    assert payload["train_idx"] == [0, 1]
    assert payload["test_idx"] == [2, 3]


def test_vae_builds_loaders(tmp_path: Path):
    raw_path, images_path = _write_raw(tmp_path, _sample(n_per_class=2))
    data_path = tmp_path / "images.parquet"
    pixels_path = tmp_path / "pixels.npy"
    _stage("02_engineer.py")["run"](raw_path, images_path, data_path, pixels_path)
    split = _stage("03_split.py")["run"](data_path, tmp_path / "split.json", test_size=0.5)
    saved = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    contract = ModelingContract.load(MODELING_CONTRACT_PATH)
    _batch, pixels = contract.read(data_path, pixels_path)
    train_rows = np.asarray(split["train_idx"])
    test_rows = np.asarray(split["test_idx"])
    _train_loader, test_loader = VAE.data_loaders(
        contract.feature_matrix(pixels, train_rows),
        contract.feature_matrix(pixels, test_rows),
        batch_size=4,
        seed=7,
    )
    test_x, test_y = next(iter(test_loader))
    taken = len(test_x)
    assert set(saved) == {"train_idx", "test_idx"}
    assert "train_loader" not in split
    assert tuple(test_x.shape[1:]) == (1, 28, 28)
    assert len(test_loader.dataset) == len(test_rows)
    assert np.allclose(test_x.numpy(), pixels[test_rows[:taken]])
    assert np.allclose(test_y.numpy(), test_x.numpy())


def test_holdout_roundtrip(tmp_path: Path):
    raw_path, images_path = _write_raw(tmp_path, _sample())
    data_path = tmp_path / "images.parquet"
    pixels_path = tmp_path / "pixels.npy"
    split_path = tmp_path / "split.json"
    params_path = tmp_path / "best_params.json"
    storage_path = tmp_path / "mnist_betta_vae.db"
    history_path = tmp_path / "optimization_history.html"
    importances_path = tmp_path / "param_importances.html"
    parallel_path = tmp_path / "parallel_coordinate.html"
    slice_path = tmp_path / "parameter_slice.html"
    beta_curve_path = tmp_path / "loss_vs_beta.png"
    model_path = tmp_path / "mnist_betta_vae.pt"
    plot_path = tmp_path / "loss.png"
    generated_path = tmp_path / "generated_samples.png"
    tune_paths = {
        "storage_path": storage_path,
        "history_path": history_path,
        "importances_path": importances_path,
        "parallel_path": parallel_path,
        "slice_path": slice_path,
        "beta_curve_path": beta_curve_path,
    }
    _stage("02_engineer.py")["run"](raw_path, images_path, data_path, pixels_path)
    split = _stage("03_split.py")["run"](data_path, split_path, test_size=0.5)
    tuned = _stage("04_tune.py")["run"](
        data_path,
        pixels_path,
        split_path,
        params_path,
        n_trials=1,
        epochs=1,
        n_splits=2,
        **tune_paths,
    )
    for path in (history_path, importances_path, parallel_path, slice_path):
        path.unlink()
    resumed = _stage("04_tune.py")["run"](
        data_path,
        pixels_path,
        split_path,
        params_path,
        n_trials=1,
        epochs=1,
        n_splits=2,
        **tune_paths,
    )
    _stage("05_train.py")["run"](
        data_path,
        pixels_path,
        split_path,
        params_path,
        model_path,
        epochs=1,
        plot_path=plot_path,
    )
    written = _stage("06_eval.py")["run"](
        data_path,
        pixels_path,
        split_path,
        model_path,
        tmp_path / "evaluation_train.json",
        tmp_path / "evaluation_test.json",
        reconstructions_train_path=tmp_path / "reconstructions_train.png",
        reconstructions_test_path=tmp_path / "reconstructions_test.png",
        generated_samples_path=generated_path,
        latent_traversal_path=tmp_path / "latent_traversals.png",
    )
    explored = _stage("07_explore.py")["run"](
        data_path,
        pixels_path,
        split_path,
        params_path,
        tmp_path / "beta_exploration.json",
        models_dir=tmp_path / "beta_models",
        plots_dir=tmp_path / "beta_plots",
        epochs=1,
    )

    assert set(tuned["params"]) >= {
        "latent_dim",
        "hidden_dim",
        "dropout",
        "lr",
        "weight_decay",
        "batch_size",
        "in_features",
        "beta",
    }
    assert tuned["params"]["in_features"] == 784
    assert tuned["params"]["latent_dim"] == LATENT_DIM
    assert tuned["params"]["beta"] == pytest.approx(BETAS[0])
    assert hasattr(VAE.from_hparams(tuned["params"]), "fc_mu")
    assert hasattr(VAE.from_hparams(tuned["params"]), "fc_logvar")
    stored = optuna.load_study(
        study_name="mnist_betta_vae",
        storage="sqlite:///" + storage_path.resolve().as_posix(),
    )
    assert stored.direction == optuna.study.StudyDirection.MINIMIZE
    assert len(stored.trials) == 1
    assert stored.best_value == pytest.approx(float(tuned["val_loss"]))
    assert "trials" not in resumed
    assert float(resumed["val_loss"]) == pytest.approx(float(tuned["val_loss"]))
    saved_params = json.loads(params_path.read_text(encoding="utf-8"))
    assert set(saved_params) == {"params", "val_loss", "n_trials", "epochs", "n_splits"}
    for path in (plot_path, beta_curve_path, generated_path, tmp_path / "latent_traversals.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
    for path in (history_path, importances_path, parallel_path, slice_path):
        html = path.read_text(encoding="utf-8").lower()
        assert "<html" in html
        assert "plotly" in html
    for name in ("train", "test"):
        metrics = written[name]
        assert metrics["n_train"] == len(split["train_idx"])
        assert metrics["n_test"] == len(split["test_idx"])
        assert float(metrics["mse"]) >= 0.0
        assert float(metrics["mae"]) >= 0.0
        assert float(metrics["kl"]) >= 0.0
        assert metrics["latent_dim"] == tuned["params"]["latent_dim"]
        assert metrics["dataset"] == "mnist_digits"
        assert float(metrics["beta"]) == pytest.approx(BETAS[0])
    for stem in ("reconstructions_train.png", "reconstructions_test.png"):
        assert (tmp_path / stem).read_bytes().startswith(b"\x89PNG")
    runs = explored["runs"]
    assert isinstance(runs, list)
    assert [float(row["beta"]) for row in runs] == pytest.approx(list(BETAS))
    assert (tmp_path / "beta_plots" / "metrics_vs_beta.png").read_bytes().startswith(b"\x89PNG")
    for row in runs:
        assert isinstance(row, dict)
        assert 0 <= int(row["active_units"]) <= LATENT_DIM
        assert float(row["mse"]) >= 0.0
        assert float(row["diversity"]) >= 0.0
        assert len(row["kl_per_dim"]) == LATENT_DIM
        assert len(row["traversal_response"]) == LATENT_DIM
        assert (tmp_path / "beta_plots" / str(row["traversal"])).read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "beta_models" / str(row["model"])).is_file()
    saved = json.loads((tmp_path / "evaluation_test.json").read_text(encoding="utf-8"))
    assert set(saved) == {
        "mse",
        "mae",
        "kl",
        "elbo",
        "beta",
        "latent_dim",
        "dataset",
        "n_train",
        "n_test",
    }


def test_mock_sql_and_local_dvc():
    statement = SQL.read_text(encoding="utf-8")
    source = (ROOT / "src" / "mnist_betta_vae" / "01_get_data.py").read_text(encoding="utf-8")
    config = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    assert "select" in statement.lower()
    assert "MNIST" in source
    assert "FashionMNIST" in source
    assert "psycopg" not in source
    assert "sqlite3" not in source
    assert "demo" in config
    assert "s3://" not in config


def test_package_files_resolve_beside_the_module():
    assert DATA_CONTRACT_PATH.is_file()
    assert MODELING_CONTRACT_PATH.is_file()
    assert DATA_CONTRACT_PATH.parent == MODELING_CONTRACT_PATH.parent
    assert project_root() == ROOT


def test_data_contract_keeps_the_extract():
    raw = pd.DataFrame(
        {
            "image_id": [1],
            "label": [3],
            "image_split": ["train"],
            "dataset": ["mnist_digits"],
        }
    )
    frame = DataContract.load(DATA_CONTRACT_PATH).validate(
        DataContract.load(DATA_CONTRACT_PATH).apply(raw)
    )
    assert list(frame.columns) == ["image_id", "label", "image_split", "dataset"]
    assert frame.loc[0, "dataset"] == "mnist_digits"
    assert int(frame.loc[0, "label"]) == 3


def test_beta_weights_the_reported_elbo():
    model = VAE(input_dim=784, latent_dim=4, hidden_dim=32, dropout=0.0)
    images = np.zeros((4, 1, 28, 28), dtype=np.float32)
    low = model.score(images, batch_size=4, beta=0.1)
    high = model.score(images, batch_size=4, beta=10.0)
    sse = float(low["mse"]) * 784
    assert float(high["kl"]) == pytest.approx(float(low["kl"]))
    assert float(low["elbo"]) == pytest.approx(-(sse + 0.1 * float(low["kl"])))
    assert float(high["elbo"]) == pytest.approx(-(sse + 10.0 * float(high["kl"])))


def test_prior_traversal_walks_each_axis():
    model = VAE(input_dim=784, latent_dim=4, hidden_dim=32, dropout=0.0)
    frames = model.traverse(steps=5, span=2.0)
    assert frames.shape == (4, 5, 1, 28, 28)
    assert float(frames.min()) >= -1.0 - 1e-4
    assert float(frames.max()) <= 1.0 + 1e-4
