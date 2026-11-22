"""Demo coverage: contract checks, normalized pixels, and a one-trial GAN search."""

from __future__ import annotations

import importlib
import json
import runpy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pandera.errors
import pytest

from mnist_gan_comparison.config import DATA_CONTRACT_PATH, MODELING_CONTRACT_PATH, project_root
from input_contracts.data_contract import DataContract
from mnist_gan_comparison.model import VanillaGAN
from input_contracts.model_contract import ModelingContract

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "src" / "mnist_gan_comparison" / "queries" / "images.sql"

_STAGES: dict[str, dict] = {}


def _stage(filename: str) -> dict:
    loaded = _STAGES.get(filename)
    if loaded is None:
        loaded = runpy.run_path(str(ROOT / "src" / "mnist_gan_comparison" / filename))
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
    tmp_path: Path, sample: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[Path, Path]:
    raw_path = tmp_path / "raw_images.parquet"
    images_path = tmp_path / "raw_images.npy"
    _stage("01_get_data.py")["load_data"](SQL, raw_path, images_path, sample=sample)
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

    assert list(batch.keys()) == [
        "image_id",
        "label",
        "label_name",
        "mnist_split",
        "pixel_mean",
    ]
    assert label_names is not None
    assert label_names[int(batch["label_name"][0])] == "zero"
    assert label_names[int(batch["label_name"][1])] == "one"
    assert int(batch["image_id"][0]) == 1
    assert pixels.shape == (10, 1, 28, 28)
    assert np.allclose(pixels[0], 1.0)
    assert np.allclose(pixels[1], -1.0)
    assert float(batch["pixel_mean"][0]) == pytest.approx(1.0)
    assert float(batch["pixel_mean"][1]) == pytest.approx(-1.0)
    contract.validate(batch)


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


def test_official_mnist_split_is_kept(tmp_path: Path):
    frame_path = tmp_path / "images.parquet"
    pd.DataFrame({"mnist_split": ["train", "train", "test", "test"]}).to_parquet(
        frame_path, index=False
    )
    payload = _stage("03_split.py")["run"](frame_path, tmp_path / "split.json", test_size=0.5)
    assert payload["train_idx"] == [0, 1]
    assert payload["test_idx"] == [2, 3]


def test_gan_builds_loaders(tmp_path: Path):
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
    _train_loader, test_loader = VanillaGAN.data_loaders(
        contract.feature_matrix(pixels, train_rows),
        contract.feature_matrix(pixels, test_rows),
        batch_size=4,
        seed=7,
    )
    (test_x,) = next(iter(test_loader))
    taken = len(test_x)
    assert set(saved) == {"train_idx", "test_idx"}
    assert "train_loader" not in split
    assert tuple(test_x.shape[1:]) == (1, 28, 28)
    assert len(test_loader.dataset) == len(test_rows)
    assert np.allclose(test_x.numpy(), pixels[test_rows[:taken]])


def test_generator_stays_in_pixel_range():
    model = VanillaGAN(noise_dim=8, hidden_dim=16, output_dim=784)
    images = model.generate(4)
    assert images.shape == (4, 1, 28, 28)
    assert float(images.min()) >= -1.0
    assert float(images.max()) <= 1.0


def test_both_datasets_share_the_table_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    get_data = importlib.import_module("mnist_gan_comparison.01_get_data")
    seen: list[str] = []

    def fake(dataset: str = "digits") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seen.append(dataset)
        images = np.zeros((20, 28, 28, 1), dtype=np.uint8)
        labels = np.tile(np.arange(10, dtype=np.int64), 2)
        splits = np.array(["train"] * 10 + ["test"] * 10)
        return images, labels, splits

    monkeypatch.setattr(get_data, "_download", fake)
    columns = None
    for dataset in ("digits", "fashion"):
        frame = get_data.load_data(
            SQL,
            tmp_path / f"{dataset}.parquet",
            tmp_path / f"{dataset}.npy",
            dataset=dataset,
        )
        columns = list(frame.columns) if columns is None else columns
        assert list(frame.columns) == columns
    assert seen == ["digits", "fashion"]
    assert columns == ["image_id", "label", "mnist_split"]


def test_holdout_roundtrip(tmp_path: Path):
    raw_path, images_path = _write_raw(tmp_path, _sample())
    data_path = tmp_path / "images.parquet"
    pixels_path = tmp_path / "pixels.npy"
    split_path = tmp_path / "split.json"
    params_path = tmp_path / "best_params.json"
    storage_path = tmp_path / "mnist_gan.db"
    history_path = tmp_path / "optimization_history.html"
    importances_path = tmp_path / "param_importances.html"
    parallel_path = tmp_path / "parallel_coordinate.html"
    slice_path = tmp_path / "parameter_slice.html"
    noise_curve_path = tmp_path / "stability_vs_noise.png"
    model_path = tmp_path / "mnist_gan.pt"
    plot_path = tmp_path / "loss.png"
    generations = tmp_path / "generations"
    tune_paths = {
        "storage_path": storage_path,
        "history_path": history_path,
        "importances_path": importances_path,
        "parallel_path": parallel_path,
        "slice_path": slice_path,
        "noise_curve_path": noise_curve_path,
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
    history = _stage("05_train.py")["run"](
        data_path,
        pixels_path,
        split_path,
        params_path,
        model_path,
        epochs=1,
        plot_path=plot_path,
        generations_dir=generations,
    )
    written = _stage("06_eval.py")["run"](
        data_path,
        pixels_path,
        split_path,
        model_path,
        tmp_path / "evaluation_train.json",
        tmp_path / "evaluation_test.json",
        generated_path=tmp_path / "generated.png",
        real_path=tmp_path / "real_test.png",
    )

    assert set(tuned["params"]) >= {
        "noise_dim",
        "hidden_dim",
        "batch_size",
        "in_features",
        "lr_g",
        "lr_d",
        "beta1",
    }
    assert tuned["params"]["in_features"] == 784
    assert tuned["params"]["noise_dim"] == 64
    assert float(tuned["val_gap"]) >= 0.0
    stored = optuna.load_study(
        study_name="mnist_gan",
        storage="sqlite:///" + storage_path.resolve().as_posix(),
    )
    assert stored.direction == optuna.study.StudyDirection.MINIMIZE
    assert len(stored.trials) == 1
    assert stored.best_value == pytest.approx(float(tuned["val_gap"]))
    assert "trials" not in resumed
    assert float(resumed["val_gap"]) == pytest.approx(float(tuned["val_gap"]))
    saved_params = json.loads(params_path.read_text(encoding="utf-8"))
    assert set(saved_params) == {"params", "val_gap", "n_trials", "epochs", "n_splits"}
    assert set(history) == {
        "train_discriminator",
        "train_generator",
        "eval_discriminator",
        "eval_generator",
        "sample_std",
    }
    assert len(history["train_generator"]) == 1
    for path in (plot_path, noise_curve_path, generations / "epoch_0001.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
    for path in (history_path, importances_path, parallel_path, slice_path):
        html = path.read_text(encoding="utf-8").lower()
        assert "<html" in html
        assert "plotly" in html
    for name in ("train", "test"):
        metrics = written[name]
        assert metrics["n_train"] == len(split["train_idx"])
        assert metrics["n_test"] == len(split["test_idx"])
        assert float(metrics["discriminator_loss"]) >= 0.0
        assert float(metrics["generator_loss"]) >= 0.0
        assert metrics["noise_dim"] == tuned["params"]["noise_dim"]
        assert metrics["mode_collapse"] in (True, False)
    for stem in ("generated.png", "real_test.png"):
        assert (tmp_path / stem).read_bytes().startswith(b"\x89PNG")
    saved = json.loads((tmp_path / "evaluation_test.json").read_text(encoding="utf-8"))
    assert set(saved) == {
        "discriminator_loss",
        "generator_loss",
        "sample_std",
        "real_std",
        "mode_collapse",
        "noise_dim",
        "n_train",
        "n_test",
    }


def test_mock_sql_and_local_dvc():
    statement = SQL.read_text(encoding="utf-8")
    source = (ROOT / "src" / "mnist_gan_comparison" / "01_get_data.py").read_text(encoding="utf-8")
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


def test_data_contract_maps_label_name():
    raw = pd.DataFrame({"image_id": [1], "label": [3], "mnist_split": ["train"]})
    frame = DataContract.load(DATA_CONTRACT_PATH).validate(
        DataContract.load(DATA_CONTRACT_PATH).apply(raw)
    )
    assert frame.loc[0, "label_name"] == "three"
    assert list(frame.columns) == ["image_id", "label", "mnist_split", "label_name"]
