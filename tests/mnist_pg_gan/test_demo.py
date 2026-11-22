"""Demo coverage: contract checks, fade-in, and a one-trial progressive search."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pandera.errors
import pytest
import torch
from torch.nn import functional as F

from mnist_pg_gan.config import DATA_CONTRACT_PATH, MODELING_CONTRACT_PATH, project_root
from input_contracts.data_contract import DataContract
from mnist_pg_gan.model import ProgressiveGAN
from input_contracts.model_contract import ModelingContract

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "src" / "mnist_pg_gan" / "queries" / "images.sql"

_STAGES: dict[str, dict] = {}


def _stage(filename: str) -> dict:
    loaded = _STAGES.get(filename)
    if loaded is None:
        loaded = runpy.run_path(str(ROOT / "src" / "mnist_pg_gan" / filename))
        _STAGES[filename] = loaded
    return loaded


def _sample(n_per_class: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = n_per_class * 10
    images = np.zeros((count, 32, 32, 3), dtype=np.uint8)
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
        "cifar_split",
        "pixel_mean",
    ]
    assert label_names is not None
    assert label_names[int(batch["label_name"][0])] == "airplane"
    assert label_names[int(batch["label_name"][1])] == "automobile"
    assert int(batch["image_id"][0]) == 1
    assert pixels.shape == (10, 3, 32, 32)
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


def test_official_cifar_split_is_kept(tmp_path: Path):
    frame_path = tmp_path / "images.parquet"
    pd.DataFrame({"cifar_split": ["train", "train", "test", "test"]}).to_parquet(
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
    _train_loader, test_loader = ProgressiveGAN.data_loaders(
        contract.feature_matrix(pixels, train_rows),
        contract.feature_matrix(pixels, test_rows),
        batch_size=4,
        seed=7,
    )
    (test_x,) = next(iter(test_loader))
    taken = len(test_x)
    assert set(saved) == {"train_idx", "test_idx"}
    assert "train_loader" not in split
    assert tuple(test_x.shape[1:]) == (3, 32, 32)
    assert len(test_loader.dataset) == len(test_rows)
    assert np.allclose(test_x.numpy(), pixels[test_rows[:taken]])


def test_generator_stays_in_pixel_range():
    model = ProgressiveGAN(noise_dim=8, width=8, image_size=32, channels=3)
    images = model.generate(4)
    assert images.shape == (4, 3, 32, 32)
    assert float(images.min()) >= -1.0
    assert float(images.max()) <= 1.0


def test_fade_in_matches_the_lower_resolution():
    model = ProgressiveGAN(noise_dim=8, width=8, image_size=8, channels=3)
    noise = torch.randn(2, 8)
    low = model.generator(noise, stage=0, alpha=1.0)
    faded = model.generator(noise, stage=1, alpha=0.0)
    assert tuple(low.shape[-2:]) == (4, 4)
    assert tuple(faded.shape[-2:]) == (8, 8)
    upsampled = F.interpolate(low, scale_factor=2, mode="nearest")
    assert torch.allclose(faded, upsampled)

    model.zero_grad(set_to_none=True)
    model.generator(noise, stage=1, alpha=0.0).sum().backward()
    faded_grad = model.generator.to_rgb[1].weight.grad
    assert faded_grad is not None
    assert float(faded_grad.abs().sum()) == 0.0

    model.zero_grad(set_to_none=True)
    model.generator(noise, stage=1, alpha=1.0).sum().backward()
    grown_grad = model.generator.to_rgb[1].weight.grad
    assert grown_grad is not None
    assert float(grown_grad.abs().sum()) > 0.0

    real = torch.rand(2, 3, 8, 8) * 2 - 1
    pooled = F.avg_pool2d(real, 2)
    score_low = model.discriminator(pooled, stage=0, alpha=1.0)
    score_fade = model.discriminator(real, stage=1, alpha=0.0)
    assert score_low.shape == (2, 1)
    assert torch.allclose(score_low, score_fade)


def test_holdout_roundtrip(tmp_path: Path):
    raw_path, images_path = _write_raw(tmp_path, _sample())
    data_path = tmp_path / "images.parquet"
    pixels_path = tmp_path / "pixels.npy"
    split_path = tmp_path / "split.json"
    params_path = tmp_path / "best_params.json"
    storage_path = tmp_path / "mnist_pg_gan.db"
    history_path = tmp_path / "optimization_history.html"
    importances_path = tmp_path / "param_importances.html"
    parallel_path = tmp_path / "parallel_coordinate.html"
    slice_path = tmp_path / "parameter_slice.html"
    noise_curve_path = tmp_path / "stability_vs_noise.png"
    model_path = tmp_path / "mnist_pg_gan.pt"
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
        "width",
        "batch_size",
        "image_size",
        "channels",
        "lr_g",
        "lr_d",
        "beta1",
    }
    assert tuned["params"]["image_size"] == 32
    assert tuned["params"]["channels"] == 3
    assert tuned["params"]["noise_dim"] == 8
    assert tuned["params"]["width"] == 8
    assert float(tuned["val_gap"]) >= 0.0
    stored = optuna.load_study(
        study_name="mnist_pg_gan",
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
        "resolution",
        "alpha",
        "stage",
    }
    assert history["resolution"] == [4.0, 8.0, 8.0, 16.0, 16.0, 32.0, 32.0]
    assert history["alpha"] == [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
    assert len(history["train_generator"]) == 7
    for path in (
        plot_path,
        noise_curve_path,
        generations / "epoch_0001_res04_stable.png",
        generations / "epoch_0007_res32_stable.png",
        generations / "growth_timeline.png",
    ):
        assert path.read_bytes().startswith(b"\x89PNG")
    growth = json.loads((generations / "growth.json").read_text(encoding="utf-8"))
    assert growth["display_size"] == 32
    assert [frame["resolution"] for frame in growth["frames"]] == [4, 8, 8, 16, 16, 32, 32]
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
        assert metrics["image_size"] == 32
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
        "image_size",
        "n_train",
        "n_test",
    }


def test_mock_sql_and_local_dvc():
    statement = SQL.read_text(encoding="utf-8")
    source = (ROOT / "src" / "mnist_pg_gan" / "01_get_data.py").read_text(encoding="utf-8")
    config = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    assert "select" in statement.lower()
    assert "CIFAR10" in source
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
    raw = pd.DataFrame({"image_id": [1], "label": [3], "cifar_split": ["train"]})
    frame = DataContract.load(DATA_CONTRACT_PATH).validate(
        DataContract.load(DATA_CONTRACT_PATH).apply(raw)
    )
    assert frame.loc[0, "label_name"] == "cat"
    assert list(frame.columns) == ["image_id", "label", "cifar_split", "label_name"]
