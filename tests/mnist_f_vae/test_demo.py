"""Demo coverage: contract checks, RealNVP posterior, and a small Optuna search."""

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
from torch import nn

from mnist_f_vae.config import (
    DATA_CONTRACT_PATH,
    LATENT_SIZES,
    MODELING_CONTRACT_PATH,
    project_root,
)
from input_contracts.data_contract import DataContract
from mnist_f_vae.model import FVAE, CouplingLayer, LatentFlow, _flow_kl
from input_contracts.model_contract import ModelingContract

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "src" / "mnist_f_vae" / "queries" / "images.sql"

_STAGES: dict[str, dict] = {}


def _stage(filename: str) -> dict:
    loaded = _STAGES.get(filename)
    if loaded is None:
        loaded = runpy.run_path(str(ROOT / "src" / "mnist_f_vae" / filename))
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


def test_coupling_logdet_matches_jacobian():
    torch.manual_seed(0)
    mask = torch.tensor([1.0, 0.0])
    layer = CouplingLayer(2, 4, mask, scale=1.0)
    nn.init.normal_(layer.st.net[-1].weight, std=0.2)
    nn.init.normal_(layer.st.net[-1].bias, std=0.2)
    x = torch.randn(1, 2)
    _y, log_det = layer(x)
    epsilon = 1e-3
    jacobian = torch.zeros(2, 2)
    flat = x.detach().reshape(-1)
    for index in range(2):
        plus = flat.clone()
        minus = flat.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        y_plus, _ = layer(plus.view_as(x))
        y_minus, _ = layer(minus.view_as(x))
        jacobian[:, index] = ((y_plus - y_minus) / (2 * epsilon)).reshape(-1)
    _sign, logabsdet = torch.linalg.slogdet(jacobian)
    assert torch.allclose(log_det, logabsdet, atol=1e-2, rtol=1e-2)


def test_latent_flow_inverts_and_starts_as_identity():
    torch.manual_seed(1)
    flow = LatentFlow(dim=6, hidden=8, n_flows=4, scale=1.0)
    z0 = torch.randn(3, 6)
    zk, log_det = flow(z0)
    assert torch.allclose(zk, z0, atol=1e-5)
    assert torch.allclose(log_det, torch.zeros(3), atol=1e-5)
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    zk, log_det = flow(z0)
    restored = flow.inverse(zk)
    assert restored.shape == z0.shape
    assert torch.allclose(restored, z0, atol=1e-4, rtol=1e-4)
    assert log_det.shape == (3,)
    assert torch.isfinite(log_det).all()


def test_identity_flow_kl_matches_gaussian():
    torch.manual_seed(2)
    model = FVAE(
        input_dim=16,
        latent_dim=4,
        hidden_dim=8,
        dropout=0.0,
        n_flows=2,
        flow_hidden=8,
        flow_scale=1.0,
    )
    model.eval()
    mu, logvar = model.encode(torch.randn(32, 16))
    draws = []
    for _ in range(256):
        z0 = model.reparameterize(mu, logvar)
        zk, log_det = model.flow(z0)
        draws.append(_flow_kl(mu, logvar, z0, zk, log_det))
    monte_carlo = torch.stack(draws).mean(dim=0)
    analytic = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    assert torch.allclose(monte_carlo, analytic, atol=0.2, rtol=0.1)


def test_sample_stays_in_tanh_range():
    torch.manual_seed(3)
    model = FVAE(
        input_dim=784,
        latent_dim=4,
        hidden_dim=8,
        dropout=0.0,
        n_flows=2,
        flow_hidden=8,
        flow_scale=1.0,
    )
    model.eval()
    drawn = model.sample(4)
    assert drawn.shape == (4, 1, 28, 28)
    assert float(drawn.min()) >= -1.0
    assert float(drawn.max()) <= 1.0


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


def test_fvae_builds_loaders(tmp_path: Path):
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
    _train_loader, test_loader = FVAE.data_loaders(
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
    storage_path = tmp_path / "mnist_f_vae.db"
    history_path = tmp_path / "optimization_history.html"
    importances_path = tmp_path / "param_importances.html"
    parallel_path = tmp_path / "parallel_coordinate.html"
    slice_path = tmp_path / "parameter_slice.html"
    latent_curve_path = tmp_path / "loss_vs_latent.png"
    model_path = tmp_path / "mnist_f_vae.pt"
    plot_path = tmp_path / "loss.png"
    generated_path = tmp_path / "generated_samples.png"
    tune_paths = {
        "storage_path": storage_path,
        "history_path": history_path,
        "importances_path": importances_path,
        "parallel_path": parallel_path,
        "slice_path": slice_path,
        "latent_curve_path": latent_curve_path,
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
        "n_flows",
        "flow_hidden",
        "flow_scale",
    }
    assert tuned["params"]["in_features"] == 784
    assert tuned["params"]["latent_dim"] == LATENT_SIZES[0]
    assert tuned["params"]["beta"] == pytest.approx(1.0)
    assert hasattr(FVAE.from_hparams(tuned["params"]), "fc_mu")
    assert hasattr(FVAE.from_hparams(tuned["params"]), "fc_logvar")
    assert hasattr(FVAE.from_hparams(tuned["params"]), "flow")
    stored = optuna.load_study(
        study_name="mnist_f_vae",
        storage="sqlite:///" + storage_path.resolve().as_posix(),
    )
    assert stored.direction == optuna.study.StudyDirection.MINIMIZE
    assert len(stored.trials) == 1
    assert stored.best_value == pytest.approx(float(tuned["val_loss"]))
    assert "trials" not in resumed
    assert float(resumed["val_loss"]) == pytest.approx(float(tuned["val_loss"]))
    saved_params = json.loads(params_path.read_text(encoding="utf-8"))
    assert set(saved_params) == {"params", "val_loss", "n_trials", "epochs", "n_splits"}
    for path in (plot_path, latent_curve_path, generated_path):
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
        assert np.isfinite(float(metrics["kl"]))
        assert np.isfinite(float(metrics["elbo"]))
        assert np.isfinite(float(metrics["bpd"]))
        assert np.isfinite(float(metrics["log_det"]))
        assert metrics["latent_dim"] == tuned["params"]["latent_dim"]
        assert metrics["n_flows"] == tuned["params"]["n_flows"]
        assert metrics["dataset"] == "mnist_digits"
        assert float(metrics["beta"]) == pytest.approx(1.0)
    for stem in ("reconstructions_train.png", "reconstructions_test.png"):
        assert (tmp_path / stem).read_bytes().startswith(b"\x89PNG")
    saved = json.loads((tmp_path / "evaluation_test.json").read_text(encoding="utf-8"))
    assert set(saved) == {
        "mse",
        "mae",
        "kl",
        "elbo",
        "bpd",
        "log_det",
        "beta",
        "latent_dim",
        "n_flows",
        "dataset",
        "n_train",
        "n_test",
    }


def test_mock_sql_and_local_dvc():
    statement = SQL.read_text(encoding="utf-8")
    source = (ROOT / "src" / "mnist_f_vae" / "01_get_data.py").read_text(encoding="utf-8")
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
