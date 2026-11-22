"""The saved checkpoint resolves locally, then from the DVC remote cache."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from mnist_pg_gan import model as model_module
from mnist_pg_gan.model import ProgressiveGAN


def _hparams() -> dict[str, float | int]:
    return {
        "noise_dim": 4,
        "width": 8,
        "image_size": 8,
        "channels": 3,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "beta1": 0.5,
        "batch_size": 8,
    }


def _checkpoint_bytes() -> bytes:
    buffer = io.BytesIO()
    spec = _hparams()
    module = ProgressiveGAN(
        noise_dim=int(spec["noise_dim"]),
        width=int(spec["width"]),
        image_size=int(spec["image_size"]),
        channels=int(spec["channels"]),
    )
    torch.save(
        {"state_dict": module.state_dict(), "columns": ["pixels"], "hparams": spec},
        buffer,
    )
    return buffer.getvalue()


def _isolate_model_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(model_module, "_session", None)
    monkeypatch.setattr(model_module, "LOCAL_MODEL", tmp_path / "absent.pt")
    monkeypatch.setenv("MNIST_PG_GAN_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("MNIST_PG_GAN_MODEL", raising=False)
    monkeypatch.delenv("MNIST_PG_GAN_REV", raising=False)
    monkeypatch.delenv("MNIST_PG_GAN_REPO", raising=False)


def test_missing_named_model_stays_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail(repo: str, rev: str | None) -> str:
        raise AssertionError((repo, rev))

    monkeypatch.setattr(model_module, "_remote_md5", fail)
    with pytest.raises(FileNotFoundError, match="progressive gan not found"):
        ProgressiveGAN.load(tmp_path / "missing.pt")


def test_local_artifact_skips_the_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "mnist_pg_gan.pt"
    model_path.write_bytes(_checkpoint_bytes())
    monkeypatch.setattr(model_module, "_session", None)
    monkeypatch.delenv("MNIST_PG_GAN_MODEL", raising=False)

    def fail(repo: str, rev: str | None) -> str:
        raise AssertionError((repo, rev))

    monkeypatch.setattr(model_module, "_remote_md5", fail)
    loaded = ProgressiveGAN.load(model_path)
    assert loaded["columns"] == ["pixels"]
    assert isinstance(loaded["model"], ProgressiveGAN)


def test_checkout_model_skips_the_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "mnist_pg_gan.pt"
    model_path.write_bytes(_checkpoint_bytes())
    monkeypatch.setattr(model_module, "_session", None)
    monkeypatch.setattr(model_module, "LOCAL_MODEL", model_path)
    monkeypatch.delenv("MNIST_PG_GAN_MODEL", raising=False)

    def fail(repo: str, rev: str | None) -> str:
        raise AssertionError((repo, rev))

    monkeypatch.setattr(model_module, "_remote_md5", fail)
    loaded = ProgressiveGAN.load()
    assert loaded["columns"] == ["pixels"]


def test_cache_miss_reads_with_dvc_api_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import dvc.api

    _isolate_model_lookup(monkeypatch, tmp_path)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    payload = _checkpoint_bytes()
    calls = {"open": 0}

    def md5(repo: str, rev: str | None) -> str:
        assert repo == model_module._MODEL_REPO
        assert rev is None
        return "abc123"

    def fake_open(path: str, **kwargs: object) -> io.BytesIO:
        calls["open"] += 1
        assert path == model_module._MODEL_RELATIVE
        assert kwargs["repo"] == model_module._MODEL_REPO
        assert kwargs["remote"] == "demo"
        assert kwargs["mode"] == "rb"
        assert "rev" not in kwargs
        return io.BytesIO(payload)

    monkeypatch.setattr(model_module, "_remote_md5", md5)
    monkeypatch.setattr(dvc.api, "open", fake_open)
    loaded = ProgressiveGAN.load()
    again = ProgressiveGAN.load()
    assert loaded["columns"] == ["pixels"]
    assert again["columns"] == ["pixels"]
    assert calls["open"] == 1
    assert (tmp_path / "cache" / "abc123.pt").is_file()


def test_cache_hit_skips_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _isolate_model_lookup(monkeypatch, tmp_path)
    cached = tmp_path / "cache" / "abc123.pt"
    cached.parent.mkdir()
    cached.write_bytes(_checkpoint_bytes())
    monkeypatch.setattr(model_module, "_remote_md5", lambda repo, rev: "abc123")

    def fail(*args: object, **kwargs: object) -> bytes:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(model_module, "_read_remote_model", fail)
    loaded = ProgressiveGAN.load()
    assert isinstance(loaded["model"], ProgressiveGAN)


def test_s3_cache_miss_requires_aws_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _isolate_model_lookup(monkeypatch, tmp_path)
    monkeypatch.setattr(model_module, "_remote_md5", lambda repo, rev: "abc123")
    monkeypatch.setattr(model_module, "_remote_url", lambda name: "s3://bucket/prefix")
    monkeypatch.setattr(model_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    def fail(*args: object, **kwargs: object) -> bytes:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(model_module, "_read_remote_model", fail)
    with pytest.raises(RuntimeError, match="AWS_ACCESS_KEY_ID"):
        ProgressiveGAN.load()
