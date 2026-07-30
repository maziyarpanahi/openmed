"""Tests for the memory-mapped / lazy MLX weight-loading toggle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openmed.mlx.models import _load_mlx_weights, _mlx_mmap_enabled


def _module_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


_MLX_AVAILABLE = _module_importable("mlx.core")


def test_mmap_enabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENMED_MLX_MMAP", raising=False)

    assert _mlx_mmap_enabled() is True


@pytest.mark.parametrize("value", ["", "   "])
def test_mmap_empty_value_falls_back_to_default(monkeypatch, value):
    monkeypatch.setenv("OPENMED_MLX_MMAP", value)

    assert _mlx_mmap_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " off "])
def test_mmap_disabled_for_falsey_values(monkeypatch, value):
    monkeypatch.setenv("OPENMED_MLX_MMAP", value)

    assert _mlx_mmap_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_mmap_enabled_for_truthy_values(monkeypatch, value):
    monkeypatch.setenv("OPENMED_MLX_MMAP", value)

    assert _mlx_mmap_enabled() is True


def test_load_mlx_weights_lazy_path_does_not_eval(monkeypatch):
    monkeypatch.delenv("OPENMED_MLX_MMAP", raising=False)
    arrays = {"a": object(), "b": object()}
    fake_mx = SimpleNamespace(load=MagicMock(return_value=arrays), eval=MagicMock())

    result = _load_mlx_weights(fake_mx, Path("weights.safetensors"))

    fake_mx.load.assert_called_once_with("weights.safetensors")
    fake_mx.eval.assert_not_called()
    assert result == arrays


def test_load_mlx_weights_eager_path_evaluates(monkeypatch):
    monkeypatch.setenv("OPENMED_MLX_MMAP", "0")
    arrays = {"a": object(), "b": object()}
    fake_mx = SimpleNamespace(load=MagicMock(return_value=arrays), eval=MagicMock())

    _load_mlx_weights(fake_mx, Path("weights.safetensors"))

    fake_mx.eval.assert_called_once_with(*arrays.values())


def test_load_mlx_weights_eager_path_handles_empty_weights(monkeypatch):
    monkeypatch.setenv("OPENMED_MLX_MMAP", "0")
    fake_mx = SimpleNamespace(load=MagicMock(return_value={}), eval=MagicMock())

    result = _load_mlx_weights(fake_mx, Path("weights.safetensors"))

    fake_mx.eval.assert_not_called()
    assert result == {}


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="requires mlx")
@pytest.mark.parametrize("mmap_flag", ["1", "0"])
def test_load_mlx_weights_roundtrip_matches(tmp_path, monkeypatch, mmap_flag):
    import mlx.core as mx

    monkeypatch.setenv("OPENMED_MLX_MMAP", mmap_flag)
    expected = {
        "layer.weight": mx.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer.scales": mx.array([0.5, 0.25]),  # quantized-style aux tensor
    }
    weights_path = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(weights_path), expected)

    loaded = _load_mlx_weights(mx, weights_path)

    assert set(loaded) == set(expected)
    for key, value in expected.items():
        assert bool(mx.all(loaded[key] == value)), key
