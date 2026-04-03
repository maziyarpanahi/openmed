"""Tests for the inference backend abstraction layer."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from openmed.core.backends import (
    HuggingFaceBackend,
    MLXBackend,
    get_backend,
    _BACKENDS,
)


class TestHuggingFaceBackend:

    @patch("openmed.core.backends.HuggingFaceBackend.is_available", return_value=True)
    def test_is_available_when_installed(self, _):
        backend = HuggingFaceBackend()
        assert backend.is_available() is True

    @patch("openmed.core.backends.HuggingFaceBackend.is_available", return_value=False)
    def test_not_available_when_missing(self, _):
        backend = HuggingFaceBackend()
        assert backend.is_available() is False


class TestMLXBackend:

    @patch("platform.system", return_value="Linux")
    def test_not_available_on_linux(self, _):
        backend = MLXBackend()
        assert backend.is_available() is False

    @patch("platform.system", return_value="Darwin")
    def test_not_available_without_mlx_package(self, _):
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            backend = MLXBackend()
            assert backend.is_available() is False


class TestGetBackend:

    @patch.object(HuggingFaceBackend, "is_available", return_value=True)
    def test_explicit_hf(self, _):
        backend = get_backend("hf")
        assert isinstance(backend, HuggingFaceBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("unknown_backend")

    @patch.object(MLXBackend, "is_available", return_value=False)
    def test_explicit_mlx_unavailable_raises(self, _):
        with pytest.raises(RuntimeError, match="not available"):
            get_backend("mlx")

    @patch.object(HuggingFaceBackend, "is_available", return_value=True)
    @patch.object(MLXBackend, "is_available", return_value=False)
    def test_auto_detect_falls_back_to_hf(self, _, __):
        backend = get_backend(None)
        assert isinstance(backend, HuggingFaceBackend)

    @patch.object(MLXBackend, "is_available", return_value=True)
    def test_auto_detect_prefers_mlx(self, _):
        backend = get_backend(None)
        assert isinstance(backend, MLXBackend)

    @patch.object(HuggingFaceBackend, "is_available", return_value=True)
    def test_config_passed_to_backend(self, _):
        config = MagicMock()
        backend = get_backend("hf", config=config)
        assert backend._config is config

    @patch.object(HuggingFaceBackend, "is_available", return_value=False)
    @patch.object(MLXBackend, "is_available", return_value=False)
    def test_no_backends_available_raises(self, _, __):
        with pytest.raises(RuntimeError, match="No inference backend"):
            get_backend(None)


class TestBackendRegistry:

    def test_hf_in_registry(self):
        assert "hf" in _BACKENDS

    def test_mlx_in_registry(self):
        assert "mlx" in _BACKENDS


class TestOpenMedConfigBackendField:

    def test_default_backend_is_none(self):
        from openmed.core.config import OpenMedConfig
        config = OpenMedConfig()
        assert config.backend is None

    def test_backend_can_be_set(self):
        from openmed.core.config import OpenMedConfig
        config = OpenMedConfig(backend="mlx")
        assert config.backend == "mlx"
