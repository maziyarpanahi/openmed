"""Tests for PyTorch device selection and MPS tuning."""

from __future__ import annotations

import sys
import types
from unittest.mock import Mock, patch

from openmed.core.config import OpenMedConfig
from openmed.core.models import ModelLoader
from openmed.torch.device import (
    LEGACY_DEVICE_ENV_VAR,
    MPS_ENV_DEFAULTS,
    TORCH_DEVICE_ENV_VAR,
    apply_mps_tuning,
    resolve_torch_device,
)


def _fake_torch(*, mps: bool = False, cuda: bool = False) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return torch


def test_auto_resolves_to_mps_before_cuda(monkeypatch) -> None:
    monkeypatch.delenv(TORCH_DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_DEVICE_ENV_VAR, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=True))

    assert resolve_torch_device() == "mps"


def test_auto_resolves_to_cuda_when_mps_unavailable(monkeypatch) -> None:
    monkeypatch.delenv(TORCH_DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_DEVICE_ENV_VAR, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=False, cuda=True))

    assert resolve_torch_device() == "cuda"


def test_auto_resolves_to_cpu_without_accelerators(monkeypatch) -> None:
    monkeypatch.delenv(TORCH_DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_DEVICE_ENV_VAR, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=False, cuda=False))

    assert resolve_torch_device() == "cpu"


def test_explicit_device_override_is_honored(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=True))

    assert resolve_torch_device("cpu") == "cpu"
    assert resolve_torch_device("cuda:1") == "cuda:1"
    assert resolve_torch_device("gpu") == "cuda"


def test_environment_device_override_is_honored(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=True))
    monkeypatch.setenv(TORCH_DEVICE_ENV_VAR, "cpu")

    assert resolve_torch_device() == "cpu"


def test_openmed_torch_device_takes_precedence(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=False, cuda=True))
    monkeypatch.setenv(TORCH_DEVICE_ENV_VAR, "mps")
    monkeypatch.setenv(LEGACY_DEVICE_ENV_VAR, "cpu")

    assert resolve_torch_device() == "mps"


def test_apply_mps_tuning_sets_expected_knobs(monkeypatch) -> None:
    for key in MPS_ENV_DEFAULTS:
        monkeypatch.delenv(key, raising=False)

    tuning = apply_mps_tuning()

    assert tuning.recommended_dtype == "float32"
    assert dict(tuning.env) == dict(MPS_ENV_DEFAULTS)
    for key, value in MPS_ENV_DEFAULTS.items():
        assert key in tuning.env
        assert tuning.env[key] == value


def test_apply_mps_tuning_preserves_existing_values(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    tuning = apply_mps_tuning()

    assert tuning.env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.pipeline")
def test_hf_pipeline_uses_resolved_mps_device(mock_pipeline, monkeypatch) -> None:
    monkeypatch.delenv(TORCH_DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_DEVICE_ENV_VAR, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=True))
    for key in MPS_ENV_DEFAULTS:
        monkeypatch.delenv(key, raising=False)

    loader = ModelLoader(OpenMedConfig(backend="hf"))
    loader.create_pipeline("test-model")

    _, kwargs = mock_pipeline.call_args
    assert kwargs["device"] == "mps"
    assert kwargs["model"] == "OpenMed/test-model"
    assert kwargs["use_fast"] is True
    assert all(key in kwargs for key in ("model", "device", "use_fast"))
    for key, value in MPS_ENV_DEFAULTS.items():
        assert key in tuning_env_subset()
        assert tuning_env_subset()[key] == value


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.pipeline")
def test_hf_pipeline_honors_config_device_override(mock_pipeline, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=True))

    loader = ModelLoader(OpenMedConfig(backend="hf", device="cpu"))
    loader.create_pipeline("test-model")

    _, kwargs = mock_pipeline.call_args
    assert kwargs["device"] == -1


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.AutoModelForTokenClassification")
@patch("openmed.core.models.AutoTokenizer")
@patch("openmed.core.models.AutoConfig")
def test_load_model_moves_model_to_resolved_device(
    mock_config_class,
    mock_tokenizer_class,
    mock_model_class,
    monkeypatch,
) -> None:
    monkeypatch.delenv(TORCH_DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_DEVICE_ENV_VAR, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=False))

    config = Mock()
    config.num_labels = 3
    config.problem_type = "token_classification"
    config.architectures = ["BertForTokenClassification"]
    mock_config_class.from_pretrained.return_value = config
    mock_tokenizer_class.from_pretrained.return_value = Mock()

    model = Mock()
    model.config = config
    mock_model_class.from_pretrained.return_value = model

    loader = ModelLoader(OpenMedConfig(backend="hf"))
    loader.load_model("test-model")

    model.to.assert_called_once_with("mps")


def tuning_env_subset() -> dict[str, str]:
    import os

    return {key: os.environ[key] for key in MPS_ENV_DEFAULTS if key in os.environ}
