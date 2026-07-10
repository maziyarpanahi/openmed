"""Tests for PyTorch attention backend selection."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openmed.core.config import TORCH_ATTENTION_BACKEND_ENV_VAR, OpenMedConfig
from openmed.core.models import ModelLoader
from openmed.torch import attention
from openmed.torch.attention import select_attn_implementation


@pytest.fixture(autouse=True)
def clear_attention_log_cache() -> None:
    attention._LOGGED_BACKENDS.clear()


def _set_availability(monkeypatch, *, flash: bool, sdpa: bool) -> None:
    monkeypatch.setattr(
        attention,
        "_flash_attention_2_is_available",
        lambda: flash,
    )
    monkeypatch.setattr(attention, "_sdpa_is_available", lambda: sdpa)


def test_auto_defers_to_transformers_when_accelerated_backends_exist(
    monkeypatch,
) -> None:
    flash_available = Mock(return_value=True)
    sdpa_available = Mock(return_value=True)
    monkeypatch.setattr(
        attention,
        "_flash_attention_2_is_available",
        flash_available,
    )
    monkeypatch.setattr(attention, "_sdpa_is_available", sdpa_available)

    assert select_attn_implementation("auto") is None
    flash_available.assert_not_called()
    sdpa_available.assert_not_called()


def test_flash_attention_2_requires_package_and_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        attention.importlib.util,
        "find_spec",
        lambda name: object() if name == "flash_attn" else None,
    )
    monkeypatch.setattr(
        attention,
        "_import_torch",
        lambda: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    assert attention._flash_attention_2_is_available() is False

    monkeypatch.setattr(
        attention,
        "_import_torch",
        lambda: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )

    assert attention._flash_attention_2_is_available() is True


def test_explicit_unavailable_flash_attention_2_downgrades_with_warning(
    monkeypatch,
    caplog,
) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)
    caplog.set_level(logging.WARNING)

    selected = select_attn_implementation("flash_attention_2")

    assert selected == "eager"
    assert "flash_attention_2 is unavailable" in caplog.text
    assert "using eager instead" in caplog.text


def test_explicit_unavailable_sdpa_downgrades_to_eager_with_warning(
    monkeypatch,
    caplog,
) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=False)
    caplog.set_level(logging.WARNING)

    selected = select_attn_implementation("sdpa")

    assert selected == "eager"
    assert "sdpa is unavailable" in caplog.text
    assert "using eager instead" in caplog.text


def test_explicit_available_sdpa_is_preserved(monkeypatch) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)

    assert select_attn_implementation("sdpa") == "sdpa"


def test_auto_delegation_logs_once(monkeypatch, caplog) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)
    caplog.set_level(logging.INFO)

    assert select_attn_implementation("auto") is None
    assert select_attn_implementation("auto") is None

    messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("Using Transformers automatic attention")
    ]
    assert messages == ["Using Transformers automatic attention backend selection"]


def test_config_accepts_attention_backend_from_env(monkeypatch) -> None:
    monkeypatch.setenv(TORCH_ATTENTION_BACKEND_ENV_VAR, "eager")

    config = OpenMedConfig()

    assert config.torch_attention_backend == "eager"
    assert config.to_dict()["torch_attention_backend"] == "eager"


def test_config_dict_round_trips_attention_backend() -> None:
    config = OpenMedConfig.from_dict({"torch_attention_backend": "sdpa"})

    assert config.torch_attention_backend == "sdpa"
    assert config.to_dict()["torch_attention_backend"] == "sdpa"


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.AutoModelForTokenClassification")
@patch("openmed.core.models.AutoTokenizer")
@patch("openmed.core.models.AutoConfig")
@pytest.mark.parametrize(
    ("attention_backend", "expected_backend"),
    [
        pytest.param("auto", None, id="deberta-auto"),
        pytest.param("eager", "eager", id="explicit-eager"),
    ],
)
def test_loader_only_forwards_explicit_attention_backends(
    mock_config_class,
    mock_tokenizer_class,
    mock_model_class,
    monkeypatch,
    attention_backend,
    expected_backend,
) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)
    hf_config = Mock()
    hf_config.num_labels = 3
    hf_config.problem_type = "token_classification"
    hf_config.architectures = ["DebertaV2ForTokenClassification"]
    mock_config_class.from_pretrained.return_value = hf_config
    mock_tokenizer_class.from_pretrained.return_value = Mock()
    model = Mock()
    model.config = hf_config
    mock_model_class.from_pretrained.return_value = model

    loader = ModelLoader(
        OpenMedConfig(
            backend="hf",
            device="cpu",
            torch_attention_backend=attention_backend,
        )
    )
    loader.load_model("test-model")

    _, model_kwargs = mock_model_class.from_pretrained.call_args
    _, config_kwargs = mock_config_class.from_pretrained.call_args
    _, tokenizer_kwargs = mock_tokenizer_class.from_pretrained.call_args
    if expected_backend is None:
        assert "attn_implementation" not in model_kwargs
    else:
        assert model_kwargs["attn_implementation"] == expected_backend
    assert "attn_implementation" not in config_kwargs
    assert "attn_implementation" not in tokenizer_kwargs


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.pipeline")
@pytest.mark.parametrize(
    ("attention_backend", "expected_backend"),
    [
        pytest.param("auto", None, id="auto"),
        pytest.param("eager", "eager", id="explicit-eager"),
    ],
)
def test_pipeline_only_forwards_explicit_attention_backends(
    mock_pipeline,
    monkeypatch,
    attention_backend,
    expected_backend,
) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)

    loader = ModelLoader(
        OpenMedConfig(
            backend="hf",
            device="cpu",
            torch_attention_backend=attention_backend,
        )
    )
    loader.create_pipeline("test-model")

    _, pipeline_kwargs = mock_pipeline.call_args
    if expected_backend is None:
        assert "model_kwargs" not in pipeline_kwargs
    else:
        assert (
            pipeline_kwargs["model_kwargs"]["attn_implementation"] == expected_backend
        )
