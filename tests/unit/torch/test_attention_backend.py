"""Tests for PyTorch attention backend selection."""

from __future__ import annotations

import logging
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


def test_auto_selects_flash_attention_2_when_available(monkeypatch) -> None:
    _set_availability(monkeypatch, flash=True, sdpa=True)

    assert select_attn_implementation("auto") == "flash_attention_2"


def test_auto_selects_sdpa_when_flash_attention_2_is_unavailable(monkeypatch) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)

    assert select_attn_implementation("auto") == "sdpa"


def test_auto_selects_eager_when_no_accelerated_backend_is_available(
    monkeypatch,
) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=False)

    assert select_attn_implementation("auto") == "eager"


def test_explicit_unavailable_flash_attention_2_downgrades_with_warning(
    monkeypatch,
    caplog,
) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)
    caplog.set_level(logging.WARNING)

    selected = select_attn_implementation("flash_attention_2")

    assert selected == "sdpa"
    assert "flash_attention_2 is unavailable" in caplog.text
    assert "using sdpa instead" in caplog.text


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


def test_selection_logs_chosen_backend_once(monkeypatch, caplog) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)
    caplog.set_level(logging.INFO)

    assert select_attn_implementation("auto") == "sdpa"
    assert select_attn_implementation("auto") == "sdpa"

    messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("Using PyTorch attention backend:")
    ]
    assert messages == ["Using PyTorch attention backend: sdpa"]


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
def test_load_model_threads_attention_backend_to_model_from_pretrained(
    mock_config_class,
    mock_tokenizer_class,
    mock_model_class,
    monkeypatch,
) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)
    hf_config = Mock()
    hf_config.num_labels = 3
    hf_config.problem_type = "token_classification"
    hf_config.architectures = ["BertForTokenClassification"]
    mock_config_class.from_pretrained.return_value = hf_config
    mock_tokenizer_class.from_pretrained.return_value = Mock()
    model = Mock()
    model.config = hf_config
    mock_model_class.from_pretrained.return_value = model

    loader = ModelLoader(
        OpenMedConfig(
            backend="hf",
            device="cpu",
            torch_attention_backend="auto",
        )
    )
    loader.load_model("test-model")

    _, model_kwargs = mock_model_class.from_pretrained.call_args
    _, config_kwargs = mock_config_class.from_pretrained.call_args
    _, tokenizer_kwargs = mock_tokenizer_class.from_pretrained.call_args
    assert model_kwargs["attn_implementation"] == "sdpa"
    assert "attn_implementation" not in config_kwargs
    assert "attn_implementation" not in tokenizer_kwargs


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.pipeline")
def test_hf_pipeline_threads_attention_backend_through_model_kwargs(
    mock_pipeline,
    monkeypatch,
) -> None:
    _set_availability(monkeypatch, flash=False, sdpa=True)

    loader = ModelLoader(
        OpenMedConfig(
            backend="hf",
            device="cpu",
            torch_attention_backend="auto",
        )
    )
    loader.create_pipeline("test-model")

    _, pipeline_kwargs = mock_pipeline.call_args
    assert pipeline_kwargs["model_kwargs"]["attn_implementation"] == "sdpa"
