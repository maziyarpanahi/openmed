"""Tests for load-time bitsandbytes 4-bit model loading."""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import Mock

from openmed.core import models as models_module
from openmed.core.config import OpenMedConfig
from openmed.core.models import ModelLoader
from openmed.torch import loader_quant


class FakeBitsAndBytesConfig:
    """Small stand-in that records constructor kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


def test_openmed_config_round_trips_bnb_4bit_options() -> None:
    config = OpenMedConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False)

    payload = config.to_dict()
    restored = OpenMedConfig.from_dict(payload)

    assert payload["load_in_4bit"] is True
    assert payload["bnb_4bit_use_double_quant"] is False
    assert restored.load_in_4bit is True
    assert restored.bnb_4bit_use_double_quant is False


def test_load_model_threads_bnb_4bit_quantization_config(monkeypatch) -> None:
    _install_fake_bnb(monkeypatch)
    auto_model, model = _install_fake_hf_loader(monkeypatch)

    loader = ModelLoader(
        OpenMedConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False)
    )
    monkeypatch.setattr(loader, "_resolve_torch_device", lambda prefer=None: "cuda")

    loader.load_model("test-model")

    kwargs = auto_model.from_pretrained.call_args.kwargs
    quantization_config = kwargs["quantization_config"]
    assert isinstance(quantization_config, FakeBitsAndBytesConfig)
    assert quantization_config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": False,
    }
    assert "load_in_4bit" not in kwargs
    assert "bnb_4bit_use_double_quant" not in kwargs
    model.to.assert_not_called()


def test_create_pipeline_threads_bnb_4bit_model_kwargs(monkeypatch) -> None:
    _install_fake_bnb(monkeypatch)
    monkeypatch.setattr("openmed.core.backends._module_available", lambda _: True)
    pipeline = Mock()
    monkeypatch.setattr(models_module, "pipeline", pipeline)
    monkeypatch.setattr(models_module, "HF_AVAILABLE", True)

    loader = ModelLoader(OpenMedConfig(backend="hf", load_in_4bit=True))
    monkeypatch.setattr(loader, "_resolve_torch_device", lambda prefer=None: "cuda")

    loader.create_pipeline("test-model")

    kwargs = pipeline.call_args.kwargs
    quantization_config = kwargs["model_kwargs"]["quantization_config"]
    assert isinstance(quantization_config, FakeBitsAndBytesConfig)
    assert quantization_config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
    assert "load_in_4bit" not in kwargs
    assert "bnb_4bit_use_double_quant" not in kwargs


def test_missing_bitsandbytes_warns_and_falls_back(monkeypatch, caplog) -> None:
    auto_model, model = _install_fake_hf_loader(monkeypatch)

    def missing_bitsandbytes(module_name):
        if module_name == "bitsandbytes":
            raise ImportError("not installed")
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        loader_quant.importlib,
        "import_module",
        missing_bitsandbytes,
    )
    loader = ModelLoader(OpenMedConfig(load_in_4bit=True))
    monkeypatch.setattr(loader, "_resolve_torch_device", lambda prefer=None: "cuda")

    with caplog.at_level(logging.WARNING):
        loader.load_model("test-model")

    kwargs = auto_model.from_pretrained.call_args.kwargs
    assert "quantization_config" not in kwargs
    model.to.assert_called_once_with("cuda")
    assert "optional bitsandbytes package is not installed" in caplog.text
    assert "loading without 4-bit quantization" in caplog.text


def test_cpu_only_4bit_request_warns_and_falls_back(monkeypatch, caplog) -> None:
    _install_fake_bnb(monkeypatch)
    auto_model, model = _install_fake_hf_loader(monkeypatch)

    loader = ModelLoader(OpenMedConfig(load_in_4bit=True))
    monkeypatch.setattr(loader, "_resolve_torch_device", lambda prefer=None: "cpu")

    with caplog.at_level(logging.WARNING):
        loader.load_model("test-model")

    kwargs = auto_model.from_pretrained.call_args.kwargs
    assert "quantization_config" not in kwargs
    model.to.assert_called_once_with("cpu")
    assert "requires a CUDA device" in caplog.text
    assert "loading without 4-bit quantization" in caplog.text


def _install_fake_bnb(monkeypatch) -> None:
    fake_bnb = types.ModuleType("bitsandbytes")
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _install_fake_hf_loader(monkeypatch) -> tuple[Mock, Mock]:
    monkeypatch.setattr(models_module, "HF_AVAILABLE", True)

    config = Mock()
    config.num_labels = 3
    config.problem_type = "token_classification"
    config.architectures = ["BertForTokenClassification"]

    auto_config = Mock()
    auto_config.from_pretrained.return_value = config
    tokenizer = Mock()
    auto_tokenizer = Mock()
    auto_tokenizer.from_pretrained.return_value = tokenizer
    model = Mock()
    model.config = config
    auto_model = Mock()
    auto_model.from_pretrained.return_value = model

    monkeypatch.setattr(models_module, "AutoConfig", auto_config)
    monkeypatch.setattr(models_module, "AutoTokenizer", auto_tokenizer)
    monkeypatch.setattr(models_module, "AutoModelForTokenClassification", auto_model)
    return auto_model, model
