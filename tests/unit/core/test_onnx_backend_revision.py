"""Capture ONNX loader options without downloading or loading a model."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from openmed.core.backends import OnnxBackend


@pytest.fixture
def loader(monkeypatch):
    model = SimpleNamespace(variant="fp32", tokenizer=object())
    load = Mock(return_value=model)
    module = ModuleType("openmed.onnx.inference")
    module.load_onnx_model = load
    monkeypatch.setitem(sys.modules, "openmed.onnx.inference", module)
    return load


@pytest.mark.parametrize(
    "configured,requested",
    [
        (None, "synthetic-revision"),
        ("configured-revision", "requested-revision"),
    ],
)
def test_explicit_revision_reaches_loader(loader, configured, requested):
    backend = OnnxBackend(SimpleNamespace(pii_model_revision=configured))
    backend.create_pipeline("synthetic/model", revision=requested)
    assert loader.call_args.kwargs["revision"] == requested


@pytest.mark.parametrize(
    "configured,expected",
    [
        (None, "main"),
        ("configured-revision", "configured-revision"),
        ("", "main"),
    ],
)
def test_omitted_revision_retains_config_fallback(loader, configured, expected):
    backend = OnnxBackend(SimpleNamespace(pii_model_revision=configured))
    backend.create_pipeline("synthetic/model")
    assert loader.call_args.kwargs["revision"] == expected


def test_none_revision_retains_config_fallback(loader):
    OnnxBackend(SimpleNamespace(pii_model_revision="configured")).create_pipeline(
        "synthetic/model", revision=None
    )
    assert loader.call_args.kwargs["revision"] == "configured"


def test_other_loader_safety_options_are_unchanged(loader):
    config = SimpleNamespace(
        cache_dir="synthetic-cache", hf_token="synthetic-only", local_only=True
    )
    pipeline = OnnxBackend(config).create_pipeline("synthetic/model", revision="v1")
    options = loader.call_args.kwargs
    assert options["local_files_only"] is True
    assert options["providers"] == ("CPUExecutionProvider",)
    assert options["cache_dir"] == "synthetic-cache"
    assert options["token"] == "synthetic-only"
    assert options["session_options"] is None
    assert pipeline.model is loader.return_value


def test_int8_requirement_is_not_weakened(loader):
    with pytest.raises(RuntimeError, match="requires model_int8.onnx"):
        OnnxBackend(SimpleNamespace(onnx_variant="int8")).create_pipeline(
            "synthetic/model", revision="synthetic-revision"
        )


def test_unsupported_task_still_fails_before_loading(loader):
    with pytest.raises(ValueError, match="token-classification"):
        OnnxBackend().create_pipeline("synthetic/model", task="unsupported")
    loader.assert_not_called()
