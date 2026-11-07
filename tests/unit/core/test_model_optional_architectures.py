"""Advisory architecture checks must not block otherwise loadable models."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import openmed.core.models as models
from openmed.core.config import OpenMedConfig
from openmed.core.errors import ModelLoadError


@pytest.fixture
def synthetic_loader(monkeypatch, tmp_path):
    config_factory = Mock()
    tokenizer = object()
    tokenizer_factory = Mock(return_value=tokenizer)
    model_factory = Mock()
    monkeypatch.setattr(
        models, "AutoConfig", SimpleNamespace(from_pretrained=config_factory)
    )
    monkeypatch.setattr(
        models, "AutoTokenizer", SimpleNamespace(from_pretrained=tokenizer_factory)
    )
    monkeypatch.setattr(
        models,
        "AutoModelForTokenClassification",
        SimpleNamespace(from_pretrained=model_factory),
    )
    # No real model files, accelerators, integrity downloads or weights are used.
    loader = models.ModelLoader(OpenMedConfig(backend="onnx", cache_dir=str(tmp_path)))
    monkeypatch.setattr(
        loader, "_prepare_model_reference", lambda *args, **kwargs: str(tmp_path)
    )
    monkeypatch.setattr(loader, "_resolve_torch_device", lambda prefer: "cpu")
    monkeypatch.setattr(loader, "_resolve_attn_implementation", lambda prefer: None)
    monkeypatch.setattr(
        loader, "_build_load_quantization_config", lambda *args, **kwargs: None
    )
    # Force a fresh cache entry without altering the tokenizer cache itself.
    return loader, config_factory, tokenizer_factory, model_factory, tokenizer


@pytest.mark.parametrize("architecture", [None, [], (), "missing"])
def test_missing_architecture_metadata_does_not_block_loading(
    synthetic_loader, architecture, caplog
):
    loader, config_factory, tokenizer_factory, model_factory, tokenizer = (
        synthetic_loader
    )
    config = SimpleNamespace(num_labels=2, problem_type=None)
    if architecture != "missing":
        config.architectures = architecture
    model = SimpleNamespace(config=config, to=Mock())
    config_factory.return_value = config
    model_factory.return_value = model
    with caplog.at_level(logging.WARNING, logger="openmed.core.models"):
        result = loader.load_model("synthetic/model", force_reload=True)
    assert result == {"model": model, "tokenizer": tokenizer, "config": config}
    assert model_factory.call_count == 1
    assert tokenizer_factory.call_count == 1
    assert model_factory.call_args.kwargs["local_files_only"] is True
    assert "may not be a TokenClassification model" in caplog.text
    model.to.assert_called_once_with("cpu")


@pytest.mark.parametrize(
    "architectures",
    [
        ["BertForTokenClassification"],
        ["SomeModel", "BertForTokenClassification"],
        [None, "BertForTokenClassification"],
    ],
)
def test_known_architecture_can_be_found_without_only_using_first_entry(
    synthetic_loader, architectures, caplog
):
    loader, cf, tf, mf, tokenizer = synthetic_loader
    config = SimpleNamespace(
        num_labels=2, problem_type=None, architectures=architectures
    )
    cf.return_value = config
    mf.return_value = SimpleNamespace(config=config)
    with caplog.at_level(logging.WARNING, logger="openmed.core.models"):
        loader.load_model("synthetic/model", force_reload=True)
    assert "may not be a TokenClassification model" not in caplog.text
    assert mf.call_count == 1


def test_factory_failure_is_still_wrapped_and_chained(synthetic_loader):
    loader, cf, tf, mf, tokenizer = synthetic_loader
    cf.return_value = SimpleNamespace(num_labels=2, problem_type=None, architectures=[])
    error = OSError("synthetic missing model weights")
    mf.side_effect = error
    with pytest.raises(ModelLoadError) as caught:
        loader.load_model("synthetic/model", force_reload=True)
    assert caught.value.__cause__ is error
    assert loader._models == {}
    assert loader._tokenizers == {}


def test_cached_model_path_remains_unchanged(synthetic_loader):
    loader, cf, tf, mf, tokenizer = synthetic_loader
    config = SimpleNamespace(num_labels=2, problem_type=None, architectures=None)
    model = SimpleNamespace(config=config)
    cf.return_value = config
    mf.return_value = model
    first = loader.load_model("synthetic/model", force_reload=True)
    second = loader.load_model("synthetic/model")
    assert first == second
    assert mf.call_count == 1
    assert tf.call_count == 1
