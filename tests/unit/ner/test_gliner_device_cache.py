"""Offline GLiNER cache isolation with a mutable model double."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openmed.ner.families import gliner


class ModelDouble:
    def __init__(self):
        self.device = "cpu"
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = device
        return self

    def predict_entities(self, text, labels, **kwargs):
        return (text, labels, kwargs)


@pytest.fixture
def loader(monkeypatch):
    gliner.clear_gliner_cache()
    monkeypatch.setattr(gliner, "ensure_gliner_available", lambda: None)
    factory = Mock(side_effect=lambda *_a, **_k: ModelDouble())
    module = SimpleNamespace(GLiNER=SimpleNamespace(from_pretrained=factory))
    original_import = gliner.importlib.import_module
    monkeypatch.setattr(
        gliner.importlib,
        "import_module",
        lambda name, *a, **k: (
            module if name == "gliner" else original_import(name, *a, **k)
        ),
    )
    yield factory
    gliner.clear_gliner_cache()


@pytest.mark.parametrize(
    "first_device,second_device",
    [
        ("cpu", "cuda:0"),
        ("cuda:0", "cpu"),
        ("cuda:0", "cuda:1"),
    ],
)
def test_second_device_does_not_move_first_handle(loader, first_device, second_device):
    first = gliner.load_gliner_handle("synthetic/model", device=first_device)
    second = gliner.load_gliner_handle("synthetic/model", device=second_device)
    assert first.model.device == first_device
    assert second.model.device == second_device
    assert first.model is not second.model
    assert loader.call_count == 2


def test_same_device_reuses_instance_without_repeated_move(loader):
    first = gliner.load_gliner_handle("synthetic/model", device="cpu")
    second = gliner.load_gliner_handle("synthetic/model", device="cpu")
    assert first.model is second.model
    assert first.model.moves == ["cpu"]
    assert loader.call_count == 1


def test_default_handle_is_not_moved_by_explicit_device(loader):
    first = gliner.load_gliner_handle("synthetic/model")
    gliner.load_gliner_handle("synthetic/model", device="cuda:0")
    assert first.model.device == "cpu"
    assert first.model.moves == []


def test_cache_clear_creates_new_instance(loader):
    first = gliner.load_gliner_handle("synthetic/model", device="cpu")
    gliner.clear_gliner_cache()
    second = gliner.load_gliner_handle("synthetic/model", device="cpu")
    assert first.model is not second.model


def test_original_loader_options_are_forwarded(loader):
    gliner.load_gliner_handle(
        "synthetic/model",
        cache_dir="synthetic-cache",
        token="synthetic-only",
        device="cpu",
    )
    loader.assert_called_once_with(
        "synthetic/model", cache_dir="synthetic-cache", token="synthetic-only"
    )


def test_prediction_options_are_unchanged(loader):
    handle = gliner.load_gliner_handle("synthetic/model", device="cpu")
    result = handle.predict_entities(
        "synthetic", ["label"], threshold=0.25, flat_ner=False, batch_size=2
    )
    assert result == (
        "synthetic",
        ["label"],
        {
            "threshold": 0.25,
            "flat_ner": False,
            "batch_size": 2,
        },
    )


def test_no_device_still_caches(loader):
    first = gliner.load_gliner_handle("synthetic/model")
    second = gliner.load_gliner_handle("synthetic/model", device="")
    assert first.model is second.model
    assert first.model.moves == []
