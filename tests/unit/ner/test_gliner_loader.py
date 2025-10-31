from __future__ import annotations

from unittest import mock

import pytest

from openmed.ner.families import gliner as gliner_module


@pytest.fixture(autouse=True)
def reset_cache():
    gliner_module._availability_cache = None
    gliner_module.clear_gliner_cache()
    yield
    gliner_module._availability_cache = None
    gliner_module.clear_gliner_cache()


def test_load_gliner_handle_uses_from_pretrained(monkeypatch):
    fake_model = mock.MagicMock()

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            FakeGLiNER.calls.append((model_id, kwargs))
            return fake_model

    FakeGLiNER.calls = []

    fake_module = mock.MagicMock()
    fake_module.GLiNER = FakeGLiNER

    monkeypatch.setattr(gliner_module, "ensure_gliner_available", lambda: None)
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)

    handle = gliner_module.load_gliner_handle("gliner-model", cache_dir="/tmp/cache", token="abc")
    assert handle.model is fake_model
    assert FakeGLiNER.calls[0][0] == "gliner-model"
    assert FakeGLiNER.calls[0][1]["cache_dir"] == "/tmp/cache"
    assert FakeGLiNER.calls[0][1]["token"] == "abc"


def test_clear_gliner_cache(monkeypatch):
    fake_model = mock.MagicMock()

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return fake_model

    fake_module = mock.MagicMock()
    fake_module.GLiNER = FakeGLiNER

    monkeypatch.setattr(gliner_module, "ensure_gliner_available", lambda: None)
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)

    first = gliner_module.load_gliner_handle("gliner-model")
    second = gliner_module.load_gliner_handle("gliner-model")
    assert first.model is second.model

    gliner_module.clear_gliner_cache()
    third = gliner_module.load_gliner_handle("gliner-model")
    assert third.model is fake_model
