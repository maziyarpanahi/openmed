from __future__ import annotations

import importlib
from unittest import mock

import pytest

from openmed.ner.exceptions import MissingDependencyError
from openmed.ner.families import ensure_gliner_available, is_gliner_available


@pytest.fixture(autouse=True)
def reset_availability_cache():
    from openmed.ner.families import gliner as gliner_module

    gliner_module._availability_cache = None
    yield
    gliner_module._availability_cache = None


def test_is_gliner_available_false_when_missing() -> None:
    with mock.patch("importlib.import_module", side_effect=ImportError()):
        assert is_gliner_available(force_refresh=True) is False


def test_ensure_gliner_available_raises_when_missing() -> None:
    with mock.patch("importlib.import_module", side_effect=ImportError()):
        with pytest.raises(MissingDependencyError) as exc:
            ensure_gliner_available()
        assert "gliner" in str(exc.value)


def test_ensure_gliner_available_success() -> None:
    def import_module(name):
        module = mock.MagicMock()
        module.GLiNER.from_pretrained.return_value = object()
        return module

    with mock.patch("importlib.import_module", side_effect=import_module):
        assert ensure_gliner_available() is None
