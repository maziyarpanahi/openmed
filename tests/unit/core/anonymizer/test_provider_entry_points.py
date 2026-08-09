"""Tests for lazy anonymizer provider entry-point discovery."""

from __future__ import annotations

import importlib
import logging

import pytest

from openmed.core.anonymizer import Anonymizer, registry


@pytest.fixture(autouse=True)
def reset_provider_discovery():
    """Keep process-scoped discovery isolated between synthetic plugins."""

    registry._reset_provider_discovery_for_tests()
    yield
    registry._reset_provider_discovery_for_tests()


def test_entry_point_registrar_is_discovered_once_and_used_by_anonymizer(
    monkeypatch,
):
    label = "SYNTHETIC_PLUGIN_ID"
    marker = "SYNTHETIC-PLUGIN-SURROGATE"
    previous = registry.LABEL_GENERATORS.get(label)
    enumeration_calls = 0
    load_calls = 0

    def generator(faker, original, *, locale):
        del faker, original, locale
        return marker

    def register() -> None:
        registry.register_label_generator(label, generator)

    class FakeEntryPoint:
        name = "synthetic-anonymizer"

        def load(self):
            nonlocal load_calls
            load_calls += 1
            return register

    def fake_entry_points(*, group=None):
        nonlocal enumeration_calls
        enumeration_calls += 1
        assert group == registry.PROVIDER_ENTRY_POINT_GROUP
        return (FakeEntryPoint(),)

    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        fake_entry_points,
    )

    try:
        anonymizer = Anonymizer(lang="en")
        assert anonymizer.surrogate("synthetic input", label) == marker
        assert anonymizer.surrogate("another input", label) == marker
        assert enumeration_calls == 1
        assert load_calls == 1
    finally:
        if previous is None:
            registry.LABEL_GENERATORS.pop(label, None)
        else:
            registry.LABEL_GENERATORS[label] = previous


def test_broken_entry_point_warns_and_does_not_break_openmed_import(
    monkeypatch,
    caplog,
):
    class BrokenEntryPoint:
        name = "broken-anonymizer"

        def load(self):
            raise ImportError("synthetic plugin import failure")

    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda *, group=None: (BrokenEntryPoint(),),
    )
    caplog.set_level(logging.WARNING, logger=registry.__name__)

    registry.discover_provider_plugins()
    imported = importlib.import_module("openmed")

    assert imported.__name__ == "openmed"
    assert "broken-anonymizer" in caplog.text
    assert "ImportError" in caplog.text
