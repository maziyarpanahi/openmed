from __future__ import annotations

import logging

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer import registry as anonymizer_registry
from openmed.plugins import registry as plugin_registry
from openmed.plugins.protocols import (
    COMPONENT_ANONYMIZER_PROVIDER,
    PluginComponentMetadata,
)


class FakeEntryPoint:
    def __init__(self, name, loaded):
        self.name = name
        self.value = f"synthetic_plugins:{name}"
        self.group = plugin_registry.PLUGIN_ENTRY_POINT_GROUP
        self._loaded = loaded

    def load(self):
        if isinstance(self._loaded, BaseException):
            raise self._loaded
        return self._loaded


class ToyAnonymizerProvider:
    metadata = PluginComponentMetadata(
        plugin_id="synthetic-openmed",
        component_id="toy-id-provider",
        kind=COMPONENT_ANONYMIZER_PROVIDER,
        labels=("ID_NUM",),
        languages=("en",),
    )

    def __init__(self):
        self.calls = 0

    def replacement_for(self, span, surface, **kwargs):
        self.calls += 1
        assert span.doc_id == "anonymizer-provider-runtime"
        assert span.start == 0
        assert span.end == len(surface)
        assert span.canonical_label == "ID_NUM"
        assert span.detector == "plugin:synthetic-openmed:toy-id-provider"
        assert span.text_hash.startswith("hmac-sha256:")
        assert kwargs["locale"] == "en_US"
        assert "faker" in kwargs
        return "PLUGIN-ID-000"


@pytest.fixture(autouse=True)
def reset_plugin_state():
    anonymizer_registry._reset_anonymizer_provider_plugins_for_tests()
    plugin_registry._reset_plugin_registry_for_tests()
    yield
    anonymizer_registry._reset_anonymizer_provider_plugins_for_tests()
    plugin_registry._reset_plugin_registry_for_tests()


def _patch_entry_points(monkeypatch, *entry_points):
    calls = 0

    def fake_entry_points(*, group=None):
        nonlocal calls
        if group == anonymizer_registry.PROVIDER_ENTRY_POINT_GROUP:
            return ()
        calls += 1
        assert group == plugin_registry.PLUGIN_ENTRY_POINT_GROUP
        return entry_points

    monkeypatch.setattr(
        plugin_registry.importlib_metadata,
        "entry_points",
        fake_entry_points,
    )
    return lambda: calls


def _patch_legacy_entry_points(monkeypatch, *entry_points):
    calls = 0

    def fake_entry_points(*, group=None):
        nonlocal calls
        if group == plugin_registry.PLUGIN_ENTRY_POINT_GROUP:
            return ()
        calls += 1
        assert group == anonymizer_registry.PROVIDER_ENTRY_POINT_GROUP
        return entry_points

    monkeypatch.setattr(
        anonymizer_registry.importlib_metadata,
        "entry_points",
        fake_entry_points,
    )
    return lambda: calls


def test_legacy_registrar_group_is_discovered_once_and_used(monkeypatch):
    previous = anonymizer_registry.LABEL_GENERATORS["ID_NUM"]
    load_calls = 0

    def generator(faker, original, *, locale):
        del faker, original, locale
        return "LEGACY-PLUGIN-ID"

    def register():
        nonlocal load_calls
        load_calls += 1
        anonymizer_registry.register_label_generator("ID_NUM", generator)

    calls = _patch_legacy_entry_points(
        monkeypatch,
        FakeEntryPoint("legacy-provider", register),
    )

    try:
        anonymizer = Anonymizer(locale="en_US", consistent=True, seed=7)
        assert anonymizer.surrogate("MRN-12345", "ID_NUM") == "LEGACY-PLUGIN-ID"
        assert anonymizer.surrogate("MRN-67890", "ID_NUM") == "LEGACY-PLUGIN-ID"
        assert calls() == 1
        assert load_calls == 1
    finally:
        anonymizer_registry.LABEL_GENERATORS["ID_NUM"] = previous


def test_legacy_registrar_failure_is_sanitized(monkeypatch, caplog):
    calls = _patch_legacy_entry_points(
        monkeypatch,
        FakeEntryPoint("broken-legacy", RuntimeError("MRN-12345")),
    )

    with caplog.at_level(logging.WARNING):
        replacement = Anonymizer(locale="en_US").surrogate("MRN-12345", "ID_NUM")

    assert replacement
    assert calls() == 1
    assert "broken-legacy" in caplog.text
    assert "MRN-12345" not in caplog.text


def test_provider_is_discovered_on_first_anonymizer_use(monkeypatch):
    provider = ToyAnonymizerProvider()
    calls = _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("toy-provider", provider),
    )

    anonymizer = Anonymizer(lang="en", locale="en_US", consistent=True, seed=7)
    assert anonymizer.surrogate("MRN-12345", "ID_NUM") == "PLUGIN-ID-000"
    assert anonymizer.surrogate("MRN-67890", "ID_NUM") == "PLUGIN-ID-000"

    assert provider.calls == 2
    assert calls() == 1


def test_broken_provider_is_warned_and_builtin_fallback_survives(
    monkeypatch,
    caplog,
):
    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("broken-provider", RuntimeError("private source value")),
    )

    with caplog.at_level(logging.WARNING):
        replacement = Anonymizer(locale="en_US").surrogate("MRN-12345", "ID_NUM")

    assert replacement
    assert "broken-provider" in caplog.text
    assert "private source value" not in caplog.text


def test_runtime_failure_is_sanitized_and_uses_builtin_fallback(
    monkeypatch,
    caplog,
):
    class FailingProvider(ToyAnonymizerProvider):
        def replacement_for(self, span, surface, **kwargs):
            raise RuntimeError(surface)

    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("runtime-failure", FailingProvider()),
    )

    with caplog.at_level(logging.WARNING):
        replacement = Anonymizer(locale="en_US").surrogate("MRN-12345", "ID_NUM")

    assert replacement
    assert "synthetic-openmed:toy-id-provider" in caplog.text
    assert "MRN-12345" not in caplog.text


def test_provider_for_other_language_leaves_builtin_generator_active(monkeypatch):
    class FrenchProvider(ToyAnonymizerProvider):
        metadata = PluginComponentMetadata(
            plugin_id="synthetic-openmed",
            component_id="french-id-provider",
            kind=COMPONENT_ANONYMIZER_PROVIDER,
            labels=("ID_NUM",),
            languages=("fr",),
        )

    provider = FrenchProvider()
    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("french-provider", provider),
    )

    replacement = Anonymizer(locale="en_US").surrogate("MRN-12345", "ID_NUM")

    assert replacement
    assert provider.calls == 0
