from __future__ import annotations

import pytest

from openmed.plugins import registry
from openmed.plugins.protocols import (
    COMPONENT_EXPORTER,
    COMPONENT_RECOGNIZER,
    PLUGIN_SDK_VERSION,
    PluginComponentMetadata,
)
from openmed.plugins.registry import (
    REASON_INVALID_LABEL,
    REASON_INVALID_METADATA,
    REASON_NETWORK_EGRESS_OPT_IN_REQUIRED,
    REASON_NON_PERMISSIVE_LICENSE_OPT_IN_REQUIRED,
    REASON_PROTOCOL_VERSION_MISMATCH,
    discover_plugins,
    is_permissive_license,
    iter_plugins,
)


@pytest.fixture(autouse=True)
def reset_plugin_registry():
    registry._reset_plugin_registry_for_tests()
    yield
    registry._reset_plugin_registry_for_tests()


class FakeEntryPoint:
    def __init__(self, name, loaded):
        self.name = name
        self._loaded = loaded

    def load(self):
        if isinstance(self._loaded, BaseException):
            raise self._loaded
        return self._loaded


class ToyRecognizer:
    metadata = PluginComponentMetadata(
        plugin_id="acme-openmed",
        component_id="toy-person",
        kind=COMPONENT_RECOGNIZER,
        labels=("PERSON",),
        languages=("en",),
    )

    def recognize(self, text: str, **kwargs):
        return ()


class ToyExporter:
    metadata = PluginComponentMetadata(
        plugin_id="acme-openmed",
        component_id="toy-exporter",
        kind=COMPONENT_EXPORTER,
        labels=("PERSON",),
    )

    def export(self, spans, **kwargs):
        return {"spans": [span.to_dict() for span in spans]}


def _patch_entry_points(monkeypatch, *entry_points):
    calls = 0

    def fake_entry_points(*, group=None):
        nonlocal calls
        calls += 1
        assert group == registry.PLUGIN_ENTRY_POINT_GROUP
        return entry_points

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", fake_entry_points)
    return lambda: calls


def test_valid_entry_point_is_discovered_once(monkeypatch):
    calls = _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("toy", lambda: (ToyRecognizer(), ToyExporter())),
    )

    result = discover_plugins()
    assert [item.metadata.qualified_id for item in result.registrations] == [
        "acme-openmed:toy-exporter",
        "acme-openmed:toy-person",
    ]
    assert result.quarantined == ()

    recognizers = iter_plugins(COMPONENT_RECOGNIZER)
    assert [item.metadata.component_id for item in recognizers] == ["toy-person"]
    assert calls() == 1


def test_protocol_version_mismatch_is_quarantined(monkeypatch):
    class FuturePlugin:
        metadata = PluginComponentMetadata(
            plugin_id="acme-openmed",
            component_id="future",
            kind=COMPONENT_RECOGNIZER,
            sdk_version="2.0.0",
            labels=("PERSON",),
        )

    _patch_entry_points(monkeypatch, FakeEntryPoint("future", FuturePlugin()))

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_PROTOCOL_VERSION_MISMATCH
    assert result.quarantined[0].plugin_id == "acme-openmed"
    assert "2.0.0" in result.quarantined[0].message


def test_policy_restricted_plugins_require_explicit_opt_in(monkeypatch):
    class NetworkPlugin:
        metadata = PluginComponentMetadata(
            plugin_id="remote-plugin",
            component_id="network-person",
            kind=COMPONENT_RECOGNIZER,
            network_egress=True,
            labels=("PERSON",),
        )

    class RestrictedLicensePlugin:
        metadata = PluginComponentMetadata(
            plugin_id="restricted-plugin",
            component_id="person",
            kind=COMPONENT_RECOGNIZER,
            license="GPL-3.0-only",
            labels=("PERSON",),
        )

    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("network", NetworkPlugin()),
        FakeEntryPoint("restricted", RestrictedLicensePlugin()),
    )

    result = discover_plugins()
    assert result.registrations == ()
    assert [record.reason for record in result.quarantined] == [
        REASON_NETWORK_EGRESS_OPT_IN_REQUIRED,
        REASON_NON_PERMISSIVE_LICENSE_OPT_IN_REQUIRED,
    ]

    opted_in = discover_plugins(
        allow_network_egress=True,
        opt_in_plugins=("restricted-plugin:person",),
    )
    assert [item.metadata.qualified_id for item in opted_in.registrations] == [
        "remote-plugin:network-person",
        "restricted-plugin:person",
    ]
    assert opted_in.registrations[1].loaded_by_policy_opt_in is True


def test_unsupported_recognizer_label_is_quarantined(monkeypatch):
    class BadLabelPlugin:
        metadata = {
            "plugin_id": "bad-labels",
            "component_id": "alien",
            "kind": COMPONENT_RECOGNIZER,
            "sdk_version": PLUGIN_SDK_VERSION,
            "license": "Apache-2.0",
            "labels": ("ALIEN",),
        }

    _patch_entry_points(monkeypatch, FakeEntryPoint("bad-label", BadLabelPlugin()))

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_INVALID_LABEL
    assert "ALIEN" in result.quarantined[0].message


def test_malformed_plugin_fixture_is_quarantined_with_specific_reason(monkeypatch):
    class MissingMetadata:
        pass

    _patch_entry_points(monkeypatch, FakeEntryPoint("malformed", MissingMetadata()))

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_INVALID_METADATA
    assert "metadata" in result.quarantined[0].message


def test_license_policy_accepts_only_permissive_expressions():
    assert is_permissive_license("Apache-2.0")
    assert is_permissive_license("Apache-2.0 OR MIT")
    assert not is_permissive_license("GPL-3.0-only")
    assert not is_permissive_license("")
