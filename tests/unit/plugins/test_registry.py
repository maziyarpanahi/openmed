from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from openmed.plugins import registry
from openmed.plugins.protocols import (
    COMPONENT_ANONYMIZER_PROVIDER,
    COMPONENT_EXPORTER,
    COMPONENT_INTEROP_ADAPTER,
    COMPONENT_LANGUAGE_PACK,
    COMPONENT_RECOGNIZER,
    PLUGIN_COMPONENT_KINDS,
    PLUGIN_SDK_VERSION,
    PluginComponentMetadata,
)
from openmed.plugins.registry import (
    REASON_DUPLICATE_COMPONENT,
    REASON_ENTRY_POINT_ENUMERATION_FAILED,
    REASON_INVALID_LABEL,
    REASON_INVALID_METADATA,
    REASON_LOAD_ERROR,
    REASON_MISSING_LABELS,
    REASON_NETWORK_EGRESS_OPT_IN_REQUIRED,
    REASON_NON_PERMISSIVE_LICENSE_OPT_IN_REQUIRED,
    REASON_PROTOCOL_VERSION_MISMATCH,
    REASON_UNKNOWN_COMPONENT_KIND,
    PluginRegistry,
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
        self.value = f"synthetic_plugins:{name}"
        self.group = registry.PLUGIN_ENTRY_POINT_GROUP
        self._loaded = loaded

    def load(self):
        if isinstance(self._loaded, BaseException):
            raise self._loaded
        return self._loaded


class ToyRecognizer:
    metadata = PluginComponentMetadata(
        plugin_id="synthetic-openmed",
        component_id="toy-person",
        kind=COMPONENT_RECOGNIZER,
        labels=("PERSON",),
        languages=("en",),
    )

    def recognize(self, text: str, **kwargs):
        return ()


class ToyExporter:
    metadata = PluginComponentMetadata(
        plugin_id="synthetic-openmed",
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


def test_sdk_declares_each_child_issue_component_kind():
    assert PLUGIN_COMPONENT_KINDS == {
        COMPONENT_RECOGNIZER,
        COMPONENT_ANONYMIZER_PROVIDER,
        COMPONENT_EXPORTER,
        COMPONENT_INTEROP_ADAPTER,
        COMPONENT_LANGUAGE_PACK,
    }


def test_valid_entry_point_is_discovered_once(monkeypatch):
    calls = _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("toy", lambda: (ToyRecognizer(), ToyExporter())),
    )

    result = discover_plugins()
    assert [item.metadata.qualified_id for item in result.registrations] == [
        "synthetic-openmed:toy-exporter",
        "synthetic-openmed:toy-person",
    ]
    assert result.quarantined == ()

    recognizers = iter_plugins(COMPONENT_RECOGNIZER)
    assert [item.metadata.component_id for item in recognizers] == ["toy-person"]
    assert calls() == 1


def test_protocol_version_mismatch_is_quarantined(monkeypatch):
    class FuturePlugin:
        metadata = PluginComponentMetadata(
            plugin_id="synthetic-openmed",
            component_id="future",
            kind=COMPONENT_RECOGNIZER,
            sdk_version="2.0.0",
            labels=("PERSON",),
        )

    _patch_entry_points(monkeypatch, FakeEntryPoint("future", FuturePlugin()))

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_PROTOCOL_VERSION_MISMATCH
    assert result.quarantined[0].plugin_id == "synthetic-openmed"
    assert "2.0.0" in result.quarantined[0].message


def test_malformed_protocol_version_is_quarantined(monkeypatch):
    class MalformedVersionPlugin:
        metadata = PluginComponentMetadata(
            plugin_id="synthetic-openmed",
            component_id="bad-version",
            kind=COMPONENT_EXPORTER,
            sdk_version="1",
        )

    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("bad-version", MalformedVersionPlugin()),
    )

    result = discover_plugins()

    assert result.quarantined[0].reason == REASON_PROTOCOL_VERSION_MISMATCH


def test_unknown_component_kind_is_quarantined(monkeypatch):
    class UnknownComponent:
        metadata = PluginComponentMetadata(
            plugin_id="synthetic-openmed",
            component_id="unknown",
            kind="model_downloader",
        )

    _patch_entry_points(monkeypatch, FakeEntryPoint("unknown", UnknownComponent()))

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_UNKNOWN_COMPONENT_KIND


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
        FakeEntryPoint("1-network", NetworkPlugin()),
        FakeEntryPoint("2-restricted", RestrictedLicensePlugin()),
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
    assert opted_in.registrations[0].loaded_by_policy_opt_in is False
    assert opted_in.registrations[1].loaded_by_policy_opt_in is True


def test_non_boolean_network_declaration_is_quarantined(monkeypatch):
    class AmbiguousNetworkPlugin:
        metadata = {
            "plugin_id": "synthetic-openmed",
            "component_id": "ambiguous-network",
            "kind": COMPONENT_EXPORTER,
            "network_egress": "false",
        }

    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("ambiguous-network", AmbiguousNetworkPlugin()),
    )

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_INVALID_METADATA
    assert "network_egress" in result.quarantined[0].message


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


def test_recognizer_without_declared_labels_is_quarantined(monkeypatch):
    class MissingLabelsPlugin:
        metadata = PluginComponentMetadata(
            plugin_id="missing-labels",
            component_id="recognizer",
            kind=COMPONENT_RECOGNIZER,
        )

    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("missing-labels", MissingLabelsPlugin()),
    )

    result = discover_plugins()

    assert result.quarantined[0].reason == REASON_MISSING_LABELS


def test_anonymizer_provider_without_declared_labels_is_quarantined(monkeypatch):
    class MissingLabelsPlugin:
        metadata = PluginComponentMetadata(
            plugin_id="missing-labels",
            component_id="anonymizer-provider",
            kind=COMPONENT_ANONYMIZER_PROVIDER,
        )

    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("missing-anonymizer-labels", MissingLabelsPlugin()),
    )

    result = discover_plugins()

    assert result.quarantined[0].reason == REASON_MISSING_LABELS
    assert "anonymizer provider" in result.quarantined[0].message


def test_malformed_plugin_fixture_is_quarantined_with_specific_reason(monkeypatch):
    class MissingMetadata:
        pass

    _patch_entry_points(monkeypatch, FakeEntryPoint("malformed", MissingMetadata()))

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_INVALID_METADATA
    assert "metadata" in result.quarantined[0].message


def test_malformed_component_does_not_hide_valid_sibling(monkeypatch):
    class MissingMetadata:
        pass

    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("mixed", (MissingMetadata(), ToyExporter())),
    )

    result = discover_plugins()

    assert [item.metadata.component_id for item in result.registrations] == [
        "toy-exporter"
    ]
    assert result.quarantined[0].reason == REASON_INVALID_METADATA


def test_entry_point_load_failure_does_not_escape_discovery(monkeypatch):
    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("broken-load", RuntimeError("synthetic failure")),
    )

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_LOAD_ERROR
    assert result.quarantined[0].message.endswith("RuntimeError")
    assert "synthetic failure" not in result.quarantined[0].message


def test_entry_point_enumeration_failure_is_structured(monkeypatch):
    def fail_enumeration(*, group=None):
        raise RuntimeError("synthetic internal detail")

    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        fail_enumeration,
    )

    result = discover_plugins()

    assert result.registrations == ()
    assert result.quarantined[0].reason == REASON_ENTRY_POINT_ENUMERATION_FAILED
    assert result.quarantined[0].message.endswith("RuntimeError")
    assert "synthetic internal detail" not in result.quarantined[0].message


def test_duplicate_component_is_quarantined_deterministically(monkeypatch):
    _patch_entry_points(
        monkeypatch,
        FakeEntryPoint("1-primary", ToyRecognizer()),
        FakeEntryPoint("2-duplicate", ToyRecognizer()),
    )

    result = discover_plugins()

    assert len(result.registrations) == 1
    assert result.registrations[0].entry_point_name == "1-primary"
    assert result.quarantined[0].reason == REASON_DUPLICATE_COMPONENT


def test_license_policy_accepts_only_permissive_expressions():
    assert is_permissive_license("Apache-2.0")
    assert is_permissive_license("Apache-2.0 OR MIT")
    assert is_permissive_license("CC-BY-4.0")
    assert not is_permissive_license("Apache-2.0 OR GPL-3.0-only")
    assert not is_permissive_license("GPL-3.0-only")
    assert not is_permissive_license("MIT!")
    assert not is_permissive_license("")


def test_registry_snapshot_does_not_trigger_discovery(monkeypatch):
    calls = _patch_entry_points(monkeypatch, FakeEntryPoint("toy", ToyRecognizer()))

    result = PluginRegistry().snapshot()

    assert result.registrations == ()
    assert result.quarantined == ()
    assert calls() == 0


def test_bare_import_does_not_import_or_discover_plugins():
    code = """
    import importlib.metadata
    import sys

    def fail_discovery(*args, **kwargs):
        raise AssertionError("plugin discovery was triggered")

    importlib.metadata.entry_points = fail_discovery

    import openmed

    assert "openmed.plugins" not in sys.modules
    assert not any(name.startswith("openmed.plugins.") for name in sys.modules)

    import openmed.plugins
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
