"""Offline tests for importless plugin metadata quarantine reports."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import socket

from openmed.plugins.quarantine import (
    CATEGORY_AVAILABLE,
    CATEGORY_DISABLED,
    CATEGORY_QUARANTINED,
    REASON_DUPLICATE_NAME,
    REASON_INVALID_API_VERSION,
    REASON_INVALID_METADATA,
    REASON_MISSING_CAPABILITIES,
    REASON_UNSUPPORTED_API_VERSION,
    REASON_UNSUPPORTED_CAPABILITY,
    build_quarantine_report,
)


def _plugin(name: str, *, capabilities: tuple[str, ...] = ("recognizer",)) -> dict:
    return {
        "name": name,
        "api_version": "1.0.0",
        "capabilities": capabilities,
    }


def test_report_does_not_enumerate_or_load_entry_points(monkeypatch) -> None:
    def fail_entry_points(*args, **kwargs):
        del args, kwargs
        raise AssertionError("static quarantine preflight enumerated entry points")

    def fail_network(*args, **kwargs):
        del args, kwargs
        raise AssertionError("static quarantine preflight attempted network I/O")

    monkeypatch.setattr(importlib_metadata, "entry_points", fail_entry_points)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket, "socket", fail_network)

    report = build_quarantine_report([_plugin("local-recognizer")])

    assert [record.name for record in report.available] == ["local-recognizer"]
    assert report.quarantined == ()


def test_api_and_capability_failures_are_safe_categories() -> None:
    report = build_quarantine_report(
        [
            _plugin("valid"),
            {**_plugin("future"), "api_version": "2.0.0"},
            {"name": "missing-capabilities", "api_version": "1.0.0"},
            {
                **_plugin("unknown-capability"),
                "capabilities": ["credential_reader"],
            },
            {**_plugin("malformed-api"), "api_version": "not-a-version"},
        ]
    )

    assert [record.name for record in report.available] == ["valid"]
    reasons = {record.name: record.reason for record in report.quarantined}
    assert reasons == {
        "future": REASON_UNSUPPORTED_API_VERSION,
        "missing-capabilities": REASON_MISSING_CAPABILITIES,
        "unknown-capability": REASON_UNSUPPORTED_CAPABILITY,
        "malformed-api": REASON_INVALID_API_VERSION,
    }
    assert all(record.category == CATEGORY_QUARANTINED for record in report.quarantined)
    assert all(
        "credential_reader" not in str(record.to_dict())
        for record in report.quarantined
    )


def test_disabled_plugin_is_reported_without_reading_free_form_fields() -> None:
    secret = "raw-credential-value"
    report = build_quarantine_report(
        [
            {
                "name": "disabled-exporter",
                "disabled": True,
                "api_version": secret,
                "credentials": secret,
                "description": secret,
            }
        ]
    )

    assert [record.name for record in report.disabled] == ["disabled-exporter"]
    assert report.disabled[0].reason == "disabled"
    assert report.disabled[0].api_version is None
    assert secret not in report.to_json()


def test_duplicate_resolution_is_independent_of_input_order() -> None:
    recognizer = _plugin("same-name", capabilities=("recognizer",))
    exporter = _plugin("same-name", capabilities=("exporter",))

    forward = build_quarantine_report([recognizer, exporter])
    reverse = build_quarantine_report([exporter, recognizer])

    assert forward.to_json() == reverse.to_json()
    assert len(forward.available) == 1
    assert len(forward.quarantined) == 1
    assert forward.quarantined[0].reason == REASON_DUPLICATE_NAME


def test_compatibility_aliases_and_normalization_are_supported() -> None:
    report = build_quarantine_report(
        {
            "plugin_id": "legacy-plugin",
            "sdk_version": "1",
            "kind": "language-pack",
        }
    )

    assert len(report.available) == 1
    assert report.available[0].name == "legacy-plugin"
    assert report.available[0].api_version == "1.0.0"
    assert report.available[0].capabilities == ("language_pack",)


def test_malformed_metadata_never_echoes_sensitive_values() -> None:
    secret = "raw-phi-or-token-value"
    report = build_quarantine_report(
        {
            "name": "unsafe name containing spaces",
            "api_version": secret,
            "capabilities": ["recognizer", secret],
            "credentials": secret,
        }
    )

    record = report.quarantined[0]
    assert record.reason == REASON_INVALID_API_VERSION
    assert record.name.startswith("redacted:")
    assert secret not in str(record.to_dict())
    assert secret not in report.to_json()


def test_report_serialization_is_detached_and_categories_are_counted() -> None:
    payload = [_plugin("zeta"), {**_plugin("alpha"), "disabled": True}]
    report = build_quarantine_report(payload)
    serialized = report.to_dict()

    serialized["available"][0]["capabilities"].append("mutated")

    assert report.available[0].capabilities == ("recognizer",)
    assert report.counts == {
        CATEGORY_AVAILABLE: 1,
        CATEGORY_DISABLED: 1,
        CATEGORY_QUARANTINED: 0,
    }
    assert report.as_dict() == report.to_dict()


def test_invalid_record_is_structured_instead_of_raising() -> None:
    report = build_quarantine_report([None, "not-a-mapping"])

    assert len(report.quarantined) == 2
    assert all(
        record.reason == REASON_INVALID_METADATA for record in report.quarantined
    )
