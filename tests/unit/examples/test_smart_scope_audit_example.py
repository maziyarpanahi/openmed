"""Smoke tests for the SMART scope audit example."""

from __future__ import annotations

import json
from typing import Any

from examples import smart_scope_audit as example


def _walk_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_walk_string_values(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(_walk_string_values(item))
        return strings
    return []


def test_smart_scope_audit_example_is_deterministic_and_offline(capsys) -> None:
    first = example.main()
    printed = json.loads(capsys.readouterr().out)
    second = example.build_report()

    assert printed == first == second
    assert first["privacy"] == {
        "offline": True,
        "contains_oauth_tokens": False,
        "contains_endpoints": False,
        "contains_launch_context": False,
        "contains_patient_data": False,
    }
    assert [audit["workflow_id"] for audit in first["audits"]] == [
        "patient-read-summary",
        "patient-read-missing",
        "user-write-overbroad",
        "system-export-mixed",
    ]
    assert {audit["status"] for audit in first["audits"]} == {
        "pass",
        "missing",
        "excessive",
    }


def test_smart_scope_audit_example_contains_no_sensitive_runtime_values() -> None:
    report = example.build_report()
    all_text = "\n".join(_walk_string_values(report)).lower()

    assert "bearer" not in all_text
    assert "access_token" not in all_text
    assert "refresh_token" not in all_text
    assert "http://" not in all_text
    assert "https://" not in all_text
    assert "launch" not in all_text
    assert "patient/" in all_text
    assert "patient/synthetic-patient" not in all_text
