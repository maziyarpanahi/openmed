"""Tests for deterministic offline dependency risk reports."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.risk import (
    RISK_CATEGORIES,
    dependency_risk_report,
    dependency_risk_report_json,
    parse_advisory_snapshot,
    write_dependency_risk_report,
)

_LOCKFILE = """
version = 1
revision = 3

[[package]]
name = "urllib3"
version = "2.0.0"

[[package]]
name = "certifi"
version = "2026.1.1"
"""

_SNAPSHOT = {
    "dependencies": [
        {
            "name": "urllib3",
            "version": "2.0.0",
            "vulns": [
                {
                    "id": "CVE-2026-0001",
                    "severity": "high",
                    "description": "synthetic advisory explanation",
                    "url": "https://example.invalid/synthetic-advisory",
                }
            ],
        },
        {"name": "certifi", "version": "2026.1.1", "vulns": []},
        {
            "name": "not-locked",
            "version": "1.0.0",
            "vulns": [{"id": "GHSA-synthetic-0001"}],
        },
    ]
}


def test_report_correlates_locked_versions_and_emits_safe_package_rows() -> None:
    report = dependency_risk_report(_SNAPSHOT, _LOCKFILE)

    assert report["packages"] == [
        {"name": "certifi", "risk_category": "none", "version": "2026.1.1"},
        {"name": "urllib3", "risk_category": "high", "version": "2.0.0"},
    ]
    assert report["offline"] is True
    assert report["summary"] == {
        "affected_packages": 1,
        "advisory_matches": 1,
        "risk_categories": {
            "critical": 0,
            "high": 1,
            "medium": 0,
            "low": 0,
            "unknown": 0,
            "none": 1,
        },
        "total_packages": 2,
        "unmatched_advisories": 1,
    }


def test_report_is_deterministic_and_omits_advisory_source_fields() -> None:
    first = dependency_risk_report_json(_SNAPSHOT, _LOCKFILE)
    second = dependency_risk_report_json(_SNAPSHOT, _LOCKFILE)

    assert first == second
    assert json.loads(first)["schema_version"] == 1
    assert "synthetic advisory explanation" not in first
    assert "example.invalid" not in first
    assert "CVE-2026-0001" not in first


def test_unknown_severity_is_conservative_and_version_mismatch_is_not_safe() -> None:
    report = dependency_risk_report(
        {
            "dependencies": [
                {
                    "name": "demo-package",
                    "version": "1.0.0",
                    "vulns": [{"id": "PYSEC-2026-0001"}],
                },
                {
                    "name": "stale-package",
                    "version": "1.0.0",
                    "vulns": [{"id": "GHSA-synthetic-0002", "severity": "low"}],
                },
            ]
        },
        {
            "packages": [
                {"name": "demo-package", "version": "1.0.0"},
                {"name": "stale-package", "version": "2.0.0"},
            ]
        },
    )

    rows = {row["name"]: row for row in report["packages"]}
    assert rows["demo-package"]["risk_category"] == "unknown"
    assert rows["stale-package"]["risk_category"] == "unknown"
    assert report["summary"]["advisory_matches"] == 1
    assert report["summary"]["unmatched_advisories"] == 1


def test_severity_aliases_and_cvss_scores_are_normalized() -> None:
    findings = parse_advisory_snapshot(
        {
            "packages": [
                {
                    "name": "synthetic-package",
                    "advisories": [
                        {"id": "A-1", "severity": "moderate"},
                        {"id": "A-2", "severity": 9.8},
                    ],
                }
            ]
        }
    )

    assert [finding.risk_category for finding in findings] == ["medium", "critical"]
    assert RISK_CATEGORIES == ("critical", "high", "medium", "low", "unknown", "none")


def test_versionless_advisory_is_counted_once_for_duplicate_lock_versions() -> None:
    report = dependency_risk_report(
        {"advisories": [{"package": "multi-version", "severity": "low"}]},
        {
            "packages": [
                {"name": "multi-version", "version": "1.0.0"},
                {"name": "multi-version", "version": "2.0.0"},
            ]
        },
    )

    assert report["summary"]["advisory_matches"] == 1
    assert report["summary"]["unmatched_advisories"] == 0
    assert [row["risk_category"] for row in report["packages"]] == ["low", "low"]


def test_unversioned_editable_root_is_not_reported_as_a_dependency() -> None:
    report = dependency_risk_report(
        {"dependencies": []},
        {
            "packages": [
                {"name": "openmed", "source": {"editable": "."}},
                {"name": "demo-package", "version": "1.0.0"},
            ]
        },
    )

    assert [row["name"] for row in report["packages"]] == ["demo-package"]


def test_report_reads_local_files_and_writes_json(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    snapshot = tmp_path / "advisories.json"
    output = tmp_path / "reports" / "dependency-risk.json"
    lockfile.write_text(_LOCKFILE, encoding="utf-8")
    snapshot.write_text(json.dumps(_SNAPSHOT), encoding="utf-8")

    returned = write_dependency_risk_report(snapshot, lockfile, output)

    assert returned == output
    assert json.loads(output.read_text(encoding="utf-8"))["packages"]


def test_report_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline dependency report attempted a socket")

    monkeypatch.setattr(socket, "socket", fail_socket)

    assert dependency_risk_report(_SNAPSHOT, _LOCKFILE)["offline"] is True


def test_malformed_input_errors_do_not_echo_payload_values() -> None:
    secret_detail = "synthetic-sensitive-source-detail"

    with pytest.raises(ValueError) as exc_info:
        parse_advisory_snapshot(
            {"dependencies": secret_detail, "description": secret_detail}
        )

    assert secret_detail not in str(exc_info.value)
