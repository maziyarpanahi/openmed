"""Focused tests for the privacy-safe surrogate-map auditor."""

from __future__ import annotations

import json

import pytest

from openmed.risk import (
    FAILURE_CATEGORIES,
    SurrogateAuditInputError,
    audit_surrogate_maps,
)


def _clean_maps() -> dict[str, dict[str, object]]:
    return {
        "child_table": {
            "entries": [
                {"key_hash": "hmac-sha256:subject-a", "surrogate": "SYN-A"},
                {"key_hash": "hmac-sha256:subject-b", "surrogate": "SYN-B"},
            ],
            "cardinality": 2,
        },
        "parent_table": {
            "entries": [
                {"key_hash": "hmac-sha256:subject-b", "surrogate": "SYN-B"},
                {"key_hash": "hmac-sha256:subject-a", "surrogate": "SYN-A"},
            ],
            "cardinality": 2,
        },
    }


def test_clean_audit_is_deterministic_and_counts_only() -> None:
    first = audit_surrogate_maps(
        _clean_maps(),
        [{"parent_table": "parent_table", "child_table": "child_table"}],
    )
    second = audit_surrogate_maps(
        {
            "parent_table": {
                "entries": list(reversed(_clean_maps()["parent_table"]["entries"])),
                "cardinality": 2,
            },
            "child_table": {
                "entries": list(reversed(_clean_maps()["child_table"]["entries"])),
                "cardinality": 2,
            },
        },
        [{"parent": "parent_table", "child": "child_table"}],
    )

    assert first.passed
    assert first.to_dict() == second.to_dict()
    assert first.failure_categories == {category: 0 for category in FAILURE_CATEGORIES}
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()


def test_audit_returns_all_required_failure_categories() -> None:
    report = audit_surrogate_maps(
        {
            "parent_table": {
                "entries": [
                    {"key_hash": "hmac-sha256:subject-a", "surrogate": "SYN-A"},
                    {"key_hash": "hmac-sha256:subject-a", "surrogate": "SYN-A-2"},
                    {"key_hash": "hmac-sha256:subject-b", "surrogate": "SYN-A"},
                ],
                "cardinality": 3,
            },
            "child_table": {
                "entries": [
                    {"key_hash": "hmac-sha256:subject-a", "surrogate": "SYN-A-3"},
                    {
                        "key_hash": "hmac-sha256:missing",
                        "surrogate": "SYN-MISSING",
                    },
                ],
                "cardinality": 2,
            },
        },
        [("parent_table", "child_table")],
    )

    assert not report.passed
    assert report.failure_categories == {
        "cardinality": 1,
        "collision": 2,
        "orphan": 1,
        "cross_table_consistency": 1,
    }
    assert [failure.to_dict() for failure in report.failures] == [
        {"category": "cardinality", "count": 1},
        {"category": "collision", "count": 2},
        {"category": "orphan", "count": 1},
        {"category": "cross_table_consistency", "count": 1},
    ]


def test_implicit_comparison_checks_consistency_but_not_orphans() -> None:
    report = audit_surrogate_maps(
        {
            "parent_table": {"hmac-sha256:subject-a": "SYN-A"},
            "child_table": {
                "hmac-sha256:subject-a": "SYN-DIFFERENT",
                "hmac-sha256:missing": "SYN-MISSING",
            },
        }
    )

    assert report.failure_categories["cross_table_consistency"] == 1
    assert report.failure_categories["orphan"] == 0
    assert report.relationships_checked == 1


def test_reports_and_input_errors_never_include_map_values() -> None:
    key_canary = "synthetic-hash-canary"
    surrogate_canary = "synthetic-surrogate-canary"
    report = audit_surrogate_maps(
        {
            "parent_table": {
                "entries": [
                    {"key_hash": key_canary, "surrogate": surrogate_canary},
                ]
            }
        }
    )
    serialized = json.dumps(report.to_dict(), sort_keys=True) + repr(report)
    assert key_canary not in serialized
    assert surrogate_canary not in serialized

    with pytest.raises(SurrogateAuditInputError) as exc_info:
        audit_surrogate_maps({"parent_table": {"entries": "synthetic-invalid-canary"}})
    assert "synthetic-invalid-canary" not in str(exc_info.value)


def test_parallel_metadata_and_package_alias_are_supported() -> None:
    report = audit_surrogate_maps(
        {
            "name": "synthetic_table",
            "key_hashes": ["hmac-sha256:subject-a", "hmac-sha256:subject-b"],
            "surrogates": ["SYN-A", "SYN-B"],
        },
        expected_cardinality={"synthetic_table": 2},
    )

    assert report.passed
    assert report.checked_maps == 1
    assert report.checked_entries == 2
    assert report.checked_keys == 2
