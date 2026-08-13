"""Focused tests for deterministic, privacy-safe budget migrations."""

from __future__ import annotations

import copy
import json

import pytest

from openmed.risk import (
    BudgetMigrationError,
    BudgetMigrationRejected,
    DPGenerationBudgetAccountant,
    EpsilonPolicy,
    compare_budget_migration,
    enforce_budget_migration,
)


def _entry(
    release_id: str,
    *,
    spent_epsilon: float = 0.5,
    spent_delta: float = 1e-6,
    max_epsilon: float = 2.0,
    max_delta: float = 1e-5,
    composition: str = "basic",
    policy_fingerprint: str = "policy-v1",
    sequence: int = 1,
) -> dict[str, object]:
    return {
        "release_id": release_id,
        "spent_epsilon": spent_epsilon,
        "spent_delta": spent_delta,
        "max_epsilon": max_epsilon,
        "max_delta": max_delta,
        "composition": composition,
        "policy_fingerprint": policy_fingerprint,
        "sequence": sequence,
    }


def _snapshot(*entries: dict[str, object]) -> dict[str, object]:
    return {"schema_version": "openmed.dp_budget_ledger.v1", "entries": list(entries)}


def test_monotonic_append_is_deterministic_and_aggregate_only() -> None:
    before = _snapshot(_entry("release-a"))
    after = _snapshot(
        _entry("release-b", spent_epsilon=0.25, sequence=2),
        _entry("release-a", spent_epsilon=0.75, spent_delta=2e-6),
    )
    after["source_value"] = "Synthetic Patient"

    first = compare_budget_migration(before, after)
    second = compare_budget_migration(
        before,
        _snapshot(
            _entry("release-a", spent_epsilon=0.75, spent_delta=2e-6),
            _entry("release-b", spent_epsilon=0.25, sequence=2),
        ),
    )

    assert first.passed is True
    assert first.valid is True
    assert first.counts == {
        "added": 1,
        "after_entries": 2,
        "before_entries": 1,
        "changed": 1,
        "issues": 0,
        "removed": 0,
        "unchanged": 0,
    }
    assert first.after_totals["spent_epsilon"] == pytest.approx(1.0)
    assert first.to_dict() == second.to_dict()
    encoded = first.to_json()
    assert json.loads(encoded) == first.to_dict()
    assert "Synthetic Patient" not in encoded
    assert "release-a" in encoded
    assert "release-b" in encoded


def test_spend_limit_composition_and_fingerprint_regressions_are_blocked() -> None:
    before = _snapshot(_entry("release-a"))
    after = _snapshot(
        _entry(
            "release-a",
            spent_epsilon=0.25,
            spent_delta=5e-7,
            max_epsilon=3.0,
            max_delta=2e-5,
            composition="advanced",
            policy_fingerprint="policy-v2",
        )
    )

    report = compare_budget_migration(before, after)

    assert report.passed is False
    assert {issue.kind for issue in report.issues} == {
        "budget_limit_increased",
        "composition_changed",
        "policy_fingerprint_changed",
        "spent_budget_decreased",
    }
    assert {issue.field for issue in report.issues} == {
        "composition",
        "max_delta",
        "max_epsilon",
        "policy_fingerprint",
        "spent_delta",
        "spent_epsilon",
    }
    with pytest.raises(BudgetMigrationRejected) as exc_info:
        enforce_budget_migration(before, after)
    assert exc_info.value.report == report
    assert "Synthetic Patient" not in str(exc_info.value)


def test_release_removal_and_sequence_reuse_are_reported() -> None:
    before = _snapshot(_entry("release-a", sequence=1))
    after = _snapshot(_entry("release-b", sequence=1))

    report = compare_budget_migration(before, after)

    assert report.passed is False
    assert report.counts["added"] == 1
    assert report.counts["removed"] == 1
    assert {issue.kind for issue in report.issues} == {
        "missing_release",
        "reused_release_sequence",
    }


def test_missing_or_sensitive_entries_fail_without_echoing_values() -> None:
    malformed = {
        "entries": [
            {
                "release_id": "release-a",
                "spent_epsilon": 0.5,
                "spent_delta": 1e-6,
                "max_epsilon": 2.0,
                "max_delta": 1e-5,
                "composition": "basic",
                "source_value": "Synthetic Patient",
            }
        ]
    }
    with pytest.raises(BudgetMigrationError) as missing:
        compare_budget_migration(malformed, malformed)
    assert "Synthetic Patient" not in str(missing.value)

    invalid_number = _snapshot(_entry("release-a"))
    invalid_number["entries"][0]["spent_epsilon"] = "Synthetic Patient"  # type: ignore[index]
    with pytest.raises(BudgetMigrationError) as unsafe_number:
        compare_budget_migration(invalid_number, invalid_number)
    assert "Synthetic Patient" not in str(unsafe_number.value)
    assert unsafe_number.value.__cause__ is None

    sensitive = _snapshot(_entry("123-45-6789"))
    with pytest.raises(BudgetMigrationError) as unsafe:
        compare_budget_migration(sensitive, sensitive)
    assert "123-45-6789" not in str(unsafe.value)

    duplicate = _snapshot(_entry("release-a"), _entry("release-a", sequence=2))
    with pytest.raises(BudgetMigrationError):
        compare_budget_migration(duplicate, duplicate)


def test_accountant_compositions_are_verified_with_derived_policy_fingerprints() -> (
    None
):
    policy = EpsilonPolicy(
        scope="release-a",
        max_epsilon=2.0,
        max_delta=1e-5,
        composition="basic",
    )
    accountant = DPGenerationBudgetAccountant({"release-a": policy})
    accountant.guard_generation(0.5, 1e-6, "release-a")
    payload = accountant.to_dict()

    report = compare_budget_migration(payload, copy.deepcopy(payload))

    assert report.passed is True
    assert report.release_identifiers == ("release-a",)
    assert len(report.policy_fingerprints) == 1
    assert report.policy_fingerprints[0].startswith("sha256:")
