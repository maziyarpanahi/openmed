"""Tests for evidence-bound medication-change summary views."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.summary_medication_changes import (
    SUMMARY_MEDICATION_CHANGES_ADVISORY,
    MedicationChangeRelation,
    render_medication_change_summary,
)


def _evidence(character: str) -> str:
    return f"sha256:{character * 64}"


def test_renders_confirmed_structured_changes_with_evidence() -> None:
    result = render_medication_change_summary(
        (
            MedicationChangeRelation(
                medication="Synthetic A",
                change_type="started",
                assertion_status="confirmed",
                evidence_ids=(_evidence("a"),),
                effective_time="2026-01-01",
            ),
            MedicationChangeRelation(
                medication="Synthetic A",
                change_type="dose_changed",
                assertion_status="confirmed",
                evidence_ids=(_evidence("b"),),
                previous_dose="5 mg",
                new_dose="10 mg",
                effective_time="2026-02-01",
            ),
            MedicationChangeRelation(
                medication="Synthetic A",
                change_type="stopped",
                assertion_status="confirmed",
                evidence_ids=(_evidence("c"),),
                effective_time="2026-03-01",
            ),
        )
    )

    assert [statement.text for statement in result.statements] == [
        "Started Synthetic A.",
        "Changed Synthetic A dose from 5 mg to 10 mg.",
        "Stopped Synthetic A.",
    ]
    assert [statement.evidence_ids for statement in result.statements] == [
        (_evidence("a"),),
        (_evidence("b"),),
        (_evidence("c"),),
    ]
    assert result.review_required is False
    assert result.advisory == SUMMARY_MEDICATION_CHANGES_ADVISORY


@pytest.mark.parametrize("assertion_status", ["possible", "conditional", "refuted"])
def test_non_confirmed_assertions_are_withheld_for_review(
    assertion_status: str,
) -> None:
    result = render_medication_change_summary(
        (
            MedicationChangeRelation(
                medication="Synthetic B",
                change_type="started",
                assertion_status=assertion_status,
                evidence_ids=(_evidence("d"),),
            ),
        )
    )

    assert result.statements == ()
    assert [issue.code for issue in result.issues] == ["assertion_not_confirmed"]
    assert result.review_required is True


def test_incomplete_dose_change_emits_value_free_review_code() -> None:
    result = render_medication_change_summary(
        (
            MedicationChangeRelation(
                medication="Synthetic C",
                change_type="dose_changed",
                assertion_status="confirmed",
                evidence_ids=(),
                previous_dose="patient-specific-sensitive-dose",
                new_dose=None,
            ),
        )
    )

    assert result.statements == ()
    assert {issue.code for issue in result.issues} == {
        "incomplete_dose_change",
        "missing_evidence",
    }
    encoded = result.to_json()
    assert "Synthetic C" not in encoded
    assert "patient-specific-sensitive-dose" not in encoded


def test_conflicting_records_are_withheld_instead_of_inferring_regimen() -> None:
    result = render_medication_change_summary(
        (
            MedicationChangeRelation(
                medication="Synthetic D",
                change_type="started",
                assertion_status="confirmed",
                evidence_ids=(_evidence("e"),),
                effective_time="2026-04-01",
            ),
            MedicationChangeRelation(
                medication="Synthetic D",
                change_type="stopped",
                assertion_status="confirmed",
                evidence_ids=(_evidence("f"),),
                effective_time="2026-04-01",
            ),
        )
    )

    assert result.statements == ()
    assert result.issues[0].to_dict() == {
        "code": "conflicting_records",
        "record_indexes": [0, 1],
    }
    assert "current_regimen" not in result.to_dict()


def test_same_change_merges_multiple_evidence_identifiers() -> None:
    relations = tuple(
        MedicationChangeRelation(
            medication="Synthetic E",
            change_type="started",
            assertion_status="confirmed",
            evidence_ids=(identifier,),
            effective_time="2026-05-01",
        )
        for identifier in (_evidence("1"), _evidence("2"))
    )

    result = render_medication_change_summary(relations)

    assert len(result.statements) == 1
    assert result.statements[0].evidence_ids == (_evidence("1"), _evidence("2"))


def test_missing_and_invalid_fields_are_reviewed_without_echoing_values() -> None:
    result = render_medication_change_summary(
        (
            MedicationChangeRelation(
                medication="private-medication-value",
                change_type=None,
                assertion_status="unexpected-private-status",
                evidence_ids=("private-evidence",),
                effective_time="private-time",
            ),
        )
    )

    assert {issue.code for issue in result.issues} == {
        "invalid_effective_time",
        "invalid_evidence_identifier",
        "missing_change_type",
        "unsupported_assertion_status",
    }
    encoded = result.to_json()
    assert "private-medication-value" not in encoded
    assert "unexpected-private-status" not in encoded
    assert "private-evidence" not in encoded
    assert "private-time" not in encoded


def test_renderer_is_deterministic_and_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    relation = MedicationChangeRelation(
        medication="Synthetic F",
        change_type="started",
        assertion_status="confirmed",
        evidence_ids=(_evidence("3"),),
    )

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("socket.socket", fail_network)
    first = render_medication_change_summary((relation,)).to_json()
    second = render_medication_change_summary((relation,)).to_json()

    assert first == second
    assert json.loads(first)["statement_count"] == 1
