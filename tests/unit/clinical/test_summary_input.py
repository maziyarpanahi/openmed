"""Focused tests for the guarded summary-input contract."""

from __future__ import annotations

import pytest

from openmed.clinical.summary_input import (
    REJECTION_CATEGORIES,
    SummaryEvidence,
    SummaryInputContract,
    SummaryInputValidationError,
    SummarySourceReference,
    build_summary_input,
    guard_summary_input,
    validate_summary_input,
)

POLICY_FINGERPRINT = "sha256:" + "a" * 64
OTHER_POLICY_FINGERPRINT = "sha256:" + "b" * 64


def _evidence(
    *,
    source_id: str = "source:synthetic-note-001",
    evidence_type: str = "structured_fact",
    policy_fingerprint: str = POLICY_FINGERPRINT,
    review_status: str = "approved",
    fields: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "evidence_type": evidence_type,
        "source_ref": {"source_id": source_id, "start": 2, "end": 8},
        "policy_fingerprint": policy_fingerprint,
        "review_status": review_status,
        "fields": fields or {"category": "problem", "count": 1},
    }


def test_valid_evidence_is_admitted_in_deterministic_order() -> None:
    first = _evidence(source_id="source:synthetic-note-002")
    second = _evidence(source_id="source:synthetic-note-001")

    result = validate_summary_input(
        [first, second], policy_fingerprint=POLICY_FINGERPRINT
    )

    assert result.valid
    assert result.accepted_count == 2
    assert result.rejected_count == 0
    assert [item.source_ref.source_id for item in result.evidence] == [
        "source:synthetic-note-001",
        "source:synthetic-note-002",
    ]
    assert result.to_dict()["rejection_counts"] == {
        category: 0 for category in REJECTION_CATEGORIES
    }


def test_typed_evidence_and_contract_round_trip() -> None:
    evidence = SummaryEvidence(
        evidence_type="structured_measurement",
        source_ref=SummarySourceReference("source:synthetic-measurement-001"),
        policy_fingerprint=POLICY_FINGERPRINT,
        review_status="verified",
        fields={"category": "measurement", "unit": "mg/dL"},
    )
    contract = SummaryInputContract(policy_fingerprint=POLICY_FINGERPRINT)

    result = contract.validate([evidence])

    assert result.valid
    assert result.evidence == (evidence,)
    assert build_summary_input([evidence], policy_fingerprint=POLICY_FINGERPRINT) == (
        evidence,
    )
    assert evidence.to_dict()["review_status"] == "approved"


def test_raw_fields_are_rejected_without_echoing_their_value() -> None:
    raw_value = "synthetic-private-token"
    candidate = _evidence(fields={"text": raw_value})

    result = validate_summary_input(candidate, policy_fingerprint=POLICY_FINGERPRINT)

    assert not result.valid
    assert result.rejection_counts == {"raw_field": 1}
    assert raw_value not in result.to_json()
    with pytest.raises(SummaryInputValidationError) as raised:
        guard_summary_input(candidate, policy_fingerprint=POLICY_FINGERPRINT)
    assert raw_value not in str(raised.value)


def test_untyped_and_unknown_evidence_are_rejected() -> None:
    result = validate_summary_input(
        ["synthetic raw record", _evidence(evidence_type="raw_text")],
        policy_fingerprint=POLICY_FINGERPRINT,
    )

    assert not result.valid
    assert result.rejection_counts == {
        "invalid_container": 1,
        "unknown_evidence_type": 1,
    }


def test_missing_provenance_and_review_fields_have_stable_categories() -> None:
    candidate = {
        "evidence_type": "structured_fact",
        "fields": {"category": "problem"},
    }

    result = validate_summary_input(candidate)

    assert result.rejection_counts == {
        "missing_source_reference": 1,
        "missing_policy_fingerprint": 1,
        "missing_review_status": 1,
    }
    assert result.rejected_count == 1


def test_policy_mismatch_and_unverified_status_are_rejected() -> None:
    candidate = _evidence(
        policy_fingerprint=OTHER_POLICY_FINGERPRINT,
        review_status="pending",
    )

    result = validate_summary_input(
        candidate, expected_policy_fingerprint=POLICY_FINGERPRINT
    )

    assert result.rejection_counts == {
        "policy_fingerprint_mismatch": 1,
        "unverified_review_status": 1,
    }


def test_invalid_source_references_and_non_scalar_fields_fail_closed() -> None:
    invalid_source = _evidence(source_id="source:123456789")
    invalid_source["fields"] = {"category": ["problem"]}

    result = validate_summary_input(invalid_source)

    assert result.rejection_counts == {
        "invalid_source_reference": 1,
        "invalid_safe_field": 1,
    }


def test_require_valid_returns_only_guarded_evidence() -> None:
    evidence = _evidence()

    assert (
        SummaryInputContract(policy_fingerprint=POLICY_FINGERPRINT)
        .require_valid([evidence])[0]
        .source_ref.source_id
        == "source:synthetic-note-001"
    )
