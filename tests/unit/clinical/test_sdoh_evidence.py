"""Synthetic offline tests for the PHI-safe SDOH evidence contract."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.sdoh_evidence import (
    ASSERTION_REFUSED,
    ASSERTION_UNKNOWN,
    EVIDENCE_REFUSED,
    NEEDS_REVIEW,
    SOCIAL_HISTORY,
    SDOHEvidence,
    SDOHSourceSpan,
    build_sdoh_evidence_report,
    evidence_from_sdoh_finding,
    validate_sdoh_evidence,
)


def test_evidence_round_trip_contains_offsets_but_no_source_value() -> None:
    evidence = SDOHEvidence(
        evidence_type="self_report",
        assertion=ASSERTION_UNKNOWN,
        source_section=SOCIAL_HISTORY,
        source_span=(12, 27),
        review_status=NEEDS_REVIEW,
        determinant="housing_insecurity",
    )

    payload = evidence.to_dict()

    assert payload == {
        "evidence_type": "self_report",
        "assertion": "unknown",
        "source_section": "social_history",
        "source_span": [12, 27],
        "review_status": "needs_review",
        "determinant": "housing_insecurity",
    }
    assert "synthetic source value" not in json.dumps(payload)
    assert validate_sdoh_evidence(payload) == evidence
    assert SDOHEvidence.from_json(evidence.to_json()) == evidence


def test_unknown_and_refused_are_explicit_non_affirmative_states() -> None:
    refused = SDOHEvidence(
        evidence_type=EVIDENCE_REFUSED,
        assertion=ASSERTION_REFUSED,
        source_section=SOCIAL_HISTORY,
        source_span=SDOHSourceSpan(4, 9),
    )

    assert refused.assertion == "refused"
    assert refused.review_status == NEEDS_REVIEW
    assert refused.to_dict()["assertion"] == "refused"
    assert refused.to_dict()["source_span"] == [4, 9]


@pytest.mark.parametrize(
    "source_span",
    [None, (2, 2), (-1, 4), ("2", 4), (True, 4)],
)
def test_source_span_is_required_and_non_empty(source_span) -> None:
    with pytest.raises((TypeError, ValueError), match="source span"):
        SDOHEvidence(
            evidence_type="inferred",
            assertion="unknown",
            source_section="social_history",
            source_span=source_span,
        )


def test_mapping_requires_explicit_assertion_and_source_span() -> None:
    with pytest.raises(ValueError, match="explicit assertion"):
        SDOHEvidence.from_dict(
            {
                "evidence_type": "self_report",
                "source_section": "social_history",
                "source_span": [0, 8],
            }
        )

    with pytest.raises(ValueError, match="source span"):
        SDOHEvidence.from_dict(
            {
                "evidence_type": "self_report",
                "assertion": "refused",
                "source_section": "social_history",
            }
        )


def test_report_is_sorted_and_discards_untrusted_raw_fields() -> None:
    report = build_sdoh_evidence_report(
        [
            {
                "evidence_type": "self_report",
                "assertion": "present",
                "source_section": "social_history",
                "source_span": [30, 39],
                "text": "synthetic source value",
            },
            {
                "evidence_type": "structured",
                "assertion": "refused",
                "source_section": "social_history",
                "source_span": [4, 9],
                "value": "synthetic source value",
            },
        ]
    )

    payload = report.to_dict()

    assert [item["source_span"] for item in payload["findings"]] == [
        [4, 9],
        [30, 39],
    ]
    serialized = json.dumps(payload, sort_keys=True)
    assert "synthetic source value" not in serialized
    assert "diagnosis" in payload["disclaimer"]


def test_existing_sdoh_shape_can_be_wrapped_without_copying_value() -> None:
    upstream = {
        "category": "employment",
        "value": "synthetic source value",
        "status": None,
        "span": (7, 18),
    }

    evidence = evidence_from_sdoh_finding(upstream)

    assert evidence.determinant == "employment"
    assert evidence.assertion == ASSERTION_UNKNOWN
    assert evidence.source_span == (7, 18)
    assert "synthetic source value" not in evidence.to_json()


def test_uncontrolled_determinant_values_are_not_serialized() -> None:
    with pytest.raises(ValueError, match="controlled label"):
        SDOHEvidence(
            evidence_type="self_report",
            assertion="unknown",
            source_section="social_history",
            source_span=(0, 8),
            determinant="synthetic source value",
        )
