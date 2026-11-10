from __future__ import annotations

import json

import pytest

from openmed.clinical.review_override_reasons import (
    REVIEW_OVERRIDE_REASON_SCHEMA_VERSION,
    ReviewOverride,
    ReviewOverrideReasonCode,
    aggregate_review_overrides,
    create_review_override,
)


def test_reason_codes_are_versioned_and_complete() -> None:
    assert REVIEW_OVERRIDE_REASON_SCHEMA_VERSION == 1
    assert [code.value for code in ReviewOverrideReasonCode] == [
        "accept",
        "correct",
        "reject",
        "defer",
        "insufficient_evidence",
    ]


def test_local_note_is_excluded_from_safe_surfaces() -> None:
    sensitive_note = "local-only synthetic patient detail"
    override = create_review_override("correct", local_note=sensitive_note)

    assert override.to_local_dict()["local_note"] == sensitive_note
    assert "local_note" not in override.to_dict()
    assert "local_note" not in override.to_telemetry_dict()
    assert "local_note" not in override.to_aggregate_dict()

    exposed = " ".join(
        (
            repr(override),
            json.dumps(override.to_dict(), sort_keys=True),
            json.dumps(override.to_telemetry_dict(), sort_keys=True),
            json.dumps(override.to_aggregate_dict(), sort_keys=True),
        )
    )
    assert sensitive_note not in exposed


def test_explicit_local_round_trip_preserves_note() -> None:
    original = ReviewOverride(
        ReviewOverrideReasonCode.DEFER,
        local_note="synthetic local reviewer note",
    )

    assert ReviewOverride.from_local_dict(original.to_local_dict()) == original


def test_aggregate_is_deterministic_and_never_reads_notes() -> None:
    class NoteThatMustNotBeRead(str):
        def __str__(self) -> str:
            raise AssertionError("aggregate read local note content")

    overrides = [
        ReviewOverride("reject", local_note=NoteThatMustNotBeRead("private")),
        ReviewOverride("accept"),
        ReviewOverride("reject"),
        {"reason_code": "insufficient_evidence", "local_note": "ignored"},
    ]

    report = aggregate_review_overrides(overrides)

    assert report.to_dict() == {
        "schema_version": 1,
        "total": 4,
        "counts": {
            "accept": 1,
            "correct": 0,
            "reject": 2,
            "defer": 0,
            "insufficient_evidence": 1,
        },
    }
    assert report.to_dict() == aggregate_review_overrides(reversed(overrides)).to_dict()
    assert "private" not in json.dumps(report.to_dict(), sort_keys=True)


def test_invalid_values_fail_without_echoing_input() -> None:
    sensitive_value = "synthetic-patient-name"

    with pytest.raises(ValueError) as error:
        ReviewOverride(sensitive_value)
    assert sensitive_value not in str(error.value)

    with pytest.raises(ValueError, match="schema version"):
        ReviewOverride("accept", schema_version=2)

    with pytest.raises(TypeError, match="local_note"):
        ReviewOverride("accept", local_note=object())
