"""Tests for the clinical decision-support guardrail layer (OM-802).

These tests assert the CDS transparency invariants that keep OpenMed out of the
autonomous medical-device category: every guarded clinical suggestion carries a
mandatory disclaimer, is traceable to at least one source span, is never an
autonomous decision, and round-trips through serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from openmed.clinical import (
    CLINICAL_DECISION_SUPPORT_DISCLAIMER,
    GuardedSuggestion,
    GuardrailValidationError,
    SourceSpan,
    build_guarded_suggestion,
    guarded_suggestion,
    validate_guarded_suggestion,
)


@dataclass(frozen=True)
class _SpanLike:
    """Minimal span-like object (e.g. an EntitySpan) for coercion tests."""

    start: int
    end: int
    label: str | None = None
    text: str | None = None


def _synthetic_span() -> SourceSpan:
    # Synthetic offsets/excerpt only -- no real PHI.
    return SourceSpan(start=12, end=27, label="lab", text="creatinine 1.8")


def test_disclaimer_always_present_and_non_empty():
    guarded = build_guarded_suggestion(
        "Consider evaluating renal function trend",
        [_synthetic_span()],
        0.81,
    )
    assert guarded.disclaimer == CLINICAL_DECISION_SUPPORT_DISCLAIMER
    assert guarded.disclaimer.strip()
    assert "not" in guarded.disclaimer.lower()
    assert "clinician" in guarded.disclaimer.lower()
    assert guarded.to_dict()["disclaimer"] == CLINICAL_DECISION_SUPPORT_DISCLAIMER


def test_blank_disclaimer_is_rejected():
    with pytest.raises(GuardrailValidationError):
        build_guarded_suggestion(
            "x",
            [_synthetic_span()],
            0.5,
            disclaimer="   ",
        )


def test_traceability_required_untraced_suggestion_is_rejected():
    with pytest.raises(GuardrailValidationError):
        build_guarded_suggestion("untraceable suggestion", [], 0.5)

    with pytest.raises(GuardrailValidationError):
        GuardedSuggestion(
            suggestion="untraceable suggestion",
            source_spans=(),
            confidence=0.5,
        )


def test_validate_rejects_mapping_without_source_spans():
    payload = {
        "suggestion": "no spans here",
        "confidence": 0.5,
        "disclaimer": CLINICAL_DECISION_SUPPORT_DISCLAIMER,
        "source_spans": [],
    }
    with pytest.raises(GuardrailValidationError):
        validate_guarded_suggestion(payload)


def test_autonomous_decision_flag_is_always_false():
    guarded = build_guarded_suggestion("suggestion", [_synthetic_span()], 0.6)
    assert guarded.autonomous_decision is False
    assert guarded.requires_clinician_review is True
    assert guarded.to_dict()["autonomous_decision"] is False
    assert guarded.to_dict()["requires_clinician_review"] is True


def test_constructing_an_autonomous_decision_is_rejected():
    with pytest.raises(GuardrailValidationError):
        GuardedSuggestion(
            suggestion="autonomous action",
            source_spans=(_synthetic_span(),),
            confidence=0.9,
            autonomous_decision=True,
        )


def test_validate_rejects_autonomous_decision_mapping():
    payload = build_guarded_suggestion("suggestion", [_synthetic_span()], 0.6).to_dict()
    payload["autonomous_decision"] = True
    with pytest.raises(GuardrailValidationError):
        validate_guarded_suggestion(payload)


def test_requires_clinician_review_cannot_be_disabled():
    with pytest.raises(GuardrailValidationError):
        GuardedSuggestion(
            suggestion="suggestion",
            source_spans=(_synthetic_span(),),
            confidence=0.6,
            requires_clinician_review=False,
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_out_of_range_confidence_is_rejected(confidence):
    with pytest.raises(GuardrailValidationError):
        build_guarded_suggestion("suggestion", [_synthetic_span()], confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_boundary_confidence_is_accepted(confidence):
    guarded = build_guarded_suggestion("suggestion", [_synthetic_span()], confidence)
    assert guarded.confidence == confidence


def test_synthetic_suggestion_round_trips_through_serialization():
    guarded = build_guarded_suggestion(
        {"advice": "reassess anticoagulation dose", "priority": "routine"},
        [
            SourceSpan(start=0, end=8, label="medication", text="warfarin"),
            _SpanLike(start=30, end=40, label="lab", text="INR 3.4"),
        ],
        0.77,
        provenance={"producer": "medication_change", "rule_id": "mc-014"},
    )

    encoded = json.dumps(guarded.to_dict())
    restored = GuardedSuggestion.from_dict(json.loads(encoded))

    assert restored.to_dict() == guarded.to_dict()
    assert restored.suggestion == guarded.suggestion
    assert restored.confidence == guarded.confidence
    assert restored.disclaimer == CLINICAL_DECISION_SUPPORT_DISCLAIMER
    assert restored.autonomous_decision is False
    assert restored.requires_clinician_review is True
    assert [span.offset_key() for span in restored.source_spans] == [(0, 8), (30, 40)]
    assert restored.provenance == {
        "producer": "medication_change",
        "rule_id": "mc-014",
    }


def test_source_span_coercion_from_mapping_and_object():
    guarded = build_guarded_suggestion(
        "suggestion",
        [
            {"start": 5, "end": 9, "label": "problem", "text": "sepsis"},
            _SpanLike(start=11, end=15, label="vital", text="temp"),
        ],
        0.5,
    )
    assert all(isinstance(span, SourceSpan) for span in guarded.source_spans)
    assert guarded.source_spans[0].label == "problem"
    assert guarded.source_spans[1].text == "temp"


def test_invalid_source_span_offsets_are_rejected():
    with pytest.raises(ValueError):
        SourceSpan(start=10, end=4)
    with pytest.raises(ValueError):
        SourceSpan(start=-1, end=2)


def test_decorator_wraps_tuple_producer_into_validated_envelope():
    @guarded_suggestion
    def suggest_monitoring():
        return (
            "increase glucose monitoring frequency",
            [{"start": 3, "end": 7, "label": "lab", "text": "220 mg/dL"}],
            0.72,
        )

    result = suggest_monitoring()
    assert isinstance(result, GuardedSuggestion)
    # Decorator output passes the hard validation gate.
    validate_guarded_suggestion(result)
    assert result.disclaimer == CLINICAL_DECISION_SUPPORT_DISCLAIMER
    assert result.autonomous_decision is False
    assert result.source_spans[0].offset_key() == (3, 7)
    assert result.confidence == pytest.approx(0.72)


def test_decorator_accepts_four_tuple_with_provenance():
    @guarded_suggestion
    def suggest_with_provenance():
        return (
            "review potassium",
            [_synthetic_span()],
            0.6,
            {"producer": "lab_trend"},
        )

    result = suggest_with_provenance()
    assert result.provenance == {"producer": "lab_trend"}


def test_decorator_passes_through_prebuilt_guarded_suggestion():
    prebuilt = build_guarded_suggestion("suggestion", [_synthetic_span()], 0.5)

    @guarded_suggestion
    def producer():
        return prebuilt

    assert producer().to_dict() == prebuilt.to_dict()


def test_decorator_rejects_untraced_producer_output():
    @guarded_suggestion
    def bad_producer():
        return ("no spans", [], 0.5)

    with pytest.raises(GuardrailValidationError):
        bad_producer()


def test_decorator_rejects_unsupported_return_shape():
    @guarded_suggestion
    def bad_shape():
        return 42

    with pytest.raises(GuardrailValidationError):
        bad_shape()


def test_validate_rejects_non_envelope_inputs():
    with pytest.raises(GuardrailValidationError):
        validate_guarded_suggestion("not an envelope")
    with pytest.raises(GuardrailValidationError):
        validate_guarded_suggestion(None)


def test_to_dict_is_deterministic_and_phi_offset_based():
    guarded = build_guarded_suggestion(
        "suggestion",
        [SourceSpan(start=1, end=2, label="problem", text="dm")],
        0.5,
    )
    first = guarded.to_dict()
    second = guarded.to_dict()
    assert first == second
    # Envelope keys are stable and offset/confidence-forward.
    assert set(first) == {
        "schema_version",
        "suggestion",
        "confidence",
        "disclaimer",
        "requires_clinician_review",
        "autonomous_decision",
        "source_spans",
        "provenance",
    }
    assert first["source_spans"][0]["start"] == 1
    assert first["source_spans"][0]["end"] == 2
