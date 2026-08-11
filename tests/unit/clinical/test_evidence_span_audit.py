"""Focused tests for the privacy-safe evidence-span overlap audit."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    EvidenceSpan,
    OverlapKind,
    audit_evidence_span_overlaps,
    audit_evidence_spans,
)


def _span(
    evidence_id: str,
    start: int,
    end: int,
    *,
    source_id: str = "synthetic-source",
) -> EvidenceSpan:
    return EvidenceSpan(
        source_id=source_id,
        evidence_id=evidence_id,
        start=start,
        end=end,
    )


def test_classifies_exact_nested_partial_and_cross_source_pairs() -> None:
    audit = audit_evidence_spans(
        [
            _span("e-partial-left", 0, 5),
            _span("e-exact-left", 10, 20),
            _span("e-nested", 12, 18),
            _span("e-exact-right", 10, 20),
            _span("e-partial-right", 18, 25),
            _span("e-cross-source", 15, 22, source_id="synthetic-peer"),
        ]
    )

    assert audit.counts == {
        "exact": 1,
        "nested": 2,
        "partial": 2,
        "cross_source": 4,
    }
    assert audit.overlap_count == 9
    assert {overlap.kind for overlap in audit.overlaps} == {
        OverlapKind.EXACT,
        OverlapKind.NESTED,
        OverlapKind.PARTIAL,
        OverlapKind.CROSS_SOURCE,
    }


def test_touching_half_open_ranges_do_not_overlap() -> None:
    audit = audit_evidence_spans([_span("left", 0, 4), _span("right", 4, 9)])

    assert audit.overlaps == ()
    assert audit.counts == {
        "exact": 0,
        "nested": 0,
        "partial": 0,
        "cross_source": 0,
    }


def test_mapping_inputs_ignore_raw_text_and_preserve_opaque_references() -> None:
    audit = audit_evidence_spans(
        [
            {
                "source_id": "synthetic-note",
                "evidence_id": "opaque-a",
                "start": 3,
                "end": 9,
                "text": "SENSITIVE_SYNTHETIC_SURFACE",
            },
            {
                "source": "synthetic-note",
                "id": "opaque-b",
                "start_offset": 5,
                "end_offset": 11,
                "raw_value": "SENSITIVE_SYNTHETIC_VALUE",
            },
        ]
    )

    serialized = audit.to_json()
    assert "SENSITIVE_SYNTHETIC_SURFACE" not in serialized
    assert "SENSITIVE_SYNTHETIC_VALUE" not in serialized
    assert {span.evidence_id for span in audit.spans} == {"opaque-a", "opaque-b"}
    assert audit.overlaps[0].kind is OverlapKind.PARTIAL


def test_input_order_does_not_change_report_or_fingerprint() -> None:
    spans = [
        _span("opaque-z", 20, 28),
        _span("opaque-a", 10, 18),
        _span("opaque-m", 15, 24),
    ]

    first = audit_evidence_spans(spans)
    second = audit_evidence_spans(reversed(spans))

    assert first.to_dict() == second.to_dict()
    assert first.fingerprint.startswith("sha256:")
    assert first.overlaps[0].fingerprint.startswith("sha256:")


def test_audit_retains_all_findings_without_resolution() -> None:
    audit = audit_evidence_span_overlaps([_span("outer", 1, 12), _span("inner", 4, 8)])

    report = audit.to_dict()
    assert report["overlap_count"] == 1
    assert report["overlaps"][0]["kind"] == "nested"
    assert "winner" not in report["overlaps"][0]
    assert "resolution" not in report


def test_invalid_inputs_do_not_echo_identifier_or_text_values() -> None:
    with pytest.raises(ValueError) as exc_info:
        audit_evidence_spans(
            [
                {
                    "source_id": "opaque-source",
                    "evidence_id": "opaque-evidence",
                    "start": 7,
                    "end": 7,
                    "text": "SENSITIVE_SYNTHETIC_SURFACE",
                }
            ]
        )

    message = str(exc_info.value)
    assert "SENSITIVE_SYNTHETIC_SURFACE" not in message
    assert "opaque-evidence" not in message
    assert "end must be greater than start" in message


def test_report_is_json_serializable_with_stable_counts() -> None:
    report = audit_evidence_spans([_span("opaque-a", 0, 2)]).to_dict()

    assert json.loads(json.dumps(report)) == report
    assert report["counts"] == {
        "exact": 0,
        "nested": 0,
        "partial": 0,
        "cross_source": 0,
    }
