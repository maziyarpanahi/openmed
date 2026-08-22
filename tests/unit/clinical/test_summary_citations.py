"""Tests for deterministic, privacy-safe summary citation consistency."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    REASON_DUPLICATE_CITATION,
    REASON_DUPLICATE_SOURCE_EVIDENCE,
    REASON_INVALID_SOURCE_EVIDENCE,
    REASON_MISSING_CITATION,
    REASON_MISSING_SOURCE_EVIDENCE,
    REASON_MISSING_SOURCE_RECORD,
    REASON_UNAVAILABLE_SPAN,
    REASON_UNSUPPORTED_CLAIM,
    assert_summary_citation_gate,
    build_summary_citation_report,
    compute_summary_citation_metrics,
    summary_citation_metadata,
)


def test_valid_identifier_and_span_citations_are_complete_and_order_stable():
    evidence = [
        {
            "evidence_id": "synthetic-evidence-a",
            "span": {"start": 0, "end": 8},
            "value": "synthetic source alpha",
        },
        {
            "id": "synthetic-evidence-b",
            "start": 12,
            "end": 20,
            "value": "synthetic source beta",
        },
    ]
    claims = [
        {
            "claim": "synthetic claim alpha",
            "citations": [{"evidence_id": "synthetic-evidence-a"}],
        },
        {
            "text": "synthetic claim beta",
            "citations": [{"start": 12, "end": 20}],
        },
    ]

    first = compute_summary_citation_metrics(claims, evidence)
    second = compute_summary_citation_metrics(
        list(reversed(claims)), list(reversed(evidence))
    )

    assert first.to_json() == second.to_json()
    assert first.coverage == 1.0
    assert first.abstention_rate == 0.0
    assert first.claim_count == 2
    assert first.supported_claim_count == 2
    assert first.valid_citation_count == 2
    assert first.referenced_evidence_count == 2
    assert first.evidence_coverage == 1.0
    assert first.source_evidence_available is True
    assert first.passed is True
    assert first.failure_reasons == ()


def test_missing_records_spans_and_duplicate_citations_fail_review():
    evidence = [{"id": "synthetic-evidence-a", "start": 0, "end": 8}]
    claims = [
        {
            "citations": [
                {"evidence_id": "synthetic-evidence-a"},
                {"evidence_id": "synthetic-evidence-a"},
            ]
        },
        {"citations": [{"evidence_id": "unavailable-evidence"}]},
        {"claim": "synthetic claim without a citation"},
        {"citations": [{"start": 40, "end": 48}]},
    ]

    result = compute_summary_citation_metrics(claims, evidence)

    assert result.claim_count == 4
    assert result.supported_claim_count == 1
    assert result.unsupported_claim_count == 3
    assert result.abstention_count == 3
    assert result.coverage == 0.25
    assert result.abstention_rate == 0.75
    assert result.citation_count == 4
    assert result.valid_citation_count == 2
    assert result.invalid_citation_count == 2
    assert result.duplicate_citation_count == 1
    assert result.missing_citation_count == 1
    assert result.missing_source_record_count == 1
    assert result.unavailable_span_count == 1
    assert result.passed is False
    assert set(result.failure_reasons) == {
        REASON_DUPLICATE_CITATION,
        "invalid_citation",
        REASON_MISSING_CITATION,
        REASON_MISSING_SOURCE_RECORD,
        REASON_UNAVAILABLE_SPAN,
        REASON_UNSUPPORTED_CLAIM,
    }


def test_malformed_source_evidence_fails_closed_without_rendering_raw_values():
    evidence = [
        {"id": "synthetic-evidence-a"},
        {"id": "synthetic-evidence-a", "value": "duplicate-sensitive-value"},
        {"value": "orphan-sensitive-value"},
    ]
    claims = [
        {
            "claim": "summary-sensitive-value",
            "citations": [{"evidence_id": "synthetic-evidence-a"}],
        }
    ]

    result = compute_summary_citation_metrics(claims, evidence)
    serialized = json.dumps(result.to_dict(), ensure_ascii=False)

    assert result.fail_closed is True
    assert result.coverage == 0.0
    assert result.invalid_evidence_record_count == 1
    assert result.duplicate_evidence_record_count == 1
    assert result.passed is False
    assert set(result.failure_reasons) == {
        REASON_DUPLICATE_SOURCE_EVIDENCE,
        REASON_INVALID_SOURCE_EVIDENCE,
    }
    assert "duplicate-sensitive-value" not in serialized
    assert "orphan-sensitive-value" not in serialized
    assert "summary-sensitive-value" not in serialized


def test_missing_evidence_fails_closed_and_gate_error_is_aggregate_only():
    result = compute_summary_citation_metrics(
        [{"claim": "synthetic unsupported claim", "citations": []}], None
    )

    assert result.coverage == 0.0
    assert result.abstention_rate == 1.0
    assert result.source_evidence_available is False
    assert result.fail_closed is True
    assert result.failure_reasons == (
        REASON_MISSING_CITATION,
        REASON_MISSING_SOURCE_EVIDENCE,
        REASON_UNSUPPORTED_CLAIM,
    )

    with pytest.raises(AssertionError) as error:
        assert_summary_citation_gate(
            [{"claim": "synthetic unsupported claim", "citations": []}],
            None,
        )
    assert REASON_MISSING_SOURCE_EVIDENCE in str(error.value)
    assert "synthetic unsupported claim" not in str(error.value)


def test_report_and_metadata_are_deterministic_and_raw_value_free():
    report = build_summary_citation_report(
        [
            {
                "claim": "report-sensitive-summary",
                "citations": [{"evidence_id": "report-evidence"}],
            }
        ],
        [{"id": "report-evidence", "value": "report-sensitive-source"}],
    )
    serialized = report.to_json() + report.to_markdown()

    assert report.passed is True
    assert report["metrics"]["coverage"] == 1.0
    assert "report-sensitive-summary" not in serialized
    assert "report-sensitive-source" not in serialized
    assert "report-evidence" not in serialized
    assert summary_citation_metadata() == {
        "suite": "summary_citation_consistency",
        "schema_version": 1,
        "synthetic": True,
        "matching": "opaque_evidence_ids_or_exact_source_spans",
        "metrics": ["coverage", "abstention_rate", "duplicate_citation_count"],
        "fail_closed_on_missing_source_evidence": True,
    }
