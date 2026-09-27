"""Focused tests for deterministic, privacy-safe summary fact coverage."""

from __future__ import annotations

import json

import pytest

from openmed.eval.summary_coverage import (
    REASON_DUPLICATE_SOURCE_FACT,
    REASON_INVALID_SOURCE_EVIDENCE,
    REASON_MISSING_SOURCE_EVIDENCE,
    REASON_MISSING_SUMMARY_CITATION,
    REASON_OMITTED_SOURCE_FACT,
    REASON_UNKNOWN_CITATION,
    REASON_UNSUPPORTED_SUMMARY_FACT,
    assert_summary_coverage_gate,
    build_summary_coverage_report,
    compute_summary_fact_coverage,
    summary_coverage_metadata,
)


def test_coverage_counts_citations_omissions_and_unsupported_facts() -> None:
    source_facts = [
        {
            "id": "source-a",
            "value": "synthetic-source-alpha",
            "evidence": {"start": 0, "end": 5},
        },
        {
            "id": "source-b",
            "value": "synthetic-source-beta",
            "evidence": {"start": 8, "end": 12},
        },
    ]
    summary_citations = [
        {
            "claim": "synthetic-summary-grounded",
            "citations": [{"source_fact_id": "source-a"}],
        },
        {
            "claim": "synthetic-summary-unsupported",
            "citations": [{"source_fact_id": "not-in-source"}],
        },
    ]

    result = compute_summary_fact_coverage(source_facts, summary_citations)

    assert result.recall == 0.5
    assert result.source_fact_count == 2
    assert result.cited_fact_count == 1
    assert result.omission_count == 1
    assert result.unsupported_fact_count == 1
    assert result.invalid_citation_count == 1
    assert result.source_evidence_available is True
    assert result.fail_closed is False
    assert result.passed is False
    assert set(result.failure_reasons) == {
        REASON_OMITTED_SOURCE_FACT,
        REASON_UNKNOWN_CITATION,
        REASON_UNSUPPORTED_SUMMARY_FACT,
    }


def test_identifier_and_span_citations_produce_a_deterministic_pass() -> None:
    source_facts = [
        {"fact_id": "fact-a", "evidence": {"start": 10, "end": 16}},
        {"fact_id": "fact-b", "evidence": {"start": 20, "end": 26}},
    ]
    summary_citations = [
        {"citations": [{"start": 10, "end": 16}]},
        {"citations": [{"source_fact_id": "fact-b"}]},
    ]

    first = build_summary_coverage_report(source_facts, summary_citations)
    second = build_summary_coverage_report(
        list(reversed(source_facts)),
        list(reversed(summary_citations)),
    )

    assert first.coverage.recall == 1.0
    assert first.coverage.omissions == 0
    assert first.coverage.unsupported_facts == 0
    assert first.passed is True
    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()


def test_missing_source_evidence_fails_closed_without_a_vacuous_score() -> None:
    result = compute_summary_fact_coverage(
        source_facts=None,
        summary_citations=[{"source_fact_id": "fact-a"}],
    )

    assert result.recall == 0.0
    assert result.source_fact_count == 0
    assert result.source_evidence_available is False
    assert result.fail_closed is True
    assert result.unsupported_fact_count == 1
    assert result.failure_reasons == (
        REASON_MISSING_SOURCE_EVIDENCE,
        REASON_UNSUPPORTED_SUMMARY_FACT,
    )

    with pytest.raises(AssertionError, match=REASON_MISSING_SOURCE_EVIDENCE):
        assert_summary_coverage_gate(None, [{"source_fact_id": "fact-a"}])


def test_missing_citations_are_unsupported_and_source_facts_are_omitted() -> None:
    result = compute_summary_fact_coverage(
        [{"id": "fact-a"}, {"id": "fact-b"}],
        [{"claim": "synthetic-summary"}, {"citations": []}],
    )

    assert result.recall == 0.0
    assert result.omission_count == 2
    assert result.unsupported_fact_count == 2
    assert result.missing_citation_count == 2
    assert set(result.failure_reasons) == {
        REASON_MISSING_SUMMARY_CITATION,
        REASON_OMITTED_SOURCE_FACT,
        REASON_UNSUPPORTED_SUMMARY_FACT,
    }


def test_malformed_source_records_fail_closed_without_echoing_values() -> None:
    result = compute_summary_fact_coverage(
        [
            {"id": "fact-a"},
            {"id": "fact-a", "value": "duplicate-sensitive-value"},
            {"value": "orphan-sensitive-value"},
        ],
        [{"claim": "summary-sensitive-value", "fact_id": "fact-a"}],
    )

    serialized = json.dumps(result.to_dict(), ensure_ascii=False)
    assert result.fail_closed is True
    assert result.invalid_source_fact_count == 1
    assert result.duplicate_source_fact_count == 1
    assert result.failure_reasons == (
        REASON_DUPLICATE_SOURCE_FACT,
        REASON_INVALID_SOURCE_EVIDENCE,
    )
    assert "duplicate-sensitive-value" not in serialized
    assert "orphan-sensitive-value" not in serialized
    assert "summary-sensitive-value" not in serialized


def test_explicit_empty_source_evidence_fails_closed() -> None:
    result = compute_summary_fact_coverage(
        [{"id": "fact-a"}],
        [{"source_fact_id": "fact-a"}],
        source_evidence=[],
    )

    assert result.recall == 0.0
    assert result.source_evidence_available is False
    assert result.fail_closed is True
    assert result.failure_reasons == (REASON_MISSING_SOURCE_EVIDENCE,)


def test_report_and_metadata_are_raw_value_free() -> None:
    report = build_summary_coverage_report(
        [{"id": "fact-a", "value": "report-sensitive-source"}],
        [{"claim": "report-sensitive-summary", "source_fact_id": "fact-a"}],
    )
    serialized = report.to_json() + report.to_markdown()

    assert "report-sensitive-source" not in serialized
    assert "report-sensitive-summary" not in serialized
    assert report.to_dict()["metrics"]["summary_fact_coverage"]["passed"] is True
    assert summary_coverage_metadata() == {
        "suite": "summary_fact_coverage",
        "schema_version": 1,
        "synthetic": True,
        "matching": "opaque_fact_ids_or_source_offsets",
        "metrics": ["recall", "unsupported_fact_count", "omission_count"],
        "fail_closed_on_missing_source_evidence": True,
    }
