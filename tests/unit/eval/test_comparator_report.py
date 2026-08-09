"""Focused tests for the counts-only comparator report renderer."""

from __future__ import annotations

import json

from openmed.eval.comparator_report import (
    ComparatorReport,
    build_comparator_report,
    fingerprint_environment,
    render_comparator_report_json,
    render_comparator_report_markdown,
)
from openmed.eval.comparators import (
    STATUS_NOT_AVAILABLE,
    ComparatorMatrixReport,
    ComparatorMatrixRow,
)
from openmed.eval.report import BenchmarkReport

RAW_FIXTURE_VALUE = "Patient Ada Lovelace MRN-493021"


def test_comparator_report_is_counts_only_and_aggregates_failures() -> None:
    environment = {
        "python": "3.12.4",
        "platform": "synthetic",
        "raw": RAW_FIXTURE_VALUE,
    }
    matrix = _matrix(environment)

    report = build_comparator_report(matrix)
    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert isinstance(report, ComparatorReport)
    assert RAW_FIXTURE_VALUE not in serialized
    assert payload["environment_fingerprint"] == fingerprint_environment(environment)
    assert payload["summary"] == {
        "comparator_count": 2,
        "failure_count": 2,
        "failed_count": 0,
        "not_available_count": 1,
        "scored_count": 1,
    }
    assert payload["failure_summary"] == {
        "by_category": {"execution": 1, "not_available": 1},
        "systems_affected": 2,
        "total": 2,
    }
    assert payload["rows"][0]["metric_counts"] == {
        "character_recall": {"denominator": 4, "numerator": 3},
        "exact_span_f1": {
            "false_negatives": 1,
            "false_positives": 0,
            "true_positives": 2,
        },
        "leakage_rate": {
            "leaked_graphemes": 1,
            "total_graphemes": 4,
        },
        "relaxed_span_f1": {
            "false_negatives": 1,
            "false_positives": 0,
            "true_positives": 2,
        },
    }
    assert payload["rows"][1]["failure_count"] == 1
    assert "raw" not in payload


def test_comparator_report_json_and_markdown_are_deterministic() -> None:
    matrix = _matrix({"python": "3.12", "platform": "synthetic"})

    first_json = render_comparator_report_json(matrix)
    second_json = render_comparator_report_json(matrix)
    first_markdown = render_comparator_report_markdown(matrix)
    second_markdown = render_comparator_report_markdown(matrix)

    assert first_json == second_json
    assert first_markdown == second_markdown
    assert "## Metric Definitions" in first_markdown
    assert "## Aggregate Failures" in first_markdown
    assert "Environment fingerprint" in first_markdown
    assert RAW_FIXTURE_VALUE not in first_json
    assert RAW_FIXTURE_VALUE not in first_markdown


def test_comparator_report_accepts_json_ready_mappings() -> None:
    report = build_comparator_report(
        {
            "suite": "mapping-suite",
            "model_name": "OpenMed",
            "device": "cpu",
            "fixture_count": 3,
            "rows": [
                {
                    "system": "mapping-comparator",
                    "status": "scored",
                    "fixture_count": 3,
                    "metrics": {
                        "leakage_rate": 0.0,
                        "character_recall": 1.0,
                        "exact_span_f1": 1.0,
                        "relaxed_span_f1": 1.0,
                    },
                }
            ],
        },
        environment={"python": "3.12", "platform": "synthetic"},
    )

    assert report.fixture_count == 3
    assert report.rows[0].metrics["character_recall"] == 1.0
    assert report.failure_summary["total"] == 0


def _matrix(environment: dict[str, str]) -> ComparatorMatrixReport:
    benchmark = BenchmarkReport(
        suite="synthetic-comparator",
        model_name="OpenMed",
        device="cpu",
        fixture_count=2,
        metrics={
            "leakage": {
                "overall": 0.25,
                "leaked_graphemes": 1,
                "total_graphemes": 4,
            },
            "character_recall": {"rate": 0.75, "numerator": 3, "denominator": 4},
            "exact_span_f1": {
                "f1": 0.8,
                "true_positives": 2,
                "false_positives": 0,
                "false_negatives": 1,
            },
            "relaxed_span_f1": {
                "f1": 0.9,
                "true_positives": 2,
                "false_positives": 0,
                "false_negatives": 1,
            },
            "failures": [{"message": RAW_FIXTURE_VALUE}],
        },
        metadata={"raw_text": RAW_FIXTURE_VALUE},
    )
    return ComparatorMatrixReport(
        suite="synthetic-comparator",
        model_name="OpenMed",
        device="cpu",
        fixture_count=2,
        generated_at="2026-08-09T00:00:00Z",
        metadata={
            "environment": environment,
            "fixture_ids": [RAW_FIXTURE_VALUE],
            "raw": RAW_FIXTURE_VALUE,
        },
        rows=(
            ComparatorMatrixRow(
                system="OpenMed",
                status="scored",
                fixture_count=2,
                benchmark_report=benchmark,
            ),
            ComparatorMatrixRow(
                system="missing-extra",
                status=STATUS_NOT_AVAILABLE,
                fixture_count=2,
                reason=f"dependency missing for {RAW_FIXTURE_VALUE}",
                metadata={"raw_text": RAW_FIXTURE_VALUE},
            ),
        ),
    )
