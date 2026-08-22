"""Focused tests for the counts-only comparator report renderer."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from openmed.eval.comparator_report import (
    DEFAULT_METRIC_DEFINITIONS,
    ComparatorReportRow,
    CountsOnlyComparatorReport,
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


def test_counts_only_report_has_a_distinct_public_type() -> None:
    from openmed.eval import ComparatorReport as BenchmarkComparatorReport
    from openmed.eval import (
        CountsOnlyComparatorReport as PublicCountsOnlyComparatorReport,
    )

    assert PublicCountsOnlyComparatorReport is CountsOnlyComparatorReport
    assert PublicCountsOnlyComparatorReport is not BenchmarkComparatorReport


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

    assert isinstance(report, CountsOnlyComparatorReport)
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


def test_source_identifiers_are_hashed_even_when_they_look_like_plain_names() -> None:
    raw_values = {
        "suite": "AdaLovelace",
        "model": "GraceHopper",
        "device": "AlanTuring",
        "system": "KatherineJohnson",
    }
    report = build_comparator_report(
        {
            "suite": raw_values["suite"],
            "model_name": raw_values["model"],
            "device": raw_values["device"],
            "fixture_count": 1,
            "rows": [
                {
                    "system": raw_values["system"],
                    "status": "scored",
                    "fixture_count": 1,
                }
            ],
        },
        environment={"python": "3.12"},
    )

    rendered = report.to_json()
    assert all(value not in rendered for value in raw_values.values())
    assert report.suite.startswith("sha256:")
    assert report.rows[0].system.startswith("sha256:")


def test_failure_counts_are_bounded_without_materializing_each_event() -> None:
    report = build_comparator_report(
        {
            "suite": "comparator",
            "model_name": "OpenMed",
            "device": "cpu",
            "rows": [],
            "failure_count": 10**30,
        },
        environment={"python": "3.12"},
    )

    assert report.failure_summary == {
        "by_category": {"other": 1_000_000_000},
        "systems_affected": 0,
        "total": 1_000_000_000,
    }


def test_systems_affected_counts_unique_sanitized_systems() -> None:
    report = build_comparator_report(
        {
            "rows": [
                {"system": "OpenMed", "status": "failed"},
                {"system": "OpenMed", "status": "failed"},
            ]
        },
        environment={"python": "3.12"},
    )

    assert report.failure_summary["systems_affected"] == 1
    assert report.failure_summary["total"] == 2


def test_sanitized_report_state_is_deeply_immutable() -> None:
    row = ComparatorReportRow(
        system="OpenMed",
        status="scored",
        fixture_count=1,
        metrics={"leakage_rate": 0.0},
        metric_counts={"leakage_rate": {"total_graphemes": 1}},
    )
    report = CountsOnlyComparatorReport(
        suite="comparator",
        model_name="OpenMed",
        device="cpu",
        fixture_count=1,
        rows=(row,),
    )

    with pytest.raises(TypeError):
        row.metrics["leakage_rate"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        row.metric_counts["leakage_rate"]["total_graphemes"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        report.failure_summary["total"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        DEFAULT_METRIC_DEFINITIONS["leakage_rate"] = RAW_FIXTURE_VALUE  # type: ignore[index]

    detached = report.to_dict()
    detached["rows"][0]["metrics"]["leakage_rate"] = 1.0
    assert report.rows[0].metrics["leakage_rate"] == 0.0


def test_hostile_values_cannot_invoke_conversion_hooks() -> None:
    hostile = _HostileValue()
    report = build_comparator_report(
        {
            "suite": hostile,
            "model_name": hostile,
            "device": hostile,
            "fixture_count": hostile,
            "rows": [
                {
                    "system": hostile,
                    "status": hostile,
                    "fixture_count": hostile,
                    "reason": hostile,
                    "metrics": {"leakage_rate": hostile},
                }
            ],
        },
        environment={"hostile": hostile},
    )

    assert report.suite == "comparator"
    assert report.rows[0].system == "system"
    assert RAW_FIXTURE_VALUE not in report.to_json()


def test_custom_mapping_is_rejected_without_reading_it() -> None:
    with pytest.raises(TypeError, match="plain dictionary"):
        build_comparator_report(_HostileMapping())  # type: ignore[arg-type]


def test_report_rejects_unbounded_json_indent_and_unknown_schema() -> None:
    report = build_comparator_report(
        {"rows": []},
        environment={"python": "3.12"},
    )

    with pytest.raises(ValueError, match="indent"):
        report.to_json(indent=1000)
    with pytest.raises(ValueError, match="schema version"):
        CountsOnlyComparatorReport.from_dict({"schema_version": 999, "rows": []})
    with pytest.raises(ValueError, match="artifact type"):
        CountsOnlyComparatorReport.from_dict({"artifact_type": "other", "rows": []})


def test_write_failure_does_not_echo_the_output_path() -> None:
    report = build_comparator_report(
        {"rows": []},
        environment={"python": "3.12"},
    )
    unsafe_path = f"{RAW_FIXTURE_VALUE}\0.json"

    with pytest.raises(OSError) as captured:
        report.write_json(unsafe_path)

    assert RAW_FIXTURE_VALUE not in str(captured.value)


def test_sanitized_report_round_trip_does_not_rehash_identifiers() -> None:
    report = build_comparator_report(
        {
            "suite": "AdaLovelace",
            "model_name": "GraceHopper",
            "device": "AlanTuring",
            "rows": [
                {
                    "system": "KatherineJohnson",
                    "status": "scored",
                    "fixture_count": 1,
                }
            ],
        },
        environment={"python": "3.12"},
    )

    restored = CountsOnlyComparatorReport.from_dict(report.to_dict())

    assert restored.to_dict() == report.to_dict()


def test_loaded_report_reconciles_row_counts_and_failure_status() -> None:
    report = CountsOnlyComparatorReport.from_dict(
        {
            "artifact_type": "openmed.eval.comparator_report",
            "fixture_count": 0,
            "rows": [
                {
                    "system": "OpenMed",
                    "status": "not_available",
                    "fixture_count": 3,
                    "failure_count": 0,
                }
            ],
            "schema_version": 1,
        }
    )

    assert report.fixture_count == 3
    assert report.rows[0].failure_count == 1
    assert report.failure_summary["total"] == 1
    assert report.failure_summary["systems_affected"] == 1


class _HostileValue:
    def __bool__(self) -> bool:
        raise AssertionError(RAW_FIXTURE_VALUE)

    def __float__(self) -> float:
        raise AssertionError(RAW_FIXTURE_VALUE)

    def __int__(self) -> int:
        raise AssertionError(RAW_FIXTURE_VALUE)

    def __str__(self) -> str:
        raise AssertionError(RAW_FIXTURE_VALUE)


class _HostileMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise AssertionError(RAW_FIXTURE_VALUE)

    def __iter__(self) -> Iterator[str]:
        raise AssertionError(RAW_FIXTURE_VALUE)

    def __len__(self) -> int:
        raise AssertionError(RAW_FIXTURE_VALUE)


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
