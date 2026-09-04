"""Focused tests for aggregate tabular re-identification reports."""

from __future__ import annotations

import json

import pytest

from openmed.risk import (
    TabularRiskReport,
    TabularRiskThresholds,
    render_tabular_risk_json,
    render_tabular_risk_markdown,
    tabular_risk_report,
)


def _synthetic_rows() -> list[dict[str, object]]:
    return [
        {
            "record_id": "synthetic-row-001",
            "age": 30,
            "region": "north",
            "diagnosis": "synthetic-condition-a",
        },
        {
            "record_id": "synthetic-row-002",
            "age": 30,
            "region": "north",
            "diagnosis": "synthetic-condition-b",
        },
        {
            "record_id": "synthetic-row-003",
            "age": 40,
            "region": "south",
            "diagnosis": "synthetic-condition-c",
        },
        {
            "record_id": "synthetic-row-004",
            "age": 40,
            "region": "south",
            "diagnosis": "synthetic-condition-d",
        },
        {
            "record_id": "synthetic-row-005",
            "age": 91,
            "region": "west",
            "diagnosis": "synthetic-condition-e",
        },
    ]


def test_report_is_aggregate_only_and_deterministic() -> None:
    rows = _synthetic_rows()
    first = tabular_risk_report(
        rows,
        quasi_identifiers=["age", "region"],
        generalization={"age": "synthetic-ten-year", "region": "synthetic-district"},
        suppressed_rows=[4],
        thresholds={
            "target_k": 2,
            "max_suppression_rate": 0.25,
            "min_generalization_coverage": 1.0,
        },
    )
    second = tabular_risk_report(
        list(reversed(rows)),
        quasi_identifiers=["region", "age"],
        generalization={"region": "synthetic-district", "age": "synthetic-ten-year"},
        suppressed_rows=[0],
        thresholds=TabularRiskThresholds(
            minimum_k=2,
            max_suppression_rate=0.25,
            min_generalization_coverage=1.0,
        ),
    )

    assert isinstance(first, TabularRiskReport)
    assert first.to_dict() == second.to_dict()
    assert json.loads(first.to_json()) == first.to_dict()
    assert first.row_count == 4
    assert first.source_row_count == 5
    assert first.suppressed_row_count == 1
    assert first.minimum_k == 2
    assert first.risk_score == pytest.approx(0.5)
    assert first.meets_thresholds is True

    serialized = first.to_json()
    markdown = first.to_markdown()
    for raw_value in (
        "synthetic-row-001",
        "synthetic-condition-a",
        "synthetic-ten-year",
        "synthetic-district",
        "north",
    ):
        assert raw_value not in serialized
        assert raw_value not in markdown
    assert "class_fingerprints" not in serialized
    assert "suppression offsets" in markdown


def test_threshold_failures_are_explicit_and_json_renderer_is_allowlisted() -> None:
    report = tabular_risk_report(
        [
            {"age": 20, "region": "a"},
            {"age": 21, "region": "b"},
        ],
        quasi_identifiers=["age", "region"],
        thresholds={
            "minimum_k": 2,
            "max_singleton_rate": 0.0,
            "max_reidentification_risk": 0.5,
        },
    )

    assert report["status"] == {
        "meets_minimum_k": False,
        "meets_max_singleton_rate": False,
        "meets_max_reidentification_risk": False,
        "meets_max_suppression_rate": True,
        "meets_min_generalization_coverage": True,
        "meets_thresholds": False,
        "outcome": "review",
    }
    assert report.meets_thresholds is False

    unsafe = report.to_dict()
    unsafe["raw_rows"] = [{"secret": "synthetic-raw-value"}]
    unsafe["risk"]["unexpected"] = "synthetic-raw-value"
    rendered = render_tabular_risk_json(unsafe)
    markdown = render_tabular_risk_markdown(unsafe)
    assert "synthetic-raw-value" not in rendered
    assert "synthetic-raw-value" not in markdown
    assert "review" in markdown


def test_schema_only_and_suppression_count_inputs_are_supported() -> None:
    report = tabular_risk_report(
        [{"age_band": "30-39"}, {"age_band": "30-39"}],
        schema={"age_band": "categorical", "unused": "integer"},
        quasi_identifiers=["age_band"],
        suppression_count=3,
    )

    assert report["schema"]["column_count"] == 2
    assert report["row_counts"] == {
        "source": 5,
        "analyzed": 2,
        "suppressed": 3,
        "suppression_rate": pytest.approx(0.6),
    }
    assert report["equivalence_classes"]["minimum_k"] == 2
    assert report["generalization"]["quasi_identifier_coverage"] == 0.0


def test_invalid_inputs_do_not_echo_source_values() -> None:
    with pytest.raises(ValueError, match="non-finite") as error:
        tabular_risk_report(
            [{"qi": float("inf"), "secret": "synthetic-sensitive-value"}],
            quasi_identifiers=["qi"],
        )
    assert "synthetic-sensitive-value" not in str(error.value)

    with pytest.raises(ValueError, match="unknown schema columns") as error:
        tabular_risk_report(
            [{"qi": "synthetic-qi-value"}],
            quasi_identifiers=["missing"],
        )
    assert "synthetic-qi-value" not in str(error.value)


def test_inferred_quasi_identifiers_exclude_identifier_like_columns() -> None:
    report = tabular_risk_report(
        [
            {"patient_id": "synthetic-patient-a", "age": 30},
            {"patient_id": "synthetic-patient-b", "age": 30},
        ]
    )

    assert report["quasi_identifiers"] == {
        "columns": ["age"],
        "count": 1,
        "inferred": True,
    }
    assert "synthetic-patient-a" not in json.dumps(report, sort_keys=True)
