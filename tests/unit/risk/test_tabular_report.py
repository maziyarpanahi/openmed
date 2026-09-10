"""Focused tests for aggregate tabular re-identification reports."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from openmed.risk import (
    TabularRiskReport,
    TabularRiskThresholds,
    render_tabular_risk_json,
    render_tabular_risk_markdown,
    tabular_risk_report,
)

tabular_report_module = importlib.import_module("openmed.risk.tabular_report")


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
    assert "synthetic-patient-a" not in report.to_json()


def test_threshold_and_suppression_mappings_are_closed() -> None:
    rows = [{"age": 30}, {"age": 30}]

    with pytest.raises(ValueError, match="unknown fields") as error:
        tabular_risk_report(
            rows,
            quasi_identifiers=["age"],
            thresholds={"minimum_k": 2, "secret-value": "synthetic-sensitive"},
        )
    assert "synthetic-sensitive" not in str(error.value)

    with pytest.raises(ValueError, match="duplicate aliases"):
        tabular_risk_report(
            rows,
            quasi_identifiers=["age"],
            thresholds={"minimum_k": 2, "target_k": 2},
        )

    with pytest.raises(ValueError, match="unknown fields"):
        tabular_risk_report(rows, suppression={"count": 0, "typo": 1})
    with pytest.raises(ValueError, match="duplicate aliases"):
        tabular_risk_report(rows, suppression={"count": 0, "suppressed_count": 0})


def test_bounded_rows_cells_and_scalar_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tabular_report_module, "MAX_TABULAR_RISK_ROWS", 2)
    with pytest.raises(ValueError, match="item limit"):
        tabular_risk_report([{"age": 1}, {"age": 2}, {"age": 3}])

    monkeypatch.setattr(tabular_report_module, "MAX_TABULAR_RISK_ROWS", 10_000)
    monkeypatch.setattr(tabular_report_module, "MAX_TABULAR_RISK_TOTAL_CELLS", 1)
    with pytest.raises(ValueError, match="cell limit"):
        tabular_risk_report([{"age": 1, "region": "north"}])

    monkeypatch.setattr(tabular_report_module, "MAX_TABULAR_RISK_TOTAL_CELLS", 100)
    monkeypatch.setattr(tabular_report_module, "MAX_TABULAR_RISK_CELL_STRING_CHARS", 4)
    with pytest.raises(ValueError, match="oversized string"):
        tabular_risk_report([{"age": "12345"}])

    with pytest.raises(ValueError, match="out-of-range integer"):
        tabular_risk_report([{"age": 1 << 64}])


def test_column_and_suppression_declarations_reject_duplicates() -> None:
    rows = [{"age": 30}, {"age": 30}]

    with pytest.raises(ValueError, match="safe column identifiers"):
        tabular_risk_report([{"unsafe column": "synthetic-sensitive"}])
    with pytest.raises(ValueError, match="schema columns must be unique"):
        tabular_risk_report(rows, schema=["age", "age"])
    with pytest.raises(ValueError, match="quasi_identifiers must be unique"):
        tabular_risk_report(rows, quasi_identifiers=["age", "age"])
    with pytest.raises(ValueError, match="generalization columns must be unique"):
        tabular_risk_report(rows, generalization=["age", "age"])
    with pytest.raises(ValueError, match="indices must be unique"):
        tabular_risk_report(rows, suppressed_rows=[0, 0])


def test_report_is_deeply_immutable_and_recomputes_renderer_status() -> None:
    report = tabular_risk_report(
        [{"age": 20}, {"age": 21}],
        quasi_identifiers=["age"],
        thresholds={"minimum_k": 2, "max_singleton_rate": 0.0},
    )

    detached_status = report["status"]
    detached_status["outcome"] = "pass"
    assert report["status"]["outcome"] == "review"
    with pytest.raises(TypeError):
        report["status"] = {"outcome": "pass"}  # type: ignore[index]
    with pytest.raises(TypeError):
        json.dumps(report)

    unsafe = report.to_dict()
    unsafe["status"] = {"meets_thresholds": True, "outcome": "pass"}
    unsafe["risk"] = {"max_reidentification_risk": 0.0}
    rendered = json.loads(render_tabular_risk_json(unsafe))
    assert rendered["status"]["outcome"] == "review"
    assert rendered["status"]["meets_thresholds"] is False
    assert rendered["risk"]["max_reidentification_risk"] == 1.0


def test_renderer_rejects_unbounded_format_controls() -> None:
    report = tabular_risk_report([{"age": 30}], quasi_identifiers=["age"])

    with pytest.raises(ValueError, match="indentation"):
        render_tabular_risk_json(report, indent=100)
    with pytest.raises(ValueError, match="title"):
        render_tabular_risk_markdown(report, title="# unsafe\nsynthetic-sensitive")


class _ExplodingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("synthetic-sensitive-value")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-value")

    def __len__(self) -> int:
        raise RuntimeError("synthetic-sensitive-value")

    def items(self) -> Any:
        raise RuntimeError("synthetic-sensitive-value")


class _ExplodingFrame:
    def to_dict(self, orient: str) -> Any:
        raise RuntimeError("synthetic-sensitive-value")


def test_hostile_adapters_fail_without_echoing_values() -> None:
    with pytest.raises(ValueError, match="could not be read") as mapping_error:
        tabular_risk_report([_ExplodingMapping()])
    assert "synthetic-sensitive-value" not in str(mapping_error.value)

    with pytest.raises(TypeError, match="supported records") as frame_error:
        tabular_risk_report(_ExplodingFrame())
    assert "synthetic-sensitive-value" not in str(frame_error.value)
