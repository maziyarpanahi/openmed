"""Tests for per-type relation calibration and abstention."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical.relations.calibration import (
    DEFAULT_MIN_RETAINED_RELATION_ACCURACY,
    RELATION_STATUS_UNCERTAIN,
    RelationCalibrationConsistencyError,
    apply_relation_abstention,
    assert_relation_consistency_gate,
    fit_relation_calibrator,
    relation_calibration_report,
)
from openmed.core.decoding import SpanEdge
from openmed.eval.calibrate import write_calibration_artifacts

FIXTURE_PATH = (
    Path(__file__).parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "relation_calibration.jsonl"
)
GENERATED_AT = "2026-08-04T00:00:00+00:00"


def _fixture_rows() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _calibration_and_evaluation() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    rows = _fixture_rows()
    return (
        [row for row in rows if row["split"] == "calibration"],
        [row for row in rows if row["split"] == "evaluation"],
    )


def _held_out_report() -> tuple[object, dict[str, object]]:
    calibration, evaluation = _calibration_and_evaluation()
    calibrator = fit_relation_calibrator(calibration)
    report = relation_calibration_report(
        evaluation,
        calibrator=calibrator,
        generated_at=GENERATED_AT,
    )
    return calibrator, report


def test_committed_gold_reduces_ece_and_passes_retained_accuracy_gate() -> None:
    calibrator, report = _held_out_report()

    assert set(calibrator.groups) == {
        "adverse_event",
        "drug_to_dose",
        "temporally_before",
    }
    assert {group.method for group in calibrator.groups.values()} == {"isotonic"}
    ece = report["expected_calibration_error"]
    assert ece["calibrated"] < ece["raw"]
    assert set(report["per_type_reliability"]) == set(calibrator.groups)

    selective = report["selective_prediction"]
    curve = selective["risk_coverage_table"]
    assert [row["empirical_risk"] for row in curve] == sorted(
        row["empirical_risk"] for row in curve
    )
    assert all(row["retained_count"] <= row["total_count"] for row in curve)
    assert all("retained_weight" in row for row in curve)

    pooled = report["operating_points"]["*"]
    assert pooled["abstention_rate"] <= 0.35
    assert pooled["accuracy"] > selective["full_coverage_accuracy"]
    assert pooled["accuracy"] >= DEFAULT_MIN_RETAINED_RELATION_ACCURACY
    assert report["consistency_gate"]["passed"] is True
    assert_relation_consistency_gate(report)


def test_sparse_relation_type_uses_pooled_temperature_fallback() -> None:
    records = [
        {"relation_type": "adverse_event", "score": 0.8, "correct": True},
        {"relation_type": "adverse_event", "score": 0.2, "correct": False},
        {"relation_type": "drug_to_dose", "score": 0.7, "correct": True},
        {"relation_type": "drug_to_dose", "score": 0.3, "correct": False},
    ]

    calibrator = fit_relation_calibrator(records)

    assert calibrator.groups["adverse_event"].method == "pooled_temperature"
    assert calibrator.groups["drug_to_dose"].temperature == pytest.approx(
        calibrator.fallback.temperature
    )
    assert 0.0 <= calibrator.predict(relation_type="unknown", score=0.6) <= 1.0


def test_uncertain_relation_is_not_asserted_but_remains_audited() -> None:
    calibrator, report = _held_out_report()
    operating_points = report["operating_points"]
    relations = (
        SpanEdge("med-node", "dose-easy", "drug_to_dose", 0.90),
        SpanEdge("med-node", "dose-ambiguous", "drug_to_dose", 0.50),
    )

    result = apply_relation_abstention(relations, calibrator, operating_points)

    assert len(result.relations) == 2
    assert len(result.asserted) == 1
    assert len(result.asserted_edges) == 1
    assert len(result.uncertain) == 1
    assert result.uncertain[0].status == RELATION_STATUS_UNCERTAIN
    assert result.uncertain[0].asserted is False
    assert (
        result.uncertain[0].relation.metadata["relation_calibration"]["status"]
        == RELATION_STATUS_UNCERTAIN
    )
    serialized_audit = json.dumps(result.audit_trace, sort_keys=True)
    assert len(result.audit_trace) == 2
    assert "relation_key_hash" in serialized_audit
    assert "med-node" not in serialized_audit
    assert "dose-ambiguous" not in serialized_audit


def test_consistency_gate_raises_below_pinned_floor() -> None:
    _, report = _held_out_report()
    report["operating_points"]["*"]["accuracy"] = 0.50

    with pytest.raises(RelationCalibrationConsistencyError):
        assert_relation_consistency_gate(report)


def test_relation_report_artifact_is_versioned_and_byte_reproducible(
    tmp_path: Path,
) -> None:
    _, report = _held_out_report()
    span_samples = [
        {
            "model_id": "artifact-test",
            "label": "NAME",
            "language": "en",
            "score": 0.9,
            "target": True,
        },
        {
            "model_id": "artifact-test",
            "label": "NAME",
            "language": "en",
            "score": 0.1,
            "target": False,
        },
    ]

    first = write_calibration_artifacts(
        span_samples,
        artifact_dir=tmp_path / "first",
        model_id="artifact-test",
        suite="relation-calibration",
        conformal_alpha=None,
        generated_at=GENERATED_AT,
        relation_report=report,
    )
    second = write_calibration_artifacts(
        span_samples,
        artifact_dir=tmp_path / "second",
        model_id="artifact-test",
        suite="relation-calibration",
        conformal_alpha=None,
        generated_at=GENERATED_AT,
        relation_report=report,
    )

    assert first.relation_report_path is not None
    assert second.relation_report_path is not None
    assert first.relation_report_path.read_bytes() == (
        second.relation_report_path.read_bytes()
    )
    payload = json.loads(first.relation_report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == "openmed.calibration.relation_report"
    assert payload["generated_at"] == GENERATED_AT
