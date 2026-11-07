"""Tests for relation reliability and selective-prediction metrics."""

from __future__ import annotations

import pytest

from openmed.eval.metrics import (
    area_under_risk_coverage,
    relation_reliability_report,
    risk_coverage_curve,
    selective_prediction_report,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "relation_type": "adverse_event",
            "confidence": 0.95,
            "correct": True,
            "weight": 2.0,
        },
        {
            "relation_type": "drug_to_dose",
            "confidence": 0.80,
            "correct": True,
            "weight": 1.0,
        },
        {
            "relation_type": "adverse_event",
            "confidence": 0.40,
            "correct": False,
            "weight": 5.0,
        },
    ]


def test_risk_coverage_uses_raw_counts_and_keeps_weights_distinct() -> None:
    curve = risk_coverage_curve(_records())

    assert [row["retained_count"] for row in curve] == [1, 2, 3]
    assert [row["coverage"] for row in curve] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert [row["retained_weight"] for row in curve] == pytest.approx([2.0, 3.0, 8.0])
    assert curve[-1]["empirical_risk"] == pytest.approx(5 / 8)
    assert curve[0]["per_type"]["drug_to_dose"]["retained_count"] == 0


def test_selective_report_uses_fixed_trapezoidal_and_oracle_convention() -> None:
    report = selective_prediction_report(_records())
    curve = report["risk_coverage_table"]

    assert report["aurc"] == pytest.approx(area_under_risk_coverage(curve))
    assert report["excess_aurc"] == pytest.approx(
        report["aurc"] - report["oracle_aurc"]
    )
    assert report["excess_aurc"] == pytest.approx(0.0)
    assert "not finite-sample" in report["note"]
    assert "Trapezoidal" in report["aurc_convention"]


def test_relation_reliability_report_emits_independent_type_slices() -> None:
    report = relation_reliability_report(_records(), n_bins=5)

    assert set(report["per_type"]) == {"adverse_event", "drug_to_dose"}
    assert report["per_type"]["adverse_event"]["sample_count"] == 2
    assert len(report["per_type"]["drug_to_dose"]["reliability"]) == 5
