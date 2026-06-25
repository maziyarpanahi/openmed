from __future__ import annotations

import json

import pytest

from openmed.eval import ThresholdSweepReport, sweep_confidence_thresholds
from openmed.eval.threshold_sweep import ThresholdSweepPoint


def _gold_spans() -> list[dict[str, object]]:
    return [
        {"start": 0, "end": 5, "label": "PERSON", "text": "Alice"},
        {"start": 10, "end": 20, "label": "DATE", "text": "2030-01-01"},
        {"start": 30, "end": 47, "label": "EMAIL", "text": "alice@example.test"},
        {"start": 60, "end": 68, "label": "PHONE", "text": "555-0100"},
    ]


def _predicted_spans() -> list[dict[str, object]]:
    return [
        {
            "start": 0,
            "end": 5,
            "label": "PERSON",
            "confidence": 0.95,
            "text": "Alice",
        },
        {
            "start": 10,
            "end": 20,
            "label": "DATE",
            "confidence": 0.85,
            "text": "2030-01-01",
        },
        {
            "start": 30,
            "end": 47,
            "label": "EMAIL",
            "confidence": 0.60,
            "text": "alice@example.test",
        },
        {
            "start": 80,
            "end": 89,
            "label": "PERSON",
            "confidence": 0.90,
            "text": "Synthetic",
        },
        {
            "start": 90,
            "end": 98,
            "label": "PHONE",
            "confidence": 0.50,
            "text": "555-0199",
        },
    ]


def test_threshold_sweep_reproduces_hand_counted_precision_recall() -> None:
    report = sweep_confidence_thresholds(
        _gold_spans(),
        _predicted_spans(),
        thresholds=[0.0, 0.6, 0.86, 1.0],
        precision_floor=0.7,
        recall_floor=0.5,
    )

    points = {point.threshold: point for point in report.curve_points}
    assert points[0.0].true_positives == 3
    assert points[0.0].false_positives == 2
    assert points[0.0].false_negatives == 1
    assert points[0.0].precision == pytest.approx(3 / 5)
    assert points[0.0].recall == pytest.approx(3 / 4)

    assert points[0.6].true_positives == 3
    assert points[0.6].false_positives == 1
    assert points[0.6].false_negatives == 1
    assert points[0.6].precision == pytest.approx(3 / 4)
    assert points[0.6].recall == pytest.approx(3 / 4)

    assert points[0.86].true_positives == 1
    assert points[0.86].false_positives == 1
    assert points[0.86].false_negatives == 3
    assert points[0.86].precision == pytest.approx(1 / 2)
    assert points[0.86].recall == pytest.approx(1 / 4)

    assert points[1.0].true_positives == 0
    assert points[1.0].false_positives == 0
    assert points[1.0].false_negatives == 4
    assert points[1.0].precision == 1.0
    assert points[1.0].recall == 0.0


def test_threshold_sweep_recall_is_monotonic_as_threshold_rises() -> None:
    report = sweep_confidence_thresholds(_gold_spans(), _predicted_spans())

    thresholds = [point.threshold for point in report.curve_points]
    recalls = [point.recall for point in report.curve_points]
    assert thresholds == sorted(thresholds)
    assert recalls == sorted(recalls, reverse=True)


def test_threshold_sweep_recommendations_respect_floors() -> None:
    report = sweep_confidence_thresholds(
        _gold_spans(),
        _predicted_spans(),
        thresholds=[0.0, 0.6, 0.86, 1.0],
        precision_floor=0.7,
        recall_floor=0.5,
    )

    recall_point = report.recall_maximizing_point
    precision_point = report.precision_maximizing_point
    assert recall_point is not None
    assert precision_point is not None
    assert recall_point.threshold == pytest.approx(0.6)
    assert recall_point.precision >= report.precision_floor
    assert precision_point.threshold == pytest.approx(0.6)
    assert precision_point.recall >= report.recall_floor


def test_threshold_sweep_output_is_deterministic_and_phi_free() -> None:
    report = sweep_confidence_thresholds(
        _gold_spans(),
        _predicted_spans(),
        thresholds=[0.0, 0.6],
    )

    assert report.to_json() == report.to_json()
    assert report.to_markdown() == report.to_markdown()

    payload = json.loads(report.to_json())
    point_keys = set(payload["curve_points"][0])
    assert point_keys == {
        "threshold",
        "predicted_count",
        "gold_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
    }

    serialized = report.to_json() + report.to_markdown()
    for forbidden in (
        "Alice",
        "2030-01-01",
        "alice@example.test",
        "555-0100",
        "PERSON",
        "EMAIL",
    ):
        assert forbidden not in serialized


def test_threshold_sweep_deterministic_tie_breaks() -> None:
    report = sweep_confidence_thresholds(
        _gold_spans(),
        _predicted_spans(),
        thresholds=[0.0, 0.1],
    )

    assert report.recall_maximizing_point is not None
    assert report.precision_maximizing_point is not None
    assert report.recall_maximizing_point.threshold == 0.0
    assert report.precision_maximizing_point.threshold == pytest.approx(0.1)


def test_threshold_sweep_exports_from_eval_package() -> None:
    report = sweep_confidence_thresholds(
        _gold_spans(),
        _predicted_spans(),
        thresholds=[0.5],
    )

    assert isinstance(report, ThresholdSweepReport)
    assert isinstance(report.curve_points[0], ThresholdSweepPoint)
