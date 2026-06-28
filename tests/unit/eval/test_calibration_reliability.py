"""Unit tests for calibration and reliability diagram data."""

from __future__ import annotations

import json

import pytest

from openmed.eval.harness import BenchmarkFixture, run_benchmark
from openmed.eval.metrics import expected_calibration_error, reliability_bins


def test_perfectly_calibrated_scores_have_zero_ece() -> None:
    records = [{"confidence": 0.2, "correct": index < 2} for index in range(10)] + [
        {"confidence": 0.8, "correct": index < 8} for index in range(10)
    ]

    bins = reliability_bins(records, n_bins=5)

    assert expected_calibration_error(bins) == pytest.approx(0.0)
    assert bins[1]["mean_confidence"] == pytest.approx(0.2)
    assert bins[1]["empirical_accuracy"] == pytest.approx(0.2)
    assert bins[1]["accuracy"] == pytest.approx(0.2)
    assert bins[1]["count"] == 10
    assert bins[4]["mean_confidence"] == pytest.approx(0.8)
    assert bins[4]["empirical_accuracy"] == pytest.approx(0.8)
    assert bins[4]["count"] == 10


def test_overconfident_scores_have_high_ece() -> None:
    records = [{"confidence": 0.95, "correct": False} for _ in range(20)]

    bins = reliability_bins(records, n_bins=10)

    assert expected_calibration_error(bins) == pytest.approx(0.95)
    assert expected_calibration_error(bins) > 0.9


def test_reliability_bins_preserve_empty_bins_and_counts() -> None:
    bins = reliability_bins(
        [
            {"confidence": 0.05, "correct": True},
            {"confidence": 1.0, "correct": False},
        ],
        n_bins=4,
    )

    assert len(bins) == 4
    assert sum(item["count"] for item in bins) == 2
    assert bins[0]["count"] == 1
    assert bins[1]["count"] == 0
    assert bins[1]["mean_confidence"] == 0.0
    assert bins[1]["empirical_accuracy"] == 0.0
    assert bins[2]["count"] == 0
    assert bins[3]["count"] == 1


def test_harness_attaches_calibration_metrics_only_when_enabled() -> None:
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-1",
            "text": "Patient John",
            "language": "en",
            "gold_spans": [{"start": 8, "end": 12, "label": "PERSON"}],
        }
    )

    def runner(fixture, model_name, device):
        return [
            {"start": 8, "end": 12, "label": "PERSON", "confidence": 0.9},
            {"start": 0, "end": 7, "label": "PERSON", "confidence": 0.8},
        ]

    default_report = run_benchmark(
        [fixture],
        suite="golden",
        model_name="test-model",
        runner=runner,
    )
    assert "calibration" not in default_report.metrics

    report = run_benchmark(
        [fixture],
        suite="golden",
        model_name="test-model",
        runner=runner,
        calibration=True,
        calibration_bins=5,
    )

    calibration = report.metrics["calibration"]
    assert calibration["n_bins"] == 5
    assert len(calibration["reliability_bins"]) == 5
    assert sum(item["count"] for item in calibration["reliability_bins"]) == 2

    populated_bin = calibration["reliability_bins"][4]
    assert populated_bin["mean_confidence"] == pytest.approx(0.85)
    assert populated_bin["empirical_accuracy"] == pytest.approx(0.5)
    assert calibration["expected_calibration_error"] == pytest.approx(0.35)

    payload = json.loads(report.to_json())
    assert payload["metrics"]["calibration"] == calibration
