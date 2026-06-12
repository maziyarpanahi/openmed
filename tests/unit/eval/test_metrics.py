"""Unit tests for eval benchmark metrics and reports."""

from __future__ import annotations

import json

import pytest

from openmed.eval.harness import BenchmarkFixture, run_benchmark
from openmed.eval.metrics import (
    EvalSpan,
    compute_character_recall,
    compute_exact_span_f1,
    compute_leakage_rate,
    compute_recall_slices,
    compute_relaxed_span_f1,
)
from openmed.eval.report import BenchmarkReport


def test_eval_modules_import_cleanly():
    import openmed.eval.harness
    import openmed.eval.metrics
    import openmed.eval.report
    import openmed.eval.suites

    assert openmed.eval.harness.run_benchmark
    assert openmed.eval.metrics.compute_leakage_rate
    assert openmed.eval.report.BenchmarkReport
    assert openmed.eval.suites.DEFAULT_SUITES == ("golden", "i2b2", "n2c2", "shield")


def test_leakage_rate_is_char_weighted_and_sliced():
    text = "SSN 123-45-6789 John"
    gold = [
        {"start": 4, "end": 15, "label": "SSN", "language": "en", "device": "cpu"},
        {"start": 16, "end": 20, "label": "PERSON", "language": "en", "device": "cpu"},
    ]
    predicted = [
        {"start": 16, "end": 20, "label": "PERSON", "language": "en", "device": "cpu"},
    ]

    result = compute_leakage_rate(gold, predicted, source_text=text)

    assert result.overall == pytest.approx(11 / 15)
    assert result["overall"] == pytest.approx(11 / 15)
    assert result.by_label["SSN"] == pytest.approx(1.0)
    assert result.by_label["PERSON"] == pytest.approx(0.0)
    assert result.by_language["en"] == pytest.approx(11 / 15)
    assert result.by_device["cpu"] == pytest.approx(11 / 15)
    assert result.leaked_chars_by_label["SSN"] == 11
    assert result.total_chars_by_label["PERSON"] == 4


def test_zero_gold_spans_do_not_report_leakage_and_have_full_recall():
    leakage = compute_leakage_rate([], [])
    recall = compute_character_recall([], [])
    exact = compute_exact_span_f1([], [])
    relaxed = compute_relaxed_span_f1([], [])

    assert leakage.overall == 0.0
    assert leakage.total_chars == 0
    assert recall.rate == 1.0
    assert exact.f1 == 1.0
    assert relaxed.f1 == 1.0


def test_exact_and_relaxed_f1_differ_on_boundary_drift():
    gold = [EvalSpan(start=0, end=4, label="PERSON", text="John")]
    predicted = [EvalSpan(start=0, end=3, label="PERSON", text="Joh")]

    exact = compute_exact_span_f1(gold, predicted)
    relaxed = compute_relaxed_span_f1(gold, predicted)

    assert exact.f1 == 0.0
    assert relaxed.f1 == 1.0
    assert exact.false_negatives == 1
    assert relaxed.true_positives == 1


def test_recall_slices_cover_label_language_and_device_edges():
    gold = [
        EvalSpan(start=0, end=4, label="PERSON", language="en", device="cpu"),
        EvalSpan(start=10, end=21, label="SSN", language="fr", device="coreml"),
    ]
    predicted = [
        EvalSpan(start=0, end=4, label="PERSON", language="en", device="cpu"),
        EvalSpan(start=10, end=15, label="SSN", language="fr", device="coreml"),
    ]

    recall = compute_recall_slices(gold, predicted)

    assert recall.overall == pytest.approx(9 / 15)
    assert recall.by_label["PERSON"] == pytest.approx(1.0)
    assert recall.by_label["SSN"] == pytest.approx(5 / 11)
    assert recall.by_language["fr"] == pytest.approx(5 / 11)
    assert recall.by_device["coreml"] == pytest.approx(5 / 11)
    assert "mlx-8bit" in recall.by_device


def test_benchmark_report_serializes_deterministically_to_json_and_markdown():
    report = BenchmarkReport(
        suite="golden",
        model_name="privacy-filter",
        device="cpu",
        fixture_count=2,
        generated_at="2026-06-11T00:00:00Z",
        metrics={
            "leakage": {"overall": 0.25, "by_label": {"SSN": 1.0, "PERSON": 0.0}},
            "exact_span_f1": {"f1": 0.8},
        },
        metadata={"z": 1, "a": {"b": True}},
    )

    assert report.to_json() == report.to_json()
    assert json.loads(report.to_json())["metrics"]["leakage"]["overall"] == 0.25
    markdown = report.to_markdown()

    assert markdown == report.to_markdown()
    assert "| `exact_span_f1.f1` | 0.8 |" in markdown
    assert "| `a.b` | true |" in markdown
    assert "| `z` | 1 |" in markdown


def test_harness_runs_with_injected_runner_without_loading_models():
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-1",
            "text": "Patient John",
            "language": "en",
            "gold_spans": [{"start": 8, "end": 12, "label": "PERSON"}],
        }
    )

    def runner(fixture, model_name, device):
        assert model_name == "test-model"
        assert device == "cpu"
        return [{"start": 8, "end": 12, "label": "PERSON"}]

    report = run_benchmark(
        [fixture],
        suite="golden",
        model_name="test-model",
        runner=runner,
        generated_at="2026-06-11T00:00:00Z",
    )

    assert report.fixture_count == 1
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["exact_span_f1"]["f1"] == 1.0
