"""Tests for repeated-run flaky-eval detection."""

from __future__ import annotations

import pytest

from openmed.eval import detect_flaky_eval as exported_detect_flaky_eval
from openmed.eval.flaky import DEFAULT_FLAKY_TOLERANCE, detect_flaky_eval
from openmed.eval.report import BenchmarkReport


def test_deterministic_callable_reports_stable_and_respects_n_runs() -> None:
    calls = 0

    def run() -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {"leakage": 0.0, "recall": 1.0}

    report = detect_flaky_eval(run, n_runs=4)

    assert calls == 4
    assert report.n_runs == 4
    assert report.stable is True
    assert report.verdict == "stable"
    assert report.flaky_metrics == ()
    assert report.metric("leakage").minimum == pytest.approx(0.0)
    assert report.metric("leakage").maximum == pytest.approx(0.0)
    assert report.metric("leakage").spread == pytest.approx(0.0)
    assert report.metric("leakage").tolerance == pytest.approx(DEFAULT_FLAKY_TOLERANCE)
    assert report.metric("leakage").verdict == "stable"
    assert report.to_dict()["metrics"]["leakage"]["min"] == pytest.approx(0.0)


def test_jittery_callable_exceeding_tolerance_is_flagged_flaky() -> None:
    values = iter([0.90, 0.91, 0.95])

    report = detect_flaky_eval(
        lambda: {"exact_span_f1": next(values)},
        n_runs=3,
        tolerance=0.02,
    )

    metric = report.metric("exact_span_f1")
    assert report.stable is False
    assert report.verdict == "flaky"
    assert report.flaky_metrics == ("exact_span_f1",)
    assert metric.minimum == pytest.approx(0.90)
    assert metric.maximum == pytest.approx(0.95)
    assert metric.spread == pytest.approx(0.05)
    assert metric.tolerance == pytest.approx(0.02)
    assert metric.verdict == "flaky"


def test_per_metric_tolerance_overrides_default_near_zero_fallback() -> None:
    runs = iter(
        [
            {"leakage": 0.0, "recall": 0.940},
            {"leakage": 1e-9, "recall": 0.950},
            {"leakage": 0.0, "recall": 0.945},
        ]
    )

    report = detect_flaky_eval(
        lambda: next(runs),
        n_runs=3,
        tolerance={"recall": 0.02},
    )

    assert report.metric("recall").stable is True
    assert report.metric("recall").spread == pytest.approx(0.01)
    assert report.metric("recall").tolerance == pytest.approx(0.02)
    assert report.metric("leakage").stable is False
    assert report.metric("leakage").spread == pytest.approx(1e-9)
    assert report.metric("leakage").tolerance == pytest.approx(DEFAULT_FLAKY_TOLERANCE)
    assert report.flaky_metrics == ("leakage",)


def test_benchmark_report_metrics_are_flattened_for_variance_checks() -> None:
    reports = iter(
        [
            _benchmark_report(leakage=0.0, recall=1.0),
            _benchmark_report(leakage=0.0, recall=1.0),
        ]
    )

    report = exported_detect_flaky_eval(lambda: next(reports), n_runs=2)

    assert report.stable is True
    assert report.metric("leakage.overall").spread == pytest.approx(0.0)
    assert report.metric("recall.overall").spread == pytest.approx(0.0)


def _benchmark_report(*, leakage: float, recall: float) -> BenchmarkReport:
    return BenchmarkReport(
        suite="golden",
        model_name="privacy-filter",
        device="cpu",
        fixture_count=1,
        metrics={
            "leakage": {"overall": leakage},
            "recall": {"overall": recall},
            "status": "green",
        },
    )
