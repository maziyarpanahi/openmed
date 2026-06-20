"""Unit tests for bootstrap confidence intervals on benchmark metrics."""

from __future__ import annotations

import pytest

from openmed.eval.harness import BenchmarkFixture, run_benchmark
from openmed.eval.metrics import (
    BootstrapCI,
    bootstrap_ci,
    compute_confidence_intervals,
)


def _rate(docs):
    """Aggregate ``(numerator, denominator)`` document pairs into a rate."""
    numerator = sum(doc[0] for doc in docs)
    denominator = sum(doc[1] for doc in docs)
    return numerator / denominator if denominator else 0.0


def test_bootstrap_ci_brackets_point_and_is_reproducible():
    docs = [(2, 10), (0, 8), (5, 5), (1, 4), (3, 9), (0, 6), (4, 7), (2, 3)]

    ci = bootstrap_ci(docs, _rate, n_resamples=200, alpha=0.05, seed=7)

    assert isinstance(ci, BootstrapCI)
    assert not ci.degenerate
    assert ci.lower <= ci.point <= ci.upper
    assert ci.point == pytest.approx(_rate(docs))

    # A fixed seed is fully reproducible.
    again = bootstrap_ci(docs, _rate, n_resamples=200, alpha=0.05, seed=7)
    assert again.to_dict() == ci.to_dict()

    # A different seed may shift the bounds but must still bracket the point.
    other = bootstrap_ci(docs, _rate, n_resamples=200, alpha=0.05, seed=11)
    assert other.lower <= other.point <= other.upper
    assert other.to_dict() != ci.to_dict()


def test_bootstrap_ci_single_document_is_zero_width_and_flagged():
    ci = bootstrap_ci([(2, 10)], _rate, n_resamples=200, seed=0)

    assert ci.degenerate is True
    assert ci.point == pytest.approx(0.2)
    assert ci.lower == ci.upper == pytest.approx(0.2)


def test_bootstrap_ci_empty_corpus_is_zero_width_and_flagged():
    ci = bootstrap_ci([], _rate, n_resamples=200, seed=0)

    assert ci.degenerate is True
    assert ci.point == 0.0
    assert ci.lower == ci.upper == 0.0


def test_bootstrap_ci_all_zero_sample_collapses_without_flag():
    # Multiple documents that all evaluate to zero: every resample is zero, so
    # the interval is zero-width but not flagged degenerate (it could have
    # varied in principle).
    docs = [(0, 10), (0, 5), (0, 8)]

    ci = bootstrap_ci(docs, _rate, n_resamples=200, seed=3)

    assert ci.degenerate is False
    assert ci.point == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_compute_confidence_intervals_covers_required_metrics():
    per_document_spans = [
        (
            [{"start": 8, "end": 12, "label": "PERSON"}],
            [{"start": 8, "end": 12, "label": "PERSON"}],
        ),
        (
            [{"start": 4, "end": 15, "label": "SSN"}],
            [],
        ),
    ]

    intervals = compute_confidence_intervals(
        per_document_spans, n_resamples=200, seed=0
    )

    assert set(intervals) == {
        "leakage",
        "character_recall",
        "exact_span_f1",
        "relaxed_span_f1",
    }
    for payload in intervals.values():
        assert payload["lower"] <= payload["point"] <= payload["upper"]


def _two_fixture_setup():
    fixtures = [
        BenchmarkFixture.from_mapping(
            {
                "id": "a",
                "text": "Patient John",
                "language": "en",
                "gold_spans": [{"start": 8, "end": 12, "label": "PERSON"}],
            }
        ),
        BenchmarkFixture.from_mapping(
            {
                "id": "b",
                "text": "SSN 123-45-6789",
                "language": "en",
                "gold_spans": [{"start": 4, "end": 15, "label": "SSN"}],
            }
        ),
    ]

    def runner(fixture, model_name, device):
        if fixture.fixture_id == "a":
            return [{"start": 8, "end": 12, "label": "PERSON"}]
        return []

    return fixtures, runner


def test_harness_omits_confidence_intervals_by_default():
    fixtures, runner = _two_fixture_setup()

    report = run_benchmark(
        fixtures,
        suite="golden",
        model_name="test-model",
        runner=runner,
        generated_at="2026-06-11T00:00:00Z",
    )

    assert "confidence_interval" not in report.metrics["leakage"]
    assert "confidence_interval" not in report.metrics["character_recall"]
    assert "confidence_interval" not in report.metrics["exact_span_f1"]


def test_harness_attaches_confidence_intervals_when_enabled():
    fixtures, runner = _two_fixture_setup()

    report = run_benchmark(
        fixtures,
        suite="golden",
        model_name="test-model",
        runner=runner,
        generated_at="2026-06-11T00:00:00Z",
        confidence_intervals=True,
        ci_resamples=200,
        ci_seed=0,
    )

    for key in ("leakage", "character_recall", "exact_span_f1", "relaxed_span_f1"):
        ci = report.metrics[key]["confidence_interval"]
        assert ci["lower"] <= ci["point"] <= ci["upper"]
        assert ci["degenerate"] is False

    # The CI point estimate matches the reported corpus point estimate.
    assert report.metrics["leakage"]["confidence_interval"]["point"] == pytest.approx(
        report.metrics["leakage"]["overall"]
    )

    # Reproducible under a fixed seed.
    again = run_benchmark(
        fixtures,
        suite="golden",
        model_name="test-model",
        runner=runner,
        generated_at="2026-06-11T00:00:00Z",
        confidence_intervals=True,
        ci_resamples=200,
        ci_seed=0,
    )
    assert (
        again.metrics["leakage"]["confidence_interval"]
        == report.metrics["leakage"]["confidence_interval"]
    )
