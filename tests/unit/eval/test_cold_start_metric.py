"""Tests for the cold_start_ms edge metric surfaced by run_benchmark."""

from __future__ import annotations

import time

from openmed.eval.harness import BenchmarkFixture, run_benchmark


def _fixture(fid: str) -> BenchmarkFixture:
    return BenchmarkFixture.from_mapping(
        {
            "id": fid,
            "text": "Patient John",
            "language": "en",
            "gold_spans": [{"start": 8, "end": 12, "label": "PERSON"}],
        }
    )


def _make_runner(cold_sleep_s: float):
    call_count = [0]

    def runner(fixture, model_name, device):
        if call_count[0] == 0:
            time.sleep(cold_sleep_s)
        call_count[0] += 1
        return [{"start": 8, "end": 12, "label": "PERSON"}]

    return runner


def test_cold_start_ms_field_is_present_and_populated():
    """cold_start_ms key exists in latency dict and holds a positive value."""
    fixtures = [_fixture("f1"), _fixture("f2")]
    report = run_benchmark(
        fixtures,
        suite="cold-start-test",
        model_name="test-model",
        runner=_make_runner(cold_sleep_s=0.01),
    )
    latency = report.metrics["latency"]
    assert "cold_start_ms" in latency
    assert latency["cold_start_ms"] is not None
    assert latency["cold_start_ms"] > 0.0
    assert latency["count"] == 1


def test_cold_start_ms_ge_first_steady_state_latency():
    """cold_start_ms is >= p50 of steady-state calls and count excludes it."""
    fixtures = [_fixture(f"f{i}") for i in range(4)]
    report = run_benchmark(
        fixtures,
        suite="cold-start-test",
        model_name="test-model",
        runner=_make_runner(cold_sleep_s=0.02),
    )
    latency = report.metrics["latency"]
    assert latency["cold_start_ms"] >= latency["p50_ms"]
    assert latency["count"] == 3
