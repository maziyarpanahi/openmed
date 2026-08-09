"""Focused tests for the offline comparator benchmark harness."""

from __future__ import annotations

import json

import pytest

from openmed.eval.comparator import (
    STATUS_NOT_AVAILABLE,
    STATUS_SCORED,
    ComparatorAdapter,
    ComparatorAdapterUnavailable,
    ComparatorBudget,
    ComparatorExecutionError,
    ComparatorFixture,
    ComparatorReport,
    run_comparator_benchmark,
)


def test_comparator_scores_required_metrics_with_shared_budget() -> None:
    fixture = _fixture()
    report = run_comparator_benchmark(
        [fixture],
        [
            ComparatorAdapter(name="exact", runner=_exact_runner),
            ComparatorAdapter(name="weak", runner=_person_only_runner),
        ],
        suite="synthetic-deidentification",
        budget=ComparatorBudget(max_latency_ms=5.0, max_memory_bytes=200),
        clock=_clock([0.0, 0.002, 0.0, 0.002]),
        memory_sampler=_sampler([100, 120, 100, 120]),
        generated_at="2026-08-09T00:00:00Z",
    )

    assert isinstance(report, ComparatorReport)
    assert report.fixture_count == 1
    assert [row.adapter for row in report.results] == ["exact", "weak"]

    exact = report.result("exact")
    assert exact.status == STATUS_SCORED
    assert exact.metrics is not None
    assert exact.metrics.precision == 1.0
    assert exact.metrics.recall == 1.0
    assert exact.metrics.critical_leakage == 0.0
    assert exact.metrics.critical_leakage_count == 0
    assert exact.metrics.latency.p95_ms == pytest.approx(2.0)
    assert exact.metrics.memory.peak_bytes == 120
    assert exact.metrics.within_budget is True

    weak = report.result("weak")
    assert weak.metrics is not None
    assert weak.metrics.precision == 1.0
    assert weak.metrics.recall == pytest.approx(0.5)
    assert weak.metrics.critical_leakage == 1.0
    assert weak.metrics.critical_leakage_count == 1
    assert weak.metrics.critical_span_count == 1


def test_comparator_report_is_deterministic_and_source_safe() -> None:
    fixture = _fixture()

    def run_once():
        return run_comparator_benchmark(
            [fixture],
            [ComparatorAdapter(name="exact", runner=_exact_runner)],
            metadata={"fixture_note": fixture.text, "owner": "synthetic"},
            clock=_clock([1.0, 1.005]),
            memory_sampler=_sampler([50, 75]),
        )

    first = run_once()
    second = run_once()
    first_json = first.to_json()
    second_json = second.to_json()

    assert first_json == second_json
    assert first.reproducibility_hash == second.reproducibility_hash
    assert fixture.text not in first_json
    assert fixture.text not in first.to_markdown()
    assert fixture.to_dict()["fixture_digest"] not in fixture.to_dict().get("text", "")
    payload = json.loads(first_json)
    assert payload["fixture_digests"]
    assert payload["results"][0]["metrics"]["memory"]["peak_bytes"] == 75


def test_missing_or_network_adapters_are_reported_without_running() -> None:
    called = False

    def should_not_run(text: str, language: str):
        nonlocal called
        called = True
        return _exact_runner(text, language)

    report = run_comparator_benchmark(
        [_fixture()],
        [
            ComparatorAdapter(name="missing"),
            ComparatorAdapter(
                name="remote",
                runner=should_not_run,
                requires_network=True,
            ),
        ],
    )

    assert [row.status for row in report.results] == [
        STATUS_NOT_AVAILABLE,
        STATUS_NOT_AVAILABLE,
    ]
    assert all(row.metrics is None for row in report.results)
    assert called is False
    assert all(
        row.reason == "adapter is not available for this offline run"
        for row in report.results
    )


def test_adapter_errors_do_not_echo_raw_values() -> None:
    secret = "SYNTHETIC-ONLY-IDENTIFIER-0001"

    def broken_runner(text: str, language: str):
        raise RuntimeError(f"unexpected adapter payload: {secret}")

    with pytest.raises(ComparatorExecutionError) as excinfo:
        run_comparator_benchmark(
            [_fixture()],
            [ComparatorAdapter(name="broken", runner=broken_runner)],
        )

    assert secret not in str(excinfo.value)
    assert "unexpected adapter payload" not in str(excinfo.value)


def test_unavailable_exception_is_not_a_failed_benchmark() -> None:
    def unavailable_runner(text: str, language: str):
        raise ComparatorAdapterUnavailable("optional local package is absent")

    report = run_comparator_benchmark(
        [_fixture()],
        [ComparatorAdapter(name="optional", runner=unavailable_runner)],
    )

    assert report.result("optional").status == STATUS_NOT_AVAILABLE
    assert report.result("optional").metrics is None


def _fixture() -> ComparatorFixture:
    text = "Synthetic note for Rowan Vale; code 000-00-0000."
    person_start = text.index("Rowan Vale")
    code_start = text.index("000-00-0000")
    return ComparatorFixture(
        fixture_id="synthetic-001",
        text=text,
        gold_spans=(
            {
                "start": person_start,
                "end": person_start + len("Rowan Vale"),
                "label": "PERSON",
            },
            {
                "start": code_start,
                "end": code_start + len("000-00-0000"),
                "label": "SSN",
            },
        ),
        metadata={"synthetic": True, "phi_free": True},
    )


def _exact_runner(text: str, language: str):
    del language
    person_start = text.index("Rowan Vale")
    code_start = text.index("000-00-0000")
    return (
        {
            "start": person_start,
            "end": person_start + len("Rowan Vale"),
            "label": "PERSON",
        },
        {
            "start": code_start,
            "end": code_start + len("000-00-0000"),
            "label": "SSN",
        },
    )


def _person_only_runner(text: str, language: str):
    del language
    person_start = text.index("Rowan Vale")
    return (
        {
            "start": person_start,
            "end": person_start + len("Rowan Vale"),
            "label": "PERSON",
        },
    )


def _clock(values: list[float]):
    iterator = iter(values)
    last = values[-1]

    def tick() -> float:
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return tick


def _sampler(values: list[int]):
    iterator = iter(values)
    last = values[-1]

    def sample() -> int:
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return sample
