"""Tests for repeated-run flaky-eval detection."""

from __future__ import annotations

import pytest

from openmed.eval import detect_flaky_eval as exported_detect_flaky_eval
from openmed.eval.flaky import (
    DEFAULT_FLAKY_TOLERANCE,
    NondeterministicGateRunError,
    detect_flaky_eval,
    detect_gate_suite_flakiness,
)
from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
    apply_flakiness_quarantine,
)
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


def test_seed_sweep_quarantines_synthetic_gate_that_flips_release_verdict() -> None:
    stability = detect_gate_suite_flakiness(
        lambda seed: _gate_report(g1a_passed=seed % 2 == 0),
        seeds=(0, 1, 2, 3, 4, 5),
    )

    assert stability.gate("G1a").unstable is True
    assert stability.gate("G1a").flip_rate == pytest.approx(0.5)
    assert stability.quarantined_gates == ("G1a",)

    blocked = apply_flakiness_quarantine(
        _gate_report(g1a_passed=True),
        stability,
    )

    assert blocked.decision == QUARANTINED
    flakiness_check = next(
        check for check in blocked.gate_results if check.gate == "flakiness"
    )
    assert flakiness_check.passed is False
    assert "G1a" in flakiness_check.reason
    assert blocked.stability_summary["quarantined_gates"] == ["G1a"]


def test_same_seed_nondeterminism_probe_fails_loudly_on_drift() -> None:
    calls = 0

    def drifting_gate(seed: int) -> GateReport:
        nonlocal calls
        calls += 1
        return _gate_report(g1a_passed=calls % 2 == 0)

    with pytest.raises(NondeterministicGateRunError, match="same-seed"):
        detect_gate_suite_flakiness(drifting_gate, seeds=(17, 18))


def test_gate_flip_rate_estimates_and_bounds_are_deterministic() -> None:
    seeds = (0, 1, 2, 3, 4, 5)

    first = detect_gate_suite_flakiness(
        lambda seed: _gate_report(g1a_passed=seed in {0, 2, 4}),
        seeds=seeds,
    )
    second = detect_gate_suite_flakiness(
        lambda seed: _gate_report(g1a_passed=seed in {0, 2, 4}),
        seeds=seeds,
    )

    assert first.to_dict() == second.to_dict()
    g1a = first.to_dict()["gates"]["G1a"]
    assert g1a["flip_rate"] == pytest.approx(0.5)
    assert g1a["flip_rate_bounds"]["lower"] == pytest.approx(
        second.to_dict()["gates"]["G1a"]["flip_rate_bounds"]["lower"]
    )
    assert g1a["flip_rate_bounds"]["upper"] == pytest.approx(
        second.to_dict()["gates"]["G1a"]["flip_rate_bounds"]["upper"]
    )


def test_stable_gate_is_not_quarantined_across_twenty_seed_sweep() -> None:
    stability = detect_gate_suite_flakiness(
        lambda seed: _gate_report(g1a_passed=True, g1a_score=0.99),
        seeds=tuple(range(20)),
    )

    assert stability.nondeterminism_probe is not None
    assert stability.nondeterminism_probe.stable is True
    assert stability.stable is True
    assert stability.quarantined_gates == ()
    assert stability.gate("G1a").stable is True
    assert stability.gate("G1a").metric_variance["score"] == pytest.approx(0.0)


def test_quarantine_ledger_persists_until_stability_window_is_satisfied(
    tmp_path,
) -> None:
    ledger_path = tmp_path / "flakiness-ledger.json"

    unstable = detect_gate_suite_flakiness(
        lambda seed: _gate_report(g1a_passed=seed % 2 == 0),
        seeds=(0, 1, 2, 3),
        ledger_path=ledger_path,
        stability_window=2,
    )
    first_stable = detect_gate_suite_flakiness(
        lambda seed: _gate_report(g1a_passed=True),
        seeds=(0, 1, 2, 3),
        ledger_path=ledger_path,
        stability_window=2,
    )
    second_stable = detect_gate_suite_flakiness(
        lambda seed: _gate_report(g1a_passed=True),
        seeds=(0, 1, 2, 3),
        ledger_path=ledger_path,
        stability_window=2,
    )

    assert unstable.quarantined_gates == ("G1a",)
    assert first_stable.gate("G1a").unstable is False
    assert first_stable.quarantined_gates == ("G1a",)
    assert first_stable.gate("G1a").reason == "awaiting stability window 1/2"
    assert second_stable.quarantined_gates == ()
    assert (
        second_stable.gate("G1a").ledger_state["reason"] == "stability window satisfied"
    )


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


def _gate_report(
    *,
    g1a_passed: bool,
    g1a_score: float = 0.99,
) -> GateReport:
    return GateReport(
        repo_id="OpenMed/unit-model",
        family="PII",
        tier="Tiny",
        param_count=44_000_000,
        format="mlx-fp",
        per_label_recall={"PERSON": 0.995},
        per_label_precision={"PERSON": 0.98},
        critical_leakage_count=0,
        residual_leakage_rate=0.0,
        quant_recall_delta=0.0,
        p50_ms=50.0,
        p95_ms=120.0,
        ram_mb=128.0,
        eval_set_hash="sha256:eval",
        leakage_fixture_hash="sha256:leakage",
        decision=RELEASABLE if g1a_passed else QUARANTINED,
        gate_results=(
            GateCheck(
                "G1a",
                g1a_passed,
                reason="ok" if g1a_passed else "synthetic flip",
                details={"score": g1a_score},
            ),
            GateCheck("G2", True, details={"score": 0.99}),
        ),
    )
