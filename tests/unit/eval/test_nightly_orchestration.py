"""Tests for nightly benchmark orchestration and gate-trend emission.

All inputs are synthetic and generated algorithmically; no real PHI is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.baseline import BASELINE_PATH
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.history import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    RunLedgerConflict,
    load_run_ledger,
)
from openmed.eval.nightly import (
    GATE_FAIL,
    GATE_PASS,
    NIGHTLY_SCHEMA_VERSION,
    VERDICT_BLOCKED,
    VERDICT_RELEASABLE,
    GateOutcome,
    NightlyGate,
    benchmark_suite_runner,
    run_nightly,
)
from openmed.eval.release_gates import ReleaseGate
from openmed.eval.report import BenchmarkReport

_GENERATED_AT = "2026-07-30T00:00:00+00:00"
_MANIFEST = "sha256:" + "a" * 64
_MODEL = "synthetic-model"
_SUITE = "synthetic-pii"


def _report(
    *, leakage: float, f1: float, generated_at: str = _GENERATED_AT
) -> BenchmarkReport:
    """Build a deterministic synthetic benchmark report."""

    return BenchmarkReport(
        suite=_SUITE,
        model_name=_MODEL,
        device="cpu",
        fixture_count=4,
        metrics={
            "leakage": {"overall": leakage},
            "exact_span_f1": {"f1": f1},
        },
        generated_at=generated_at,
        metadata={},
    )


def _seed_runner(reports_by_seed):
    """Return a per-seed runner backed by a fixed mapping."""

    def _run(seed: int) -> BenchmarkReport:
        return reports_by_seed[seed]

    return _run


def _leakage_gate(threshold: float = 0.05, *, blocking: bool = True) -> NightlyGate:
    return NightlyGate(
        name="leakage",
        metric="leakage.overall",
        threshold=threshold,
        direction=LOWER_IS_BETTER,
        blocking=blocking,
    )


def _span_f1_gate(threshold: float = 0.90, *, blocking: bool = False) -> NightlyGate:
    return NightlyGate(
        name="span_f1",
        metric="exact_span_f1.f1",
        threshold=threshold,
        direction=HIGHER_IS_BETTER,
        blocking=blocking,
    )


def _run(tmp_path: Path, reports_by_seed, gates, **kwargs):
    ledger_path = tmp_path / "gates" / "nightly_ledger.json"
    return run_nightly(
        _seed_runner(reports_by_seed),
        seeds=sorted(reports_by_seed),
        gates=gates,
        manifest_hash=_MANIFEST,
        ledger_path=ledger_path,
        generated_at=_GENERATED_AT,
        **kwargs,
    )


def test_run_nightly_runs_offline_over_synthetic_fixtures(tmp_path: Path):
    """The orchestrator runs offline over committed-style fixtures via the harness."""

    fixtures = [
        BenchmarkFixture(
            fixture_id="f1", text="Amina visited the clinic.", gold_spans=()
        ),
        BenchmarkFixture(
            fixture_id="f2", text="Jordan took amoxicillin.", gold_spans=()
        ),
    ]
    runner = benchmark_suite_runner(
        fixtures,
        suite=_SUITE,
        model_name=_MODEL,
        model_runner=lambda fixture, model_name, device: [],
        generated_at=_GENERATED_AT,
    )
    ledger_path = tmp_path / "gates" / "nightly_ledger.json"
    result = run_nightly(
        runner,
        seeds=[0, 1],
        gates=[_leakage_gate(threshold=1.0), _span_f1_gate(threshold=0.0)],
        manifest_hash=_MANIFEST,
        ledger_path=ledger_path,
        generated_at=_GENERATED_AT,
    )

    assert ledger_path.exists()
    assert result.verdict.suite == _SUITE
    assert result.verdict.model_id == _MODEL
    assert result.verdict.seeds == (0, 1)
    # Both gates are permissive, so the night is releasable.
    assert result.verdict.verdict == VERDICT_RELEASABLE
    assert len(result.verdict.trend) == 2


def test_ledger_and_trend_artifacts_are_byte_stable(tmp_path: Path):
    """Two independent runs on fixed inputs reproduce identical bytes."""

    reports = {0: _report(leakage=0.02, f1=0.95), 1: _report(leakage=0.03, f1=0.94)}
    gates = [_leakage_gate(), _span_f1_gate()]

    first = _run(tmp_path / "a", reports, gates)
    second = _run(tmp_path / "b", reports, gates)

    assert first.verdict.to_json() == second.verdict.to_json()
    assert first.verdict.to_markdown() == second.verdict.to_markdown()
    assert first.ledger_path.read_bytes() == second.ledger_path.read_bytes()
    # Idempotent within a single serialization call too.
    assert first.verdict.to_json() == first.verdict.to_json()
    assert first.verdict.to_markdown().encode(
        "utf-8"
    ) == first.verdict.to_markdown().encode("utf-8")


def test_ledger_append_is_idempotent_and_preserves_prior_rows(tmp_path: Path):
    """Re-running identical inputs never mutates or grows the ledger."""

    reports = {0: _report(leakage=0.02, f1=0.95), 1: _report(leakage=0.03, f1=0.94)}
    gates = [_leakage_gate()]
    ledger_path = tmp_path / "gates" / "nightly_ledger.json"

    run_nightly(
        _seed_runner(reports),
        seeds=[0, 1],
        gates=gates,
        manifest_hash=_MANIFEST,
        ledger_path=ledger_path,
        generated_at=_GENERATED_AT,
    )
    first_bytes = ledger_path.read_bytes()
    first_count = len(load_run_ledger(ledger_path).entries)

    run_nightly(
        _seed_runner(reports),
        seeds=[0, 1],
        gates=gates,
        manifest_hash=_MANIFEST,
        ledger_path=ledger_path,
        generated_at=_GENERATED_AT,
    )
    assert ledger_path.read_bytes() == first_bytes
    assert len(load_run_ledger(ledger_path).entries) == first_count == 2


def test_conflicting_rerun_raises(tmp_path: Path):
    """A different payload for an existing run key raises a ledger conflict."""

    ledger_path = tmp_path / "gates" / "nightly_ledger.json"
    run_nightly(
        _seed_runner({0: _report(leakage=0.02, f1=0.95)}),
        seeds=[0],
        gates=[_leakage_gate()],
        manifest_hash=_MANIFEST,
        ledger_path=ledger_path,
        generated_at=_GENERATED_AT,
    )
    with pytest.raises(RunLedgerConflict):
        run_nightly(
            _seed_runner({0: _report(leakage=0.09, f1=0.10)}),
            seeds=[0],
            gates=[_leakage_gate()],
            manifest_hash=_MANIFEST,
            ledger_path=ledger_path,
            generated_at=_GENERATED_AT,
        )


def test_blocking_leakage_failure_blocks_verdict(tmp_path: Path):
    """A blocking leakage gate failure quarantines the night."""

    reports = {0: _report(leakage=0.02, f1=0.95), 1: _report(leakage=0.08, f1=0.95)}
    result = _run(tmp_path, reports, [_leakage_gate(threshold=0.05), _span_f1_gate()])

    assert result.verdict.verdict == VERDICT_BLOCKED
    assert result.verdict.blocking_failures == ("leakage",)
    assert result.verdict.advisory_failures == ()
    assert result.verdict.releasable is False


def test_advisory_failure_does_not_block_verdict(tmp_path: Path):
    """An advisory F1 gate failure is recorded but does not block."""

    reports = {0: _report(leakage=0.01, f1=0.95), 1: _report(leakage=0.01, f1=0.80)}
    result = _run(
        tmp_path,
        reports,
        [_leakage_gate(threshold=0.05), _span_f1_gate(threshold=0.90)],
    )

    assert result.verdict.verdict == VERDICT_RELEASABLE
    assert result.verdict.blocking_failures == ()
    assert result.verdict.advisory_failures == ("span_f1",)


def test_gate_trend_reflects_pass_fail_history(tmp_path: Path):
    """The trend surfaces per-run pass/fail for each gate in seed order."""

    reports = {
        0: _report(leakage=0.02, f1=0.95),
        1: _report(leakage=0.03, f1=0.95),
        2: _report(leakage=0.08, f1=0.95),
    }
    result = _run(tmp_path, reports, [_leakage_gate(threshold=0.05)])

    statuses = [point.status("leakage") for point in result.verdict.trend]
    assert statuses == [GATE_PASS, GATE_PASS, GATE_FAIL]
    seeds = [point.seed for point in result.verdict.trend]
    assert seeds == [0, 1, 2]


def test_trend_window_limits_recent_runs(tmp_path: Path):
    """Only the trailing ``trend_window`` rows are surfaced in the trend."""

    reports = {seed: _report(leakage=0.01, f1=0.95) for seed in range(5)}
    result = _run(tmp_path, reports, [_leakage_gate()], trend_window=3)

    assert len(result.verdict.trend) == 3
    assert [point.seed for point in result.verdict.trend] == [2, 3, 4]


def test_missing_metric_fails_gate():
    """A gate over an absent metric fails deterministically."""

    gate = NightlyGate(name="missing", metric="not.present", threshold=1.0)
    outcome = gate.evaluate({"leakage.overall": 0.1})
    assert isinstance(outcome, GateOutcome)
    assert outcome.status == GATE_FAIL
    assert outcome.value is None


def test_verdict_payload_carries_schema_version(tmp_path: Path):
    """The serialized verdict declares the nightly schema version."""

    reports = {0: _report(leakage=0.01, f1=0.95)}
    result = _run(tmp_path, reports, [_leakage_gate()])
    payload = json.loads(result.verdict.to_json())
    assert payload["schema_version"] == NIGHTLY_SCHEMA_VERSION
    assert payload["verdict"] == VERDICT_RELEASABLE


def test_markdown_is_self_describing_and_phi_free(tmp_path: Path):
    """The Markdown trend summary is human-readable and leaks no raw text."""

    reports = {0: _report(leakage=0.02, f1=0.95), 1: _report(leakage=0.08, f1=0.95)}
    result = _run(tmp_path, reports, [_leakage_gate(threshold=0.05), _span_f1_gate()])
    markdown = result.verdict.to_markdown()

    assert markdown.startswith("# Nightly Benchmark Trend: synthetic-pii")
    assert markdown.endswith("\n")
    assert "## Gate Trend" in markdown
    assert "## Blocking Failures" in markdown
    assert "`leakage`" in markdown
    # No raw fixture prose ever reaches the artifact.
    assert "Amina" not in markdown
    assert "amoxicillin" not in markdown


def test_release_gate_consumes_verdict_without_mutating_baseline(tmp_path: Path):
    """The release gate consumes the verdict and never mutates the baseline store."""

    baseline_src = BASELINE_PATH
    baseline_copy = tmp_path / "baseline.json"
    baseline_copy.write_bytes(baseline_src.read_bytes())
    before = baseline_copy.read_bytes()

    reports = {0: _report(leakage=0.01, f1=0.95), 1: _report(leakage=0.01, f1=0.95)}
    result = _run(tmp_path, reports, [_leakage_gate(), _span_f1_gate()])

    gate = ReleaseGate(baseline_path=baseline_copy)
    gate_report = gate.preview(result.verdict.to_benchmark_report())

    assert gate_report.decision in {"RELEASABLE", "QUARANTINED"}
    assert baseline_copy.read_bytes() == before


def test_run_nightly_requires_seeds_and_gates(tmp_path: Path):
    """Empty seed or gate sets are rejected."""

    ledger_path = tmp_path / "gates" / "nightly_ledger.json"
    with pytest.raises(ValueError):
        run_nightly(
            _seed_runner({}),
            seeds=[],
            gates=[_leakage_gate()],
            manifest_hash=_MANIFEST,
            ledger_path=ledger_path,
        )
    with pytest.raises(ValueError):
        run_nightly(
            _seed_runner({0: _report(leakage=0.01, f1=0.95)}),
            seeds=[0],
            gates=[],
            manifest_hash=_MANIFEST,
            ledger_path=ledger_path,
        )


def test_duplicate_gate_names_rejected(tmp_path: Path):
    """Gate names must be unique within a nightly run."""

    ledger_path = tmp_path / "gates" / "nightly_ledger.json"
    with pytest.raises(ValueError):
        run_nightly(
            _seed_runner({0: _report(leakage=0.01, f1=0.95)}),
            seeds=[0],
            gates=[_leakage_gate(), _leakage_gate(threshold=0.9)],
            manifest_hash=_MANIFEST,
            ledger_path=ledger_path,
        )


def test_invalid_gate_direction_rejected():
    """A gate with an unknown direction is rejected at construction."""

    with pytest.raises(ValueError):
        NightlyGate(name="bad", metric="x", threshold=1.0, direction="sideways")
