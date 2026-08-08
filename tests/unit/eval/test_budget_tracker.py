"""Focused synthetic tests for the release budget tracker."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openmed.eval.budget_tracker import (
    BENCHMARK_REFRESH,
    OVER,
    TRAINING,
    BudgetPolicy,
    BudgetThresholds,
    BudgetTrackingError,
    CostCarbonFactors,
    StageTiming,
    StageTimingDocument,
    append_budget_entry,
    build_budget_entry,
    load_budget_ledger,
    load_stage_timings,
    rolling_budget_breakdown,
    rolling_weekly_totals,
)
from openmed.eval.fleet_metrics import (
    compute_fleet_budget_metrics,
    compute_fleet_budget_metrics_from_ledger,
)

AS_OF = datetime(2026, 8, 4, 12, tzinfo=timezone.utc)
FACTORS = CostCarbonFactors(
    gpu_cost_per_hour_usd=2.0,
    runner_cost_per_minute_usd=0.1,
    gpu_power_kw=0.5,
    runner_power_kw=0.1,
    carbon_intensity_kg_per_kwh=0.4,
)


def _threshold(cost: float) -> BudgetThresholds:
    return BudgetThresholds(
        estimated_cost_usd=cost,
        carbon_kg=1_000,
        energy_kwh=1_000,
        gpu_hours=1_000,
        runner_minutes=10_000,
        wall_clock_seconds=1_000_000,
    )


def _policy(*, per_run_warn: float, per_run_over: float) -> BudgetPolicy:
    return BudgetPolicy(
        per_run_warn=_threshold(per_run_warn),
        per_run_over=_threshold(per_run_over),
        weekly_warn=_threshold(1_000),
        weekly_over=_threshold(2_000),
    )


def _stage(
    candidate_id: str,
    family: str,
    tier: str,
    *,
    runner_minutes: float,
    workload: str = BENCHMARK_REFRESH,
) -> StageTiming:
    return StageTiming(
        stage="eval",
        candidate_id=candidate_id,
        family=family,
        tier=tier,
        workload=workload,
        runner_minutes=runner_minutes,
        wall_clock_seconds=runner_minutes * 60,
    )


def test_aggregation_is_deterministic_and_splits_training_from_refresh() -> None:
    stages = (
        StageTiming(
            stage="train",
            candidate_id="ner-base",
            family="NER",
            tier="Base",
            workload=TRAINING,
            gpu_hours=2,
            wall_clock_seconds=7_200,
        ),
        StageTiming(
            stage="eval",
            candidate_id="pii-small",
            family="PII",
            tier="Small",
            workload=BENCHMARK_REFRESH,
            gpu_hours=1,
            wall_clock_seconds=3_600,
        ),
        StageTiming(
            stage="quantize",
            candidate_id="pii-small",
            family="PII",
            tier="Small",
            workload=BENCHMARK_REFRESH,
            runner_minutes=30,
            wall_clock_seconds=1_800,
        ),
    )

    first = build_budget_entry(
        run_id="run-aggregation",
        stages=stages,
        recorded_at=AS_OF,
        factors=FACTORS,
    )
    replay = build_budget_entry(
        run_id="run-aggregation",
        stages=reversed(stages),
        recorded_at=AS_OF,
        factors=FACTORS,
    )

    assert first.totals.to_dict() == replay.totals.to_dict()
    assert first.aggregation_hash == replay.aggregation_hash
    assert first.record_hash == replay.record_hash
    assert first.totals.gpu_hours == 3.0
    assert first.totals.runner_minutes == 30.0
    assert first.totals.wall_clock_seconds == 12_600.0
    assert first.totals.energy_kwh == 1.55
    assert first.totals.carbon_kg == 0.62
    assert first.totals.estimated_cost_usd == 9.0
    assert first.by_workload[TRAINING].estimated_cost_usd == 4.0
    assert first.by_workload[BENCHMARK_REFRESH].estimated_cost_usd == 5.0


def test_per_run_budget_excess_has_advisory_over_verdict() -> None:
    entry = build_budget_entry(
        run_id="run-over",
        stages=(_stage("pii-small", "PII", "Small", runner_minutes=90),),
        recorded_at=AS_OF,
        factors=FACTORS,
        policy=_policy(per_run_warn=4, per_run_over=8),
    )

    assert entry.totals.estimated_cost_usd == 9.0
    assert entry.per_run_budget.verdict == OVER
    assert entry.per_run_budget.exceeded_metrics == ("estimated_cost_usd",)
    assert entry.per_run_budget.gating is False
    assert entry.verdict == OVER
    assert entry.throttle_recommended is True


def test_ledger_requires_orchestrator_link_and_excludes_raw_phi(tmp_path: Path) -> None:
    timings = StageTimingDocument(
        run_id="run-safe",
        orchestrator_run_id="run-safe",
        stages=(_stage("synthetic-one", "PII", "Small", runner_minutes=1),),
    )
    timings_path = timings.write_json(tmp_path / "stage-timings.json")
    assert load_stage_timings(timings_path) == timings

    entry = build_budget_entry(
        run_id=timings.run_id,
        stages=timings.stages,
        recorded_at=AS_OF,
        factors=FACTORS,
    )
    release_ledger = tmp_path / "release-runs.jsonl"
    release_ledger.write_text(
        json.dumps({"record_type": "nightly-release", "run_id": "run-safe"}) + "\n",
        encoding="utf-8",
    )
    budget_ledger = tmp_path / "budget-ledger.jsonl"

    append_budget_entry(
        entry,
        ledger_path=budget_ledger,
        orchestrator_ledger_path=release_ledger,
    )

    (loaded,) = load_budget_ledger(budget_ledger)
    assert loaded.run_id == loaded.orchestrator_run_id == "run-safe"
    persisted = budget_ledger.read_text(encoding="utf-8")
    assert "Synthetic Patient" not in persisted
    assert "123-45-6789" not in persisted

    unsafe_ledger = tmp_path / "unsafe-budget-ledger.jsonl"
    unsafe_payload = entry.to_dict()
    unsafe_payload["raw_text"] = "Synthetic Patient Example"
    unsafe_ledger.write_text(json.dumps(unsafe_payload) + "\n", encoding="utf-8")
    with pytest.raises(BudgetTrackingError, match="unsupported fields"):
        load_budget_ledger(unsafe_ledger)

    inconsistent_ledger = tmp_path / "inconsistent-budget-ledger.jsonl"
    inconsistent_payload = entry.to_dict()
    inconsistent_payload["totals"]["estimated_cost_usd"] = 999
    inconsistent_ledger.write_text(
        json.dumps(inconsistent_payload) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BudgetTrackingError, match="committed stage timings"):
        load_budget_ledger(inconsistent_ledger)

    with pytest.raises(BudgetTrackingError, match="safe identifier"):
        StageTiming(
            stage="eval",
            candidate_id="patient@example.com",
            family="PII",
            tier="Small",
            workload=BENCHMARK_REFRESH,
            wall_clock_seconds=1,
        )

    with pytest.raises(BudgetTrackingError, match="PHI-shaped"):
        StageTimingDocument(
            run_id="123-45-6789",
            orchestrator_run_id="123-45-6789",
            stages=(_stage("synthetic-three", "PII", "Small", runner_minutes=1),),
        )

    with pytest.raises(BudgetTrackingError, match="PHI-shaped"):
        build_budget_entry(
            run_id="123-45-6789",
            stages=(_stage("synthetic-four", "PII", "Small", runner_minutes=1),),
        )

    missing_link = build_budget_entry(
        run_id="run-missing",
        stages=(_stage("synthetic-two", "PII", "Small", runner_minutes=1),),
        recorded_at=AS_OF,
    )
    with pytest.raises(BudgetTrackingError, match="absent from release ledger"):
        append_budget_entry(
            missing_link,
            ledger_path=tmp_path / "other-budget-ledger.jsonl",
            orchestrator_ledger_path=release_ledger,
        )


def test_rolling_weekly_and_family_cost_match_committed_entries(
    tmp_path: Path,
) -> None:
    old = build_budget_entry(
        run_id="run-old",
        stages=(_stage("old", "PII", "Small", runner_minutes=50),),
        recorded_at="2026-07-20T12:00:00Z",
        factors=FACTORS,
    )
    recent_pii = build_budget_entry(
        run_id="run-pii",
        stages=(_stage("pii", "PII", "Small", runner_minutes=10),),
        history=(old,),
        recorded_at="2026-08-01T12:00:00Z",
        factors=FACTORS,
    )
    recent_ner = build_budget_entry(
        run_id="run-ner",
        stages=(_stage("ner", "NER", "Base", runner_minutes=20),),
        history=(old, recent_pii),
        recorded_at=AS_OF,
        factors=FACTORS,
    )
    release_ledger = tmp_path / "release-runs.jsonl"
    release_ledger.write_text(
        "".join(
            json.dumps({"record_type": "nightly-release", "run_id": run_id}) + "\n"
            for run_id in ("run-old", "run-pii", "run-ner")
        ),
        encoding="utf-8",
    )
    budget_ledger = tmp_path / "budget-ledger.jsonl"
    for entry in (old, recent_pii, recent_ner):
        append_budget_entry(
            entry,
            ledger_path=budget_ledger,
            orchestrator_ledger_path=release_ledger,
        )
    entries = load_budget_ledger(budget_ledger)

    totals = rolling_weekly_totals(entries, as_of=AS_OF)
    by_family = rolling_budget_breakdown(
        entries,
        dimension="family",
        as_of=AS_OF,
    )
    fleet = compute_fleet_budget_metrics(entries, as_of=AS_OF)
    fleet_from_ledger = compute_fleet_budget_metrics_from_ledger(
        budget_ledger,
        as_of=AS_OF,
    )

    assert totals.estimated_cost_usd == 3.0
    assert by_family["PII"].estimated_cost_usd == 1.0
    assert by_family["NER"].estimated_cost_usd == 2.0
    assert sum(item.estimated_cost_usd for item in by_family.values()) == 3.0
    assert fleet.run_count == 2
    assert fleet.totals.to_dict() == totals.to_dict()
    assert fleet_from_ledger.to_dict() == fleet.to_dict()
    assert fleet.to_dict()["queue_cost_priority"][0]["family_tier"] == "PII/Small"
