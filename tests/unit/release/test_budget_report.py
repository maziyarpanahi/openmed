"""Focused tests for advisory release-budget reporting and CI outputs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from openmed.eval.budget_tracker import (
    BENCHMARK_REFRESH,
    OVER,
    StageTiming,
    write_stage_timings,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "budget_report.py"
SPEC = importlib.util.spec_from_file_location("release_budget_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
budget_report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget_report
SPEC.loader.exec_module(budget_report)


def test_over_budget_warns_in_ci_without_returning_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "run-ci-over"
    timings = tmp_path / "stage-timings.json"
    write_stage_timings(
        (
            StageTiming(
                stage="eval",
                candidate_id="synthetic-gpu",
                family="PII",
                tier="Small",
                workload=BENCHMARK_REFRESH,
                gpu_hours=40,
                wall_clock_seconds=144_000,
            ),
        ),
        run_id=run_id,
        path=timings,
    )
    release_ledger = tmp_path / "release-runs.jsonl"
    release_ledger.write_text(
        json.dumps({"record_type": "nightly-release", "run_id": run_id}) + "\n",
        encoding="utf-8",
    )
    budget_ledger = tmp_path / "budget-ledger.jsonl"
    report = tmp_path / "budget-report.json"
    github_output = tmp_path / "github-output.txt"

    result = budget_report.main(
        [
            "record",
            "--timings",
            str(timings),
            "--ledger",
            str(budget_ledger),
            "--orchestrator-ledger",
            str(release_ledger),
            "--recorded-at",
            "2026-08-04T12:00:00Z",
            "--output",
            str(report),
            "--github-output",
            str(github_output),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "::warning title=Advisory release budget OVER::" in captured.out
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["advisory"] is True
    assert payload["run"]["verdict"] == OVER
    assert payload["run"]["per_run_budget"]["gating"] is False
    assert payload["run"]["orchestrator_run_id"] == run_id
    outputs = github_output.read_text(encoding="utf-8")
    assert "verdict=OVER" in outputs
    assert "throttle_recommended=true" in outputs


def test_rolling_over_budget_can_select_optional_queue_throttle(
    tmp_path: Path,
) -> None:
    run_id = "run-weekly-over"
    timings = tmp_path / "stage-timings.json"
    write_stage_timings(
        (
            StageTiming(
                stage="eval",
                candidate_id="synthetic-gpu",
                family="PII",
                tier="Small",
                workload=BENCHMARK_REFRESH,
                gpu_hours=40,
                wall_clock_seconds=144_000,
            ),
        ),
        run_id=run_id,
        path=timings,
    )
    release_ledger = tmp_path / "release-runs.jsonl"
    release_ledger.write_text(
        json.dumps({"record_type": "nightly-release", "run_id": run_id}) + "\n",
        encoding="utf-8",
    )
    budget_ledger = tmp_path / "budget-ledger.jsonl"
    assert (
        budget_report.main(
            [
                "record",
                "--timings",
                str(timings),
                "--ledger",
                str(budget_ledger),
                "--orchestrator-ledger",
                str(release_ledger),
                "--recorded-at",
                "2026-08-04T12:00:00Z",
            ]
        )
        == 0
    )
    github_output = tmp_path / "preflight-output.txt"

    result = budget_report.main(
        [
            "status",
            "--ledger",
            str(budget_ledger),
            "--as-of",
            "2026-08-04T13:00:00Z",
            "--throttle-on-over",
            "--github-output",
            str(github_output),
        ]
    )

    assert result == 0
    outputs = github_output.read_text(encoding="utf-8")
    assert "verdict=OVER" in outputs
    assert "max_candidates=1" in outputs
