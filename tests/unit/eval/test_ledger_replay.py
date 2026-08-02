"""Tests for benchmark ledger backfill and deterministic replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.history import (
    IMPROVEMENT,
    REGRESSION,
    UNCHANGED,
    BenchmarkRunLedger,
    BenchmarkRunLedgerEntry,
    RunLedgerConflict,
    load_run_ledger,
)
from openmed.eval.ledger_replay import (
    REPLAY_SCHEMA_VERSION,
    backfill_reports,
    entry_from_report,
    main,
    replay_ledger,
)
from openmed.eval.report import BenchmarkReport

MANIFEST_HASH = "sha256:" + "b" * 64


def _report(
    *,
    seed: int,
    leakage: float,
    recall: float,
    generated_at: str,
    manifest_hash: str = MANIFEST_HASH,
) -> BenchmarkReport:
    """Build a synthetic PHI-free benchmark report for one run."""

    return BenchmarkReport(
        suite="golden",
        model_name="OpenMed/pii-small-mlx",
        device="mlx-fp",
        fixture_count=4,
        generated_at=generated_at,
        metrics={
            "leakage": {"overall": leakage},
            "per_label_recall": {"PERSON": recall},
        },
        metadata={
            "family": "PII",
            "manifest_hash": manifest_hash,
            "seed": seed,
        },
    )


def _write_report(path: Path, report: BenchmarkReport) -> Path:
    report.write_json(path)
    return path


def _entry(
    *,
    seed: int,
    leakage: float,
    recall: float,
    generated_at: str,
) -> BenchmarkRunLedgerEntry:
    """Build a synthetic ledger entry keyed by its seed."""

    return BenchmarkRunLedgerEntry(
        model_id="OpenMed/pii-small-mlx",
        suite="golden",
        manifest_hash=MANIFEST_HASH,
        seed=seed,
        generated_at=generated_at,
        fixture_count=4,
        metrics={"leakage.overall": leakage, "per_label_recall.PERSON": recall},
    )


def _series_ledger() -> BenchmarkRunLedger:
    """Build a within-noise baseline window plus a regressed candidate."""

    ledger = BenchmarkRunLedger()
    for index in range(5):
        leakage = 0.010 + 0.0005 * ((index % 3) - 1)  # small symmetric noise
        recall = 0.990 + 0.0005 * (index % 2)
        ledger = ledger.append(
            _entry(
                seed=index,
                leakage=leakage,
                recall=recall,
                generated_at=f"2026-06-1{index}T00:00:00+00:00",
            )
        )
    # Candidate run: leakage jumps up (regression), recall improves.
    ledger = ledger.append(
        _entry(
            seed=99,
            leakage=0.050,
            recall=0.999,
            generated_at="2026-06-20T00:00:00+00:00",
        )
    )
    return ledger


def test_backfill_is_idempotent_and_byte_stable(tmp_path: Path) -> None:
    ledger_path = tmp_path / "runs.json"
    reports = [
        _write_report(
            tmp_path / f"report-{seed}.json",
            _report(
                seed=seed,
                leakage=0.01,
                recall=0.99,
                generated_at=f"2026-06-0{seed}T00:00:00+00:00",
            ),
        )
        for seed in (1, 2, 3)
    ]

    first = backfill_reports(ledger_path, reports)
    first_bytes = ledger_path.read_bytes()
    assert len(first.added) == 3
    assert first.skipped == ()

    second = backfill_reports(ledger_path, reports)
    assert second.added == ()
    assert len(second.skipped) == 3
    assert ledger_path.read_bytes() == first_bytes
    assert len(load_run_ledger(ledger_path).entries) == 3


def test_backfill_conflicting_duplicate_run_key_raises(tmp_path: Path) -> None:
    ledger_path = tmp_path / "runs.json"
    original = _write_report(
        tmp_path / "a.json",
        _report(
            seed=7, leakage=0.01, recall=0.99, generated_at="2026-06-01T00:00:00+00:00"
        ),
    )
    backfill_reports(ledger_path, [original])
    before = ledger_path.read_bytes()

    conflicting = _write_report(
        tmp_path / "b.json",
        _report(
            seed=7, leakage=0.09, recall=0.80, generated_at="2026-06-02T00:00:00+00:00"
        ),
    )
    with pytest.raises(RunLedgerConflict, match="already exists"):
        backfill_reports(ledger_path, [conflicting])

    assert ledger_path.read_bytes() == before


def test_entry_from_report_requires_manifest_hash(tmp_path: Path) -> None:
    report = BenchmarkReport(
        suite="golden",
        model_name="OpenMed/pii-small-mlx",
        device="mlx-fp",
        fixture_count=1,
        generated_at="2026-06-01T00:00:00+00:00",
        metrics={"leakage": {"overall": 0.01}},
        metadata={"seed": 3},
    )
    with pytest.raises(ValueError, match="manifest hash"):
        entry_from_report(report)

    entry = entry_from_report(report, manifest_hash=MANIFEST_HASH)
    assert entry.manifest_hash == MANIFEST_HASH
    assert entry.seed == 3


def test_replay_is_deterministic_and_byte_identical(tmp_path: Path) -> None:
    ledger = _series_ledger()

    first = replay_ledger(ledger, seed=13, n_resamples=256).to_json()
    second = replay_ledger(ledger, seed=13, n_resamples=256).to_json()
    assert first == second

    ledger_path = tmp_path / "runs.json"
    ledger.write_json(ledger_path)
    from_path = replay_ledger(ledger_path, seed=13, n_resamples=256).to_json()
    assert from_path == first


def test_replay_seed_changes_confidence_intervals() -> None:
    ledger = _series_ledger()
    low = replay_ledger(ledger, seed=1, n_resamples=256).to_json()
    high = replay_ledger(ledger, seed=2, n_resamples=256).to_json()
    assert low != high


def test_replay_flags_leakage_regression_and_recall_improvement() -> None:
    report = replay_ledger(_series_ledger(), seed=5, n_resamples=128)
    by_metric = {verdict.metric: verdict for verdict in report.verdicts}

    assert report.schema_version == REPLAY_SCHEMA_VERSION
    leakage = by_metric["leakage.overall"]
    assert leakage.direction == "lower_is_better"
    assert leakage.verdict == REGRESSION
    assert leakage.outside_ci is True

    recall = by_metric["per_label_recall.PERSON"]
    assert recall.direction == "higher_is_better"
    assert recall.verdict == IMPROVEMENT


def test_replay_single_run_group_emits_no_verdict() -> None:
    ledger = BenchmarkRunLedger().append(
        _entry(
            seed=0, leakage=0.01, recall=0.99, generated_at="2026-06-01T00:00:00+00:00"
        )
    )
    assert replay_ledger(ledger, seed=0).verdicts == ()


def test_replay_identical_series_is_unchanged() -> None:
    ledger = BenchmarkRunLedger()
    for index in range(3):
        ledger = ledger.append(
            _entry(
                seed=index,
                leakage=0.02,
                recall=0.98,
                generated_at=f"2026-06-0{index}T00:00:00+00:00",
            )
        )
    report = replay_ledger(ledger, seed=0, n_resamples=64)
    assert {verdict.verdict for verdict in report.verdicts} == {UNCHANGED}


def test_cli_backfill_then_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger_path = tmp_path / "runs.json"
    reports = [
        _write_report(
            tmp_path / f"r{seed}.json",
            _report(
                seed=seed,
                leakage=0.01,
                recall=0.99,
                generated_at=f"2026-06-0{seed}T00:00:00+00:00",
            ),
        )
        for seed in (1, 2)
    ]

    exit_code = main(["backfill", "--ledger", str(ledger_path), *map(str, reports)])
    assert exit_code == 0
    assert "2 added" in capsys.readouterr().out

    out_path = tmp_path / "verdicts.json"
    exit_code = main(
        [
            "replay",
            "--ledger",
            str(ledger_path),
            "--seed",
            "3",
            "--resamples",
            "64",
            "--output",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == REPLAY_SCHEMA_VERSION
    assert payload["seed"] == 3


def test_cli_replay_reports_malformed_ledger_row(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger_path = tmp_path / "broken.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "model_id": "OpenMed/pii-small-mlx",
                        "suite": "golden",
                        "manifest_hash": MANIFEST_HASH,
                        "seed": 0,
                        "metrics": "not-an-object",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["replay", "--ledger", str(ledger_path)])
    assert exit_code == 1
    assert "failed" in capsys.readouterr().err
