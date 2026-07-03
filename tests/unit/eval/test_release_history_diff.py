"""Tests for cross-release benchmark history diffing."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.core.baseline import write_baseline_store
from openmed.eval.history import (
    IMPROVEMENT,
    REGRESSION,
    UNCHANGED,
    BenchmarkRunLedger,
    BenchmarkRunLedgerEntry,
    RunLedgerConflict,
    append_run_to_ledger,
    diff_against_baseline,
    load_run_ledger,
    metric_history,
)
from openmed.eval.report import BenchmarkReport


def _report(
    *,
    leakage: float,
    recall: float,
    p95_ms: float,
    generated_at: str = "2026-06-14T00:00:00+00:00",
    released: str = "2026-06-14",
) -> BenchmarkReport:
    return BenchmarkReport(
        suite="golden",
        model_name="OpenMed/pii-small-mlx",
        device="mlx-fp",
        fixture_count=2,
        generated_at=generated_at,
        metrics={
            "leakage": {"overall": leakage},
            "latency": {"p95_ms": p95_ms},
            "per_label_recall": {"PERSON": recall},
        },
        metadata={
            "family": "PII",
            "tier": "Small",
            "format": "mlx-fp",
            "released": released,
        },
    )


def _baseline_entry(
    *,
    leakage: float = 0.01,
    recall: float = 0.99,
    p95_ms: float = 120.0,
) -> dict[str, object]:
    return {
        "key": "pii::small::mlx-fp",
        "family": "PII",
        "tier": "Small",
        "format": "mlx-fp",
        "metrics": {
            "leakage": {"overall": leakage},
            "latency": {"p95_ms": p95_ms},
            "per_label_recall": {"PERSON": recall},
        },
        "reproducibility_hash": "sha256:" + "a" * 64,
        "released": "2026-06-01",
    }


def _baseline_store() -> dict[str, object]:
    return {"schema_version": 1, "entries": {"pii::small::mlx-fp": _baseline_entry()}}


def _ledger_entry(
    *,
    seed: int = 7,
    leakage: float = 0.01,
    recall: float = 0.99,
    generated_at: str = "2026-06-14T00:00:00+00:00",
) -> BenchmarkRunLedgerEntry:
    return BenchmarkRunLedgerEntry(
        model_id="OpenMed/pii-small-mlx",
        suite="golden",
        manifest_hash="sha256:" + "b" * 64,
        seed=seed,
        generated_at=generated_at,
        fixture_count=2,
        metrics={
            "leakage.overall": leakage,
            "per_label_recall.PERSON": recall,
        },
        metadata={"device": "mlx-fp"},
    )


def test_degraded_current_report_flags_regressions_and_ranks_them(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    write_baseline_store(_baseline_store(), baseline_path)

    diff = diff_against_baseline(
        _report(leakage=0.025, recall=0.981, p95_ms=160.0),
        baseline_path=baseline_path,
    )

    assert diff.baseline_key == "pii::small::mlx-fp"
    assert diff.metrics["leakage.overall"].verdict == REGRESSION
    assert diff.metrics["per_label_recall.PERSON"].verdict == REGRESSION
    assert diff.metrics["latency.p95_ms"].verdict == REGRESSION
    assert [delta.metric for delta in diff.largest_regressions] == [
        "latency.p95_ms",
        "leakage.overall",
        "per_label_recall.PERSON",
    ]


def test_improved_current_report_flags_improvements() -> None:
    diff = diff_against_baseline(
        _report(leakage=0.0, recall=0.995, p95_ms=90.0),
        _baseline_entry(),
    )

    assert diff.metrics["leakage.overall"].verdict == IMPROVEMENT
    assert diff.metrics["per_label_recall.PERSON"].verdict == IMPROVEMENT
    assert diff.metrics["latency.p95_ms"].verdict == IMPROVEMENT
    assert {delta.metric for delta in diff.largest_improvements} == {
        "latency.p95_ms",
        "leakage.overall",
        "per_label_recall.PERSON",
    }


def test_identical_reports_show_zero_delta() -> None:
    diff = diff_against_baseline(
        _report(leakage=0.01, recall=0.99, p95_ms=120.0),
        _baseline_entry(),
    )

    assert all(delta.verdict == UNCHANGED for delta in diff.metrics.values())
    assert all(delta.delta == pytest.approx(0.0) for delta in diff.metrics.values())
    assert diff.largest_regressions == ()
    assert diff.largest_improvements == ()


def test_metric_history_returns_values_in_release_order() -> None:
    reports = [
        _report(
            leakage=0.03,
            recall=0.98,
            p95_ms=150.0,
            generated_at="2026-06-01T00:00:00+00:00",
            released="2026-06-01",
        ),
        _report(
            leakage=0.02,
            recall=0.985,
            p95_ms=130.0,
            generated_at="2026-06-08T00:00:00+00:00",
            released="2026-06-08",
        ),
        _report(
            leakage=0.01,
            recall=0.99,
            p95_ms=120.0,
            generated_at="2026-06-14T00:00:00+00:00",
            released="2026-06-14",
        ),
    ]

    history = metric_history(reports, "leakage.overall")

    assert [point.value for point in history] == [0.03, 0.02, 0.01]
    assert [point.index for point in history] == [0, 1, 2]
    assert [point.release for point in history] == [
        "2026-06-01",
        "2026-06-08",
        "2026-06-14",
    ]


def test_run_ledger_append_is_idempotent_and_byte_stable(tmp_path: Path) -> None:
    ledger_path = tmp_path / "run-ledger.json"
    entry = _ledger_entry()

    first = append_run_to_ledger(ledger_path, entry)
    first_bytes = ledger_path.read_bytes()
    second = append_run_to_ledger(
        ledger_path,
        BenchmarkRunLedgerEntry.from_mapping(entry.to_dict()),
    )

    assert second.entries == first.entries == (entry,)
    assert ledger_path.read_bytes() == first_bytes
    assert load_run_ledger(ledger_path).entries == (entry,)


def test_run_ledger_rejects_conflicting_duplicate_key(tmp_path: Path) -> None:
    ledger_path = tmp_path / "run-ledger.json"
    append_run_to_ledger(ledger_path, _ledger_entry())
    before = ledger_path.read_bytes()

    with pytest.raises(RunLedgerConflict, match="already exists"):
        append_run_to_ledger(ledger_path, _ledger_entry(leakage=0.02))

    assert ledger_path.read_bytes() == before


def test_run_ledger_preserves_prior_row_order() -> None:
    first = _ledger_entry(seed=1, generated_at="2026-06-14T00:00:00+00:00")
    second = _ledger_entry(seed=2, generated_at="2026-06-15T00:00:00+00:00")

    ledger = BenchmarkRunLedger().append(first).append(second)
    updated = ledger.append(first)

    assert updated is ledger
    assert [entry.key for entry in updated.entries] == [first.key, second.key]


def test_run_ledger_entry_from_report_flattens_numeric_metrics() -> None:
    entry = BenchmarkRunLedgerEntry.from_report(
        _report(leakage=0.025, recall=0.981, p95_ms=160.0),
        manifest_hash="sha256:" + "c" * 64,
        seed=3,
    )

    assert entry.model_id == "OpenMed/pii-small-mlx"
    assert entry.suite == "golden"
    assert entry.metrics == {
        "latency.p95_ms": 160.0,
        "leakage.overall": 0.025,
        "per_label_recall.PERSON": 0.981,
    }
    assert entry.metadata["device"] == "mlx-fp"
