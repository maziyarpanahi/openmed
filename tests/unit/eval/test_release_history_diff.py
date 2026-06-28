"""Tests for cross-release benchmark history diffing."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.core.baseline import write_baseline_store
from openmed.eval.history import (
    IMPROVEMENT,
    REGRESSION,
    UNCHANGED,
    diff_against_baseline,
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
