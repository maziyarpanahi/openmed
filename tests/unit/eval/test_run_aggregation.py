"""Tests for deterministic multi-run benchmark aggregation."""

from __future__ import annotations

import random
import statistics

import pytest

from openmed.eval.history import (
    AGGREGATION_SCHEMA_VERSION,
    BenchmarkRunLedger,
    BenchmarkRunLedgerEntry,
    MetricDistribution,
    RunAggregation,
    aggregate_run_ledger,
    paired_run_significance,
)

MANIFEST = "sha256:" + "b" * 64
CANDIDATE_MANIFEST = "sha256:" + "c" * 64


def _entry(
    *,
    seed: int,
    leakage: float,
    recall: float,
    manifest_hash: str = MANIFEST,
    model_id: str = "OpenMed/pii-small-mlx",
    suite: str = "golden",
    generated_at: str | None = None,
) -> BenchmarkRunLedgerEntry:
    """Build one synthetic ledger row from algorithmic metric values."""

    if generated_at is None:
        generated_at = f"2026-06-{10 + seed:02d}T00:00:00+00:00"
    return BenchmarkRunLedgerEntry(
        model_id=model_id,
        suite=suite,
        manifest_hash=manifest_hash,
        seed=seed,
        generated_at=generated_at,
        fixture_count=4,
        metrics={
            "leakage.overall": leakage,
            "per_label_recall.PERSON": recall,
        },
        metadata={"device": "mlx-fp"},
    )


def _synthetic_ledger(seeds: range = range(6)) -> BenchmarkRunLedger:
    """Deterministic within-noise ledger: values are a fixed function of seed."""

    ledger = BenchmarkRunLedger()
    for seed in seeds:
        leakage = round(0.010 + 0.001 * ((seed * 7) % 5), 6)
        recall = round(0.980 + 0.002 * ((seed * 3) % 4), 6)
        ledger = ledger.append(_entry(seed=seed, leakage=leakage, recall=recall))
    return ledger


def test_aggregation_computes_metric_distributions() -> None:
    ledger = _synthetic_ledger()
    aggregation = aggregate_run_ledger(ledger, n_resamples=200, seed=11)

    assert aggregation.schema_version == AGGREGATION_SCHEMA_VERSION
    dist = aggregation.distribution(
        "OpenMed/pii-small-mlx", "golden", MANIFEST, "leakage.overall"
    )
    assert isinstance(dist, MetricDistribution)

    leakage_values = [round(0.010 + 0.001 * ((seed * 7) % 5), 6) for seed in range(6)]
    assert dist.count == 6
    assert dist.seeds == tuple(range(6))
    assert dist.values == tuple(leakage_values)
    assert dist.mean == pytest.approx(statistics.fmean(leakage_values))
    assert dist.median == pytest.approx(statistics.median(leakage_values))
    assert dist.variance == pytest.approx(statistics.variance(leakage_values))
    assert dist.stdev == pytest.approx(statistics.stdev(leakage_values))
    assert dist.minimum == min(leakage_values)
    assert dist.maximum == max(leakage_values)
    assert dist.confidence_interval.lower <= dist.mean <= dist.confidence_interval.upper


def test_confidence_intervals_are_deterministic_and_byte_identical() -> None:
    ledger = _synthetic_ledger()

    first = aggregate_run_ledger(ledger, n_resamples=256, alpha=0.05, seed=7)
    second = aggregate_run_ledger(ledger, n_resamples=256, alpha=0.05, seed=7)

    assert first.to_json() == second.to_json()
    dist = first.distribution(
        "OpenMed/pii-small-mlx", "golden", MANIFEST, "leakage.overall"
    )
    assert dist is not None
    assert dist.confidence_interval.n_resamples == 256
    assert dist.confidence_interval.alpha == 0.05


def test_aggregation_is_order_independent() -> None:
    ledger = _synthetic_ledger()
    entries = list(ledger.entries)

    shuffled = entries[:]
    random.Random(2026).shuffle(shuffled)
    reversed_entries = list(reversed(entries))

    canonical = aggregate_run_ledger(entries, n_resamples=200, seed=3).to_json()
    from_shuffled = aggregate_run_ledger(shuffled, n_resamples=200, seed=3).to_json()
    from_reversed = aggregate_run_ledger(
        reversed_entries, n_resamples=200, seed=3
    ).to_json()

    assert canonical == from_shuffled == from_reversed


def test_trailing_window_keeps_most_recent_runs() -> None:
    ledger = _synthetic_ledger(range(8))

    windowed = aggregate_run_ledger(ledger, window=3, n_resamples=64, seed=5)
    dist = windowed.distribution(
        "OpenMed/pii-small-mlx", "golden", MANIFEST, "leakage.overall"
    )

    assert windowed.window == 3
    assert dist is not None
    # generated_at increases with seed, so the trailing window is the top seeds.
    assert dist.seeds == (5, 6, 7)
    assert dist.count == 3


def test_single_run_yields_degenerate_zero_width_interval() -> None:
    ledger = BenchmarkRunLedger().append(_entry(seed=1, leakage=0.02, recall=0.99))

    aggregation = aggregate_run_ledger(ledger, n_resamples=100, seed=1)
    dist = aggregation.distribution(
        "OpenMed/pii-small-mlx", "golden", MANIFEST, "leakage.overall"
    )

    assert dist is not None
    assert dist.count == 1
    assert dist.variance == 0.0
    assert dist.stdev == 0.0
    assert dist.confidence_interval.degenerate is True
    assert dist.confidence_interval.lower == dist.confidence_interval.upper == 0.02


def test_paired_run_significance_pairs_by_seed_and_is_deterministic() -> None:
    baseline = _synthetic_ledger()
    candidate = BenchmarkRunLedger()
    for entry in baseline.entries:
        candidate = candidate.append(
            _entry(
                seed=entry.seed,
                leakage=entry.metrics["leakage.overall"] + 0.05,
                recall=entry.metrics["per_label_recall.PERSON"],
                manifest_hash=CANDIDATE_MANIFEST,
            )
        )

    first = paired_run_significance(
        baseline, candidate, "leakage.overall", n_resamples=500, seed=9
    )
    second = paired_run_significance(
        baseline, candidate, "leakage.overall", n_resamples=500, seed=9
    )

    assert first.to_dict() == second.to_dict()
    assert first.observed_delta == pytest.approx(0.05)
    # A uniform +0.05 shift is only matched when every paired seed flips sign
    # together, so the permutation p-value is small but deterministic.
    assert first.p_value < 0.05
    assert first.n_resamples == 500


def test_paired_run_significance_is_order_independent() -> None:
    baseline = _synthetic_ledger()
    candidate_entries = [
        _entry(
            seed=entry.seed,
            leakage=entry.metrics["leakage.overall"] + 0.001 * (entry.seed % 3),
            recall=entry.metrics["per_label_recall.PERSON"],
            manifest_hash=CANDIDATE_MANIFEST,
        )
        for entry in baseline.entries
    ]
    baseline_entries = list(baseline.entries)

    shuffled_baseline = baseline_entries[:]
    shuffled_candidate = candidate_entries[:]
    random.Random(11).shuffle(shuffled_baseline)
    random.Random(13).shuffle(shuffled_candidate)

    canonical = paired_run_significance(
        baseline_entries, candidate_entries, "leakage.overall", n_resamples=300, seed=4
    )
    reordered = paired_run_significance(
        shuffled_baseline,
        shuffled_candidate,
        "leakage.overall",
        n_resamples=300,
        seed=4,
    )

    assert canonical.to_dict() == reordered.to_dict()


def test_aggregation_rejects_invalid_window() -> None:
    ledger = _synthetic_ledger()
    with pytest.raises(ValueError, match="window"):
        aggregate_run_ledger(ledger, window=0)


def test_run_aggregation_default_is_empty() -> None:
    aggregation = RunAggregation()
    assert aggregation.distributions == ()
    assert aggregation.window is None
    assert aggregation.to_dict()["distributions"] == []
