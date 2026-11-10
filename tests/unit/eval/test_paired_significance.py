"""Unit tests for paired significance testing between benchmark runs."""

from __future__ import annotations

import pytest

from openmed.eval.metrics import PairedSignificance, paired_significance


def test_identical_series_yield_high_p_value() -> None:
    docs = [(9, 10), (7, 8), (3, 5), (1, 4), (12, 12)]

    result = paired_significance(
        docs,
        docs,
        "character_recall",
        n_resamples=500,
        seed=7,
    )

    assert isinstance(result, PairedSignificance)
    assert result.observed_delta == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_clearly_separated_series_yield_low_p_value() -> None:
    better = [(10, 10)] * 16
    worse = [(2, 10)] * 16

    result = paired_significance(
        better,
        worse,
        "character_recall",
        n_resamples=2000,
        seed=3,
    )

    assert result.observed_delta == pytest.approx(0.8)
    assert result.p_value < 0.01


def test_fixed_seed_is_reproducible() -> None:
    run_a = [(5, 10), (8, 10), (9, 10), (6, 10), (7, 10)]
    run_b = [(4, 10), (6, 10), (8, 10), (6, 10), (2, 10)]

    first = paired_significance(run_a, run_b, "recall", n_resamples=250, seed=13)
    second = paired_significance(run_a, run_b, "recall", n_resamples=250, seed=13)

    assert second.to_dict() == first.to_dict()


def test_mismatched_length_inputs_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_significance(
            [(1, 1)],
            [(1, 1), (1, 1)],
            "leakage_rate",
            n_resamples=100,
            seed=0,
        )


def test_builtin_leakage_statistic_compares_document_rates() -> None:
    lower_leakage = [(0, 10), (1, 10), (1, 20)]
    higher_leakage = [(4, 10), (3, 10), (5, 20)]

    result = paired_significance(
        lower_leakage,
        higher_leakage,
        "leakage",
        n_resamples=500,
        seed=5,
    )

    assert result.observed_delta == pytest.approx(-0.25)


def test_builtin_f1_statistic_aggregates_counts() -> None:
    perfect = [(1, 0, 0), (2, 0, 0), (1, 0, 0)]
    mixed = [(1, 1, 1), (1, 1, 1), (0, 0, 0)]

    result = paired_significance(perfect, mixed, "f1", n_resamples=500, seed=11)

    assert result.observed_delta == pytest.approx(0.5)
