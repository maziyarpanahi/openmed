"""Aggregate fidelity evidence and statistical negative controls."""

import math

import pytest

from openmed.eval.synthetic_fidelity import (
    FidelityCounts,
    FidelityEvidence,
    synthetic_fidelity_report,
    wilson_interval,
)

METRICS = ("constraint_validity", "fact_consistency", "task_utility")


def evidence(successes=10, trials=10, *, distribution=None, unique=None):
    counts = {name: FidelityCounts(successes, trials) for name in METRICS}
    distribution = {"class_a": trials} if distribution is None else distribution
    return FidelityEvidence(
        counts, distribution, sum(distribution.values()) if unique is None else unique
    )


def report(value, **kwargs):
    return synthetic_fidelity_report(
        value, minimum_rates={name: 0.95 for name in METRICS}, **kwargs
    )


def test_each_fidelity_axis_is_independent_and_high_utility_cannot_mask_invalidity():
    for axis in METRICS:
        counts = {name: FidelityCounts(10, 10) for name in METRICS}
        counts[axis] = FidelityCounts(0, 10)
        result = report(FidelityEvidence(counts, {"class_a": 10}, 10))
        assert result["verdict"] == "fail"
        assert result["failed_metrics"] == [axis]


def test_no_trials_and_small_samples_are_inconclusive_not_perfect():
    result = report(evidence(0, 0))
    assert result["verdict"] == "inconclusive"
    assert result["metrics"]["task_utility"]["rate"] is None
    assert result["unique_record_fraction"] is None
    assert report(evidence(4, 4))["verdict"] == "inconclusive"


def test_pinned_minimum_boundary_is_inclusive():
    assert report(evidence(95, 100))["verdict"] == "meets_declared_limits"
    assert report(evidence(94, 100))["verdict"] == "fail"


def test_wilson_interval_known_binomial_control():
    lower, upper = wilson_interval(FidelityCounts(5, 10))
    assert lower == pytest.approx(0.236593090512564, abs=1e-12)
    assert upper == pytest.approx(0.763406909487436, abs=1e-12)
    assert wilson_interval(FidelityCounts(0, 0)) is None


def test_wilson_extreme_samples_are_bounded_and_not_exact_certainty():
    for successes in (0, 10):
        lower, upper = wilson_interval(FidelityCounts(successes, 10))
        assert 0 <= lower <= upper <= 1
        assert upper - lower > 0


def test_distribution_distance_and_record_diversity_are_separate():
    generated = evidence(distribution={"class_a": 5, "class_b": 5}, unique=2)
    reference = evidence(distribution={"class_a": 10})
    result = report(generated, reference=reference)
    assert result["unique_record_fraction"] == 0.2
    assert result["reference_total_variation"] == 0.5
    assert report(generated, reference=generated)["reference_total_variation"] == 0


def test_reference_small_samples_suppress_comparison_and_rates():
    result = report(evidence(), reference=evidence(2, 2))
    assert result["reference_total_variation"] is None
    assert result["metrics"]["task_utility"]["reference_rate"] is None


def test_evidence_snapshots_mutable_mappings():
    metrics = {name: FidelityCounts(10, 10) for name in METRICS}
    distribution = {"class_a": 10}
    value = FidelityEvidence(metrics, distribution, 10)
    metrics["task_utility"] = FidelityCounts(0, 10)
    distribution["class_a"] = 1
    assert value.metrics["task_utility"].successes == 10
    assert value.distribution["class_a"] == 10
    with pytest.raises(TypeError):
        value.distribution["class_a"] = 1


@pytest.mark.parametrize("counts", [(-1, 10), (11, 10), (True, 1), (1.5, 10)])
def test_invalid_counts_fail(counts):
    with pytest.raises(ValueError):
        FidelityCounts(*counts)


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), -1, True])
def test_invalid_pinned_rates_fail(rate):
    with pytest.raises(ValueError):
        synthetic_fidelity_report(
            evidence(), minimum_rates={name: rate for name in METRICS}
        )


def test_report_does_not_retain_reference_category_names():
    reference = evidence(distribution={"CONTROLLED_REFERENCE_CATEGORY": 10})
    result = report(evidence(), reference=reference)
    assert "CONTROLLED_REFERENCE_CATEGORY" not in str(result)
    assert math.isfinite(result["reference_total_variation"])
