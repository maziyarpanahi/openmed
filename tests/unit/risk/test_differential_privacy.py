"""Offline synthetic tests for aggregate differential privacy mechanisms."""

from __future__ import annotations

import math
import random

import pytest

from openmed.risk import release_dp_aggregate
from openmed.risk.differential_privacy import (
    DifferentialPrivacy,
    PrivacyBudget,
    PrivacyBudgetExceeded,
    gaussian_noise,
    gaussian_scale,
    laplace_noise,
    laplace_scale,
    release_count,
    release_histogram,
    release_mean,
    release_sum,
    utility_report,
)


def test_package_alias_keeps_dispatch_release_available() -> None:
    released = release_dp_aggregate(
        4,
        kind="count",
        epsilon=1.0,
        rng=_FixedRandom(0.5),
    )

    assert released == pytest.approx(4.0)


def test_laplace_scale_and_seeded_noise_follow_sensitivity_formula() -> None:
    sensitivity = 3.0
    epsilon = 0.6
    expected_scale = sensitivity / epsilon

    assert laplace_scale(sensitivity, epsilon) == pytest.approx(expected_scale)
    unit = 0.75
    expected_noise = -expected_scale * math.log(1.0 - 2.0 * (unit - 0.5))
    assert laplace_noise(sensitivity, epsilon, rng=_FixedRandom(unit)) == pytest.approx(
        expected_noise
    )


def test_gaussian_scale_and_seeded_noise_follow_sensitivity_formula() -> None:
    sensitivity = 2.0
    epsilon = 0.8
    delta = 1e-5
    expected_scale = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    assert gaussian_scale(sensitivity, epsilon, delta) == pytest.approx(expected_scale)
    assert gaussian_noise(
        sensitivity,
        epsilon,
        delta,
        rng=_FixedGaussian(1.5),
    ) == pytest.approx(1.5 * expected_scale)


def test_gaussian_count_release_charges_delta_and_scale() -> None:
    delta = 1e-4
    budget = PrivacyBudget(epsilon=1.0, delta=0.01)

    released = release_count(
        9,
        epsilon=0.5,
        delta=delta,
        mechanism="gaussian",
        budget=budget,
        rng=_FixedGaussian(0.0),
    )

    expected_scale = gaussian_scale(1.0, 0.5, delta)
    assert released == pytest.approx(9.0)
    assert budget.spent_delta == pytest.approx(delta)
    assert budget.spends[0].mechanism == "gaussian"
    assert budget.spends[0].noise_scale == pytest.approx(expected_scale)


def test_budget_composes_queries_and_rejects_without_mutating() -> None:
    budget = PrivacyBudget(epsilon=1.0, delta=0.1)

    release_count(4, epsilon=0.4, budget=budget, rng=random.Random(3))
    release_sum(
        12,
        epsilon=0.3,
        sensitivity=2,
        delta=0.02,
        budget=budget,
        rng=random.Random(4),
    )

    assert budget.spent_epsilon == pytest.approx(0.7)
    assert budget.spent_delta == pytest.approx(0.02)
    assert budget.query_count == 2
    assert budget.remaining_epsilon == pytest.approx(0.3)

    with pytest.raises(PrivacyBudgetExceeded) as caught:
        release_count(8, epsilon=0.31, budget=budget, rng=random.Random(5))

    assert caught.value.decision.reason == "epsilon exceeds budget"
    assert budget.query_count == 2
    assert budget.spent_epsilon == pytest.approx(0.7)


def test_budget_rejects_delta_overrun_without_mutating() -> None:
    budget = PrivacyBudget(epsilon=1.0, delta=0.01)

    with pytest.raises(PrivacyBudgetExceeded) as caught:
        release_count(
            8,
            epsilon=0.2,
            delta=0.02,
            mechanism="gaussian",
            budget=budget,
            rng=_FixedGaussian(0.0),
        )

    assert caught.value.decision.reason == "delta exceeds budget"
    assert budget.query_count == 0


def test_histogram_is_one_budget_query_and_keeps_only_aggregate_keys() -> None:
    budget = PrivacyBudget(epsilon=1.0)
    released = release_histogram(
        {"low": 3, "high": 7},
        epsilon=0.5,
        budget=budget,
        rng=random.Random(8),
    )

    assert set(released) == {"low", "high"}
    assert len(released) == 2
    assert budget.query_count == 1
    assert budget.spends[0].aggregate == "histogram"
    assert "3" not in repr(budget.to_dict())
    assert "7" not in repr(budget.to_dict())


def test_mean_can_derive_bounded_sensitivity() -> None:
    budget = PrivacyBudget(epsilon=1.0)
    released = release_mean(
        50,
        epsilon=0.5,
        lower=0,
        upper=100,
        count=10,
        budget=budget,
        rng=_FixedRandom(0.5),
    )

    assert released == pytest.approx(50.0)
    assert budget.spends[0].sensitivity == pytest.approx(10.0)


def test_utility_report_exposes_error_tradeoff_without_source_values() -> None:
    report = utility_report([0.5, 1.0, 2.0], sensitivity=2.0)

    assert [point.noise_scale for point in report.points] == pytest.approx(
        [4.0, 2.0, 1.0]
    )
    assert report.points[0].root_mean_square_error == pytest.approx(4.0 * math.sqrt(2))
    assert "patient" not in repr(report.to_dict()).lower()


def test_gaussian_requires_positive_delta_and_facade_shares_budget() -> None:
    with pytest.raises(ValueError, match="delta greater than zero"):
        release_count(2, epsilon=0.5, mechanism="gaussian", delta=0.0)

    layer = DifferentialPrivacy(
        epsilon=1.0,
        rng=_FixedRandom(0.5),
    )
    assert layer.count(5, epsilon=0.4) == pytest.approx(5.0)
    assert layer.budget.spent_epsilon == pytest.approx(0.4)


class _FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class _FixedGaussian:
    def __init__(self, value: float) -> None:
        self.value = value

    def gauss(self, mu: float, sigma: float) -> float:
        return mu + self.value * sigma
