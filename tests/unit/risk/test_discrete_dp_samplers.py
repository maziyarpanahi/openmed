"""Offline synthetic tests for exact discrete differential-privacy samplers."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict
from fractions import Fraction

import pytest

from openmed.risk import (
    AggregateDPBudgetLedger,
    DiscreteSamplerError,
    DPRandomSource,
    PrivacyBudget,
    PrivacyBudgetExceeded,
    discrete_gaussian_sigma_squared,
    discrete_sampling,
    release_aggregate,
    release_count,
    release_discrete_aggregate,
    release_discrete_count,
    release_discrete_histogram,
    release_snapped,
    sample_discrete_gaussian,
    sample_discrete_laplace,
    snapped_epsilon,
)

_SAMPLES = 20_000
# Upper 1e-4 standard normal quantile for the Wilson-Hilferty approximation.
_Z_CRITICAL = 3.719


def _discrete_laplace_pmf(scale: float):
    weight = math.tanh(1.0 / (2.0 * scale))

    def pmf(value: int) -> float:
        return weight * math.exp(-abs(value) / scale)

    return pmf


def _discrete_gaussian_pmf(sigma_squared: float):
    radius = int(40 * math.sqrt(sigma_squared)) + 40
    weights = {
        value: math.exp(-(value * value) / (2.0 * sigma_squared))
        for value in range(-radius, radius + 1)
    }
    total = math.fsum(weights.values())

    def pmf(value: int) -> float:
        return weights.get(value, 0.0) / total

    return pmf


def _chi_square_critical(degrees: int) -> float:
    term = 2.0 / (9.0 * degrees)
    return degrees * (1.0 - term + _Z_CRITICAL * math.sqrt(term)) ** 3


def _assert_matches_pmf(samples: list[int], pmf) -> None:
    total = len(samples)
    counts = Counter(samples)
    statistic = 0.0
    bins = 0
    tail_observed = total
    tail_probability = 1.0
    radius = 0
    while total * pmf(radius + 1) >= 5.0:
        radius += 1
    for value in range(-radius, radius + 1):
        expected = total * pmf(value)
        observed = counts.get(value, 0)
        statistic += (observed - expected) ** 2 / expected
        bins += 1
        tail_observed -= observed
        tail_probability -= pmf(value)
    if total * tail_probability >= 1.0:
        expected = total * tail_probability
        statistic += (tail_observed - expected) ** 2 / expected
        bins += 1
    assert statistic <= _chi_square_critical(bins - 1)
    distance = 0.5 * sum(
        abs(counts.get(value, 0) / total - pmf(value))
        for value in range(-radius, radius + 1)
    )
    assert distance + tail_observed / total / 2 <= 0.03


class _CountingSource(DPRandomSource):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def randbelow(self, upper: int) -> int:
        self.calls += 1
        return super().randbelow(upper)


class _AlwaysRejectSource(DPRandomSource):
    def randbelow(self, upper: int) -> int:
        return upper - 1


@pytest.mark.parametrize("scale", [Fraction(3, 2), 1 / 0.7, 4])
def test_discrete_laplace_matches_reference_pmf(scale) -> None:
    rng = DPRandomSource.for_testing(20261016)
    samples = [sample_discrete_laplace(scale, rng=rng) for _ in range(_SAMPLES)]

    assert all(type(sample) is int for sample in samples)
    _assert_matches_pmf(samples, _discrete_laplace_pmf(float(scale)))


@pytest.mark.parametrize("sigma_squared", [Fraction(1, 4), Fraction(9, 4), 6])
def test_discrete_gaussian_matches_reference_pmf(sigma_squared) -> None:
    rng = DPRandomSource.for_testing(3875)
    samples = [
        sample_discrete_gaussian(sigma_squared, rng=rng) for _ in range(_SAMPLES)
    ]

    assert all(type(sample) is int for sample in samples)
    _assert_matches_pmf(samples, _discrete_gaussian_pmf(float(sigma_squared)))


def test_discrete_count_release_is_integer_and_matches_laplace_reference() -> None:
    rng = DPRandomSource.for_testing(11)
    budget = PrivacyBudget(epsilon=1.0)
    count = 782341
    released = [
        release_discrete_count(count, epsilon=0.5, rng=rng) for _ in range(_SAMPLES)
    ]
    charged = release_discrete_count(count, epsilon=0.5, budget=budget, rng=rng)

    assert all(type(value) is int for value in [*released, charged])
    _assert_matches_pmf(
        [value - count for value in released],
        _discrete_laplace_pmf(2.0),
    )
    assert budget.spends[0].sampler == "discrete_laplace"
    assert budget.spends[0].mechanism == "laplace"
    assert budget.spends[0].noise_scale == pytest.approx(2.0)


def test_discrete_gaussian_count_matches_calibrated_reference() -> None:
    sigma_squared = discrete_gaussian_sigma_squared(1.0, 1.0, 1e-5)
    rng = DPRandomSource.for_testing(12)
    released = [
        release_discrete_count(
            40,
            epsilon=1.0,
            delta=1e-5,
            mechanism="gaussian",
            rng=rng,
        )
        for _ in range(_SAMPLES)
    ]

    assert all(type(value) is int for value in released)
    _assert_matches_pmf(
        [value - 40 for value in released],
        _discrete_gaussian_pmf(float(sigma_squared)),
    )


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0, 4.0])
@pytest.mark.parametrize("delta", [1e-10, 1e-5, 1e-3])
def test_discrete_gaussian_calibration_meets_zcdp_conversion(epsilon, delta) -> None:
    sigma_squared = discrete_gaussian_sigma_squared(1.0, epsilon, delta)

    assert isinstance(sigma_squared, Fraction)
    # Independent check: rho-zCDP implies (rho + 2 sqrt(rho log(1/delta)), delta).
    rho = 1.0 / (2.0 * float(sigma_squared))
    achieved = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    assert achieved <= epsilon
    assert achieved >= 0.999 * epsilon
    scaled = discrete_gaussian_sigma_squared(3.0, epsilon, delta)
    assert float(scaled) == pytest.approx(9.0 * float(sigma_squared), rel=1e-6)


def test_discrete_histogram_is_one_integer_query_and_keeps_keys() -> None:
    budget = PrivacyBudget(epsilon=2.0, delta=1e-3)
    rng = DPRandomSource.for_testing(5)
    released = release_discrete_histogram(
        {"low": 3, "high": 7},
        epsilon=0.5,
        budget=budget,
        rng=rng,
    )
    sequence = release_discrete_histogram(
        [4, 0, 9],
        epsilon=0.5,
        delta=1e-4,
        mechanism="gaussian",
        budget=budget,
        rng=rng,
    )

    assert set(released) == {"low", "high"}
    assert all(type(value) is int for value in released.values())
    assert len(sequence) == 3 and all(type(value) is int for value in sequence)
    assert budget.query_count == 2
    assert [spend.aggregate for spend in budget.spends] == ["histogram"] * 2
    assert [spend.sampler for spend in budget.spends] == [
        "discrete_laplace",
        "discrete_gaussian",
    ]
    assert budget.spends[1].to_dict()["calibration_method"] == (
        "zcdp_discrete_gaussian"
    )


def test_seeded_test_mode_is_deterministic_and_explicit() -> None:
    first = DPRandomSource.for_testing(7)
    second = DPRandomSource.for_testing(7)

    assert first.test_mode is True
    assert DPRandomSource().test_mode is False
    assert [sample_discrete_laplace(3, rng=first) for _ in range(50)] == [
        sample_discrete_laplace(3, rng=second) for _ in range(50)
    ]
    assert "7" not in repr(first)


def test_operational_mode_rejects_seeds_and_uses_system_randomness(
    monkeypatch,
) -> None:
    with pytest.raises(TypeError):
        DPRandomSource(7)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DPRandomSource(seed=7)  # type: ignore[call-arg]
    for seed in (True, 1.5, "7", b"7", None):
        with pytest.raises(TypeError):
            DPRandomSource.for_testing(seed)  # type: ignore[arg-type]
    for rng in (random.Random(7), random.SystemRandom(), 7):
        with pytest.raises(TypeError, match="DPRandomSource"):
            release_discrete_count(3, epsilon=1.0, rng=rng)  # type: ignore[arg-type]

    calls: list[int] = []
    system_randbelow = discrete_sampling.secrets.randbelow

    def counting_randbelow(upper: int) -> int:
        calls.append(upper)
        return system_randbelow(upper)

    monkeypatch.setattr(discrete_sampling.secrets, "randbelow", counting_randbelow)
    assert type(release_discrete_count(3, epsilon=1.0)) is int
    assert type(sample_discrete_gaussian(2)) is int
    assert calls


def test_snapped_release_lands_on_declared_grid_within_bound() -> None:
    grid = 2.0**-3
    rng = DPRandomSource.for_testing(21)
    for value in (12.3, -4.06, 0.0, 1e9, -1e9):
        released = release_snapped(
            value,
            epsilon=0.5,
            sensitivity=0.3,
            bound=100.0,
            grid=grid,
            rng=rng,
        )
        assert type(released) is float
        assert Fraction(released) % Fraction(grid) == 0
        assert -100.0 <= released <= 100.0


def test_snapped_release_charges_documented_adjusted_epsilon() -> None:
    epsilon, sensitivity, grid = 0.5, 0.3, 2.0**-3
    # Independent derivation: rounding to the grid can move neighbors by
    # ceil(sensitivity / grid) grid steps; noise is scaled for sensitivity.
    exact = (
        math.ceil(Fraction(sensitivity) / Fraction(grid))
        * Fraction(grid)
        * Fraction(epsilon)
        / Fraction(sensitivity)
    )
    budget = PrivacyBudget(epsilon=1.0)

    release_snapped(
        12.3,
        epsilon=epsilon,
        sensitivity=sensitivity,
        bound=100.0,
        grid=grid,
        budget=budget,
        rng=DPRandomSource.for_testing(2),
    )

    charged = budget.spends[0].epsilon
    assert charged == snapped_epsilon(epsilon, sensitivity=sensitivity, grid=grid)
    assert Fraction(charged) >= exact
    assert Fraction(charged) - exact <= Fraction(math.ulp(charged))
    assert charged > epsilon
    assert budget.spends[0].sampler == "grid_discrete_laplace"
    assert budget.spends[0].to_dict()["calibration_method"] == "grid_snapped_laplace"
    assert budget.spends[0].noise_scale == pytest.approx(sensitivity / epsilon)
    assert snapped_epsilon(0.5, sensitivity=1.0, grid=2.0**-4) == 0.5


def test_snapped_noise_matches_grid_laplace_reference() -> None:
    rng = DPRandomSource.for_testing(31)
    grid = 0.25
    released = [
        release_snapped(
            10.0,
            epsilon=1.0,
            sensitivity=1.0,
            bound=1024.0,
            grid=grid,
            rng=rng,
        )
        for _ in range(_SAMPLES // 2)
    ]
    steps = [int((Fraction(value) - 10) / Fraction(grid)) for value in released]

    _assert_matches_pmf(steps, _discrete_laplace_pmf(4.0))


@pytest.mark.parametrize(
    "grid,bound",
    [
        (0.3, 3.0),
        (3.0, 3.0),
        (0.0, 1.0),
        (-0.25, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (True, 1.0),
        (2.0**-1074, 1.0),
        (0.25, 1.1),
        (0.25, 0.0),
        (0.25, -1.0),
        (0.25, float("inf")),
        (2.0**-10, 2.0**43),
    ],
)
def test_snapped_release_rejects_invalid_grid_and_bound(grid, bound) -> None:
    budget = PrivacyBudget(epsilon=1.0)
    with pytest.raises((TypeError, ValueError)):
        release_snapped(
            1.0,
            epsilon=0.5,
            sensitivity=1.0,
            bound=bound,
            grid=grid,
            budget=budget,
            rng=DPRandomSource.for_testing(1),
        )
    assert budget.query_count == 0


@pytest.mark.parametrize(
    "scale",
    [0, -1, Fraction(0), Fraction(-1, 2), float("nan"), float("inf"), True, "2"],
)
def test_samplers_reject_invalid_parameters(scale) -> None:
    rng = DPRandomSource.for_testing(1)
    with pytest.raises((TypeError, ValueError)):
        sample_discrete_laplace(scale, rng=rng)
    with pytest.raises((TypeError, ValueError)):
        sample_discrete_gaussian(scale, rng=rng)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 3.0},
        {"count": True},
        {"count": "3"},
        {"epsilon": 0},
        {"epsilon": -1},
        {"epsilon": float("nan")},
        {"epsilon": float("inf")},
        {"epsilon": True},
        {"sensitivity": 0},
        {"sensitivity": -1},
        {"sensitivity": float("inf")},
        {"sensitivity": False},
        {"mechanism": "uniform"},
        {"mechanism": "gaussian", "delta": 0.0},
        {"delta": 1.0},
    ],
)
def test_discrete_count_rejects_invalid_parameters(kwargs) -> None:
    budget = PrivacyBudget(epsilon=10.0, delta=0.5)
    arguments = {"count": 3, "epsilon": 1.0, **kwargs}
    count = arguments.pop("count")
    with pytest.raises((TypeError, ValueError)):
        release_discrete_count(
            count,
            budget=budget,
            rng=DPRandomSource.for_testing(1),
            **arguments,
        )
    assert budget.query_count == 0


def test_budget_rejection_happens_before_any_noise_is_drawn() -> None:
    budget = PrivacyBudget(epsilon=0.5)
    source = _CountingSource()

    with pytest.raises(PrivacyBudgetExceeded):
        release_discrete_count(5, epsilon=0.6, budget=budget, rng=source)
    with pytest.raises(PrivacyBudgetExceeded):
        release_snapped(
            5.0,
            epsilon=0.5,
            sensitivity=0.3,
            bound=8.0,
            grid=0.125,
            budget=budget,
            rng=source,
        )

    assert source.calls == 0
    assert budget.query_count == 0


def test_rejection_bound_fails_closed_without_values() -> None:
    budget = PrivacyBudget(epsilon=1.0)

    with pytest.raises(DiscreteSamplerError) as caught:
        release_discrete_count(
            782341,
            epsilon=1.0,
            budget=budget,
            rng=_AlwaysRejectSource(),
        )

    assert "782341" not in str(caught.value)
    assert budget.query_count == 0


def test_records_do_not_depend_on_or_contain_noise_realizations(caplog) -> None:
    def run(seed: int) -> tuple[list[object], dict[str, object]]:
        budget = PrivacyBudget(epsilon=10.0, delta=1e-3)
        rng = DPRandomSource.for_testing(seed)
        released: list[object] = [
            release_discrete_count(782341, epsilon=1.0, budget=budget, rng=rng),
            release_discrete_count(
                782341,
                epsilon=1.0,
                delta=1e-4,
                mechanism="gaussian",
                budget=budget,
                rng=rng,
            ),
            release_discrete_histogram(
                {"synthetic_bin": 782341}, epsilon=1.0, budget=budget, rng=rng
            ),
            release_snapped(
                7823.5,
                epsilon=1.0,
                sensitivity=1.0,
                bound=65536.0,
                grid=0.5,
                budget=budget,
                rng=rng,
            ),
        ]
        return released, budget.to_dict()

    first_values, first_record = run(101)
    second_values, second_record = run(202)

    assert first_values != second_values
    assert first_record == second_record
    serialized = json.dumps(first_record) + repr(first_record) + caplog.text
    assert "782341" not in serialized
    assert "7823" not in serialized
    for value in first_values:
        assert repr(value) not in json.dumps(first_record["spends"])


def test_spend_records_name_their_sampler() -> None:
    budget = PrivacyBudget(epsilon=10.0, delta=0.1)

    release_count(4, epsilon=0.5, budget=budget, rng=random.Random(3))
    release_count(
        4,
        epsilon=0.5,
        delta=1e-4,
        mechanism="gaussian",
        budget=budget,
        rng=random.Random(4),
    )
    budget.spend(0.5)

    assert [spend.sampler for spend in budget.spends] == [
        "float_laplace",
        "float_gaussian",
        "unspecified",
    ]
    assert [spend.to_dict()["sampler"] for spend in budget.spends] == [
        "float_laplace",
        "float_gaussian",
        "unspecified",
    ]
    with pytest.raises(ValueError):
        budget.spend(0.1, sampler="mersenne")


def test_discrete_aggregate_release_records_sampler_and_test_mode() -> None:
    ledger = AggregateDPBudgetLedger(max_epsilon=2.0, max_delta=0.0)
    release = release_discrete_aggregate(
        {"synthetic_count": 782341, "synthetic_other": 12},
        ledger=ledger,
        epsilon=0.5,
        rng=DPRandomSource.for_testing(9),
    )
    scalar = release_discrete_aggregate(17, ledger=ledger, epsilon=0.5)

    assert all(type(value) is int for value in release.value.values())
    assert type(scalar.value) is int
    assert release.test_only is True
    assert scalar.test_only is False
    record = release.to_dict()
    assert record["spend"]["sampler"] == "discrete_laplace"
    assert "782341" not in json.dumps(record["spend"]) + repr(ledger.to_dict())
    for item in (asdict(release), vars(release)):
        assert "noise" not in item and "seed" not in item
    with pytest.raises(TypeError):
        release_discrete_aggregate({"synthetic_count": 1.5}, ledger=ledger, epsilon=0.5)
    with pytest.raises(TypeError, match="row-level"):
        release_discrete_aggregate([1, 2], ledger=ledger, epsilon=0.5)
    with pytest.raises(TypeError, match="DPRandomSource"):
        release_discrete_aggregate(1, ledger=ledger, epsilon=0.5, rng=random.Random(1))
    assert len(ledger.spends) == 2


def test_legacy_aggregate_ledger_keeps_manual_spend_records_unchanged() -> None:
    ledger = AggregateDPBudgetLedger(max_epsilon=2.0, max_delta=0.0)
    manual = ledger.spend(0.25, label="count")
    release_aggregate(3.0, ledger=ledger, epsilon=0.25, seed="synthetic")

    assert "sampler" not in manual.to_dict()
    assert ledger.spends[1].to_dict()["sampler"] == "float_laplace"
