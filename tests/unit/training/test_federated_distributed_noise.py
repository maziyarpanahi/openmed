"""Synthetic, offline contracts for dropout-aware distributed noise shares."""

from __future__ import annotations

import hashlib
import json
import math
import random
from fractions import Fraction

import pytest

from openmed.training import federated_distributed_noise as noise_module
from openmed.training.federated_distributed_noise import (
    DISTRIBUTED_NOISE_REASON_CODES,
    FEDERATED_DISTRIBUTED_NOISE_SCHEMA_VERSION,
    SHARE_VARIANCE_DEFICIT,
    SHARE_VARIANCE_MARGIN,
    DiscreteGaussianShareSampler,
    DistributedNoiseCalibration,
    DistributedNoiseError,
    DistributedNoiseMechanismRecord,
    calibrate_distributed_noise,
    noised_client_share,
    unmask_noised_aggregate,
)

_MODULUS = 2**32
_BASE = {
    "client_count": 5,
    "recovery_threshold": 3,
    "target_aggregate_variance": 300,
    "dimension": 8,
    "modulus": _MODULUS,
    "clip_l2_norm": 16,
    "tail_bound_bits": 40,
}


def _calibration(**overrides):
    return calibrate_distributed_noise(**{**_BASE, **overrides})


def _code(excinfo) -> str:
    error = excinfo.value
    assert str(error) == error.code
    assert error.code in DISTRIBUTED_NOISE_REASON_CODES
    return error.code


def _exact_pmf(sigma_squared: float, support: int) -> dict[int, float]:
    weights = {
        k: math.exp(-(k * k) / (2.0 * sigma_squared))
        for k in range(-support, support + 1)
    }
    total = math.fsum(weights.values())
    return {k: w / total for k, w in weights.items()}


def _exact_variance(sigma_squared: float) -> float:
    pmf = _exact_pmf(sigma_squared, 60 + int(40 * math.sqrt(sigma_squared)))
    return math.fsum(k * k * p for k, p in pmf.items())


# Reference vectors: (n, t, target aggregate variance, expected share scale).
_REFERENCE_VECTORS = (
    (5, 3, 300, Fraction(104857700, 1048576)),
    (10, 7, 70, Fraction(10485770, 1048576)),
    (4, 4, Fraction(9, 2), Fraction(9437193, 8388608)),
    (8, 1, 2, Fraction(1048577, 524288)),
    (12, 9, Fraction(45, 2), Fraction(5242885, 2097152)),
)


@pytest.mark.parametrize(("n", "t", "target", "expected_share"), _REFERENCE_VECTORS)
def test_reference_vectors_meet_target_for_every_survivor_count(
    n, t, target, expected_share
):
    calibration = _calibration(
        client_count=n, recovery_threshold=t, target_aggregate_variance=target
    )
    target = Fraction(target)
    assert calibration.share_scale == expected_share
    assert calibration.share_scale == target / t * (1 + SHARE_VARIANCE_MARGIN)
    for survivors in range(t, n + 1):
        assert calibration.aggregate_scale(survivors) == survivors * expected_share
        lower = calibration.aggregate_variance_lower_bound(survivors)
        assert lower == survivors * expected_share * (1 - SHARE_VARIANCE_DEFICIT)
        assert lower >= target
    # The quorum is the binding case and the margin stays small there.
    assert calibration.aggregate_variance_lower_bound(t) < target * (
        1 + 2 * SHARE_VARIANCE_MARGIN
    )


@pytest.mark.parametrize(("n", "t", "target", "expected_share"), _REFERENCE_VECTORS)
def test_calibrating_to_client_count_would_miss_target_after_dropout(
    n, t, target, expected_share
):
    target = Fraction(target)
    naive_share = target / n
    if t < n:
        assert t * naive_share < target
    assert t * expected_share * (1 - SHARE_VARIANCE_DEFICIT) >= target


@pytest.mark.parametrize("sigma_squared", [1, Fraction(3, 2), 2, Fraction(1049, 1000)])
def test_share_variance_deficit_bound_holds_against_exact_pmf(sigma_squared):
    exact = _exact_variance(float(sigma_squared))
    assert exact < float(sigma_squared)
    assert exact >= float(sigma_squared) * (1 - float(SHARE_VARIANCE_DEFICIT))


def test_survivors_below_threshold_abort_before_unmasking():
    calibration = _calibration()
    calls = []

    def unmask():
        calls.append(True)
        return [0] * calibration.dimension

    for survivors in (0, 1, calibration.recovery_threshold - 1):
        with pytest.raises(DistributedNoiseError) as excinfo:
            unmask_noised_aggregate(calibration, survivors, unmask)
        assert _code(excinfo) == "survivors_below_threshold"
        with pytest.raises(DistributedNoiseError) as excinfo:
            calibration.require_recovery_quorum(survivors)
        assert _code(excinfo) == "survivors_below_threshold"
        with pytest.raises(DistributedNoiseError):
            calibration.aggregate_variance_lower_bound(survivors)
    assert calls == []


@pytest.mark.parametrize("survivors", [6, -1, True, 3.0, "3", None])
def test_invalid_survivor_counts_are_refused_before_unmasking(survivors):
    calibration = _calibration()
    calls = []
    with pytest.raises(DistributedNoiseError) as excinfo:
        unmask_noised_aggregate(calibration, survivors, lambda: calls.append(1))
    assert _code(excinfo) == "invalid_survivor_count"
    assert calls == []


def test_fake_quorum_round_trip_recovers_signal_plus_bounded_noise():
    calibration = _calibration()
    sampler = DiscreteGaussianShareSampler.for_testing(20261025)
    signal_rng = random.Random(7)
    signals = [
        [signal_rng.randint(-5, 5) for _ in range(calibration.dimension)]
        for _ in range(calibration.client_count)
    ]
    shares = [
        noised_client_share(calibration, signal, sampler, allow_test_sampler=True)
        for signal in signals
    ]
    for share in shares:
        assert len(share) == calibration.dimension
        assert all(0 <= value < calibration.modulus for value in share)
    for survivors in range(calibration.recovery_threshold, len(shares) + 1):
        modular_sum = [
            sum(column) % calibration.modulus for column in zip(*shares[:survivors])
        ]
        decoded = unmask_noised_aggregate(
            calibration, survivors, lambda total=modular_sum: list(total)
        )
        expected_signal = [sum(column) for column in zip(*signals[:survivors])]
        noise = [value - base for value, base in zip(decoded, expected_signal)]
        assert all(abs(value) <= calibration.noise_tail_bound for value in noise)
        assert any(value != 0 for value in noise)


def test_unmask_port_failure_and_malformed_sums_fail_closed():
    calibration = _calibration()

    def failing():
        raise RuntimeError("secret detail 12345")

    with pytest.raises(DistributedNoiseError) as excinfo:
        unmask_noised_aggregate(calibration, 3, failing)
    assert _code(excinfo) == "unmask_failed"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__
    assert "12345" not in str(excinfo.value)

    bad_sums = (
        [0] * (calibration.dimension - 1),
        [calibration.modulus] + [0] * (calibration.dimension - 1),
        [-1] + [0] * (calibration.dimension - 1),
        [True] + [0] * (calibration.dimension - 1),
        [0.0] * calibration.dimension,
        None,
        "0" * calibration.dimension,
    )
    for bad in bad_sums:
        with pytest.raises(DistributedNoiseError) as excinfo:
            unmask_noised_aggregate(calibration, 4, lambda value=bad: value)
        assert _code(excinfo) == "invalid_unmasked_sum"


def test_centered_decoding_of_modular_sum():
    calibration = _calibration(modulus=1_000_003)
    residues = [0, 1, 500_001, 500_002, 1_000_002, 7, 999_990, 3]
    decoded = unmask_noised_aggregate(calibration, 5, lambda: residues)
    assert decoded == (0, 1, 500_001, -500_001, -1, 7, -13, 3)


def test_modulus_boundary_rejects_wraparound_risk():
    reference = _calibration()
    needed = 2 * (reference.aggregate_signal_bound + reference.noise_tail_bound) + 1
    assert reference.aggregate_signal_bound == 5 * 16
    accepted = _calibration(modulus=needed)
    assert accepted.modulus == needed
    for modulus in (needed - 1, needed - 2, 3, 2**8):
        with pytest.raises(DistributedNoiseError) as excinfo:
            _calibration(modulus=modulus)
        assert _code(excinfo) == "modulus_wraparound_risk"


def test_noise_tail_bound_reference_and_tail_probability():
    calibration = _calibration()
    n = calibration.client_count
    sigma_squared = calibration.share_scale
    exponent = 40 + 1 + 3  # tail bits + log2(2) + ceil(log2(8)).
    exact_square = 2 * n * sigma_squared * Fraction(6932, 10000) * exponent
    k = calibration.noise_tail_bound
    assert k == 175
    assert k * k >= exact_square > (k - 1) ** 2
    variance_proxy = float(n * sigma_squared)
    tail = 2 * calibration.dimension * math.exp(-(k * k) / (2 * variance_proxy))
    assert tail <= 2.0**-40


def test_sensitivity_tau_and_zcdp_bounds_match_cited_formulas():
    calibration = _calibration()
    d = calibration.dimension
    sigma_squared = calibration.share_scale
    assert calibration.l2_sensitivity_squared_bound == (16 + 3) ** 2
    assert calibration.l2_sensitivity_squared_bound >= (16 + math.sqrt(d)) ** 2
    true_tau = 10 * math.fsum(
        math.exp(-2 * math.pi**2 * float(sigma_squared) * k / (k + 1))
        for k in range(1, calibration.client_count)
    )
    assert calibration.sum_closeness_tau_bound >= true_tau
    expected_rho = (
        calibration.l2_sensitivity_squared_bound
        / (calibration.recovery_threshold * sigma_squared)
        + 2 * calibration.sum_closeness_tau_bound * d
    ) / 2
    assert calibration.zcdp_rho_bound == expected_rho
    small = _calibration(target_aggregate_variance=3)
    small_tau = 10 * math.fsum(
        math.exp(-2 * math.pi**2 * float(small.share_scale) * k / (k + 1))
        for k in range(1, small.client_count)
    )
    assert small_tau > 0
    assert small.sum_closeness_tau_bound >= small_tau
    assert small.sum_closeness_tau_bound <= 4 * small_tau


def _chi_square(counts, pmf, width: int, draws: int) -> float:
    """Chi-square over |k| <= width plus one pooled tail bin."""
    bins = range(-width, width + 1)
    expected = [draws * pmf[k] for k in bins]
    observed = [counts.get(k, 0) for k in bins]
    tail_expected = draws - math.fsum(expected)
    tail_observed = draws - sum(observed)
    return (
        math.fsum((o - e) ** 2 / e for o, e in zip(observed, expected))
        + (tail_observed - tail_expected) ** 2 / tail_expected
    )


# (sigma**2, half-width w, chi-square 0.999 quantile for 2w + 1 dof).
_PMF_CASES = (
    (Fraction(1, 4), 1, 16.27),
    (Fraction(3, 2), 3, 24.32),
    (5, 6, 34.53),
)


@pytest.mark.parametrize(("sigma_squared", "width", "critical"), _PMF_CASES)
def test_sampler_matches_exact_pmf_under_seeded_test_rng(
    sigma_squared, width, critical
):
    sampler = DiscreteGaussianShareSampler.for_testing(3860)
    draws = 20_000
    counts: dict[int, int] = {}
    for _ in range(draws):
        value = sampler.sample(sigma_squared)
        assert type(value) is int
        counts[value] = counts.get(value, 0) + 1
    pmf = _exact_pmf(float(sigma_squared), 60)
    assert set(counts) <= set(pmf)
    total_variation = 0.5 * math.fsum(
        abs(counts.get(k, 0) / draws - p) for k, p in pmf.items()
    )
    assert total_variation < 0.015
    assert _chi_square(counts, pmf, width, draws) < critical
    # Negative control: the same draws reject a 1.5x mis-scaled reference.
    wrong = _exact_pmf(1.5 * float(sigma_squared), 60)
    assert _chi_square(counts, wrong, width, draws) > critical
    mean = sum(k * c for k, c in counts.items()) / draws
    assert abs(mean) < 5 * math.sqrt(float(sigma_squared) / draws)


def test_seeded_test_sampler_is_deterministic():
    first = DiscreteGaussianShareSampler.for_testing(11)
    second = DiscreteGaussianShareSampler.for_testing(11)
    values = [first.sample(Fraction(1049, 1000)) for _ in range(200)]
    assert values == [second.sample(Fraction(1049, 1000)) for _ in range(200)]
    assert first.is_test_mode


@pytest.mark.parametrize("gamma", [Fraction(0), Fraction(1, 2), 1, Fraction(5, 2)])
def test_bernoulli_exp_matches_exact_probability(gamma):
    rng = random.Random(42)
    trials = 6_000
    hits = sum(
        noise_module._bernoulli_exp_neg(rng, Fraction(gamma)) for _ in range(trials)
    )
    p = math.exp(-float(gamma))
    spread = 4.5 * math.sqrt(max(p * (1 - p), 1e-12) / trials)
    assert abs(hits / trials - p) <= spread + 1e-12


def test_large_scale_sampling_stays_integer_and_centered():
    sampler = DiscreteGaussianShareSampler.for_testing(5)
    sigma_squared = Fraction(10**6, 3)
    values = [sampler.sample(sigma_squared) for _ in range(400)]
    assert all(type(value) is int for value in values)
    spread = math.sqrt(float(sigma_squared))
    assert abs(sum(values) / len(values)) < 0.3 * spread
    assert max(abs(value) for value in values) < 7 * spread


def test_operational_mode_requires_exact_system_random():
    class DerivedSystemRandom(random.SystemRandom):
        pass

    for rng in (random.Random(1), DerivedSystemRandom(), None, object(), 7):
        with pytest.raises(DistributedNoiseError) as excinfo:
            DiscreteGaussianShareSampler(rng)
        assert _code(excinfo) == "randomness_not_operational"
    sampler = DiscreteGaussianShareSampler(random.SystemRandom())
    assert not sampler.is_test_mode
    assert type(sampler.sample(2)) is int


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.0, "1", None])
def test_test_mode_seed_is_validated(seed):
    with pytest.raises(DistributedNoiseError) as excinfo:
        DiscreteGaussianShareSampler.for_testing(seed)
    assert _code(excinfo) == "invalid_test_seed"


@pytest.mark.parametrize(
    "scale", [0, -1, Fraction(-1, 2), True, float("nan"), float("inf"), "1", 2**200]
)
def test_sampler_rejects_invalid_scales(scale):
    sampler = DiscreteGaussianShareSampler.for_testing(1)
    with pytest.raises(DistributedNoiseError) as excinfo:
        sampler.sample(scale)
    assert _code(excinfo) == "invalid_sampler_scale"


def test_client_share_refuses_test_sampler_unless_explicit():
    calibration = _calibration()
    signal = [0] * calibration.dimension
    with pytest.raises(DistributedNoiseError) as excinfo:
        noised_client_share(
            calibration, signal, DiscreteGaussianShareSampler.for_testing(1)
        )
    assert _code(excinfo) == "test_sampler_refused"
    with pytest.raises(DistributedNoiseError) as excinfo:
        noised_client_share(calibration, signal, random.SystemRandom())
    assert _code(excinfo) == "invalid_sampler"
    with pytest.raises(DistributedNoiseError) as excinfo:
        noised_client_share(
            object(), signal, DiscreteGaussianShareSampler.for_testing(1)
        )
    assert _code(excinfo) == "invalid_calibration"
    share = noised_client_share(
        calibration, signal, DiscreteGaussianShareSampler(random.SystemRandom())
    )
    assert len(share) == calibration.dimension


@pytest.mark.parametrize(
    ("signal", "code"),
    [
        ([0] * 7, "invalid_signal"),
        ([0] * 9, "invalid_signal"),
        ([True] + [0] * 7, "invalid_signal"),
        ([0.0] * 8, "invalid_signal"),
        ("00000000", "invalid_signal"),
        (None, "invalid_signal"),
        ([17] + [0] * 7, "signal_out_of_bounds"),
        ([-17] + [0] * 7, "signal_out_of_bounds"),
        ([16, 16, 0, 0, 0, 0, 0, 0], "signal_out_of_bounds"),
    ],
)
def test_client_share_validates_signal_before_sampling(signal, code):
    calibration = _calibration()
    sampler = DiscreteGaussianShareSampler.for_testing(9)
    state = sampler._rng.getstate()
    with pytest.raises(DistributedNoiseError) as excinfo:
        noised_client_share(calibration, signal, sampler, allow_test_sampler=True)
    assert _code(excinfo) == code
    assert sampler._rng.getstate() == state


def test_signal_at_enforced_bounds_is_accepted():
    calibration = _calibration()
    sampler = DiscreteGaussianShareSampler.for_testing(9)
    for signal in ([16] + [0] * 7, [-16, 10, 0, 0, 0, 0, 0, 0]):
        share = noised_client_share(
            calibration, signal, sampler, allow_test_sampler=True
        )
        assert len(share) == 8


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"client_count": 1}, "invalid_client_count"),
        ({"client_count": True}, "invalid_client_count"),
        ({"client_count": 2**21}, "invalid_client_count"),
        ({"client_count": 5.0}, "invalid_client_count"),
        ({"recovery_threshold": 0}, "invalid_recovery_threshold"),
        ({"recovery_threshold": 6}, "invalid_recovery_threshold"),
        ({"recovery_threshold": True}, "invalid_recovery_threshold"),
        ({"target_aggregate_variance": 0}, "invalid_target_variance"),
        ({"target_aggregate_variance": -3}, "invalid_target_variance"),
        ({"target_aggregate_variance": float("nan")}, "invalid_target_variance"),
        ({"target_aggregate_variance": float("inf")}, "invalid_target_variance"),
        ({"target_aggregate_variance": True}, "invalid_target_variance"),
        ({"target_aggregate_variance": "300"}, "invalid_target_variance"),
        ({"target_aggregate_variance": 2**200}, "invalid_target_variance"),
        ({"target_aggregate_variance": 2}, "share_scale_below_minimum"),
        ({"dimension": 0}, "invalid_dimension"),
        ({"dimension": True}, "invalid_dimension"),
        ({"modulus": 2}, "invalid_modulus"),
        ({"modulus": True}, "invalid_modulus"),
        ({"modulus": 2**600}, "invalid_modulus"),
        ({"clip_l2_norm": 0}, "invalid_clip_bound"),
        ({"clip_l2_norm": float("inf")}, "invalid_clip_bound"),
        ({"clip_l2_norm": False}, "invalid_clip_bound"),
        ({"tail_bound_bits": 19}, "invalid_tail_bound"),
        ({"tail_bound_bits": 257}, "invalid_tail_bound"),
        ({"tail_bound_bits": True}, "invalid_tail_bound"),
    ],
)
def test_configuration_negative_controls(overrides, code):
    with pytest.raises(DistributedNoiseError) as excinfo:
        _calibration(**overrides)
    assert _code(excinfo) == code


def test_float_inputs_convert_exactly():
    calibration = _calibration(target_aggregate_variance=300.5, clip_l2_norm=15.5)
    assert calibration.target_aggregate_variance == Fraction(601, 2)
    assert calibration.clip_l2_norm == Fraction(31, 2)
    assert calibration.aggregate_signal_bound == 5 * 16


def test_calibration_is_immutable_and_repr_is_value_free():
    calibration = _calibration(modulus=987_654_321)
    with pytest.raises(AttributeError):
        calibration.share_scale = Fraction(1)
    text = repr(calibration)
    assert "987654321" not in text and "300" not in text
    record = calibration.mechanism_record()
    assert "987654321" not in repr(record)
    sampler = DiscreteGaussianShareSampler.for_testing(123456789)
    assert "123456789" not in repr(sampler)
    assert type(calibration) is DistributedNoiseCalibration


def test_mechanism_record_declares_boundary_and_is_value_free():
    calibration = _calibration()
    record = calibration.mechanism_record()
    assert isinstance(record, DistributedNoiseMechanismRecord)
    payload = record.to_dict()
    assert payload["schema_version"] == FEDERATED_DISTRIBUTED_NOISE_SCHEMA_VERSION
    assert payload["trust_boundary"] == "coordinator_untrusted"
    assert payload["noise_placement"] == "client_shares"
    assert payload["adjacency"] == "zero_one_client_update"
    assert payload["participation"] == "fixed_survivor_set"
    assert payload["dropout_model"] == "data_independent"
    assert payload["calibration_basis"] == "recovery_threshold"
    assert payload["survivor_rule"] == "abort_below_recovery_threshold"
    assert payload["sampler"] == "cks20_exact_discrete_gaussian"
    assert payload["recovery_threshold"] == 3
    assert payload["client_count"] == 5
    assert Fraction(payload["share_scale"]) == calibration.share_scale
    assert Fraction(payload["target_aggregate_variance"]) == 300
    assert Fraction(payload["zcdp_rho_bound"]) >= calibration.zcdp_rho_bound
    assert Fraction(payload["zcdp_rho_bound"]) <= calibration.zcdp_rho_bound * (
        1 + Fraction(1, 10**5)
    )
    assert Fraction(payload["sum_closeness_tau_bound"]) >= (
        calibration.sum_closeness_tau_bound
    )
    body = {key: value for key, value in payload.items() if key != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    assert payload["digest"] == (
        "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()
    )
    assert record.digest == payload["digest"]
    lowered = canonical.lower()
    for forbidden in (
        "seed",
        "draw",
        "sample_value",
        "client_id",
        "site",
        "share_values",
    ):
        assert forbidden not in lowered


def test_record_is_independent_of_noise_draws_and_seeds():
    calibration = _calibration()
    before = calibration.mechanism_record().to_dict()
    shares = []
    for seed in (1, 2):
        sampler = DiscreteGaussianShareSampler.for_testing(seed)
        shares.append(
            noised_client_share(calibration, [1] * 8, sampler, allow_test_sampler=True)
        )
    assert shares[0] != shares[1]
    after = calibration.mechanism_record().to_dict()
    assert before == after
    text = json.dumps(after)
    for share in shares:
        for value in share:
            if value > 10**6:
                assert str(value) not in text


def test_record_digest_changes_with_calibration():
    first = _calibration().mechanism_record().digest
    second = _calibration(recovery_threshold=4).mechanism_record().digest
    assert first != second
    assert first == _calibration().mechanism_record().digest


def test_sampler_round_limit_fails_closed(monkeypatch):
    monkeypatch.setattr(noise_module, "_bernoulli_exp_neg", lambda rng, gamma: False)
    sampler = DiscreteGaussianShareSampler.for_testing(1)
    with pytest.raises(DistributedNoiseError) as excinfo:
        sampler.sample(4)
    assert _code(excinfo) == "sampler_round_limit"
