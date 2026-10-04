"""Offline contract tests for the round-aware Renyi privacy accountant."""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import FrozenInstanceError

import pytest

from openmed.training import federated
from openmed.training.federated import rdp_accountant

DEFAULT_RENYI_ORDERS = rdp_accountant.DEFAULT_RENYI_ORDERS
MAX_ACCOUNTED_ROUNDS = rdp_accountant.MAX_ACCOUNTED_ROUNDS
MAX_CLIPPING_NORM = rdp_accountant.MAX_CLIPPING_NORM
MAX_NOISE_MULTIPLIER = rdp_accountant.MAX_NOISE_MULTIPLIER
MAX_RENYI_ORDERS = rdp_accountant.MAX_RENYI_ORDERS
MAX_RENYI_ORDER = rdp_accountant.MAX_RENYI_ORDER
RDP_ACCOUNTANT_SCHEMA_VERSION = rdp_accountant.RDP_ACCOUNTANT_SCHEMA_VERSION
RDP_ACCOUNTING_REASON_CODES = rdp_accountant.RDP_ACCOUNTING_REASON_CODES
FederatedRenyiRound = rdp_accountant.FederatedRenyiRound
RdpAccountantError = rdp_accountant.RdpAccountantError
RenyiAccountingReport = rdp_accountant.RenyiAccountingReport
RenyiFederationPolicy = rdp_accountant.RenyiFederationPolicy
RenyiRoundDecision = rdp_accountant.RenyiRoundDecision
account_federated_rounds = rdp_accountant.account_federated_rounds
compose_renyi_curve = rdp_accountant.compose_renyi_curve
evaluate_federation_round = rdp_accountant.evaluate_federation_round
fingerprint_accounting_report = rdp_accountant.fingerprint_accounting_report
renyi_to_dp = rdp_accountant.renyi_to_dp
round_renyi_epsilon = rdp_accountant.round_renyi_epsilon

REPORT_FIELDS = ("round_count", "orders", "epsilons", "delta", "epsilon")
CLIENT_SENTINEL = "site-4412-participant-private-value"
MIN_DELTA = 1e-5


def make_round(
    *,
    round_index: int = 0,
    sampling_rate: float = 1.0,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 1.0,
) -> FederatedRenyiRound:
    """Build a round with explicit, synthetic mechanics."""

    return FederatedRenyiRound(
        round_index=round_index,
        sampling_rate=sampling_rate,
        clipping_norm=clipping_norm,
        noise_multiplier=noise_multiplier,
    )


def schedule(count: int, **kwargs: float) -> list[FederatedRenyiRound]:
    """Build ``count`` rounds with sequential indices."""

    return [make_round(round_index=index, **kwargs) for index in range(count)]


class OversizedSequence:
    """A sequence that reports a length above a cap without materializing."""

    def __len__(self) -> int:
        return MAX_ACCOUNTED_ROUNDS + 1

    def __getitem__(self, index: int) -> object:
        raise IndexError(index)


def test_subpackage_reexports_the_module_surface() -> None:
    assert federated.__all__ == rdp_accountant.__all__
    for name in rdp_accountant.__all__:
        assert getattr(federated, name) is getattr(rdp_accountant, name)


def test_training_package_lazily_exposes_accountant_names() -> None:
    import openmed.training as training

    for name in rdp_accountant.__all__:
        assert getattr(training, name) is getattr(rdp_accountant, name)


def test_schema_version_names_the_contract() -> None:
    assert RDP_ACCOUNTANT_SCHEMA_VERSION == (
        "openmed.training.federated.rdp_accountant.v1"
    )


def test_default_orders_are_strictly_increasing_integers() -> None:
    assert DEFAULT_RENYI_ORDERS == (2, 4, 8, 16, 32, 64)
    assert all(type(order) is int for order in DEFAULT_RENYI_ORDERS)
    assert list(DEFAULT_RENYI_ORDERS) == sorted(set(DEFAULT_RENYI_ORDERS))
    assert len(DEFAULT_RENYI_ORDERS) <= MAX_RENYI_ORDERS


def test_reason_codes_are_a_fixed_vocabulary() -> None:
    assert RDP_ACCOUNTING_REASON_CODES == (
        "within_policy",
        "epsilon_exceeded",
        "delta_exceeded",
        "round_limit_exceeded",
    )


@pytest.mark.parametrize("order", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("noise_multiplier", [0.5, 1.0, 2.0, 8.0])
def test_full_participation_round_matches_gaussian_rdp(
    order: int, noise_multiplier: float
) -> None:
    round_ = make_round(sampling_rate=1.0, noise_multiplier=noise_multiplier)
    expected = order / (2.0 * noise_multiplier * noise_multiplier)
    assert round_renyi_epsilon(round_, order) == expected


def test_subsampled_round_matches_the_convexity_bound() -> None:
    round_ = make_round(sampling_rate=0.5, noise_multiplier=1.0)
    expected = math.log(1.0 - 0.5 + 0.5 * math.exp(1.0))
    assert round_renyi_epsilon(round_, 2) == pytest.approx(expected, rel=1e-15)
    assert round_renyi_epsilon(round_, 2) < 1.0


@pytest.mark.parametrize("sampling_rate", [0.001, 0.01, 0.1, 0.5, 0.99])
@pytest.mark.parametrize("order", [2, 8, 64])
def test_subsampled_bound_is_below_full_participation(
    sampling_rate: float, order: int
) -> None:
    sampled = make_round(sampling_rate=sampling_rate, noise_multiplier=1.0)
    full = make_round(sampling_rate=1.0, noise_multiplier=1.0)
    assert 0.0 < round_renyi_epsilon(sampled, order)
    assert round_renyi_epsilon(sampled, order) < round_renyi_epsilon(full, order)


@pytest.mark.parametrize("order", [2, 8, 64])
def test_round_epsilon_decreases_with_the_noise_multiplier(order: int) -> None:
    values = [
        round_renyi_epsilon(make_round(noise_multiplier=value), order)
        for value in (0.5, 1.0, 2.0, 4.0, 16.0)
    ]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("order", [2, 8, 64])
def test_round_epsilon_increases_with_the_sampling_rate(order: int) -> None:
    values = [
        round_renyi_epsilon(make_round(sampling_rate=value), order)
        for value in (0.01, 0.05, 0.2, 0.5, 1.0)
    ]
    assert values == sorted(values)


def test_subsampled_bound_needs_no_overflow_for_tiny_noise() -> None:
    round_ = make_round(sampling_rate=0.01, noise_multiplier=1e-3)
    value = round_renyi_epsilon(round_, 64)
    assert math.isfinite(value)
    assert value > 0.0
    full = round_renyi_epsilon(make_round(noise_multiplier=1e-3), 64)
    assert value < full


def test_clipping_norm_does_not_move_the_bound() -> None:
    small = make_round(clipping_norm=1.0, noise_multiplier=2.0)
    large = make_round(clipping_norm=1e6, noise_multiplier=2.0)
    assert round_renyi_epsilon(small, 8) == round_renyi_epsilon(large, 8)


def test_compose_curve_sums_identical_rounds() -> None:
    single = compose_renyi_curve(schedule(1), DEFAULT_RENYI_ORDERS)
    doubled = compose_renyi_curve(schedule(2), DEFAULT_RENYI_ORDERS)
    assert doubled == tuple(2.0 * value for value in single)
    assert single == tuple(order / 2.0 for order in DEFAULT_RENYI_ORDERS)


def test_compose_curve_ignores_input_order() -> None:
    forwards = schedule(4, sampling_rate=0.25, noise_multiplier=2.0)
    backwards = list(reversed(forwards))
    assert compose_renyi_curve(forwards) == compose_renyi_curve(backwards)


def test_compose_curve_of_an_empty_schedule_is_all_zero() -> None:
    assert compose_renyi_curve(()) == (0.0,) * len(DEFAULT_RENYI_ORDERS)


def test_compose_curve_is_monotone_in_round_count() -> None:
    curves = [
        compose_renyi_curve(schedule(count, sampling_rate=0.1), (2, 8))
        for count in range(0, 5)
    ]
    for earlier, later in zip(curves, curves[1:]):
        assert earlier[0] < later[0]
        assert earlier[1] < later[1]


def test_renyi_to_dp_matches_the_closed_form_for_one_order() -> None:
    delta = 1e-6
    assert renyi_to_dp((1.0,), (2,), delta) == 1.0 + math.log(1.0 / delta)


def test_renyi_to_dp_picks_the_lowest_conversion() -> None:
    orders = (2, 4, 8, 16)
    curve = (8.0, 4.0, 2.0, 1.0)
    delta = 1e-5
    expected = min(
        value + math.log(1.0 / delta) / (order - 1)
        for order, value in zip(orders, curve)
    )
    assert renyi_to_dp(curve, orders, delta) == expected


def test_account_without_rounds_reports_zero_epsilon() -> None:
    report = account_federated_rounds((), delta=MIN_DELTA)
    assert report.round_count == 0
    assert report.epsilons == (0.0,) * len(DEFAULT_RENYI_ORDERS)
    assert report.epsilon == 0.0
    assert report.delta == MIN_DELTA


def test_account_reports_the_composed_curve() -> None:
    rounds = schedule(3, sampling_rate=0.2, noise_multiplier=4.0)
    report = account_federated_rounds(rounds, orders=(2, 8), delta=MIN_DELTA)
    assert report.round_count == 3
    assert report.orders == (2, 8)
    assert report.epsilons == compose_renyi_curve(rounds, (2, 8))
    assert report.epsilon == renyi_to_dp(report.epsilons, (2, 8), MIN_DELTA)


def test_account_epsilon_grows_with_the_round_count() -> None:
    values = [
        account_federated_rounds(schedule(count), delta=MIN_DELTA).epsilon
        for count in range(0, 5)
    ]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert all(value > 0.0 for value in values[1:])


def test_account_epsilon_shrinks_with_more_noise() -> None:
    quiet = account_federated_rounds(schedule(3, noise_multiplier=8.0), delta=MIN_DELTA)
    loud = account_federated_rounds(schedule(3, noise_multiplier=1.0), delta=MIN_DELTA)
    assert quiet.epsilon < loud.epsilon


def test_golden_report_for_two_full_participation_rounds() -> None:
    report = account_federated_rounds(schedule(2), orders=(2, 4), delta=0.5)
    slack = math.log(2.0)
    assert report.epsilons == (2.0, 4.0)
    assert report.epsilon == min(2.0 + slack, 4.0 + slack / 3.0)
    assert report.to_dict() == {
        "round_count": 2,
        "orders": [2, 4],
        "epsilons": [2.0, 4.0],
        "delta": 0.5,
        "epsilon": min(2.0 + slack, 4.0 + slack / 3.0),
    }


def test_report_to_dict_field_order_is_fixed() -> None:
    report = account_federated_rounds(schedule(1), delta=MIN_DELTA)
    assert tuple(report.to_dict()) == REPORT_FIELDS


def test_report_json_is_canonical_and_reproducible() -> None:
    first = account_federated_rounds(
        schedule(2, sampling_rate=0.3, noise_multiplier=3.0), delta=MIN_DELTA
    )
    second = account_federated_rounds(
        schedule(2, sampling_rate=0.3, noise_multiplier=3.0), delta=MIN_DELTA
    )
    assert first.to_json() == second.to_json()
    assert first.to_json() == json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":")
    )
    assert json.loads(first.to_json()) == first.to_dict()
    assert ", " not in first.to_json()


def test_report_fingerprint_is_domain_separated_and_stable() -> None:
    report = account_federated_rounds(schedule(2), delta=MIN_DELTA)
    digest = fingerprint_accounting_report(report)
    assert digest == fingerprint_accounting_report(report)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    other = account_federated_rounds(schedule(2, noise_multiplier=2.0), delta=MIN_DELTA)
    assert fingerprint_accounting_report(other) != digest


def test_fingerprint_rejects_a_foreign_report() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        fingerprint_accounting_report("not-a-report")  # type: ignore[arg-type]
    assert caught.value.category == "report_invalid"


def test_gate_allows_a_round_within_policy() -> None:
    policy = RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA, orders=(2, 4))
    decision = evaluate_federation_round(policy, (), make_round())
    assert decision.allowed is True
    assert decision.reason_code == "within_policy"
    assert decision.report.round_count == 1
    assert decision.report.epsilon <= policy.max_epsilon


def test_gate_blocks_a_round_beyond_the_epsilon_ceiling() -> None:
    policy = RenyiFederationPolicy(max_epsilon=1.0, max_delta=MIN_DELTA, orders=(2, 4))
    decision = evaluate_federation_round(policy, (), make_round())
    assert decision.allowed is False
    assert decision.reason_code == "epsilon_exceeded"
    assert decision.report.epsilon > policy.max_epsilon


def test_gate_blocks_a_target_delta_above_the_policy() -> None:
    policy = RenyiFederationPolicy(max_epsilon=50.0, max_delta=MIN_DELTA, orders=(2, 4))
    decision = evaluate_federation_round(
        policy, (), make_round(), target_delta=MIN_DELTA * 10.0
    )
    assert decision.allowed is False
    assert decision.reason_code == "delta_exceeded"
    assert decision.report.delta == policy.max_delta


def test_gate_blocks_when_the_round_limit_is_reached() -> None:
    policy = RenyiFederationPolicy(
        max_epsilon=50.0, max_delta=MIN_DELTA, orders=(2, 4), max_rounds=1
    )
    decision = evaluate_federation_round(policy, schedule(1), make_round(round_index=9))
    assert decision.allowed is False
    assert decision.reason_code == "round_limit_exceeded"
    assert decision.report.round_count == 2


def test_gate_charges_the_candidate_on_top_of_the_schedule() -> None:
    policy = RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA, orders=(2, 4))
    candidate = make_round(round_index=9)
    assert evaluate_federation_round(policy, (), candidate).allowed is True
    decision = evaluate_federation_round(policy, schedule(3), candidate)
    assert decision.allowed is False
    assert decision.reason_code == "epsilon_exceeded"
    assert decision.report.round_count == 4


def test_gate_treats_a_smaller_target_delta_as_stricter() -> None:
    policy = RenyiFederationPolicy(max_epsilon=1e6, max_delta=MIN_DELTA, orders=(2,))
    relaxed = evaluate_federation_round(
        policy, (), make_round(), target_delta=MIN_DELTA
    )
    strict = evaluate_federation_round(policy, (), make_round(), target_delta=1e-9)
    assert strict.report.epsilon > relaxed.report.epsilon
    assert strict.report.delta == 1e-9


def test_decision_payload_has_a_fixed_shape() -> None:
    policy = RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA)
    decision = evaluate_federation_round(policy, (), make_round())
    payload = decision.to_dict()
    assert tuple(payload) == ("allowed", "reason_code", "report")
    assert payload["report"] == decision.report.to_dict()
    assert payload["reason_code"] in RDP_ACCOUNTING_REASON_CODES


def test_decision_json_is_canonical_and_reproducible() -> None:
    policy = RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA)
    first = evaluate_federation_round(policy, (), make_round())
    second = evaluate_federation_round(policy, (), make_round())
    assert first.to_json() == second.to_json()
    assert first.to_json() == json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":")
    )
    assert json.loads(first.to_json()) == first.to_dict()


def test_decision_rejects_unknown_reason_codes() -> None:
    report = account_federated_rounds((), delta=MIN_DELTA)
    with pytest.raises(RdpAccountantError) as caught:
        RenyiRoundDecision(allowed=True, reason_code="maybe", report=report)
    assert caught.value.category == "reason_code_invalid"


def test_decision_rejects_a_non_boolean_verdict() -> None:
    report = account_federated_rounds((), delta=MIN_DELTA)
    with pytest.raises(RdpAccountantError) as caught:
        RenyiRoundDecision(
            allowed=1,  # type: ignore[arg-type]
            reason_code="within_policy",
            report=report,
        )
    assert caught.value.category == "allowed_invalid"


def test_decision_rejects_a_foreign_report() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        RenyiRoundDecision(
            allowed=True,
            reason_code="within_policy",
            report="report",  # type: ignore[arg-type]
        )
    assert caught.value.category == "report_invalid"


@pytest.mark.parametrize(
    "value",
    [0, -0.5, 1.0000001, "0.5", True, None, float("nan"), float("inf")],
)
def test_sampling_rate_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        make_round(sampling_rate=value)  # type: ignore[arg-type]
    assert caught.value.category == "sampling_rate_invalid"


@pytest.mark.parametrize(
    "value", [0, -1.0, MAX_CLIPPING_NORM * 10.0, "1.0", False, None]
)
def test_clipping_norm_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        make_round(clipping_norm=value)  # type: ignore[arg-type]
    assert caught.value.category == "clipping_norm_invalid"


@pytest.mark.parametrize(
    "value", [0, -1.0, MAX_NOISE_MULTIPLIER * 10.0, "2.0", True, None]
)
def test_noise_multiplier_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        make_round(noise_multiplier=value)  # type: ignore[arg-type]
    assert caught.value.category == "noise_multiplier_invalid"


@pytest.mark.parametrize("value", [-1, 1.5, True, "0", None, MAX_ACCOUNTED_ROUNDS + 1])
def test_round_index_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        make_round(round_index=value)  # type: ignore[arg-type]
    assert caught.value.category == "round_index_invalid"


@pytest.mark.parametrize("value", [1, 0, -2, MAX_RENYI_ORDER + 1, 2.0, "2", True])
def test_order_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        round_renyi_epsilon(make_round(), value)  # type: ignore[arg-type]
    assert caught.value.category == "order_invalid"


@pytest.mark.parametrize(
    "value",
    [
        (),
        (2, 2),
        (4, 2),
        (1,),
        (MAX_RENYI_ORDER + 1,),
        (2, "4"),
        "24",
        24,
    ],
)
def test_orders_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        compose_renyi_curve((), value)  # type: ignore[arg-type]
    assert caught.value.category in {"orders_invalid", "orders_entry_invalid"}


def test_orders_validation_rejects_too_many_entries() -> None:
    too_many = tuple(range(2, MAX_RENYI_ORDERS + 3))
    with pytest.raises(RdpAccountantError) as caught:
        compose_renyi_curve((), too_many)
    assert caught.value.category == "orders_invalid"


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.5, "0.1", True, None, float("nan")])
def test_delta_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        account_federated_rounds((), delta=value)  # type: ignore[arg-type]
    assert caught.value.category == "delta_invalid"


@pytest.mark.parametrize("value", ["rounds", b"rounds", 4, {"round_index": 0}, None])
def test_schedule_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        compose_renyi_curve(value)  # type: ignore[arg-type]
    assert caught.value.category == "rounds_invalid"


def test_schedule_validation_rejects_foreign_entries() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        compose_renyi_curve([make_round(), {"round_index": 1}])  # type: ignore[list-item]
    assert caught.value.category == "rounds_invalid"


def test_schedule_validation_rejects_duplicate_round_indices() -> None:
    rounds = [make_round(round_index=3), make_round(round_index=3)]
    with pytest.raises(RdpAccountantError) as caught:
        compose_renyi_curve(rounds)
    assert caught.value.category == "duplicate_round_index"


def test_schedule_validation_caps_the_round_count() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        compose_renyi_curve(OversizedSequence())  # type: ignore[arg-type]
    assert caught.value.category == "rounds_invalid"


def test_round_epsilon_rejects_a_foreign_round() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        round_renyi_epsilon("round", 2)  # type: ignore[arg-type]
    assert caught.value.category == "round_invalid"


def test_curve_and_order_mismatch_is_rejected() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        renyi_to_dp((1.0,), (2, 4), MIN_DELTA)
    assert caught.value.category == "epsilons_invalid"


@pytest.mark.parametrize("value", [-1.0, float("nan"), "1.0", True])
def test_curve_entries_are_validated(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        renyi_to_dp((value,), (2,), MIN_DELTA)  # type: ignore[arg-type]
    assert caught.value.category == "epsilons_invalid"


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), "1.0", True])
def test_policy_epsilon_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        RenyiFederationPolicy(
            max_epsilon=value,  # type: ignore[arg-type]
            max_delta=MIN_DELTA,
        )
    assert caught.value.category == "max_epsilon_invalid"


@pytest.mark.parametrize("value", [0, 1, -0.5, 1.5, "0.1", True, None])
def test_policy_delta_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        RenyiFederationPolicy(
            max_epsilon=10.0,
            max_delta=value,  # type: ignore[arg-type]
        )
    assert caught.value.category == "max_delta_invalid"


@pytest.mark.parametrize("value", [0, -1, MAX_ACCOUNTED_ROUNDS + 1, 1.5, "2"])
def test_policy_round_limit_validation(value: object) -> None:
    with pytest.raises(RdpAccountantError) as caught:
        RenyiFederationPolicy(
            max_epsilon=10.0,
            max_delta=MIN_DELTA,
            max_rounds=value,  # type: ignore[arg-type]
        )
    assert caught.value.category == "max_rounds_invalid"


def test_policy_rejects_an_invalid_order_grid() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA, orders=(4, 2))
    assert caught.value.category == "orders_invalid"


def test_gate_rejects_a_foreign_policy() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        evaluate_federation_round("policy", (), make_round())  # type: ignore[arg-type]
    assert caught.value.category == "policy_invalid"


def test_gate_rejects_a_foreign_candidate() -> None:
    policy = RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA)
    with pytest.raises(RdpAccountantError) as caught:
        evaluate_federation_round(policy, (), "round")  # type: ignore[arg-type]
    assert caught.value.category == "candidate_invalid"


def test_gate_rejects_a_schedule_with_duplicate_indices() -> None:
    policy = RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA)
    rounds = [make_round(round_index=1), make_round(round_index=1)]
    with pytest.raises(RdpAccountantError) as caught:
        evaluate_federation_round(policy, rounds, make_round(round_index=2))
    assert caught.value.category == "duplicate_round_index"


def test_report_rejects_a_curve_of_the_wrong_length() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        RenyiAccountingReport(
            round_count=1,
            orders=(2, 4),
            epsilons=(1.0,),
            delta=MIN_DELTA,
            epsilon=1.0,
        )
    assert caught.value.category == "epsilons_invalid"


def test_report_rejects_a_negative_epsilon() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        RenyiAccountingReport(
            round_count=0,
            orders=(2,),
            epsilons=(0.0,),
            delta=MIN_DELTA,
            epsilon=-1.0,
        )
    assert caught.value.category == "epsilon_invalid"


def test_accounting_error_is_a_value_error_with_a_category() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        make_round(sampling_rate=1.25)
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.category == "sampling_rate_invalid"
    assert str(error) == getattr(error, "args")[0]
    assert "1.25" not in str(error)


def test_error_messages_never_echo_rejected_values() -> None:
    with pytest.raises(RdpAccountantError) as caught:
        make_round(noise_multiplier=1.25e7)
    message = str(caught.value)
    assert caught.value.category == "noise_multiplier_invalid"
    assert "12500000" not in message
    assert "1.25e+07" not in message
    with pytest.raises(RdpAccountantError) as caught:
        compose_renyi_curve((), (2, 3.5))
    assert "3.5" not in str(caught.value)


def test_rounds_and_reports_are_frozen_and_slotted() -> None:
    round_ = make_round()
    policy = RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA)
    report = account_federated_rounds((round_,), delta=MIN_DELTA)
    decision = evaluate_federation_round(policy, (), round_)
    for value, attribute in (
        (round_, "sampling_rate"),
        (policy, "max_epsilon"),
        (report, "epsilon"),
        (decision, "allowed"),
    ):
        assert not hasattr(value, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, 1)


def test_payloads_carry_mechanics_and_counts_only() -> None:
    round_ = make_round()
    report = account_federated_rounds((round_,), delta=MIN_DELTA)
    payload = json.dumps(
        {
            "round": round_.to_dict(),
            "report": report.to_dict(),
            "decision": evaluate_federation_round(
                RenyiFederationPolicy(max_epsilon=10.0, max_delta=MIN_DELTA),
                (),
                round_,
            ).to_dict(),
        },
        default=str,
    )
    assert CLIENT_SENTINEL not in payload
    forbidden = ("client", "site", "patient", "record", "token", "payload", "text")
    for name in forbidden:
        assert name not in payload


def test_round_payload_has_a_fixed_shape() -> None:
    assert make_round().to_dict() == {
        "round_index": 0,
        "sampling_rate": 1.0,
        "clipping_norm": 1.0,
        "noise_multiplier": 1.0,
    }


def test_accountant_source_stays_offline_and_self_contained() -> None:
    source = pathlib.Path(rdp_accountant.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import socket",
        "import ssl",
        "import urllib",
        "import requests",
        "http.client",
        "boto3",
        "openmed.risk",
        "numpy",
    ):
        assert forbidden not in source
