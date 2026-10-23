"""Tests for privacy accounting of private hyperparameter and checkpoint selection.

Reference values come from Papernot and Steinke, "Hyperparameter Tuning with
Renyi Differential Privacy" (ICLR 2022), Definition 1, Theorem 2 (equation 5),
Corollaries 3 and 4 and Theorem 6, and from Liu and Talwar, "Private Selection
from Private Candidates" (STOC 2019), Theorems 1.3 and 3.2. All fixtures are
synthetic; no training is run.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable

import pytest

from openmed.risk.privacy_budget import PrivacyBudgetLedger
from openmed.risk.private_selection import (
    OUTCOME_BANDS,
    PRIVATE_SELECTION_REASONS,
    PrivateSelectionError,
    RandomStoppingScheme,
    TrialMechanism,
    TrialPrivacyCost,
    TuningLedger,
    TuningPromotionEvidence,
    band_outcome,
    poisson_rdp,
    random_stopping_pure_dp_epsilon,
    rdp_to_dp_epsilon,
    truncated_negative_binomial_mean,
    truncated_negative_binomial_rdp,
    truncated_negative_binomial_zcdp_rdp,
    verify_promotion_evidence,
)

SITE_SENTINEL = "site-boston-general"
SEED_SENTINEL = "918273645"


class _ScriptedCsprng(random.SystemRandom):
    """Synthetic CSPRNG double that replays scripted uniforms for tests."""

    def __init__(self, values: Iterable[float]) -> None:
        super().__init__()
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


class _ReplayCsprng(random.SystemRandom):
    """Synthetic CSPRNG double backed by a test-only generator for statistics."""

    def __init__(self, generator: random.Random) -> None:
        super().__init__()
        self._generator = generator

    def random(self) -> float:
        return self._generator.random()


def _digest(index: int) -> str:
    return "sha256:" + format(index, "x").rjust(64, "0")


def _mechanism(noise: float = 1.0, rounds: int = 10) -> TrialMechanism:
    return TrialMechanism(
        noise_multiplier=noise,
        clipping_norm=1.0,
        sampling_rate=0.01,
        round_count=rounds,
        learning_rate=0.05,
    )


def _pure_accountant(epsilon_per_round: float = 0.1):
    def accountant(mechanism: TrialMechanism) -> TrialPrivacyCost:
        return TrialPrivacyCost(epsilon=epsilon_per_round * mechanism.round_count)

    return accountant


def _approx_accountant(mechanism: TrialMechanism) -> TrialPrivacyCost:
    return TrialPrivacyCost(epsilon=0.5 * mechanism.noise_multiplier, delta=1e-6)


def _reason(exc_info: pytest.ExceptionInfo[PrivateSelectionError]) -> str:
    assert exc_info.value.reason in PRIVATE_SELECTION_REASONS
    return exc_info.value.reason


# ---------------------------------------------------------------------------
# Published reference values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0, 2.0])
def test_geometric_pure_dp_bound_is_three_epsilon(epsilon: float) -> None:
    # Liu and Talwar (2019) Theorem 3.2; Papernot and Steinke Corollary 3, eta=1.
    assert random_stopping_pure_dp_epsilon(epsilon, eta=1.0) == pytest.approx(
        3.0 * epsilon, rel=0, abs=1e-15
    )


@pytest.mark.parametrize(
    ("eta", "factor"), [(0.0, 2.0), (0.5, 2.5), (1.0, 3.0), (2.0, 4.0)]
)
def test_truncated_negative_binomial_pure_dp_bound_matches_corollary_3(
    eta: float, factor: float
) -> None:
    assert random_stopping_pure_dp_epsilon(0.4, eta=eta) == pytest.approx(
        factor * 0.4, rel=0, abs=1e-15
    )


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.25, 0.5, 0.9])
def test_geometric_mean_is_one_over_gamma(gamma: float) -> None:
    # Definition 1 with eta=1; Liu and Talwar Theorem 1.3(b).
    assert truncated_negative_binomial_mean(1.0, gamma) == pytest.approx(1.0 / gamma)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_logarithmic_mean_matches_definition_1(gamma: float) -> None:
    expected = (1.0 / gamma - 1.0) / math.log(1.0 / gamma)
    assert truncated_negative_binomial_mean(0.0, gamma) == pytest.approx(expected)


@pytest.mark.parametrize(("eta", "gamma"), [(-0.5, 0.05), (0.5, 0.1), (3.0, 0.3)])
def test_negative_binomial_mean_matches_definition_1(eta: float, gamma: float) -> None:
    expected = eta * (1.0 - gamma) / (gamma * (1.0 - gamma**eta))
    assert truncated_negative_binomial_mean(eta, gamma) == pytest.approx(expected)


def test_theorem_2_equation_5_reference_value() -> None:
    # eps + (1+eta)(1-1/lam_hat) eps_hat + (1+eta) log(1/gamma)/lam_hat
    #     + log E[K]/(lam-1), with eta=0, gamma=0.1, lam=2, lam_hat=4.
    mean = 9.0 / math.log(10.0)
    expected = 1.0 + 0.75 * 0.5 + math.log(10.0) / 4.0 + math.log(mean) / 1.0
    value = truncated_negative_binomial_rdp(2.0, 1.0, 4.0, 0.5, eta=0.0, gamma=0.1)
    assert value == pytest.approx(expected, rel=1e-12)
    assert value == pytest.approx(3.313838, abs=1e-6)


def test_theorem_2_tends_to_corollary_3_for_pure_dp() -> None:
    # Pure DP is (inf, eps)-RDP; equation 5 tends to (2+eta) eps as both
    # orders grow, which is how the paper derives Corollary 3.
    value = truncated_negative_binomial_rdp(1e9, 0.3, 1e9, 0.3, eta=1.0, gamma=0.2)
    assert value == pytest.approx(0.9, abs=1e-6)


@pytest.mark.parametrize("order", [1.5, 3.0, 8.0, 32.0])
def test_corollary_4_matches_theorem_2_at_the_published_optimum(order: float) -> None:
    rho, eta, gamma = 0.1, 0.0, 0.05
    mean = truncated_negative_binomial_mean(eta, gamma)
    order_hat = math.sqrt(math.log(1.0 / gamma) / rho)
    threshold = 1.0 + math.sqrt(math.log(mean) / rho)
    corollary = truncated_negative_binomial_zcdp_rdp(order, rho, eta=eta, gamma=gamma)
    if order > threshold:
        theorem = truncated_negative_binomial_rdp(
            order, rho * order, order_hat, rho * order_hat, eta=eta, gamma=gamma
        )
    else:
        theorem = truncated_negative_binomial_rdp(
            threshold,
            rho * threshold,
            order_hat,
            rho * order_hat,
            eta=eta,
            gamma=gamma,
        )
    assert corollary == pytest.approx(theorem, rel=1e-12)


def test_theorem_6_poisson_reference_value_and_condition() -> None:
    # eps + mu * delta_hat + log(mu)/(lam-1) when exp(eps_hat) <= 1 + 1/(lam-1).
    value = poisson_rdp(2.0, 0.5, 0.5, 1e-6, mu=10.0)
    assert value == pytest.approx(0.5 + 10.0 * 1e-6 + math.log(10.0), rel=1e-12)
    with pytest.raises(PrivateSelectionError) as exc_info:
        poisson_rdp(4.0, 0.5, 0.5, 1e-6, mu=10.0)
    assert _reason(exc_info) == "bound_not_applicable"


def test_rdp_conversion_matches_mironov_proposition_3() -> None:
    curve = ((2.0, 1.0), (8.0, 2.0))
    expected = min(1.0 + math.log(1e5), 2.0 + math.log(1e5) / 7.0)
    assert rdp_to_dp_epsilon(curve, 1e-5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Trial-count distributions and the injected CSPRNG.
# ---------------------------------------------------------------------------


def _geometric_scheme(epsilon: float = 1.0, gamma: float = 0.5) -> RandomStoppingScheme:
    return RandomStoppingScheme(
        distribution="geometric",
        gamma=gamma,
        per_trial=TrialPrivacyCost(epsilon=epsilon),
    )


@pytest.mark.parametrize(("uniform", "expected"), [(0.3, 1), (0.6, 2), (0.8, 3)])
def test_geometric_draw_uses_inverse_cdf_of_the_declared_distribution(
    uniform: float, expected: int
) -> None:
    scheme = _geometric_scheme()
    assert scheme.draw_trial_count(_ScriptedCsprng([uniform])) == expected


@pytest.mark.parametrize(("uniform", "expected"), [(0.2, 0), (0.5, 1), (0.9, 2)])
def test_poisson_draw_can_return_zero_trials(uniform: float, expected: int) -> None:
    scheme = RandomStoppingScheme(
        distribution="poisson",
        mu=1.0,
        target_delta=1e-6,
        per_trial=TrialPrivacyCost(epsilon=0.1),
    )
    assert scheme.draw_trial_count(_ScriptedCsprng([uniform])) == expected


@pytest.mark.parametrize(
    ("distribution", "kwargs"),
    [
        ("geometric", {"gamma": 0.2}),
        ("logarithmic", {"gamma": 0.1}),
        ("truncated_negative_binomial", {"eta": 0.5, "gamma": 0.15}),
        ("poisson", {"mu": 4.0, "target_delta": 1e-6}),
    ],
)
def test_draws_follow_the_declared_mean(distribution: str, kwargs: dict) -> None:
    scheme = RandomStoppingScheme(
        distribution=distribution,
        per_trial=TrialPrivacyCost(epsilon=0.1),
        **kwargs,
    )
    rng = _ReplayCsprng(random.Random(20221))
    draws = [scheme.draw_trial_count(rng) for _ in range(20_000)]
    mean = sum(draws) / len(draws)
    variance = sum((value - mean) ** 2 for value in draws) / len(draws)
    tolerance = 5.0 * math.sqrt(variance / len(draws))
    assert abs(mean - scheme.expected_trial_count) <= tolerance


@pytest.mark.parametrize(
    "rng", [random.Random(7), object(), "csprng"], ids=["mersenne", "object", "text"]
)
def test_non_csprng_generators_are_rejected(rng: object) -> None:
    ledger = TuningLedger(
        _pure_accountant(),
        metric_source="public",
        scheme=_geometric_scheme(),
    )
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.draw_trial_count(rng)  # type: ignore[arg-type]
    assert _reason(exc_info) == "rng_not_csprng"


def test_system_csprng_is_accepted_by_default() -> None:
    ledger = TuningLedger(
        _pure_accountant(),
        metric_source="public",
        scheme=_geometric_scheme(gamma=0.9),
    )
    assert ledger.draw_trial_count() >= 1


# ---------------------------------------------------------------------------
# Fixed trial counts under a random-stopping scheme.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["trial_count", "trials", "fixed_trial_count"])
def test_scheme_mapping_with_a_fixed_trial_count_is_rejected(key: str) -> None:
    with pytest.raises(PrivateSelectionError) as exc_info:
        RandomStoppingScheme.from_mapping(
            {
                "distribution": "geometric",
                "gamma": 0.2,
                "per_trial": {"epsilon": 0.5},
                key: 5,
            }
        )
    assert _reason(exc_info) == "fixed_trial_count"


@pytest.mark.parametrize("distribution", ["fixed", "point_mass", "constant"])
def test_point_mass_distribution_is_rejected(distribution: str) -> None:
    with pytest.raises(PrivateSelectionError) as exc_info:
        RandomStoppingScheme(
            distribution=distribution,
            per_trial=TrialPrivacyCost(epsilon=0.5),
        )
    assert _reason(exc_info) == "fixed_trial_count"


def test_recording_trials_without_drawing_the_count_is_rejected() -> None:
    ledger = TuningLedger(
        _pure_accountant(), metric_source="public", scheme=_geometric_scheme()
    )
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_trial(_mechanism(), _digest(1), outcome_band="high")
    assert _reason(exc_info) == "trial_count_not_drawn"
    assert ledger.records == ()


def test_running_a_different_number_of_trials_than_drawn_is_rejected() -> None:
    ledger = TuningLedger(
        _pure_accountant(), metric_source="public", scheme=_geometric_scheme()
    )
    assert ledger.draw_trial_count(_ScriptedCsprng([0.6])) == 2
    ledger.record_trial(_mechanism(), _digest(1), outcome_band="high")
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.account()
    assert _reason(exc_info) == "trial_count_incomplete"
    ledger.record_trial(_mechanism(), _digest(2), outcome_band="low")
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_trial(_mechanism(), _digest(3), outcome_band="low")
    assert _reason(exc_info) == "trial_count_exceeded"


def test_trial_count_cannot_be_redrawn() -> None:
    ledger = TuningLedger(
        _pure_accountant(), metric_source="public", scheme=_geometric_scheme()
    )
    ledger.draw_trial_count(_ScriptedCsprng([0.1]))
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.draw_trial_count(_ScriptedCsprng([0.9]))
    assert _reason(exc_info) == "trial_count_already_drawn"


def test_heavy_tailed_scheme_is_rejected_before_any_trial() -> None:
    with pytest.raises(PrivateSelectionError) as exc_info:
        _geometric_scheme(gamma=0.001)
    assert _reason(exc_info) == "scheme_tail_too_heavy"


# ---------------------------------------------------------------------------
# Default charging: composition of every trial plus selection.
# ---------------------------------------------------------------------------


def _scored_ledger(trials: int = 4) -> TuningLedger:
    ledger = TuningLedger(
        _pure_accountant(0.1),
        metric_source="private",
        scoring_cost=TrialPrivacyCost(epsilon=0.05),
    )
    bands = ("low", "high", "moderate", "very_high")
    for index in range(1, trials + 1):
        ledger.record_trial(
            _mechanism(rounds=10),
            _digest(index),
            outcome_band=bands[(index - 1) % len(bands)],
        )
    return ledger


def test_best_of_k_without_a_scheme_charges_every_trial_and_its_scoring() -> None:
    ledger = _scored_ledger(4)
    ledger.select(4)
    account = ledger.account()
    # 4 trials of 1.0 epsilon plus 4 private candidate scores of 0.05.
    assert account.accounting == "composition"
    assert account.method == "basic_composition"
    assert account.epsilon == pytest.approx(4.2)
    assert account.delta == 0.0
    assert account.trial_count == 4
    selected_only = ledger.records[3].training_cost.epsilon
    assert account.epsilon >= 4 * selected_only


def test_abandoned_trials_are_still_charged() -> None:
    ledger = TuningLedger(_approx_accountant, metric_source="public")
    ledger.record_trial(_mechanism(noise=1.0), _digest(1), outcome_band="high")
    ledger.record_trial(
        _mechanism(noise=2.0),
        _digest(2),
        status="abandoned",
        outcome_band="not_evaluated",
    )
    ledger.select(1)
    account = ledger.account()
    assert account.epsilon == pytest.approx(1.5)
    assert account.delta == pytest.approx(2e-6)


def test_public_metric_selection_still_charges_all_trials() -> None:
    ledger = TuningLedger(_pure_accountant(0.1), metric_source="public")
    for index in range(1, 4):
        ledger.record_trial(_mechanism(), _digest(index), outcome_band="high")
    ledger.select(2)
    assert ledger.account().epsilon == pytest.approx(3.0)


def test_private_metric_source_requires_a_declared_scoring_cost() -> None:
    with pytest.raises(PrivateSelectionError) as exc_info:
        TuningLedger(_pure_accountant(), metric_source="private")
    assert _reason(exc_info) == "scoring_cost_missing"
    with pytest.raises(PrivateSelectionError) as exc_info:
        TuningLedger(
            _pure_accountant(),
            metric_source="private",
            scoring_cost=TrialPrivacyCost(epsilon=0.0),
        )
    assert _reason(exc_info) == "scoring_cost_missing"


# ---------------------------------------------------------------------------
# Early stopping, checkpoint and round selection on private metrics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", ["early_stopping", "checkpoint_selection", "round_selection"]
)
def test_in_trial_selection_is_recorded_and_charged(kind: str) -> None:
    ledger = _scored_ledger(2)
    before = ledger.account().epsilon
    ledger.record_selection(1, kind, evaluation_count=3)
    after = ledger.account()
    assert after.epsilon == pytest.approx(before + 3 * 0.05)
    assert dict(after.selection_counts)[kind] == 3
    assert dict(ledger.records[0].selection_evaluations)[kind] == 3


def test_in_trial_selection_on_public_metrics_is_recorded_without_extra_charge() -> (
    None
):
    ledger = TuningLedger(_pure_accountant(0.1), metric_source="public")
    ledger.record_trial(_mechanism(), _digest(1), outcome_band="high")
    ledger.record_selection(1, "early_stopping", evaluation_count=5)
    account = ledger.account()
    assert account.epsilon == pytest.approx(1.0)
    assert dict(account.selection_counts)["early_stopping"] == 5


def test_best_of_k_is_recorded_only_through_select() -> None:
    ledger = _scored_ledger(2)
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_selection(1, "best_of_k")
    assert _reason(exc_info) == "selection_kind_invalid"
    ledger.select(2)
    assert dict(ledger.account().selection_counts)["best_of_k"] == 1


def test_selected_trial_must_be_completed_and_evaluated() -> None:
    ledger = TuningLedger(_pure_accountant(), metric_source="public")
    ledger.record_trial(_mechanism(), _digest(1), status="abandoned")
    ledger.record_trial(_mechanism(), _digest(2))
    for index in (1, 2):
        with pytest.raises(PrivateSelectionError) as exc_info:
            ledger.select(index)
        assert _reason(exc_info) == "selected_trial_not_eligible"
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.select(3)
    assert _reason(exc_info) == "unknown_trial"


# ---------------------------------------------------------------------------
# Declared random-stopping scheme charges.
# ---------------------------------------------------------------------------


def test_geometric_scheme_charges_three_times_the_declared_trial_cost() -> None:
    scheme = RandomStoppingScheme(
        distribution="geometric",
        gamma=0.5,
        per_trial=TrialPrivacyCost(epsilon=1.1),
    )
    ledger = TuningLedger(
        _pure_accountant(0.1),
        metric_source="private",
        scoring_cost=TrialPrivacyCost(epsilon=0.05),
        scheme=scheme,
    )
    trials = ledger.draw_trial_count(_ScriptedCsprng([0.8]))
    assert trials == 3
    for index in range(1, trials + 1):
        ledger.record_trial(_mechanism(), _digest(index), outcome_band="moderate")
    ledger.select(2)
    account = ledger.account()
    assert account.accounting == "random_stopping"
    assert account.method == "tnb_pure_dp"
    assert account.epsilon == pytest.approx(3.3)
    assert account.delta == 0.0
    # The realized trial count is not exposed under random stopping.
    assert account.trial_count is None
    assert "trial_count" not in account.to_dict()


def test_scheme_charge_does_not_depend_on_the_realized_trial_count() -> None:
    charges = []
    for uniform in (0.1, 0.95):
        ledger = TuningLedger(
            _pure_accountant(0.1),
            metric_source="public",
            scheme=_geometric_scheme(epsilon=1.0),
        )
        trials = ledger.draw_trial_count(_ScriptedCsprng([uniform]))
        for index in range(1, trials + 1):
            ledger.record_trial(_mechanism(), _digest(index), outcome_band="high")
        charges.append(ledger.account().epsilon)
    assert charges[0] == charges[1] == pytest.approx(3.0)


def test_trial_exceeding_the_declared_per_trial_cost_is_rejected() -> None:
    ledger = TuningLedger(
        _pure_accountant(0.1),
        metric_source="private",
        scoring_cost=TrialPrivacyCost(epsilon=0.05),
        scheme=_geometric_scheme(epsilon=1.1),
    )
    ledger.draw_trial_count(_ScriptedCsprng([0.6]))
    ledger.record_trial(_mechanism(), _digest(1), outcome_band="high")
    # A second private checkpoint evaluation would push trial 1 above 1.1.
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_selection(1, "checkpoint_selection", evaluation_count=2)
    assert _reason(exc_info) == "trial_exceeds_declared_cost"
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_trial(_mechanism(rounds=20), _digest(2), outcome_band="high")
    assert _reason(exc_info) == "trial_exceeds_declared_cost"


def test_rdp_scheme_charges_theorem_2_converted_to_approximate_dp() -> None:
    rho = 0.1
    orders = (2.0, 4.0, 8.0, 16.0, 32.0)
    curve = tuple((order, rho * order) for order in orders)
    scheme = RandomStoppingScheme(
        distribution="logarithmic",
        gamma=0.1,
        target_delta=1e-6,
        per_trial=TrialPrivacyCost(epsilon=5.0, delta=1e-7, rdp=curve),
    )
    expected_curve = []
    for order in orders:
        expected_curve.append(
            (
                order,
                min(
                    truncated_negative_binomial_rdp(
                        order, rho * order, hat, rho * hat, eta=0.0, gamma=0.1
                    )
                    for hat in orders
                ),
            )
        )
    bound = scheme.bound()
    assert bound.method == "tnb_rdp"
    assert bound.delta == 1e-6
    assert bound.epsilon == pytest.approx(rdp_to_dp_epsilon(expected_curve, 1e-6))


def test_poisson_scheme_charges_theorem_6() -> None:
    orders = (2.0, 3.0, 4.0)
    curve = tuple((order, 0.05 * order) for order in orders)
    scheme = RandomStoppingScheme(
        distribution="poisson",
        mu=5.0,
        target_delta=1e-6,
        per_trial=TrialPrivacyCost(epsilon=0.4, delta=1e-8, rdp=curve),
    )
    usable = [
        (order, 0.05 * order + 5.0 * 1e-8 + math.log(5.0) / (order - 1.0))
        for order in orders
        if math.exp(0.4) <= 1.0 + 1.0 / (order - 1.0)
    ]
    assert [order for order, _ in usable] == [2.0, 3.0]
    bound = scheme.bound()
    assert bound.method == "poisson_rdp"
    assert bound.epsilon == pytest.approx(rdp_to_dp_epsilon(usable, 1e-6))


def test_approximate_dp_trial_without_rdp_curve_cannot_use_a_scheme() -> None:
    with pytest.raises(PrivateSelectionError) as exc_info:
        RandomStoppingScheme(
            distribution="geometric",
            gamma=0.5,
            per_trial=TrialPrivacyCost(epsilon=1.0, delta=1e-6),
        )
    assert _reason(exc_info) == "bound_not_applicable"


# ---------------------------------------------------------------------------
# Charging the shared privacy-budget ledger.
# ---------------------------------------------------------------------------


def test_charge_records_the_composed_account_once() -> None:
    budget = PrivacyBudgetLedger({"tuning": {"epsilon": 10.0, "delta": 1e-5}})
    ledger = _scored_ledger(3)
    ledger.select(1)
    decision = ledger.charge(budget, "tuning")
    assert decision.allowed is True
    assert budget.render_counts_only()["spent_epsilon"] == pytest.approx(3.15)
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.charge(budget, "tuning")
    assert _reason(exc_info) == "already_charged"
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_trial(_mechanism(), _digest(9), outcome_band="high")
    assert _reason(exc_info) == "ledger_closed"


def test_charge_over_budget_fails_closed_without_mutation() -> None:
    budget = PrivacyBudgetLedger({"tuning": {"epsilon": 2.0, "delta": 1e-5}})
    ledger = _scored_ledger(3)
    ledger.select(1)
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.charge(budget, "tuning")
    assert _reason(exc_info) == "budget_exceeded"
    assert budget.spends == ()
    assert "3.15" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Promotion evidence.
# ---------------------------------------------------------------------------


def _charged_ledger() -> TuningLedger:
    budget = PrivacyBudgetLedger({"tuning": {"epsilon": 10.0, "delta": 1e-5}})
    ledger = _scored_ledger(3)
    ledger.select(2)
    ledger.charge(budget, "tuning")
    return ledger


def test_promotion_evidence_exposes_the_composed_account() -> None:
    ledger = _charged_ledger()
    evidence = ledger.promotion_evidence()
    assert evidence.epsilon == pytest.approx(3.15)
    assert evidence.coverage == "all_trials"
    assert evidence.trial_count == 3
    assert evidence.selected_lineage_digest == _digest(2)
    assert evidence.ledger_digest == ledger.digest
    account = verify_promotion_evidence(evidence, ledger)
    assert account.epsilon == pytest.approx(3.15)
    roundtrip = TuningPromotionEvidence.from_mapping(evidence.to_dict())
    assert verify_promotion_evidence(roundtrip, ledger) == account


def test_evidence_citing_only_the_selected_trial_is_refused() -> None:
    ledger = _charged_ledger()
    selected = ledger.records[1]
    selected_only = {
        "selected_lineage_digest": selected.lineage_digest,
        "epsilon": selected.training_cost.epsilon,
        "delta": 0.0,
    }
    with pytest.raises(PrivateSelectionError) as exc_info:
        verify_promotion_evidence(selected_only, ledger)
    assert _reason(exc_info) == "evidence_selected_trial_only"

    payload = ledger.promotion_evidence().to_dict()
    payload.update(coverage="selected_trial", trial_count=1)
    with pytest.raises(PrivateSelectionError) as exc_info:
        verify_promotion_evidence(payload, ledger)
    assert _reason(exc_info) == "evidence_selected_trial_only"

    payload = ledger.promotion_evidence().to_dict()
    payload.update(trial_count=1)
    with pytest.raises(PrivateSelectionError) as exc_info:
        verify_promotion_evidence(payload, ledger)
    assert _reason(exc_info) == "evidence_selected_trial_only"


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"epsilon": 1.0}, "evidence_account_understated"),
        ({"delta": -1.0}, "evidence_invalid"),
        ({"trial_count": 2}, "evidence_coverage_incomplete"),
        ({"ledger_digest": "sha256:" + "f" * 64}, "evidence_ledger_mismatch"),
        ({"selected_lineage_digest": _digest(1)}, "evidence_ledger_mismatch"),
        ({"coverage": "sampled"}, "evidence_coverage_incomplete"),
    ],
)
def test_tampered_evidence_is_refused(update: dict, reason: str) -> None:
    ledger = _charged_ledger()
    payload = ledger.promotion_evidence().to_dict()
    payload.update(update)
    with pytest.raises(PrivateSelectionError) as exc_info:
        verify_promotion_evidence(payload, ledger)
    assert _reason(exc_info) == reason


def test_evidence_requires_charge_and_selection() -> None:
    ledger = _scored_ledger(2)
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.promotion_evidence()
    assert _reason(exc_info) == "not_charged"
    budget = PrivacyBudgetLedger({"tuning": {"epsilon": 10.0, "delta": 1e-5}})
    ledger.charge(budget, "tuning")
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.promotion_evidence()
    assert _reason(exc_info) == "no_selected_trial"


def test_random_stopping_evidence_omits_the_realized_trial_count() -> None:
    budget = PrivacyBudgetLedger({"tuning": {"epsilon": 10.0, "delta": 1e-5}})
    ledger = TuningLedger(
        _pure_accountant(0.1), metric_source="public", scheme=_geometric_scheme()
    )
    trials = ledger.draw_trial_count(_ScriptedCsprng([0.8]))
    for index in range(1, trials + 1):
        ledger.record_trial(_mechanism(), _digest(index), outcome_band="high")
    ledger.select(1)
    ledger.charge(budget, "tuning")
    evidence = ledger.promotion_evidence()
    assert evidence.trial_count is None
    assert "trial_count" not in evidence.to_dict()
    assert "trial_count" not in ledger.render()
    assert evidence.epsilon == pytest.approx(3.0)
    verify_promotion_evidence(evidence, ledger)


# ---------------------------------------------------------------------------
# Record hygiene: bands only, no seeds, no site identities.
# ---------------------------------------------------------------------------


def test_band_outcome_maps_metric_values_to_controlled_codes() -> None:
    assert band_outcome(0.1) == "low"
    assert band_outcome(0.3) == "moderate"
    assert band_outcome(0.6) == "high"
    assert band_outcome(0.95) == "very_high"
    assert set(OUTCOME_BANDS) >= {"not_evaluated", "low", "very_high"}


@pytest.mark.parametrize("band", [0.8731, "0.8731", "excellent", None, True])
def test_raw_metric_values_are_not_accepted_as_outcomes(band: object) -> None:
    ledger = TuningLedger(_pure_accountant(), metric_source="public")
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_trial(
            _mechanism(),
            _digest(1),
            outcome_band=band,  # type: ignore[arg-type]
        )
    assert _reason(exc_info) == "outcome_not_banded"
    assert "0.8731" not in str(exc_info.value)


@pytest.mark.parametrize(
    "extra", [{"seed": 918273645}, {"site_id": SITE_SENTINEL}, {"rng_state": "x"}]
)
def test_mechanism_mapping_refuses_seeds_and_site_identities(extra: dict) -> None:
    payload = {
        "noise_multiplier": 1.0,
        "clipping_norm": 1.0,
        "sampling_rate": 0.01,
        "round_count": 5,
        **extra,
    }
    with pytest.raises(PrivateSelectionError) as exc_info:
        TrialMechanism.from_mapping(payload)
    assert _reason(exc_info) == "invalid_field"
    assert SITE_SENTINEL not in str(exc_info.value)
    assert SEED_SENTINEL not in str(exc_info.value)


@pytest.mark.parametrize(
    "digest", [SITE_SENTINEL, "sha256:" + "z" * 64, "sha256:" + "a" * 63, 7]
)
def test_lineage_digest_must_be_a_sha256_reference(digest: object) -> None:
    ledger = TuningLedger(_pure_accountant(), metric_source="public")
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_trial(_mechanism(), digest)  # type: ignore[arg-type]
    assert _reason(exc_info) == "lineage_digest_invalid"
    assert SITE_SENTINEL not in str(exc_info.value)


def test_serialized_records_hold_only_bands_digests_and_mechanism_numbers() -> None:
    ledger = _charged_ledger()
    exported = json.dumps(
        {
            "records": [record.to_dict() for record in ledger.records],
            "render": ledger.render(),
            "evidence": ledger.promotion_evidence().to_dict(),
        },
        sort_keys=True,
    )
    assert SITE_SENTINEL not in exported
    assert SEED_SENTINEL not in exported
    for forbidden in ("seed", "site", "metric_value", "score", "rng"):
        assert forbidden not in exported
    for record in ledger.records:
        payload = record.to_dict()
        assert payload["outcome_band"] in OUTCOME_BANDS
        assert set(payload) == {
            "trial_index",
            "lineage_digest",
            "status",
            "outcome_band",
            "mechanism",
            "training_cost",
            "selection_evaluations",
        }
        assert set(payload["mechanism"]) <= {
            "noise_multiplier",
            "clipping_norm",
            "sampling_rate",
            "round_count",
            "learning_rate",
        }


def test_ledger_does_not_retain_the_injected_generator() -> None:
    rng = _ScriptedCsprng([0.3])
    ledger = TuningLedger(
        _pure_accountant(), metric_source="public", scheme=_geometric_scheme()
    )
    ledger.draw_trial_count(rng)
    assert all(value is not rng for value in vars(ledger).values())
    assert not any(character.isdigit() for character in repr(ledger))


def test_accountant_failures_do_not_leak_exception_text() -> None:
    def broken(_: TrialMechanism) -> TrialPrivacyCost:
        raise RuntimeError(SITE_SENTINEL)

    ledger = TuningLedger(broken, metric_source="public")
    with pytest.raises(PrivateSelectionError) as exc_info:
        ledger.record_trial(_mechanism(), _digest(1))
    assert _reason(exc_info) == "accountant_failed"
    assert SITE_SENTINEL not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def test_error_messages_are_fixed_codes() -> None:
    error = PrivateSelectionError("budget_exceeded", "context")
    assert str(error) == "private selection refused: budget_exceeded (context)"
    assert PrivateSelectionError("no_trials").reason == "no_trials"


# ---------------------------------------------------------------------------
# Cross-module flow.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tuning_program_end_to_end_with_shared_budget() -> None:
    budget = PrivacyBudgetLedger({"tuning": {"epsilon": 5.0, "delta": 1e-5}})
    ledger = TuningLedger(
        _approx_accountant,
        metric_source="private",
        scoring_cost=TrialPrivacyCost(epsilon=0.02, delta=1e-8),
    )
    ledger.record_trial(_mechanism(noise=1.0), _digest(1), outcome_band="moderate")
    ledger.record_selection(1, "early_stopping", evaluation_count=2)
    ledger.record_trial(
        _mechanism(noise=1.5),
        _digest(2),
        status="abandoned",
        outcome_band="not_evaluated",
    )
    ledger.record_selection(2, "early_stopping", evaluation_count=1)
    ledger.record_trial(_mechanism(noise=0.8), _digest(3), outcome_band="high")
    ledger.record_selection(3, "checkpoint_selection", evaluation_count=4)
    ledger.select(3)
    ledger.charge(budget, "tuning")
    evidence = ledger.promotion_evidence()
    # Training: 0.5 + 0.75 + 0.4; scoring: (2 + 1 + 4 + 2 scored) * 0.02.
    assert evidence.epsilon == pytest.approx(1.65 + 9 * 0.02)
    assert evidence.delta == pytest.approx(3e-6 + 9e-8)
    assert budget.render_counts_only()["spent_epsilon"] == pytest.approx(
        evidence.epsilon
    )
    verify_promotion_evidence(evidence, ledger)
    rendered = ledger.render()
    assert rendered["account"]["epsilon"] == pytest.approx(evidence.epsilon)
    assert dict(map(tuple, rendered["selection_counts"])) == {
        "best_of_k": 1,
        "checkpoint_selection": 4,
        "early_stopping": 3,
        "round_selection": 0,
    }
