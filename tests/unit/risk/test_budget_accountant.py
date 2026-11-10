"""Tests for the DP generation budget accountant and epsilon-policy gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openmed.risk import (
    BudgetExceeded,
    DPGenerationBudgetAccountant,
    EpsilonPolicy,
    GenerationSpend,
    epsilon_policy_for,
    load_epsilon_policies,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "risk" / "dp_budget_scenarios.json"
)


def _load_scenarios() -> list[dict]:
    payload = json.loads(SCENARIO_FIXTURE.read_text(encoding="utf-8"))
    assert payload["kind"] == "meta"
    assert payload["synthetic"] is True
    return list(payload["scenarios"])


def _basic_epsilon(epsilons: list[float]) -> float:
    return math.fsum(epsilons)


def _advanced_epsilon(epsilons: list[float], delta_prime: float) -> float:
    sum_squares = math.fsum(epsilon * epsilon for epsilon in epsilons)
    tail = math.fsum(epsilon * math.expm1(epsilon) for epsilon in epsilons)
    return math.sqrt(2.0 * math.log(1.0 / delta_prime) * sum_squares) + tail


def _record_prior(accountant: DPGenerationBudgetAccountant, scenario: dict) -> None:
    for spend in scenario["prior_spends"]:
        accountant.guard_generation(
            spend["epsilon"],
            spend["delta"],
            scenario["scope"],
            family=scenario["family"],
            mechanism=spend["mechanism"],
        )


def test_committed_config_loads_and_validates_policies():
    policies = load_epsilon_policies()

    assert set(policies) >= {
        "clinical_release_default",
        "research_broad",
        "strict_minimal",
    }
    basic = epsilon_policy_for("research_broad")
    assert basic.composition == "basic"
    assert basic.max_epsilon == 8.0
    assert basic.delta_prime == 0.0

    advanced = epsilon_policy_for("clinical_release_default")
    assert advanced.composition == "advanced"
    assert 0.0 < advanced.delta_prime < advanced.max_delta


@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["name"])
def test_check_budget_matches_fixture_expectations(scenario: dict):
    accountant = DPGenerationBudgetAccountant.from_config()
    _record_prior(accountant, scenario)

    request = scenario["request"]
    decision = accountant.check_budget(
        request["epsilon"],
        request["delta"],
        scenario["scope"],
        family=scenario["family"],
        mechanism=request["mechanism"],
    )

    assert decision.allowed is scenario["expected_allowed"]
    if "expected_projected_epsilon" in scenario:
        assert decision.projected_epsilon == pytest.approx(
            scenario["expected_projected_epsilon"]
        )
    # check_budget must never mutate the ledger.
    assert len(accountant.ledger) == len(scenario["prior_spends"])


def test_budget_exceeded_blocks_generation_and_reports_remaining():
    accountant = DPGenerationBudgetAccountant.from_config()
    over = next(s for s in _load_scenarios() if s["name"] == "basic_over_budget")
    _record_prior(accountant, over)
    ledger_before = accountant.ledger

    request = over["request"]
    with pytest.raises(BudgetExceeded) as excinfo:
        accountant.guard_generation(
            request["epsilon"],
            request["delta"],
            over["scope"],
            family=over["family"],
            mechanism=request["mechanism"],
        )

    decision = excinfo.value.decision
    assert decision.allowed is False
    assert decision.reason == "epsilon exceeds policy"
    # Ledger already sits at the ceiling, so remaining headroom is zero.
    assert decision.remaining_epsilon == pytest.approx(0.0)
    assert decision.projected_epsilon == pytest.approx(8.5)
    # A blocked request must leave the ledger untouched.
    assert accountant.ledger == ledger_before


def test_repeated_basic_composition_equals_rule_result():
    accountant = DPGenerationBudgetAccountant.from_config()
    epsilons = [0.5, 0.75, 1.25, 2.0]
    for epsilon in epsilons:
        accountant.guard_generation(epsilon, 1e-6, "research_broad", family="labs")

    composition = accountant.compose("research_broad")

    assert composition.query_count == len(epsilons)
    assert composition.epsilon == pytest.approx(_basic_epsilon(epsilons))
    assert composition.remaining_epsilon == pytest.approx(
        8.0 - _basic_epsilon(epsilons)
    )


def test_repeated_advanced_composition_equals_rule_result():
    accountant = DPGenerationBudgetAccountant.from_config()
    policy = epsilon_policy_for("clinical_release_default")
    epsilons = [0.1] * 10
    for epsilon in epsilons:
        accountant.guard_generation(
            epsilon, 1e-8, "clinical_release_default", family="vitals"
        )

    composition = accountant.compose("clinical_release_default")

    assert composition.composition == "advanced"
    assert composition.epsilon == pytest.approx(
        _advanced_epsilon(epsilons, policy.delta_prime)
    )


def test_advanced_gate_bites_once_cumulative_spend_crosses_policy():
    accountant = DPGenerationBudgetAccountant.from_config()
    # 18 draws of 0.1 compose to ~2.92 < 3.0 under advanced composition; the
    # 19th crosses the ceiling and must be refused.
    for _ in range(18):
        decision = accountant.guard_generation(
            0.1, 1e-8, "clinical_release_default", family="vitals"
        )
        assert decision.allowed is True

    with pytest.raises(BudgetExceeded):
        accountant.guard_generation(
            0.1, 1e-8, "clinical_release_default", family="vitals"
        )
    assert len(accountant.ledger) == 18


def test_ledger_contains_only_numeric_aggregates_and_identifiers():
    accountant = DPGenerationBudgetAccountant.from_config()
    accountant.guard_generation(1.0, 1e-6, "research_broad", family="patient_labs")
    accountant.guard_generation(2.0, 1e-6, "research_broad", family="patient_labs")

    payload = accountant.to_dict(salt="unit-test")
    encoded = json.dumps(payload, sort_keys=True)

    allowed_keys = {
        "sequence",
        "scope",
        "family",
        "epsilon",
        "delta",
        "mechanism",
        "spend_hash",
    }
    for entry in payload["ledger"]:
        assert set(entry) == allowed_keys
        for key, value in entry.items():
            # Numeric aggregates plus scope/family/mechanism identifiers only;
            # never nested raw rows or record structures.
            assert isinstance(value, (int, float, str))
            assert not isinstance(value, (list, dict))
    # A PHI-shaped value can never have entered a numeric-only ledger.
    assert "123-45-6789" not in encoded
    assert json.loads(encoded) == payload


def test_generation_spend_rejects_non_scalar_payloads():
    with pytest.raises(ValueError):
        GenerationSpend(
            sequence=1,
            scope="research_broad",
            family=["row-0", "row-1"],  # type: ignore[arg-type]
            epsilon=1.0,
            delta=1e-6,
            mechanism="gaussian",
        )


def test_epsilon_policy_validation_rejects_bad_configuration():
    with pytest.raises(ValueError, match="composition"):
        EpsilonPolicy(scope="x", max_epsilon=1.0, max_delta=1e-6, composition="rdp")
    with pytest.raises(ValueError, match="delta_prime"):
        EpsilonPolicy(
            scope="x",
            max_epsilon=1.0,
            max_delta=1e-6,
            composition="advanced",
        )
    with pytest.raises(ValueError, match="delta_prime must be smaller"):
        EpsilonPolicy(
            scope="x",
            max_epsilon=1.0,
            max_delta=1e-6,
            composition="advanced",
            delta_prime=1e-3,
        )


def test_exported_budget_accountant_api_is_available_from_package():
    import openmed.risk as risk

    assert hasattr(risk, "DPGenerationBudgetAccountant")
    assert "BudgetExceeded" in risk.__all__
    assert "EpsilonPolicy" in risk.__all__
