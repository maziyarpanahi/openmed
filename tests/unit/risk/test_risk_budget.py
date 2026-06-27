"""Tests for per-document risk-budget enforcement."""

from __future__ import annotations

import json

import pytest

from openmed.core.policy import list_policies
from openmed.risk import (
    DEFAULT_POLICY_BUDGETS,
    RiskBudget,
    RiskBudgetExceeded,
    budget_for_policy,
    evaluate_budget,
    risk_report,
)


def test_under_budget_risk_report_returns_clean_verdict():
    report = risk_report(
        [
            {"record_id": "a", "age": 44},
            {"record_id": "b", "age": 44},
        ]
    )
    budget = RiskBudget(
        name="unit",
        max_residual_qi_weight=3.0,
        max_surviving_direct_ids=0,
        min_k=2,
        max_singleton_records=0,
    )

    verdict = evaluate_budget(report, budget)

    assert verdict.within_budget is True
    assert verdict.violations == ()
    payload = verdict.to_dict()
    assert payload["within_budget"] is True
    assert payload["breakdown"]["k_min"] == {
        "consumed": 2,
        "limit": 2,
        "comparison": "min",
    }


def test_surviving_direct_identifier_violation_lists_consumed_and_limit_values():
    report = {
        **risk_report("Patient [NAME] lives in [LOCATION]."),
        "surviving_direct_ids": 2,
    }
    budget = RiskBudget(name="unit", max_surviving_direct_ids=0)

    verdict = evaluate_budget(report, budget)

    assert verdict.within_budget is False
    assert [violation.metric for violation in verdict.violations] == [
        "surviving_direct_ids"
    ]
    violation = verdict.to_dict()["violations"][0]
    assert violation["consumed"] == 2
    assert violation["limit"] == 0


def test_strict_mode_raises_only_when_over_budget():
    report = {**risk_report("Patient [NAME]."), "surviving_direct_ids": 1}
    budget = RiskBudget(name="unit", max_surviving_direct_ids=0)

    with pytest.raises(RiskBudgetExceeded) as excinfo:
        evaluate_budget(report, budget, strict=True)

    assert excinfo.value.verdict.to_dict()["violations"][0]["metric"] == (
        "surviving_direct_ids"
    )

    clean = {**report, "surviving_direct_ids": 0}
    assert evaluate_budget(clean, budget, strict=True).within_budget is True


def test_verdict_is_json_serializable_and_deterministic_for_fixed_input():
    report = {
        **risk_report(
            {
                "doc_id": "note-1",
                "text": "94-year-old patient from Riverton.",
            }
        ),
        "surviving_direct_ids": [],
    }
    budget = RiskBudget(
        name="unit",
        max_residual_qi_weight=3.0,
        max_surviving_direct_ids=0,
        max_singleton_records=1,
    )

    first = evaluate_budget(report, budget).to_dict()
    second = evaluate_budget(report, budget).to_dict()

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first


def test_default_budgets_cover_every_bundled_policy_and_strict_is_stricter():
    assert set(DEFAULT_POLICY_BUDGETS) == set(list_policies())

    strict = budget_for_policy("strict_no_leak")
    clinical = budget_for_policy("clinical_minimal_redaction")

    assert strict.max_residual_qi_weight == 0.0
    assert clinical.max_residual_qi_weight is not None
    assert strict.max_residual_qi_weight < clinical.max_residual_qi_weight
    assert budget_for_policy("gdpr").name == "gdpr_pseudonymization"


def test_exported_budget_api_is_available_from_package():
    import openmed.risk as risk

    assert hasattr(risk, "evaluate_budget")
    assert "RiskBudgetExceeded" in risk.__all__
