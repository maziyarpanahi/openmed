"""Synthetic offline tests for deterministic clinical-summary budgets."""

from __future__ import annotations

import hashlib
import json

import pytest

from openmed.clinical.summary_length_budget import (
    SUMMARY_LENGTH_BUDGET_DISCLAIMER,
    SummaryEvidenceClassPolicy,
    SummaryLengthBudgetError,
    SummaryLengthBudgetPolicy,
    build_summary_length_budget,
)


def _opaque(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def test_budget_is_stable_for_equivalent_demand_orderings() -> None:
    first = build_summary_length_budget(
        64,
        {
            "safety": 12,
            "active_problems": 80,
            "medications": 20,
            "follow_up": 20,
        },
    )
    second = build_summary_length_budget(
        total_tokens=64,
        evidence=(
            {"evidence_class": "follow_up", "token_count": 20},
            {"evidence_class": "medications", "token_count": 20},
            {"evidence_class": "active_problems", "token_count": 80},
            {"evidence_class": "safety", "token_count": 12},
        ),
    )

    assert first.to_json() == second.to_json()
    assert first.allocated_tokens == 64
    assert first.deferred_evidence_classes == (
        "active_problems",
        "follow_up",
        "medications",
    )
    assert first.budget_for("safety") == 12


def test_budget_respects_minimums_weights_and_global_cap() -> None:
    policy = SummaryLengthBudgetPolicy(
        policy_id="synthetic_summary",
        classes=(
            SummaryEvidenceClassPolicy(
                "safety", priority=0, weight=4, minimum_tokens=2
            ),
            SummaryEvidenceClassPolicy(
                "problems", priority=1, weight=1, minimum_tokens=2
            ),
            SummaryEvidenceClassPolicy(
                "follow_up", priority=2, weight=1, minimum_tokens=2
            ),
        ),
    )

    budget = build_summary_length_budget(
        7,
        {"safety": 10, "problems": 10, "follow_up": 10},
        policy=policy,
    )

    assert budget.allocated_tokens == 7
    assert budget.unused_tokens == 0
    assert budget.budget_for("safety") == 3
    assert budget.budget_for("problems") == 2
    assert budget.budget_for("follow_up") == 2
    assert budget.deferred_evidence_classes == (
        "follow_up",
        "safety",
        _opaque("problems"),
    )


def test_severely_constrained_budget_keeps_priority_order_deterministically() -> None:
    policy = SummaryLengthBudgetPolicy(
        policy_id="synthetic_priority",
        classes=(
            SummaryEvidenceClassPolicy("safety", priority=0, minimum_tokens=4),
            SummaryEvidenceClassPolicy("problems", priority=1, minimum_tokens=4),
            SummaryEvidenceClassPolicy("follow_up", priority=2, minimum_tokens=4),
        ),
    )

    budget = build_summary_length_budget(
        5,
        {"follow_up": 10, "problems": 10, "safety": 10},
        policy=policy,
    )

    assert budget.budget_for("safety") == 4
    assert budget.budget_for("problems") == 1
    assert budget.budget_for("follow_up") == 0
    assert budget.deferred_evidence_classes == (
        "follow_up",
        "safety",
        _opaque("problems"),
    )
    assert budget.allocation_for("follow_up").status == "deferred"


def test_report_identifies_partial_and_full_defer_without_source_values() -> None:
    budget = build_summary_length_budget(
        24,
        {
            "active_problems": 40,
            "medications": 40,
            "follow_up": 40,
        },
    )

    payload = budget.to_dict()
    assert payload["truncation"] == {
        "occurred": True,
        "deferred_evidence_classes": [
            "active_problems",
            "follow_up",
            "medications",
        ],
        "deferred_tokens": 96,
    }
    assert budget.truncation == budget.truncation_metadata
    assert budget.truncation.occurred is True
    statuses = {
        item["evidence_class"]: item["status"]
        for item in payload["allocations"]
        if item["requested_tokens"]
    }
    assert statuses == {
        "active_problems": "truncated",
        "follow_up": "deferred",
        "medications": "truncated",
    }
    assert payload["requires_clinician_review"] is True
    assert payload["autonomous_decision"] is False
    assert payload["disclaimer"] == SUMMARY_LENGTH_BUDGET_DISCLAIMER
    assert "synthetic_private_marker" not in json.dumps(payload)
    assert "synthetic_record_token" not in budget.to_json()


def test_mapping_policy_is_closed_and_serialization_is_byte_stable() -> None:
    budget = build_summary_length_budget(
        total_tokens=12,
        evidence={"safety": {"available_tokens": 4}},
        policy={
            "policy_id": "synthetic_mapping",
            "classes": [
                {"name": "safety", "priority": 0, "weight": 2},
                {"name": "follow_up", "priority": 1, "weight": 1},
            ],
        },
    )

    expected = budget.to_json()
    assert expected == budget.to_json()
    assert json.loads(expected)["policy_id"] == _opaque("synthetic_mapping")
    assert budget.budget_for("follow_up") == 0


def test_unknown_or_sensitive_inputs_fail_with_fixed_value_free_errors() -> None:
    with pytest.raises(SummaryLengthBudgetError) as unknown:
        build_summary_length_budget(32, {"patient_name": 8})
    assert unknown.value.reason_code == "unknown_evidence_class"
    assert "patient_name" not in str(unknown.value)

    with pytest.raises(SummaryLengthBudgetError) as invalid:
        build_summary_length_budget(
            32,
            [
                {
                    "evidence_class": "safety",
                    "token_count": 8,
                    "text": "synthetic_private_marker",
                }
            ],
        )
    assert invalid.value.reason_code == "invalid_evidence"
    assert "synthetic_private_marker" not in str(invalid.value)


def test_no_model_or_network_is_needed_for_budget_planning(monkeypatch) -> None:
    def fail_if_network_is_attempted(*args, **kwargs):
        raise AssertionError("network access is outside the budget planner")

    monkeypatch.setattr("socket.socket", fail_if_network_is_attempted)
    budget = build_summary_length_budget(
        32,
        evidence_classes=("safety", "active_problems"),
    )

    assert budget.allocated_tokens == 32
    assert budget.truncated is True
    assert budget.deferred_evidence_classes == ("active_problems", "safety")


def test_typed_demand_cannot_bypass_approval() -> None:
    from openmed.clinical.summary_length_budget import SummaryEvidenceDemand

    demand = SummaryEvidenceDemand("safety", 8)
    object.__setattr__(demand, "approved", False)
    with pytest.raises(SummaryLengthBudgetError):
        build_summary_length_budget(32, [demand])


def test_typed_policy_is_revalidated() -> None:
    item = SummaryEvidenceClassPolicy("safety", 0)
    policy = SummaryLengthBudgetPolicy(classes=(item,))
    object.__setattr__(policy.classes[0], "weight", 0)
    with pytest.raises(SummaryLengthBudgetError):
        build_summary_length_budget(32, {"safety": 8}, policy=policy)


def test_deferred_metadata_requires_positive_tokens() -> None:
    from openmed.clinical.summary_length_budget import SummaryTruncationMetadata

    with pytest.raises(SummaryLengthBudgetError):
        SummaryTruncationMetadata(("safety",), 0)


def test_iterator_failure_has_no_sensitive_exception_context() -> None:
    def broken():
        raise ValueError("SYNTHETIC_PRIVATE_CANARY")
        yield

    with pytest.raises(SummaryLengthBudgetError) as caught:
        build_summary_length_budget(32, broken())
    assert caught.value.__context__ is None


def test_cycles_fail_closed() -> None:
    cycle = {}
    cycle["evidence"] = cycle
    with pytest.raises(SummaryLengthBudgetError):
        build_summary_length_budget(32, cycle)


def test_custom_policy_identifiers_are_opaque() -> None:
    policy = SummaryLengthBudgetPolicy(
        policy_id="patient_jane_doe",
        classes=(SummaryEvidenceClassPolicy("patient_jane_doe", 0),),
    )
    budget = build_summary_length_budget(8, {"patient_jane_doe": 12}, policy=policy)
    assert "jane_doe" not in budget.to_json() + repr(budget) + repr(policy)


def test_direct_policy_rejects_unbounded_classes() -> None:
    from itertools import repeat

    with pytest.raises(SummaryLengthBudgetError):
        SummaryLengthBudgetPolicy(
            classes=repeat(SummaryEvidenceClassPolicy("safety", 0))
        )


def test_mapping_failure_is_sanitized() -> None:
    class Broken(dict):
        def items(self):
            raise ValueError("SYNTHETIC_PRIVATE_CANARY")

    with pytest.raises(SummaryLengthBudgetError) as caught:
        build_summary_length_budget(32, {"safety": 4}, policy=Broken())
    assert caught.value.__context__ is None
