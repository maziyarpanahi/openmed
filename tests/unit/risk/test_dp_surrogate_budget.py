"""Tests for DP surrogate budget accounting."""

from __future__ import annotations

import json

import pytest

from openmed.risk import (
    DPSurrogateBudget,
    DPSurrogateBudgetExceeded,
    DPSurrogateSensitivity,
    DPSurrogateSensitivityRegistry,
)


def test_dp_surrogate_budget_reports_conservative_composition():
    budget = DPSurrogateBudget(target_epsilon=4.0, target_delta=1e-6)

    budget.spend(
        label="name",
        draw_kind="categorical",
        mechanism="exponential",
        epsilon=0.05,
    )
    budget.spend(
        label="date",
        draw_kind="date_offset",
        mechanism="discrete_laplace",
        epsilon=0.05,
    )

    composition = budget.composition()

    assert composition.query_count == 2
    assert composition.reported_epsilon >= composition.basic_epsilon
    assert composition.reported_epsilon >= composition.zcdp_epsilon
    assert composition.reported_epsilon >= composition.rdp_epsilon
    assert composition.remaining_epsilon == pytest.approx(
        budget.target_epsilon - composition.reported_epsilon
    )
    assert composition.basic_delta == 0.0


def test_dp_surrogate_budget_exhaustion_raises_before_recording():
    budget = DPSurrogateBudget(target_epsilon=0.5, target_delta=1e-6)

    budget.spend(
        label="name",
        draw_kind="categorical",
        mechanism="exponential",
        epsilon=0.01,
    )
    before = budget.spends

    with pytest.raises(DPSurrogateBudgetExceeded) as excinfo:
        budget.spend(
            label="date",
            draw_kind="date_offset",
            mechanism="discrete_gaussian",
            epsilon=0.2,
        )

    assert budget.spends == before
    assert excinfo.value.attempted_spend.label == "DATE"
    assert excinfo.value.attempted_composition.reported_epsilon > 0.5


def test_dp_surrogate_sensitivity_registry_validates_and_exports_entries():
    registry = DPSurrogateSensitivityRegistry.defaults().with_entry(
        "medical-record-number",
        "categorical",
        l1=2.0,
        l2=3.0,
    )

    sensitivity = registry.for_label("medical record number", "categorical")

    assert sensitivity == DPSurrogateSensitivity(
        label="MEDICAL_RECORD_NUMBER",
        draw_kind="categorical",
        l1=2.0,
        l2=3.0,
    )
    assert registry.for_label("unknown", "numeric").to_dict() == {
        "label": "UNKNOWN",
        "draw_kind": "numeric",
        "l1": 1.0,
        "l2": 1.0,
    }
    with pytest.raises(ValueError, match="l1 must be positive"):
        registry.with_entry("name", "categorical", l1=0.0)
    with pytest.raises(ValueError, match="draw_kind"):
        registry.for_label("name", "free_text")


def test_dp_surrogate_accounting_payload_is_deterministic_and_hash_only():
    budget = DPSurrogateBudget(target_epsilon=4.0, target_delta=1e-6)
    spend = budget.spend(
        label="person",
        draw_kind="categorical",
        mechanism="exponential",
        epsilon=0.05,
    )

    first = budget.to_dict(salt="unit-test")
    second = budget.to_dict(salt="unit-test")
    encoded = json.dumps(first, sort_keys=True)

    assert first == second
    assert json.loads(encoded) == first
    assert first["spends"][0]["spend_hash"] == spend.spend_hash(salt="unit-test")
    assert "Alice" not in encoded
    assert "surrogate" not in first["spends"][0]


def test_exported_dp_surrogate_budget_api_is_available_from_package():
    import openmed.risk as risk

    assert hasattr(risk, "DPSurrogateBudget")
    assert "DPSurrogateSensitivityRegistry" in risk.__all__
