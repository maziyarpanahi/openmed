"""Tests for offline SMART scope comparison helpers."""

from __future__ import annotations

import pytest

from openmed.interop.smart_scope_audit import (
    audit_smart_scopes,
    normalize_smart_scope,
    parse_smart_scope,
)


def test_normalizes_context_and_operation_order() -> None:
    assert normalize_smart_scope(" Patient/SyntheticObservation.SR ") == (
        "patient/SyntheticObservation.rs"
    )

    scope = parse_smart_scope("user/SyntheticCarePlan.uc")
    assert scope.to_dict() == {
        "scope": "user/SyntheticCarePlan.cu",
        "context": "user",
        "resource_type": "SyntheticCarePlan",
        "operations": [
            {"code": "c", "name": "create"},
            {"code": "u", "name": "update"},
        ],
    }


def test_reports_missing_and_excessive_atomic_operations() -> None:
    result = audit_smart_scopes(
        workflow_id="synthetic-workflow",
        required_scopes=(
            "patient/SyntheticCondition.r",
            "patient/SyntheticObservation.r",
        ),
        declared_scopes=(
            "patient/SyntheticObservation.rs",
            "patient/SyntheticMedication.r",
        ),
    )

    assert result.status == "missing"
    assert [scope.value for scope in result.missing_scopes] == [
        "patient/SyntheticCondition.r"
    ]
    assert [scope.value for scope in result.excessive_scopes] == [
        "patient/SyntheticMedication.r",
        "patient/SyntheticObservation.s",
    ]


@pytest.mark.parametrize(
    "scope",
    [
        "",
        "launch/patient",
        "patient/*.r",
        "patient/SyntheticObservation.*",
        "patient/SyntheticObservation.rx",
        "tenant/SyntheticObservation.r",
    ],
)
def test_rejects_out_of_scope_or_wildcard_values(scope: str) -> None:
    with pytest.raises(ValueError):
        parse_smart_scope(scope)
