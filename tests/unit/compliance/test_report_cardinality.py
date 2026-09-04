"""Focused tests for privacy report cardinality budgets."""

from __future__ import annotations

import json

import pytest

from openmed.compliance import (
    ReportCardinalityBudget,
    check_report_cardinality,
)

SENSITIVE_VALUE = "synthetic-sensitive-value"


def test_typed_report_within_budget_is_deterministic() -> None:
    report = {
        "sections": [
            {"measurements": [1, 2]},
            {"measurements": [3]},
        ],
        "summary": {"count": 3},
    }
    budget = ReportCardinalityBudget(
        max_items_per_field=4,
        max_unique_keys=4,
        max_nesting_depth=3,
        max_aggregate_items=20,
    )

    first = check_report_cardinality(report, budget)
    second = check_report_cardinality(report, budget)

    assert first.allowed is True
    assert first.within_budget is True
    assert first.failed_closed is False
    assert first.violations == ()
    assert first.to_json() == second.to_json()


def test_item_and_unique_key_limits_report_paths_and_counts_only() -> None:
    report = {
        "events": [SENSITIVE_VALUE, SENSITIVE_VALUE, SENSITIVE_VALUE],
        "metadata": {
            "opaque-key-1": 1,
            "opaque-key-2": 2,
            "opaque-key-3": 3,
        },
    }
    budget = ReportCardinalityBudget(
        max_items_per_field=2,
        max_unique_keys=2,
        max_nesting_depth=3,
        max_aggregate_items=20,
    )

    result = check_report_cardinality(report, budget)
    serialized = result.to_json()

    assert result.allowed is False
    assert {
        (violation.path, violation.rule, violation.count, violation.limit)
        for violation in result.violations
    } == {
        ("$.events", "items_per_field", 3, 2),
        ("$.metadata", "items_per_field", 3, 2),
        ("$.metadata", "unique_keys", 3, 2),
    }
    assert SENSITIVE_VALUE not in serialized
    assert "opaque-key-1" not in serialized
    assert all(
        set(violation.to_dict()) == {"count", "limit", "path", "rule"}
        for violation in result.violations
    )


def test_nesting_and_aggregate_limits_fail_closed() -> None:
    report = {
        "groups": [
            {"values": [1, 2]},
            {"values": [3, 4]},
        ]
    }

    nesting_result = check_report_cardinality(
        report,
        ReportCardinalityBudget(
            max_items_per_field=10,
            max_unique_keys=10,
            max_nesting_depth=1,
            max_aggregate_items=100,
        ),
    )
    aggregate_result = check_report_cardinality(
        report,
        ReportCardinalityBudget(
            max_items_per_field=10,
            max_unique_keys=10,
            max_nesting_depth=5,
            max_aggregate_items=4,
        ),
    )

    assert any(
        (violation.path, violation.count, violation.limit) == ("$.groups[*]", 2, 1)
        and violation.rule == "nesting_depth"
        for violation in nesting_result.violations
    )
    assert any(
        violation.rule == "aggregate_items"
        and violation.path == "$.groups[*].values"
        and violation.count == 6
        and violation.limit == 4
        for violation in aggregate_result.violations
    )


def test_cycles_unknown_shapes_and_non_string_keys_fail_closed() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    unknown = {"payload": object()}
    non_string_key = {1: SENSITIVE_VALUE}

    cyclic_result = check_report_cardinality(cyclic)
    unknown_result = check_report_cardinality(unknown)
    key_result = check_report_cardinality(non_string_key)  # type: ignore[arg-type]

    assert any(violation.rule == "cycle" for violation in cyclic_result.violations)
    assert any(
        violation.rule == "unsupported_shape" for violation in unknown_result.violations
    )
    assert any(
        violation.rule == "non_string_key" for violation in key_result.violations
    )
    assert SENSITIVE_VALUE not in key_result.to_json()
    assert "object at" not in unknown_result.to_json()


def test_invalid_budget_fails_closed_without_echoing_input() -> None:
    result = check_report_cardinality(
        {"field": SENSITIVE_VALUE},
        budget=object(),  # type: ignore[arg-type]
    )

    assert result.allowed is False
    assert result.violations[0].rule == "invalid_budget"
    assert SENSITIVE_VALUE not in json.dumps(result.to_dict())


def test_budget_rejects_boolean_and_negative_limits() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        ReportCardinalityBudget(max_items_per_field=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative integer"):
        ReportCardinalityBudget(max_unique_keys=-1)
