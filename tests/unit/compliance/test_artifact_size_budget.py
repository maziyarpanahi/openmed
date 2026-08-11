"""Focused tests for counts-only audit-artifact size budgets."""

from __future__ import annotations

import json

import pytest

from openmed.compliance import (
    ArtifactDescriptor,
    ArtifactSectionDescriptor,
    ArtifactSizeBudget,
    check_artifact_size_budget,
    evaluate_artifact_size_budget,
)


def _descriptor() -> ArtifactDescriptor:
    return ArtifactDescriptor(
        total_bytes=100,
        sections=(
            ArtifactSectionDescriptor(
                size_bytes=40,
                record_count=3,
                nesting_depth=2,
            ),
            ArtifactSectionDescriptor(
                size_bytes=60,
                record_count=4,
                nesting_depth=4,
            ),
        ),
    )


def test_descriptor_within_all_limits_returns_count_only_report() -> None:
    budget = ArtifactSizeBudget(
        max_total_bytes=100,
        max_section_bytes=60,
        max_record_count=7,
        max_nesting_depth=4,
    )

    result = evaluate_artifact_size_budget(_descriptor(), budget)

    assert result.within_budget is True
    assert result.exceeded_categories == ()
    assert result.to_dict() == {
        "within_budget": True,
        "exceeded_categories": [],
        "observed": {
            "total_bytes": 100,
            "section_bytes": 60,
            "record_count": 7,
            "nesting_depth": 4,
        },
        "limits": {
            "total_bytes": 100,
            "section_bytes": 60,
            "record_count": 7,
            "nesting_depth": 4,
        },
        "violations": [],
    }


def test_all_exceeded_categories_are_deterministic_and_count_only() -> None:
    budget = ArtifactSizeBudget(
        max_total_bytes=99,
        max_section_bytes=59,
        max_record_count=6,
        max_nesting_depth=3,
    )

    first = evaluate_artifact_size_budget(_descriptor(), budget)
    second = evaluate_artifact_size_budget(_descriptor(), budget)

    assert first.to_dict() == second.to_dict()
    assert first.exceeded_budget_categories == (
        "total_bytes",
        "section_bytes",
        "record_count",
        "nesting_depth",
    )
    assert first.to_dict()["violations"] == [
        {"category": "total_bytes", "observed": 100, "limit": 99},
        {"category": "section_bytes", "observed": 60, "limit": 59},
        {"category": "record_count", "observed": 7, "limit": 6},
        {"category": "nesting_depth", "observed": 4, "limit": 3},
    ]
    assert "sections" not in first.to_dict()
    assert json.loads(json.dumps(first.to_dict(), sort_keys=True)) == first.to_dict()


def test_mapping_input_uses_counts_and_does_not_echo_free_form_fields() -> None:
    descriptor = {
        "total_size_bytes": 24,
        "record_count_total": 5,
        "max_nesting_depth": 2,
        "sections": {
            "synthetic-summary": {
                "bytes": 24,
                "records": 5,
                "depth": 2,
                "content": "synthetic-only-value",
            }
        },
        "content": "synthetic-only-artifact-value",
    }
    budget = {
        "total_bytes": 24,
        "max_bytes_per_section": 24,
        "max_records": 5,
        "nesting_depth": 2,
    }

    result = check_artifact_size_budget(descriptor, budget)
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert result.within_budget is True
    assert "synthetic-summary" not in serialized
    assert "synthetic-only-value" not in serialized
    assert "synthetic-only-artifact-value" not in serialized


def test_explicit_aggregates_cannot_understate_section_measurements() -> None:
    descriptor = ArtifactDescriptor(
        total_bytes=1,
        record_count=1,
        max_nesting_depth=1,
        sections=(
            ArtifactSectionDescriptor(
                size_bytes=8,
                record_count=3,
                nesting_depth=4,
            ),
        ),
    )
    budget = ArtifactSizeBudget(
        max_total_bytes=7,
        max_record_count=2,
        max_nesting_depth=3,
    )

    result = evaluate_artifact_size_budget(descriptor, budget)

    assert result.observed == {
        "total_bytes": 8,
        "section_bytes": 8,
        "record_count": 3,
        "nesting_depth": 4,
    }
    assert result.exceeded_categories == (
        "total_bytes",
        "record_count",
        "nesting_depth",
    )


@pytest.mark.parametrize(
    ("factory", "value"),
    [
        (lambda value: ArtifactSizeBudget(max_total_bytes=value), True),
        (lambda value: ArtifactSectionDescriptor(size_bytes=value), -1),
        (lambda value: ArtifactDescriptor(total_bytes=value), "not-a-count"),
    ],
)
def test_count_fields_reject_non_safe_values(factory, value) -> None:
    with pytest.raises((TypeError, ValueError), match="non-negative integer"):
        factory(value)


def test_unknown_or_unconfigured_dimensions_are_not_reported() -> None:
    descriptor = ArtifactDescriptor(total_bytes=8)
    result = evaluate_artifact_size_budget(
        descriptor,
        ArtifactSizeBudget(max_total_bytes=10),
    )

    assert result.within_budget is True
    assert result.limits == {
        "total_bytes": 10,
        "section_bytes": None,
        "record_count": None,
        "nesting_depth": None,
    }
