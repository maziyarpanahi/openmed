"""Focused tests for the privacy-safe quasi-identifier profiler."""

from __future__ import annotations

import json

from openmed.risk import (
    QIProfiler,
    apply_generalization_plan,
    profile_quasi_identifiers,
)


def _synthetic_rows() -> list[dict[str, object]]:
    return [
        {
            "age": age,
            "city": "Riverton",
            "visit_date": "2025-01-01",
            "diagnosis": f"synthetic-{age}",
        }
        for age in range(30, 38)
    ]


def test_profiles_and_ranks_marginal_uniqueness_contribution() -> None:
    rows = [
        {
            "age": 30 + index // 2,
            "city": ("Riverton", "Lakeside", "Summit")[index // 2],
            "visit_date": f"2025-01-{index + 1:02d}",
            "diagnosis": "synthetic-condition",
        }
        for index in range(6)
    ]

    report = profile_quasi_identifiers(rows, target_k=2)

    assert report.columns[0].column == "visit_date"
    assert report.columns[0].marginal_uniqueness_contribution == 1.0
    assert report.columns[0].marginal_reidentification_risk > 0.0
    assert report.columns[0].offsets == tuple(range(6))
    assert report.quasi_identifiers == ("age", "city", "visit_date")
    assert next(item for item in report.columns if item.column == "diagnosis").role == (
        "sensitive"
    )


def test_generalization_plan_raises_achieved_k_and_is_reproducible() -> None:
    rows = _synthetic_rows()
    report = QIProfiler(target_k=2).profile(rows)

    applied = report.plan.apply(rows)
    rescored = report.plan.rescore(rows)

    assert report.before.achieved_k == 1
    assert report.after.achieved_k >= 2
    assert report.plan.after_achieved_k == report.after.achieved_k
    assert rescored.achieved_k == report.after.achieved_k
    assert applied[0]["age"] == "30-34"
    assert applied[-1]["age"] == "35-39"
    assert apply_generalization_plan(rows, report.plan) == applied


def test_suppression_offsets_are_applied_without_raw_values_in_report() -> None:
    rows = [
        {"age": 30, "city": "Riverton", "note": "synthetic-a"},
        {"age": 30, "city": "Riverton", "note": "synthetic-b"},
        {"age": 40, "city": "Lakeside", "note": "synthetic-c"},
        {"age": 40, "city": "Lakeside", "note": "synthetic-d"},
        {"age": 91, "city": "Summit", "note": "synthetic-e"},
    ]

    report = profile_quasi_identifiers(
        rows,
        quasi_identifiers=["age", "city"],
        target_k=2,
    )
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    retained = report.plan.apply(rows)

    assert report.before.achieved_k == 1
    assert report.after.achieved_k == 2
    assert report.plan.suppressed_offsets == (4,)
    assert len(retained) == 4
    assert "Riverton" not in serialized
    assert "Lakeside" not in serialized
    assert "Summit" not in serialized
    assert "synthetic-e" not in serialized
    assert report.to_dict()["plan"]["columns"][0]["affected_offsets"] == []


def test_empty_candidate_set_is_reported_without_exposing_source_values() -> None:
    rows = [
        {"status": "active", "note": "synthetic note"},
        {"status": "active", "note": "synthetic note"},
    ]

    report = profile_quasi_identifiers(rows, target_k=2)

    assert report.quasi_identifiers == ()
    assert report.plan.suppression_count == 0
    assert report.after.achieved_k == 2
    assert "synthetic note" not in report.to_json()
