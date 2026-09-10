"""Tests for the offline l-diversity/t-closeness checker (issue #943)."""

from __future__ import annotations

import json
import math

import pytest

from openmed.risk import (
    LDiversityChecker,
    LDiversityEngine,
    analyze_l_diversity,
    check_l_diversity,
)


def _synthetic_rows() -> list[dict[str, object]]:
    return [
        {"group": "A", "diagnosis": "flu"},
        {"group": "A", "diagnosis": "flu"},
        {"group": "A", "diagnosis": "flu"},
        {"group": "A", "diagnosis": "cold"},
        {"group": "B", "diagnosis": "flu"},
        {"group": "B", "diagnosis": "cold"},
        {"group": "B", "diagnosis": "asthma"},
        {"group": "B", "diagnosis": "asthma"},
    ]


def test_reference_l_diversity_and_variational_t_closeness_values() -> None:
    report = analyze_l_diversity(
        _synthetic_rows(),
        quasi_identifiers=["group"],
        sensitive_attribute="diagnosis",
        target_l=2,
        target_t=0.25,
    )

    first, second = report.equivalence_classes
    assert [item.size for item in report.equivalence_classes] == [4, 4]
    assert [item.distinct for item in report.equivalence_classes] == [2, 3]
    assert first.entropy == pytest.approx(0.8112781244591328)
    assert second.entropy == pytest.approx(1.5)
    assert first.t_closeness == pytest.approx(0.25)
    assert second.t_closeness == pytest.approx(0.25)
    assert report.achieved_l == 2
    assert report.achieved_t == pytest.approx(0.25)
    assert report.meets_target is True


def test_homogeneous_classes_are_flagged_as_l_one_violations() -> None:
    rows = [
        {"group": "A", "diagnosis": "flu"},
        {"group": "A", "diagnosis": "flu"},
        {"group": "B", "diagnosis": "cold"},
        {"group": "B", "diagnosis": "asthma"},
    ]

    report = check_l_diversity(
        rows,
        ["group"],
        "diagnosis",
        target_l=2,
        target_t=1.0,
    )

    assert report.achieved_distinct == 1
    assert report.l_violating_class_count == 1
    assert report.violating_classes[0].distinct == 1
    assert report.violating_classes[0].row_indices == (0, 1)
    assert report.meets_l is False
    assert report.meets_t is True
    assert report.meets_target is False


def test_entropy_threshold_and_t_violations_are_reported_separately() -> None:
    rows = [
        {"group": "A", "diagnosis": "flu"},
        {"group": "A", "diagnosis": "cold"},
        {"group": "B", "diagnosis": "flu"},
        {"group": "B", "diagnosis": "flu"},
    ]

    report = LDiversityEngine(
        ["group"],
        "diagnosis",
        target_l=2,
        target_t=0.1,
        l_metric="entropy",
    ).analyze(rows)

    assert report.l_threshold == pytest.approx(1.0)
    assert report.achieved_l == pytest.approx(0.0)
    assert report.l_violating_class_count == 1
    assert report.t_violating_class_count == 2
    assert report.violating_class_count == 2
    assert report.t_violating_classes[0].t_closeness == pytest.approx(0.25)


def test_report_is_deterministic_json_safe_and_does_not_emit_sensitive_values() -> None:
    first = LDiversityChecker(
        ["group"],
        "diagnosis",
        target_l=2,
        target_t=0.25,
    ).check(_synthetic_rows())
    second = analyze_l_diversity(
        _synthetic_rows(),
        ["group"],
        "diagnosis",
        target_l=2,
        target_t=0.25,
    )
    serialized = json.dumps(first.to_dict(), sort_keys=True)

    assert first == second
    assert json.loads(serialized) == first.to_dict()
    assert "flu" not in serialized
    assert "asthma" not in serialized
    assert all(
        item.class_hash.startswith("sha256:") for item in first.equivalence_classes
    )


def test_mapping_access_and_plural_sensitive_attribute_alias() -> None:
    report = analyze_l_diversity(
        _synthetic_rows(),
        ["group"],
        sensitive_attributes=["diagnosis"],
        target_l=2,
        target_t=0.25,
    )

    assert report["class_count"] == 2
    assert report["l_diversity"]["achieved"] == 2
    assert report["t_closeness"]["achieved"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_l": 0}, "target_l"),
        ({"target_l": True}, "target_l"),
        ({"target_t": -0.1}, "target_t"),
        ({"target_t": 1.1}, "target_t"),
        ({"l_metric": "ratio"}, "l_metric"),
        ({"t_distance": "earth_mover"}, "t_distance"),
    ],
)
def test_invalid_policy_values_fail_closed(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LDiversityEngine(["group"], "diagnosis", **kwargs)  # type: ignore[arg-type]


def test_empty_table_does_not_claim_to_meet_policy() -> None:
    report = analyze_l_diversity(
        [],
        ["group"],
        "diagnosis",
        target_l=1,
        target_t=1.0,
    )

    assert report.class_count == 0
    assert report.achieved_l == 0
    assert report.achieved_t == 0.0
    assert report.meets_target is False


def test_entropy_reference_uses_base_two_shannon_bits() -> None:
    rows = [
        {"group": "A", "diagnosis": "one"},
        {"group": "A", "diagnosis": "two"},
        {"group": "A", "diagnosis": "three"},
        {"group": "A", "diagnosis": "three"},
    ]

    report = analyze_l_diversity(
        rows,
        ["group"],
        "diagnosis",
        target_l=2,
        target_t=1.0,
        l_metric="entropy",
    )

    expected = -(0.25 * math.log2(0.25) * 2 + 0.5 * math.log2(0.5))
    assert report.achieved_entropy == pytest.approx(expected)
