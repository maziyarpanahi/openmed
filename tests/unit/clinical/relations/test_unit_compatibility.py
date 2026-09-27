"""Offline unit-dimension checks for quantitative relation candidates."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.relations import (
    UNIT_COMPATIBLE,
    UNIT_INCOMPATIBLE,
    UNIT_UNKNOWN,
    SpanReference,
    UnitCompatibilityReport,
    check_unit_compatibility,
    validate_quantitative_relation,
    validate_quantitative_relations,
)
from openmed.clinical.relations.candidate import RelationCandidate


@pytest.mark.parametrize(
    ("left", "right", "relation_kind", "status"),
    [
        ("mg", "g", "dose", UNIT_COMPATIBLE),
        ("mg/h", "g/s", "rate", UNIT_COMPATIBLE),
        ("mg/dL", "g/L", "concentration", UNIT_COMPATIBLE),
        ("mg/dL", "mmol/L", "laboratory", UNIT_INCOMPATIBLE),
        ("mg", "mL", "dose", UNIT_INCOMPATIBLE),
        ("mg", "g", "rate", UNIT_INCOMPATIBLE),
        ("unknown-unit", "mg", "laboratory", UNIT_UNKNOWN),
    ],
)
def test_check_unit_compatibility_is_dimension_only(left, right, relation_kind, status):
    result = check_unit_compatibility(
        left,
        right,
        relation_kind=relation_kind,
    )

    assert result.status == status
    assert result.review_required is (status != UNIT_COMPATIBLE)
    assert result.decision == ("accept" if status == UNIT_COMPATIBLE else "review")
    assert "magnitude" not in result.to_dict()
    assert "converted" not in result.to_dict()


def test_unknown_and_missing_units_are_explicit_review_findings():
    unknown = validate_quantitative_relation(
        {"relation_type": "drug_to_dose", "unit": "units"}
    )
    missing = validate_quantitative_relation(
        {"relation_type": "lab_result", "value_unit": "mg/dL"}
    )

    assert unknown.status == UNIT_UNKNOWN
    assert unknown.review_required
    assert missing.status == UNIT_UNKNOWN
    assert missing.reason == "missing_comparison_unit"
    assert missing.review_required


def test_unsupported_explicit_relation_kind_fails_closed():
    result = validate_quantitative_relation(
        {"relation_type": "drug_to_dose", "unit": "mg"},
        relation_kind="not-a-quantitative-kind",
    )

    assert result.status == UNIT_UNKNOWN
    assert result.reason == "unsupported_relation_kind"
    assert result.review_required


def test_quantitative_category_checks_reject_wrong_dimensions():
    rate = validate_quantitative_relation(
        {"relation_type": "drug_to_rate", "unit": "mg"}
    )
    concentration = validate_quantitative_relation(
        {"relation_kind": "concentration", "unit": "mg/h"}
    )

    assert rate.status == UNIT_INCOMPATIBLE
    assert rate.reason == "relation_dimension_mismatch"
    assert concentration.status == UNIT_INCOMPATIBLE
    assert concentration.reason == "relation_dimension_mismatch"


def test_nested_quantitative_endpoints_are_compared():
    result = validate_quantitative_relation(
        {
            "relation_kind": "dose",
            "left": {"unit": "mg"},
            "right": {"unit": "mL"},
        }
    )

    assert result.status == UNIT_INCOMPATIBLE
    assert result.reason == "dimension_mismatch"


def test_relation_candidate_adapter_uses_offsets_without_source_text():
    candidate = RelationCandidate(
        relation_type="drug_to_dose",
        head=SpanReference(
            text="synthetic medicine",
            label="Drug",
            start=0,
            end=19,
            score=0.9,
        ),
        attribute=SpanReference(
            text="5 mg",
            label="Dose",
            start=20,
            end=24,
            score=0.9,
        ),
        score=0.9,
        confidence=0.9,
        features={},
        explanation=(),
    )

    result = validate_quantitative_relation(candidate)

    assert result.status == UNIT_COMPATIBLE
    assert result.left_unit == "mg"
    assert result.head_offset == (0, 19)
    assert result.tail_offset == (20, 24)
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "synthetic medicine" not in serialized
    assert "5 mg" not in serialized


def test_batch_report_is_stable_and_value_free():
    candidates = [
        {
            "relation_type": "drug_to_dose",
            "unit": "mg",
            "value": "synthetic-only-value",
        },
        {
            "relation_type": "lab_result",
            "value_unit": "mg/dL",
            "reference_unit": "mmol/L",
        },
        {"relation_type": "drug_to_rate", "unit": "unknown-unit"},
    ]

    report = validate_quantitative_relations(candidates)

    assert isinstance(report, UnitCompatibilityReport)
    assert [result.candidate_index for result in report] == [0, 1, 2]
    assert [result.status for result in report] == [
        UNIT_COMPATIBLE,
        UNIT_INCOMPATIBLE,
        UNIT_UNKNOWN,
    ]
    assert report.has_review_findings
    assert len(report.compatible) == 1
    assert len(report.review_required) == 2
    serialized = report.to_json()
    assert "synthetic-only-value" not in serialized
    assert "magnitude" not in serialized
    assert serialized == report.to_json()


def test_report_rejects_string_iterables():
    with pytest.raises(TypeError, match="iterable of relation candidates"):
        validate_quantitative_relations("not a candidate list")
