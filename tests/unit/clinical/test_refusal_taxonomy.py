"""Synthetic offline tests for the clinical refusal taxonomy."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    REFUSAL_CATEGORY_VALUES,
    ClinicalRefusal,
    RefusalCategory,
    RefusalReason,
    RefusalReport,
    RefusalTaxonomy,
    aggregate_refusals,
    build_refusal,
    remediation_hint_for,
    serialize_refusal_report,
    serialize_refusals,
)


def test_categories_are_stable_and_have_fixed_hints() -> None:
    assert REFUSAL_CATEGORY_VALUES == (
        "missing_evidence",
        "policy_block",
        "ambiguity",
        "unsupported_request",
    )

    for category in RefusalCategory:
        hint = remediation_hint_for(category)
        assert hint
        assert category.value in RefusalTaxonomy.categories
        assert RefusalTaxonomy.remediation_hints[category.value] == hint


def test_reason_serializes_only_taxonomy_fields() -> None:
    reason = build_refusal("policy-block")

    assert isinstance(reason, ClinicalRefusal)
    assert reason.category is RefusalCategory.POLICY_BLOCK
    assert reason.to_dict() == {
        "category": "policy_block",
        "count": 1,
        "remediation_hint": (
            "Review the applicable policy and use an approved workflow."
        ),
    }


def test_report_aggregates_in_taxonomy_order_and_round_trips() -> None:
    report = aggregate_refusals(
        [
            "unsupported_request",
            RefusalReason("policy_block", count=2),
            "ambiguity",
            {"category": "policy-block", "count": 1, "message": "synthetic"},
        ]
    )

    assert isinstance(report, RefusalReport)
    assert report.categories == ("policy_block", "ambiguity", "unsupported_request")
    assert dict(report.counts) == {
        "policy_block": 3,
        "ambiguity": 1,
        "unsupported_request": 1,
    }
    assert report.total_count == 5
    assert dict(report.remediation_hints) == {
        "policy_block": ("Review the applicable policy and use an approved workflow."),
        "ambiguity": "Clarify the request or supply disambiguating context.",
        "unsupported_request": (
            "Use a supported clinical workflow or consult the capability guidance."
        ),
    }
    assert report.to_dict() == {
        "counts": {
            "policy_block": 3,
            "ambiguity": 1,
            "unsupported_request": 1,
        },
        "remediation_hints": {
            "policy_block": (
                "Review the applicable policy and use an approved workflow."
            ),
            "ambiguity": "Clarify the request or supply disambiguating context.",
            "unsupported_request": (
                "Use a supported clinical workflow or consult the capability guidance."
            ),
        },
    }

    restored = RefusalReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored.to_dict() == report.to_dict()
    assert restored.to_safe_dict() == serialize_refusal_report(report)


def test_serialization_does_not_retain_untrusted_fields() -> None:
    synthetic_note_text = "synthetic note value that must not be retained"
    report = RefusalReport.from_dict(
        {
            "counts": {"missing_evidence": 2},
            "remediation_hints": {"missing_evidence": synthetic_note_text},
            "message": synthetic_note_text,
        }
    )

    encoded = json.dumps(report.to_dict())
    assert synthetic_note_text not in encoded
    assert report.to_dict() == {
        "counts": {"missing_evidence": 2},
        "remediation_hints": {
            "missing_evidence": (
                "Provide the minimum required evidence and retry when it is available."
            )
        },
    }


def test_invalid_category_and_count_errors_do_not_echo_input() -> None:
    synthetic_sensitive_value = "synthetic-patient-note-value"

    with pytest.raises(ValueError) as category_error:
        build_refusal(synthetic_sensitive_value)
    assert str(category_error.value) == "unsupported refusal category"
    assert synthetic_sensitive_value not in str(category_error.value)

    with pytest.raises(ValueError) as count_error:
        build_refusal("ambiguity", count=-1)
    assert str(count_error.value) == "refusal count must be a positive integer"
    assert synthetic_sensitive_value not in str(count_error.value)


def test_empty_report_has_no_free_form_or_category_content() -> None:
    report = RefusalReport()

    assert report.total_count == 0
    assert report.categories == ()
    assert report.to_dict() == {"counts": {}, "remediation_hints": {}}
    assert serialize_refusals(None) == report.to_dict()


def test_taxonomy_facade_matches_functional_api() -> None:
    categories = ["missing_evidence", "missing_evidence", "ambiguity"]

    assert RefusalTaxonomy.classify("missing_evidence") == build_refusal(
        "missing_evidence"
    )
    assert (
        RefusalTaxonomy.aggregate(categories).to_dict()
        == aggregate_refusals(categories).to_dict()
    )
    assert RefusalTaxonomy.serialize(categories) == serialize_refusals(categories)


def test_from_dict_ignores_extra_fields_without_copying_them() -> None:
    reason = RefusalReason.from_dict(
        {
            "category": "unsupported_request",
            "count": 2,
            "raw_note": "synthetic raw note",
        }
    )

    assert reason.to_dict()["category"] == "unsupported_request"
    assert "raw_note" not in reason.to_dict()
