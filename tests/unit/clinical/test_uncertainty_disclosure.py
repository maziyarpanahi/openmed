"""Tests for the privacy-safe uncertainty disclosure completeness audit."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from openmed.clinical import (
    DUPLICATE_EVIDENCE_REFERENCES,
    INVALID_DISPLAY_HINTS,
    INVALID_REVIEW_STATE,
    MISSING_EVIDENCE_REFERENCES,
    MISSING_REASON_CODES,
    MISSING_REQUIRED_CATEGORY,
    MISSING_UNCERTAINTY_CATEGORIES,
    UncertaintyDisclosureReport,
    audit_uncertainty_disclosure,
    audit_uncertainty_disclosures,
)


def _complete_claim(claim_id: str = "synthetic-claim-001") -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "uncertainty_disclosure": {
            "uncertainty_categories": ["epistemic"],
            "reason_codes": ["reason.synthetic"],
            "evidence_references": ["evidence.synthetic.001"],
            "review_state": "pending",
            "display_hints": {"max_chars": 240, "max_items": 4},
        },
    }


def test_complete_synthetic_claim_is_a_clean_report():
    report = audit_uncertainty_disclosures([_complete_claim()])

    assert isinstance(report, UncertaintyDisclosureReport)
    assert report.is_complete
    assert report.summary == {
        "checked_claims": 1,
        "compliant_claims": 1,
        "non_compliant_claims": 0,
        "complete": True,
    }
    assert report.findings == ()
    assert all(count == 0 for count in report.issue_counts.values())


def test_report_is_json_safe_and_has_only_opaque_claim_identity():
    claim = _complete_claim("synthetic-opaque-claim-007")
    report = audit_uncertainty_disclosures(
        [
            claim,
            {
                "claim_id": "synthetic-incomplete-008",
                "uncertainty_disclosure": {},
            },
        ]
    )

    encoded = json.dumps(report.to_dict(), sort_keys=True)
    assert "synthetic-opaque-claim-007" not in encoded
    assert "synthetic-incomplete-008" not in encoded
    assert "epistemic" not in encoded
    assert "reason.synthetic" not in encoded
    assert "evidence.synthetic.001" not in encoded
    assert all(finding.claim_key.startswith("sha256:") for finding in report.findings)
    assert report.issue_counts[MISSING_UNCERTAINTY_CATEGORIES] == 1
    assert report.issue_counts[MISSING_REASON_CODES] == 1


def test_missing_and_duplicate_fields_are_reported_without_values():
    claim = {
        "claim_id": "synthetic-sensitive-looking-id",
        "uncertainty_disclosure": {
            "uncertainty_categories": ["epistemic", "epistemic"],
            "reason_codes": [],
            "evidence_references": ["evidence.synthetic.002", "evidence.synthetic.002"],
            "review_state": "reviewed",
            "display_hints": {"max_chars": 120, "max_items": 2},
        },
    }

    report = audit_uncertainty_disclosures([claim])
    finding = report.findings[0]

    assert finding.claim_key != claim["claim_id"]
    assert MISSING_REASON_CODES in finding.issue_codes
    assert DUPLICATE_EVIDENCE_REFERENCES in finding.issue_codes
    assert "epistemic" not in repr(report)
    assert "evidence.synthetic.002" not in repr(report)


@pytest.mark.parametrize(
    ("review_state", "display_hints"),
    [
        ("clinical conclusion", {"max_chars": 120, "max_items": 2}),
        ("pending", {"max_chars": 0, "max_items": 2}),
        ("pending", {"max_chars": 120, "max_items": True}),
    ],
)
def test_invalid_review_state_or_display_bound_is_flagged(review_state, display_hints):
    claim = _complete_claim()
    disclosure = claim["uncertainty_disclosure"]
    assert isinstance(disclosure, dict)
    disclosure["review_state"] = review_state
    disclosure["display_hints"] = display_hints

    report = audit_uncertainty_disclosures([claim])
    issue_codes = report.findings[0].issue_codes

    if review_state == "clinical conclusion":
        assert INVALID_REVIEW_STATE in issue_codes
    else:
        assert INVALID_REVIEW_STATE not in issue_codes
    if display_hints["max_chars"] == 0 or display_hints["max_items"] is True:
        assert INVALID_DISPLAY_HINTS in issue_codes
    else:
        assert INVALID_DISPLAY_HINTS not in issue_codes


def test_required_categories_and_reference_count_are_structural_checks():
    report = audit_uncertainty_disclosures(
        [_complete_claim()],
        required_categories=("epistemic", "aleatoric"),
        min_evidence_references=2,
    )

    finding = report.findings[0]
    assert MISSING_REQUIRED_CATEGORY in finding.issue_codes
    assert MISSING_EVIDENCE_REFERENCES in finding.issue_codes
    assert "aleatoric" not in repr(report)


def test_top_level_aliases_and_metadata_container_are_supported():
    claim = {
        "id": "synthetic-alias-claim",
        "metadata": {
            "uncertainty": {
                "categories": ["data_quality"],
                "reason_code": "reason.synthetic.alias",
                "evidence_refs": [{"ref_id": "provenance.synthetic.alias"}],
                "review_status": "in-review",
                "display_bounds": {"max_length": 80, "max_count": 1},
            }
        },
    }

    report = audit_uncertainty_disclosures(claim)

    assert report.is_complete


@dataclass(frozen=True)
class _SyntheticClaim:
    claim_id: str
    uncertainty_disclosure: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "uncertainty_disclosure": self.uncertainty_disclosure,
        }


def test_to_dict_claim_objects_and_singular_alias_work():
    claim = _complete_claim("synthetic-object-claim")
    obj = _SyntheticClaim(
        claim_id=claim["claim_id"],
        uncertainty_disclosure=claim["uncertainty_disclosure"],
    )

    report = audit_uncertainty_disclosure([obj])

    assert report.is_complete
    assert report.to_dict()["summary"]["checked_claims"] == 1


def test_invalid_claim_shape_does_not_echo_input():
    sensitive_marker = "synthetic-no-raw-value-marker"

    with pytest.raises(TypeError, match="each claim must be a mapping") as exc_info:
        audit_uncertainty_disclosures([sensitive_marker])

    assert sensitive_marker not in str(exc_info.value)


def test_empty_input_is_deterministically_complete():
    report = audit_uncertainty_disclosures([])

    assert report.is_complete
    assert report.checked_claims == 0
    assert report.compliant_claims == 0
    assert report.findings == ()


def test_claims_without_ids_still_have_order_independent_opaque_keys():
    claims = [
        {"uncertainty_disclosure": {"reason_codes": ["reason.synthetic.a"]}},
        {"uncertainty_disclosure": {"reason_codes": ["reason.synthetic.b"]}},
    ]

    assert (
        audit_uncertainty_disclosures(claims).to_dict()
        == audit_uncertainty_disclosures(list(reversed(claims))).to_dict()
    )
