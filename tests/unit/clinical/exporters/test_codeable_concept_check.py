"""Tests for CodeableConcept structural consistency checks (OM-414)."""

import json

import pytest

from openmed.clinical.exporters import check_codeable_concept
from openmed.clinical.exporters.fhir import to_operation_outcome


def _codes(findings):
    return [finding["finding_code"] for finding in findings]


def test_codeless_concept_without_text_produces_hard_finding():
    findings = check_codeable_concept({})

    assert _codes(findings) == ["missing-text-when-codeless"]
    assert findings[0]["severity"] == "error"
    assert findings[0]["code"] == "required"
    assert findings[0]["expression"] == ["CodeableConcept.text"]


def test_empty_coding_array_is_reported_deterministically():
    findings = check_codeable_concept({"coding": []})

    assert _codes(findings) == [
        "missing-text-when-codeless",
        "empty-coding-array",
    ]
    assert [finding["severity"] for finding in findings] == ["error", "warning"]


def test_empty_coding_array_with_text_reports_only_empty_coding_array():
    findings = check_codeable_concept({"coding": [], "text": "Creatinine"})

    assert _codes(findings) == ["empty-coding-array"]
    assert findings[0]["code"] == "structure"


@pytest.mark.parametrize("text", ["Creatinine", "creatinine", "  CREATININE  "])
def test_matching_display_and_text_produces_no_findings(text):
    concept = {
        "coding": [
            {
                "system": "http://loinc.org",
                "code": "2160-0",
                "display": "Creatinine",
            }
        ],
        "text": text,
    }

    assert check_codeable_concept(concept) == []


def test_text_with_no_matching_display_produces_soft_mismatch():
    findings = check_codeable_concept(
        {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "2160-0",
                    "display": "Creatinine",
                }
            ],
            "text": "Glucose",
        },
        expression="Observation.code",
    )

    assert _codes(findings) == ["text-display-mismatch"]
    assert findings[0]["severity"] == "warning"
    assert findings[0]["code"] == "invariant"
    assert findings[0]["expression"] == ["Observation.code.text"]


def test_coding_without_text_or_display_produces_soft_mismatch():
    findings = check_codeable_concept(
        {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "2160-0",
                }
            ]
        }
    )

    assert _codes(findings) == ["text-display-mismatch"]
    assert findings[0]["severity"] == "warning"


def test_coding_display_can_provide_fallback_when_text_is_missing():
    findings = check_codeable_concept(
        {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "2160-0",
                    "display": "Creatinine",
                }
            ]
        }
    )

    assert findings == []


def test_findings_are_json_serializable_and_deterministic():
    concept = {"coding": [], "text": ""}

    first = check_codeable_concept(concept)
    second = check_codeable_concept(concept)

    assert first == second
    assert json.loads(json.dumps(first)) == first


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        [],
        "not a mapping",
        {"coding": None, "text": None},
        {"coding": ["bad", 42, {"display": object()}], "text": object()},
    ],
)
def test_checker_never_raises_on_malformed_input(malformed):
    findings = check_codeable_concept(malformed)

    assert isinstance(findings, list)
    json.dumps(findings)


def test_findings_map_directly_to_operation_outcome_builder():
    findings = check_codeable_concept(
        {"coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
        expression="Observation.code",
    )

    outcome = to_operation_outcome(findings)

    assert outcome["resourceType"] == "OperationOutcome"
    assert outcome["issue"] == [
        {
            "severity": "warning",
            "code": "invariant",
            "diagnostics": findings[0]["diagnostics"],
            "expression": ["Observation.code.text"],
        }
    ]
