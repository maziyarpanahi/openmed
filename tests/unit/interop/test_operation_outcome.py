"""Offline tests for the privacy-safe FHIR validation report."""

from __future__ import annotations

import json

from openmed.interop.fhir.operation_outcome import (
    FHIRValidationReport,
    ValidationFinding,
    build_operation_outcome,
    render_counts,
    render_json,
)


def test_structural_and_policy_findings_map_to_safe_operation_outcome_issues():
    findings = [
        ValidationFinding(
            category="policy",
            path="Patient.identifier[0].value",
            severity="error",
            diagnostics="Synthetic-PHI-0001 violated the local policy.",
        ),
        ValidationFinding(
            category="structural",
            expression="Patient.name",
            severity="warning",
            message="Synthetic-PHI-0001 is missing a required element.",
        ),
    ]

    outcome = build_operation_outcome(findings)

    assert outcome["resourceType"] == "OperationOutcome"
    assert outcome["issue"] == [
        {
            "severity": "error",
            "code": "business-rule",
            "diagnostics": "Policy validation failed; details redacted.",
            "expression": ["Patient.identifier[0].value"],
        },
        {
            "severity": "warning",
            "code": "structure",
            "diagnostics": "Structural validation failed; details redacted.",
            "expression": ["Patient.name"],
        },
    ]

    assert "Synthetic-PHI-0001" not in json.dumps(outcome)


def test_expression_literals_are_redacted_without_losing_the_structural_path():
    outcome = build_operation_outcome(
        [
            {
                "category": "policy",
                "path": "Patient.identifier.where(value='Synthetic-MRN-0001')",
                "message": "Synthetic-MRN-0001 must not be emitted.",
            }
        ]
    )

    expression = outcome["issue"][0]["expression"]
    assert expression == ['Patient.identifier.where(value="[REDACTED]")']
    assert "Synthetic-MRN-0001" not in json.dumps(outcome)


def test_json_and_counts_renderings_are_deterministic_and_counts_only():
    findings = [
        {
            "category": "policy",
            "path": "Patient.telecom[0].value",
            "severity": "warning",
            "message": "Synthetic-PHI-0002 is not permitted.",
        },
        {
            "category": "structural",
            "path": "Patient.name",
            "severity": "error",
            "message": "Synthetic-PHI-0003 is not permitted.",
        },
    ]

    json_output = render_json(findings)
    reversed_json_output = render_json(list(reversed(findings)))
    counts_output = render_counts(list(reversed(findings)))

    assert json_output == reversed_json_output
    assert "Patient.name" in json_output
    assert "Synthetic-PHI-0002" not in json_output
    assert counts_output == (
        "total=2\n"
        "category.policy=1\n"
        "category.structural=1\n"
        "code.business-rule=1\n"
        "code.structure=1\n"
        "severity.error=1\n"
        "severity.warning=1\n"
    )
    assert "Patient.name" not in counts_output
    assert "Synthetic-PHI-0003" not in counts_output


def test_result_buckets_and_explicit_codes_are_supported_offline():
    result = {
        "structural_failures": [{"path": "Observation.code", "code": "required"}],
        "policy_failures": [{"path": "Observation.subject", "code": "security"}],
    }

    outcome = build_operation_outcome(result)

    assert [issue["code"] for issue in outcome["issue"]] == [
        "required",
        "security",
    ]
    assert [issue["expression"] for issue in outcome["issue"]] == [
        ["Observation.code"],
        ["Observation.subject"],
    ]


def test_empty_report_is_valid_and_has_zero_counts():
    report = FHIRValidationReport()

    assert report.to_operation_outcome() == {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": "information",
                "code": "informational",
                "diagnostics": "No issues detected.",
            }
        ],
    }
    assert report.to_counts_text() == "total=0\n"
