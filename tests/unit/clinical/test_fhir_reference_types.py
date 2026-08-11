"""Focused tests for offline FHIR reference target-type validation."""

import json

from openmed.clinical.exporters.fhir import to_bundle
from openmed.clinical.exporters.fhir.reference_types import (
    check_reference_targets,
    find_reference_target_issues,
)


def _issue(outcome: dict, index: int = 0) -> dict:
    return outcome["issue"][index]


def test_valid_r4_relative_references_pass_without_network_resolution():
    resources = [
        {"resourceType": "Patient", "id": "subject-a"},
        {"resourceType": "Encounter", "id": "visit-a"},
        {
            "resourceType": "Observation",
            "id": "observation-a",
            "subject": {"reference": "Patient/subject-a"},
            "encounter": {"reference": "Encounter/visit-a"},
        },
    ]

    assert check_reference_targets(resources) == {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": "information",
                "code": "informational",
                "diagnostics": "No issues detected.",
            }
        ],
    }


def test_missing_target_is_reported_without_echoing_the_reference_value():
    sensitive_synthetic_id = "synthetic-private-reference-001"
    outcome = check_reference_targets(
        [
            {
                "resourceType": "Observation",
                "id": "observation-a",
                "subject": {"reference": f"Patient/{sensitive_synthetic_id}"},
            }
        ]
    )

    assert _issue(outcome) == {
        "severity": "error",
        "code": "not-found",
        "diagnostics": (
            "Referenced resource is missing from the local validation set."
        ),
        "expression": ["resources[0].subject"],
    }
    assert sensitive_synthetic_id not in json.dumps(outcome)


def test_untyped_reference_with_multiple_local_ids_is_ambiguous():
    outcome = check_reference_targets(
        [
            {"resourceType": "Patient", "id": "shared-synthetic-id"},
            {"resourceType": "Group", "id": "shared-synthetic-id"},
            {
                "resourceType": "Observation",
                "id": "observation-a",
                "subject": {"reference": "shared-synthetic-id"},
            },
        ]
    )

    assert _issue(outcome)["code"] == "multiple-matches"
    assert _issue(outcome)["expression"] == ["resources[2].subject"]
    assert "shared-synthetic-id" not in json.dumps(outcome)


def test_disallowed_target_type_is_reported_even_when_target_exists():
    outcome = check_reference_targets(
        [
            {"resourceType": "Observation", "id": "other-observation"},
            {
                "resourceType": "Observation",
                "id": "observation-a",
                "subject": {"reference": "Observation/other-observation"},
            },
        ]
    )

    assert _issue(outcome) == {
        "severity": "error",
        "code": "structure",
        "diagnostics": "Reference target type is not allowed for this field.",
        "expression": ["resources[1].subject"],
    }


def test_contained_reference_is_explicitly_non_blocking_and_value_free():
    outcome = check_reference_targets(
        [
            {
                "resourceType": "Condition",
                "id": "condition-a",
                "subject": {"reference": "#contained-synthetic-subject"},
            }
        ]
    )

    assert _issue(outcome) == {
        "severity": "information",
        "code": "not-supported",
        "diagnostics": (
            "Contained references are not resolved by this local target check."
        ),
        "expression": ["resources[0].subject"],
    }
    assert "contained-synthetic-subject" not in json.dumps(outcome)


def test_r5_medication_usage_uses_the_r5_allowlist():
    resources = [
        {"resourceType": "Patient", "id": "subject-a"},
        {
            "resourceType": "MedicationUsage",
            "id": "usage-a",
            "subject": {"reference": "Patient/subject-a"},
        },
    ]

    assert (
        check_reference_targets(resources, fhir_version="R5")["issue"][0]["code"]
        == "informational"
    )


def test_bundle_urn_references_are_checked_against_local_full_urls():
    bundle = to_bundle(
        [
            {"resourceType": "Patient", "id": "subject-a"},
            {
                "resourceType": "Observation",
                "id": "observation-a",
                "subject": {"reference": "Patient/subject-a"},
            },
        ],
        doc_id="synthetic-document",
    )

    assert check_reference_targets(bundle)["issue"][0]["code"] == "informational"


def test_external_absolute_and_logical_references_are_not_dereferenced():
    resources = [
        {
            "resourceType": "Observation",
            "id": "observation-a",
            "subject": {"reference": "https://synthetic.example/Patient/a"},
        },
        {
            "resourceType": "Observation",
            "id": "observation-b",
            "subject": {
                "identifier": {
                    "system": "https://synthetic.example/ids",
                    "value": "synthetic-subject-a",
                }
            },
        },
    ]

    assert check_reference_targets(resources)["issue"][0]["code"] == "informational"


def test_findings_are_deterministic_and_do_not_mutate_input():
    resources = [
        {
            "resourceType": "Condition",
            "id": "condition-a",
            "subject": {"reference": "Patient/missing-a"},
        }
    ]
    snapshot = json.dumps(resources, sort_keys=True)

    first = find_reference_target_issues(resources)
    second = find_reference_target_issues(resources)

    assert first == second
    assert first[0].reason == "missing"
    assert first[0].path == "resources[0].subject"
    assert json.dumps(resources, sort_keys=True) == snapshot
