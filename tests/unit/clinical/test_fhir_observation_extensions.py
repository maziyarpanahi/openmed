"""Tests for conservative, offline FHIR Observation extension checks."""

import json
from copy import deepcopy

import pytest

from openmed.clinical.exporters.fhir import (
    COREFERENCE_EVIDENCE_EXTENSION_URL,
    FHIR_R4,
    FHIR_R5,
    OBSERVATION_UNKNOWN_STATE_EXTENSION_URL,
    ObservationExtensionSpec,
    check_observation_extensions,
    validate_observation_extensions,
)


def _observation(extensions):
    return {"resourceType": "Observation", "extension": extensions}


def _unknown_state(state="unknown"):
    return {
        "url": OBSERVATION_UNKNOWN_STATE_EXTENSION_URL,
        "valueCode": state,
    }


@pytest.mark.parametrize("version", [FHIR_R4, FHIR_R5, "4.0.1", "5.0.0"])
def test_allowlisted_unknown_state_is_valid_in_r4_and_r5(version):
    resource = _observation([_unknown_state()])

    first = check_observation_extensions(resource, fhir_version=version)
    second = check_observation_extensions(resource, fhir_version=version)

    assert first == []
    assert first == second


def test_unknown_url_and_value_are_rejected_without_echoing_the_value():
    sensitive_synthetic_value = "SYNTHETIC-SENSITIVE-VALUE-DO-NOT-EMIT"
    resource = _observation(
        [
            {
                "url": "https://synthetic.example/fhir/StructureDefinition/raw",
                "valueString": sensitive_synthetic_value,
            }
        ]
    )

    findings = check_observation_extensions(resource)

    assert [finding["finding_code"] for finding in findings] == [
        "unsupported-extension-url"
    ]
    assert sensitive_synthetic_value not in json.dumps(findings)


def test_known_url_rejects_unsupported_value_type_and_inferred_marker():
    resource = _observation(
        [
            {
                "url": OBSERVATION_UNKNOWN_STATE_EXTENSION_URL,
                "valueString": "SYNTHETIC-VALUE",
                "inferred": True,
            }
        ]
    )

    findings = check_observation_extensions(resource)

    assert [finding["finding_code"] for finding in findings] == [
        "inferred-extension-content",
        "unsupported-extension-fields",
        "unsupported-extension-value-type",
    ]
    assert "SYNTHETIC-VALUE" not in json.dumps(findings)


def test_unknown_state_must_be_explicit_and_allowlisted():
    invalid = check_observation_extensions(_observation([_unknown_state("inferred")]))
    unsupported = check_observation_extensions(
        _observation([_unknown_state("not-stated")])
    )

    assert [finding["finding_code"] for finding in invalid] == [
        "inferred-extension-content"
    ]
    assert [finding["finding_code"] for finding in unsupported] == [
        "invalid-explicit-state"
    ]
    assert all("inferred" not in finding["diagnostics"] for finding in unsupported)


def test_cardinality_and_extension_shape_are_checked():
    duplicate = _observation([_unknown_state(), _unknown_state()])
    malformed = _observation(
        [
            {
                **_unknown_state(),
                "extension": [],
            }
        ]
    )

    duplicate_findings = check_observation_extensions(duplicate)
    malformed_findings = check_observation_extensions(malformed)

    assert [finding["finding_code"] for finding in duplicate_findings] == [
        "extension-cardinality"
    ]
    assert [finding["finding_code"] for finding in malformed_findings] == [
        "mixed-extension-content"
    ]


def test_nested_allowlisted_coreference_extension_is_valid():
    extension = {
        "url": COREFERENCE_EVIDENCE_EXTENSION_URL,
        "extension": [
            {"url": "clusterId", "valueString": "synthetic-cluster"},
            {
                "url": "representative",
                "extension": [
                    {"url": "start", "valueUnsignedInt": 3},
                    {"url": "end", "valueUnsignedInt": 8},
                    {"url": "textHash", "valueString": "hash"},
                ],
            },
            {
                "url": "supportingMention",
                "extension": [
                    {"url": "start", "valueUnsignedInt": 20},
                    {"url": "end", "valueUnsignedInt": 26},
                    {"url": "textHash", "valueString": "hash-2"},
                ],
            },
        ],
    }

    assert check_observation_extensions(_observation([extension])) == []


def test_custom_rule_can_be_scoped_to_one_fhir_mode():
    custom_url = "https://synthetic.example/fhir/StructureDefinition/flag"
    rules = {
        custom_url: ObservationExtensionSpec(
            value_types=("valueBoolean",),
            fhir_versions=(FHIR_R5,),
        )
    }
    resource = _observation([{"url": custom_url, "valueBoolean": True}])

    r4 = check_observation_extensions(resource, allowed_extensions=rules)
    r5 = check_observation_extensions(resource, allowed_extensions=rules, mode="R5")

    assert [finding["finding_code"] for finding in r4] == ["unsupported-fhir-version"]
    assert r5 == []


def test_custom_rule_checks_the_runtime_shape_of_the_allowlisted_value():
    custom_url = "https://synthetic.example/fhir/StructureDefinition/flag"
    resource = _observation([{"url": custom_url, "valueBoolean": "yes"}])

    findings = check_observation_extensions(
        resource,
        allowed_extensions={
            custom_url: {"value_types": ["valueBoolean"]},
        },
    )

    assert [finding["finding_code"] for finding in findings] == [
        "invalid-extension-value-shape"
    ]


def test_checker_does_not_mutate_the_observation_and_outcome_is_fhir_shaped():
    resource = _observation([_unknown_state()])
    snapshot = deepcopy(resource)

    outcome = validate_observation_extensions(resource)

    assert resource == snapshot
    assert outcome == {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": "information",
                "code": "informational",
                "diagnostics": "No issues detected.",
            }
        ],
    }


def test_malformed_resource_reports_only_safe_paths():
    sensitive_synthetic_value = "SYNTHETIC-RESOURCE-VALUE"
    resource = {
        "resourceType": "Patient",
        "extension": sensitive_synthetic_value,
    }

    findings = check_observation_extensions(resource)

    assert [finding["finding_code"] for finding in findings] == [
        "invalid-resource-type",
        "extension-not-array",
    ]
    assert sensitive_synthetic_value not in json.dumps(findings)
