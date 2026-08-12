"""Focused tests for offline FHIR profile-declaration consistency checks."""

from __future__ import annotations

import copy
import json

from openmed.clinical.exporters.fhir import (
    DUPLICATE_PROFILE_DECLARATION,
    FHIR_R4,
    FHIR_R5,
    MISSING_PROFILE_DECLARATION,
    PROFILE_FHIR_VERSION_MISMATCH,
    PROFILE_RESOURCE_TYPE_MISMATCH,
    PROFILE_VALIDATION_MODE_MISMATCH,
    UNKNOWN_PROFILE_DECLARATION,
    check_profile_declarations,
    to_bundle,
    validate_profile_declarations,
)

PATIENT_PROFILE = "https://synthetic.example/fhir/StructureDefinition/patient"
OBSERVATION_PROFILE = "https://synthetic.example/fhir/StructureDefinition/observation"
UNKNOWN_PROFILE = "https://synthetic.example/fhir/StructureDefinition/unknown"

PROFILE_CATALOG = {
    PATIENT_PROFILE: {
        "resource_type": "Patient",
        "fhir_versions": [FHIR_R4, FHIR_R5],
        "validation_modes": ["strict", "advisory"],
    },
    OBSERVATION_PROFILE: {
        "resource_type": "Observation",
        "fhir_versions": [FHIR_R4],
    },
}


def _patient(*profiles: str) -> dict:
    return {
        "resourceType": "Patient",
        "id": "synthetic-patient",
        "meta": {"profile": list(profiles)},
    }


def _codes(findings: list[dict]) -> list[str]:
    return [finding["finding_code"] for finding in findings]


def test_known_declarations_are_deterministic_and_value_free() -> None:
    resource = _patient(PATIENT_PROFILE)
    first = check_profile_declarations(resource, PROFILE_CATALOG)
    second = check_profile_declarations(resource, PROFILE_CATALOG)

    assert first == second == []
    assert resource["meta"]["profile"] == [PATIENT_PROFILE]


def test_missing_profile_declaration_is_classified() -> None:
    findings = check_profile_declarations(
        {"resourceType": "Patient"},
        PROFILE_CATALOG,
    )

    assert _codes(findings) == [MISSING_PROFILE_DECLARATION]
    assert findings[0]["expression"] == ["Patient.meta.profile"]


def test_duplicate_and_unknown_declarations_are_classified() -> None:
    findings = check_profile_declarations(
        _patient(PATIENT_PROFILE, PATIENT_PROFILE, UNKNOWN_PROFILE),
        PROFILE_CATALOG,
    )

    assert _codes(findings) == [
        UNKNOWN_PROFILE_DECLARATION,
        DUPLICATE_PROFILE_DECLARATION,
    ]
    serialized = json.dumps(findings)
    assert UNKNOWN_PROFILE not in serialized
    assert PATIENT_PROFILE not in serialized


def test_resource_type_and_fhir_version_conflicts_are_reported() -> None:
    wrong_type = check_profile_declarations(
        {"resourceType": "Observation", "meta": {"profile": [PATIENT_PROFILE]}},
        PROFILE_CATALOG,
    )
    wrong_version = check_profile_declarations(
        {
            "resourceType": "Observation",
            "meta": {"profile": [OBSERVATION_PROFILE]},
        },
        PROFILE_CATALOG,
        fhir_version=FHIR_R5,
    )

    assert _codes(wrong_type) == [PROFILE_RESOURCE_TYPE_MISMATCH]
    assert _codes(wrong_version) == [PROFILE_FHIR_VERSION_MISMATCH]


def test_validation_mode_alias_and_required_catalog_are_checked() -> None:
    mode_findings = check_profile_declarations(
        _patient(PATIENT_PROFILE),
        {
            PATIENT_PROFILE: {
                "resource_type": "Patient",
                "validation_modes": ["strict"],
            }
        },
        validation_mode="advisory",
    )
    required_findings = check_profile_declarations(
        {"resourceType": "Observation", "meta": {"profile": []}},
        {OBSERVATION_PROFILE: {"resource_type": "Observation"}},
        expected_profiles={"Observation": [OBSERVATION_PROFILE]},
        require_profile=False,
    )

    assert _codes(mode_findings) == [PROFILE_VALIDATION_MODE_MISMATCH]
    assert _codes(required_findings) == [MISSING_PROFILE_DECLARATION]


def test_bundle_input_and_r5_alias_are_supported_without_mutation() -> None:
    bundle = to_bundle(
        [
            _patient(PATIENT_PROFILE),
            {
                "resourceType": "Observation",
                "meta": {"profile": [OBSERVATION_PROFILE]},
            },
        ],
        doc_id="synthetic-document",
    )
    snapshot = copy.deepcopy(bundle)

    findings = check_profile_declarations(
        bundle,
        PROFILE_CATALOG,
        fhir_mode="5.0.0",
        require_profile=False,
    )

    assert _codes(findings) == [PROFILE_FHIR_VERSION_MISMATCH]
    assert findings[0]["expression"] == ["Bundle.entry[1].resource.meta.profile[0]"]
    assert bundle == snapshot


def test_operation_outcome_adapter_has_a_safe_fhir_shape() -> None:
    outcome = validate_profile_declarations(
        _patient(UNKNOWN_PROFILE),
        PROFILE_CATALOG,
    )

    assert outcome == {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": "error",
                "code": "not-found",
                "diagnostics": (
                    "Profile declaration is not present in the injected local catalog."
                ),
                "expression": ["Patient.meta.profile[0]"],
            }
        ],
    }
    assert UNKNOWN_PROFILE not in json.dumps(outcome)
