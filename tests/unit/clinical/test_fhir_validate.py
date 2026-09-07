"""Tests for dependency-free FHIR R4 base structural validation."""

from __future__ import annotations

import json
import socket
from importlib import resources
from typing import Any

from openmed.clinical.exporters.fhir import (
    BASE_R4_RESOURCE_TYPES,
    ValidationFinding,
    ValidationResult,
    from_validation_result,
    validate_bundle,
    validate_resource,
)
from openmed.interop.fhir import validation_result as interop_validation_result


def _observation(**updates: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": "synthetic-observation",
        "status": "final",
        "code": {"text": "synthetic measurement"},
        "subject": {"reference": "Patient/synthetic"},
        "valueQuantity": {
            "value": 7.2,
            "unit": "mmol/L",
            "system": "http://unitsofmeasure.org",
            "code": "mmol/L",
        },
    }
    resource.update(updates)
    return resource


def _condition(**updates: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Condition",
        "verificationStatus": {
            "coding": [
                {
                    "system": (
                        "http://terminology.hl7.org/CodeSystem/condition-ver-status"
                    ),
                    "code": "confirmed",
                }
            ]
        },
        "code": {"text": "synthetic condition"},
        "subject": {"reference": "Patient/synthetic"},
    }
    resource.update(updates)
    return resource


def test_well_formed_observation_validates_cleanly() -> None:
    result = validate_resource(_observation())

    assert isinstance(result, ValidationResult)
    assert result.is_valid
    assert result.errors == ()
    assert result.warnings == ()
    assert result.findings == ()


def test_observation_missing_status_has_required_element_error() -> None:
    observation = _observation()
    observation.pop("status")

    result = validate_resource(observation)

    assert not result.is_valid
    assert result.errors == (
        ValidationFinding(
            severity="error",
            location="Observation.status",
            message="Required element is missing or empty.",
            code="required",
        ),
    )


def test_condition_outside_required_verification_binding_is_flagged() -> None:
    result = validate_resource(
        _condition(
            verificationStatus={
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/condition-ver-status"
                        ),
                        "code": "synthetic-invalid-status",
                    }
                ]
            }
        )
    )

    assert [(item.code, item.location) for item in result.errors] == [
        ("code-invalid", "Condition.verificationStatus")
    ]
    assert "synthetic-invalid-status" not in result.errors[0].message


def test_required_binding_requires_the_fixed_code_system() -> None:
    result = validate_resource(
        _condition(
            verificationStatus={
                "coding": [
                    {"system": "https://openmed.example/wrong", "code": "confirmed"}
                ]
            }
        )
    )

    assert [item.code for item in result.errors] == ["code-invalid"]


def test_cardinality_and_primitive_datatype_violations_are_structured() -> None:
    result = validate_resource(
        _observation(
            status=["final", "amended"],
            valueBoolean="true",
            valueQuantity=None,
        )
    )

    assert ("structure", "Observation.status") in {
        (item.code, item.location) for item in result.errors
    }
    assert ("invalid", "Observation.valueBoolean") in {
        (item.code, item.location) for item in result.errors
    }


def test_choice_elements_enforce_one_selected_value_and_selected_type() -> None:
    result = validate_resource(
        _observation(valueQuantity=None, valueString="synthetic", valueBoolean=True)
    )

    assert any(
        item.code == "structure"
        and item.location == "Observation.value"
        and "cardinality" in item.message
        for item in result.errors
    )

    wrong_choice = validate_resource(_observation(valueQuantity="not-a-quantity"))
    assert ("invalid", "Observation.valueQuantity") in {
        (item.code, item.location) for item in wrong_choice.errors
    }


def test_repeating_elements_require_json_arrays() -> None:
    result = validate_resource(_observation(performer={"reference": "Practitioner/x"}))

    assert ("structure", "Observation.performer") in {
        (item.code, item.location) for item in result.errors
    }


def test_bundle_validates_every_entry_and_qualifies_locations() -> None:
    missing_status = _observation()
    missing_status.pop("status")
    invalid_condition = _condition(
        verificationStatus={
            "coding": [
                {
                    "system": (
                        "http://terminology.hl7.org/CodeSystem/condition-ver-status"
                    ),
                    "code": "synthetic-invalid-status",
                }
            ]
        }
    )
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {"resource": missing_status},
            {"resource": invalid_condition},
        ],
    }

    result = validate_bundle(bundle)

    assert [(item.code, item.location) for item in result.errors] == [
        ("required", "Bundle.entry[0].resource.status"),
        ("code-invalid", "Bundle.entry[1].resource.verificationStatus"),
    ]


def test_all_bundled_resource_types_have_a_clean_minimal_resource() -> None:
    resources = [
        _condition(),
        _observation(),
        {
            "resourceType": "MedicationStatement",
            "status": "active",
            "medicationCodeableConcept": {"text": "synthetic medication"},
            "subject": {"reference": "Patient/synthetic"},
        },
        {
            "resourceType": "Procedure",
            "status": "completed",
            "subject": {"reference": "Patient/synthetic"},
        },
        {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"text": "synthetic report"},
        },
        {
            "resourceType": "AllergyIntolerance",
            "patient": {"reference": "Patient/synthetic"},
        },
        {
            "resourceType": "Immunization",
            "status": "completed",
            "vaccineCode": {"text": "synthetic vaccine"},
            "patient": {"reference": "Patient/synthetic"},
            "occurrenceDateTime": "2026-01-02T03:04:05Z",
        },
        {
            "resourceType": "Encounter",
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
            },
        },
    ]

    result = validate_bundle(
        {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": resource} for resource in resources],
        }
    )

    assert result.is_valid
    assert result.findings == ()


def test_unsupported_and_malformed_resources_never_raise() -> None:
    unsupported = validate_resource({"resourceType": "Patient"})
    malformed_values = [None, [], "resource", {}, {"resourceType": 7}]

    assert unsupported.is_valid
    assert unsupported.warnings[0].code == "not-supported"
    for malformed in malformed_values:
        result = validate_resource(malformed)  # type: ignore[arg-type]
        assert result.errors

    malformed_bundle = validate_bundle(  # type: ignore[arg-type]
        {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": [{}, {"resource": []}],
        }
    )
    assert len(malformed_bundle.errors) == 2


def test_validation_does_not_open_network_connections(monkeypatch: Any) -> None:
    def fail_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("base R4 validator attempted network access")

    monkeypatch.setattr(socket, "socket", fail_network)

    assert validate_resource(_observation()).is_valid


def test_result_adapts_to_operation_outcome() -> None:
    observation = _observation()
    observation.pop("status")

    outcome = from_validation_result(validate_resource(observation))

    assert outcome["issue"] == [
        {
            "severity": "error",
            "code": "required",
            "diagnostics": "Required element is missing or empty.",
            "expression": ["Observation.status"],
        }
    ]


def test_versioned_interop_validator_reuses_base_r4_findings() -> None:
    observation = _observation()
    observation.pop("status")

    result = interop_validation_result(observation, "R4")

    required_status = [
        issue
        for issue in result.as_dict()["issues"]
        if issue["code"] == "required" and issue["path"] == "Observation.status"
    ]
    assert required_status == [
        {
            "severity": "error",
            "code": "required",
            "path": "Observation.status",
            "diagnostics": "Required element is missing or empty.",
        }
    ]


def test_bundled_constraints_are_the_scoped_permissive_r4_subset() -> None:
    path = resources.files("openmed.clinical.exporters.fhir").joinpath(
        "definitions", "r4_base.json"
    )
    definitions = json.loads(path.read_text(encoding="utf-8"))

    assert definitions["license"] == "CC0-1.0"
    assert definitions["fhirVersion"] == "4.0.1"
    assert set(definitions["resources"]) == BASE_R4_RESOURCE_TYPES
    serialized = json.dumps(definitions).casefold()
    assert not any(
        restricted in serialized
        for restricted in ("cpt", "mimic", "snomed", "umls", "n2c2", "i2b2")
    )
