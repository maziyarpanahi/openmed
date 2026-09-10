"""Tests for the dependency-free US Core STU9 conformance subset."""

from __future__ import annotations

import json
import socket
from importlib import resources
from typing import Any

from openmed.clinical.exporters.fhir import (
    US_CORE_VERSION,
    ConformanceResult,
    check_us_core,
    from_validation_result,
)

CONDITION_PROFILE = (
    "http://hl7.org/fhir/us/core/StructureDefinition/"
    "us-core-condition-problems-health-concerns"
)


def _coding(system: str, code: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code}]}


def _condition(**updates: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Condition",
        "meta": {"lastUpdated": "2026-09-07T08:00:00Z"},
        "extension": [
            {
                "url": "http://hl7.org/fhir/StructureDefinition/condition-assertedDate",
                "valueDateTime": "2026-09-01",
            }
        ],
        "clinicalStatus": _coding(
            "http://terminology.hl7.org/CodeSystem/condition-clinical", "active"
        ),
        "verificationStatus": _coding(
            "http://terminology.hl7.org/CodeSystem/condition-ver-status",
            "confirmed",
        ),
        "category": [
            _coding(
                "http://terminology.hl7.org/CodeSystem/condition-category",
                "problem-list-item",
            )
        ],
        "code": {"text": "synthetic condition"},
        "subject": {"reference": "Patient/synthetic"},
        "onsetDateTime": "2026-08-01",
        "abatementDateTime": "2026-08-02",
        "recordedDate": "2026-08-03",
    }
    resource.update(updates)
    return resource


def _observation(**updates: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "meta": {"lastUpdated": "2026-09-07T08:00:00Z"},
        "status": "final",
        "category": [
            _coding(
                "http://terminology.hl7.org/CodeSystem/observation-category",
                "laboratory",
            )
        ],
        "code": {"text": "synthetic laboratory test"},
        "subject": {"reference": "Patient/synthetic"},
        "encounter": {"reference": "Encounter/synthetic"},
        "effectiveDateTime": "2026-09-01",
        "performer": [{"reference": "Practitioner/synthetic"}],
        "valueQuantity": {"value": 7.2, "unit": "synthetic"},
        "interpretation": [{"text": "synthetic interpretation"}],
        "specimen": {"reference": "Specimen/synthetic"},
        "referenceRange": [{"text": "synthetic range"}],
    }
    resource.update(updates)
    return resource


def _medication_request(**updates: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "MedicationRequest",
        "status": "active",
        "intent": "order",
        "category": [
            _coding(
                "http://terminology.hl7.org/CodeSystem/medicationrequest-category",
                "outpatient",
            )
        ],
        "reportedBoolean": False,
        "medicationCodeableConcept": {"text": "synthetic medication"},
        "subject": {"reference": "Patient/synthetic"},
        "encounter": {"reference": "Encounter/synthetic"},
        "authoredOn": "2026-09-01",
        "requester": {"reference": "Practitioner/synthetic"},
        "dosageInstruction": [
            {
                "text": "synthetic dosage",
                "timing": {"code": {"text": "synthetic schedule"}},
                "route": {"text": "synthetic route"},
                "doseAndRate": [{"doseQuantity": {"value": 1}}],
            }
        ],
        "dispenseRequest": {
            "numberOfRepeatsAllowed": 1,
            "quantity": {"value": 30},
        },
    }
    resource.update(updates)
    return resource


def _allergy(**updates: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "AllergyIntolerance",
        "clinicalStatus": _coding(
            "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
            "active",
        ),
        "verificationStatus": _coding(
            "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
            "confirmed",
        ),
        "code": {"text": "synthetic allergy"},
        "patient": {"reference": "Patient/synthetic"},
        "reaction": [
            {
                "manifestation": [{"text": "synthetic manifestation"}],
                "severity": "mild",
            }
        ],
    }
    resource.update(updates)
    return resource


def test_absent_must_support_element_is_a_sanitized_warning() -> None:
    resource = _condition()
    resource.pop("clinicalStatus")

    result = check_us_core(resource)

    assert isinstance(result, ConformanceResult)
    assert result.is_valid
    assert any(
        item.code == "incomplete" and item.location == "Condition.clinicalStatus"
        for item in result.warnings
    )
    assert all("synthetic" not in item.message for item in result.findings)


def test_required_category_binding_violation_is_an_error() -> None:
    result = check_us_core(
        _condition(
            category=[
                _coding(
                    "http://terminology.hl7.org/CodeSystem/condition-category",
                    "encounter-diagnosis",
                )
            ]
        )
    )

    assert ("code-invalid", "Condition.category[us-core]") in {
        (item.code, item.location) for item in result.errors
    }


def test_fully_populated_condition_is_clean() -> None:
    result = check_us_core(_condition())

    assert result.profile == CONDITION_PROFILE
    assert result.resource_type == "Condition"
    assert result.findings == ()


def test_supported_profiles_are_clean_and_run_base_r4_first() -> None:
    resources_to_check = [
        _observation(),
        _medication_request(),
        _allergy(),
    ]

    for resource in resources_to_check:
        result = check_us_core(resource)
        assert result.findings == (), (resource["resourceType"], result.findings)

    invalid_medication = _medication_request(status="synthetic-invalid")
    result = check_us_core(invalid_medication)
    assert ("code-invalid", "MedicationRequest.status") in {
        (item.code, item.location) for item in result.errors
    }


def test_observation_value_or_absent_reason_satisfies_must_support_choice() -> None:
    observation = _observation()
    observation.pop("valueQuantity")
    observation["dataAbsentReason"] = {"text": "synthetic reason"}

    result = check_us_core(observation)

    assert not any(
        "value[x]|dataAbsentReason" in item.location for item in result.warnings
    )


def test_allergy_reaction_requires_manifestation_and_valid_severity() -> None:
    result = check_us_core(_allergy(reaction=[{"severity": "synthetic-invalid"}]))

    assert ("required", "AllergyIntolerance.reaction.manifestation") in {
        (item.code, item.location) for item in result.errors
    }
    assert ("code-invalid", "AllergyIntolerance.reaction[0].severity") in {
        (item.code, item.location) for item in result.errors
    }


def test_profile_resolution_accepts_meta_id_https_and_version() -> None:
    encounter_profile = (
        "https://hl7.org/fhir/us/core/StructureDefinition/"
        f"us-core-condition-encounter-diagnosis|{US_CORE_VERSION}"
    )
    resource = _condition(
        meta={"profile": [encounter_profile]},
        category=[
            _coding(
                "http://terminology.hl7.org/CodeSystem/condition-category",
                "encounter-diagnosis",
            )
        ],
        encounter={"reference": "Encounter/synthetic"},
    )

    result = check_us_core(resource)

    assert result.profile and result.profile.endswith(
        "us-core-condition-encounter-diagnosis"
    )
    assert result.is_valid
    assert check_us_core(_allergy(), "us-core-allergyintolerance").is_valid


def test_unknown_version_and_resource_mismatch_are_structured_errors() -> None:
    wrong_version = check_us_core(_condition(), f"{CONDITION_PROFILE}|8.0.1")
    wrong_resource = check_us_core(
        _observation(), "us-core-condition-problems-health-concerns"
    )

    assert [(item.code, item.location) for item in wrong_version.errors] == [
        ("not-supported", "Condition.meta.profile")
    ]
    assert [(item.code, item.location) for item in wrong_resource.errors] == [
        ("value", "Observation.meta.profile")
    ]


def test_base_and_profile_required_findings_are_deduplicated() -> None:
    resource = _condition()
    resource.pop("subject")

    result = check_us_core(resource)

    assert [item.location for item in result.errors].count("Condition.subject") == 1


def test_result_adapts_to_operation_outcome() -> None:
    resource = _condition()
    resource.pop("clinicalStatus")

    outcome = from_validation_result(check_us_core(resource))

    assert any(
        issue["severity"] == "warning"
        and issue["code"] == "incomplete"
        and issue["expression"] == ["Condition.clinicalStatus"]
        for issue in outcome["issue"]
    )


def test_check_is_offline_and_constraints_are_permissive(monkeypatch: Any) -> None:
    def fail_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("US Core checker attempted network access")

    monkeypatch.setattr(socket, "socket", fail_network)
    assert check_us_core(_condition()).is_valid

    path = resources.files("openmed.clinical.exporters.fhir").joinpath(
        "definitions", "us_core_constraints.json"
    )
    definitions = json.loads(path.read_text(encoding="utf-8"))
    assert definitions["license"] == "CC0-1.0"
    assert definitions["usCoreVersion"] == US_CORE_VERSION
    assert definitions["bindings"]["condition-code"] == {
        "external": True,
        "strength": "extensible",
        "url": "http://hl7.org/fhir/us/core/ValueSet/us-core-condition-code",
    }
    serialized = json.dumps(definitions).casefold()
    assert not any(
        restricted in serialized
        for restricted in ("cpt", "mimic", "snomed", "umls", "n2c2", "i2b2")
    )
