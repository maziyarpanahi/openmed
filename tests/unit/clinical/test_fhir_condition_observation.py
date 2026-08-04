"""Focused offline tests for FHIR R4 Condition and Observation exporters."""

from __future__ import annotations

from typing import Any

from openmed.clinical.context import (
    CERTAIN,
    FAMILY_EXPERIENCER,
    NEGATED,
    PATIENT_EXPERIENCER,
    RECENT,
    ClinicalAssertion,
)
from openmed.clinical.exporters import to_fhir
from openmed.clinical.grounding import Candidate, GroundedSpan

_FHIR_R4_ELEMENT_MODEL = {
    "Condition": {
        "required": {"resourceType", "verificationStatus", "code", "subject"},
        "allowed": {
            "resourceType",
            "id",
            "clinicalStatus",
            "verificationStatus",
            "code",
            "subject",
        },
    },
    "Observation": {
        "required": {"resourceType", "status", "code"},
        "allowed": {
            "resourceType",
            "id",
            "status",
            "code",
            "subject",
            "valueBoolean",
            "valueQuantity",
            "valueString",
        },
    },
}


def _span(
    text: str,
    label: str,
    system: str,
    code: str,
    *,
    assertion: ClinicalAssertion | None = None,
    value: Any = None,
    unit: str | None = None,
) -> GroundedSpan:
    metadata: dict[str, Any] = {}
    if value is not None:
        metadata["value"] = value
    if unit is not None:
        metadata["unit"] = unit
    return GroundedSpan(
        text=text,
        start=0,
        end=len(text),
        canonical_label=label,
        assertion=assertion,
        candidates=(
            Candidate(
                system=system,
                code=code,
                display=text,
                score=0.99,
                source="synthetic",
                matched_alias=text,
                match_kind="exact",
                vocab_version="synthetic-r4-v1",
            ),
        ),
        metadata=metadata,
    )


def _assert_r4_shape(resource: dict[str, Any], resource_type: str) -> None:
    model = _FHIR_R4_ELEMENT_MODEL[resource_type]
    assert resource["resourceType"] == resource_type
    assert model["required"] <= resource.keys()
    assert set(resource) <= model["allowed"]


def _coding(resource: dict[str, Any], element: str) -> dict[str, Any]:
    return resource[element]["coding"][0]


def test_condition_maps_assertion_status_and_codeable_concept() -> None:
    condition = to_fhir(
        _span("synthetic pneumonia", "CONDITION", "ICD10CM", "J18.9"),
        resource="Condition",
        subject_reference="Patient/synthetic",
    )

    assert condition is not None
    _assert_r4_shape(condition, "Condition")
    assert _coding(condition, "clinicalStatus")["code"] == "active"
    assert _coding(condition, "verificationStatus")["code"] == "confirmed"
    assert _coding(condition, "code") == {
        "system": "http://hl7.org/fhir/sid/icd-10-cm",
        "code": "J18.9",
        "display": "synthetic pneumonia",
        "version": "synthetic-r4-v1",
    }


def test_negated_condition_is_refuted_and_family_history_is_excluded() -> None:
    negated = to_fhir(
        _span(
            "synthetic pneumonia",
            "CONDITION",
            "ICD10CM",
            "J18.9",
            assertion=ClinicalAssertion(
                temporality=RECENT,
                certainty=CERTAIN,
                negation=NEGATED,
                experiencer=PATIENT_EXPERIENCER,
            ),
        ),
        resource="Condition",
        subject_reference="Patient/synthetic",
    )
    family_history = to_fhir(
        _span(
            "synthetic colon cancer",
            "CONDITION",
            "ICD10CM",
            "C18.9",
            assertion=ClinicalAssertion(
                temporality=RECENT,
                certainty=CERTAIN,
                experiencer=FAMILY_EXPERIENCER,
            ),
        ),
        resource="Condition",
        subject_reference="Patient/synthetic",
    )

    assert negated is not None
    _assert_r4_shape(negated, "Condition")
    assert _coding(negated, "verificationStatus")["code"] == "refuted"
    assert "clinicalStatus" not in negated
    assert family_history is None


def test_observation_uses_loinc_and_preserves_quantity_value() -> None:
    observation = to_fhir(
        _span(
            "synthetic glucose",
            "LAB_TEST",
            "LOINC",
            "2345-7",
            value=7.2,
            unit="mmol/L",
        ),
        resource="Observation",
        subject_reference="Patient/synthetic",
    )

    assert observation is not None
    _assert_r4_shape(observation, "Observation")
    assert observation["status"] == "final"
    assert _coding(observation, "code")["system"] == "http://loinc.org"
    assert _coding(observation, "code")["code"] == "2345-7"
    assert observation["valueQuantity"] == {
        "value": 7.2,
        "unit": "mmol/L",
        "system": "http://unitsofmeasure.org",
        "code": "mmol/L",
    }


def test_observation_without_value_omits_value_x() -> None:
    observation = to_fhir(
        _span("synthetic hemoglobin", "LAB_TEST", "LOINC", "718-7"),
        resource="Observation",
        subject_reference="Patient/synthetic",
    )

    assert observation is not None
    _assert_r4_shape(observation, "Observation")
    assert observation["status"] == "final"
    assert not any(key.startswith("value") for key in observation)
