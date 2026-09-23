"""Focused offline tests for allergy, immunization, and encounter export."""

from __future__ import annotations

from typing import Any

import pytest

from openmed.clinical.context import (
    CERTAIN,
    NEGATED,
    PATIENT_EXPERIENCER,
    RECENT,
    ClinicalAssertion,
)
from openmed.clinical.exporters.fhir import (
    FHIR_R4_REFERENCE_TARGETS,
    to_fhir,
    validate_resource,
)
from openmed.clinical.grounding import Candidate, GroundedSpan


def _span(
    label: str | None,
    system: str,
    code: str,
    *,
    start: int = 0,
    assertion: ClinicalAssertion | None = None,
    metadata: dict[str, Any] | None = None,
) -> GroundedSpan:
    text = f"synthetic-{code}"
    return GroundedSpan(
        text=text,
        start=start,
        end=start + len(text),
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
        metadata=metadata or {},
    )


def _coding(resource: dict[str, Any], field: str) -> dict[str, Any]:
    return resource[field]["coding"][0]


def test_allergy_uses_rxnorm_and_maps_refuted_assertion() -> None:
    allergy = to_fhir(
        _span(
            "ALLERGEN",
            "RXNORM",
            "12345",
            assertion=ClinicalAssertion(
                temporality=RECENT,
                certainty=CERTAIN,
                negation=NEGATED,
                experiencer=PATIENT_EXPERIENCER,
            ),
        ),
        resource="AllergyIntolerance",
        subject_reference="Patient/synthetic",
    )

    assert allergy is not None
    assert allergy["resourceType"] == "AllergyIntolerance"
    assert _coding(allergy, "code")["system"] == (
        "http://www.nlm.nih.gov/research/umls/rxnorm"
    )
    assert _coding(allergy, "code")["code"] == "12345"
    assert _coding(allergy, "verificationStatus")["code"] == "refuted"
    assert "clinicalStatus" not in allergy
    assert "criticality" not in allergy
    assert validate_resource(allergy).is_valid


def test_allergy_includes_only_extracted_criticality() -> None:
    allergy = to_fhir(
        _span(
            "ALLERGEN",
            "SNOMED",
            "91936005",
            metadata={"criticality": "high"},
        ),
        resource="AllergyIntolerance",
    )

    assert allergy is not None
    assert allergy["criticality"] == "high"
    assert validate_resource(allergy).is_valid

    with pytest.raises(ValueError, match="criticality must be one of"):
        to_fhir(
            _span(
                "ALLERGEN",
                "SNOMED",
                "91936005",
                metadata={"criticality": "severe"},
            ),
            resource="AllergyIntolerance",
        )


def test_immunization_uses_cvx_and_assertion_status() -> None:
    immunization = to_fhir(
        _span(
            "VACCINE_NAME",
            "CVX",
            "207",
            metadata={
                "occurrence_datetime": "2026-01-02T03:04:05Z",
                "lot_number": "synthetic-lot",
            },
        ),
        resource="Immunization",
        subject_reference="Patient/synthetic",
    )

    assert immunization is not None
    assert immunization["resourceType"] == "Immunization"
    assert immunization["status"] == "completed"
    assert _coding(immunization, "vaccineCode")["system"] == (
        "http://hl7.org/fhir/sid/cvx"
    )
    assert immunization["occurrenceDateTime"] == "2026-01-02T03:04:05Z"
    assert immunization["lotNumber"] == "synthetic-lot"
    assert validate_resource(immunization).is_valid

    refuted = to_fhir(
        _span(
            "VACCINE_NAME",
            "SNOMED",
            "787859002",
            assertion=ClinicalAssertion(
                temporality=RECENT,
                certainty=CERTAIN,
                negation=NEGATED,
                experiencer=PATIENT_EXPERIENCER,
            ),
        ),
        resource="Immunization",
    )
    assert refuted is not None
    assert refuted["status"] == "not-done"
    assert refuted["occurrenceString"] == "unknown"
    assert validate_resource(refuted).is_valid


def test_encounter_exports_class_type_period_and_stable_id() -> None:
    span = _span(
        None,
        "SNOMED",
        "185349003",
        metadata={
            "encounter_class": "AMB",
            "period": {
                "start": "2026-01-02T03:04:05Z",
                "end": "2026-01-02T03:34:05Z",
            },
        },
    )

    first = to_fhir(span, resource="Encounter", document_id="synthetic-doc")
    second = to_fhir(span, resource="Encounter", document_id="synthetic-doc")

    assert first is not None
    assert first == second
    assert first["resourceType"] == "Encounter"
    assert first["status"] == "in-progress"
    assert first["class"] == {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
        "code": "AMB",
    }
    assert first["type"][0]["coding"][0]["code"] == "185349003"
    assert first["period"] == span.metadata["period"]
    assert first["id"].startswith("openmed-")
    assert validate_resource(first).is_valid


def test_facade_routes_allergy_and_vaccine_labels_and_keeps_references() -> None:
    spans = (
        _span(
            "ALLERGEN",
            "RXNORM",
            "12345",
            metadata={"encounter_reference": "Encounter/synthetic-visit"},
        ),
        _span(
            "VACCINE_NAME",
            "CVX",
            "207",
            start=20,
            metadata={"encounter_reference": "Encounter/synthetic-visit"},
        ),
    )

    bundle = to_fhir(spans, doc_id="synthetic-new-resources")

    assert [entry["resource"]["resourceType"] for entry in bundle["entry"]] == [
        "AllergyIntolerance",
        "Immunization",
    ]
    assert bundle.summary.exported_by_label == {
        "ALLERGEN": 1,
        "VACCINE_NAME": 1,
    }
    assert all(
        entry["resource"]["encounter"] == {"reference": "Encounter/synthetic-visit"}
        for entry in bundle["entry"]
    )
    assert FHIR_R4_REFERENCE_TARGETS["AllergyIntolerance"]["patient"] == {"Patient"}
    assert FHIR_R4_REFERENCE_TARGETS["Immunization"]["patient"] == {"Patient"}


def test_encounter_requires_an_explicit_class() -> None:
    with pytest.raises(ValueError, match="requires encounter_class metadata"):
        to_fhir(_span(None, "SNOMED", "185349003"), resource="Encounter")
