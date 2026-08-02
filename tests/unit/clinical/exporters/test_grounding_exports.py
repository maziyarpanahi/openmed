"""Focused deterministic FHIR R4 and OMOP export tests."""

from __future__ import annotations

from openmed.clinical.context import (
    CERTAIN,
    FAMILY_EXPERIENCER,
    NEGATED,
    PATIENT_EXPERIENCER,
    RECENT,
    ClinicalAssertion,
)
from openmed.clinical.exporters import (
    CORE_OMOP_TABLES,
    achilles_smoke_check,
    to_fhir,
    to_omop,
)
from openmed.clinical.grounding import Candidate, GroundedSpan


def _span(
    text: str,
    start: int,
    label: str,
    system: str,
    code: str,
    *,
    assertion: ClinicalAssertion | None = None,
    value: float | None = None,
    unit: str | None = None,
) -> GroundedSpan:
    metadata = {}
    if value is not None:
        metadata["value"] = value
    if unit is not None:
        metadata["unit"] = unit
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
                score=0.91,
                source="sparse",
                matched_alias=text,
                match_kind="exact",
                vocab_version="synthetic-v1",
            ),
        ),
        metadata=metadata,
    )


def _spans() -> tuple[GroundedSpan, ...]:
    return (
        _span("Aster syndrome", 0, "CONDITION", "ICD10CM", "C-1"),
        _span("Novo tablet", 16, "MEDICATION", "RXNORM", "D-1"),
        _span(
            "Elin panel",
            29,
            "LAB_TEST",
            "LOINC",
            "M-1",
            value=7.2,
            unit="mg/dL",
        ),
        _span("Juno procedure", 51, "PROCEDURE", "SNOMED", "P-1"),
    )


def test_to_fhir_emits_valid_core_r4_resource_shapes() -> None:
    bundle = to_fhir(
        _spans(),
        document_id="synthetic-doc",
        subject_reference="Patient/synthetic",
    )

    assert bundle is not None
    resources = [entry["resource"] for entry in bundle["entry"]]
    assert [resource["resourceType"] for resource in resources] == [
        "Condition",
        "MedicationStatement",
        "Observation",
        "Procedure",
    ]
    assert resources[2]["valueQuantity"] == {
        "value": 7.2,
        "unit": "mg/dL",
        "system": "http://unitsofmeasure.org",
        "code": "mg/dL",
    }
    serialized = str(bundle)
    assert "'_score'" not in serialized
    assert "openmed.ai/fhir/StructureDefinition" not in serialized
    assert resources[0]["code"]["coding"][0]["version"] == "synthetic-v1"
    assert to_fhir(_spans(), document_id="synthetic-doc") == to_fhir(
        _spans(), document_id="synthetic-doc"
    )


def test_negated_condition_is_refuted() -> None:
    span = _span(
        "Aster syndrome",
        0,
        "CONDITION",
        "ICD10CM",
        "C-1",
        assertion=ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            negation=NEGATED,
            experiencer=PATIENT_EXPERIENCER,
        ),
    )

    condition = to_fhir(span, subject_reference="Patient/synthetic")

    assert condition is not None
    assert condition["verificationStatus"]["coding"][0]["code"] == "refuted"
    assert "clinicalStatus" not in condition


def test_nonpatient_span_is_excluded_before_resource_inference() -> None:
    span = GroundedSpan(
        text="Aster syndrome",
        start=0,
        end=14,
        assertion=ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            experiencer=FAMILY_EXPERIENCER,
        ),
    )

    assert to_fhir(span, subject_reference="Patient/synthetic") is None


def test_to_omop_emits_four_core_tables_and_passes_smoke() -> None:
    spans = _spans()
    document_text = "Aster syndrome. Novo tablet. Elin panel 7.2 mg/dL. Juno procedure."
    concept_ids = {
        (span.candidates[0].system, span.candidates[0].code): 910000 + index
        for index, span in enumerate(spans, start=1)
    }

    tables = to_omop(
        spans,
        document_text=document_text,
        document_id="synthetic-doc",
        person_id="synthetic-person",
        concept_resolver=concept_ids,
    )

    assert set(CORE_OMOP_TABLES) <= set(tables.tables)
    assert len(tables.table("condition_occurrence")) == 1
    assert len(tables.table("drug_exposure")) == 1
    assert len(tables.table("measurement")) == 1
    assert len(tables.table("procedure_occurrence")) == 1
    assert tables.table("condition_occurrence")[0]["condition_concept_id"] == 910001
    assert tables.table("drug_exposure")[0]["drug_source_value"] == "Novo tablet"
    assert achilles_smoke_check(tables) == ()


def test_to_omop_preserves_unmapped_source_and_excludes_refuted() -> None:
    negated = _span(
        "Refuted syndrome",
        16,
        "CONDITION",
        "ICD10CM",
        "C-2",
        assertion=ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            negation=NEGATED,
            experiencer=PATIENT_EXPERIENCER,
        ),
    )
    present = _span("Aster syndrome", 0, "CONDITION", "ICD10CM", "C-1")

    tables = to_omop(
        (present, negated),
        document_text="Aster syndrome. Refuted syndrome.",
    )

    rows = tables.table("condition_occurrence")
    assert len(rows) == 1
    assert rows[0]["condition_concept_id"] == 0
    assert rows[0]["condition_source_value"] == "Aster syndrome"
