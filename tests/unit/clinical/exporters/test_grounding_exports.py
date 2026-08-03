"""Focused deterministic FHIR R4 and OMOP export tests."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import CoreferenceChain
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
    COREFERENCE_EVIDENCE_EXTENSION_URL,
    achilles_smoke_check,
    to_fhir,
    to_omop,
)
from openmed.clinical.grounding import Candidate, GroundedSpan
from openmed.core.schemas import OpenMedSpan, hmac_text_hash

_COREFERENCE_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "clinical"
    / "medication_coreference_collapse.json"
)
_SYNTHETIC_HASH_SECRET = "synthetic-medication-coreference-secret"


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


def _coreference_chain(case: dict) -> CoreferenceChain:
    members = tuple(
        OpenMedSpan(
            doc_id=case["document_id"],
            start=mention["start"],
            end=mention["end"],
            text_hash=hmac_text_hash(
                case["text"][mention["start"] : mention["end"]],
                _SYNTHETIC_HASH_SECRET,
            ),
            entity_type="DRUG",
            canonical_label="MEDICATION",
        )
        for mention in case["mentions"]
    )
    return CoreferenceChain(
        chain_id="coref-synthetic-medication",
        members=members,
        representative=members[case["representative_index"]],
        confidence=0.98,
    )


def _fhir_source_evidence(extension: dict) -> tuple[int, int, str]:
    values = {
        item["url"]: item.get("valueUnsignedInt", item.get("valueString"))
        for item in extension["extension"]
    }
    return values["start"], values["end"], values["textHash"]


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _string_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _string_values(child)]
    return []


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


def test_to_fhir_collapses_five_coreferent_medication_surfaces() -> None:
    case = json.loads(_COREFERENCE_FIXTURE.read_text(encoding="utf-8"))
    chain = _coreference_chain(case)
    grounding = case["grounding"]
    candidate = Candidate(
        system=grounding["system"],
        code=grounding["code"],
        display=grounding["display"],
        score=0.99,
        source="synthetic",
        matched_alias=grounding["display"],
        match_kind="exact",
        vocab_version="synthetic-v1",
    )
    spans = tuple(
        GroundedSpan(
            text=mention["surface"],
            start=mention["start"],
            end=mention["end"],
            candidates=(candidate,),
            canonical_label="MEDICATION",
        )
        for mention in case["mentions"]
    )

    bundle = to_fhir(
        spans,
        document_id=case["document_id"],
        coreference_chains=(chain,),
    )

    assert bundle is not None
    assert len(bundle["entry"]) == 1
    statement = bundle["entry"][0]["resource"]
    assert statement["resourceType"] == case["expected_resource_type"]
    evidence = next(
        extension
        for extension in statement["extension"]
        if extension["url"] == COREFERENCE_EVIDENCE_EXTENSION_URL
    )
    supporting = [
        item for item in evidence["extension"] if item["url"] == "supportingMention"
    ]
    assert len(supporting) == len(case["mentions"]) == 5
    assert [_fhir_source_evidence(item) for item in supporting] == [
        (member.start, member.end, member.text_hash) for member in chain.members
    ]

    evidence_values = {value.casefold() for value in _string_values(evidence)}
    for mention in case["mentions"]:
        assert mention["surface"].casefold() not in evidence_values
    assert (
        to_fhir(
            reversed(spans),
            document_id=case["document_id"],
            coreference_chains=(chain,),
        )
        == bundle
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
