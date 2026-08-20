"""Focused tests for the condition and drug OMOP table exporters."""

from __future__ import annotations

from datetime import date

from openmed.clinical.context import (
    CERTAIN,
    NEGATED,
    PATIENT_EXPERIENCER,
    RECENT,
    ClinicalAssertion,
)
from openmed.clinical.exporters import (
    to_condition_occurrence,
    to_drug_exposure,
    to_omop,
)
from openmed.clinical.exporters.omop.condition_occurrence import (
    CONDITION_OCCURRENCE_COLUMNS,
)
from openmed.clinical.exporters.omop.drug_exposure import DRUG_EXPOSURE_COLUMNS
from openmed.clinical.grounding import Candidate, GroundedSpan
from openmed.interop.omop import VocabularyRouter


def _span(
    text: str,
    label: str,
    system: str,
    code: str,
    *,
    assertion: ClinicalAssertion | None = None,
    metadata: dict[str, object] | None = None,
) -> GroundedSpan:
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
                score=1.0,
                source="synthetic",
            ),
        ),
        metadata=metadata or {},
    )


def test_condition_occurrence_has_exact_cdm_columns_and_athena_mapping() -> None:
    span = _span(
        "Aster syndrome",
        "CONDITION",
        "ICD10CM",
        "C-1",
        metadata={"condition_type_concept_id": 32817},
    )

    rows = to_condition_occurrence(
        span,
        concept_resolver={
            ("ICD10CM", "C-1"): {
                "target_concept_id": 1001,
                "source_concept_id": 9001,
            }
        },
        person_id=7,
        visit_occurrence_id=8,
        condition_start_date=date(2026, 8, 4),
    )

    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == CONDITION_OCCURRENCE_COLUMNS
    assert row["condition_concept_id"] == 1001
    assert row["condition_source_value"] == "Aster syndrome"
    assert row["condition_source_concept_id"] == 9001
    assert row["condition_start_date"] == "2026-08-04"
    assert row["condition_type_concept_id"] == 32817
    assert row["person_id"] == 7
    assert row["visit_occurrence_id"] == 8


def test_unmapped_condition_preserves_source_value_and_uses_zero_concept() -> None:
    row = to_omop(
        _span("Unknown syndrome", "CONDITION", "ICD10CM", "C-unknown"),
        table="condition_occurrence",
    )[0]

    assert row["condition_concept_id"] == 0
    assert row["condition_source_value"] == "Unknown syndrome"
    assert row["condition_source_concept_id"] == 0
    assert row["person_id"] is None
    assert row["visit_occurrence_id"] is None


def test_condition_accepts_injected_athena_router() -> None:
    router = VocabularyRouter(
        {
            "SYNTHETIC": {
                "C-2": {
                    "concept_id": 1002,
                    "concept_name": "Synthetic condition",
                    "domain_id": "Condition",
                    "vocabulary_id": "SYNTHETIC",
                    "concept_class_id": "Clinical Finding",
                    "standard_concept": "S",
                    "concept_code": "C-2",
                }
            }
        }
    )

    row = to_condition_occurrence(
        _span("Synthetic condition", "CONDITION", "SYNTHETIC", "C-2"),
        resolver=router,
    )[0]

    assert row["condition_concept_id"] == 1002
    assert row["condition_source_concept_id"] == 1002


def test_refuted_condition_is_excluded_by_default() -> None:
    span = _span(
        "Refuted syndrome",
        "CONDITION",
        "ICD10CM",
        "C-refuted",
        assertion=ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            negation=NEGATED,
            experiencer=PATIENT_EXPERIENCER,
        ),
    )

    assert to_condition_occurrence(span) == ()


def test_drug_exposure_emits_mapped_row_with_exact_cdm_columns() -> None:
    span = _span("Novo tablet", "MEDICATION", "RXNORM", "D-1")

    rows = to_omop(
        span,
        table="drug_exposure",
        person_id="synthetic-person",
        visit_id="synthetic-visit",
        note_date="2026-08-04",
        concept_resolver={("RXNORM", "D-1"): 2001},
    )

    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == DRUG_EXPOSURE_COLUMNS
    assert row["drug_concept_id"] == 2001
    assert row["drug_source_value"] == "Novo tablet"
    assert row["drug_source_concept_id"] == 0
    assert row["drug_exposure_start_date"] == "2026-08-04"
    assert row["person_id"] > 0
    assert row["visit_occurrence_id"] > 0


def test_drug_exposure_unmapped_code_keeps_source_text() -> None:
    row = to_drug_exposure(
        _span("Mystery tablet", "MEDICATION", "RXNORM", "D-unknown")
    )[0]

    assert row["drug_concept_id"] == 0
    assert row["drug_source_value"] == "Mystery tablet"
    assert row["drug_source_concept_id"] == 0
