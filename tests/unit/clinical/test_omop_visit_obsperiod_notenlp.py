"""Focused tests for supporting OMOP CDM table exporters."""

from __future__ import annotations

from openmed.clinical.context import (
    CERTAIN,
    FAMILY_EXPERIENCER,
    HISTORICAL,
    NEGATED,
    ClinicalAssertion,
)
from openmed.clinical.exporters import (
    to_note_nlp,
    to_observation_period,
    to_omop,
    to_visit_occurrence,
)
from openmed.clinical.exporters.omop import (
    NOTE_NLP_COLUMNS,
    OBSERVATION_PERIOD_COLUMNS,
    VISIT_OCCURRENCE_COLUMNS,
)
from openmed.clinical.grounding import Candidate, GroundedSpan


def _span(
    *,
    start: int = 12,
    assertion: ClinicalAssertion | None = None,
    metadata: dict[str, object] | None = None,
) -> GroundedSpan:
    text = "Synthetic condition"
    return GroundedSpan(
        text=text,
        start=start,
        end=start + len(text),
        canonical_label="CONDITION",
        assertion=assertion,
        candidates=(
            Candidate(
                system="ICD10CM",
                code="SYNTH-CODE",
                display=text,
                score=1.0,
                source="synthetic",
            ),
        ),
        metadata=metadata or {},
    )


def test_supporting_table_column_sets_match_cdm_v54() -> None:
    assert OBSERVATION_PERIOD_COLUMNS == (
        "observation_period_id",
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
        "period_type_concept_id",
    )
    assert VISIT_OCCURRENCE_COLUMNS == (
        "visit_occurrence_id",
        "person_id",
        "visit_concept_id",
        "visit_start_date",
        "visit_start_datetime",
        "visit_end_date",
        "visit_end_datetime",
        "visit_type_concept_id",
        "provider_id",
        "care_site_id",
        "visit_source_value",
        "visit_source_concept_id",
        "admitted_from_concept_id",
        "admitted_from_source_value",
        "discharged_to_concept_id",
        "discharged_to_source_value",
        "preceding_visit_occurrence_id",
    )
    assert NOTE_NLP_COLUMNS == (
        "note_nlp_id",
        "note_id",
        "section_concept_id",
        "snippet",
        "offset",
        "lexical_variant",
        "note_nlp_concept_id",
        "note_nlp_source_concept_id",
        "nlp_system",
        "nlp_date",
        "nlp_datetime",
        "term_exists",
        "term_temporal",
        "term_modifiers",
    )


def test_visit_occurrence_is_document_scoped_and_exact() -> None:
    row = to_omop(
        _span(metadata={"visit_concept_id": 9202}),
        table="visit_occurrence",
        document_id="synthetic-document",
        person_id="synthetic-person",
        visit_id="synthetic-encounter",
        note_date="2026-09-07",
    )[0]

    assert tuple(row) == VISIT_OCCURRENCE_COLUMNS
    assert row["person_id"] > 0
    assert row["visit_occurrence_id"] > 0
    assert row["visit_concept_id"] == 9202
    assert row["visit_start_date"] == "2026-09-07"
    assert row["visit_end_date"] == "2026-09-07"
    occurrence = to_omop(
        _span(metadata={"condition_start_date": "2026-09-07"}),
        table="condition_occurrence",
        person_id="synthetic-person",
        visit_id="synthetic-encounter",
    )[0]
    assert row["visit_occurrence_id"] == occurrence["visit_occurrence_id"]
    assert (
        row
        == to_visit_occurrence(
            _span(metadata={"visit_concept_id": 9202}),
            document_id="synthetic-document",
            person_id="synthetic-person",
            visit_id="synthetic-encounter",
            note_date="2026-09-07",
        )[0]
    )


def test_observation_period_bounds_available_clinical_dates() -> None:
    spans = (
        _span(start=0, metadata={"condition_start_date": "2026-01-03"}),
        _span(start=30, metadata={"condition_end_date": "2026-08-19"}),
    )

    row = to_omop(
        spans,
        table="observation_period",
        document_id="synthetic-document",
        person_id=7,
    )[0]

    assert tuple(row) == OBSERVATION_PERIOD_COLUMNS
    assert row["person_id"] == 7
    assert row["observation_period_start_date"] == "2026-01-03"
    assert row["observation_period_end_date"] == "2026-08-19"
    assert row["period_type_concept_id"] == 0


def test_observation_period_is_empty_without_dates() -> None:
    assert to_observation_period(_span(), person_id=7) == ()


def test_note_nlp_carries_mapping_offsets_and_assertion_context() -> None:
    assertion = ClinicalAssertion(
        temporality=HISTORICAL,
        certainty=CERTAIN,
        negation=NEGATED,
        experiencer=FAMILY_EXPERIENCER,
    )

    row = to_note_nlp(
        _span(assertion=assertion),
        note_id="synthetic-note",
        nlp_date="2026-09-07",
        concept_resolver={
            ("ICD10CM", "SYNTH-CODE"): {
                "target_concept_id": 1001,
                "source_concept_id": 9001,
            }
        },
    )[0]

    assert tuple(row) == NOTE_NLP_COLUMNS
    assert row["note_id"] > 0
    assert row["offset"] == 12
    assert row["lexical_variant"] == "Synthetic condition"
    assert row["note_nlp_concept_id"] == 1001
    assert row["note_nlp_source_concept_id"] == 9001
    assert row["nlp_system"] == "openmed"
    assert row["nlp_date"] == "2026-09-07"
    assert row["term_exists"] == "N"
    assert row["term_temporal"] == "historical"
    assert row["term_modifiers"] == (
        '{"certainty":"certain","experiencer":"family",'
        '"negation":"negated","temporality":"historical"}'
    )


def test_note_nlp_unasserted_unmapped_span_is_retained() -> None:
    row = to_omop(
        _span(),
        table="note_nlp",
        document_id="synthetic-document",
        note_date="2026-09-07",
    )[0]

    assert row["note_nlp_concept_id"] == 0
    assert row["note_nlp_source_concept_id"] == 0
    assert row["term_exists"] == "Y"
    assert row["term_temporal"] is None
    assert row["term_modifiers"] is None
