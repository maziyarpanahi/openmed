"""Focused tests for measurement and procedure OMOP table exporters."""

from __future__ import annotations

from openmed.clinical.exporters import (
    to_measurement,
    to_omop,
    to_procedure_occurrence,
)
from openmed.clinical.exporters.omop import (
    MEASUREMENT_COLUMNS,
    PROCEDURE_OCCURRENCE_COLUMNS,
)
from openmed.clinical.grounding import Candidate, GroundedSpan


def test_omop_v54_column_sets_are_exact() -> None:
    assert MEASUREMENT_COLUMNS == (
        "measurement_id",
        "person_id",
        "measurement_concept_id",
        "measurement_date",
        "measurement_datetime",
        "measurement_time",
        "measurement_type_concept_id",
        "operator_concept_id",
        "value_as_number",
        "value_as_concept_id",
        "unit_concept_id",
        "range_low",
        "range_high",
        "provider_id",
        "visit_occurrence_id",
        "visit_detail_id",
        "measurement_source_value",
        "measurement_source_concept_id",
        "unit_source_value",
        "unit_source_concept_id",
        "value_source_value",
        "measurement_event_id",
        "meas_event_field_concept_id",
    )
    assert PROCEDURE_OCCURRENCE_COLUMNS == (
        "procedure_occurrence_id",
        "person_id",
        "procedure_concept_id",
        "procedure_date",
        "procedure_datetime",
        "procedure_end_date",
        "procedure_end_datetime",
        "procedure_type_concept_id",
        "modifier_concept_id",
        "quantity",
        "provider_id",
        "visit_occurrence_id",
        "visit_detail_id",
        "procedure_source_value",
        "procedure_source_concept_id",
        "modifier_source_value",
    )


def _span(
    text: str,
    label: str,
    system: str,
    code: str,
    *,
    metadata: dict[str, object] | None = None,
) -> GroundedSpan:
    return GroundedSpan(
        text=text,
        start=0,
        end=len(text),
        canonical_label=label,
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


def test_measurement_has_exact_cdm_columns_and_lab_values() -> None:
    span = _span(
        "Synthetic glucose",
        "LAB_TEST",
        "LOINC",
        "SYNTH-GLUCOSE",
        metadata={
            "value_as_number": "7.4",
            "unit": "mmol/L",
            "unit_concept_id": 8753,
            "unit_source_concept_id": 9448,
            "reference_range": {"low": 3.9, "high": 7.8},
            "measurement_type_concept_id": 32817,
        },
    )

    rows = to_omop(
        span,
        table="measurement",
        person_id=7,
        visit_id=8,
        note_date="2026-09-07",
        concept_resolver={
            ("LOINC", "SYNTH-GLUCOSE"): {
                "target_concept_id": 3001,
                "source_concept_id": 9001,
            }
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == MEASUREMENT_COLUMNS
    assert row["measurement_concept_id"] == 3001
    assert row["measurement_source_value"] == "Synthetic glucose"
    assert row["measurement_source_concept_id"] == 9001
    assert row["measurement_date"] == "2026-09-07"
    assert row["value_as_number"] == 7.4
    assert row["value_source_value"] == "7.4"
    assert row["unit_concept_id"] == 8753
    assert row["unit_source_value"] == "mmol/L"
    assert row["unit_source_concept_id"] == 9448
    assert row["range_low"] == 3.9
    assert row["range_high"] == 7.8


def test_unmapped_measurement_preserves_source_and_unmapped_unit() -> None:
    row = to_measurement(
        _span(
            "Synthetic assay",
            "LAB_TEST",
            "LOINC",
            "SYNTH-ASSAY",
            metadata={"unit": "arbitrary-unit"},
        )
    )[0]

    assert row["measurement_concept_id"] == 0
    assert row["measurement_source_value"] == "Synthetic assay"
    assert row["measurement_source_concept_id"] == 0
    assert row["unit_concept_id"] == 0
    assert row["unit_source_concept_id"] == 0
    assert row["unit_source_value"] == "arbitrary-unit"


def test_procedure_occurrence_has_exact_cdm_columns_and_mapping() -> None:
    span = _span(
        "Synthetic imaging procedure",
        "PROCEDURE",
        "SNOMED",
        "SYNTH-PROCEDURE",
        metadata={
            "procedure_type_concept_id": 32817,
            "procedure_end_date": "2026-09-08",
            "quantity": 1,
        },
    )

    rows = to_procedure_occurrence(
        span,
        person_id="synthetic-person",
        visit_occurrence_id="synthetic-visit",
        procedure_date="2026-09-07",
        resolver={
            ("SNOMED", "SYNTH-PROCEDURE"): {
                "standard_concept_id": 4001,
                "source_concept_id": 9002,
            }
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == PROCEDURE_OCCURRENCE_COLUMNS
    assert row["procedure_concept_id"] == 4001
    assert row["procedure_source_value"] == "Synthetic imaging procedure"
    assert row["procedure_source_concept_id"] == 9002
    assert row["procedure_date"] == "2026-09-07"
    assert row["procedure_end_date"] == "2026-09-08"
    assert row["quantity"] == 1
    assert row["person_id"] > 0
    assert row["visit_occurrence_id"] > 0


def test_unmapped_procedure_routes_through_public_to_omop() -> None:
    row = to_omop(
        _span(
            "Unmapped procedure",
            "PROCEDURE",
            "LOCAL",
            "SYNTH-UNMAPPED",
        ),
        table="procedure_occurrence",
    )[0]

    assert row["procedure_concept_id"] == 0
    assert row["procedure_source_value"] == "Unmapped procedure"
    assert row["procedure_source_concept_id"] == 0
