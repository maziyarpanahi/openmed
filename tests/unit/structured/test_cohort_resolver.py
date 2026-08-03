"""Acceptance tests for local DuckDB/Parquet cohort resolution."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.interop.duckdb_udf import cohort_resolve
from openmed.interop.omop import (
    deterministic_omop_id,
    load_grounded_jsonl,
    write_omop_duckdb,
    write_omop_parquet,
)
from openmed.structured.cohort import (
    COHORT_ADVISORY,
    ConceptSet,
    Criterion,
    Expression,
    PhenotypeDefinition,
    TemporalWindow,
    load_athena_hierarchy,
    resolve_phenotype,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cohort"
GROUNDING = FIXTURES / "synthetic_grounded.jsonl"
ATHENA = FIXTURES / "athena"
PHENOTYPES = FIXTURES / "phenotypes"


def _patient_id(source_value: str) -> int:
    return deterministic_omop_id("person", source_value)


def _tables():
    return load_grounded_jsonl(GROUNDING)


def test_fixture_coverage_descendants_negation_and_phi_free_provenance(
    caplog,
) -> None:
    tables = _tables()
    connection = write_omop_duckdb(tables)
    definition = PhenotypeDefinition.load(PHENOTYPES / "diabetes_on_metformin.json")
    hierarchy = load_athena_hierarchy(ATHENA)
    try:
        result = resolve_phenotype(definition, connection, hierarchy=hierarchy)
    finally:
        connection.close()

    expected = {
        _patient_id("raw-person-alpha"),
        _patient_id("raw-person-epsilon"),
    }
    predicted = result.patient_id_set
    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted)
    recall = true_positives / len(expected)

    assert predicted == expected
    assert precision == recall == 1.0
    assert _patient_id("raw-person-beta") not in predicted
    assert _patient_id("raw-person-eta") not in predicted
    assert result.provenance.matched_patient_count == 2
    assert result.to_dict()["advisory"] == COHORT_ADVISORY
    diabetes_provenance = result.provenance.concept_sets[0]
    assert diabetes_provenance.expanded_concept_ids == (201826, 443238)
    assert {item.concept_id for item in diabetes_provenance.matched_members} == {
        201826,
        443238,
    }

    beta_id = _patient_id("raw-person-beta")
    beta_conditions = [
        row
        for row in tables.table("condition_occurrence")
        if row["person_id"] == beta_id
    ]
    note_nlp = {row["note_nlp_id"]: row for row in tables.table("note_nlp")}
    assert note_nlp[beta_conditions[0]["note_nlp_id"]]["term_exists"] == "N"

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    for raw_marker in (
        "raw-person-",
        "synthetic-alpha",
        "Synthetic diabetes marker",
        "Synthetic metformin marker",
    ):
        assert raw_marker not in serialized
    assert caplog.records == []
    assert set(result.evidence) == expected
    assert all(
        pointer.source_note_hash
        for rows in result.evidence.values()
        for pointer in rows
    )


def test_round_trip_definition_resolves_identically() -> None:
    original = PhenotypeDefinition.load(PHENOTYPES / "diabetes_on_metformin.json")
    reloaded = PhenotypeDefinition.from_json(original.to_json_bytes())
    hierarchy = load_athena_hierarchy(ATHENA)
    connection = write_omop_duckdb(_tables())
    try:
        first = resolve_phenotype(original, connection, hierarchy=hierarchy)
        second = resolve_phenotype(reloaded, connection, hierarchy=hierarchy)
    finally:
        connection.close()

    assert original.to_json_bytes() == reloaded.to_json_bytes()
    assert first.patient_ids == second.patient_ids
    assert first.to_dict() == second.to_dict()


def test_occurrence_and_not_expressions_use_the_patient_universe() -> None:
    hierarchy = load_athena_hierarchy(ATHENA)
    connection = write_omop_duckdb(_tables())
    try:
        recurrent = resolve_phenotype(
            PhenotypeDefinition.load(PHENOTYPES / "recurrent_diabetes.json"),
            connection,
            hierarchy=hierarchy,
        )
        without_metformin = resolve_phenotype(
            PhenotypeDefinition.load(PHENOTYPES / "diabetes_without_metformin.json"),
            connection,
            hierarchy=hierarchy,
        )
    finally:
        connection.close()

    expected = (_patient_id("raw-person-zeta"),)
    assert recurrent.patient_ids == expected
    assert without_metformin.patient_ids == expected


def test_absolute_window_and_parquet_source_match_duckdb(tmp_path: Path) -> None:
    definition = PhenotypeDefinition(
        id="recent-metformin",
        name="Recent metformin",
        concept_sets=(ConceptSet("metformin", "OMOP", (1503297,)),),
        expression=Expression.leaf(
            Criterion(
                id="recent-metformin",
                concept_set="metformin",
                temporal=TemporalWindow(start_date="2026-02-01"),
            )
        ),
    )
    tables = _tables()
    parquet_directory = write_omop_parquet(tables, tmp_path / "omop-parquet")
    connection = write_omop_duckdb(tables)
    try:
        through_adapter = cohort_resolve(connection, definition.to_dict())
    finally:
        connection.close()
    through_parquet = resolve_phenotype(
        definition,
        parquet_directory=parquet_directory,
    )

    expected = (_patient_id("raw-person-epsilon"),)
    assert through_adapter.patient_ids == expected
    assert through_parquet.patient_ids == expected
