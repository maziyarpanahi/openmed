"""Acceptance tests for local DuckDB/Parquet cohort resolution."""

from __future__ import annotations

import json
from dataclasses import replace
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
    CohortResolver,
    ConceptSet,
    Criterion,
    Expression,
    PhenotypeDefinition,
    TemporalWindow,
    load_athena_hierarchy,
    resolve_phenotype,
)

DOMAIN_CONCEPTS = (
    ("condition_occurrence", "condition_concept_id"),
    ("drug_exposure", "drug_concept_id"),
    ("measurement", "measurement_concept_id"),
    ("procedure_occurrence", "procedure_concept_id"),
    ("observation", "observation_concept_id"),
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cohort"
GROUNDING = FIXTURES / "synthetic_grounded.jsonl"
ATHENA = FIXTURES / "athena"
PHENOTYPES = FIXTURES / "phenotypes"


def _patient_id(source_value: str) -> int:
    return deterministic_omop_id("person", source_value)


def _tables():
    return load_grounded_jsonl(GROUNDING)


def test_empty_source_and_unmatched_concepts_are_distinct() -> None:
    connection = write_omop_duckdb(_tables())
    definition = PhenotypeDefinition.load(PHENOTYPES / "diabetes_on_metformin.json")
    try:
        source_count = connection.execute("SELECT COUNT(*) FROM person").fetchone()[0]
        for table, _ in DOMAIN_CONCEPTS:
            connection.execute(f"DELETE FROM {table}")
        unmatched = resolve_phenotype(
            definition, connection, hierarchy=load_athena_hierarchy(ATHENA)
        )
        assert unmatched.patient_ids == ()
        assert unmatched.coverage.source_patient_count == source_count > 0
        assert unmatched.coverage.unmapped_source_count == 0
        assert all(item.matched_count == 0 for item in unmatched.coverage.concept_sets)
        assert unmatched.review_required is True
        assert unmatched.warnings[0].sub_reasons == ("concept_set_unmatched",)
        connection.close()
        tables = _tables()
        connection = write_omop_duckdb(
            replace(
                tables,
                tables={
                    name: rows if name == "concept" else ()
                    for name, rows in tables.tables.items()
                },
            )
        )
        empty = resolve_phenotype(
            definition, connection, hierarchy=load_athena_hierarchy(ATHENA)
        )
        assert empty.coverage.source_patient_count == 0
        assert empty.warnings[0].sub_reasons == (
            "concept_set_unmatched",
            "no_source_patients",
        )
    finally:
        connection.close()


def test_unmapped_sources_are_counted_without_values_or_patient_keys() -> None:
    connection = write_omop_duckdb(_tables())
    definition = PhenotypeDefinition.load(PHENOTYPES / "diabetes_on_metformin.json")
    try:
        event_count = sum(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table, _ in DOMAIN_CONCEPTS
        )
        for table, concept in DOMAIN_CONCEPTS:
            connection.execute(f"UPDATE {table} SET {concept} = 0")
        result = resolve_phenotype(
            definition, connection, hierarchy=load_athena_hierarchy(ATHENA)
        )
        assert result.patient_ids == ()
        assert result.coverage.unmapped_source_count == event_count > 0
        assert result.warnings[0].sub_reasons == (
            "concept_set_unmatched",
            "unmapped_sources_present",
        )
        diagnostic = json.dumps(
            {
                "coverage": result.coverage.to_dict(),
                "warnings": [warning.to_dict() for warning in result.warnings],
            }
        )
        assert "raw-person" not in diagnostic
        assert "diabetes" not in diagnostic
        assert "metformin" not in diagnostic
        assert "source_value" not in diagnostic
    finally:
        connection.close()


def test_coverage_counts_source_concepts_even_when_eligibility_is_empty() -> None:
    definition = PhenotypeDefinition(
        id="empty-window",
        name="Synthetic empty window",
        concept_sets=(ConceptSet("metformin", "OMOP", (1503297,)),),
        expression=Expression.leaf(
            Criterion(
                id="empty-window",
                concept_set="metformin",
                temporal=TemporalWindow(start_date="1900-01-01", end_date="1900-01-02"),
            )
        ),
    )
    connection = write_omop_duckdb(_tables())
    try:
        result = resolve_phenotype(
            definition, connection, hierarchy=load_athena_hierarchy(ATHENA)
        )
    finally:
        connection.close()
    assert result.patient_ids == ()
    assert result.coverage.concept_sets[0].matched_count == 1
    assert "concept_set_unmatched" not in result.warnings[0].sub_reasons
    assert result.provenance.concept_sets[0].matched_members == ()


def test_unavailable_coverage_stays_unknown_and_preserves_matches(caplog) -> None:
    connection = write_omop_duckdb(_tables())
    definition = PhenotypeDefinition.load(PHENOTYPES / "diabetes_on_metformin.json")

    class UnavailableCoverage:
        def execute(self, sql, parameters=()):
            if sql.startswith("WITH source_events") or sql.startswith(
                "SELECT COUNT(DISTINCT"
            ):
                raise RuntimeError("synthetic-private-source-sentinel")
            return connection.execute(sql, parameters)

    try:
        expected = resolve_phenotype(
            definition, connection, hierarchy=load_athena_hierarchy(ATHENA)
        )
        result = CohortResolver(
            UnavailableCoverage(), hierarchy=load_athena_hierarchy(ATHENA)
        ).resolve(definition)
        for table, _ in DOMAIN_CONCEPTS:
            connection.execute(f"DELETE FROM {table}")
        empty = CohortResolver(
            UnavailableCoverage(), hierarchy=load_athena_hierarchy(ATHENA)
        ).resolve(definition)
    finally:
        connection.close()
    assert result.patient_ids == expected.patient_ids
    assert result.evidence == expected.evidence
    assert result.coverage.source_patient_count is None
    assert result.coverage.unmapped_source_count is None
    assert all(
        item.matched_count is None and item.expanded_count > 0
        for item in result.coverage.concept_sets
    )
    assert result.review_required is False
    assert empty.review_required is True
    assert empty.warnings[0].sub_reasons == ("coverage_unknown",)
    assert caplog.records == []
    assert "synthetic-private-source-sentinel" not in json.dumps(empty.to_dict())


def test_empty_parquet_snapshot_has_the_same_coverage_as_duckdb(tmp_path) -> None:
    tables = _tables()
    empty_tables = replace(
        tables,
        tables={
            name: (() if name in dict(DOMAIN_CONCEPTS) else rows)
            for name, rows in tables.tables.items()
        },
    )
    write_omop_parquet(empty_tables, tmp_path)
    definition = PhenotypeDefinition.load(PHENOTYPES / "diabetes_on_metformin.json")
    connection = write_omop_duckdb(empty_tables)
    try:
        direct = resolve_phenotype(
            definition, connection, hierarchy=load_athena_hierarchy(ATHENA)
        )
    finally:
        connection.close()
    parquet = resolve_phenotype(
        definition, parquet_directory=tmp_path, hierarchy=load_athena_hierarchy(ATHENA)
    )
    assert parquet.to_dict() == direct.to_dict()
    assert parquet.review_required is True


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
