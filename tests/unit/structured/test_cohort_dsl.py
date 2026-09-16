"""Focused tests for the stable cohort phenotype definition schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.structured.cohort import (
    AssertionFilter,
    ConceptSet,
    Criterion,
    Expression,
    OccurrenceCount,
    PhenotypeDefinition,
    PhenotypeDefinitionError,
    TemporalWindow,
    phenotype_from_json,
    phenotype_to_json,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cohort" / "phenotypes"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "diabetes_on_metformin.json",
        "diabetes_without_metformin.json",
        "recurrent_diabetes.json",
    ],
)
def test_worked_definitions_round_trip_to_byte_stable_json(
    fixture_name: str,
    tmp_path: Path,
) -> None:
    definition = PhenotypeDefinition.load(FIXTURES / fixture_name)
    canonical = phenotype_to_json(definition)
    reloaded = phenotype_from_json(canonical)
    output = reloaded.write(tmp_path / fixture_name)

    assert reloaded == definition
    assert reloaded.to_json() == canonical
    assert output.read_bytes() == canonical.encode("utf-8")
    assert reloaded.sha256 == definition.sha256


def test_programmatic_schema_covers_boolean_count_assertion_and_temporal_forms() -> (
    None
):
    first = Expression.leaf(
        Criterion(
            id="first",
            concept_set="conditions",
            occurrence=OccurrenceCount(minimum=2, maximum=4),
            assertion=AssertionFilter(
                negation=("affirmed",),
                temporality=("recent", "historical"),
                certainty=("certain",),
                experiencer=("patient",),
            ),
            temporal=TemporalWindow(
                start_date="2025-01-01",
                end_date="2026-12-31",
            ),
        )
    )
    second = Expression.leaf(
        Criterion(
            id="second",
            concept_set="medications",
            temporal=TemporalWindow(
                anchor_criterion="first",
                days_before=7,
                days_after=30,
            ),
        )
    )
    definition = PhenotypeDefinition(
        id="programmatic-example",
        name="Programmatic example",
        concept_sets=(
            ConceptSet("conditions", "OMOP", (201826,), True),
            ConceptSet("medications", "OMOP", (1503297,)),
        ),
        expression=Expression.any_of(first, Expression.exclude(second)),
    )

    assert PhenotypeDefinition.from_json(definition.to_json()) == definition
    assert definition.expression.to_dict()["operator"] == "or"
    assert definition.criteria()[0].occurrence.maximum == 4


def test_schema_rejects_unknown_fields_and_invalid_references() -> None:
    payload = {
        "id": "invalid",
        "name": "Invalid",
        "concept_sets": [
            {
                "id": "known",
                "vocabulary": "OMOP",
                "concept_ids": [201826],
                "include_descendants": False,
            }
        ],
        "expression": {
            "operator": "criterion",
            "criterion": {
                "id": "leaf",
                "concept_set": "missing",
                "occurrence": {"minimum": 1},
                "assertion": {"negation": ["affirmed"]},
            },
        },
        "executable_sql": "SELECT *",
    }

    with pytest.raises(PhenotypeDefinitionError, match="unknown fields"):
        PhenotypeDefinition.from_dict(payload)

    payload.pop("executable_sql")
    with pytest.raises(PhenotypeDefinitionError, match="unknown concept sets"):
        PhenotypeDefinition.from_dict(payload)
