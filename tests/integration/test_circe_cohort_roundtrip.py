"""Offline native membership checks across protected Circe JSON round trips."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

from openmed.interop.omop import load_grounded_jsonl, write_omop_duckdb
from openmed.structured.cohort import (
    CohortSourceSnapshot,
    ConceptSet,
    Criterion,
    Expression,
    OccurrenceCount,
    PhenotypeDefinition,
    TemporalWindow,
    export_circe_expression,
    export_cohort_definition,
    import_circe_expression,
    load_athena_hierarchy,
    resolve_phenotype,
)

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cohort"


def snapshot():
    return CohortSourceSnapshot(
        "snapshot_circeintegration01",
        "sha256:" + "a" * 64,
        "synthetic-omop",
        ("synthetic",),
    )


@pytest.mark.parametrize(
    "mode", ["relative", "count_range", "absolute_any", "exclusion"]
)
def test_synthetic_membership_roundtrip(mode):
    definition = PhenotypeDefinition.load(
        FIXTURES / "phenotypes" / "diabetes_on_metformin.json"
    )
    condition, drug = definition.criteria()
    condition = replace(
        condition,
        occurrence=OccurrenceCount(1, 2)
        if mode == "count_range"
        else OccurrenceCount(),
    )
    if mode == "absolute_any":
        drug = replace(
            drug,
            temporal=TemporalWindow(start_date="2026-01-01", end_date="2026-12-31"),
        )
        other = replace(drug, id="other-drug", occurrence=OccurrenceCount(2))
        root = Expression.all_of(
            Expression.leaf(condition),
            Expression.any_of(Expression.leaf(drug), Expression.leaf(other)),
        )
        domains = {
            condition.id: "ConditionOccurrence",
            drug.id: "DrugExposure",
            other.id: "DrugExposure",
        }
    elif mode == "exclusion":
        drug = replace(drug, temporal=None)
        root = Expression.all_of(
            Expression.leaf(condition), Expression.exclude(Expression.leaf(drug))
        )
        domains = {condition.id: "ConditionOccurrence", drug.id: "DrugExposure"}
    else:
        root = Expression.all_of(Expression.leaf(condition), Expression.leaf(drug))
        domains = {condition.id: "ConditionOccurrence", drug.id: "DrugExposure"}
    definition = replace(definition, expression=root)
    envelope = export_cohort_definition(definition, source_snapshot=snapshot()).value
    conversion = export_circe_expression(
        envelope,
        criterion_domains=domains,
        primary_criterion_id=condition.id,
        strict=False,
    ).value
    imported = import_circe_expression(
        conversion.expression,
        source_snapshot=snapshot(),
        expected_expression_digest=conversion.target_digest,
        strict=False,
    ).value
    assert imported.exchange is not None
    connection = write_omop_duckdb(
        load_grounded_jsonl(FIXTURES / "synthetic_grounded.jsonl")
    )
    try:
        hierarchy = load_athena_hierarchy(FIXTURES / "athena")
        first = resolve_phenotype(definition, connection, hierarchy=hierarchy)
        second = resolve_phenotype(
            imported.exchange.definition, connection, hierarchy=hierarchy
        )
        assert first.patient_ids == second.patient_ids
        assert first.patient_ids
    finally:
        connection.close()
    # This is native synthetic membership evidence, not OHDSI execution parity.
    assert conversion.losses and imported.losses


def _evaluate(node, met):
    if node.criterion:
        return node.criterion.concept_set in met
    children = [_evaluate(child, met) for child in node.children]
    if node.operator == "not":
        return not children[0]
    return all(children) if node.operator == "and" else any(children)


def test_at_least_threshold_truth_table_and_export_roundtrip():
    sets = [
        ConceptSet(f"cs_{index}", "Synthetic", (900001 + index,)) for index in range(4)
    ]
    definition = PhenotypeDefinition(
        "threshold",
        "Synthetic threshold",
        tuple(sets),
        Expression.leaf(Criterion("primary", sets[0].id)),
    )
    envelope = export_cohort_definition(definition, source_snapshot=snapshot()).value
    expression = export_circe_expression(
        envelope,
        criterion_domains={"primary": "ConditionOccurrence"},
        primary_criterion_id="primary",
        strict=False,
    ).value.expression
    expression["ConceptSets"] = [
        {
            "id": index,
            "expression": {
                "items": [
                    {
                        "concept": {
                            "CONCEPT_ID": item.concept_ids[0],
                            "VOCABULARY_ID": "Synthetic",
                        }
                    }
                ]
            },
        }
        for index, item in enumerate(sets)
    ]
    expression["InclusionRules"] = [
        {
            "expression": {
                "Type": "AT_LEAST",
                "Count": 2,
                "CriteriaList": [
                    {
                        "Criteria": {"ConditionOccurrence": {"CodesetId": index}},
                        "StartWindow": {"Start": {"Coeff": -1}, "End": {"Coeff": 1}},
                        "IgnoreObservationPeriod": True,
                        "Occurrence": {"Type": 2, "Count": 1},
                    }
                    for index in range(1, 4)
                ],
            }
        }
    ]
    imported = import_circe_expression(
        expression, source_snapshot=snapshot(), strict=False
    ).value.exchange
    criteria = imported.definition.criteria()
    conversion = export_circe_expression(
        imported,
        criterion_domains={item.id: "ConditionOccurrence" for item in criteria},
        primary_criterion_id=criteria[0].id,
        strict=False,
    ).value
    restored = import_circe_expression(
        conversion.expression, source_snapshot=snapshot(), strict=False
    ).value.exchange
    # Every possible membership combination includes negative controls.
    for bits in product((False, True), repeat=4):
        met = {f"cs_{index}" for index, bit in enumerate(bits) if bit}
        expected = bits[0] and sum(bits[1:]) >= 2
        assert _evaluate(imported.definition.expression, met) == expected
        assert _evaluate(restored.definition.expression, met) == expected
