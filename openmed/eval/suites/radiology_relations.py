"""Synthetic RadGraph-style radiology entity-and-relation evaluation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from openmed.eval.datasets.dua_stubs import (
    RadiologyEntityRelationFixture,
    load_synthetic_radiology_fixtures,
)
from openmed.eval.metrics import (
    EvalSpan,
    compute_radiology_entity_relation_metrics,
    normalize_radiology_entity,
)
from openmed.eval.relation_metrics import (
    EvalRelation,
    normalize_eval_relations,
)
from openmed.eval.report import BenchmarkReport

RADIOLOGY_ENTITY_RELATION = "radiology-entity-relation"
RADIOLOGY_MEDICAL_DEVICE_DISCLAIMER = (
    "Synthetic radiology evaluation only; not a medical device, clinical ground "
    "truth, or a substitute for clinician review."
)


def radiology_entity_relation_suite_metadata() -> dict[str, Any]:
    """Return provenance and safety metadata for the synthetic suite."""
    return {
        "suite": RADIOLOGY_ENTITY_RELATION,
        "schema_version": 1,
        "source": "committed synthetic fixtures",
        "synthetic_only": True,
        "redistribution": "safe; no RadGraph, MIMIC-CXR, DUA, or production rows",
        "medical_device_disclaimer": RADIOLOGY_MEDICAL_DEVICE_DISCLAIMER,
        "task": "radiology_entity_relation",
        "radiology_entity_relation_required": True,
    }


def score_radiology_entity_relation_fixtures(
    fixtures: Sequence[RadiologyEntityRelationFixture],
    predictions_by_fixture: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score radiology entities, relations, and finding uncertainty."""
    if not fixtures:
        raise ValueError("radiology entity-and-relation evaluation requires fixtures")
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("duplicate radiology fixture ids")
    predictions = predictions_by_fixture or {}

    gold_entities: list[EvalSpan] = []
    predicted_entities: list[EvalSpan] = []
    gold_relations: list[EvalRelation] = []
    predicted_relations: list[EvalRelation] = []
    offset = 0
    for fixture in fixtures:
        raw_prediction = predictions.get(fixture.fixture_id, {})
        if not isinstance(raw_prediction, Mapping):
            raise ValueError("radiology fixture prediction must be a mapping")
        prediction_entities = _prediction_entities(
            raw_prediction.get("entities") or raw_prediction.get("spans") or [],
            fixture=fixture,
        )
        prediction_relations = normalize_eval_relations(
            raw_prediction.get("relations") or [],
            entity_spans=prediction_entities,
            fixture_id=fixture.fixture_id,
            default_language=fixture.language,
            source_text=fixture.text,
        )

        gold_entities.extend(_shift_entities(fixture.gold_spans, offset))
        predicted_entities.extend(_shift_entities(prediction_entities.values(), offset))
        gold_relations.extend(_shift_relations(fixture.gold_relations, offset))
        predicted_relations.extend(_shift_relations(prediction_relations, offset))
        offset += len(fixture.text) + 1

    metrics = compute_radiology_entity_relation_metrics(
        gold_entities,
        predicted_entities,
        gold_relations,
        predicted_relations,
    )
    return {
        "suite": RADIOLOGY_ENTITY_RELATION,
        "fixture_count": len(fixtures),
        "entity_count": len(gold_entities),
        "relation_count": len(gold_relations),
        "metrics": {"radiology_entity_relation": metrics},
        "metadata": {
            **radiology_entity_relation_suite_metadata(),
            "fixture_ids": fixture_ids,
            "relation_types": sorted(
                {relation.relation_type for relation in gold_relations}
            ),
        },
    }


def build_radiology_entity_relation_report(
    fixtures: Sequence[RadiologyEntityRelationFixture],
    predictions_by_fixture: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str,
    device: str = "cpu",
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Build a benchmark report suitable for G13 and model scorecards."""
    scored = score_radiology_entity_relation_fixtures(
        fixtures,
        predictions_by_fixture,
    )
    return BenchmarkReport(
        suite=RADIOLOGY_ENTITY_RELATION,
        model_name=model_name,
        device=device,
        fixture_count=int(scored["fixture_count"]),
        generated_at=generated_at,
        metrics=scored["metrics"],
        metadata=scored["metadata"],
    )


def run_synthetic_radiology_entity_relation_eval(
    predictions_by_fixture: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str,
    device: str = "cpu",
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Score predictions against the committed synthetic offline fixtures."""
    return build_radiology_entity_relation_report(
        load_synthetic_radiology_fixtures(),
        predictions_by_fixture,
        model_name=model_name,
        device=device,
        generated_at=generated_at,
    )


def _prediction_entities(
    raw_entities: Any,
    *,
    fixture: RadiologyEntityRelationFixture,
) -> dict[str, EvalSpan]:
    if isinstance(raw_entities, Mapping):
        rows = [(str(entity_id), row) for entity_id, row in raw_entities.items()]
    elif isinstance(raw_entities, list | tuple):
        rows = []
        for index, row in enumerate(raw_entities, start=1):
            if not isinstance(row, Mapping):
                raise ValueError("radiology predicted entity must be a mapping")
            entity_id = str(
                row.get("id")
                or row.get("entity_id")
                or row.get("span_id")
                or f"prediction-{index}"
            )
            rows.append((entity_id, row))
    else:
        raise ValueError("radiology predicted entities must be a mapping or list")

    entities: dict[str, EvalSpan] = {}
    for entity_id, row in rows:
        if entity_id in entities:
            raise ValueError(f"duplicate radiology predicted entity id: {entity_id}")
        entities[entity_id] = normalize_radiology_entity(
            row,
            default_language=fixture.language,
            source_text=fixture.text,
        )
    return entities


def _shift_entities(entities: Any, offset: int) -> list[EvalSpan]:
    return [
        replace(entity, start=entity.start + offset, end=entity.end + offset)
        for entity in entities
    ]


def _shift_relations(relations: Any, offset: int) -> list[EvalRelation]:
    return [
        replace(
            relation,
            head=replace(
                relation.head,
                start=relation.head.start + offset,
                end=relation.head.end + offset,
            ),
            tail=replace(
                relation.tail,
                start=relation.tail.start + offset,
                end=relation.tail.end + offset,
            ),
        )
        for relation in relations
    ]


__all__ = [
    "RADIOLOGY_ENTITY_RELATION",
    "RADIOLOGY_MEDICAL_DEVICE_DISCLAIMER",
    "build_radiology_entity_relation_report",
    "radiology_entity_relation_suite_metadata",
    "run_synthetic_radiology_entity_relation_eval",
    "score_radiology_entity_relation_fixtures",
]
