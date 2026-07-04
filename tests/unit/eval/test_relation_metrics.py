"""Tests for synthetic relation extraction metrics."""

from __future__ import annotations

import json

import pytest

from openmed.eval.metrics import EvalSpan
from openmed.eval.relation_metrics import (
    EvalRelation,
    compute_relation_metrics,
    compute_strict_relation_f1,
)
from openmed.eval.suites.relations import (
    RelationFixture,
    load_relation_fixtures,
    relation_trap_summary,
    score_relation_fixtures,
)


def test_strict_and_relaxed_relation_f1_differ_on_boundary_drift():
    gold = [
        _relation(
            "TREATS",
            _span(0, 7, "MEDICATION", "Aspirin"),
            _span(15, 20, "CONDITION", "fever"),
        )
    ]
    predicted = [
        _relation(
            "TREATS",
            _span(0, 7, "MEDICATION", "Aspirin"),
            _span(14, 20, "CONDITION", " fever"),
        )
    ]

    metrics = compute_relation_metrics(gold, predicted)

    assert metrics["strict"]["f1"] == 0.0
    assert metrics["strict"]["false_negatives"] == 1
    assert metrics["relaxed"]["f1"] == 1.0
    assert metrics["by_type"]["TREATS"]["relaxed"]["true_positives"] == 1


def test_relation_f1_counts_wrong_type_missing_and_extra_predictions():
    gold = [
        _relation(
            "TREATS",
            _span(0, 7, "MEDICATION", "Aspirin"),
            _span(15, 20, "CONDITION", "fever"),
        ),
        _relation(
            "TEMPORALLY_BEFORE",
            _span(0, 9, "MEDICATION", "Metformin"),
            _span(26, 33, "GLYCEMIC_MEASURE", "Glucose"),
            scope="document",
        ),
    ]
    predicted = [
        _relation(
            "CAUSES",
            _span(0, 7, "MEDICATION", "Aspirin"),
            _span(15, 20, "CONDITION", "fever"),
        ),
        _relation(
            "TREATS",
            _span(40, 47, "MEDICATION", "Insulin"),
            _span(50, 62, "CONDITION", "headache"),
        ),
    ]

    strict = compute_strict_relation_f1(gold, predicted)

    assert strict.true_positives == 0
    assert strict.false_positives == 2
    assert strict.false_negatives == 2
    assert strict.f1 == 0.0


def test_relation_metrics_include_type_and_scope_breakdowns():
    gold = [
        _relation(
            "TREATS",
            _span(0, 7, "MEDICATION", "Aspirin"),
            _span(15, 20, "CONDITION", "fever"),
        ),
        _relation(
            "TEMPORALLY_BEFORE",
            _span(0, 9, "MEDICATION", "Metformin"),
            _span(26, 33, "GLYCEMIC_MEASURE", "Glucose"),
            scope="document",
        ),
    ]
    predicted = [gold[0]]

    metrics = compute_relation_metrics(gold, predicted)

    assert metrics["counts"]["relation_types"] == ["TEMPORALLY_BEFORE", "TREATS"]
    assert metrics["counts"]["scopes"] == ["sentence", "document"]
    assert metrics["strict"]["precision"] == 1.0
    assert metrics["strict"]["recall"] == pytest.approx(0.5)
    assert metrics["by_type"]["TREATS"]["strict"]["f1"] == 1.0
    assert metrics["by_type"]["TEMPORALLY_BEFORE"]["strict"]["f1"] == 0.0
    assert metrics["by_scope"]["sentence"]["strict"]["f1"] == 1.0
    assert metrics["by_scope"]["document"]["strict"]["f1"] == 0.0


def test_committed_relation_fixture_loads_and_scores_exact_predictions():
    fixtures = load_relation_fixtures()
    predictions = {
        fixture.fixture_id: list(fixture.gold_relations) for fixture in fixtures
    }

    scorecard = score_relation_fixtures(fixtures, predictions)
    traps = relation_trap_summary(fixtures)

    assert [fixture.fixture_id for fixture in fixtures] == [
        "relation-sentence-treatment",
        "relation-assertion-negated",
        "relation-document-temporal",
    ]
    assert scorecard["fixture_count"] == 3
    assert scorecard["relation_count"] == 4
    assert scorecard["metrics"]["strict"]["f1"] == 1.0
    assert scorecard["metrics"]["relaxed"]["f1"] == 1.0
    assert scorecard["metrics"]["by_scope"]["sentence"]["strict"]["f1"] == 1.0
    assert scorecard["metrics"]["by_scope"]["document"]["strict"]["f1"] == 1.0
    assert traps["total"] == 2
    assert traps["by_kind"]["assertion"]["zero_tolerance"] is True
    assert traps["by_kind"]["temporal"]["relation_ids"] == [
        "rel-metformin-glucose",
        "rel-monday-tuesday",
    ]


def test_relation_fixture_loader_rejects_duplicate_fixture_ids(tmp_path):
    path = tmp_path / "relations.jsonl"
    rows = [_fixture_row("duplicate"), _fixture_row("duplicate")]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate relation fixture ids"):
        load_relation_fixtures(path)


def test_relation_fixture_schema_rejects_bad_references_and_traps():
    bad_reference = _fixture_row("bad-reference")
    bad_reference["relations"][0]["tail"] = "missing-span"

    with pytest.raises(ValueError, match="unknown relation argument span id"):
        RelationFixture.from_mapping(bad_reference)

    bad_trap = _fixture_row("bad-trap")
    bad_trap["traps"] = [
        {
            "id": "trap-bad",
            "kind": "assertion",
            "relation_ids": ["missing-relation"],
        }
    ]

    with pytest.raises(ValueError, match="unknown relation ids"):
        RelationFixture.from_mapping(bad_trap)


def _relation(
    relation_type: str,
    head: EvalSpan,
    tail: EvalSpan,
    *,
    scope: str = "sentence",
) -> EvalRelation:
    return EvalRelation(
        relation_type=relation_type,
        head=head,
        tail=tail,
        scope=scope,
    )


def _span(start: int, end: int, label: str, text: str) -> EvalSpan:
    return EvalSpan(start=start, end=end, label=label, text=text)


def _fixture_row(fixture_id: str) -> dict[str, object]:
    return {
        "id": fixture_id,
        "schema_version": 1,
        "language": "en",
        "text": "Aspirin treats fever.",
        "entities": [
            {
                "id": "e-aspirin",
                "start": 0,
                "end": 7,
                "label": "MEDICATION",
                "text": "Aspirin",
            },
            {
                "id": "e-fever",
                "start": 15,
                "end": 20,
                "label": "CONDITION",
                "text": "fever",
            },
        ],
        "relations": [
            {
                "id": "rel-aspirin-fever",
                "type": "treats",
                "head": "e-aspirin",
                "tail": "e-fever",
                "scope": "sentence",
            }
        ],
        "traps": [],
        "metadata": {
            "synthetic": True,
            "category": "relation_gold",
            "schema_version": 1,
        },
    }
