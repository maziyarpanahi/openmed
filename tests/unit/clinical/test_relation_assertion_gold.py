"""Golden-fixture accuracy for relation-assertion propagation."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical.relations.assertion_filter import propagate_relation_assertion
from openmed.clinical.relations.candidate import RelationCandidate, SpanReference
from openmed.eval.metrics import relation_assertion_consistency

_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "relation_assertion.jsonl"
)


def _rows() -> list[dict]:
    with _FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _relation(row: dict) -> RelationCandidate:
    text = row["text"]
    head_start = text.index(row["head"])
    attr_start = text.index(row["attribute"])
    return RelationCandidate(
        relation_type=row["relation_type"],
        head=SpanReference(
            text=row["head"],
            label="MEDICATION",
            start=head_start,
            end=head_start + len(row["head"]),
            score=1.0,
        ),
        attribute=SpanReference(
            text=row["attribute"],
            label="dose",
            start=attr_start,
            end=attr_start + len(row["attribute"]),
            score=1.0,
        ),
        score=1.0,
        confidence=1.0,
        features={},
        explanation=(),
    )


def test_gold_relation_assertion_accuracy_at_least_090():
    rows = _rows()
    assert rows, "fixture must not be empty"

    predicted = []
    gold = []
    for index, row in enumerate(rows):
        result = propagate_relation_assertion(_relation(row), row["text"])
        predicted.append((index, result.status))
        gold.append((index, row["expected_status"]))

    metric = relation_assertion_consistency(predicted, gold)
    assert metric.score >= 0.90, metric.violations


def test_gold_covers_all_four_statuses():
    statuses = {row["expected_status"] for row in _rows()}
    assert statuses == {"confirmed", "refuted", "conditional", "possible"}
