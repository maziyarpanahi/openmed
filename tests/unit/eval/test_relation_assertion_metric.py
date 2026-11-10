"""Tests for the relation-assertion consistency metric."""

from __future__ import annotations

from openmed.eval.metrics import ConsistencyMetric, relation_assertion_consistency


def test_perfect_consistency():
    gold = [("r1", "confirmed"), ("r2", "refuted"), ("r3", "conditional")]
    predicted = [("r1", "confirmed"), ("r2", "refuted"), ("r3", "conditional")]

    metric = relation_assertion_consistency(predicted, gold)

    assert isinstance(metric, ConsistencyMetric)
    assert metric.score == 1.0
    assert metric.consistent == 3
    assert metric.total == 3
    assert metric.violations == {}


def test_mismatch_is_recorded_as_violation():
    gold = [("r1", "confirmed"), ("r2", "refuted")]
    predicted = [("r1", "confirmed"), ("r2", "confirmed")]

    metric = relation_assertion_consistency(predicted, gold)

    assert metric.score == 0.5
    assert metric.consistent == 1
    assert metric.total == 2
    assert "r2" in metric.violations


def test_empty_inputs_score_one():
    metric = relation_assertion_consistency([], [])
    assert metric.score == 1.0
    assert metric.total == 0
