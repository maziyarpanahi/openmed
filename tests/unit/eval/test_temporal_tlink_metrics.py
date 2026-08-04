"""Tests for synthetic temporal TLINK metrics and the blocking gate."""

from __future__ import annotations

import json

import pytest

from openmed.eval.golden.loader import list_fixture_paths
from openmed.eval.metrics import (
    TEMPORAL_AWARENESS_F1_FLOOR,
    assert_temporal_consistency_gate,
    compute_temporal_awareness_f1,
    compute_temporal_closure_consistency,
    evaluate_temporal_consistency_gate,
)
from openmed.eval.suites.temporal_tlinks import (
    TEMPORAL_TLINK_FIXTURE_PATH,
    assert_temporal_tlink_gate,
    evaluate_temporal_tlink_fixtures,
    load_temporal_tlink_fixtures,
)


def test_temporal_awareness_uses_inverse_normalization_reduction_and_closure():
    gold = [
        ("BEFORE", "event-a", "event-b"),
        ("BEFORE", "event-b", "event-c"),
        ("OVERLAP", "event-d", "event-e"),
    ]
    predicted = [
        ("AFTER", "event-b", "event-a"),
        ("BEFORE", "event-b", "event-c"),
        ("BEFORE", "event-a", "event-c"),
        ("OVERLAP", "event-e", "event-d"),
    ]

    metric = compute_temporal_awareness_f1(gold, predicted)

    assert metric.precision == 1.0
    assert metric.recall == 1.0
    assert metric.f1 == 1.0
    assert metric.predicted_reduced_relations == 3
    assert metric.gold_reduced_relations == 3


def test_temporal_awareness_does_not_treat_one_implied_edge_as_a_full_timeline():
    metric = compute_temporal_awareness_f1(
        [
            ("BEFORE", "event-a", "event-b"),
            ("BEFORE", "event-b", "event-c"),
        ],
        [("BEFORE", "event-a", "event-c")],
    )

    assert metric.precision == 1.0
    assert metric.recall == 0.0
    assert metric.f1 == 0.0


def test_temporal_closure_consistency_reports_cycles_without_node_ids():
    valid = compute_temporal_closure_consistency(
        [
            ("BEFORE", "event-a", "event-b"),
            ("BEFORE", "event-b", "event-c"),
        ]
    )
    contradictory = compute_temporal_closure_consistency(
        [
            ("BEFORE", "synthetic-secret-a", "synthetic-secret-b"),
            ("AFTER", "synthetic-secret-a", "synthetic-secret-b"),
        ]
    )
    serialized = json.dumps(contradictory.to_dict(), sort_keys=True)

    assert valid.score == 1.0
    assert valid.violations == {}
    assert contradictory.score == 0.0
    assert contradictory.violations
    assert "synthetic-secret" not in serialized


def test_merge_gate_blocks_a_contradiction_even_when_awareness_clears_floor():
    gold = [
        ("BEFORE", "synthetic-secret-a", "synthetic-secret-b"),
        ("BEFORE", "synthetic-secret-b", "synthetic-secret-c"),
    ]
    contradictory = [
        *gold,
        ("BEFORE", "synthetic-secret-c", "synthetic-secret-a"),
    ]

    result = evaluate_temporal_consistency_gate(gold, contradictory)

    assert result.awareness.f1 >= TEMPORAL_AWARENESS_F1_FLOOR
    assert result.blocking is True
    assert result.passed is False
    assert result.failure_reasons == ("transitive_closure_violation",)
    with pytest.raises(AssertionError, match="transitive_closure_violation") as error:
        assert_temporal_consistency_gate(gold, contradictory)
    assert "synthetic-secret" not in str(error.value)


def test_committed_temporal_gold_is_synthetic_complete_and_specialized():
    fixtures = load_temporal_tlink_fixtures()
    relation_types = {
        relation_type
        for fixture in fixtures
        for relation_type, _, _ in fixture.gold_tlinks
    }

    assert [fixture.fixture_id for fixture in fixtures] == [
        "temporal-discharge-dct-ordering",
        "temporal-discharge-duration-ordering",
    ]
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    assert all(fixture.metadata["contains_real_phi"] is False for fixture in fixtures)
    assert all(fixture.contradictory_candidate_traps for fixture in fixtures)
    assert {"BEFORE", "OVERLAP", "CONTAINS", "BEGINS_ON", "ENDS_ON"} <= (relation_types)
    assert TEMPORAL_TLINK_FIXTURE_PATH.name not in {
        path.name for path in list_fixture_paths()
    }


def test_temporal_fixture_loader_rejects_real_phi_declarations(tmp_path):
    row = json.loads(
        TEMPORAL_TLINK_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[0]
    )
    row["metadata"]["contains_real_phi"] = True
    fixture_path = tmp_path / "temporal.jsonl"
    fixture_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains_real_phi=false"):
        load_temporal_tlink_fixtures(fixture_path)


def test_committed_temporal_gold_clears_the_blocking_gate():
    result = evaluate_temporal_tlink_fixtures()
    asserted = assert_temporal_tlink_gate()

    assert result == asserted
    assert result.passed is True
    assert result.gate.blocking is True
    assert result.gate.awareness.f1 == 1.0
    assert result.gate.awareness.f1 >= TEMPORAL_AWARENESS_F1_FLOOR
    assert result.gate.consistency.score == 1.0
    assert result.gate.consistency.violations == {}
    assert result.pruned_contradictory_trap_count == result.contradictory_trap_count
    assert "Document date" not in json.dumps(result.to_dict(), sort_keys=True)
