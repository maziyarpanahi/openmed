"""Synthetic multi-encounter timelines, delayed results and causal negative controls."""

from dataclasses import replace

import pytest

from openmed.training.synthetic.longitudinal import (
    TrajectoryEvent,
    generate_trajectory,
    validate_trajectory,
)


def events():
    return (
        TrajectoryEvent("visit_one", "encounter", 0, 0, "visit_one"),
        TrajectoryEvent("condition", "condition", 0, 0, "visit_one", ("visit_one",)),
        TrajectoryEvent("sample", "laboratory", 1, 3, "visit_one", ("condition",)),
        TrajectoryEvent("visit_two", "encounter", 5, 5, "visit_two", ("sample",)),
        TrajectoryEvent(
            "followup", "outcome", 5, 6, "visit_two", ("sample", "visit_two")
        ),
    )


def test_reproducible_identity_runtime_contract_graph_and_observable_documents():
    t = generate_trajectory(events(), seed=42)
    assert t == generate_trajectory(tuple(reversed(events())), seed=42)
    assert validate_trajectory(t) == ()
    assert len({f.encounter_id for f in t.facts}) == 2
    assert all(f.subject_id == t.subject_id for f in t.facts)
    sample = next(f for f in t.facts if f.fact_type == "laboratory")
    assert sample.effective_time["date"] == "2000-01-02"
    assert sample.attributes["observed_at"] == "2000-01-04T00:00:00Z"
    assert "NOTE" not in str(t.report())


def test_uniform_date_shift_preserves_identity_parents_and_intervals():
    a = generate_trajectory(events(), seed=42)
    b = generate_trajectory(events(), seed=42, date_shift_days=30)
    assert a.subject_id == b.subject_id
    assert [f.fact_id for f in a.facts] == [f.fact_id for f in b.facts]
    assert [f.parent_fact_ids for f in a.facts] == [f.parent_fact_ids for f in b.facts]
    assert b.facts[0].effective_time["date"] == "2000-01-31"
    assert validate_trajectory(b) == ()


@pytest.mark.parametrize(
    "bad",
    [
        (TrajectoryEvent("event", "condition", 0, 0, "missing"),),
        (
            TrajectoryEvent("visit_one", "encounter", 0, 0, "visit_one"),
            TrajectoryEvent("early", "outcome", 0, 0, "visit_one", ("later",)),
            TrajectoryEvent("later", "condition", 1, 1, "visit_one"),
        ),
        (
            TrajectoryEvent("visit_one", "encounter", 0, 0, "visit_one"),
            TrajectoryEvent("a", "condition", 0, 0, "visit_one", ("b",)),
            TrajectoryEvent("b", "condition", 0, 0, "visit_one", ("a",)),
        ),
    ],
)
def test_missing_encounters_future_parents_and_same_day_cycles_reject(bad):
    with pytest.raises(ValueError):
        generate_trajectory(bad)


def test_independent_validator_detects_document_and_causal_time_mutations():
    t = generate_trajectory(events())
    documents = (replace(t.documents[0], text="changed synthetic text"),) + t.documents[
        1:
    ]
    assert "document_hash_mismatch" in validate_trajectory(
        replace(t, documents=documents)
    )
    facts = list(t.facts)
    facts[-1] = replace(facts[-1], effective_time={"date": "1999-01-01"})
    assert "causal_time_mismatch" in validate_trajectory(replace(t, facts=tuple(facts)))


def test_invalid_time_and_oversized_iterator_are_controlled():
    with pytest.raises(ValueError):
        TrajectoryEvent("x", "condition", 3, 2, "visit")
    with pytest.raises(ValueError, match="collection"):
        generate_trajectory(iter([events()[0]] * 1001))


def test_malformed_effective_date_is_a_controlled_validation_finding():
    trajectory = generate_trajectory(events())
    changed = replace(trajectory.facts[0], effective_time={"date": "invalid"})
    assert "invalid_event_time" in validate_trajectory(
        replace(trajectory, facts=(changed,) + trajectory.facts[1:])
    )


def test_empty_trajectory_and_invalid_report_metadata_are_rejected():
    original = generate_trajectory(events())
    empty = replace(original, facts=(), artifacts=(), evidence=(), documents=())
    assert validate_trajectory(empty) == ("empty_trajectory",)
    with pytest.raises(ValueError, match="report"):
        empty.report()
    with pytest.raises(ValueError, match="report"):
        replace(original, subject_id="raw source sentinel").report()


@pytest.mark.parametrize(
    "parent_observed,should_fail",
    [("2000-01-01T00:00:00-12:00", True), ("2000-01-01T03:00:00+12:00", False)],
)
def test_causal_observation_order_compares_actual_timezone_instants(
    parent_observed, should_fail
):
    original = generate_trajectory(
        (
            TrajectoryEvent("parent", "encounter", 0, 0, "parent"),
            TrajectoryEvent("child", "condition", 0, 0, "parent", ("parent",)),
        )
    )
    facts = tuple(
        replace(
            fact,
            attributes=dict(
                fact.attributes,
                observed_at="2000-01-01T01:00:00Z"
                if fact.parent_fact_ids
                else parent_observed,
            ),
        )
        for fact in original.facts
    )
    report = validate_trajectory(replace(original, facts=facts))
    assert ("causal_time_mismatch" in report) is should_fail
