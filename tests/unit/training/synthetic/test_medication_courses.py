"""Synthetic medication gold labels and temporal negative controls."""

from dataclasses import replace

import pytest

from openmed.training.synthetic.medication_courses import (
    AdherenceEvent,
    DoseChange,
    MedicationCourseSpec,
    MedicationOutcome,
    generate_medication_course,
    validate_medication_course,
)


def spec(**overrides):
    args = dict(
        medication_code="synthetic_med",
        indication_code="synthetic_indication",
        dose_unit="synthetic_unit",
        route="oral",
        start_day=0,
        stop_day=10,
        initial_dose=2,
        changes=(DoseChange(4, 3),),
        adherence=(AdherenceEvent(1, "taken"), AdherenceEvent(5, "missed")),
        outcomes=(MedicationOutcome(7, "synthetic_event"),),
    )
    args.update(overrides)
    return MedicationCourseSpec(**args)


def test_reproducibility_regimen_continuity_and_gold_relations():
    course = generate_medication_course(spec(), seed=42)
    assert course == generate_medication_course(spec(), seed=42)
    assert course.course_id != generate_medication_course(spec(), seed=43).course_id
    assert [(r.start_day, r.end_day, r.dose) for r in course.regimens] == [
        (0, 4, 2),
        (4, 10, 3),
    ]
    assert validate_medication_course(course) == ()
    assert {r["label"] for r in course.relations} == {
        "indication_for",
        "modifies_regimen",
        "adherence_to",
        "discontinues",
        "observed_after_start",
    }
    assert not any("caused" in r["label"] for r in course.relations)


def test_adherence_at_titration_day_uses_new_regimen():
    course = generate_medication_course(spec(adherence=(AdherenceEvent(4, "taken"),)))
    event = next(e for e in course.events if e["kind"] == "adherence")
    assert event["regimen_id"] == course.regimens[1].regimen_id


def test_delayed_outcome_requires_explicit_followup_window():
    with pytest.raises(ValueError, match="outcome timeline"):
        spec(outcomes=(MedicationOutcome(12, "synthetic_event"),))
    course = generate_medication_course(
        spec(outcomes=(MedicationOutcome(12, "synthetic_event"),), followup_days=3)
    )
    assert validate_medication_course(course) == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"stop_day": 0},
        {"initial_dose": float("nan")},
        {"initial_dose": True},
        {"changes": (DoseChange(10, 3),)},
        {"changes": (DoseChange(4, 3), DoseChange(4, 4))},
        {"adherence": (AdherenceEvent(10, "taken"),)},
        {"adherence": (AdherenceEvent(1, "taken"), AdherenceEvent(1, "missed"))},
        {"medication_code": "private raw text"},
    ],
)
def test_impossible_timelines_reject_without_echo(overrides):
    with pytest.raises(ValueError) as exc:
        spec(**overrides)
    assert "private raw text" not in str(exc.value)


def test_independent_validator_rejects_regimen_gap_and_dangling_relation():
    course = generate_medication_course(spec())
    broken = replace(
        course, regimens=(course.regimens[0], replace(course.regimens[1], start_day=5))
    )
    assert "regimen_discontinuity" in validate_medication_course(broken)
    broken = replace(
        course,
        relations=course.relations
        + ({"source": "missing", "target": course.course_id, "label": "adherence_to"},),
    )
    assert "dangling_relation" in validate_medication_course(broken)


def test_independent_validator_rejects_adherence_after_stop():
    course = generate_medication_course(spec())
    events = tuple(
        {**event, "day": 20} if event["kind"] == "adherence" else event
        for event in course.events
    )
    assert "adherence_outside_regimen" in validate_medication_course(
        replace(course, events=events)
    )


def test_missing_gold_links_and_altered_indication_are_detected():
    course = generate_medication_course(spec())
    assert "gold_relation_mismatch" in validate_medication_course(
        replace(course, relations=())
    )
    events = tuple(
        {**e, "code": "changed_indication"} if e["kind"] == "indication" else e
        for e in course.events
    )
    assert "indication_mismatch" in validate_medication_course(
        replace(course, events=events)
    )


def test_malformed_events_and_late_outcome_have_controlled_categories():
    course = generate_medication_course(spec())
    assert "invalid_event" in validate_medication_course(replace(course, events=({},)))
    events = tuple(
        {**e, "day": 30} if e["kind"] == "outcome" else e for e in course.events
    )
    assert "outcome_outside_followup" in validate_medication_course(
        replace(course, events=events)
    )


def test_invalid_adherence_regimen_reference_returns_controlled_category():
    course = generate_medication_course(spec())
    for reference in [[], {}, None, "raw submitted reference"]:
        events = tuple(
            dict(event, regimen_id=reference) if event["kind"] == "adherence" else event
            for event in course.events
        )
        assert "invalid_event" in validate_medication_course(
            replace(course, events=events)
        )
