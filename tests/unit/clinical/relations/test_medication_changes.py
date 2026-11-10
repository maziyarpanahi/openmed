"""Tests for guarded medication-change relation candidates."""

import json

from openmed.clinical.relations.medication_changes import (
    generate_medication_change_candidates,
)


def _span(text: str, value: str, label: str, **extra):
    start = text.index(value)
    return {"label": label, "start": start, "end": start + len(value), **extra}


def test_links_change_action_time_and_assertion_without_prescribing() -> None:
    text = "Plan: do not start aspirin today."
    spans = [_span(text, "aspirin", "MEDICATION")]

    (candidate,) = generate_medication_change_candidates(text, spans)

    assert candidate.change_type == "start"
    assert candidate.event_time is not None
    assert candidate.event_assertion.negation == "negated"
    assert candidate.review_required is True
    assert candidate.prescribing_action is False
    payload = json.dumps(candidate.to_dict(), sort_keys=True)
    assert "aspirin" not in payload
    assert "today" not in payload


def test_supports_stop_hold_resume_and_dose_changes_deterministically() -> None:
    cases = (
        ("Stop aspirin.", "stop"),
        ("Hold aspirin.", "hold"),
        ("Resume aspirin.", "resume"),
        ("Increase aspirin.", "dose_increase"),
        ("Decrease aspirin.", "dose_decrease"),
    )
    for text, expected in cases:
        spans = [_span(text, "aspirin", "DRUG")]
        first = generate_medication_change_candidates(text, spans)
        second = generate_medication_change_candidates(text, spans)
        assert first == second
        assert [candidate.change_type for candidate in first] == [expected]


def test_requires_an_upstream_medication_span() -> None:
    assert generate_medication_change_candidates("Stop aspirin.", ()) == ()
