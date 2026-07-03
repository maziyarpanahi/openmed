"""Tests for normalized clinical timeline resolution (OM-609)."""

from __future__ import annotations

from pathlib import Path

from openmed.clinical import (
    HISTORICAL,
    RECENT,
    TIMELINE_ASSISTIVE_DISCLAIMER,
    detect_timexes,
    evaluate_timeline_gold,
    resolve_temporality,
    resolve_timeline,
)

ROOT = Path(__file__).resolve().parents[3]
GOLD_FIXTURE = ROOT / "tests" / "fixtures" / "clinical" / "timeline_gold.json"


def test_timex_detection_types_and_provenance_offsets() -> None:
    text = (
        "On 2026-06-01 she had surgery. Three weeks ago pain began. "
        "Symptoms lasted for 3 weeks and aspirin is daily."
    )

    timexes = detect_timexes(text)
    by_text = {timex.text: timex for timex in timexes}

    assert by_text["2026-06-01"].timex_type == "DATE"
    assert by_text["Three weeks ago"].direction == "past"
    assert by_text["for 3 weeks"].timex_type == "DURATION"
    assert by_text["daily"].timex_type == "SET"
    for timex in timexes:
        assert text[timex.start : timex.end] == timex.text


def test_resolve_timeline_chained_anchors_and_reference_provenance() -> None:
    text = (
        "Surgery was performed on 2026-06-01. On post-op day 2, fever resolved. "
        "Last admission was 3 weeks ago. Since last admission, dyspnea improved."
    )

    timeline = resolve_timeline(text, reference_date="2026-06-15")
    events = _events_by_timex_text(timeline)

    assert events["2026-06-01"].normalized_value == "2026-06-01/2026-06-01"
    assert events["post-op day 2"].normalized_value == "2026-06-03/2026-06-03"
    assert events["3 weeks ago"].normalized_value == "2026-05-25/2026-05-25"
    assert events["Since last admission"].normalized_value == "2026-05-25/2026-06-15"
    assert events["3 weeks ago"].interval is not None
    assert events["3 weeks ago"].interval.lower_bound.isoformat() == "2026-05-22"
    assert events["3 weeks ago"].interval.upper_bound.isoformat() == "2026-05-28"
    assert events["Since last admission"].reference_date_provenance == {
        "required": True,
        "provided": True,
        "source": "user_supplied",
        "value": "2026-06-15",
    }
    assert "not a clinical decision" in timeline.disclaimer
    assert timeline.disclaimer == TIMELINE_ASSISTIVE_DISCLAIMER


def test_no_reference_date_keeps_relative_only_ordering() -> None:
    text = "Three weeks ago cough worsened. In two days repeat labs."

    timeline = resolve_timeline(text)
    events = _events_by_timex_text(timeline)

    assert events["Three weeks ago"].normalized_value is None
    assert events["In two days"].normalized_value is None
    assert timeline.reference_date is None
    assert timeline.reference_date_provenance == {
        "required": True,
        "provided": False,
        "source": "not_supplied",
        "value": None,
    }
    document_relations = {
        (relation.evidence, relation.target_id): relation.relation
        for relation in timeline.relations
    }
    assert document_relations[("Three weeks ago", "document_reference")] == "before"
    assert document_relations[("In two days", "document_reference")] == "after"


def test_timeline_reconciles_future_absolute_date_with_context_temporality() -> None:
    text = "History of follow-up in 2 days was entered for scheduling."

    assert resolve_temporality(text) == HISTORICAL
    timeline = resolve_timeline(text, reference_date="2026-06-15")
    event = _events_by_timex_text(timeline)["in 2 days"]

    assert event.normalized_value == "2026-06-17/2026-06-17"
    assert event.temporality == RECENT


def test_synthetic_gold_corpus_meets_timeline_ci_gates() -> None:
    result = evaluate_timeline_gold(GOLD_FIXTURE)

    assert result.value_accuracy >= 0.85, result.to_dict()
    assert result.ordering_consistency >= 0.90, result.to_dict()
    assert result.failures == ()


def _events_by_timex_text(timeline) -> dict[str, object]:
    return {event.timex.text: event for event in timeline.events}
