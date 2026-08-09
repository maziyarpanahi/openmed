"""Tests for normalized clinical timeline resolution (OM-609)."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import (
    EVENT_ANCHORING_ADVISORY,
    HISTORICAL,
    RECENT,
    TIMELINE_ASSISTIVE_DISCLAIMER,
    TimeExpr,
    anchor_events,
    detect_timexes,
    evaluate_timeline_gold,
    extract_timex,
    normalize_temporal,
    order_events,
    resolve_temporality,
    resolve_timeline,
)
from openmed.core.audit import hash_text

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


def test_extract_timex_exposes_kinds_and_optional_document_anchoring() -> None:
    text = (
        "On 2026-06-01 she had surgery. Symptoms began 3 days ago and lasted "
        "for 2 weeks."
    )

    unanchored = extract_timex(text)
    assert all(isinstance(expression, TimeExpr) for expression in unanchored)
    by_text = {expression.text: expression for expression in unanchored}
    assert by_text["2026-06-01"].kind == "DATE"
    assert by_text["for 2 weeks"].kind == "DURATION"
    assert by_text["3 days ago"].value is None
    assert by_text["3 days ago"].relative_value == "P3D"
    assert by_text["3 days ago"].offset_days == -3

    anchored = extract_timex(text, document_time="2026-06-15")
    anchored_by_text = {expression.text: expression for expression in anchored}
    assert anchored_by_text["3 days ago"].value == "2026-06-12"
    assert anchored_by_text["3 days ago"].reference_time == "2026-06-15"


def test_compact_order_events_assigns_doc_time_rel_and_stable_partial_order() -> None:
    text = "History of MI 3 days ago. Chest pain then surgery in 2 days."
    timexes = extract_timex(text, document_time="2026-06-15")
    spans = [
        {
            "id": "history-mi",
            "label": "EVENT",
            "text": "MI",
            "start": text.index("MI"),
            "end": text.index("MI") + len("MI"),
            "context": text,
        },
        {
            "id": "surgery",
            "label": "EVENT",
            "text": "surgery",
            "start": text.index("surgery"),
            "end": text.index("surgery") + len("surgery"),
            "context": text,
        },
    ]

    timeline = order_events(reversed(spans), timexes)

    assert [event.event_id for event in timeline] == ["history-mi", "surgery"]
    assert [event.doc_time_rel for event in timeline] == ["before", "after"]
    assert timeline[0].normalized_value == "2026-06-12"
    assert timeline[1].normalized_value == "2026-06-17"
    assert timeline.is_cycle_free
    assert len(timeline.reduced_graph) == 1
    assert "History of" not in json.dumps(timeline.to_dict())


def test_order_events_orders_then_cue_deterministically() -> None:
    text = "Pain improved then surgery followed."
    spans = [
        {
            "label": "EVENT",
            "text": value,
            "start": text.index(value),
            "end": text.index(value) + len(value),
        }
        for value in ("Pain", "surgery")
    ]

    baseline = order_events(text, spans)
    reordered = order_events(text, reversed(spans))

    assert [event.text for event in baseline] == ["Pain", "surgery"]
    assert baseline.to_dict() == reordered.to_dict()
    assert len(baseline.reduced_graph) == 1


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


def test_event_anchoring_date_arithmetic_fixture() -> None:
    fixture = json.loads(GOLD_FIXTURE.read_text(encoding="utf-8"))

    assert fixture["synthetic"] is True
    for case in fixture["event_anchor_cases"]:
        result = anchor_events(
            case["text"],
            [case["event_span"]],
            case["document_creation_time"],
            case["timex_spans"],
        )
        anchor = result.anchors[0]

        assert anchor.anchor_source == case["expected"]["anchor_source"], case["id"]
        assert anchor.anchor_value == case["expected"]["anchor_value"], case["id"]
        assert anchor.dct_position == case["expected"]["dct_position"], case["id"]
        assert anchor.event_text_hash == hash_text(
            case["text"][anchor.event_start : anchor.event_end]
        )


def test_event_anchoring_autodetects_nearby_timex_without_crossing_sentences() -> None:
    text = "On 2026-06-01 symptoms resolved. Cough worsened."
    result = anchor_events(
        text,
        [
            (text.index("resolved"), text.index("resolved") + len("resolved")),
            (text.index("worsened"), text.index("worsened") + len("worsened")),
        ],
        "2026-06-15",
    )

    assert [anchor.anchor_source for anchor in result.anchors] == [
        "timex",
        "dct_fallback",
    ]
    assert result.anchors[0].anchor_value == "2026-06-01"
    assert result.anchors[1].anchor_value == "2026-06-15"


def test_event_anchoring_reuses_supplied_normalized_timex_value() -> None:
    text = "Fever started 3 days ago."
    timex = normalize_temporal(text, [(14, 24)], "2026-06-15")[0]

    result = anchor_events(text, [(6, 13)], "2026-06-20", [timex])

    assert result.anchors[0].anchor_value == "2026-06-12"
    assert result.anchors[0].dct_position == "before"
    assert result.anchors[0].timex is not None
    assert result.anchors[0].timex.normalized_value == timex.value


def test_event_anchoring_output_contains_offsets_and_hashes_not_note_text() -> None:
    text = "Fever started 3 days ago."
    result = anchor_events(text, [(6, 13)], "2026-06-15", [(14, 24)])
    payload = result.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert result.disclaimer == EVENT_ANCHORING_ADVISORY
    assert payload["anchors"][0]["event"] == {
        "span": [6, 13],
        "start": 6,
        "end": 13,
        "text_hash": hash_text("started"),
    }
    assert payload["anchors"][0]["timex"]["text_hash"] == hash_text("3 days ago")
    assert "started" not in encoded
    assert "3 days ago" not in encoded
    assert "text" not in payload["anchors"][0]["event"]
    assert "text" not in payload["anchors"][0]["timex"]


def test_synthetic_gold_corpus_meets_timeline_ci_gates() -> None:
    result = evaluate_timeline_gold(GOLD_FIXTURE)

    assert result.value_accuracy >= 0.85, result.to_dict()
    assert result.ordering_consistency >= 0.90, result.to_dict()
    assert result.failures == ()


def _events_by_timex_text(timeline) -> dict[str, object]:
    return {event.timex.text: event for event in timeline.events}
