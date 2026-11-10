"""Synthetic tests for explicit temporality buckets and stable provenance."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    BUCKETED_TIMELINE_SCHEMA_VERSION,
    ClinicalTimeline,
    build_timeline,
    extract_timex,
    normalize_conservative_temporal_interval,
)
from openmed.clinical.temporal_intervals import TemporalInterval
from openmed.clinical.timeline.timex import TimeExpr
from openmed.core.audit import hash_text


def _span(
    text: str,
    start: int,
    temporality: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "text": text,
        "label": "CONDITION",
        "start": start,
        "end": start + len(text),
        "temporality": temporality,
        **extra,
    }


def test_buckets_historical_recent_and_hypothetical_without_raw_surfaces() -> None:
    timeline = build_timeline(
        [
            _span("acute MI", 20, "recent"),
            _span("possible future MI", 40, "hypothetical"),
            _span("history of MI", 0, "historical"),
        ]
    )

    assert isinstance(timeline, ClinicalTimeline)
    assert [
        len(timeline.lanes[lane]) for lane in ("historical", "recent", "hypothetical")
    ] == [1, 1, 1]
    assert timeline.lanes["hypothetical"][0] not in timeline.lanes["recent"]
    assert timeline.to_dict()["lanes"] == {
        "historical": [0],
        "recent": [1],
        "hypothetical": [2],
    }
    payload = timeline.to_jsonl()
    assert "history of MI" not in payload
    assert "acute MI" not in payload
    assert "possible future MI" not in payload
    assert "chronology of record" in payload
    assert all(
        row["schema_version"] == BUCKETED_TIMELINE_SCHEMA_VERSION
        for row in (json.loads(line) for line in payload.splitlines())
    )


def test_normalized_times_order_within_lane_and_missing_times_use_offsets() -> None:
    spans = [
        _span("later", 30, "recent", normalized_time="2026-07-02"),
        _span("unanchored second", 60, "recent"),
        _span("earlier", 10, "recent", normalized_time="2026-06-01"),
        _span("unanchored first", 40, "recent"),
    ]
    timeline = build_timeline(spans)

    assert [event.start for event in timeline.lanes["recent"]] == [10, 30, 40, 60]
    assert [event.normalized_time for event in timeline.lanes["recent"]] == [
        "2026-06-01",
        "2026-07-02",
        None,
        None,
    ]
    assert timeline.to_jsonl() == build_timeline(reversed(spans)).to_jsonl()


def test_anchored_and_unanchored_timex_keep_source_evidence() -> None:
    source = "Fever began 3 days ago."
    anchored = extract_timex(source, document_time="2026-06-15")[0]
    relative = extract_timex(source)[0]

    known = build_timeline([_span("Fever", 0, "recent", time_expr=anchored)])
    unknown = build_timeline([_span("Fever", 0, "recent", time_expr=relative)])

    assert known.events[0].normalized_time == "2026-06-12"
    assert unknown.events[0].normalized_time is None
    evidence = known.time_evidence[0]
    assert evidence is not None
    assert (evidence.start, evidence.end) == (anchored.start, anchored.end)
    assert evidence.fingerprint == hash_text("3 days ago")
    assert "3 days ago" not in known.to_jsonl()
    assert unknown.time_evidence[0].state == "relative_or_unknown"
    assert unknown.time_evidence[0].relative_value == "P3D"
    assert unknown.time_evidence[0].direction == "past"


def test_conflicting_date_is_not_promoted_to_a_known_chronology() -> None:
    source = "03/04/2026"
    conflicting = normalize_conservative_temporal_interval(source, (0, len(source)))
    timeline = build_timeline(
        [
            _span(
                "finding",
                20,
                "historical",
                temporal_interval=conflicting,
                context=source,
            )
        ]
    )

    assert conflicting.status == "conflicting"
    assert timeline.events[0].normalized_time is None
    assert timeline.time_evidence[0].state == "conflicting"
    assert timeline.time_evidence[0].fingerprint == hash_text(source)


def test_missing_temporality_is_rejected_without_guessing() -> None:
    with pytest.raises(ValueError, match="explicit temporality"):
        build_timeline([{"text": "MI", "start": 0, "end": 2}])

    timeline = build_timeline([_span("MI", 0, "recent")])
    assert timeline.events[0].assertion.certainty == "uncertain"

    with pytest.raises(ValueError, match="normalized date"):
        build_timeline([_span("MI", 0, "recent", normalized_time="2026-02-30")])


def test_untrusted_caller_time_fields_do_not_enter_report() -> None:
    raw = "Patient Jane Doe"
    relative = TimeExpr(
        text=raw,
        start=0,
        end=len(raw),
        kind="DATE",
        value=None,
        relative_value=raw,
    )
    interval = TemporalInterval(
        kind="date",
        source_start=0,
        source_end=len(raw),
        value=raw,
        precision=raw,  # type: ignore[arg-type]
        timezone_state="unknown",
        conflicts=("ambiguous",),
    )
    timeline = build_timeline(
        [
            _span("finding", 0, "historical", time_expr=relative),
            _span("finding", 30, "recent", temporal_interval=interval),
        ]
    )

    assert raw not in timeline.to_jsonl()
    assert timeline.time_evidence[0].relative_value is None
    assert timeline.time_evidence[1].normalized_value is None
    assert timeline.time_evidence[1].precision == "unknown"


def test_supplied_negation_and_experiencer_are_not_lost():
    timeline = build_timeline(
        [
            _span(
                "finding",
                0,
                "historical",
                assertion={"negation": "negated", "experiencer": "family"},
            )
        ]
    )
    assertion = timeline.events[0].assertion
    assert assertion.negation == "negated"
    assert assertion.experiencer == "family"


def test_aware_datetime_order_uses_instants_not_wall_clock_strings():
    timeline = build_timeline(
        [
            _span("first", 10, "recent", normalized_time="2026-01-01T13:00+02:00"),
            _span("second", 0, "recent", normalized_time="2026-01-01T12:00+00:00"),
        ]
    )
    assert [event.start for event in timeline.events] == [10, 0]


def test_zero_year_month_and_date_errors_are_value_free():
    for value in ("0000-02", "2026-02-30"):
        with pytest.raises(ValueError) as caught:
            build_timeline([_span("finding", 0, "recent", normalized_time=value)])
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
