"""Tests for deterministic TIMEX3-style temporal normalization (OM-516)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from openmed.clinical import NormalizedTimex, normalize_temporal

ROOT = Path(__file__).resolve().parents[3]
GOLD_FIXTURE = (
    ROOT / "tests" / "fixtures" / "clinical" / "temporal_normalization_gold.json"
)


def test_temporal_normalization_gold_fixture() -> None:
    fixture = json.loads(GOLD_FIXTURE.read_text(encoding="utf-8"))

    assert fixture["synthetic"] is True
    for case in fixture["cases"]:
        record = normalize_temporal(
            case["text"],
            [case["span"]],
            fixture["reference_time"],
        )[0]

        assert isinstance(record, NormalizedTimex), case["id"]
        assert record.type == case["type"], case["id"]
        assert record.value == case["value"], case["id"]
        assert record.anchor == case["anchor"], case["id"]
        assert list(record.granularity_flags) == case["granularity_flags"], case["id"]
        assert record.span == tuple(case["span"]), case["id"]
        assert case["text"][record.start : record.end] == record.text, case["id"]


def test_ambiguous_and_unanchored_values_are_not_guessed() -> None:
    text = "On 03/04/2026 she recalled March and symptoms 3 weeks ago."
    expressions = ("03/04/2026", "March", "3 weeks ago")
    spans = [
        {
            "start": text.index(expression),
            "end": text.index(expression) + len(expression),
        }
        for expression in expressions
    ]

    ambiguous_date, month_only, relative = normalize_temporal(text, spans, None)

    assert ambiguous_date.value is None
    assert ambiguous_date.granularity_flags == ("day", "ambiguous")
    assert month_only.value is None
    assert month_only.granularity_flags == ("month", "ambiguous", "unanchored")
    assert relative.value is None
    assert relative.anchor is None
    assert relative.granularity_flags == ("day", "unanchored")


def test_last_next_calendar_arithmetic_and_month_end_clamping() -> None:
    text = "last month; next year; 1 month ago"
    expressions = ("last month", "next year", "1 month ago")
    spans = [
        (text.index(expression), text.index(expression) + len(expression))
        for expression in expressions
    ]

    records = normalize_temporal(text, spans, date(2024, 3, 31))

    assert [record.value for record in records] == ["2024-02", "2025", "2024-02-29"]
    assert [record.granularity_flags for record in records] == [
        ("month",),
        ("year",),
        ("day",),
    ]


def test_absolute_time_duration_and_recurring_set_types() -> None:
    text = "At 08:45:30, continue every 8 hours and observe for 2 weeks."
    expressions = ("08:45:30", "every 8 hours", "for 2 weeks")
    spans = [
        {
            "start": text.index(expression),
            "end": text.index(expression) + len(expression),
        }
        for expression in expressions
    ]

    time_record, set_record, duration = normalize_temporal(
        text,
        spans,
        "2026-06-15",
    )

    assert time_record.timex_type == "TIME"
    assert time_record.value == "2026-06-15T08:45:30"
    assert time_record.granularity_flags == ("second",)
    assert set_record.timex_type == "SET"
    assert set_record.value == "R/PT8H"
    assert duration.timex_type == "DURATION"
    assert duration.value == "P2W"


def test_normalization_is_offline_deterministic_and_emits_no_logs(caplog) -> None:
    text = "about 2 days ago"
    spans = [{"start": 0, "end": len(text)}]

    first = normalize_temporal(text, spans, "2026-06-15T12:00:00+02:00")
    second = normalize_temporal(text, spans, "2026-06-15T12:00:00+02:00")

    assert first == second
    assert first[0].value == "2026-06-13"
    assert first[0].granularity_flags == ("day", "approximate")
    assert caplog.records == []


@pytest.mark.parametrize(
    "span",
    [
        {"start": -1, "end": 2},
        {"start": 2, "end": 2},
        {"start": 0, "end": 4},
        {"start": "bad", "end": 2},
        (0,),
    ],
)
def test_invalid_spans_are_rejected(span) -> None:
    with pytest.raises(ValueError):
        normalize_temporal("abc", [span], "2026-06-15")


def test_json_representation_retains_exact_span_and_type() -> None:
    record = normalize_temporal("POD 2", [(0, 5)], "2026-06-15")[0]

    assert record.to_dict() == {
        "text": "POD 2",
        "span": [0, 5],
        "start": 0,
        "end": 5,
        "type": "DATE",
        "value": "2026-06-17",
        "anchor": "2026-06-15",
        "granularity_flags": ["day"],
    }
