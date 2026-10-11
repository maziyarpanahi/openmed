"""Tests for deterministic TIMEX3-style temporal normalization (OM-516)."""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

import pytest

from openmed.clinical import NormalizedTimex, normalize_temporal
from openmed.core.iso_temporal import (
    parse_iso_date,
    parse_iso_datetime,
    parse_iso_time,
)

ROOT = Path(__file__).resolve().parents[3]
GOLD_FIXTURE = (
    ROOT / "tests" / "fixtures" / "clinical" / "temporal_normalization_gold.json"
)

# Literal synthetic vectors are also executed unchanged by the local Python
# 3.10/3.11/3.12/3.13 conformance runner. All CI compatibility lanes run this file.
ISO_PROFILE_CASES = [
    ("date", "2026-01-05", "2026-01-05"),
    ("date", "2024-02-29", "2024-02-29"),
    ("date", "0001-01-01", "0001-01-01"),
    ("date", "20260105", None),
    ("date", "2026-W02-1", None),
    ("date", "2026-005", None),
    ("date", "2026-02-29", None),
    ("date", "2026-13-01", None),
    ("date", "2026-01-00", None),
    ("date", "0000-01-01", None),
    ("date", "2026-1-5", None),
    ("date", "2026-01-05Z", None),
    ("date", "\uff12\uff10\uff12\uff16-01-05", None),
    ("date", "synthetic-private-timestamp", None),
    ("datetime", "2026-01-05", "2026-01-05T00:00:00"),
    ("datetime", "2026-01-05T10", "2026-01-05T10:00:00"),
    ("datetime", "2026-01-05T10:30", "2026-01-05T10:30:00"),
    ("datetime", "2026-01-05 10:30:00", "2026-01-05T10:30:00"),
    ("datetime", "2026-01-05t10:30:00z", "2026-01-05T10:30:00+00:00"),
    ("datetime", "2026-01-05T10:30:00Z", "2026-01-05T10:30:00+00:00"),
    ("datetime", "2026-01-05T10:30:00-05:30", "2026-01-05T10:30:00-05:30"),
    ("datetime", "2026-01-05T10:30:00+05:30", "2026-01-05T10:30:00+05:30"),
    ("datetime", "2026-01-05T10:30:00.1Z", "2026-01-05T10:30:00.100000+00:00"),
    ("datetime", "2026-01-05T10:30:00.12Z", "2026-01-05T10:30:00.120000+00:00"),
    ("datetime", "2026-01-05T10:30:00.123Z", "2026-01-05T10:30:00.123000+00:00"),
    ("datetime", "2026-01-05T10:30:00.1234Z", "2026-01-05T10:30:00.123400+00:00"),
    ("datetime", "2026-01-05T10:30:00.12345Z", "2026-01-05T10:30:00.123450+00:00"),
    ("datetime", "2026-01-05T10:30:00.123456Z", "2026-01-05T10:30:00.123456+00:00"),
    ("datetime", "20260105T103000Z", None),
    ("datetime", "2026-W02-1T10:30:00Z", None),
    ("datetime", "2026-005T10:30:00Z", None),
    ("datetime", "2026-01-05T10:30:00+0530", None),
    ("datetime", "2026-01-05T10:30:00+05", None),
    ("datetime", "2026-01-05T10:30:00+05:30:01", None),
    ("datetime", "2026-01-05T10:30:00+24:00", None),
    ("datetime", "2026-01-05T10:30:00+05:60", None),
    ("datetime", "2026-01-05T10:30:00.1234567Z", None),
    ("datetime", "2026-01-05T10:30:00,123Z", None),
    ("datetime", "2026-01-05T24:00:00Z", None),
    ("datetime", "2026-01-05T10:60:00Z", None),
    ("datetime", "2026-01-05T10:30:60Z", None),
    ("datetime", "2026-01-05\u202810:30:00Z", None),
    ("time", "14", "14:00:00"),
    ("time", "14:30", "14:30:00"),
    ("time", "14:30:45.123456Z", "14:30:45.123456+00:00"),
    ("time", "14:30:45.1+05:30", "14:30:45.100000+05:30"),
    ("time", "143045", None),
    ("time", "14:30:45+0530", None),
    ("time", "14:30:45+05:60", None),
    ("time", "24:30:00", None),
    ("time", "14:30:45.1234567", None),
]


@pytest.mark.parametrize("kind,value,expected", ISO_PROFILE_CASES)
def test_shared_iso_profile_has_interpreter_independent_conformance(
    kind, value, expected
):
    parser = {
        "date": parse_iso_date,
        "datetime": parse_iso_datetime,
        "time": parse_iso_time,
    }[kind]
    if expected is None:
        with pytest.raises(ValueError) as error:
            parser(value)
        assert str(error.value) == "invalid_iso_temporal"
        assert error.value.__context__ is None
    else:
        assert parser(value).isoformat() == expected


@pytest.mark.parametrize("value", [None, True, 20260105, {}, [], "x" * 4096])
@pytest.mark.parametrize("parser", [parse_iso_date, parse_iso_datetime, parse_iso_time])
def test_shared_iso_parser_rejects_nonstrings_and_excessive_values_without_echo(
    parser, value
):
    with pytest.raises(ValueError, match="^invalid_iso_temporal$"):
        parser(value)


def _direct_iso_accesses(source: str) -> tuple[int, ...]:
    return tuple(
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr == "fromisoformat"
    )


def _assert_shared_iso_parsing(sources: dict[str, str]) -> None:
    offenders = [
        (module, lines)
        for module, source in sorted(sources.items())
        if (lines := _direct_iso_accesses(source))
    ]
    assert offenders == []


def test_clinical_code_uses_the_shared_iso_profile_instead_of_interpreter_grammar():
    _assert_shared_iso_parsing(
        {
            str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "openmed/clinical").rglob("*.py"))
        }
    )


def test_direct_iso_lint_detects_a_synthetic_alias_and_bound_method():
    with pytest.raises(AssertionError, match="synthetic.clinical"):
        _assert_shared_iso_parsing(
            {
                "synthetic.clinical": "from datetime import datetime as dt\n"
                "def parse(value):\n"
                "    raw_parser = dt.fromisoformat\n"
                "    return raw_parser(value)\n"
            }
        )
    assert (
        _direct_iso_accesses(
            "from openmed.core.iso_temporal import parse_iso_datetime\n"
        )
        == ()
    )


@pytest.mark.parametrize(
    "value", ["20260105", "2026-W02-1", "2026-005", "2026-01-05T10:00:00+0530"]
)
def test_reference_times_refuse_basic_week_ordinal_and_compact_offsets(value):
    with pytest.raises(ValueError) as error:
        normalize_temporal("yesterday", [(0, 9)], value)
    assert value not in str(error.value)


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


def test_two_digit_year_does_not_guess_a_century() -> None:
    record = normalize_temporal("12/31/99", [(0, 8)], None)[0]

    assert record.value is None
    assert record.granularity_flags == ("day", "ambiguous")


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


@pytest.mark.parametrize(
    ("expression", "flags"),
    [
        ("q0h", ("hour", "ambiguous")),
        ("q5h x1 day", ("hour", "bounded", "ambiguous")),
    ],
)
def test_invalid_or_inexact_sets_do_not_emit_unbounded_values(
    expression: str,
    flags: tuple[str, ...],
) -> None:
    record = normalize_temporal(expression, [(0, len(expression))], None)[0]

    assert record.timex_type == "SET"
    assert record.value is None
    assert record.granularity_flags == flags


@pytest.mark.parametrize(
    ("expression", "value", "flags"),
    [
        ("2026-06-15T08:45:00", "2026-06-15T08:45:00", ("second",)),
        ("2026-06-15T08:45", "2026-06-15T08:45:00", ("minute",)),
        (
            "2026-06-15T08:45+02:00",
            "2026-06-15T08:45:00+02:00",
            ("minute",),
        ),
        ("2026-06-15t08:45", "2026-06-15T08:45:00", ("minute",)),
        ("2026-02-30T08:45", None, ("minute", "ambiguous")),
    ],
)
def test_iso_datetime_granularity_comes_from_source_precision(
    expression: str,
    value: str | None,
    flags: tuple[str, ...],
) -> None:
    record = normalize_temporal(expression, [(0, len(expression))], None)[0]

    assert record.value == value
    assert record.granularity_flags == flags


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
