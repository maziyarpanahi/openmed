"""Focused tests for conservative clinical temporal intervals (OM-984a)."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import normalize_temporal_intervals as package_normalize
from openmed.clinical.temporal_intervals import (
    TemporalInterval,
    TemporalIntervalNormalizer,
    normalize_temporal_interval,
    normalize_temporal_intervals,
)


def _one(text: str, expression: str) -> TemporalInterval:
    start = text.index(expression)
    return normalize_temporal_interval(text, (start, start + len(expression)))


def test_dates_keep_precision_and_do_not_guess_numeric_order() -> None:
    text = "2026-04; April 5, 2026; 03/04/2026"

    month, named_day, ambiguous = normalize_temporal_intervals(
        text,
        [
            (0, 7),
            (8, 22),
            (24, len(text)),
        ],
    )

    assert month.value == "2026-04"
    assert month.precision == "month"
    assert month.timezone_state == "not_applicable"
    assert named_day.value == "2026-04-05"
    assert named_day.precision == "day"
    assert ambiguous.value is None
    assert ambiguous.status == "conflicting"
    assert ambiguous.conflicts == ("date_order",)


def test_times_expose_timezone_state_without_assuming_local_timezone() -> None:
    text = "14:30; 14:30:05+02:00; 2 PM"

    naive, explicit, hour = normalize_temporal_intervals(
        text,
        [
            (text.index("14:30"), text.index("14:30") + len("14:30")),
            (
                text.index("14:30:05"),
                text.index("14:30:05") + len("14:30:05+02:00"),
            ),
            (text.index("2 PM"), len(text)),
        ],
    )

    assert naive.value == "14:30"
    assert naive.precision == "minute"
    assert naive.timezone_state == "unknown"
    assert naive.unknown_components == ("timezone",)
    assert explicit.value == "14:30:05+02:00"
    assert explicit.precision == "second"
    assert explicit.timezone_state == "explicit"
    assert hour.value == "14"
    assert hour.precision == "hour"


def test_durations_normalize_iso_and_prose_components() -> None:
    prose = _one("for 2 hours and 30 minutes", "for 2 hours and 30 minutes")
    iso = _one("P3D", "P3D")

    assert prose.kind == "duration"
    assert prose.value == "PT2H30M"
    assert prose.precision == "minute"
    assert prose.timezone_state == "not_applicable"
    assert iso.value == "P3D"
    assert iso.precision == "day"


def test_open_ended_intervals_never_substitute_the_current_time() -> None:
    since = _one("since 2024-01-01", "since 2024-01-01")
    present = _one("2024-01-01 to present", "2024-01-01 to present")
    from_present = _one("from 2024-02 to present", "from 2024-02 to present")
    open_start = _one("../2024-02", "../2024-02")

    assert since.value == "2024-01-01/.."
    assert since.start is not None and since.start.value == "2024-01-01"
    assert since.end is None
    assert since.open_end is True
    assert present.value == "2024-01-01/.."
    assert present.open_end is True
    assert from_present.value == "2024-02/.."
    assert from_present.open_end is True
    assert from_present.end_inclusive is False
    assert open_start.value == "../2024-02"
    assert open_start.open_start is True
    assert open_start.precision == "month"


def test_interval_conflicts_and_unknown_components_are_explicit() -> None:
    reversed_interval = _one("2024-03-01/2024-02-01", "2024-03-01/2024-02-01")
    mixed_kinds = _one("2024-01-01 to 12:00", "2024-01-01 to 12:00")
    malformed = _one("2024-02-30", "2024-02-30")

    assert reversed_interval.value is None
    assert reversed_interval.conflicts == ("interval_order",)
    assert mixed_kinds.value is None
    assert mixed_kinds.conflicts == ("endpoint_kind",)
    assert malformed.value is None
    assert malformed.status == "unknown"
    assert "day" in malformed.unknown_components


def test_offsets_are_preserved_and_serialization_has_no_raw_source_text() -> None:
    text = "Synthetic patient note: since 2024-01-01"
    record = _one(text, "since 2024-01-01")

    payload = record.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert record.span == (24, len(text))
    assert payload["source_span"] == [24, len(text)]
    assert "Synthetic patient note" not in encoded
    assert "since 2024-01-01" not in encoded
    assert "source_span" in payload


def test_normalization_is_deterministic_offline_and_wrapper_is_stateless(
    caplog,
) -> None:
    text = "2026-01-01 to present"
    normalizer = TemporalIntervalNormalizer()

    first = normalizer.normalize(text)
    second = normalizer(text)

    assert first == second
    assert first[0].to_dict() == second[0].to_dict()
    assert caplog.records == []
    assert package_normalize(text)[0] == first[0]


@pytest.mark.parametrize(
    "span",
    [
        (-1, 2),
        (0, 0),
        (0, 4),
        ("bad", 2),
        (0,),
    ],
)
def test_invalid_spans_are_rejected(span) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_temporal_intervals("abc", [span])
