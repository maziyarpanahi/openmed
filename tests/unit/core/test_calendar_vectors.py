"""Pinned calendar conversion vectors shared with ports of the calendars.

``tests/fixtures/parity/calendar_conversion_v1.json`` lists, for every
registered calendar, dates near year starts, leap days and the edges of the
supported range with their proleptic Gregorian dates. The Gregorian dates are
literals from well-known anchors, not output of the converters, so a port can
reuse the file as its conformance vectors.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from openmed.core.calendars import calendar_codes, get_calendar
from openmed.core.errors import InputError

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "parity"
    / "calendar_conversion_v1.json"
)
VECTORS = json.loads(FIXTURE.read_text(encoding="utf-8"))["calendars"]
TABLE_CALENDARS = ("bikram_sambat", "hijri")

CASES = [
    pytest.param(code, case, id=f"{code}-{'-'.join(map(str, case['native']))}")
    for code, entry in sorted(VECTORS.items())
    for case in entry["cases"]
]


def test_fixture_covers_exactly_the_registered_calendars():
    assert sorted(VECTORS) == list(calendar_codes())


@pytest.mark.parametrize("code", sorted(VECTORS))
def test_every_calendar_has_at_least_five_cases_and_a_shift(code):
    assert len(VECTORS[code]["cases"]) >= 5
    assert VECTORS[code]["shifts"]


@pytest.mark.parametrize(("code", "case"), CASES)
def test_cases_convert_in_both_directions(code, case):
    calendar = get_calendar(code)
    native = tuple(case["native"])
    day_number = date.fromisoformat(case["gregorian"]).toordinal()

    assert case["day_number"] == day_number
    assert calendar.to_day_number(*native) == day_number
    assert calendar.from_day_number(day_number) == native


@pytest.mark.parametrize("code", sorted(VECTORS))
def test_shifts_land_on_the_expected_native_date(code):
    calendar = get_calendar(code)
    for shift in VECTORS[code]["shifts"]:
        assert calendar.shift(*shift["from"], shift["days"]) == tuple(shift["to"])


@pytest.mark.parametrize("code", sorted(VECTORS))
def test_cases_include_the_first_and_last_supported_dates(code):
    calendar = get_calendar(code)
    first, last = calendar.day_number_range()
    day_numbers = {case["day_number"] for case in VECTORS[code]["cases"]}

    assert first in day_numbers
    assert last in day_numbers


@pytest.mark.parametrize("code", TABLE_CALENDARS)
def test_table_calendar_edges_are_pinned_to_the_range_boundaries(code):
    calendar = get_calendar(code)
    first, last = calendar.day_number_range()
    natives = {tuple(case["native"]) for case in VECTORS[code]["cases"]}

    assert calendar.from_day_number(first)[0] == calendar.min_year
    assert calendar.from_day_number(last)[0] == calendar.max_year
    assert calendar.from_day_number(first) in natives
    assert calendar.from_day_number(last) in natives
    with pytest.raises(InputError):
        calendar.from_day_number(first - 1)
    with pytest.raises(InputError):
        calendar.from_day_number(last + 1)
