"""Arithmetic calendar converters and their registry.

Reference dates below are literals computed independently of OpenMed: Solar
Hijri pairs agree with both ``jdatetime`` (33-year arithmetic) and
``convertdate.persian`` (astronomical equinox), Ethiopian pairs with
``convertdate.coptic`` shifted by 276 years, and well-known anchors such as
Nowruz and the Ethiopian millennium (1 Meskerem 2000 = 12 September 2007).
"""

from __future__ import annotations

from datetime import date

import pytest

from openmed.core import calendars
from openmed.core.calendars import (
    ETHIOPIAN,
    GREGORIAN,
    SOLAR_HIJRI,
    THAI_BUDDHIST,
    CalendarConverter,
    calendar_codes,
    get_calendar,
    register_calendar,
)
from openmed.core.errors import ConfigurationError, InputError

SOLAR_HIJRI_REFERENCES = [
    ((1277, 1, 1), date(1898, 3, 21)),
    ((1357, 11, 22), date(1979, 2, 11)),
    ((1370, 12, 30), date(1992, 3, 20)),
    ((1375, 12, 30), date(1997, 3, 20)),
    ((1399, 12, 30), date(2021, 3, 20)),
    ((1400, 1, 1), date(2021, 3, 21)),
    ((1403, 12, 30), date(2025, 3, 20)),
    ((1404, 1, 1), date(2025, 3, 21)),
    ((1404, 2, 28), date(2025, 5, 18)),
    ((1404, 2, 31), date(2025, 5, 21)),
    ((1404, 7, 30), date(2025, 10, 22)),
    ((1404, 12, 29), date(2026, 3, 20)),
    ((1407, 12, 29), date(2029, 3, 19)),
    ((1408, 12, 30), date(2030, 3, 20)),
    ((1501, 12, 29), date(2123, 3, 20)),
]
ETHIOPIAN_REFERENCES = [
    ((1891, 1, 1), date(1898, 9, 10)),
    ((1999, 13, 6), date(2007, 9, 11)),
    ((2000, 1, 1), date(2007, 9, 12)),
    ((2011, 13, 6), date(2019, 9, 11)),
    ((2015, 13, 6), date(2023, 9, 11)),
    ((2016, 1, 1), date(2023, 9, 12)),
    ((2016, 4, 29), date(2024, 1, 8)),
    ((2016, 13, 5), date(2024, 9, 10)),
    ((2017, 1, 1), date(2024, 9, 11)),
    ((2017, 7, 12), date(2025, 3, 21)),
    ((2114, 13, 5), date(2122, 9, 11)),
]
THAI_BUDDHIST_REFERENCES = [
    ((2484, 1, 1), date(1941, 1, 1)),
    ((2543, 2, 29), date(2000, 2, 29)),
    ((2567, 2, 29), date(2024, 2, 29)),
    ((2568, 3, 21), date(2025, 3, 21)),
    ((2665, 12, 31), date(2122, 12, 31)),
]
REFERENCES = [
    *((SOLAR_HIJRI, native, gregorian) for native, gregorian in SOLAR_HIJRI_REFERENCES),
    *((ETHIOPIAN, native, gregorian) for native, gregorian in ETHIOPIAN_REFERENCES),
    *(
        (THAI_BUDDHIST, native, gregorian)
        for native, gregorian in THAI_BUDDHIST_REFERENCES
    ),
]
NATIVE_CALENDARS = (SOLAR_HIJRI, ETHIOPIAN, THAI_BUDDHIST)


@pytest.mark.parametrize(("code", "native", "gregorian"), REFERENCES)
def test_converters_match_independent_reference_dates(code, native, gregorian):
    converter = get_calendar(code)

    assert converter.to_day_number(*native) == gregorian.toordinal()
    assert converter.from_day_number(gregorian.toordinal()) == native


@pytest.mark.parametrize("code", NATIVE_CALENDARS)
def test_every_day_in_the_supported_range_round_trips_in_order(code):
    converter = get_calendar(code)
    first, last = converter.day_number_range()
    previous = converter.from_day_number(first)

    assert previous == (converter.min_year, 1, 1)
    assert converter.to_day_number(*previous) == first
    for day_number in range(first + 1, last + 1):
        current = converter.from_day_number(day_number)
        assert converter.to_day_number(*current) == day_number
        assert converter.is_valid(*current)
        year, month, day = previous
        if day < converter.month_length(year, month):
            expected = (year, month, day + 1)
        elif month < converter.months_in_year(year):
            expected = (year, month + 1, 1)
        else:
            expected = (year + 1, 1, 1)
        assert current == expected
        previous = current
    assert previous[0] == converter.max_year


def test_solar_hijri_leap_years_follow_the_33_year_rule_across_the_range():
    converter = get_calendar(SOLAR_HIJRI)
    leap_years = [
        year
        for year in range(converter.min_year, converter.max_year + 1)
        if converter.month_length(year, 12) == 30
    ]

    assert {1399, 1403, 1408} <= set(leap_years)
    assert not {1400, 1404, 1405, 1406, 1407} & set(leap_years)
    for first, second in zip(leap_years, leap_years[1:]):
        assert second - first in {4, 5}
    for year in range(converter.min_year, converter.max_year):
        length = converter.to_day_number(year + 1, 1, 1) - converter.to_day_number(
            year, 1, 1
        )
        assert length == (366 if year in leap_years else 365)


@pytest.mark.parametrize(
    ("code", "valid", "invalid"),
    [
        (SOLAR_HIJRI, (1403, 12, 30), (1404, 12, 30)),
        (SOLAR_HIJRI, (1404, 6, 31), (1404, 7, 31)),
        (ETHIOPIAN, (2015, 13, 6), (2016, 13, 6)),
        (ETHIOPIAN, (2016, 12, 30), (2016, 12, 31)),
        (ETHIOPIAN, (2016, 13, 5), (2016, 14, 1)),
        (THAI_BUDDHIST, (2567, 2, 29), (2566, 2, 29)),
        (THAI_BUDDHIST, (2543, 2, 29), (2643, 2, 29)),
    ],
)
def test_month_lengths_and_leap_days_are_validated(code, valid, invalid):
    converter = get_calendar(code)

    assert converter.is_valid(*valid)
    assert not converter.is_valid(*invalid)


@pytest.mark.parametrize(
    ("code", "outside"),
    [
        (SOLAR_HIJRI, (1276, 6, 1)),
        (SOLAR_HIJRI, (1502, 1, 1)),
        (ETHIOPIAN, (1890, 13, 5)),
        (ETHIOPIAN, (2115, 1, 1)),
        (THAI_BUDDHIST, (2483, 6, 1)),
        (THAI_BUDDHIST, (2666, 1, 1)),
    ],
)
def test_out_of_range_dates_raise_without_echoing_values(code, outside):
    converter = get_calendar(code)

    assert not converter.is_valid(*outside)
    with pytest.raises(InputError) as excinfo:
        converter.to_day_number(*outside)

    message = str(excinfo.value)
    assert code in message
    assert str(outside[0]) not in message
    assert excinfo.value.details["reason"] == "invalid_date"


@pytest.mark.parametrize("code", NATIVE_CALENDARS)
def test_day_numbers_outside_the_range_raise_and_shift_checks_bounds(code):
    converter = get_calendar(code)
    first, last = converter.day_number_range()

    for outside in (first - 1, last + 1):
        with pytest.raises(InputError) as excinfo:
            converter.from_day_number(outside)
        assert str(outside) not in str(excinfo.value)
    last_date = converter.from_day_number(last)
    with pytest.raises(InputError):
        converter.shift(*last_date, 1)
    assert converter.shift(*last_date, -1) == converter.from_day_number(last - 1)


@pytest.mark.parametrize("bad", [True, 1.0, "1404"])
def test_non_integer_components_are_never_valid(bad):
    converter = get_calendar(SOLAR_HIJRI)

    assert not converter.is_valid(bad, 1, 1)
    assert not converter.is_valid(1404, bad, 1)
    with pytest.raises(InputError):
        converter.shift(1404, 1, 1, bad)


def test_gregorian_converter_matches_datetime():
    converter = get_calendar(GREGORIAN)

    assert converter.to_day_number(2024, 2, 29) == date(2024, 2, 29).toordinal()
    assert converter.from_day_number(date(1900, 3, 1).toordinal()) == (1900, 3, 1)
    assert not converter.is_valid(1900, 2, 29)


def test_registry_lists_builtin_calendars():
    assert {GREGORIAN, SOLAR_HIJRI, ETHIOPIAN, THAI_BUDDHIST} <= set(calendar_codes())
    assert get_calendar("hebrew") is None
    assert get_calendar(None) is None


class _TableCalendar(CalendarConverter):
    """Toy table-driven calendar with two years of uneven month lengths."""

    code = "test_table"
    min_year = 1
    max_year = 2
    _months = {1: (29, 32), 2: (31, 30)}
    _epoch = date(2030, 1, 1).toordinal()

    def months_in_year(self, year):
        return 2

    def month_length(self, year, month):
        return self._months[year][month - 1]

    def _day_number(self, year, month, day):
        offset = sum(sum(self._months[item]) for item in range(1, year))
        offset += sum(self._months[year][: month - 1])
        return self._epoch + offset + day - 1

    def _date_from_day_number(self, day_number):
        remaining = day_number - self._epoch
        for year in (1, 2):
            for month, length in enumerate(self._months[year], start=1):
                if remaining < length:
                    return year, month, remaining + 1
                remaining -= length
        raise AssertionError("day number outside the toy table")


def test_registering_a_table_calendar_extends_shifting(monkeypatch):
    from openmed.core.calendar_dates import shift_calendar_date

    monkeypatch.setattr(calendars, "_REGISTRY", dict(calendars._REGISTRY))
    register_calendar(_TableCalendar())

    outcome = shift_calendar_date("0001/02/32", 1, lang="en", calendar="test_table")

    assert outcome.text == "0002/01/01"
    assert outcome.calendar == "test_table"
    with pytest.raises(ConfigurationError, match="already registered"):
        register_calendar(_TableCalendar())
    register_calendar(_TableCalendar(), replace=True)


def test_registry_rejects_malformed_converters(monkeypatch):
    monkeypatch.setattr(calendars, "_REGISTRY", dict(calendars._REGISTRY))

    class _BadCode(_TableCalendar):
        code = "Bad Code"

    class _BadRange(_TableCalendar):
        code = "test_bad_range"
        min_year = 3

    for converter in (object(), _BadCode(), _BadRange()):
        with pytest.raises(ConfigurationError):
            register_calendar(converter)


def test_language_packs_declare_default_calendars():
    from openmed.core import language_pack_catalog  # noqa: F401
    from openmed.core.language_pack import LanguagePack, get_language_pack

    assert get_language_pack("fa").default_calendar == SOLAR_HIJRI
    assert get_language_pack("am").default_calendar == ETHIOPIAN
    assert get_language_pack("th").default_calendar == THAI_BUDDHIST
    assert get_language_pack("en").default_calendar == GREGORIAN
    assert get_language_pack("ar").default_calendar == GREGORIAN

    with pytest.raises(ValueError, match="registered calendar"):
        LanguagePack(
            code="eo",
            scripts=("Latin",),
            default_model="/models/esperanto-pii",
            segmenter_id="unicode-sentence",
            recognizers=("builtin-patterns",),
            surrogate_locale="en_US",
            default_calendar="lunar_unknown",
        )
