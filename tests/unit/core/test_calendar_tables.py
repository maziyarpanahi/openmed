"""Table-driven Umm al-Qura Hijri and Bikram Sambat converters and their loader.

Reference dates below are literals computed outside OpenMed. Hijri pairs agree
with ``hijridate`` 2.6.0 and with either ICU 77.1 ``islamic-umalqura`` (1420
AH onward) or R.H. van Gent's Umm al-Qura table (before 1420 AH). Bikram
Sambat pairs agree with ``nepali-datetime`` 1.0.8.5, ``nepali`` 1.2.0 and
``pyBSDate`` 0.3.2, except where a comment names the two that agree. Tables
used to exercise the loader are synthetic.
"""

from __future__ import annotations

import copy
import json
from datetime import date

import pytest

from openmed.core import calendars
from openmed.core.calendar_tables import (
    BUNDLED_CALENDAR_TABLES,
    PERMISSIVE_TABLE_LICENSES,
    TableCalendar,
    bundled_calendar_table,
    calendar_table_digest,
    load_calendar_table,
    verify_calendar_table,
)
from openmed.core.calendars import (
    BIKRAM_SAMBAT,
    LUNAR_HIJRI,
    get_calendar,
    register_calendar,
)
from openmed.core.errors import ConfigurationError, InputError

HIJRI_REFERENCES = [
    ((1356, 1, 1), date(1937, 3, 14)),
    ((1364, 8, 28), date(1945, 8, 7)),
    ((1364, 9, 1), date(1945, 8, 8)),
    ((1400, 1, 1), date(1979, 11, 20)),
    ((1420, 1, 1), date(1999, 4, 17)),
    ((1427, 6, 1), date(2006, 6, 27)),
    ((1440, 9, 1), date(2019, 5, 6)),
    ((1445, 8, 19), date(2024, 2, 29)),
    ((1446, 9, 1), date(2025, 3, 1)),
    ((1446, 9, 21), date(2025, 3, 21)),
    ((1446, 10, 1), date(2025, 3, 30)),
    ((1447, 1, 1), date(2025, 6, 26)),
    ((1447, 2, 29), date(2025, 8, 23)),
    ((1447, 3, 1), date(2025, 8, 24)),
    ((1447, 9, 1), date(2026, 2, 18)),
    ((1447, 9, 30), date(2026, 3, 19)),
    ((1450, 12, 29), date(2029, 5, 13)),
]
BIKRAM_SAMBAT_REFERENCES = [
    ((2000, 1, 1), date(1943, 4, 14)),
    ((2035, 9, 1), date(1978, 12, 16)),
    ((2047, 3, 1), date(1990, 6, 15)),
    ((2062, 1, 30), date(2005, 5, 13)),
    # nepali and pyBSDate agree; nepali-datetime has a 31-day Baisakh 2062.
    ((2062, 2, 1), date(2005, 5, 14)),
    ((2079, 10, 1), date(2023, 1, 15)),
    ((2080, 11, 17), date(2024, 2, 29)),
    ((2081, 1, 1), date(2024, 4, 13)),
    ((2081, 12, 8), date(2025, 3, 21)),
    ((2082, 1, 1), date(2025, 4, 14)),
    ((2082, 3, 30), date(2025, 7, 14)),
    ((2082, 3, 32), date(2025, 7, 16)),
    ((2082, 4, 1), date(2025, 7, 17)),
    ((2082, 9, 29), date(2026, 1, 13)),
    ((2083, 1, 1), date(2026, 4, 14)),
    ((2083, 12, 30), date(2027, 4, 13)),
]
REFERENCES = [
    *((LUNAR_HIJRI, native, gregorian) for native, gregorian in HIJRI_REFERENCES),
    *(
        (BIKRAM_SAMBAT, native, gregorian)
        for native, gregorian in BIKRAM_SAMBAT_REFERENCES
    ),
]
TABLE_CALENDARS = (LUNAR_HIJRI, BIKRAM_SAMBAT)
GPL_LICENSE = "GPL-3.0-only"


@pytest.mark.parametrize(("code", "native", "gregorian"), REFERENCES)
def test_table_calendars_match_independent_reference_dates(code, native, gregorian):
    converter = get_calendar(code)

    assert converter.to_day_number(*native) == gregorian.toordinal()
    assert converter.from_day_number(gregorian.toordinal()) == native


@pytest.mark.parametrize(
    ("code", "years", "span"),
    [
        (LUNAR_HIJRI, (1356, 1450), (date(1937, 3, 14), date(2029, 5, 13))),
        (BIKRAM_SAMBAT, (2000, 2083), (date(1943, 4, 14), date(2027, 4, 13))),
    ],
)
def test_bundled_tables_declare_their_bounded_ranges(code, years, span):
    converter = get_calendar(code)
    first, last = converter.day_number_range()

    assert isinstance(converter, TableCalendar)
    assert (converter.min_year, converter.max_year) == years
    assert (date.fromordinal(first), date.fromordinal(last)) == span


@pytest.mark.parametrize("code", TABLE_CALENDARS)
def test_every_day_in_the_table_round_trips_in_order(code):
    converter = get_calendar(code)
    first, last = converter.day_number_range()
    previous = converter.from_day_number(first)

    assert previous == (converter.min_year, 1, 1)
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


def test_month_lengths_cover_29_to_32_days():
    hijri = get_calendar(LUNAR_HIJRI)
    bikram_sambat = get_calendar(BIKRAM_SAMBAT)
    hijri_lengths = [
        hijri.month_length(year, month)
        for year in range(hijri.min_year, hijri.max_year + 1)
        for month in range(1, 13)
    ]
    bs_lengths = {
        bikram_sambat.month_length(year, month)
        for year in range(bikram_sambat.min_year, bikram_sambat.max_year + 1)
        for month in range(1, 13)
    }

    assert bs_lengths == {29, 30, 31, 32}
    # Both Umm al-Qura tabulations print Shaban 1364 with 28 days.
    assert set(hijri_lengths) == {28, 29, 30}
    assert hijri_lengths.count(28) == 1
    assert hijri.month_length(1364, 8) == 28
    for year in range(bikram_sambat.min_year, bikram_sambat.max_year + 1):
        year_length = sum(bikram_sambat.month_length(year, m) for m in range(1, 13))
        assert year_length in {365, 366}


@pytest.mark.parametrize(
    ("code", "valid", "invalid"),
    [
        (LUNAR_HIJRI, (1447, 2, 29), (1447, 2, 30)),
        (LUNAR_HIJRI, (1447, 9, 30), (1447, 13, 1)),
        (BIKRAM_SAMBAT, (2082, 3, 32), (2082, 4, 32)),
        (BIKRAM_SAMBAT, (2082, 8, 29), (2082, 8, 30)),
    ],
)
def test_table_month_lengths_are_validated(code, valid, invalid):
    converter = get_calendar(code)

    assert converter.is_valid(*valid)
    assert not converter.is_valid(*invalid)


@pytest.mark.parametrize(
    ("code", "outside"),
    [
        (LUNAR_HIJRI, (1355, 12, 1)),
        (LUNAR_HIJRI, (1451, 1, 1)),
        (BIKRAM_SAMBAT, (1999, 12, 1)),
        (BIKRAM_SAMBAT, (2084, 1, 1)),
    ],
)
def test_dates_outside_the_table_raise_without_echoing_values(code, outside):
    converter = get_calendar(code)
    first, last = converter.day_number_range()

    assert not converter.is_valid(*outside)
    with pytest.raises(InputError) as excinfo:
        converter.to_day_number(*outside)
    message = str(excinfo.value)
    assert code in message
    assert str(outside[0]) not in message
    for day_number in (first - 1, last + 1):
        with pytest.raises(InputError) as excinfo:
            converter.from_day_number(day_number)
        assert str(day_number) not in str(excinfo.value)


@pytest.mark.parametrize("code", TABLE_CALENDARS)
def test_bundled_tables_carry_pinned_digests_and_permissive_licenses(code):
    from openmed.core import calendar_tables

    name, pinned = BUNDLED_CALENDAR_TABLES[code]
    path = calendar_tables._DATA_DIR / name
    document = json.loads(path.read_text(encoding="utf-8"))
    converter = bundled_calendar_table(code)

    assert document["sha256"] == pinned == calendar_table_digest(document)
    assert converter.table.sha256 == pinned
    assert document["license"] in PERMISSIVE_TABLE_LICENSES
    assert document["license"] == "MIT"
    assert document["copyright"].startswith("Copyright (c)")
    assert "Permission is hereby granted" in document["license_text"]
    assert document["source"]["project"]
    assert document["source"]["repository"].startswith("https://github.com/")
    assert len(document["verification"]) == 2
    assert converter.code == code
    assert (
        converter.variant
        == {
            LUNAR_HIJRI: "umm_al_qura",
            BIKRAM_SAMBAT: "nepal",
        }[code]
    )
    assert get_calendar(code).table.sha256 == pinned


def test_bundled_table_lookup_rejects_unknown_codes():
    with pytest.raises(ConfigurationError) as excinfo:
        bundled_calendar_table("hijri_diyanet")

    assert excinfo.value.details["reason"] == "unknown_table"


def _synthetic_table(**overrides):
    document = {
        "format": "openmed-calendar-table",
        "format_version": 1,
        "calendar": "hijri_test_variant",
        "variant": "test_variant",
        "version": "0.0.1",
        "license": "CC0-1.0",
        "source": {"project": "synthetic unit-test table"},
        "first_year": 1447,
        "first_day": "2025-06-25",
        "month_lengths": [
            [29, 30, 29, 30, 29, 30, 29, 30, 29, 30, 29, 30],
            [30, 29, 30, 29, 30, 29, 30, 29, 30, 29, 30, 29],
        ],
    }
    document.update(overrides)
    document["sha256"] = calendar_table_digest(document)
    return document


def _reason(excinfo) -> str:
    return excinfo.value.details["reason"]


def test_loader_accepts_a_caller_supplied_variant_table(tmp_path):
    document = _synthetic_table()
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    from_file = load_calendar_table(path, expected_sha256=document["sha256"])
    from_mapping = load_calendar_table(document)

    assert from_file.code == from_mapping.code == "hijri_test_variant"
    assert from_file.variant == "test_variant"
    assert (from_file.min_year, from_file.max_year) == (1447, 1448)
    assert from_file.to_day_number(1447, 1, 1) == date(2025, 6, 25).toordinal()
    assert from_file.from_day_number(date(2025, 6, 25).toordinal() + 29) == (
        1447,
        2,
        1,
    )
    assert from_file.table.source["project"] == "synthetic unit-test table"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda doc: doc["month_lengths"][0].__setitem__(0, 30), "digest_mismatch"),
        (lambda doc: doc.__setitem__("license", "MIT"), "digest_mismatch"),
        (lambda doc: doc.__setitem__("sha256", "0" * 64), "digest_mismatch"),
        (lambda doc: doc.__setitem__("sha256", "not-a-digest"), "invalid_digest"),
        (lambda doc: doc.pop("sha256"), "missing_field"),
    ],
)
def test_loader_rejects_tables_that_do_not_match_their_digest(mutate, reason):
    document = copy.deepcopy(_synthetic_table())
    mutate(document)

    with pytest.raises(ConfigurationError) as excinfo:
        load_calendar_table(document)

    assert _reason(excinfo) == reason


def test_loader_enforces_a_pinned_digest():
    document = _synthetic_table()

    with pytest.raises(ConfigurationError) as excinfo:
        load_calendar_table(document, expected_sha256="f" * 64)

    assert _reason(excinfo) == "digest_mismatch"


def test_loader_requires_a_permissive_license_unless_the_caller_opts_in():
    restricted = _synthetic_table(license=GPL_LICENSE)
    unlicensed = _synthetic_table(license=" ")

    with pytest.raises(ConfigurationError) as excinfo:
        load_calendar_table(restricted)
    assert _reason(excinfo) == "non_permissive_license"
    assert GPL_LICENSE not in str(excinfo.value)
    with pytest.raises(ConfigurationError) as excinfo:
        load_calendar_table(unlicensed, allow_non_permissive_license=True)
    assert _reason(excinfo) == "missing_license"

    converter = load_calendar_table(restricted, allow_non_permissive_license=True)
    assert converter.table.license == GPL_LICENSE


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"source": {}}, "missing_lineage"),
        ({"source": {"project": " "}}, "missing_lineage"),
        ({"source": "synthetic"}, "missing_lineage"),
        ({"format": "other-table"}, "unsupported_format"),
        ({"format_version": 2}, "unsupported_format"),
        ({"format_version": True}, "unsupported_format"),
        ({"calendar": "Hijri Variant"}, "invalid_calendar"),
        ({"variant": "Umm al-Qura"}, "invalid_variant"),
        ({"version": ""}, "invalid_version"),
        ({"first_year": 0}, "invalid_year"),
        ({"first_year": "1447"}, "invalid_year"),
        ({"first_day": "2025-02-30"}, "invalid_first_day"),
        ({"first_day": 739000}, "invalid_first_day"),
        ({"month_lengths": []}, "invalid_month_lengths"),
        ({"month_lengths": [[0, 30]]}, "invalid_month_lengths"),
        ({"month_lengths": [[36]]}, "invalid_month_lengths"),
        ({"month_lengths": [[30] * 15]}, "invalid_month_lengths"),
        ({"month_lengths": [[30, True]]}, "invalid_month_lengths"),
        ({"month_lengths": [30, 29]}, "invalid_month_lengths"),
        ({"first_day": "9999-12-01"}, "out_of_range"),
        ({"copyright": 7}, "invalid_field"),
        ({"verification": "checked"}, "invalid_field"),
        ({"comment": "unexpected"}, "unknown_field"),
    ],
)
def test_loader_rejects_malformed_tables_without_echoing_values(overrides, reason):
    document = _synthetic_table(**overrides)

    with pytest.raises(ConfigurationError) as excinfo:
        load_calendar_table(document)

    assert _reason(excinfo) == reason
    message = str(excinfo.value)
    for value in overrides.values():
        if isinstance(value, str) and value.strip():
            assert value not in message


def test_loader_rejects_unreadable_and_non_json_files_without_paths(tmp_path):
    missing = tmp_path / "patient-calendar-secret.json"
    broken = tmp_path / "broken-calendar.json"
    broken.write_text("{not json", encoding="utf-8")
    listed = tmp_path / "list-calendar.json"
    listed.write_text("[]", encoding="utf-8")

    for path, reason in (
        (missing, "unreadable"),
        (broken, "not_json"),
        (listed, "not_object"),
    ):
        with pytest.raises(ConfigurationError) as excinfo:
            load_calendar_table(path)
        assert _reason(excinfo) == reason
        assert str(tmp_path) not in str(excinfo.value)
        assert path.name not in str(excinfo.value)
        assert excinfo.value.__cause__ is None


def test_verify_rejects_documents_that_are_not_json_data():
    document = _synthetic_table()
    document["source"] = {"project": "synthetic", "retrieved": date(2026, 1, 1)}

    with pytest.raises(ConfigurationError) as excinfo:
        verify_calendar_table(document)

    assert _reason(excinfo) == "not_json"


def test_registering_a_caller_table_shifts_spans_tagged_with_its_code(monkeypatch):
    from openmed.core.calendar_dates import UNSUPPORTED_CALENDAR, shift_calendar_date

    monkeypatch.setattr(calendars, "_REGISTRY", dict(calendars._REGISTRY))
    before = shift_calendar_date(
        "29/01/1447", 1, lang="ar", calendar="hijri_test_variant"
    )
    register_calendar(load_calendar_table(_synthetic_table()))
    after = shift_calendar_date(
        "29/01/1447", 1, lang="ar", calendar="hijri_test_variant"
    )

    assert before.reason == UNSUPPORTED_CALENDAR
    assert before.text is None
    # The synthetic variant gives Muharram 29 days; Umm al-Qura gives it 30.
    assert after.text == "01/02/1447"
    assert shift_calendar_date("29/01/1447", 1, lang="ar").text == "30/01/1447"


def test_replacing_the_hijri_table_changes_marked_dates(monkeypatch):
    from openmed.core.calendar_dates import shift_calendar_date

    monkeypatch.setattr(calendars, "_REGISTRY", dict(calendars._REGISTRY))
    variant = load_calendar_table(_synthetic_table(calendar=LUNAR_HIJRI))

    assert shift_calendar_date("29/01/1447 هـ", 1, lang="ar").text == "30/01/1447 هـ"
    register_calendar(variant, replace=True)
    assert shift_calendar_date("29/01/1447 هـ", 1, lang="ar").text == "01/02/1447 هـ"
    # Years outside the replacement table now fall back to the placeholder.
    assert shift_calendar_date("15/09/1446 هـ", 1, lang="ar").text is None
