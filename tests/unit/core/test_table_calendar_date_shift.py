"""Shifting and synthesizing Umm al-Qura Hijri and Bikram Sambat dates.

Expected outputs are literals that follow from the independent reference dates
in ``test_calendar_tables.py``: Safar 1447 has 29 days, Ramadan 1447 has 30,
and Bikram Sambat 2082 has months of 31, 31, 32, 31, 31, 31, 30, 29, 30, 29,
30 and 30 days. Every input is synthetic.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core.calendar_dates import (
    AMBIGUOUS,
    CALENDAR_METADATA_KEY,
    INVALID_DATE,
    LUNAR_HIJRI_MARKER,
    OUT_OF_RANGE,
    SHIFTED,
    UNSUPPORTED_CALENDAR,
    calendar_date_surrogate,
    default_calendar_for,
    shift_calendar_date,
)
from openmed.core.calendars import BIKRAM_SAMBAT, GREGORIAN, LUNAR_HIJRI, get_calendar
from openmed.core.date_shift import stable_offset_for
from openmed.core.pii import _shift_date
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult

HMAC_TEST_MATERIAL = bytes(range(11, 43))
PLACEHOLDER = "[DATE_SHIFTED]"
_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")
_HIJRI_ARABIC_MONTHS = (
    "محرم",
    "صفر",
    "ربيع الأول",
    "ربيع الآخر",
    "جمادى الأولى",
    "جمادى الآخرة",
    "رجب",
    "شعبان",
    "رمضان",
    "شوال",
    "ذو القعدة",
    "ذو الحجة",
)
_BS_COLLOQUIAL_MONTHS = (
    "बैशाख",
    "जेठ",
    "असार",
    "साउन",
    "भदौ",
    "असोज",
    "कात्तिक",
    "मंसिर",
    "पुस",
    "माघ",
    "फागुन",
    "चैत",
)


def _ar(text: str) -> str:
    return text.translate(_ARABIC_DIGITS)


def _fa(text: str) -> str:
    return text.translate(_PERSIAN_DIGITS)


def _ne(text: str) -> str:
    return text.translate(_DEVANAGARI_DIGITS)


def test_calendar_codes_match_month_name_detection():
    # Month-name detection tags spans with exactly these codes.
    assert LUNAR_HIJRI == LUNAR_HIJRI_MARKER == "hijri"
    assert BIKRAM_SAMBAT == "bikram_sambat"
    assert CALENDAR_METADATA_KEY == "calendar"


@pytest.mark.parametrize(
    ("value", "days", "lang", "calendar", "expected"),
    [
        # Issue examples: Safar 1447 has 29 days and Asar 2082 has 32.
        ("28/02/1447", 3, "ar", None, "02/03/1447"),
        ("29/02/1447", 3, "ar", None, "03/03/1447"),
        ("15/09/1447 هـ", 3, "ar", None, "18/09/1447 هـ"),
        ("2082/03/30", 3, "en", BIKRAM_SAMBAT, "2082/04/01"),
        ("2082/03/32", 3, "en", BIKRAM_SAMBAT, "2082/04/03"),
        # 29-, 30-, 31- and 32-day Bikram Sambat months.
        ("2082/08/29 BS", 1, "en", None, "2082/09/01 BS"),
        ("2082/07/30 BS", 1, "en", None, "2082/08/01 BS"),
        ("2082/04/31 BS", 1, "en", None, "2082/05/01 BS"),
        ("2082/03/32 B.S.", 1, "en", None, "2082/04/01 B.S."),
        # 29- and 30-day Hijri months, year boundaries and negative offsets.
        ("29/02/1447 هـ", 1, "ar", None, "01/03/1447 هـ"),
        ("30/09/1447 AH", 1, "ar", None, "01/10/1447 AH"),
        ("29/12/1446 هـ", 1, "ar", None, "01/01/1447 هـ"),
        ("01/01/1447 هـ", -1, "ar", None, "29/12/1446 هـ"),
        ("2082/12/30 BS", 1, "en", None, "2083/01/01 BS"),
        ("2083/01/01 BS", -1, "en", None, "2082/12/30 BS"),
        # Source digits, separators, padding and era markers are kept.
        (_ar("15/09/1447 هـ"), 3, "ar", None, _ar("18/09/1447 هـ")),
        (_fa("1447-09-15 ه.ق"), 16, "fa", None, _fa("1447-10-01 ه.ق")),
        ("1447.9.5 هـ", 3, "ar", None, "1447.9.8 هـ"),
        (_ne("2082/03/32 वि.सं."), 3, "ne", None, _ne("2082/04/03 वि.सं.")),
        (_ne("2082-3-30"), 2, "ne", BIKRAM_SAMBAT, _ne("2082-3-32")),
        # Month names keep their spelling convention.
        (_ar("١٥ رمضان ١٤٤٧ هـ"), 20, "ar", None, _ar("٥ شوال ١٤٤٧ هـ")),
        (_fa("۱۵ رمضان ۱۴۴۷ ه.ق"), 50, "fa", None, _fa("۶ ذی‌القعده ۱۴۴۷ ه.ق")),
        (
            _fa("۱۵ ربیع‌الاول ۱۴۴۷"),
            40,
            "fa",
            None,
            _fa("۲۵ ربیع‌الثانی ۱۴۴۷"),
        ),
        ("1 ذوالقعدہ 1446ھ", 30, "ur", None, "2 ذوالحجہ 1446ھ"),
        ("1 ذو القعدة 1446 هـ", 30, "ar", None, "2 ذو الحجة 1446 هـ"),
        (_ne("२०८२ असार ३२ गते"), 1, "ne", None, _ne("२०८२ साउन १ गते")),
        (_ne("१५ श्रावण २०८२"), 30, "ne", None, _ne("१४ भाद्र २०८२")),
        (_ne("१५ साउन २०८२ साल"), 30, "ne", None, _ne("१४ भदौ २०८२ साल")),
        ("वि.सं. २०८२ असार १५", 3, "ne", None, "वि.सं. २०८२ असार १८"),
        # Particles from detected month-name spans stay in place.
        (
            _ar("١٥ من شهر رمضان عام ١٤٤٧ هـ"),
            20,
            "ar",
            LUNAR_HIJRI,
            _ar("٥ من شهر شوال عام ١٤٤٧ هـ"),
        ),
        ("15 ماه رمضان سال 1447", 1, "fa", LUNAR_HIJRI, "16 ماه رمضان سال 1447"),
    ],
)
def test_table_dates_shift_within_their_calendar(value, days, lang, calendar, expected):
    outcome = shift_calendar_date(value, days, lang=lang, calendar=calendar)

    assert outcome.reason == SHIFTED
    assert outcome.text == expected
    assert _shift_date(value, days, lang=lang, calendar=calendar) == expected


@pytest.mark.parametrize(
    ("value", "lang", "calendar", "reason"),
    [
        # Outside the bundled tables (1356-1450 AH, 2000-2083 BS).
        ("15/09/1355 هـ", "ar", None, OUT_OF_RANGE),
        ("15/09/1451 هـ", "ar", None, OUT_OF_RANGE),
        ("1999/12/01 BS", "en", None, OUT_OF_RANGE),
        ("2084/01/01 BS", "en", None, OUT_OF_RANGE),
        ("2084/01/01", "en", BIKRAM_SAMBAT, OUT_OF_RANGE),
        ("1300/02/28", "ar", None, OUT_OF_RANGE),
        # Undeclared variants and conflicting declarations.
        ("15/09/1447", "ar", "hijri_diyanet", UNSUPPORTED_CALENDAR),
        ("15/09/1447", "ar", "hijri_tabular", UNSUPPORTED_CALENDAR),
        ("2082/03/30", "en", "bikram_sambat_india", UNSUPPORTED_CALENDAR),
        ("15/09/1447 هـ", "ar", "hijri_diyanet", AMBIGUOUS),
        ("15/09/1447 هـ", "ar", BIKRAM_SAMBAT, AMBIGUOUS),
        ("2082/03/30 BS", "en", LUNAR_HIJRI, AMBIGUOUS),
        ("2082/03/30 BS ई.सं.", "ne", None, AMBIGUOUS),
        ("١٥ رمضان ١٤٤٧ BS", "ar", None, AMBIGUOUS),
        ("15 साउन 2082", "ne", LUNAR_HIJRI, AMBIGUOUS),
        # Days that do not exist in the table.
        ("30/02/1447 هـ", "ar", None, INVALID_DATE),
        ("30 شعبان 1364 هـ", "ar", None, INVALID_DATE),
        ("2082/08/30 BS", "en", None, INVALID_DATE),
        ("2082/04/32 BS", "en", None, INVALID_DATE),
        ("33 असार 2082", "ne", None, INVALID_DATE),
        ("2082/13/01 BS", "en", None, INVALID_DATE),
    ],
)
def test_out_of_range_and_undeclared_variants_fall_back_to_placeholder(
    value, lang, calendar, reason
):
    outcome = shift_calendar_date(value, 3, lang=lang, calendar=calendar)

    assert outcome.handled
    assert outcome.text is None
    assert outcome.reason == reason
    assert _shift_date(value, 3, lang=lang, calendar=calendar) == PLACEHOLDER


@pytest.mark.parametrize("code", [LUNAR_HIJRI, BIKRAM_SAMBAT])
def test_shifts_that_leave_the_table_fall_back_to_placeholder(code):
    converter = get_calendar(code)
    first, last = converter.day_number_range()

    for day_number, offset in ((last, 1), (first, -1), (last - 10, 11)):
        value = _render_numeric(code, converter.from_day_number(day_number))
        outcome = shift_calendar_date(value, offset, lang="ar")

        assert outcome.reason == OUT_OF_RANGE
        assert _shift_date(value, offset, lang="ar") == PLACEHOLDER
    edge = _render_numeric(code, converter.from_day_number(last))
    assert _parse_numeric(code, _shift_date(edge, -1, lang="ar")) == (
        converter.from_day_number(last - 1)
    )


def test_outcomes_and_errors_never_echo_date_values():
    values = [
        "15/09/1355 هـ",
        "30/02/1447 هـ",
        "2084/01/01 BS",
        "15/09/1447 هـ BS",
        "2082/04/32 BS",
    ]
    for value in values:
        outcome = shift_calendar_date(value, 3, lang="ar")
        serialized = json.dumps(
            [outcome.calendar, outcome.reason, outcome.handled], ensure_ascii=False
        )
        assert not re.search(r"\d", serialized)
        assert value not in repr(outcome)


@pytest.mark.parametrize(
    ("value", "lang", "expected"),
    [
        ("01/01/1447 هـ", "ar", "29/12/1447 هـ"),
        ("2083/01/01 BS", "en", "2083/12/30 BS"),
    ],
)
def test_keep_year_restores_the_table_year(value, lang, expected):
    assert _shift_date(value, -1, keep_year=True, lang=lang) == expected


def test_arabic_and_urdu_packs_read_hijri_range_years_as_hijri():
    assert default_calendar_for("ar_SA") == LUNAR_HIJRI
    assert default_calendar_for("ur") == LUNAR_HIJRI
    assert default_calendar_for("ne") == GREGORIAN

    # Gregorian years never fall in the Hijri table, so they keep shifting
    # exactly as before.
    gregorian = shift_calendar_date("28/02/2024", 3, lang="ar")
    assert not gregorian.handled
    assert _shift_date("28/02/2024", 3, lang="ar") == "02/03/2024"
    # Without a pack default, a marker or a code, numeric dates stay Gregorian.
    assert _shift_date("2082/03/30", 3, lang="ne") == "2082/04/02"


def _render_numeric(code: str, native: tuple[int, int, int]) -> str:
    year, month, day = native
    if code == LUNAR_HIJRI:
        return f"{day:02d}/{month:02d}/{year} هـ"
    return f"{year}/{month:02d}/{day:02d} BS"


def _ascii(value: str) -> str:
    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else char for char in value
    )


def _parse_numeric(code: str, value: str) -> tuple[int, int, int]:
    numbers = [int(item) for item in re.findall(r"\d+", _ascii(value))]
    if code == LUNAR_HIJRI:
        return numbers[2], numbers[1], numbers[0]
    return numbers[0], numbers[1], numbers[2]


def _render_named(code: str, native: tuple[int, int, int]) -> str:
    year, month, day = native
    if code == LUNAR_HIJRI:
        return _ar(f"{day} {_HIJRI_ARABIC_MONTHS[month - 1]} {year} هـ")
    return _ne(f"{year} {_BS_COLLOQUIAL_MONTHS[month - 1]} {day} गते")


def _parse_named(code: str, value: str) -> tuple[int, int, int]:
    names = _HIJRI_ARABIC_MONTHS if code == LUNAR_HIJRI else _BS_COLLOQUIAL_MONTHS
    month = next(
        index + 1
        for index, name in sorted(
            enumerate(names), key=lambda item: len(item[1]), reverse=True
        )
        if name in value
    )
    numbers = [int(item) for item in re.findall(r"\d+", _ascii(value))]
    if code == LUNAR_HIJRI:
        return numbers[1], month, numbers[0]
    return numbers[0], month, numbers[1]


@settings(deadline=None, max_examples=250)
@given(
    code=st.sampled_from([LUNAR_HIJRI, BIKRAM_SAMBAT]),
    position=st.floats(min_value=0.0, max_value=1.0),
    offset=st.integers(min_value=-400, max_value=400),
    named=st.booleans(),
)
def test_shifting_preserves_intervals_and_validity(code, position, offset, named):
    converter = get_calendar(code)
    first, last = converter.day_number_range()
    source_day = first + round(position * (last - first))
    native = converter.from_day_number(source_day)
    render, parse = (
        (_render_named, _parse_named) if named else (_render_numeric, _parse_numeric)
    )
    source = render(code, native)
    lang = "ar" if code == LUNAR_HIJRI else "ne"

    shifted = _shift_date(source, offset, lang=lang)

    if not first <= source_day + offset <= last:
        assert shifted == PLACEHOLDER
        return
    shifted_native = parse(code, shifted)
    assert converter.is_valid(*shifted_native)
    assert converter.to_day_number(*shifted_native) - source_day == offset
    assert shifted == render(code, shifted_native)
    assert _shift_date(shifted, -offset, lang=lang) == source


@pytest.mark.parametrize(
    ("lang", "gregorian_text", "native_text", "metadata", "code"),
    [
        ("ar", "21/03/2025", "21/09/1446 هـ", None, LUNAR_HIJRI),
        ("ar", "21/03/2025", _ar("٢١ رمضان ١٤٤٦"), None, LUNAR_HIJRI),
        ("en", "2025-03-21", "2081/12/08 BS", None, BIKRAM_SAMBAT),
        (
            "en",
            "2025-03-21",
            _ne("२०८१ चैत ८ गते"),
            {CALENDAR_METADATA_KEY: BIKRAM_SAMBAT},
            BIKRAM_SAMBAT,
        ),
    ],
)
def test_gregorian_and_table_dates_of_one_patient_keep_their_interval(
    lang, gregorian_text, native_text, metadata, code, caplog
):
    text = f"A {gregorian_text} B {native_text}"
    converter = get_calendar(code)
    gregorian_day = date(2025, 3, 21).toordinal()
    native_span = (native_text, "DATE", metadata) if metadata else (native_text, "DATE")
    named = not re.search(r"[/\-]", native_text)
    parse = _parse_named if named else _parse_numeric
    assert converter.to_day_number(*parse(code, native_text)) == gregorian_day

    for index in range(30):
        key = f"synthetic-patient-{index}"
        offset = stable_offset_for(key, max_days=365, secret=HMAC_TEST_MATERIAL)
        result = _deidentify(
            text,
            (gregorian_text, "DATE"),
            native_span,
            lang=lang,
            patient_key=key,
            date_shift_max_days=365,
            date_shift_secret=HMAC_TEST_MATERIAL,
        )
        shifted_gregorian = _after(result, "A ").split(" B ", 1)[0]
        shifted_native = _after(result, " B ")

        assert _gregorian_day(shifted_gregorian) - gregorian_day == offset
        native = parse(code, shifted_native)
        assert converter.is_valid(*native)
        assert converter.to_day_number(*native) - gregorian_day == offset
        assert key not in caplog.text
    assert native_text not in caplog.text
    assert _ascii(native_text) not in caplog.text


def test_detected_calendar_codes_select_table_calendars():
    text = "Admit ١٥ رمضان ١٤٤٧, clinic २०८२ असार ३२, other 15/09/1447"

    result = _deidentify(
        text,
        ("١٥ رمضان ١٤٤٧", "DATE", {CALENDAR_METADATA_KEY: LUNAR_HIJRI}),
        ("२०८२ असार ३२", "DATE", {CALENDAR_METADATA_KEY: BIKRAM_SAMBAT}),
        ("15/09/1447", "DATE", {CALENDAR_METADATA_KEY: "hijri_diyanet"}),
        lang="en",
        date_shift_days=1,
    )

    assert result == f"Admit ١٦ رمضان ١٤٤٧, clinic २०८२ साउन १, other {PLACEHOLDER}"


def _parse_ymd(code: str, value: str) -> tuple[int, int, int]:
    return _parse_numeric(BIKRAM_SAMBAT, value)


@pytest.mark.parametrize(
    ("original", "lang", "code", "pattern", "parse", "years"),
    [
        (
            "15/09/1447 هـ",
            "ar",
            LUNAR_HIJRI,
            r"^\d{2}/\d{2}/\d{4} هـ$",
            _parse_numeric,
            (1356, 1446),
        ),
        (
            "1447/02/28",
            "ar",
            LUNAR_HIJRI,
            r"^\d{4}/\d{2}/\d{2}$",
            _parse_ymd,
            (1356, 1446),
        ),
        (
            _ar("١٥ رمضان ١٤٤٧ هـ"),
            "ar",
            LUNAR_HIJRI,
            r"^[٠-٩]{1,2} \D+ [٠-٩]{4} هـ$",
            _parse_named,
            (1356, 1446),
        ),
        (
            "2082/03/30 BS",
            "en",
            BIKRAM_SAMBAT,
            r"^\d{4}/\d{2}/\d{2} BS$",
            _parse_numeric,
            (2000, 2081),
        ),
        (
            _ne("२०८२ असार ३२ गते"),
            "ne",
            BIKRAM_SAMBAT,
            r"^[०-९]{4} \S+ [०-९]{1,2} गते$",
            _parse_named,
            (2000, 2081),
        ),
    ],
)
def test_surrogates_stay_in_the_table_calendar_and_plausible_era(
    original, lang, code, pattern, parse, years
):
    from openmed.core.anonymizer import Anonymizer

    converter = get_calendar(code)
    for seed in range(25):
        surrogate = Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
            original, "date", lang=lang
        )
        repeat = Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
            original, "date", lang=lang
        )
        assert surrogate == repeat
        assert re.fullmatch(pattern, surrogate)
        year, month, day = parse(code, surrogate)
        assert converter.is_valid(year, month, day)
        assert years[0] <= year <= years[1]


def test_surrogates_ignore_undeclared_calendars():
    import random

    rng = random.Random(3888)

    assert calendar_date_surrogate("2082/03/30", lang="ne", rng=rng) is None
    assert calendar_date_surrogate("28/02/2024", lang="ar", rng=rng) is None


def _deidentify(text: str, *spans, method: str = "shift_dates", **run_kwargs) -> str:
    entities = []
    for span in spans:
        surface, label, *rest = span
        start = text.index(surface)
        entities.append(
            EntityPrediction(
                text=surface,
                label=label,
                start=start,
                end=start + len(surface),
                confidence=0.99,
                metadata=dict(rest[0]) if rest and rest[0] else None,
            )
        )

    def model_detector(model_text: str, **kwargs):
        assert model_text == text
        return PredictionResult(
            text=model_text,
            entities=list(entities),
            model_name="unit-test",
            timestamp=datetime.now().isoformat(),
        )

    lang = run_kwargs.pop("lang")
    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
        lang=lang,
    ).run(text, method=method, **run_kwargs)
    return result.deidentification_result.deidentified_text


def _after(text: str, prefix: str) -> str:
    return text.split(prefix, 1)[1]


def _gregorian_day(value: str) -> int:
    numbers = [int(item) for item in re.findall(r"\d+", _ascii(value))]
    if len(str(numbers[0])) == 4:
        return date(numbers[0], numbers[1], numbers[2]).toordinal()
    return date(numbers[2], numbers[1], numbers[0]).toordinal()
