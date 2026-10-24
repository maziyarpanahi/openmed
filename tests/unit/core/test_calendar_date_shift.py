"""Shifting and synthesizing Solar Hijri, Ethiopian and Thai Buddhist-era dates.

Expected outputs are literals checked against the independently computed
reference dates in ``test_calendars.py``. Every input is synthetic.
"""

from __future__ import annotations

import json
import random
import re
from datetime import date, datetime, timedelta

import pytest

from openmed.core.calendar_dates import (
    AMBIGUOUS,
    ERA_MARKERS,
    INVALID_DATE,
    MONTH_STYLES,
    OUT_OF_RANGE,
    SHIFTED,
    UNRECOGNIZED,
    UNSUPPORTED_CALENDAR,
    calendar_date_surrogate,
    default_calendar_for,
    shift_calendar_date,
)
from openmed.core.calendars import (
    ETHIOPIAN,
    GREGORIAN,
    SOLAR_HIJRI,
    THAI_BUDDHIST,
    get_calendar,
)
from openmed.core.date_shift import stable_offset_for, stable_offset_from_seed
from openmed.core.pii import _shift_date
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult

HMAC_TEST_MATERIAL = bytes(range(7, 39))
PLACEHOLDER = "[DATE_SHIFTED]"
CONTROLLED_REASONS = {
    SHIFTED,
    "gregorian",
    UNRECOGNIZED,
    AMBIGUOUS,
    UNSUPPORTED_CALENDAR,
    OUT_OF_RANGE,
    INVALID_DATE,
}
_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _fa(text: str) -> str:
    return text.translate(_PERSIAN_DIGITS)


@pytest.mark.parametrize(
    ("value", "days", "lang", "expected"),
    [
        # Issue examples: valid Solar Hijri, Thai and Pagume dates.
        (_fa("1404/07/30"), 1, "fa", _fa("1404/08/01")),
        (_fa("30 اردیبهشت 1404"), 1, "fa", _fa("31 اردیبهشت 1404")),
        (_fa("28 اردیبهشت 1404"), 3, "fa", _fa("31 اردیبهشت 1404")),
        ("28 กุมภาพันธ์ 2567", 1, "th", "29 กุมภาพันธ์ 2567"),
        ("29 กุมภาพันธ์ 2567", 1, "th", "1 มีนาคม 2567"),
        ("29/02/2567", 1, "th", "01/03/2567"),
        ("05/13/2016", 1, "am", "01/01/2017"),
        ("06/13/2015", 1, "am", "01/01/2016"),
        ("01/13/2016", 4, "am", "05/13/2016"),
        # Year boundaries and negative offsets.
        (_fa("29 اسفند 1404"), 1, "fa", _fa("1 فروردین 1405")),
        (_fa("1405/01/01"), -1, "fa", _fa("1404/12/29")),
        (_fa("1404/01/01"), -1, "fa", _fa("1403/12/30")),
        ("01/01/2016", -1, "am", "06/13/2015"),
        ("1 มกราคม 2568", -1, "th", "31 ธันวาคม 2567"),
        # Month names and era markers are kept in their source form.
        ("ጳጉሜ 5 ቀን 2016 ዓ.ም", 1, "am", "መስከረም 1 ቀን 2017 ዓ.ም"),
        ("28 ก.พ. พ.ศ. 2567", 2, "th", "1 มี.ค. พ.ศ. 2567"),
        (_fa("1404/07/30 ه.ش"), 3, "fa", _fa("1404/08/03 ه.ش")),
        ("12/07/2017 ዓ.ም", 30, "am", "12/08/2017 ዓ.ም"),
        # Digit systems, separators and padding follow the source.
        ("١٤٠٤-٠٧-٣٠", 1, "fa", "١٤٠٤-٠٨-٠١"),
        ("1404.7.3", 3, "fa", "1404.7.6"),
        ("๒๘/๐๒/๒๕๖๗", 1, "th", "๒๙/๐๒/๒๕๖๗"),
        ("5 دي 1404", 1, "fa", "6 دي 1404"),
    ],
)
def test_native_dates_shift_within_their_calendar(value, days, lang, expected):
    assert _shift_date(value, days, lang=lang) == expected


@pytest.mark.parametrize(
    ("value", "lang", "calendar"),
    [
        ("۱۴۰۴/۰۷/۳۰ ه.ش", "en", SOLAR_HIJRI),
        ("۲۸ اردیبهشت ۱۴۰۴", None, SOLAR_HIJRI),
        ("ጳጉሜ 5 2016", "en", ETHIOPIAN),
        ("15/03/2567 B.E.", None, THAI_BUDDHIST),
        ("03/15/2567 BE", "en", THAI_BUDDHIST),
        ("12/05/2016 EC", "en", ETHIOPIAN),
    ],
)
def test_markers_and_month_names_identify_the_calendar_without_a_pack(
    value, lang, calendar
):
    outcome = shift_calendar_date(value, 1, lang=lang)

    assert outcome.handled
    assert outcome.reason == SHIFTED
    assert outcome.calendar == calendar


@pytest.mark.parametrize(
    ("value", "lang", "expected"),
    [
        ("01/15/2020", "en", "01/18/2020"),
        ("2025/03/21", "fa", "2025/03/24"),
        ("21/03/2025", "th", "24/03/2025"),
        ("28 กุมภาพันธ์ 2024", "th", "2 มีนาคม 2024"),
        ("11.10.2026", "de", "14.10.2026"),
    ],
)
def test_gregorian_dates_keep_the_existing_shifter(value, lang, expected):
    outcome = shift_calendar_date(value, 3, lang=lang)

    assert not outcome.handled
    assert _shift_date(value, 3, lang=lang) == expected


@pytest.mark.parametrize(
    ("value", "lang", "expected"),
    [
        ("2025/03/21 میلادی", "fa", "2025/03/24 میلادی"),
        ("12/05/2016 እ.ኤ.አ", "am", "15/05/2016 እ.ኤ.አ"),
        ("28 กุมภาพันธ์ ค.ศ. 2024", "th", "2 มีนาคม ค.ศ. 2024"),
    ],
)
def test_explicit_gregorian_markers_override_the_pack_calendar(value, lang, expected):
    outcome = shift_calendar_date(value, 3, lang=lang)

    assert outcome.calendar == GREGORIAN
    assert _shift_date(value, 3, lang=lang) == expected


@pytest.mark.parametrize(
    ("value", "lang", "reason", "calendar"),
    [
        # Out of the validated range: never extrapolated or read as Gregorian.
        ("۱۲۵۰/۰۱/۰۱", "fa", OUT_OF_RANGE, None),
        ("۲۸ اردیبهشت ۲۰۲۵", "fa", OUT_OF_RANGE, None),
        ("15/03/2450", "th", OUT_OF_RANGE, None),
        ("01/01/1700", "am", OUT_OF_RANGE, None),
        # Dates that do not exist in their calendar.
        ("۳۰ اسفند ۱۴۰۴", "fa", INVALID_DATE, None),
        ("1404/07/31", "fa", INVALID_DATE, None),
        ("06/13/2016", "am", INVALID_DATE, None),
        ("31/01/2016", "am", INVALID_DATE, None),
        ("29/02/2566", "th", INVALID_DATE, None),
        # Conflicting or missing calendar evidence.
        ("1404/01/01 ه.ش میلادی", "fa", AMBIGUOUS, None),
        ("۲۸ اردیبهشت ۱۴۰۴ میلادی", "fa", AMBIGUOUS, None),
        ("05/07/1404", "fa", AMBIGUOUS, None),
        ("28/02/67", "th", AMBIGUOUS, None),
        ("28 ก.พ. 67", "th", AMBIGUOUS, None),
        ("04/07/30", "fa", AMBIGUOUS, None),
        # Calendars without a registered converter.
        ("15/09/1447 هـ", "ar", UNSUPPORTED_CALENDAR, None),
        ("1404/07/30", "en", UNSUPPORTED_CALENDAR, "hebrew"),
        ("1404/07/30", "en", UNSUPPORTED_CALENDAR, "Solar Hijri"),
        ("1404/07/30", "en", UNSUPPORTED_CALENDAR, 7),
        # A declared native calendar with an unparseable surface.
        ("፲፬/፭/፲፱፻፹፰", "am", UNRECOGNIZED, ETHIOPIAN),
        ("1404-07-30T10:00", "en", UNRECOGNIZED, SOLAR_HIJRI),
        ("اردیبهشت ۱۴۰۴", "en", UNRECOGNIZED, None),
    ],
)
def test_out_of_range_invalid_and_ambiguous_dates_fall_back_to_placeholder(
    value, lang, reason, calendar
):
    outcome = shift_calendar_date(value, 3, lang=lang, calendar=calendar)

    assert outcome.handled
    assert outcome.text is None
    assert outcome.reason == reason
    assert outcome.reason in CONTROLLED_REASONS
    assert _shift_date(value, 3, lang=lang, calendar=calendar) == PLACEHOLDER


def test_shift_beyond_the_supported_range_falls_back_to_placeholder():
    converter = get_calendar(SOLAR_HIJRI)
    first, last = converter.day_number_range()
    year, month, day = converter.from_day_number(last)
    value = f"{year}/{month:02d}/{day:02d}"

    assert _shift_date(value, 1, lang="fa") == PLACEHOLDER
    assert shift_calendar_date(value, 1, lang="fa").reason == OUT_OF_RANGE
    assert _shift_date(value, -1, lang="fa") == "1501/12/28"


def test_outcomes_never_echo_date_values():
    values = ["۱۲۵۰/۰۱/۰۱", "۳۰ اسفند ۱۴۰۴", "1404/01/01 ه.ش میلادی", "1404/07/31"]
    for value in values:
        outcome = shift_calendar_date(value, 3, lang="fa")
        serialized = json.dumps(
            [outcome.calendar, outcome.reason, outcome.handled], ensure_ascii=False
        )
        assert not re.search(r"\d", serialized)
        assert value not in repr(outcome)


@pytest.mark.parametrize(
    ("value", "days", "lang", "expected"),
    [
        ("1404/12/29", 2, "fa", "1404/01/02"),
        ("1407/12/29", 366, "fa", "1407/12/29"),
        ("05/13/2016", 3, "am", "03/01/2016"),
        ("06/13/2015", 365, "am", "05/13/2015"),
        ("31/12/2567", 1, "th", "01/01/2567"),
    ],
)
def test_keep_year_restores_the_native_year_and_clamps_leap_days(
    value, days, lang, expected
):
    assert _shift_date(value, days, keep_year=True, lang=lang) == expected


def _render_numeric(code: str, native: tuple[int, int, int]) -> str:
    year, month, day = native
    if code == SOLAR_HIJRI:
        return f"{year}/{month:02d}/{day:02d}"
    return f"{day:02d}/{month:02d}/{year}"


def _parse_numeric(code: str, value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.split("/")]
    if code == SOLAR_HIJRI:
        return parts[0], parts[1], parts[2]
    return parts[2], parts[1], parts[0]


@pytest.mark.parametrize(
    ("code", "lang"),
    [(SOLAR_HIJRI, "fa"), (ETHIOPIAN, "am"), (THAI_BUDDHIST, "th")],
)
def test_shifting_preserves_intervals_and_validity_over_the_range(code, lang):
    converter = get_calendar(code)
    first, last = converter.day_number_range()
    rng = random.Random(3867)

    for _ in range(600):
        offset = rng.choice([-365, -31, -1, 1, 29, 30, 31, 365, rng.randint(-400, 400)])
        source_day = rng.randint(first + 400, last - 400)
        source = _render_numeric(code, converter.from_day_number(source_day))

        shifted = _shift_date(source, offset, lang=lang)
        native = _parse_numeric(code, shifted)

        assert converter.is_valid(*native)
        assert converter.to_day_number(*native) - source_day == offset
        assert _shift_date(shifted, -offset, lang=lang) == source


@pytest.mark.parametrize(
    ("lang", "gregorian_text", "native_text", "code"),
    [
        ("fa", "2025/03/21", "1404/01/01", SOLAR_HIJRI),
        ("fa", "2025/03/21 میلادی", "۲۸ اسفند ۱۴۰۳", SOLAR_HIJRI),
        ("th", "21/03/2025", "21/03/2568", THAI_BUDDHIST),
        ("am", "21/03/2025 እ.ኤ.አ", "12/07/2017", ETHIOPIAN),
    ],
)
def test_gregorian_and_native_dates_of_one_patient_keep_their_interval(
    lang, gregorian_text, native_text, code
):
    text = f"A {gregorian_text} B {native_text}"
    gregorian_day = date(2025, 3, 21).toordinal()
    converter = get_calendar(code)
    source_native_day = _native_day(code, native_text)
    keys = [f"synthetic-patient-{index}" for index in range(40)]

    for key in keys:
        offset = stable_offset_for(key, max_days=365, secret=HMAC_TEST_MATERIAL)
        result = _deidentify(
            text,
            (gregorian_text, "DATE"),
            (native_text, "DATE"),
            lang=lang,
            patient_key=key,
            date_shift_max_days=365,
            date_shift_secret=HMAC_TEST_MATERIAL,
        )
        shifted_gregorian, shifted_native = (
            _between(result, "A ", " B "),
            _after(result, " B "),
        )

        gregorian_value = _gregorian_day(lang, shifted_gregorian)
        native_value = _native_day(code, shifted_native)
        assert gregorian_value - gregorian_day == offset
        assert native_value - source_native_day == offset
        assert converter.is_valid(*converter.from_day_number(native_value))


def test_patient_keyed_and_seeded_native_shifts_are_reproducible(caplog):
    first_text = "پذیرش ۱۴۰۴/۰۷/۳۰"
    second_text = "ترخیص ۲۸ اردیبهشت ۱۴۰۴"
    key = "synthetic-patient-reproducible"
    offset = stable_offset_for(key, max_days=60, secret=HMAC_TEST_MATERIAL)
    kwargs = {
        "lang": "fa",
        "patient_key": key,
        "date_shift_max_days": 60,
        "date_shift_secret": HMAC_TEST_MATERIAL,
    }

    first = _deidentify(first_text, ("۱۴۰۴/۰۷/۳۰", "DATE"), **kwargs)
    again = _deidentify(first_text, ("۱۴۰۴/۰۷/۳۰", "DATE"), **kwargs)
    second = _deidentify(second_text, ("۲۸ اردیبهشت ۱۴۰۴", "DATE"), **kwargs)

    assert first == again
    assert (
        _native_day(SOLAR_HIJRI, _after(first, "پذیرش "))
        - _native_day(SOLAR_HIJRI, "1404/07/30")
        == offset
    )
    assert (
        _native_day(SOLAR_HIJRI, _after(second, "ترخیص "))
        - _native_day(SOLAR_HIJRI, "۲۸ اردیبهشت ۱۴۰۴")
        == offset
    )
    assert key not in caplog.text
    assert "۱۴۰۴" not in caplog.text

    seed_offset = stable_offset_from_seed(822, max_days=30)
    seeded = [
        _deidentify(
            "ቀን 05/13/2016",
            ("05/13/2016", "DATE"),
            lang="am",
            seed=822,
            date_shift_max_days=30,
        )
        for _ in range(2)
    ]
    assert seeded[0] == seeded[1]
    assert (
        _native_day(ETHIOPIAN, _after(seeded[0], "ቀን "))
        - _native_day(ETHIOPIAN, "05/13/2016")
        == seed_offset
    )


def test_detection_stage_calendar_code_selects_the_calendar():
    text = "Visit 1404/07/30 and 30/05/2025, review 30/05/2017"

    result = _deidentify(
        text,
        ("1404/07/30", "DATE", {"calendar": SOLAR_HIJRI}),
        ("30/05/2025", "DATE", {"calendar": GREGORIAN}),
        ("30/05/2017", "DATE"),
        lang="am",
        date_shift_days=2,
    )

    # Without a code, the Amharic pack's Ethiopian default applies: 30 Tir
    # plus two days is 2 Yekatit, whereas 30 May plus two days is 1 June.
    assert result == "Visit 1404/08/02 and 01/06/2025, review 02/06/2017"


@pytest.mark.parametrize(
    ("original", "lang", "code", "pattern"),
    [
        ("۱۴۰۴/۰۷/۳۰", "fa", SOLAR_HIJRI, r"^[۰-۹]{4}/[۰-۹]{2}/[۰-۹]{2}$"),
        ("۲۸ اردیبهشت ۱۴۰۴", "fa", SOLAR_HIJRI, r"^[۰-۹]{1,2} \S+ [۰-۹]{4}$"),
        ("05/13/2016", "am", ETHIOPIAN, r"^\d{2}/\d{2}/\d{4}$"),
        ("ጳጉሜ 5 ቀን 2016 ዓ.ም", "am", ETHIOPIAN, r"^\S+ \d{1,2} ቀን \d{4} ዓ\.ም$"),
        ("29/02/2567", "th", THAI_BUDDHIST, r"^\d{2}/\d{2}/\d{4}$"),
        ("28 ก.พ. พ.ศ. 2567", "th", THAI_BUDDHIST, r"^\d{1,2} \S+ พ\.ศ\. \d{4}$"),
    ],
)
def test_replace_surrogates_stay_in_the_calendar_and_plausible_era(
    original, lang, code, pattern
):
    from openmed.core.anonymizer import Anonymizer

    converter = get_calendar(code)
    first, _ = converter.day_number_range()
    window_start = date(1925, 1, 1).toordinal()
    era_start = converter.from_day_number(max(first, window_start))[0]
    era_end = converter.from_day_number(window_start + 365 * 100)[0]
    for seed in range(30):
        surrogate = Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
            original, "date", lang=lang
        )
        repeat = Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
            original, "date", lang=lang
        )
        assert surrogate == repeat
        assert re.fullmatch(pattern, surrogate)
        native = _parse_any(code, surrogate)
        assert converter.is_valid(*native)
        assert era_start <= native[0] <= era_end


def test_replace_in_a_persian_pipeline_keeps_jalali_era_and_digits():
    for seed in range(20):
        result = _deidentify(
            "تاریخ ۱۴۰۴/۰۷/۳۰",
            ("۱۴۰۴/۰۷/۳۰", "DATE"),
            lang="fa",
            method="replace",
            seed=seed,
        )
        surrogate = _after(result, "تاریخ ")
        year, month, day = _parse_any(SOLAR_HIJRI, surrogate)
        assert re.fullmatch(r"[۰-۹]{4}/[۰-۹]{2}/[۰-۹]{2}", surrogate)
        assert 1303 <= year <= 1404
        assert get_calendar(SOLAR_HIJRI).is_valid(year, month, day)


def test_format_preserve_uses_the_source_calendar():
    from openmed.core.anonymizer import Anonymizer

    surrogate = Anonymizer(
        lang="th", consistent=True, seed=5
    ).format_preserving_surrogate("29/02/2567", "date", lang="th")

    assert surrogate is not None
    assert surrogate != "29/02/2567"
    assert get_calendar(THAI_BUDDHIST).is_valid(*_parse_any(THAI_BUDDHIST, surrogate))


def test_surrogates_ignore_plain_gregorian_and_unknown_calendars():
    rng = random.Random(1)

    assert calendar_date_surrogate("11/10/2026", lang="en", rng=rng) is None
    assert calendar_date_surrogate("2025/03/21", lang="fa", rng=rng) is None
    assert calendar_date_surrogate("15/09/1447 هـ", lang="ar", rng=rng) is None
    assert calendar_date_surrogate("28/02/67", lang="th", rng=rng) is None


def test_default_calendar_resolves_language_codes_and_locales():
    assert default_calendar_for("fa_IR") == SOLAR_HIJRI
    assert default_calendar_for("am-ET") == ETHIOPIAN
    assert default_calendar_for("th") == THAI_BUDDHIST
    assert default_calendar_for("en") == GREGORIAN
    assert default_calendar_for("xx") == GREGORIAN
    assert default_calendar_for(None) == GREGORIAN


def test_month_tables_match_the_language_pack_month_names():
    from openmed.core.pii_i18n import LANGUAGE_MONTH_NAMES

    styles = {style.key: style for style in MONTH_STYLES}

    assert [names[0] for names in styles["persian"].names] == LANGUAGE_MONTH_NAMES["fa"]
    assert [names[0] for names in styles["thai_full"].names] == LANGUAGE_MONTH_NAMES[
        "th"
    ]
    assert len(styles["amharic"].names) == 13
    assert set(ERA_MARKERS.values()) <= {
        GREGORIAN,
        SOLAR_HIJRI,
        ETHIOPIAN,
        THAI_BUDDHIST,
        "islamic",
    }


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
                metadata=dict(rest[0]) if rest else None,
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


def _between(text: str, prefix: str, suffix: str) -> str:
    return _after(text, prefix).split(suffix, 1)[0]


def _ascii(value: str) -> str:
    import unicodedata

    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else char for char in value
    )


def _parse_any(code: str, value: str) -> tuple[int, int, int]:
    from openmed.core.calendar_dates import _MONTH_LOOKUP

    folded = _ascii(value)
    numbers = [int(item) for item in re.findall(r"\d+", folded)]
    month_name = next(
        (
            name
            for name in sorted(_MONTH_LOOKUP, key=len, reverse=True)
            if name in value
        ),
        None,
    )
    if month_name is not None:
        day, year = numbers
        return year, _MONTH_LOOKUP[month_name][1], day
    if code == SOLAR_HIJRI:
        return numbers[0], numbers[1], numbers[2]
    return numbers[2], numbers[1], numbers[0]


def _native_day(code: str, value: str) -> int:
    native = _parse_any(code, value.split(" ه")[0].split(" ዓ")[0])
    return get_calendar(code).to_day_number(*native)


def _gregorian_day(lang: str, value: str) -> int:
    folded = _ascii(value).split(" ")[0]
    numbers = [int(item) for item in re.findall(r"\d+", folded)]
    if len(str(numbers[0])) == 4:
        return date(numbers[0], numbers[1], numbers[2]).toordinal()
    return date(numbers[2], numbers[1], numbers[0]).toordinal()


def test_helper_reference_dates_are_consistent():
    # Guard the test helpers with the same independent reference dates.
    assert _native_day(SOLAR_HIJRI, "۱۴۰۴/۰۱/۰۱") == date(2025, 3, 21).toordinal()
    assert (
        _native_day(SOLAR_HIJRI, "۲۸ اسفند ۱۴۰۳")
        == (date(2025, 3, 21) - timedelta(days=3)).toordinal()
    )
    assert _native_day(ETHIOPIAN, "12/07/2017") == date(2025, 3, 21).toordinal()
    assert _native_day(THAI_BUDDHIST, "21/03/2568") == date(2025, 3, 21).toordinal()
