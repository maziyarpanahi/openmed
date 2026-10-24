"""Hijri, Ethiopian, Bikram Sambat and Hebrew month-name dates (#3866).

Every input is synthetic. Detection only: these dates keep the existing shift
placeholder until calendar-aware shifting lands.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import pytest

from openmed.core.calendar_month_dates import (
    BIKRAM_SAMBAT_CALENDAR,
    CALENDAR_CODES,
    CALENDAR_MONTH_TABLES,
    ETHIOPIAN_CALENDAR,
    HEBREW_CALENDAR,
    HIJRI_CALENDAR,
    calendar_month_date_patterns,
)
from openmed.core.pii import _apply_pii_smart_merging, _shift_date
from openmed.core.pii_entity_merger import (
    PIIPattern,
    merge_entities_with_semantic_units,
)
from openmed.core.pii_i18n import get_patterns_for_language
from openmed.core.pipeline import Pipeline
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import PredictionResult

_STUB_MODEL = "synthetic/user-supplied-stub"
_I18N_DIR = Path("openmed/eval/golden/fixtures/i18n")
_NEPALI_FIXTURE = Path("tests/fixtures/pii/nepali_bikram_sambat_synthetic_notes.jsonl")

# (lang, text, exact date surface, calendar). The first seven rows are the
# forms listed in the issue; the rest cover spelling, digit and order variants.
_POSITIVE_CASES = (
    ("fa", "تاریخ تولد: ۱۵ رمضان ۱۴۴۷", "۱۵ رمضان ۱۴۴۷", HIJRI_CALENDAR),
    ("ar", "تاريخ الميلاد: ١٥ رمضان ١٤٤٧ هـ.", "١٥ رمضان ١٤٤٧ هـ", HIJRI_CALENDAR),
    (
        "ur",
        "تاریخ پیدائش: ۱۵ رمضان ۱۴۴۷ ہجری",
        "۱۵ رمضان ۱۴۴۷ ہجری",
        HIJRI_CALENDAR,
    ),
    ("am", "የትውልድ ቀን: 15 መስከረም 2017", "15 መስከረም 2017", ETHIOPIAN_CALENDAR),
    (
        "am",
        "የትውልድ ቀን: መስከረም 15 ቀን 2017 ዓ.ም",
        "መስከረም 15 ቀን 2017 ዓ.ም",
        ETHIOPIAN_CALENDAR,
    ),
    ("ne", "जन्म मिति: २०८२ असार १५", "२०८२ असार १५", BIKRAM_SAMBAT_CALENDAR),
    ("he", "תאריך לידה: ט״ו בשבט תשפ״ו", "ט״ו בשבט תשפ״ו", HEBREW_CALENDAR),
    ("ar", "ولد في 3 ذي الحجة 1420 للهجرة.", "3 ذي الحجة 1420 للهجرة", HIJRI_CALENDAR),
    (
        "ar",
        "ولد في 15 من شهر رمضان عام 1447 هـ.",
        "15 من شهر رمضان عام 1447 هـ",
        HIJRI_CALENDAR,
    ),
    ("ar", "ولد في ٢٩ شوّال ١٤١٠ه وراجع.", "٢٩ شوّال ١٤١٠ه", HIJRI_CALENDAR),
    (
        "fa",
        "تاریخ تولد: ۱۵ ربیع\u200cالاول ۱۴۴۷ ه.ق",
        "۱۵ ربیع\u200cالاول ۱۴۴۷ ه.ق",
        HIJRI_CALENDAR,
    ),
    ("fa", "تاریخ تولد: ۳ ذیقعده ۱۴۲۰ قمری", "۳ ذیقعده ۱۴۲۰ قمری", HIJRI_CALENDAR),
    (
        "ur",
        "تاریخ پیدائش: 7 جمادی الاولیٰ 1410 ہجری",
        "7 جمادی الاولیٰ 1410 ہجری",
        HIJRI_CALENDAR,
    ),
    (
        "ur",
        "تاریخ پیدائش: 1 رمضان المبارک 1440ھ۔",
        "1 رمضان المبارک 1440ھ",
        HIJRI_CALENDAR,
    ),
    (
        "am",
        "የትውልድ ቀን፡ ፲፭ ጥር ፲፱፻፹፭ ዓ.ም።",
        "፲፭ ጥር ፲፱፻፹፭ ዓ.ም",
        ETHIOPIAN_CALENDAR,
    ),
    ("am", "የትውልድ ቀን፡ ጳጉሜን 5, 2015።", "ጳጉሜን 5, 2015", ETHIOPIAN_CALENDAR),
    (
        "ne",
        "जन्म मिति: वि.सं. २०५० साल असार ३२ गते।",
        "वि.सं. २०५० साल असार ३२ गते",
        BIKRAM_SAMBAT_CALENDAR,
    ),
    ("ne", "जन्म मिति: १५ चैत्र २०८२।", "१५ चैत्र २०८२", BIKRAM_SAMBAT_CALENDAR),
    ("ne", "जन्म मिति: भदौ 5, 2045।", "भदौ 5, 2045", BIKRAM_SAMBAT_CALENDAR),
    ("he", 'תאריך לידה: ט"ו בשבט תשפ"ו.', 'ט"ו בשבט תשפ"ו', HEBREW_CALENDAR),
    ("he", "נולד בכ״ה באדר ב׳ תשפ״ד.", "כ״ה באדר ב׳ תשפ״ד", HEBREW_CALENDAR),
    ("he", "נולד ב-15 בשבט 5786.", "15 בשבט 5786", HEBREW_CALENDAR),
    ("he", "נולדה א׳ בתשרי ה׳תשפ״ו.", "א׳ בתשרי ה׳תשפ״ו", HEBREW_CALENDAR),
    ("he", "נולד ל׳ בחודש חשוון תש״ן.", "ל׳ בחודש חשוון תש״ן", HEBREW_CALENDAR),
)

# Month names used as ordinary words or names, and partial dates without a
# day or a year. None of them may be flagged.
_NEGATIVE_CASES = (
    ("fa", "در ماه رمضان روزه می\u200cگیرم."),
    ("fa", "نتیجه آزمایش صفر بود."),
    ("ar", "صام المريض شهر رمضان كاملا."),
    ("ar", "المريض رمضان علي، 45 سنة."),
    ("ar", "النتيجة 3 صفر 1 في المباراة."),
    ("ar", "يعود في 15 رمضان."),
    ("ar", "زيارة في رمضان 1447."),
    ("ur", "رمضان میں روزہ رکھا۔"),
    ("am", "መስከረም ወር ላይ ይመለሳል።"),
    ("am", "ጥር ወር ቀጠሮ አለ።"),
    ("ne", "मेरो जेठ आउनुभयो।"),
    ("ne", "माघ महिनामा फेरि आउनुहोस्।"),
    ("ne", "१५ असार मा फेरि आउनुहोस्।"),
    ("he", "ט״ו בשבט הוא חג."),
    ("he", "בני שבט לוי הגיעו."),
    ("he", "אב המשפחה הגיע."),
    ("he", "סיון לוי, בת 40."),
    ("he", "ט׳ באב תשע ימים."),
)

# Swept (label, start, end) spans recorded on master before the calendar
# tables were added. Gregorian month-name and numeric dates must not change.
_UNCHANGED_SWEEP_SNAPSHOT = (
    (
        "ar",
        "تاريخ الميلاد: 15 مارس 2020؛ المراجعة 3 سبتمبر 2021.",
        [("date", 15, 27), ("date", 38, 51)],
    ),
    ("ar", "تاريخ الميلاد: ١٥ يناير ١٩٨٠ م.", [("date", 15, 28)]),
    (
        "fa",
        "تاریخ تولد: ۱۵ مهر ۱۳۶۰؛ پذیرش ۲ فروردین ۱۴۰۴.",
        [("date", 12, 23), ("date", 31, 45)],
    ),
    ("ur", "تاریخ پیدائش: ۱۵ جنوری ۱۹۸۰۔", [("date", 14, 27)]),
    ("he", "תאריך לידה: 15 במרץ 1980.", [("date", 12, 24)]),
    ("he", "נולד ב-3 בספטמבר 1975.", [("date", 7, 21)]),
    ("am", "የትውልድ ቀን: 15/01/2017", [("date", 10, 20)]),
    ("ne", "जन्म मिति: 2045/03/15", [("date", 11, 21)]),
    ("hi", "जन्म तिथि: 15 जनवरी 1980", [("date", 11, 24)]),
    ("en", "DOB 15 September 1980; review Ramadan 15, 1447.", [("date", 4, 21)]),
)


def _empty_detector(text: str, **_kwargs: object) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="synthetic-empty-detector",
        timestamp="2026-01-01T00:00:00Z",
    )


def _pipeline(lang: str) -> Pipeline:
    return Pipeline(
        lang=lang,
        model_name=_STUB_MODEL,
        model_detector=_empty_detector,
        use_smart_merging=True,
        use_safety_sweep=True,
    )


def _date_spans(lang: str, text: str) -> list[tuple[int, int, str, str | None]]:
    return [
        (entity.start, entity.end, entity.text, entity.metadata.get("calendar"))
        for entity in safety_sweep(text, [], lang=lang)
        if entity.label == "date"
    ]


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _calendar_rows() -> list[dict]:
    rows = [
        row
        for lang in ("ar", "fa", "am", "he")
        for row in _load_jsonl(_I18N_DIR / f"{lang}.jsonl")
        if any(span.get("metadata", {}).get("calendar") for span in row["gold_spans"])
    ]
    return [*rows, *_load_jsonl(_NEPALI_FIXTURE)]


def test_month_tables_are_declarative_calendar_vocabularies() -> None:
    month_counts = {
        HIJRI_CALENDAR: 12,
        ETHIOPIAN_CALENDAR: 13,
        BIKRAM_SAMBAT_CALENDAR: 12,
        HEBREW_CALENDAR: 13,
    }
    calendars = set()
    for lang, tables in CALENDAR_MONTH_TABLES.items():
        for table in tables:
            calendars.add(table.calendar)
            assert table.calendar in CALENDAR_CODES
            assert len(table.months) == month_counts[table.calendar], lang
            assert all(table.months), lang
            for spelling in (name for month in table.months for name in month):
                assert spelling == spelling.strip()
                assert unicodedata.is_normalized("NFC", spelling)
    assert calendars == CALENDAR_CODES
    assert set(CALENDAR_MONTH_TABLES) == {"ar", "fa", "ur", "am", "ne", "he"}


def test_calendar_patterns_join_their_language_routes_only() -> None:
    for lang in CALENDAR_MONTH_TABLES:
        compiled = calendar_month_date_patterns(lang)
        assert compiled
        assert all(pattern.entity_type == "date" for pattern in compiled)
        assert all(pattern.calendar in CALENDAR_CODES for pattern in compiled)
        route = get_patterns_for_language(lang)
        assert {pattern.calendar for pattern in route} - {None} == {
            table.calendar for table in CALENDAR_MONTH_TABLES[lang]
        }
    assert calendar_month_date_patterns("ar-SA") == calendar_month_date_patterns("ar")
    for lang in ("en", "fr", "hi", "sw"):
        assert calendar_month_date_patterns(lang) == []
        assert all(
            pattern.calendar is None for pattern in get_patterns_for_language(lang)
        )


@pytest.mark.parametrize(("lang", "text", "surface", "calendar"), _POSITIVE_CASES)
def test_month_name_date_is_swept_with_its_calendar_code(
    lang: str, text: str, surface: str, calendar: str
) -> None:
    start = text.index(surface)

    assert _date_spans(lang, text) == [(start, start + len(surface), surface, calendar)]


@pytest.mark.parametrize(("lang", "text", "surface", "calendar"), _POSITIVE_CASES)
def test_empty_detector_pipeline_masks_and_tags_month_name_date(
    lang: str, text: str, surface: str, calendar: str
) -> None:
    start = text.index(surface)
    end = start + len(surface)

    result = _pipeline(lang).run(text, method="mask")

    masked = result.deidentification_result.deidentified_text
    assert surface not in masked
    assert masked == f"{text[:start]}[date]{text[end:]}"
    date_spans = [span for span in result.spans if span.canonical_label == "DATE"]
    assert [(span.start, span.end) for span in date_spans] == [(start, end)]
    assert date_spans[0].metadata["calendar"] == calendar
    assert date_spans[0].metadata["safety_sweep"]["calendar"] == calendar


@pytest.mark.parametrize(("lang", "text"), _NEGATIVE_CASES)
def test_month_word_without_day_and_year_is_not_flagged(lang: str, text: str) -> None:
    assert _date_spans(lang, text) == []
    assert _pipeline(lang).run(text, method="mask").redacted_text == text


@pytest.mark.parametrize(("lang", "text", "expected"), _UNCHANGED_SWEEP_SNAPSHOT)
def test_gregorian_and_numeric_date_sweep_is_unchanged(
    lang: str, text: str, expected: list[tuple[str, int, int]]
) -> None:
    swept = safety_sweep(text, [], lang=lang)

    assert [(entity.label, entity.start, entity.end) for entity in swept] == expected
    assert all("calendar" not in entity.metadata for entity in swept)


def test_date_prefix_letters_stay_outside_the_span() -> None:
    hebrew = "נולד ובט״ו בשבט תשפ״ו."
    amharic = "የተወለደው በመስከረም 15 ቀን 2017 ዓ.ም ነው።"

    assert [span[2] for span in _date_spans("he", hebrew)] == ["ט״ו בשבט תשפ״ו"]
    assert [span[2] for span in _date_spans("am", amharic)] == ["መስከረም 15 ቀን 2017 ዓ.ም"]


def test_spans_keep_combining_marks_with_their_letters() -> None:
    cases = (
        ("ur", "تاریخ پیدائش: 2 ربیع الاوّل 1410ھ۔"),
        ("ne", "जन्म मिति: २०५० चैत ३ गते।"),
        ("ne", "जन्म मिति: १५ चैत्र २०८२।"),
    )
    for lang, text in cases:
        spans = _date_spans(lang, text)
        assert len(spans) == 1
        start, end, _surface, _calendar = spans[0]
        assert not unicodedata.category(text[start]).startswith("M")
        assert end == len(text) or not unicodedata.category(text[end]).startswith("M")


def test_persian_and_urdu_letter_variants_and_bidi_isolates_match() -> None:
    arabic_yeh = "تاريخ تولد: ۱۵ رمضان ۱۴۴۷"
    isolated = "تاریخ پیدائش: ۱۵ رمضان \u2066۱۴۴۷\u2069 ہجری۔"

    assert [span[2] for span in _date_spans("fa", arabic_yeh)] == ["۱۵ رمضان ۱۴۴۷"]
    assert [span[2:] for span in _date_spans("ur", isolated)] == [
        ("۱۵ رمضان \u2066۱۴۴۷\u2069 ہجری", HIJRI_CALENDAR)
    ]


def test_semantic_merger_carries_the_calendar_code() -> None:
    text = "تاريخ الميلاد: ١٥ رمضان ١٤٤٧ هـ."
    result = _apply_pii_smart_merging(_empty_detector(text), _STUB_MODEL, "ar")

    assert [
        (entity.label, entity.metadata["calendar"]) for entity in result.entities
    ] == [("date", HIJRI_CALENDAR)]

    # A model span overlapping the rule keeps the calendar on the union.
    surface_start = text.index("١٥")
    model_entity = {
        "entity_type": "date",
        "score": 0.9,
        "start": surface_start,
        "end": surface_start + 8,
        "word": text[surface_start : surface_start + 8],
    }
    merged = merge_entities_with_semantic_units(
        [model_entity], text, patterns=get_patterns_for_language("ar")
    )
    assert [entity.get("calendar") for entity in merged] == [HIJRI_CALENDAR]


def test_coalesced_clusters_keep_only_an_agreed_calendar_code() -> None:
    text = "1 2 3 4"
    tagged = PIIPattern(r"1 2", "date", priority=5, calendar=HEBREW_CALENDAR)
    other = PIIPattern(r"3 4", "date", priority=5, calendar=HIJRI_CALENDAR)
    untagged = PIIPattern(r"3 4", "date", priority=5)
    model = [{"entity_type": "date", "score": 0.9, "start": 0, "end": 7, "word": text}]

    agreed = merge_entities_with_semantic_units(
        model, text, patterns=[tagged, untagged]
    )
    mixed = merge_entities_with_semantic_units(model, text, patterns=[tagged, other])

    assert [entity.get("calendar") for entity in agreed] == [HEBREW_CALENDAR]
    assert ["calendar" in entity for entity in mixed] == [False]


@pytest.mark.parametrize(("lang", "text", "surface", "calendar"), _POSITIVE_CASES[:7])
def test_detected_dates_keep_the_shift_placeholder(
    lang: str, text: str, surface: str, calendar: str
) -> None:
    assert calendar in CALENDAR_CODES
    assert _shift_date(surface, 30, lang=lang) == "[DATE_SHIFTED]"


def test_no_date_value_reaches_logs_audit_or_span_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    for lang, text, surface, calendar in _POSITIVE_CASES[:7]:
        result = _pipeline(lang).run(text, method="mask")
        date_span = next(
            span for span in result.spans if span.canonical_label == "DATE"
        )
        audit = json.dumps(result.audit_record, ensure_ascii=False, default=str)
        metadata = json.dumps(dict(date_span.metadata), ensure_ascii=False, default=str)

        assert date_span.metadata["calendar"] == calendar
        assert surface not in result.redacted_text
        assert surface not in metadata
        # ASCII digit runs also occur inside hex digests, so beyond the whole
        # surface check the day and year tokens written in native numerals.
        native_values = [
            token
            for token in surface.split()
            if not token.isascii()
            and (any(char.isnumeric() for char in token) or "״" in token)
        ]
        for leaked in (surface, *native_values):
            assert leaked not in audit
            assert leaked not in caplog.text


def test_calendar_fixtures_cover_every_calendar_with_exact_spans() -> None:
    rows = _calendar_rows()
    calendars = set()

    for row in rows:
        assert row["metadata"]["synthetic"] is True
        lang = row["language"]
        expected = {
            (
                span["start"],
                span["end"],
                span["text"],
                span.get("metadata", {}).get("calendar"),
            )
            for span in row["gold_spans"]
        }
        calendars.update(item[3] for item in expected)

        assert set(_date_spans(lang, row["text"])) == expected
        masked = _pipeline(lang).run(row["text"], method="mask").redacted_text
        assert masked == row["metadata"]["expected_output"]["text"].replace(
            "[DATE]", "[date]"
        )

    assert calendars == CALENDAR_CODES
