from __future__ import annotations

import hashlib
import json
import random
import re
import string
import unicodedata
from datetime import date

import pytest

from openmed.core.labels import DATE, DATE_OF_BIRTH
from openmed.core.pii_i18n import get_patterns_for_language
from openmed.core.safety_sweep import safety_sweep
from openmed.training.synthetic import (
    SUPPORTED_LOCALE_PHI_LANGUAGES,
    LocalePhiGenerator,
    generate_locale_phi_examples,
)
from openmed.training.synthetic.native_dates import (
    BIKRAM_SAMBAT_CALENDAR,
    DATE_CALENDARS,
    DATE_DETECTION_GAPS,
    DATE_FORMS,
    DEFAULT_NATIVE_DATE_PROFILES,
    ETHIOPIAN_CALENDAR,
    GREGORIAN_CALENDAR,
    HIJRI_CALENDAR,
    MONTH_NAME_FORM,
    NATIVE_DATE_PROFILES,
    NUMERAL_SYSTEMS,
    SOLAR_HIJRI_CALENDAR,
    THAI_BUDDHIST_CALENDAR,
    NativeDateProfile,
    ethiopic_numeral,
    generate_native_date,
    months_in_year,
    native_date_profiles_for,
    native_month_names,
    render_native_date,
    resolve_native_date_profiles,
    sample_native_date,
    sampled_month_length,
)

_DATE_FIELD_BY_LABEL = {DATE_OF_BIRTH: "date_of_birth", DATE: "date"}
_SPAN_METADATA_KEYS = {
    "calendar",
    "date_profile",
    "detection_gap",
    "field",
    "form",
    "numeral_system",
    "synthetic",
}
_GENERATED_PROFILES = tuple(
    name
    for name, profile in NATIVE_DATE_PROFILES.items()
    if profile.language in SUPPORTED_LOCALE_PHI_LANGUAGES
)
_BIDI_CONTROLS = frozenset("؜‎‏‪‫‬‭‮⁦⁧⁨⁩")
_ZERO_DIGIT = {"latn": "0", "arab": "٠", "arabext": "۰", "thai": "๐", "deva": "०"}
_ETHIOPIC_VALUES = {
    **{char: index + 1 for index, char in enumerate("፩፪፫፬፭፮፯፰፱")},
    **{char: (index + 1) * 10 for index, char in enumerate("፲፳፴፵፶፷፸፹፺")},
}


def _parse_number(text: str, numeral_system: str) -> int:
    if numeral_system == "ethi":
        total = group = 0
        for char in text:
            if char == "፻":
                total += (group or 1) * 100
                group = 0
            else:
                group += _ETHIOPIC_VALUES[char]
        return total + group
    zero = ord(_ZERO_DIGIT[numeral_system])
    assert all(zero <= ord(char) <= zero + 9 for char in text), numeral_system
    return int("".join(str(ord(char) - zero) for char in text))


def _parse_rendered(profile: NativeDateProfile, text: str) -> tuple[int, int, int]:
    """Invert a profile template so tests can check calendar validity."""

    names: tuple[str, ...] = ()
    parts = []
    for literal, field_name, _, _ in string.Formatter().parse(profile.template):
        parts.append(re.escape(literal))
        if field_name == "month_name":
            names = native_month_names(profile.calendar, profile.language)
            alternation = "|".join(map(re.escape, sorted(names, key=len)[::-1]))
            parts.append(f"(?P<month_name>{alternation})")
        elif field_name is not None:
            parts.append(rf"(?P<{field_name}>[^\s/.\-]+)")
    match = re.fullmatch("".join(parts), text)
    assert match is not None, profile.name
    year = _parse_number(match["year"], profile.numeral_system)
    day = _parse_number(match["day"], profile.numeral_system)
    if profile.form == MONTH_NAME_FORM:
        month = names.index(match["month_name"]) + 1
    else:
        month = _parse_number(match["month"], profile.numeral_system)
    return year, month, day


def _is_valid_calendar_date(calendar: str, year: int, month: int, day: int) -> bool:
    """Independent month-length rules used to check sampled output."""

    if day < 1:
        return False
    if calendar in {GREGORIAN_CALENDAR, THAI_BUDDHIST_CALENDAR}:
        offset = 543 if calendar == THAI_BUDDHIST_CALENDAR else 0
        try:
            date(year - offset, month, day)
        except ValueError:
            return False
        return True
    if calendar == ETHIOPIAN_CALENDAR:
        if 1 <= month <= 12:
            return day <= 30
        return month == 13 and day <= (6 if year % 4 == 3 else 5)
    if not 1 <= month <= 12:
        return False
    if calendar == SOLAR_HIJRI_CALENDAR:
        # Esfand 30 exists only in leap years; the sampler never draws it.
        return day <= (31 if month <= 6 else 30 if month <= 11 else 29)
    # Lunar Hijri and Bikram Sambat months always have at least 29 days.
    return day <= 29


def _date_spans(example):
    return [span for span in example.gold_spans if span.label in _DATE_FIELD_BY_LABEL]


def _training_digest(examples) -> str:
    payload = json.dumps(
        [example.to_training_item() for example in examples],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _detected_exactly(example, span) -> bool:
    date_patterns = [
        pattern
        for pattern in get_patterns_for_language(example.language, example.locale)
        if pattern.entity_type == "date"
    ]
    found = safety_sweep(
        example.text,
        [],
        lang=example.language,
        locale=example.locale,
        patterns=date_patterns,
    )
    return any((item.start, item.end) == (span.start, span.end) for item in found)


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------


def test_profile_registry_is_keyed_by_name_and_uses_known_codes():
    assert NATIVE_DATE_PROFILES
    for name, profile in NATIVE_DATE_PROFILES.items():
        assert profile.name == name
        assert profile.calendar in DATE_CALENDARS
        assert profile.numeral_system in NUMERAL_SYSTEMS
        assert profile.form in DATE_FORMS
        assert profile.detection_gap is None or (
            profile.detection_gap in DATE_DETECTION_GAPS
        )
        assert profile in native_date_profiles_for(profile.language)


@pytest.mark.parametrize(
    ("language", "calendar"),
    (
        ("fa", SOLAR_HIJRI_CALENDAR),
        ("th", THAI_BUDDHIST_CALENDAR),
        ("am", ETHIOPIAN_CALENDAR),
        ("ar", HIJRI_CALENDAR),
        ("ur", HIJRI_CALENDAR),
        ("ne", BIKRAM_SAMBAT_CALENDAR),
    ),
)
def test_each_target_language_has_native_calendar_profiles(language, calendar):
    profiles = native_date_profiles_for(language)
    calendars = {profile.calendar for profile in profiles}
    forms = {profile.form for profile in profiles if profile.calendar == calendar}

    assert calendar in calendars
    assert forms == DATE_FORMS
    default = NATIVE_DATE_PROFILES[DEFAULT_NATIVE_DATE_PROFILES[language]]
    assert default.language == language
    assert default.calendar == calendar


def test_hijri_profiles_carry_an_era_marker_inside_the_template():
    for language, marker in (("ar", "هـ"), ("ur", "ہجری")):
        for profile in native_date_profiles_for(language):
            if profile.calendar == HIJRI_CALENDAR:
                assert profile.template.endswith(f" {marker}")


def test_native_date_profiles_for_unknown_language_is_empty():
    assert native_date_profiles_for("xx") == ()


# ---------------------------------------------------------------------------
# Calendar rules and numerals against literal references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("calendar", "year", "month", "expected"),
    (
        # 2024 C.E. is a Gregorian leap year, so B.E. 2567 February has 29 days.
        (THAI_BUDDHIST_CALENDAR, 2567, 2, 29),
        (THAI_BUDDHIST_CALENDAR, 2566, 2, 28),
        (THAI_BUDDHIST_CALENDAR, 2567, 4, 30),
        (GREGORIAN_CALENDAR, 2000, 2, 29),
        (GREGORIAN_CALENDAR, 1900, 2, 28),
        # Pagume 2015 had six days: Ethiopian New Year 2016 was 12 Sept 2023.
        (ETHIOPIAN_CALENDAR, 2015, 13, 6),
        (ETHIOPIAN_CALENDAR, 2011, 13, 6),
        (ETHIOPIAN_CALENDAR, 2016, 13, 5),
        (ETHIOPIAN_CALENDAR, 2016, 1, 30),
        (SOLAR_HIJRI_CALENDAR, 1402, 1, 31),
        (SOLAR_HIJRI_CALENDAR, 1402, 6, 31),
        (SOLAR_HIJRI_CALENDAR, 1402, 7, 30),
        (SOLAR_HIJRI_CALENDAR, 1402, 11, 30),
        (SOLAR_HIJRI_CALENDAR, 1403, 12, 29),
        (HIJRI_CALENDAR, 1447, 9, 29),
        (BIKRAM_SAMBAT_CALENDAR, 2081, 3, 29),
    ),
)
def test_sampled_month_lengths_match_reference_rules(calendar, year, month, expected):
    assert sampled_month_length(calendar, year, month) == expected


def test_month_counts_and_month_range_errors():
    assert months_in_year(ETHIOPIAN_CALENDAR) == 13
    assert months_in_year(SOLAR_HIJRI_CALENDAR) == 12
    with pytest.raises(ValueError, match="outside"):
        sampled_month_length(SOLAR_HIJRI_CALENDAR, 1402, 13)
    with pytest.raises(ValueError, match="unknown date calendar"):
        months_in_year("lunisolar")
    with pytest.raises(ValueError, match="thai_buddhist"):
        sampled_month_length(THAI_BUDDHIST_CALENDAR, 543, 1)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (1, "፩"),
        (9, "፱"),
        (10, "፲"),
        (12, "፲፪"),
        (30, "፴"),
        (100, "፻"),
        (101, "፻፩"),
        (1990, "፲፱፻፺"),
        (2000, "፳፻"),
        (2016, "፳፻፲፮"),
        (9999, "፺፱፻፺፱"),
    ),
)
def test_ethiopic_numerals_match_reference_values(value, expected):
    assert ethiopic_numeral(value) == expected
    assert _parse_number(expected, "ethi") == value


@pytest.mark.parametrize("value", (0, 10000, -1))
def test_ethiopic_numerals_reject_out_of_range_values(value):
    with pytest.raises(ValueError, match="1-9999"):
        ethiopic_numeral(value)


@pytest.mark.parametrize(
    ("name", "components", "expected"),
    (
        ("fa_solar_hijri_numeric_arabext", (1402, 5, 20), "۱۴۰۲/۰۵/۲۰"),
        ("fa_solar_hijri_numeric_latn", (1402, 5, 20), "1402/05/20"),
        ("fa_solar_hijri_month_name", (1402, 5, 9), "۹ مرداد ۱۴۰۲"),
        ("th_thai_buddhist_numeric_thai", (2567, 4, 18), "๑๘/๐๔/๒๕๖๗"),
        ("th_thai_buddhist_numeric_latn", (2567, 4, 18), "18/04/2567"),
        ("th_thai_buddhist_month_name", (2567, 2, 8), "๘ กุมภาพันธ์ พ.ศ. ๒๕๖๗"),
        ("am_ethiopian_numeric_latn", (2016, 5, 12), "12/05/2016"),
        ("am_ethiopian_numeric_ethi", (2016, 5, 12), "፲፪/፭/፳፻፲፮"),
        ("am_ethiopian_month_name", (2015, 13, 6), "6 ጳጉሜ 2015"),
        ("ar_hijri_numeric_arab", (1447, 9, 15), "١٥/٠٩/١٤٤٧ هـ"),
        ("ar_hijri_month_name", (1447, 3, 1), "١ ربيع الأول ١٤٤٧ هـ"),
        ("ar_gregorian_numeric_arab", (2024, 4, 18), "١٨/٠٤/٢٠٢٤"),
        ("ur_hijri_numeric_arabext", (1447, 9, 15), "۱۵/۰۹/۱۴۴۷ ہجری"),
        ("ur_hijri_month_name", (1447, 12, 10), "۱۰ ذوالحجہ ۱۴۴۷ ہجری"),
        ("ur_gregorian_numeric_arabext", (2024, 4, 18), "۱۸/۰۴/۲۰۲۴"),
        ("ne_bikram_sambat_numeric_deva", (2081, 4, 15), "२०८१/०४/१५"),
        ("ne_bikram_sambat_month_name", (2081, 4, 15), "१५ साउन २०८१"),
    ),
)
def test_render_native_date_matches_literal_references(name, components, expected):
    profile = NATIVE_DATE_PROFILES[name]

    assert render_native_date(profile, *components) == expected
    assert _parse_rendered(profile, expected) == components


@pytest.mark.parametrize(
    ("name", "components"),
    (
        ("am_ethiopian_numeric_latn", (2016, 13, 6)),
        ("th_thai_buddhist_numeric_thai", (2566, 2, 29)),
        ("fa_solar_hijri_numeric_arabext", (1402, 7, 31)),
        ("ar_hijri_numeric_arab", (1447, 9, 31)),
        ("ar_gregorian_numeric_arab", (2023, 13, 1)),
        ("ne_bikram_sambat_numeric_deva", (2081, 4, 0)),
    ),
)
def test_render_native_date_rejects_impossible_dates_without_echoing(name, components):
    profile = NATIVE_DATE_PROFILES[name]

    with pytest.raises(ValueError) as excinfo:
        render_native_date(profile, *components)

    message = str(excinfo.value)
    assert profile.calendar in message
    assert not any(char.isdigit() for char in message)


# ---------------------------------------------------------------------------
# Direct sampling validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", tuple(NATIVE_DATE_PROFILES))
def test_sampled_dates_are_valid_in_their_calendar(name):
    profile = NATIVE_DATE_PROFILES[name]
    birth_years = set()
    visit_years = set()

    for seed in range(300):
        for field in ("date_of_birth", "date"):
            rendered, metadata = generate_native_date(
                profile, random.Random(f"{name}|{seed}|{field}"), field=field
            )
            year, month, day = _parse_rendered(profile, rendered)
            assert _is_valid_calendar_date(profile.calendar, year, month, day), name
            assert metadata == profile.span_metadata()
            (birth_years if field == "date_of_birth" else visit_years).add(year)

    # A follow-up date never precedes the date of birth.
    assert max(birth_years) < min(visit_years)


def test_ethiopian_sampling_reaches_pagume_with_the_leap_rule():
    rng = random.Random(3886)
    pagume_days = {}
    for _ in range(20000):
        year, month, day = sample_native_date(ETHIOPIAN_CALENDAR, rng, field="date")
        if month == 13:
            pagume_days.setdefault(year % 4 == 3, set()).add(day)

    assert max(pagume_days[True]) == 6
    assert max(pagume_days[False]) == 5


def test_solar_hijri_sampling_uses_31_day_months_but_never_esfand_30():
    rng = random.Random(1402)
    seen = [sample_native_date(SOLAR_HIJRI_CALENDAR, rng) for _ in range(4000)]

    assert any(day == 31 for _, month, day in seen if month <= 6)
    assert all(day <= 30 for _, month, day in seen if month > 6)
    assert all(day <= 29 for _, month, day in seen if month == 12)


def test_sample_native_date_rejects_unknown_field_and_calendar():
    with pytest.raises(ValueError, match="unknown date field"):
        sample_native_date(HIJRI_CALENDAR, random.Random(1), field="admission")
    with pytest.raises(ValueError, match="unknown date calendar"):
        sample_native_date("lunisolar", random.Random(1))


# ---------------------------------------------------------------------------
# Generator integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _GENERATED_PROFILES)
def test_profiled_examples_have_exact_spans_and_value_free_metadata(name):
    profile = NATIVE_DATE_PROFILES[name]
    language = profile.language

    for seed in range(6):
        example = LocalePhiGenerator(
            seed=seed, date_profiles={language: name}
        ).generate(language)
        date_spans = _date_spans(example)
        assert {span.label for span in date_spans} == set(_DATE_FIELD_BY_LABEL)

        for span in example.gold_spans:
            assert example.text[span.start : span.end] == span.text
        for span in date_spans:
            field = _DATE_FIELD_BY_LABEL[span.label]
            year, month, day = _parse_rendered(profile, span.text)
            assert _is_valid_calendar_date(profile.calendar, year, month, day)
            assert set(span.metadata) == _SPAN_METADATA_KEYS
            assert span.metadata == {
                "synthetic": True,
                "field": field,
                **profile.span_metadata(),
            }
            # Span metadata never carries the rendered value or any number.
            values = [str(value) for value in span.metadata.values()]
            assert all(span.text not in value for value in values)
            assert not any(char.isdigit() for value in values for char in value)
            assert not any(char in _BIDI_CONTROLS for char in span.text)
            # Spans never split a base character from its combining marks.
            assert not unicodedata.category(span.text[0]).startswith("M")
            if span.end < len(example.text):
                assert not unicodedata.category(example.text[span.end]).startswith("M")

        expected_gaps = (
            {}
            if profile.detection_gap is None
            else {field: profile.detection_gap for field in ("date_of_birth", "date")}
        )
        assert example.metadata["date_profile"] == name
        assert example.metadata["date_detection_gaps"] == expected_gaps


@pytest.mark.parametrize("name", _GENERATED_PROFILES)
def test_date_spans_are_detected_or_marked_as_known_gaps(name):
    profile = NATIVE_DATE_PROFILES[name]

    for seed in range(4):
        example = LocalePhiGenerator(
            seed=seed, date_profiles={profile.language: name}
        ).generate(profile.language)
        for span in _date_spans(example):
            if span.metadata["detection_gap"] is None:
                assert _detected_exactly(example, span), (name, seed, span.label)
            else:
                gap = span.metadata["detection_gap"]
                gaps = example.metadata["date_detection_gaps"]
                assert gap in DATE_DETECTION_GAPS
                assert gaps[span.metadata["field"]] == gap


def test_month_name_spans_cover_combining_marks_and_rtl_text():
    marked = 0
    for seed in range(12):
        thai = LocalePhiGenerator(
            seed=seed, date_profiles={"th": "th_thai_buddhist_month_name"}
        ).generate("th")
        persian = LocalePhiGenerator(
            seed=seed, date_profiles={"fa": "fa_solar_hijri_month_name"}
        ).generate("fa")
        for example in (thai, persian):
            for span in _date_spans(example):
                assert example.text[span.start : span.end] == span.text
                marked += any(unicodedata.category(char) == "Mn" for char in span.text)
        for span in _date_spans(persian):
            # Right-to-left month name next to Extended Arabic-Indic digits.
            classes = {unicodedata.bidirectional(char) for char in span.text}
            assert {"AL", "EN"} <= classes

    assert marked > 0


def test_profiled_generation_is_deterministic_per_seed():
    first = generate_locale_phi_examples(
        seed=3886, date_profiles=DEFAULT_NATIVE_DATE_PROFILES
    )
    second = generate_locale_phi_examples(
        seed=3886, date_profiles=DEFAULT_NATIVE_DATE_PROFILES
    )

    assert first == second
    assert _training_digest(first) == _training_digest(second)


def test_unseeded_profiled_generation_still_renders_valid_dates():
    profile = NATIVE_DATE_PROFILES["am_ethiopian_numeric_ethi"]
    example = LocalePhiGenerator(date_profiles={"am": profile}).generate("am")

    for span in _date_spans(example):
        assert _is_valid_calendar_date(
            profile.calendar, *_parse_rendered(profile, span.text)
        )


def test_default_output_is_unchanged_without_profiles():
    default = LocalePhiGenerator(seed=17).generate_all()

    assert LocalePhiGenerator(seed=17, date_profiles=None).generate_all() == default
    assert LocalePhiGenerator(seed=17, date_profiles={}).generate_all() == default
    assert _training_digest(generate_locale_phi_examples(seed=17)) == (
        _training_digest(default)
    )
    for example in default:
        assert "date_profile" not in example.metadata
        assert "date_detection_gaps" not in example.metadata
        for span in _date_spans(example):
            assert span.metadata == {
                "synthetic": True,
                "field": _DATE_FIELD_BY_LABEL[span.label],
            }


def test_profiles_only_change_the_requested_language_dates():
    default = {
        example.language: example
        for example in LocalePhiGenerator(seed=5).generate_all()
    }
    profiled = {
        example.language: example
        for example in LocalePhiGenerator(
            seed=5, date_profiles={"fa": "fa_solar_hijri_numeric_arabext"}
        ).generate_all()
    }

    for language, example in default.items():
        if language != "fa":
            assert profiled[language] == example
    changed = profiled["fa"]
    for before, after in zip(default["fa"].gold_spans, changed.gold_spans):
        assert before.label == after.label
        if before.label not in _DATE_FIELD_BY_LABEL:
            assert (before.text, dict(before.metadata)) == (
                after.text,
                dict(after.metadata),
            )


def test_custom_profile_instances_are_accepted():
    profile = NativeDateProfile(
        name="fa_gregorian_numeric_arabext",
        language="fa",
        calendar=GREGORIAN_CALENDAR,
        numeral_system="arabext",
        form="numeric",
        template="{year}-{month}-{day}",
    )
    example = LocalePhiGenerator(seed=3, date_profiles={"fa": profile}).generate("fa")

    for span in _date_spans(example):
        year, month, day = _parse_rendered(profile, span.text)
        assert date(year, month, day)
        assert span.metadata["date_profile"] == "fa_gregorian_numeric_arabext"


# ---------------------------------------------------------------------------
# Selection and profile validation
# ---------------------------------------------------------------------------


def test_resolve_native_date_profiles_accepts_names_and_instances():
    profile = NATIVE_DATE_PROFILES["th_thai_buddhist_month_name"]

    assert resolve_native_date_profiles(None) == {}
    assert resolve_native_date_profiles(
        {"fa": "fa_solar_hijri_month_name", "th": profile}
    ) == {"fa": NATIVE_DATE_PROFILES["fa_solar_hijri_month_name"], "th": profile}


@pytest.mark.parametrize(
    ("selection", "error", "match"),
    (
        ({"fa": "fa_lunar"}, ValueError, "unknown native date profile"),
        ({"ar": "fa_solar_hijri_month_name"}, ValueError, "renders 'fa'"),
        ({"fa": 7}, TypeError, "profile names or NativeDateProfile"),
        (["fa_solar_hijri_month_name"], TypeError, "must map language codes"),
    ),
)
def test_invalid_date_profile_selection_is_rejected(selection, error, match):
    with pytest.raises(error, match=match):
        LocalePhiGenerator(seed=1, date_profiles=selection)


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"calendar": "lunisolar"}, "unknown date calendar"),
        ({"numeral_system": "roman"}, "unknown numeral system"),
        ({"form": "spoken"}, "unknown date form"),
        ({"detection_gap": "unknown"}, "unknown date detection gap"),
        ({"template": "{year}/{month}"}, "must use exactly"),
        ({"template": "{year}/{month_name}/{day}"}, "must use exactly"),
        (
            {"form": "month_name", "template": "{day} {month_name} {year}"},
            "no 'gregorian' month names",
        ),
    ),
)
def test_native_date_profile_validates_its_fields(overrides, match):
    fields = {
        "name": "custom",
        "language": "fa",
        "calendar": GREGORIAN_CALENDAR,
        "numeral_system": "latn",
        "form": "numeric",
        "template": "{year}/{month}/{day}",
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        NativeDateProfile(**fields)


def test_native_month_names_rejects_unregistered_pairs():
    assert len(native_month_names(ETHIOPIAN_CALENDAR, "am")) == 13
    with pytest.raises(ValueError, match="month names"):
        native_month_names(GREGORIAN_CALENDAR, "fa")
