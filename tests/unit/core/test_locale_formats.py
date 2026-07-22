"""Tests for locale-aware date and number normalization helpers."""

from __future__ import annotations

import re

import pytest

from openmed.core.locale_formats import (
    BACKLOG_LOCALES,
    LOCALE_DATE_ORDER,
    LOCALE_NUMBER_SEP,
    LOCALE_PII_FORMATS,
    WIRED_LOCALES,
    format_hint,
    normalize_number,
    parse_date,
)
from openmed.core.pii_i18n import (
    AFRICAN_FR_PT_PHONE_PREFIXES,
    LOCALE_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    get_patterns_for_language,
    validate_egyptian_national_id,
    validate_moroccan_cin,
)
from openmed.core.safety_sweep import safety_sweep


def test_parse_date_respects_german_day_month_order() -> None:
    result = parse_date("03.04.2026", "de")

    assert result.normalized == (2026, 4, 3)
    assert result.order == "dmy"
    assert result.ambiguous is False


def test_parse_date_respects_english_us_month_day_order() -> None:
    result = parse_date("03/04/2026", "en")

    assert result.normalized == (2026, 3, 4)
    assert result.order == "mdy"
    assert result.ambiguous is False


def test_parse_date_respects_hungarian_year_month_day_order() -> None:
    result = parse_date("2026.03.04", "hu")

    assert result.normalized == (2026, 3, 4)
    assert result.order == "ymd"
    assert result.ambiguous is False


def test_parse_date_without_locale_signal_flags_ambiguous_date() -> None:
    result = parse_date("03/04/2026", None)

    assert result.normalized is None
    assert result.ambiguous is True
    assert {candidate.normalized for candidate in result.candidates} == {
        (2026, 3, 4),
        (2026, 4, 3),
    }


def test_parse_date_without_locale_resolves_unambiguous_component_values() -> None:
    result = parse_date("13/04/2026", None)

    assert result.normalized == (2026, 4, 13)
    assert result.order == "dmy"
    assert result.ambiguous is False


def test_normalize_number_resolves_comma_and_period_decimal_locales() -> None:
    assert normalize_number("1.234,56", "de").value == "1234.56"
    assert normalize_number("1,234.56", "en").value == "1234.56"


def test_normalize_number_without_locale_signal_flags_ambiguous_separator() -> None:
    result = normalize_number("1,234", None)

    assert result.value is None
    assert result.ambiguous is True


def test_format_hint_covers_every_wired_language() -> None:
    expected_languages = SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES

    assert expected_languages <= set(LOCALE_DATE_ORDER)
    assert expected_languages <= set(LOCALE_NUMBER_SEP)
    for lang in sorted(expected_languages):
        hint = format_hint(lang)
        assert hint.date_order == LOCALE_DATE_ORDER[lang]
        assert hint.number == LOCALE_NUMBER_SEP[lang]


def test_korean_full_language_pack_is_wired_not_backlog() -> None:
    assert "ko" in WIRED_LOCALES
    assert "ko" not in BACKLOG_LOCALES


def test_format_hint_covers_backlog_language_pack_locales() -> None:
    assert WIRED_LOCALES <= set(LOCALE_DATE_ORDER)
    assert BACKLOG_LOCALES <= set(LOCALE_DATE_ORDER)
    assert WIRED_LOCALES <= set(LOCALE_NUMBER_SEP)
    assert BACKLOG_LOCALES <= set(LOCALE_NUMBER_SEP)


@pytest.mark.parametrize("lang", sorted(WIRED_LOCALES | BACKLOG_LOCALES))
def test_format_hint_returns_date_order_and_separators(lang: str) -> None:
    hint = format_hint(lang)

    assert hint.lang == lang
    assert hint.date_order in {"dmy", "mdy", "ymd"}
    assert hint.decimal_separator
    assert hint.number.decimal_separators


@pytest.mark.parametrize("locale", ["ar_eg", "ar_ma"])
def test_north_africa_locale_format_examples_match_declared_data(locale: str) -> None:
    for spec in LOCALE_PII_FORMATS[locale]:
        for example in spec.examples:
            assert re.search(spec.pattern, example, spec.flags), (spec.name, example)


@pytest.mark.parametrize(
    ("locale", "format_name", "contextual_text", "value", "invalid", "validator"),
    [
        (
            "ar_EG",
            "egyptian_national_id",
            "الرقم القومي: 29801011234567",
            "29801011234567",
            "29913311234567",
            validate_egyptian_national_id,
        ),
        (
            "ar_MA",
            "moroccan_cin",
            "رقم البطاقة الوطنية: AB123456",
            "AB123456",
            "123456",
            validate_moroccan_cin,
        ),
    ],
)
def test_north_africa_national_ids_round_trip_through_locale_sweep(
    locale,
    format_name,
    contextual_text,
    value,
    invalid,
    validator,
) -> None:
    locale_key = locale.casefold()
    declared = next(
        spec for spec in LOCALE_PII_FORMATS[locale_key] if spec.name == format_name
    )
    runtime = next(
        pattern
        for pattern in LOCALE_PII_PATTERNS[locale_key]
        if pattern.pattern == declared.pattern
    )

    assert validator(value) is True
    assert validator(invalid) is False
    assert runtime.validator is validator
    assert runtime in get_patterns_for_language("ar", locale=locale)

    swept = safety_sweep(contextual_text, [], lang="ar", locale=locale)
    assert [(entity.text, entity.label) for entity in swept] == [(value, "national_id")]


@pytest.mark.parametrize(
    ("locale", "phone"),
    [
        ("fr_sn", "+221 77 642 18 35"),
        ("fr_ci", "+225 07 48 26 15 39"),
        ("fr_cm", "+237 6 71 24 83 59"),
        ("pt_mz", "+258 84 362 7185"),
        ("pt_ao", "+244 923 461 785"),
    ],
)
def test_african_fr_pt_phone_formats_match_country_prefix(locale, phone) -> None:
    phone_patterns = [
        pattern
        for pattern in LOCALE_PII_PATTERNS[locale]
        if pattern.entity_type == "phone_number"
    ]

    assert phone.startswith(AFRICAN_FR_PT_PHONE_PREFIXES[locale])
    assert any(
        re.fullmatch(pattern.pattern, phone, pattern.flags)
        for pattern in phone_patterns
    )


@pytest.mark.parametrize(
    ("lang", "locale", "contextual_text", "bare_value"),
    [
        ("fr", "fr_SN", "CNI: 1002199012345", "1002199012345"),
        ("pt", "pt_AO", "Bilhete de Identidade: 009523666HO041", "009523666HO041"),
    ],
)
def test_context_only_national_id_formats_do_not_claim_a_validator(
    lang,
    locale,
    contextual_text,
    bare_value,
) -> None:
    detected = safety_sweep(contextual_text, [], lang=lang, locale=locale)
    without_context = safety_sweep(bare_value, [], lang=lang, locale=locale)
    matching_patterns = [
        pattern
        for pattern in LOCALE_PII_PATTERNS[locale.casefold()]
        if pattern.entity_type == "national_id"
        and re.fullmatch(pattern.pattern, bare_value, pattern.flags)
    ]

    assert [(span.text, span.label) for span in detected] == [
        (bare_value, "national_id")
    ]
    assert without_context == []
    assert matching_patterns
    assert all(pattern.validator is None for pattern in matching_patterns)
    assert all(pattern.safety_sweep_requires_context for pattern in matching_patterns)
