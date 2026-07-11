"""Tests for locale-aware date and number normalization helpers."""

from __future__ import annotations

import pytest

from openmed.core.locale_formats import (
    BACKLOG_LOCALES,
    LOCALE_DATE_ORDER,
    LOCALE_NUMBER_SEP,
    WIRED_LOCALES,
    format_hint,
    normalize_number,
    parse_date,
)
from openmed.core.pii_i18n import NATIONAL_ID_ONLY_LANGUAGES, SUPPORTED_LANGUAGES


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
