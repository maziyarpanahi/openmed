"""Locale-aware date and number format normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Final, Mapping

DateOrder = str

_SPACE_GROUP_SEPARATORS: Final = (" ", "\u00a0", "\u202f")
_ARABIC_DECIMAL_SEPARATOR: Final = "\u066b"
_ARABIC_GROUP_SEPARATOR: Final = "\u066c"
_NUMBER_SEPARATOR_CHARS: Final = frozenset(
    {".", ",", "'", *_SPACE_GROUP_SEPARATORS, _ARABIC_DECIMAL_SEPARATOR}
)
_ALL_NUMBER_SEPARATOR_CHARS: Final = _NUMBER_SEPARATOR_CHARS | {_ARABIC_GROUP_SEPARATOR}
_DATE_RE: Final = re.compile(
    r"^\s*(?P<a>\d{1,4})(?P<sep>[./\-\s])(?P<b>\d{1,2})"
    r"(?P=sep)(?P<c>\d{1,4})\s*$"
)
_CJK_DATE_RE: Final = re.compile(
    r"^\s*(?P<year>\d{2,4})\s*(?:\u5e74|\ub144)\s*"
    r"(?P<month>\d{1,2})\s*(?:\u6708|\uc6d4)\s*"
    r"(?P<day>\d{1,2})\s*(?:\u65e5|\uc77c)?\s*$"
)


@dataclass(frozen=True, eq=False)
class NumberSeparators:
    """Locale number separators.

    Args:
        decimal: Preferred decimal separator for the locale.
        thousands: Accepted thousands/grouping separators.
        decimal_alternates: Extra decimal separators accepted for the locale.
    """

    decimal: str
    thousands: tuple[str, ...]
    decimal_alternates: tuple[str, ...] = ()

    @property
    def decimal_separators(self) -> tuple[str, ...]:
        """Return preferred and alternate decimal separators without duplicates."""
        return _ordered_unique((self.decimal, *self.decimal_alternates))

    def __eq__(self, other: object) -> bool:
        """Compare with another instance or a ``(decimal, thousands)`` tuple."""
        if isinstance(other, NumberSeparators):
            return (
                self.decimal == other.decimal
                and self.thousands == other.thousands
                and self.decimal_alternates == other.decimal_alternates
            )
        if isinstance(other, tuple):
            return (self.decimal, self.thousands) == other
        return False

    def __getitem__(self, key: str) -> str | tuple[str, ...]:
        """Expose dictionary-style access for simple table consumers."""
        if key == "decimal":
            return self.decimal
        if key == "thousands":
            return self.thousands
        if key == "decimal_alternates":
            return self.decimal_alternates
        raise KeyError(key)


@dataclass(frozen=True)
class DateCandidate:
    """One valid interpretation of a numeric date."""

    year: int
    month: int
    day: int
    order: DateOrder

    @property
    def normalized(self) -> tuple[int, int, int]:
        """Return the normalized ``(year, month, day)`` tuple."""
        return self.year, self.month, self.day


@dataclass(frozen=True, eq=False)
class ParsedDate:
    """Result from :func:`parse_date`.

    Args:
        year: Four-digit Gregorian year when a single interpretation exists.
        month: Gregorian month when a single interpretation exists.
        day: Gregorian day when a single interpretation exists.
        ambiguous: True when multiple interpretations exist or the locale order
            does not cleanly resolve the input.
        order: Date order used for the normalized value, if any.
        candidates: Valid interpretations considered by the parser.
        reason: Machine-readable explanation for non-normalized results.
    """

    year: int | None
    month: int | None
    day: int | None
    ambiguous: bool = False
    order: DateOrder | None = None
    candidates: tuple[DateCandidate, ...] = ()
    reason: str | None = None

    @property
    def normalized(self) -> tuple[int, int, int] | None:
        """Return ``(year, month, day)`` when the date is unambiguous."""
        if self.year is None or self.month is None or self.day is None:
            return None
        return self.year, self.month, self.day

    def __eq__(self, other: object) -> bool:
        """Compare with another result or a normalized ``(year, month, day)``."""
        if isinstance(other, ParsedDate):
            return (
                self.year == other.year
                and self.month == other.month
                and self.day == other.day
                and self.ambiguous == other.ambiguous
                and self.order == other.order
                and self.candidates == other.candidates
                and self.reason == other.reason
            )
        if isinstance(other, tuple):
            return self.normalized == other
        return False


@dataclass(frozen=True, eq=False)
class NormalizedNumber:
    """Result from :func:`normalize_number`.

    Args:
        value: Canonical numeric string using ``.`` as the decimal separator.
        ambiguous: True when the input has no safe single interpretation.
        decimal_separator: Separator interpreted as the decimal separator.
        thousands_separators: Grouping separators stripped from the integer part.
        reason: Machine-readable explanation for non-normalized results.
    """

    value: str | None
    ambiguous: bool = False
    decimal_separator: str | None = None
    thousands_separators: tuple[str, ...] = ()
    reason: str | None = None

    def __eq__(self, other: object) -> bool:
        """Compare with another result or the canonical numeric string."""
        if isinstance(other, NormalizedNumber):
            return (
                self.value == other.value
                and self.ambiguous == other.ambiguous
                and self.decimal_separator == other.decimal_separator
                and self.thousands_separators == other.thousands_separators
                and self.reason == other.reason
            )
        if isinstance(other, str):
            return self.value == other
        return False

    def __str__(self) -> str:
        """Return the canonical value or an empty string when unresolved."""
        return self.value or ""


@dataclass(frozen=True, eq=False)
class LocaleFormatHint:
    """Formatting hints for one OpenMed language code."""

    lang: str
    date_order: DateOrder
    number: NumberSeparators

    @property
    def decimal_separator(self) -> str:
        """Return the preferred decimal separator."""
        return self.number.decimal

    @property
    def thousands_separators(self) -> tuple[str, ...]:
        """Return accepted thousands/grouping separators."""
        return self.number.thousands

    def __eq__(self, other: object) -> bool:
        """Compare with another hint or a minimal mapping shape."""
        if isinstance(other, LocaleFormatHint):
            return (
                self.lang == other.lang
                and self.date_order == other.date_order
                and self.number == other.number
            )
        if isinstance(other, Mapping):
            return (
                other.get("lang") == self.lang
                and other.get("date_order") == self.date_order
                and other.get("decimal_separator") == self.decimal_separator
                and other.get("thousands_separators") == self.thousands_separators
            )
        return False

    def __getitem__(self, key: str) -> str | tuple[str, ...] | NumberSeparators:
        """Expose dictionary-style access for simple pattern builders."""
        if key == "lang":
            return self.lang
        if key == "date_order":
            return self.date_order
        if key == "number":
            return self.number
        if key == "decimal_separator":
            return self.decimal_separator
        if key == "thousands_separators":
            return self.thousands_separators
        raise KeyError(key)


WIRED_LOCALES: Final = frozenset(
    {"en", "fr", "de", "it", "es", "nl", "hi", "te", "pt", "ar", "ja", "ko", "tr"}
)
BACKLOG_LOCALES: Final = frozenset(
    {
        "bn",
        "cs",
        "da",
        "el",
        "fa",
        "he",
        "id",
        "ms",
        "no",
        "pl",
        "ro",
        "ru",
        "sv",
        "th",
        "tl",
        "uk",
        "vi",
        "zh",
    }
)

LOCALE_DATE_ORDER: Final[Mapping[str, DateOrder]] = {
    # Wired PII languages.
    "en": "mdy",
    "fr": "dmy",
    "de": "dmy",
    "it": "dmy",
    "es": "dmy",
    "nl": "dmy",
    "hi": "dmy",
    "te": "dmy",
    "pt": "dmy",
    "ar": "dmy",
    "ja": "ymd",
    "ko": "ymd",
    "tr": "dmy",
    # Backlog and national-ID-only locales.
    "bn": "dmy",
    "cs": "dmy",
    "da": "dmy",
    "el": "dmy",
    "fa": "ymd",
    "he": "dmy",
    "id": "dmy",
    "ms": "dmy",
    "no": "dmy",
    "pl": "dmy",
    "ro": "dmy",
    "ru": "dmy",
    "sk": "dmy",
    "sr": "dmy",
    "sv": "ymd",
    "th": "dmy",
    "tl": "dmy",
    "lv": "dmy",
    "uk": "dmy",
    "vi": "dmy",
    "zh": "ymd",
}

_DOT_DECIMAL_COMMA_GROUPS: Final = NumberSeparators(".", (",",))
_COMMA_DECIMAL_DOT_GROUPS: Final = NumberSeparators(",", (".",))
_COMMA_DECIMAL_SPACE_GROUPS: Final = NumberSeparators(
    ",",
    _SPACE_GROUP_SEPARATORS,
)
_ARABIC_NUMBER_SEPARATORS: Final = NumberSeparators(
    _ARABIC_DECIMAL_SEPARATOR,
    (_ARABIC_GROUP_SEPARATOR, ",", *_SPACE_GROUP_SEPARATORS),
    decimal_alternates=(".",),
)

LOCALE_NUMBER_SEP: Final[Mapping[str, NumberSeparators]] = {
    # Wired PII languages.
    "en": _DOT_DECIMAL_COMMA_GROUPS,
    "fr": _COMMA_DECIMAL_SPACE_GROUPS,
    "de": _COMMA_DECIMAL_DOT_GROUPS,
    "it": _COMMA_DECIMAL_DOT_GROUPS,
    "es": _COMMA_DECIMAL_DOT_GROUPS,
    "nl": _COMMA_DECIMAL_DOT_GROUPS,
    "hi": _DOT_DECIMAL_COMMA_GROUPS,
    "te": _DOT_DECIMAL_COMMA_GROUPS,
    "pt": _COMMA_DECIMAL_DOT_GROUPS,
    "ar": _ARABIC_NUMBER_SEPARATORS,
    "ja": _DOT_DECIMAL_COMMA_GROUPS,
    "ko": _DOT_DECIMAL_COMMA_GROUPS,
    "tr": _COMMA_DECIMAL_DOT_GROUPS,
    # Backlog and national-ID-only locales.
    "bn": _DOT_DECIMAL_COMMA_GROUPS,
    "cs": _COMMA_DECIMAL_SPACE_GROUPS,
    "da": _COMMA_DECIMAL_DOT_GROUPS,
    "el": _COMMA_DECIMAL_DOT_GROUPS,
    "fa": _ARABIC_NUMBER_SEPARATORS,
    "he": _DOT_DECIMAL_COMMA_GROUPS,
    "id": _COMMA_DECIMAL_DOT_GROUPS,
    "ms": _DOT_DECIMAL_COMMA_GROUPS,
    "no": _COMMA_DECIMAL_SPACE_GROUPS,
    "pl": _COMMA_DECIMAL_SPACE_GROUPS,
    "ro": _COMMA_DECIMAL_DOT_GROUPS,
    "ru": _COMMA_DECIMAL_SPACE_GROUPS,
    "sk": _COMMA_DECIMAL_SPACE_GROUPS,
    "sr": _COMMA_DECIMAL_DOT_GROUPS,
    "sv": _COMMA_DECIMAL_SPACE_GROUPS,
    "th": _DOT_DECIMAL_COMMA_GROUPS,
    "tl": _DOT_DECIMAL_COMMA_GROUPS,
    "lv": _COMMA_DECIMAL_SPACE_GROUPS,
    "uk": _COMMA_DECIMAL_SPACE_GROUPS,
    "vi": _COMMA_DECIMAL_DOT_GROUPS,
    "zh": _DOT_DECIMAL_COMMA_GROUPS,
}


def parse_date(text: str, lang: str | None = None) -> ParsedDate:
    """Parse a numeric Gregorian date using locale day/month order.

    Args:
        text: Date text such as ``"03.04.2026"`` or ``"03/04/2026"``.
        lang: Optional OpenMed language code. When omitted or unknown, the
            parser returns ``ambiguous=True`` for cross-locale date traps rather
            than silently choosing a day/month order.

    Returns:
        A :class:`ParsedDate` containing a normalized ``(year, month, day)``
        tuple when there is a single safe interpretation.
    """
    value = _normalize_digits(text).strip()
    cjk_match = _CJK_DATE_RE.match(value)
    if cjk_match:
        candidate = _date_candidate(
            (cjk_match.group("year"), cjk_match.group("month"), cjk_match.group("day")),
            "ymd",
        )
        if candidate is None:
            return ParsedDate(None, None, None, reason="invalid_date")
        return _parsed_date_from_candidate(candidate, (candidate,))

    match = _DATE_RE.match(value)
    if match is None:
        return ParsedDate(None, None, None, reason="unrecognized_date")

    parts = (match.group("a"), match.group("b"), match.group("c"))
    if len(parts[0]) == 4:
        candidate = _date_candidate(parts, "ymd")
        if candidate is None:
            return ParsedDate(None, None, None, reason="invalid_date")
        return _parsed_date_from_candidate(candidate, (candidate,))

    lang_code = _normalize_lang(lang)
    locale_order = LOCALE_DATE_ORDER.get(lang_code) if lang_code else None
    if locale_order is not None:
        candidate = _date_candidate(parts, locale_order)
        candidates = _valid_date_candidates(parts)
        if candidate is not None:
            return _parsed_date_from_candidate(candidate, candidates or (candidate,))
        return ParsedDate(
            None,
            None,
            None,
            ambiguous=bool(candidates),
            order=locale_order,
            candidates=candidates,
            reason="locale_order_invalid" if candidates else "invalid_date",
        )

    candidates = _valid_date_candidates(parts)
    if not candidates:
        return ParsedDate(None, None, None, reason="invalid_date")

    unique_values = _unique_date_values(candidates)
    if len(unique_values) == 1:
        return _parsed_date_from_candidate(candidates[0], candidates)
    return ParsedDate(
        None,
        None,
        None,
        ambiguous=True,
        candidates=candidates,
        reason="ambiguous_date",
    )


def normalize_number(text: str, lang: str | None = None) -> NormalizedNumber:
    """Normalize a locale-formatted number to a canonical numeric string.

    Args:
        text: Numeric text with optional sign, decimal separator, and grouping
            separators.
        lang: Optional OpenMed language code. Known locales use their separator
            table. Unknown or omitted locales use conservative inference and
            flag genuinely ambiguous cases instead of guessing.

    Returns:
        A :class:`NormalizedNumber` whose ``value`` uses ASCII digits and ``.``
        as the decimal separator when a single interpretation is safe.
    """
    raw = _normalize_digits(text).strip()
    if not raw:
        return NormalizedNumber(None, reason="empty")

    sign = ""
    if raw[0] in "+-":
        if raw[0] == "-":
            sign = "-"
        raw = raw[1:].strip()

    if not raw:
        return NormalizedNumber(None, reason="empty")

    lang_code = _normalize_lang(lang)
    separators = LOCALE_NUMBER_SEP.get(lang_code) if lang_code else None
    if separators is not None:
        return _normalize_number_with_locale(raw, sign, separators)
    return _normalize_number_without_locale(raw, sign)


def format_hint(lang: str) -> LocaleFormatHint:
    """Return date order and number separators for an OpenMed language code.

    Args:
        lang: OpenMed ISO 639-1 language code.

    Returns:
        Locale formatting hints for language-pack pattern builders.

    Raises:
        ValueError: If the language has no registered locale-format hint.
    """
    lang_code = _normalize_lang(lang)
    if not lang_code:
        raise ValueError("lang must be a non-empty OpenMed language code")
    try:
        date_order = LOCALE_DATE_ORDER[lang_code]
        number = LOCALE_NUMBER_SEP[lang_code]
    except KeyError as exc:
        raise ValueError(f"unsupported locale format language: {lang!r}") from exc
    return LocaleFormatHint(lang=lang_code, date_order=date_order, number=number)


def _normalize_lang(lang: str | None) -> str | None:
    if lang is None:
        return None
    normalized = str(lang).strip().lower().replace("_", "-")
    if not normalized:
        return None
    return normalized.split("-", 1)[0]


def _normalize_digits(text: str) -> str:
    normalized = []
    for char in text:
        try:
            normalized.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            normalized.append(char)
    return "".join(normalized)


def _date_candidate(
    parts: tuple[str, str, str], order: DateOrder
) -> DateCandidate | None:
    values = tuple(int(part) for part in parts)
    if order == "mdy":
        month, day, year = values
        year_width = len(parts[2])
    elif order == "dmy":
        day, month, year = values
        year_width = len(parts[2])
    elif order == "ymd":
        year, month, day = values
        year_width = len(parts[0])
    else:
        return None

    year = _expand_year(year, year_width)
    try:
        date(year, month, day)
    except ValueError:
        return None
    return DateCandidate(year=year, month=month, day=day, order=order)


def _expand_year(year: int, width: int) -> int:
    if width <= 2:
        return 2000 + year if year < 50 else 1900 + year
    return year


def _valid_date_candidates(parts: tuple[str, str, str]) -> tuple[DateCandidate, ...]:
    candidates = []
    seen: set[tuple[int, int, int, DateOrder]] = set()
    for order in ("mdy", "dmy", "ymd"):
        candidate = _date_candidate(parts, order)
        if candidate is None:
            continue
        key = (*candidate.normalized, candidate.order)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return tuple(candidates)


def _unique_date_values(
    candidates: tuple[DateCandidate, ...],
) -> tuple[tuple[int, int, int], ...]:
    values: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if candidate.normalized not in values:
            values.append(candidate.normalized)
    return tuple(values)


def _parsed_date_from_candidate(
    candidate: DateCandidate,
    candidates: tuple[DateCandidate, ...],
) -> ParsedDate:
    return ParsedDate(
        candidate.year,
        candidate.month,
        candidate.day,
        order=candidate.order,
        candidates=candidates,
    )


def _normalize_number_with_locale(
    value: str,
    sign: str,
    separators: NumberSeparators,
) -> NormalizedNumber:
    decimal_positions = _separator_positions(value, separators.decimal_separators)
    if len(decimal_positions) > 1:
        return NormalizedNumber(None, ambiguous=True, reason="multiple_decimals")

    decimal_separator = None
    fraction = None
    integer = value
    if decimal_positions:
        position, decimal_separator = decimal_positions[0]
        integer = value[:position]
        fraction = value[position + len(decimal_separator) :]

    integer_clean = _remove_separators(integer, separators.thousands)
    if not integer_clean or not integer_clean.isdigit():
        return _invalid_or_ambiguous_number(integer_clean)
    if fraction is not None and (not fraction or not fraction.isdigit()):
        return _invalid_or_ambiguous_number(fraction)

    return NormalizedNumber(
        _canonical_number(sign, integer_clean, fraction),
        decimal_separator=decimal_separator,
        thousands_separators=separators.thousands,
    )


def _normalize_number_without_locale(value: str, sign: str) -> NormalizedNumber:
    if value.isdigit():
        return NormalizedNumber(_canonical_number(sign, value, None))

    non_space_positions = _separator_positions(
        value,
        (".", ",", _ARABIC_DECIMAL_SEPARATOR),
    )
    if not non_space_positions:
        return _normalize_space_grouped_number(value, sign)

    separator_types = {separator for _, separator in non_space_positions}
    if len(separator_types) > 1:
        decimal_position, decimal_separator = max(
            non_space_positions, key=lambda item: item[0]
        )
        integer = value[:decimal_position]
        fraction = value[decimal_position + len(decimal_separator) :]
        grouping = tuple(
            separator
            for separator in _ALL_NUMBER_SEPARATOR_CHARS
            if separator != decimal_separator
        )
        integer_clean = _remove_separators(integer, grouping)
        if not integer_clean or not integer_clean.isdigit() or not fraction.isdigit():
            return NormalizedNumber(None, reason="invalid_number")
        return NormalizedNumber(
            _canonical_number(sign, integer_clean, fraction),
            decimal_separator=decimal_separator,
            thousands_separators=grouping,
        )

    separator = next(iter(separator_types))
    positions = [position for position, _ in non_space_positions]
    if len(positions) > 1:
        if _has_valid_grouping(value, separator):
            clean = value.replace(separator, "")
            return NormalizedNumber(
                _canonical_number(sign, clean, None),
                thousands_separators=(separator,),
            )
        return NormalizedNumber(None, ambiguous=True, reason="ambiguous_number")

    position = positions[0]
    digits_before = _clean_valid_space_grouping(value[:position])
    digits_after = value[position + len(separator) :]
    if digits_before is None or not digits_after.isdigit():
        return NormalizedNumber(None, reason="invalid_number")
    if len(digits_after) == 3 and 1 <= len(digits_before) <= 3:
        return NormalizedNumber(None, ambiguous=True, reason="ambiguous_number")
    return NormalizedNumber(
        _canonical_number(sign, digits_before, digits_after),
        decimal_separator=separator,
    )


def _normalize_space_grouped_number(value: str, sign: str) -> NormalizedNumber:
    for separator in _SPACE_GROUP_SEPARATORS:
        if separator in value:
            if not _has_valid_grouping(value, separator):
                return NormalizedNumber(None, ambiguous=True, reason="ambiguous_number")
            clean = value.replace(separator, "")
            if not clean.isdigit():
                return NormalizedNumber(None, reason="invalid_number")
            return NormalizedNumber(
                _canonical_number(sign, clean, None),
                thousands_separators=(separator,),
            )
    return NormalizedNumber(None, reason="invalid_number")


def _clean_valid_space_grouping(value: str) -> str | None:
    if value.isdigit():
        return value
    for separator in _SPACE_GROUP_SEPARATORS:
        if separator in value:
            if not _has_valid_grouping(value, separator):
                return None
            return value.replace(separator, "")
    return None


def _separator_positions(
    value: str,
    separators: tuple[str, ...],
) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    for separator in separators:
        start = 0
        while True:
            position = value.find(separator, start)
            if position == -1:
                break
            positions.append((position, separator))
            start = position + len(separator)
    return sorted(positions)


def _remove_separators(value: str, separators: tuple[str, ...]) -> str:
    cleaned = value
    for separator in separators:
        cleaned = cleaned.replace(separator, "")
    return cleaned


def _invalid_or_ambiguous_number(part: str) -> NormalizedNumber:
    if any(separator in part for separator in _ALL_NUMBER_SEPARATOR_CHARS):
        return NormalizedNumber(None, ambiguous=True, reason="unexpected_separator")
    return NormalizedNumber(None, reason="invalid_number")


def _has_valid_grouping(value: str, separator: str) -> bool:
    groups = value.split(separator)
    return (
        len(groups) > 1
        and all(group.isdigit() for group in groups)
        and 1 <= len(groups[0]) <= 3
        and all(len(group) == 3 for group in groups[1:])
    )


def _canonical_number(sign: str, integer: str, fraction: str | None) -> str:
    canonical_integer = integer.lstrip("0") or "0"
    if fraction is None:
        return f"{sign}{canonical_integer}"
    return f"{sign}{canonical_integer}.{fraction}"


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return tuple(unique)


__all__ = [
    "BACKLOG_LOCALES",
    "LOCALE_DATE_ORDER",
    "LOCALE_NUMBER_SEP",
    "WIRED_LOCALES",
    "DateCandidate",
    "LocaleFormatHint",
    "NormalizedNumber",
    "NumberSeparators",
    "ParsedDate",
    "format_hint",
    "normalize_number",
    "parse_date",
]
