"""Shift and synthesize dates written in non-Gregorian calendars.

This module recognizes a date surface such as ``۱۴۰۴/۰۷/۳۰``,
``۲۸ اردیبهشت ۱۴۰۴``, ``28 กุมภาพันธ์ พ.ศ. 2567`` or ``05/13/2015 ዓ.ም``,
chooses its calendar, shifts it through the shared day number of
:mod:`openmed.core.calendars`, and renders the result in the source calendar,
digit system, separators and month-name form.

The calendar is chosen from, in order:

1. explicit era markers inside the span (``ه.ش``, ``พ.ศ.``, ``ዓ.ም``,
   ``میلادی`` and similar), or a detection-stage calendar code;
2. month names that belong to a single calendar (Solar Hijri or Ethiopian);
3. the language pack's declared ``default_calendar`` when the year falls in
   that calendar's supported range.

Plain Gregorian dates are left to the existing Gregorian shifter. Conflicting
signals, two-digit years and years outside every plausible range are reported
as unshifted so callers emit their placeholder instead of guessing. Results
carry controlled reason codes and never echo the date text.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Final, Iterable, Mapping, Optional

from .calendars import (
    ETHIOPIAN,
    GREGORIAN,
    SOLAR_HIJRI,
    THAI_BUDDHIST,
    CalendarConverter,
    get_calendar,
    is_calendar_code,
)

CALENDAR_METADATA_KEY: Final = "calendar"
"""Entity metadata key a detector can set to declare a span's calendar code."""

LUNAR_HIJRI_MARKER: Final = "islamic"
"""Calendar code implied by lunar Hijri era markers; no converter ships yet."""

GREGORIAN_PLAUSIBLE_YEARS: Final = (1800, 2299)
"""Gregorian years accepted when a pack's default calendar does not apply."""

SHIFTED: Final = "shifted"
GREGORIAN_DATE: Final = "gregorian"
UNRECOGNIZED: Final = "unrecognized"
AMBIGUOUS: Final = "ambiguous"
UNSUPPORTED_CALENDAR: Final = "unsupported_calendar"
OUT_OF_RANGE: Final = "out_of_range"
INVALID_DATE: Final = "invalid_date"

# The surrogate window mirrors ``preserve_date_format``: 100 years of days
# starting on 1 January 1925, intersected with the calendar's supported range.
_SURROGATE_WINDOW_START: Final = date(1925, 1, 1).toordinal()
_SURROGATE_WINDOW_DAYS: Final = 365 * 100


@dataclass(frozen=True)
class _MonthStyle:
    """One family of month names and the calendars it can denote."""

    key: str
    calendars: tuple[str, ...]
    names: tuple[tuple[str, ...], ...]


def _with_arabic_letter_variants(name: str) -> tuple[str, ...]:
    variant = name.replace("ی", "ي").replace("ک", "ك")
    return (name,) if variant == name else (name, variant)


_SOLAR_HIJRI_MONTHS: Final = _MonthStyle(
    key="persian",
    calendars=(SOLAR_HIJRI,),
    names=tuple(
        _with_arabic_letter_variants(name)
        for name in (
            "فروردین",
            "اردیبهشت",
            "خرداد",
            "تیر",
            "مرداد",
            "شهریور",
            "مهر",
            "آبان",
            "آذر",
            "دی",
            "بهمن",
            "اسفند",
        )
    ),
)
_ETHIOPIAN_MONTHS: Final = _MonthStyle(
    key="amharic",
    calendars=(ETHIOPIAN,),
    names=(
        ("መስከረም",),
        ("ጥቅምት",),
        ("ኅዳር", "ህዳር"),
        ("ታኅሣሥ", "ታህሳስ", "ታሕሳስ"),
        ("ጥር",),
        ("የካቲት",),
        ("መጋቢት",),
        ("ሚያዝያ",),
        ("ግንቦት",),
        ("ሰኔ",),
        ("ሐምሌ", "ሀምሌ"),
        ("ነሐሴ", "ነሀሴ"),
        ("ጳጉሜ", "ጳጉሜን"),
    ),
)
# Thai month names are shared by the Buddhist-era and Common-era calendars;
# the year or an era marker decides which one a date uses.
_THAI_FULL_MONTHS: Final = _MonthStyle(
    key="thai_full",
    calendars=(THAI_BUDDHIST, GREGORIAN),
    names=tuple(
        (name,)
        for name in (
            "มกราคม",
            "กุมภาพันธ์",
            "มีนาคม",
            "เมษายน",
            "พฤษภาคม",
            "มิถุนายน",
            "กรกฎาคม",
            "สิงหาคม",
            "กันยายน",
            "ตุลาคม",
            "พฤศจิกายน",
            "ธันวาคม",
        )
    ),
)
_THAI_ABBREVIATED_MONTHS: Final = _MonthStyle(
    key="thai_abbreviated",
    calendars=(THAI_BUDDHIST, GREGORIAN),
    names=tuple(
        (name,)
        for name in (
            "ม.ค.",
            "ก.พ.",
            "มี.ค.",
            "เม.ย.",
            "พ.ค.",
            "มิ.ย.",
            "ก.ค.",
            "ส.ค.",
            "ก.ย.",
            "ต.ค.",
            "พ.ย.",
            "ธ.ค.",
        )
    ),
)
MONTH_STYLES: Final = (
    _SOLAR_HIJRI_MONTHS,
    _ETHIOPIAN_MONTHS,
    _THAI_FULL_MONTHS,
    _THAI_ABBREVIATED_MONTHS,
)

# Era markers recognized inside a date span. Lunar Hijri markers resolve to a
# calendar without a registered converter, so those dates stay unshifted.
ERA_MARKERS: Final[Mapping[str, str]] = {
    # Solar Hijri.
    "هجری شمسی": SOLAR_HIJRI,
    "هجری خورشیدی": SOLAR_HIJRI,
    "ه.ش.": SOLAR_HIJRI,
    "ه.ش": SOLAR_HIJRI,
    "هـ.ش.": SOLAR_HIJRI,
    "هـ.ش": SOLAR_HIJRI,
    "ه‍.ش.": SOLAR_HIJRI,
    "ه‍.ش": SOLAR_HIJRI,
    "شمسی": SOLAR_HIJRI,
    "خورشیدی": SOLAR_HIJRI,
    "A.P.": SOLAR_HIJRI,
    "S.H.": SOLAR_HIJRI,
    "AP": SOLAR_HIJRI,
    "SH": SOLAR_HIJRI,
    # Thai Buddhist era.
    "พุทธศักราช": THAI_BUDDHIST,
    "พ.ศ.": THAI_BUDDHIST,
    "พ.ศ": THAI_BUDDHIST,
    "B.E.": THAI_BUDDHIST,
    "BE": THAI_BUDDHIST,
    # Ethiopian (Amete Mihret) era.
    "ዓ.ም.": ETHIOPIAN,
    "ዓ.ም": ETHIOPIAN,
    "ዓ/ም": ETHIOPIAN,
    "ዓም": ETHIOPIAN,
    "E.C.": ETHIOPIAN,
    "EC": ETHIOPIAN,
    # Gregorian markers used beside native-calendar dates.
    "میلادی": GREGORIAN,
    "ميلادي": GREGORIAN,
    "م.": GREGORIAN,
    "م": GREGORIAN,
    "คริสต์ศักราช": GREGORIAN,
    "ค.ศ.": GREGORIAN,
    "ค.ศ": GREGORIAN,
    "እ.ኤ.አ.": GREGORIAN,
    "እ.ኤ.አ": GREGORIAN,
    "እኤአ": GREGORIAN,
    "G.C.": GREGORIAN,
    "GC": GREGORIAN,
    # Lunar Hijri.
    "هجری قمری": LUNAR_HIJRI_MARKER,
    "ه.ق.": LUNAR_HIJRI_MARKER,
    "ه.ق": LUNAR_HIJRI_MARKER,
    "هـ.ق.": LUNAR_HIJRI_MARKER,
    "هـ.ق": LUNAR_HIJRI_MARKER,
    "قمری": LUNAR_HIJRI_MARKER,
    "هجرية": LUNAR_HIJRI_MARKER,
    "هجري": LUNAR_HIJRI_MARKER,
    "هجرى": LUNAR_HIJRI_MARKER,
    "هـ": LUNAR_HIJRI_MARKER,
    "A.H.": LUNAR_HIJRI_MARKER,
    "AH": LUNAR_HIJRI_MARKER,
}

_MONTH_LOOKUP: Final[Mapping[str, tuple[_MonthStyle, int]]] = {
    spelling: (style, index + 1)
    for style in MONTH_STYLES
    for index, spellings in enumerate(style.names)
    for spelling in spellings
}


def _alternation(values: Iterable[str]) -> str:
    return "|".join(
        re.escape(value) for value in sorted(values, key=lambda item: -len(item))
    )


_NATIVE_MONTH_NAME_RE: Final = re.compile(
    r"(?<!\w)(?:"
    + _alternation(
        spelling
        for style in MONTH_STYLES
        if len(style.calendars) == 1
        for spellings in style.names
        for spelling in spellings
    )
    + r")(?!\w)"
)
_TOKEN_RE: Final = re.compile(
    rf"(?P<month>{_alternation(_MONTH_LOOKUP)})"
    rf"|(?P<marker>{_alternation(ERA_MARKERS)})"
    r"|(?P<number>\d+)"
    r"|(?P<sep>[./\-])"
    r"|(?P<space>[\s,،]+|ቀን)"
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str


@dataclass(frozen=True)
class _ParsedSurface:
    tokens: tuple[_Token, ...]
    year_index: Optional[int]
    other_indexes: tuple[int, ...]
    month_index: Optional[int]
    month_style: Optional[_MonthStyle]
    marker_calendars: frozenset[str]
    zero_padded: bool

    @property
    def year(self) -> Optional[int]:
        if self.year_index is None:
            return None
        return int(self.tokens[self.year_index].text)


@dataclass(frozen=True)
class _SourceDate:
    year: int
    month: int
    day: int
    month_index: int
    day_index: int


@dataclass(frozen=True)
class CalendarDateShift:
    """Outcome of shifting one date surface through a calendar converter.

    Attributes:
        text: Shifted surface rendered in the source calendar and format, or
            ``None`` when this module did not produce a shifted value.
        calendar: Resolved calendar code, when one was determined.
        reason: Controlled reason code: ``shifted``, ``gregorian``,
            ``unrecognized``, ``ambiguous``, ``unsupported_calendar``,
            ``out_of_range`` or ``invalid_date``.
        handled: ``True`` when the caller must use this outcome: ``text`` when
            present, otherwise its date placeholder. ``False`` means the
            surface is a plain Gregorian date for the existing shifter.
    """

    text: Optional[str]
    calendar: Optional[str]
    reason: str
    handled: bool


def default_calendar_for(lang: Optional[str]) -> str:
    """Return the default calendar declared by a language's pack.

    Args:
        lang: OpenMed language code or locale such as ``"fa"`` or ``"th_TH"``.

    Returns:
        The pack's ``default_calendar``, or ``"gregorian"`` for unknown or
        missing languages.
    """
    code = _language_code(lang)
    if code is None:
        return GREGORIAN
    from . import language_pack_catalog  # noqa: F401  (registers built-in packs)
    from .language_pack import get_language_pack

    pack = get_language_pack(code)
    if pack is None:
        return GREGORIAN
    return pack.default_calendar


def shift_calendar_date(
    text: str,
    days: int,
    *,
    lang: Optional[str] = None,
    calendar: Optional[str] = None,
    keep_year: bool = False,
) -> CalendarDateShift:
    """Shift a date written in its own calendar by ``days`` days.

    Args:
        text: Date span as it appears in the document.
        days: Signed day offset shared with the patient's Gregorian dates.
        lang: Language code or locale used for the pack's default calendar
            and numeric day/month order.
        calendar: Optional detection-stage calendar code for the span.
        keep_year: Restore the source year after shifting, clamping the day
            to the month length of that year.

    Returns:
        A :class:`CalendarDateShift`. ``handled`` is ``False`` only for
        surfaces that the existing Gregorian shifter should process.
    """
    if not isinstance(text, str) or isinstance(days, bool) or not isinstance(days, int):
        return CalendarDateShift(None, None, UNRECOGNIZED, handled=False)

    hint_given = calendar is not None
    if hint_given and not is_calendar_code(calendar):
        return CalendarDateShift(None, None, UNSUPPORTED_CALENDAR, handled=True)

    parsed = _parse_surface(text)
    if parsed is None:
        # A calendar-specific month name must never reach the Gregorian
        # month-name parser, which would read Ordibehesht as February.
        owned = (hint_given and calendar != GREGORIAN) or bool(
            _NATIVE_MONTH_NAME_RE.search(text)
        )
        return CalendarDateShift(None, None, UNRECOGNIZED, handled=owned)

    code, reason = _resolve_calendar(parsed, lang=lang, hint=calendar)
    if code is None:
        return CalendarDateShift(None, None, reason, handled=True)
    if code == GREGORIAN and not parsed.marker_calendars:
        return CalendarDateShift(None, code, GREGORIAN_DATE, handled=False)

    converter = get_calendar(code)
    if converter is None:
        return CalendarDateShift(None, code, UNSUPPORTED_CALENDAR, handled=True)

    source, reason = _source_date(parsed, converter, lang=lang)
    if source is None:
        return CalendarDateShift(None, code, reason, handled=True)

    target = converter.to_day_number(source.year, source.month, source.day) + days
    first, last = converter.day_number_range()
    if not first <= target <= last:
        return CalendarDateShift(None, code, OUT_OF_RANGE, handled=True)
    new_year, new_month, new_day = converter.from_day_number(target)
    if keep_year and new_year != source.year:
        new_year = source.year
        new_day = min(new_day, converter.month_length(new_year, new_month))

    rendered = _render(
        parsed,
        (new_year, new_month, new_day),
        source_month=source.month,
        month_index=source.month_index,
        day_index=source.day_index,
    )
    return CalendarDateShift(rendered, code, SHIFTED, handled=True)


def calendar_date_surrogate(
    original: str,
    *,
    lang: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Return a synthetic date in the source span's calendar and format.

    The surrogate is drawn from the same 100-year window as Gregorian date
    surrogates, intersected with the calendar's supported range, so it always
    falls inside the calendar's plausible era.

    Args:
        original: Source date span.
        lang: Language code or locale used for the pack's default calendar.
        rng: Random source; the module-level generator when omitted.

    Returns:
        A native-calendar surrogate, or ``None`` when the span is a plain
        Gregorian date or its calendar cannot be determined.
    """
    if not isinstance(original, str):
        return None
    parsed = _parse_surface(original)
    if parsed is None or parsed.year_index is None:
        return None
    code, _ = _resolve_calendar(parsed, lang=lang, hint=None)
    if code is None or (code == GREGORIAN and not parsed.marker_calendars):
        return None
    converter = get_calendar(code)
    if converter is None:
        return None

    first, last = converter.day_number_range()
    low = max(first, _SURROGATE_WINDOW_START)
    high = min(last, _SURROGATE_WINDOW_START + _SURROGATE_WINDOW_DAYS)
    if low > high:
        return None
    source = rng or random
    target = converter.from_day_number(low + source.randint(0, high - low))

    if parsed.month_index is None:
        month_index, day_index = _numeric_index_options(parsed, lang=lang)[0]
        source_month = 0
    else:
        month_index, day_index = parsed.month_index, parsed.other_indexes[0]
        source_month = _MONTH_LOOKUP[parsed.tokens[month_index].text][1]
    return _render(
        parsed,
        target,
        source_month=source_month,
        month_index=month_index,
        day_index=day_index,
    )


def _language_code(lang: Optional[str]) -> Optional[str]:
    if not isinstance(lang, str):
        return None
    normalized = lang.strip().lower().replace("_", "-")
    if not normalized:
        return None
    return normalized.split("-", 1)[0]


def _digit_zero(run: str) -> Optional[str]:
    zeros = {chr(ord(char) - unicodedata.decimal(char)) for char in run}
    return zeros.pop() if len(zeros) == 1 else None


def _render_number(value: int, *, template: str, width: int) -> str:
    zero = _digit_zero(template) or "0"
    digits = f"{value:0{width}d}"
    return "".join(chr(ord(zero) + int(char)) for char in digits)


def _has_leading_zero(run: str) -> bool:
    return len(run) > 1 and unicodedata.decimal(run[0]) == 0


def _tokenize(text: str) -> Optional[tuple[_Token, ...]]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.match(text, position)
        if match is None or match.end() == position or match.lastgroup is None:
            return None
        tokens.append(_Token(match.lastgroup, match.group()))
        position = match.end()
    return tuple(tokens)


def _parse_surface(text: str) -> Optional[_ParsedSurface]:
    tokens = _tokenize(text)
    if not tokens:
        return None

    numbers = [index for index, token in enumerate(tokens) if token.kind == "number"]
    months = [index for index, token in enumerate(tokens) if token.kind == "month"]
    if any(_digit_zero(tokens[index].text) is None for index in numbers):
        return None
    marker_calendars = frozenset(
        ERA_MARKERS[token.text] for token in tokens if token.kind == "marker"
    )

    if not months and len(numbers) == 3:
        first, middle, last = numbers
        if middle != first + 2 or last != middle + 2:
            return None
        separators = (tokens[first + 1], tokens[middle + 1])
        if any(token.kind != "sep" for token in separators):
            return None
        if separators[0].text != separators[1].text:
            return None
        outside = tokens[:first] + tokens[last + 1 :]
        if any(token.kind not in {"space", "marker"} for token in outside):
            return None
        widths = [len(tokens[index].text) for index in numbers]
        if any(width > 4 for width in widths):
            return None
        if widths[0] == 4 and widths[1] <= 2 and widths[2] <= 2:
            year_index, others = first, (middle, last)
        elif widths[2] == 4 and widths[0] <= 2 and widths[1] <= 2:
            year_index, others = last, (first, middle)
        elif all(width <= 2 for width in widths):
            year_index, others = None, (first, middle)
        else:
            return None
        # Numeric dates follow the existing shifter's two-digit padding unless
        # the source visibly drops it, as in ``1404/7/3``.
        zero_padded = all(len(tokens[index].text) == 2 for index in others)
        return _ParsedSurface(
            tokens=tokens,
            year_index=year_index,
            other_indexes=others,
            month_index=None,
            month_style=None,
            marker_calendars=marker_calendars,
            zero_padded=zero_padded,
        )

    if len(months) == 1 and len(numbers) == 2:
        (month_index,) = months
        day_index, year_index = numbers
        if not (day_index < month_index < year_index or month_index < day_index):
            return None
        if len(tokens[day_index].text) > 2:
            return None
        year_width = len(tokens[year_index].text)
        if year_width not in (2, 4):
            return None
        fillers = {"space", "marker", "sep"}
        if any(
            token.kind not in fillers
            for index, token in enumerate(tokens)
            if index not in {day_index, month_index, year_index}
        ):
            return None
        style, _ = _MONTH_LOOKUP[tokens[month_index].text]
        return _ParsedSurface(
            tokens=tokens,
            year_index=year_index if year_width == 4 else None,
            other_indexes=(day_index,),
            month_index=month_index,
            month_style=style,
            marker_calendars=marker_calendars,
            zero_padded=_has_leading_zero(tokens[day_index].text),
        )
    return None


def _resolve_calendar(
    parsed: _ParsedSurface,
    *,
    lang: Optional[str],
    hint: Optional[str],
) -> tuple[Optional[str], str]:
    explicit = set(parsed.marker_calendars)
    if hint is not None:
        explicit.add(hint)
    if len(explicit) > 1:
        return None, AMBIGUOUS
    style = parsed.month_style
    if explicit:
        code = next(iter(explicit))
        if style is not None and code not in style.calendars:
            return None, AMBIGUOUS
        return code, "explicit"
    if style is not None and len(style.calendars) == 1:
        return style.calendars[0], "month_name"

    native = style.calendars[0] if style is not None else default_calendar_for(lang)
    if native == GREGORIAN:
        return GREGORIAN, GREGORIAN_DATE
    year = parsed.year
    if year is None:
        return None, AMBIGUOUS
    converter = get_calendar(native)
    if converter is None:
        return native, UNSUPPORTED_CALENDAR
    if converter.min_year <= year <= converter.max_year:
        return native, "default_calendar"
    low, high = GREGORIAN_PLAUSIBLE_YEARS
    if low <= year <= high:
        return GREGORIAN, GREGORIAN_DATE
    return None, OUT_OF_RANGE


def _source_date(
    parsed: _ParsedSurface,
    converter: CalendarConverter,
    *,
    lang: Optional[str],
) -> tuple[Optional[_SourceDate], str]:
    year = parsed.year
    if year is None:
        return None, AMBIGUOUS
    if not converter.min_year <= year <= converter.max_year:
        return None, OUT_OF_RANGE

    if parsed.month_index is None:
        options = _numeric_index_options(parsed, lang=lang)
        candidates = [
            _SourceDate(
                year=year,
                month=int(parsed.tokens[month_index].text),
                day=int(parsed.tokens[day_index].text),
                month_index=month_index,
                day_index=day_index,
            )
            for month_index, day_index in options
        ]
    else:
        day_index = parsed.other_indexes[0]
        candidates = [
            _SourceDate(
                year=year,
                month=_MONTH_LOOKUP[parsed.tokens[parsed.month_index].text][1],
                day=int(parsed.tokens[day_index].text),
                month_index=parsed.month_index,
                day_index=day_index,
            )
        ]

    valid = [
        candidate
        for candidate in candidates
        if converter.is_valid(candidate.year, candidate.month, candidate.day)
    ]
    if not valid:
        return None, INVALID_DATE
    if len({(candidate.month, candidate.day) for candidate in valid}) > 1:
        return None, AMBIGUOUS
    return valid[0], SHIFTED


def _numeric_index_options(
    parsed: _ParsedSurface,
    *,
    lang: Optional[str],
) -> tuple[tuple[int, int], ...]:
    first, second = parsed.other_indexes
    if parsed.year_index is not None and parsed.year_index < first:
        return ((first, second),)
    from .locale_formats import date_order_for_locale

    order = date_order_for_locale(lang)
    if order == "dmy":
        return ((second, first),)
    if order == "mdy":
        return ((first, second),)
    return ((second, first), (first, second))


def _render(
    parsed: _ParsedSurface,
    value: tuple[int, int, int],
    *,
    source_month: int,
    month_index: int,
    day_index: int,
) -> str:
    year, month, day = value
    tokens = parsed.tokens
    width = 2 if parsed.zero_padded else 1
    pieces = [token.text for token in tokens]
    if parsed.year_index is not None:
        year_template = tokens[parsed.year_index].text
        pieces[parsed.year_index] = _render_number(
            year,
            template=year_template,
            width=len(year_template),
        )
    pieces[day_index] = _render_number(
        day,
        template=tokens[day_index].text,
        width=width,
    )
    if parsed.month_style is None:
        pieces[month_index] = _render_number(
            month,
            template=tokens[month_index].text,
            width=width,
        )
    elif month != source_month:
        pieces[month_index] = parsed.month_style.names[month - 1][0]
    return "".join(pieces)


__all__ = [
    "AMBIGUOUS",
    "CALENDAR_METADATA_KEY",
    "ERA_MARKERS",
    "GREGORIAN_DATE",
    "GREGORIAN_PLAUSIBLE_YEARS",
    "INVALID_DATE",
    "LUNAR_HIJRI_MARKER",
    "MONTH_STYLES",
    "OUT_OF_RANGE",
    "SHIFTED",
    "UNRECOGNIZED",
    "UNSUPPORTED_CALENDAR",
    "CalendarDateShift",
    "calendar_date_surrogate",
    "default_calendar_for",
    "shift_calendar_date",
]
