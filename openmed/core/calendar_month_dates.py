"""Month-name date patterns for non-Gregorian calendars.

Arabic, Persian and Urdu records write dates in the Lunar Hijri calendar,
Amharic records in the Ethiopian calendar, Nepali records in Bikram Sambat and
Hebrew records in the Hebrew calendar, usually with the calendar's own month
names. A date of birth written this way is a direct identifier.

This module keeps those month names as declarative data, one
:class:`CalendarMonthTable` per language and calendar, and compiles each table
into deterministic ``date`` patterns that accept the language's native digits.
:func:`openmed.core.pii_i18n.get_patterns_for_language` appends the compiled
patterns to the matching language route.

A month word only matches with both a day and a year attached, inside the
calendar's day and year ranges, so month names that are also ordinary words
(``رمضان`` in a sentence about fasting, ``שבט`` meaning "tribe") stay
unflagged. Every compiled pattern carries a value-free calendar code in
``PIIPattern.calendar``; the safety sweep and the semantic merger copy that
code into span metadata so later shifting or surrogate stages can select a
calendar without reading the date. Nothing here parses, converts or shifts a
date: shifting keeps its existing placeholder for these calendars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from .pii_entity_merger import PIIPattern

HIJRI_CALENDAR: Final = "hijri"
ETHIOPIAN_CALENDAR: Final = "ethiopian"
BIKRAM_SAMBAT_CALENDAR: Final = "bikram_sambat"
HEBREW_CALENDAR: Final = "hebrew"

#: Value-free calendar codes attached to non-Gregorian month-name dates.
CALENDAR_CODES: Final = frozenset(
    {HIJRI_CALENDAR, ETHIOPIAN_CALENDAR, BIKRAM_SAMBAT_CALENDAR, HEBREW_CALENDAR}
)

_DATE_ORDERS: Final = frozenset({"dmy", "mdy", "ymd"})

_ARABIC_INDIC_DIGITS: Final = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED_ARABIC_INDIC_DIGITS: Final = "۰۱۲۳۴۵۶۷۸۹"
_DEVANAGARI_DIGITS: Final = "०१२३४५६७८९"


@dataclass(frozen=True)
class CalendarMonthTable:
    """Month-name vocabulary for one calendar as written in one language.

    Attributes:
        calendar: Value-free calendar code recorded on every match.
        months: Accepted spellings for each month, in calendar order.
        orders: Accepted component orders: ``"dmy"``, ``"mdy"`` or ``"ymd"``.
        digits: Native decimal digit sets, each written ``0`` to ``9``,
            accepted in addition to ASCII digits.
        era_markers: Optional era words kept inside the span, written after
            the year, or before it in ``"ymd"`` order.
        month_particles: Optional words written before the month name, such
            as Arabic ``شهر`` ("month").
        day_suffixes: Optional words written after the day, such as Nepali
            ``गते``.
        year_particles: Optional words written before the year, such as
            Arabic ``عام`` ("year").
        year_suffixes: Optional words written after the year, such as Nepali
            ``साल``.
        prefix_letters: One-letter prepositions or conjunctions that attach to
            the date or the month name. A prefix before the date stays
            outside the span.
        context_words: Nearby cues that raise the match confidence.
    """

    calendar: str
    months: tuple[tuple[str, ...], ...]
    orders: tuple[str, ...] = ("dmy",)
    digits: tuple[str, ...] = ()
    era_markers: tuple[str, ...] = ()
    month_particles: tuple[str, ...] = ()
    day_suffixes: tuple[str, ...] = ()
    year_particles: tuple[str, ...] = ()
    year_suffixes: tuple[str, ...] = ()
    prefix_letters: str = ""
    context_words: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Month-name tables
# ---------------------------------------------------------------------------

# Lunar Hijri months in Arabic spelling. Compilation also accepts the Persian
# and Urdu letter forms (ی, ک, ہ), hamza-less alefs, harakat, and an optional
# space or zero-width non-joiner inside two-word names, so one table serves
# ``ربيع الأول``, ``ربیع الاول`` (also with a non-joiner) and ``ربیع الاوّل``.
_HIJRI_MONTHS: Final = (
    ("محرم", "محرم الحرام"),
    ("صفر", "صفر المظفر", "صفر الخير"),
    ("ربيع الأول", "ربيع أول"),
    ("ربيع الآخر", "ربيع الثاني", "ربيع ثاني"),
    ("جمادى الأولى", "جمادى الأول", "جمادى أولى"),
    ("جمادى الآخرة", "جمادى الآخر", "جمادى الأخرى", "جمادى الثانية", "جمادى الثاني"),
    ("رجب", "رجب المرجب"),
    ("شعبان", "شعبان المعظم"),
    ("رمضان", "رمضان المبارك", "رمضان الكريم"),
    ("شوال", "شوال المكرم"),
    ("ذو القعدة", "ذي القعدة", "ذي قعدة"),
    ("ذو الحجة", "ذي الحجة", "ذي حجة"),
)
_HIJRI_DIGITS: Final = (_ARABIC_INDIC_DIGITS, _EXTENDED_ARABIC_INDIC_DIGITS)

# Ethiopian months, including the 13th month Pagume, with common spellings.
_ETHIOPIAN_MONTHS: Final = (
    ("መስከረም",),
    ("ጥቅምት",),
    ("ኅዳር", "ህዳር", "ሕዳር"),
    ("ታኅሣሥ", "ታህሳስ", "ታኅሳስ", "ታህሣሥ", "ታሕሳስ"),
    ("ጥር",),
    ("የካቲት",),
    ("መጋቢት",),
    ("ሚያዝያ", "ሚያዚያ"),
    ("ግንቦት",),
    ("ሰኔ",),
    ("ሐምሌ", "ሀምሌ", "ሃምሌ"),
    ("ነሐሴ", "ነሃሴ", "ነሀሴ"),
    ("ጳጉሜ", "ጳጉሜን", "ጷጉሜ"),
)

# Bikram Sambat months in colloquial Nepali and formal (tatsama) spellings.
# ``मार्ग`` alone is left out because it also means "road".
_BIKRAM_SAMBAT_MONTHS: Final = (
    ("बैशाख", "वैशाख", "बैसाख"),
    ("जेठ", "जेष्ठ", "ज्येष्ठ"),
    ("असार", "आषाढ", "असाढ"),
    ("साउन", "श्रावण", "सावन"),
    ("भदौ", "भाद्र", "भाद्रपद"),
    ("असोज", "आश्विन", "असौज"),
    ("कात्तिक", "कार्तिक", "कातिक"),
    ("मंसिर", "मङ्सिर", "मंसीर", "मार्गशीर्ष"),
    ("पुस", "पौष", "पूस"),
    ("माघ",),
    ("फागुन", "फाल्गुन", "फागुण"),
    ("चैत", "चैत्र"),
)

# Hebrew months from Tishrei, with Adar I and Adar (II) for leap years. A
# geresh (׳) also matches an ASCII or typographic apostrophe, or none.
_HEBREW_MONTHS: Final = (
    ("תשרי",),
    ("חשוון", "חשון", "מרחשוון", "מרחשון"),
    ("כסלו", "כסליו"),
    ("טבת",),
    ("שבט",),
    ("אדר א׳", "אדר ראשון"),
    ("אדר", "אדר ב׳", "אדר שני"),
    ("ניסן",),
    ("אייר", "איר"),
    ("סיוון", "סיון"),
    ("תמוז",),
    ("אב", "מנחם אב"),
    ("אלול",),
)

#: Declarative month-name tables per language, compiled into date patterns.
CALENDAR_MONTH_TABLES: Final[Mapping[str, tuple[CalendarMonthTable, ...]]] = (
    MappingProxyType(
        {
            "ar": (
                CalendarMonthTable(
                    calendar=HIJRI_CALENDAR,
                    months=_HIJRI_MONTHS,
                    digits=_HIJRI_DIGITS,
                    era_markers=("هـ", "ه", "هجري", "هجرية", "للهجرة"),
                    month_particles=("من شهر", "شهر", "من"),
                    year_particles=("عام", "سنة"),
                    context_words=(
                        "تاريخ",
                        "تاريخ الميلاد",
                        "الميلاد",
                        "ميلاد",
                        "مواليد",
                        "ولد",
                        "هجري",
                        "الدخول",
                        "خروج",
                    ),
                ),
            ),
            "fa": (
                CalendarMonthTable(
                    calendar=HIJRI_CALENDAR,
                    months=_HIJRI_MONTHS,
                    digits=_HIJRI_DIGITS,
                    era_markers=("هجری قمری", "ه. ق", "هـ. ق", "ه ق", "قمری", "ق"),
                    month_particles=("ماه",),
                    year_particles=("سال",),
                    context_words=(
                        "تاریخ",
                        "تولد",
                        "تاریخ تولد",
                        "متولد",
                        "پذیرش",
                        "ترخیص",
                        "قمری",
                    ),
                ),
            ),
            "ur": (
                CalendarMonthTable(
                    calendar=HIJRI_CALENDAR,
                    months=_HIJRI_MONTHS,
                    digits=_HIJRI_DIGITS,
                    era_markers=("ہجری", "هـ", "ھ", "ہ"),
                    month_particles=("ماہ",),
                    context_words=(
                        "تاریخ",
                        "پیدائش",
                        "تاریخ پیدائش",
                        "داخلہ",
                        "ڈسچارج",
                        "ہجری",
                    ),
                ),
            ),
            "am": (
                CalendarMonthTable(
                    calendar=ETHIOPIAN_CALENDAR,
                    months=_ETHIOPIAN_MONTHS,
                    orders=("dmy", "mdy"),
                    era_markers=("ዓ. ም", "ዓ/ም", "ዓም", "ዓመተ ምሕረት"),
                    day_suffixes=("ቀን",),
                    prefix_letters="በከለ",
                    context_words=(
                        "ቀን",
                        "የትውልድ ቀን",
                        "ትውልድ",
                        "የተወለደ",
                        "የቀጠሮ ቀን",
                        "date of birth",
                        "dob",
                    ),
                ),
            ),
            "ne": (
                CalendarMonthTable(
                    calendar=BIKRAM_SAMBAT_CALENDAR,
                    months=_BIKRAM_SAMBAT_MONTHS,
                    orders=("ymd", "dmy", "mdy"),
                    digits=(_DEVANAGARI_DIGITS,),
                    era_markers=(
                        "वि. सं",
                        "बि. सं",
                        "विक्रम संवत्",
                        "विक्रम संवत",
                        "विक्रम सम्वत्",
                        "बिक्रम संवत्",
                        "बिक्रम सम्वत्",
                    ),
                    day_suffixes=("गते",),
                    year_suffixes=("साल",),
                    context_words=(
                        "मिति",
                        "जन्म मिति",
                        "जन्म",
                        "भर्ना मिति",
                        "date of birth",
                        "dob",
                    ),
                ),
            ),
            "he": (
                CalendarMonthTable(
                    calendar=HEBREW_CALENDAR,
                    months=_HEBREW_MONTHS,
                    month_particles=("חודש",),
                    prefix_letters="ובכלמהש",
                    context_words=(
                        "תאריך",
                        "תאריך לידה",
                        "לידה",
                        "נולד",
                        "נולדה",
                        "אשפוז",
                        "שחרור",
                    ),
                ),
            ),
        }
    )
)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

# Combining marks that continue a grapheme cluster. A match never starts or
# ends next to one, so a span cannot split a letter from its vowel sign.
_MARKS: Final = (
    "\u0300-\u036f\u0591-\u05c7\u064b-\u065f\u0670"
    "\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962\u0963\u135d-\u135f"
)
_NOT_AFTER_WORD: Final = rf"(?<![\w{_MARKS}])"
_NOT_BEFORE_WORD: Final = rf"(?![\w{_MARKS}])"
_LIST_SEPARATOR: Final = r"[,،]?"

_ARABIC_LETTER_CLASSES: Final = {
    **dict.fromkeys("اأإآ", "[اأإآٱ]"),
    **dict.fromkeys("يىی", "[يىی]"),
    **dict.fromkeys("ةهۃہھ", "[ةهۃہھ]"),
    **dict.fromkeys("كک", "[كک]"),
}
_ARABIC_HARAKAT: Final = "[\u064b-\u065f\u0670]*"
_ARABIC_JOINER: Final = "[ \u00a0\u200c]?"
_GERESH: Final = "[׳'’]"
_GERSHAYIM: Final = '[״"”“]'

# Hebrew-numeral days 1-30 (ט״ו and ט״ז stand for 15 and 16) and years of the
# sixth millennium from 5500, with an optional ה׳ thousands prefix. Letter
# years must carry gershayim so ordinary words such as תשע are not years.
_HEBREW_DAY: Final = rf"(?:[טיכ]{_GERSHAYIM}?[א-ט]|[א-טיכל]{_GERESH}?)"
_HEBREW_YEAR: Final = (
    rf"(?:ה{_GERESH}?)?ת[קרשת]?"
    rf"(?:[טיכלמנסעפצ]?{_GERSHAYIM}[א-ט]|{_GERSHAYIM}[יכלמנסעפצךםןףץקרשת])"
)
# Ethiopic numerals are additive: ፲፭ is 15 and ፳፻፲፯ is 2017.
_ETHIOPIC_DAY: Final = r"(?:[፲፳][፩-፱]?|፴|[፩-፱])"
_ETHIOPIC_YEAR: Final = r"(?:፲፱|፳)፻[፲-፺]?[፩-፱]?"

_CALENDAR_MAX_DAY: Final = MappingProxyType(
    {
        HIJRI_CALENDAR: 30,
        ETHIOPIAN_CALENDAR: 30,
        BIKRAM_SAMBAT_CALENDAR: 32,
        HEBREW_CALENDAR: 30,
    }
)
# Leading two digits of the accepted four-digit years: Hijri 1300-1499,
# Ethiopian 1900-2099, Bikram Sambat 1900-2199 and Hebrew 5500-5899.
_CALENDAR_YEAR_CENTURIES: Final = MappingProxyType(
    {
        HIJRI_CALENDAR: ("13", "14"),
        ETHIOPIAN_CALENDAR: ("19", "20"),
        BIKRAM_SAMBAT_CALENDAR: ("19", "20", "21"),
        HEBREW_CALENDAR: ("55", "56", "57", "58"),
    }
)
_CALENDAR_NATIVE_DAYS: Final = MappingProxyType(
    {ETHIOPIAN_CALENDAR: _ETHIOPIC_DAY, HEBREW_CALENDAR: _HEBREW_DAY}
)
_CALENDAR_NATIVE_YEARS: Final = MappingProxyType(
    {ETHIOPIAN_CALENDAR: _ETHIOPIC_YEAR, HEBREW_CALENDAR: _HEBREW_YEAR}
)


def _digit_class(values: str, digits: tuple[str, ...]) -> str:
    """Return a class matching each decimal value in ASCII and native digits."""

    return (
        "["
        + "".join(
            script[int(value)] for script in ("0123456789", *digits) for value in values
        )
        + "]"
    )


def _day_pattern(table: CalendarMonthTable) -> str:
    digit = _digit_class("0123456789", table.digits)
    last = "0" if _CALENDAR_MAX_DAY[table.calendar] == 30 else "012"
    numeric = (
        f"{_digit_class('12', table.digits)}{digit}"
        f"|{_digit_class('3', table.digits)}{_digit_class(last, table.digits)}"
        f"|{_digit_class('0', table.digits)}?{_digit_class('123456789', table.digits)}"
    )
    native = _CALENDAR_NATIVE_DAYS.get(table.calendar)
    return f"(?:{numeric}|{native})" if native else f"(?:{numeric})"


def _year_pattern(table: CalendarMonthTable) -> str:
    digit = _digit_class("0123456789", table.digits)
    centuries = "|".join(
        _digit_class(century[0], table.digits) + _digit_class(century[1], table.digits)
        for century in _CALENDAR_YEAR_CENTURIES[table.calendar]
    )
    numeric = f"(?:{centuries}){digit}{{2}}"
    native = _CALENDAR_NATIVE_YEARS.get(table.calendar)
    return f"(?:{numeric}|{native})" if native else numeric


def _spelling_regex(value: str) -> str:
    """Compile one table spelling into a variant-tolerant expression."""

    arabic_script = any("\u0600" <= char <= "\u06ff" for char in value)
    parts: list[str] = []
    for index, char in enumerate(value):
        if char == " ":
            if arabic_script:
                parts.append(_ARABIC_JOINER)
            elif index and value[index - 1] in "./":
                parts.append(r"\s*")
            else:
                parts.append(r"\s+")
        elif char == "׳":
            parts.append(f"{_GERESH}?")
        elif char in _ARABIC_LETTER_CLASSES:
            parts.append(_ARABIC_LETTER_CLASSES[char] + _ARABIC_HARAKAT)
        elif "\u0621" <= char <= "\u064a" or "\u0671" <= char <= "\u06d3":
            parts.append(re.escape(char) + _ARABIC_HARAKAT)
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def _alternation(values: tuple[str, ...]) -> str:
    """Return a longest-first alternation so shorter spellings never win."""

    ordered = sorted(set(values), key=lambda value: (-len(value), value))
    return "|".join(_spelling_regex(value) for value in ordered)


def _optional_words(
    values: tuple[str, ...], *, before: str = "", after: str = ""
) -> str:
    if not values:
        return ""
    return f"(?:{before}(?:{_alternation(values)}){after})?"


def _start_boundary(prefix_letters: str) -> str:
    """Return a look-behind that admits one or two attached prefix letters."""

    if not prefix_letters:
        return _NOT_AFTER_WORD
    prefix = f"[{prefix_letters}]"
    prefixed = (
        f"{prefix}",
        f"{prefix}{prefix}",
        f"{prefix}[-־]",
        f"{prefix}{prefix}[-־]",
    )
    alternatives = "|".join(f"(?<={_NOT_AFTER_WORD}{letters})" for letters in prefixed)
    return f"(?:{_NOT_AFTER_WORD}|{alternatives})"


def _compile_table(table: CalendarMonthTable) -> tuple[PIIPattern, ...]:
    """Compile one month-name table into one date pattern per order."""

    if table.calendar not in CALENDAR_CODES:  # pragma: no cover - data invariant
        raise ValueError("unknown calendar code in month-name table")
    if not set(table.orders) <= _DATE_ORDERS:  # pragma: no cover - data invariant
        raise ValueError("unknown date order in month-name table")

    day = _day_pattern(table)
    year = _year_pattern(table)
    months = _alternation(tuple(name for month in table.months for name in month))
    month_prefix = (
        f"(?:[{table.prefix_letters}]{{1,2}}[-־]?)?" if table.prefix_letters else ""
    )
    particles = _optional_words(
        table.month_particles, before=month_prefix, after=r"\s+"
    )
    month = f"{particles}{month_prefix}(?:{months})"
    # A month that opens the date leaves its attached prefix to the boundary.
    leading_particles = _optional_words(table.month_particles, after=r"\s+")
    leading_month = f"{leading_particles}(?:{months})"
    day_part = day + _optional_words(
        table.day_suffixes, before=r"\s*", after=_NOT_BEFORE_WORD
    )
    year_particles = _optional_words(table.year_particles, after=r"\s+")
    year_suffix = _optional_words(
        table.year_suffixes, before=r"\s*", after=_NOT_BEFORE_WORD
    )
    era_after = _optional_words(
        table.era_markers, before=r"\s*", after=_NOT_BEFORE_WORD
    )
    era_before = _optional_words(table.era_markers, after=r"\.?\s*")
    year_last = f"{year_particles}{year}{year_suffix}{era_after}"
    year_first = f"{era_before}{year}{year_suffix}"
    start = _start_boundary(table.prefix_letters)

    bodies = {
        "dmy": rf"{day_part}\s+{month}{_LIST_SEPARATOR}\s+{year_last}",
        "mdy": rf"{leading_month}\s+{day_part}{_LIST_SEPARATOR}\s+{year_last}",
        "ymd": rf"{year_first}{_LIST_SEPARATOR}\s+{month}\s+{day_part}",
    }
    return tuple(
        PIIPattern(
            f"{start}{bodies[order]}{_NOT_BEFORE_WORD}",
            "date",
            priority=9,
            flags=0,
            base_score=0.7,
            context_boost=0.25,
            context_words=list(table.context_words),
            calendar=table.calendar,
        )
        for order in table.orders
    )


_COMPILED_PATTERNS: Final[Mapping[str, tuple[PIIPattern, ...]]] = MappingProxyType(
    {
        lang: tuple(pattern for table in tables for pattern in _compile_table(table))
        for lang, tables in CALENDAR_MONTH_TABLES.items()
    }
)


def calendar_month_date_patterns(lang: str) -> list[PIIPattern]:
    """Return non-Gregorian month-name date patterns for a language route.

    Args:
        lang: ISO 639-1 language code, optionally with a region suffix such
            as ``"ar_SA"``.

    Returns:
        The compiled patterns for every calendar table registered for the
        language, or an empty list when the language has none. Each pattern
        carries its calendar code in ``PIIPattern.calendar``.
    """

    code = lang.strip().replace("-", "_").split("_", 1)[0].casefold()
    return list(_COMPILED_PATTERNS.get(code, ()))


__all__ = [
    "BIKRAM_SAMBAT_CALENDAR",
    "CALENDAR_CODES",
    "CALENDAR_MONTH_TABLES",
    "ETHIOPIAN_CALENDAR",
    "HEBREW_CALENDAR",
    "HIJRI_CALENDAR",
    "CalendarMonthTable",
    "calendar_month_date_patterns",
]
