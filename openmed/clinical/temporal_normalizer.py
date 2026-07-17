"""Deterministic TIMEX3-style normalization for supplied temporal spans.

The normalizer is deliberately rules-based and has no wall-clock or network
dependency. Relative expressions are resolved only when the caller supplies a
document reference time; otherwise the result remains explicitly unanchored.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

TimexType = Literal["DATE", "TIME", "DURATION", "SET"]

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
}
_SMALL_NUMBER_WORD = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen"
)
_NUMBER = (
    rf"(?:\d+|{_SMALL_NUMBER_WORD}|"
    rf"(?:twenty|thirty)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
)
_UNIT = r"seconds?|minutes?|hours?|days?|weeks?|months?|years?"
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY = "|".join(_WEEKDAYS)
_APPROXIMATE_PREFIX = r"(?:(?P<approx>about|around|approximately|roughly)\s+)?"

_ISO_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}"
    r"(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONTH_DAY_YEAR_RE = re.compile(
    rf"(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    r",?\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
_DAY_MONTH_YEAR_RE = re.compile(
    rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
    rf"(?P<month>{_MONTH})\s+(?P<year>\d{{4}})",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(rf"(?P<month>{_MONTH})\s+(?P<year>\d{{4}})", re.IGNORECASE)
_MONTH_ONLY_RE = re.compile(
    rf"(?:(?P<part>early|mid|late)[- ]+)?(?P<month>{_MONTH})",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(
    r"(?P<first>\d{1,2})/(?P<second>\d{1,2})/(?P<year>\d{2}|\d{4})"
)
_YEAR_RE = re.compile(r"\d{4}")
_TIME_24_RE = re.compile(
    r"(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)"
    r"(?::(?P<second>[0-5]\d))?"
)
_TIME_12_RE = re.compile(
    r"(?P<hour>0?[1-9]|1[0-2])"
    r"(?::(?P<minute>[0-5]\d))?\s*(?P<period>a\.?m\.?|p\.?m\.?)",
    re.IGNORECASE,
)
_PART_OF_DAY_RE = re.compile(
    r"(?:this\s+)?(?:morning|afternoon|evening|night|noon|midnight)",
    re.IGNORECASE,
)
_RELATIVE_OFFSET_RE = re.compile(
    rf"{_APPROXIMATE_PREFIX}(?:(?P<in>in)\s+)?"
    rf"(?P<amount>{_NUMBER})\s+(?P<unit>{_UNIT})\s*"
    r"(?P<direction>ago|earlier|prior|before|from now|later)?",
    re.IGNORECASE,
)
_VAGUE_RELATIVE_RE = re.compile(
    rf"(?:several|few|many)\s+(?P<unit>{_UNIT})\s+"
    r"(?:ago|earlier|prior|before|later)",
    re.IGNORECASE,
)
_LAST_NEXT_RE = re.compile(
    rf"(?P<direction>last|next)\s+"
    rf"(?P<target>{_MONTH}|{_WEEKDAY}|day|week|month|year)",
    re.IGNORECASE,
)
_POSTOP_RE = re.compile(
    rf"(?:pod|post[- ]?op(?:erative)?\s+day)\s*#?\s*(?P<amount>{_NUMBER})",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    rf"(?:for\s+|x\s*)?(?P<amount>{_NUMBER})\s+(?P<unit>{_UNIT})",
    re.IGNORECASE,
)
_BOUNDED_SET_RE = re.compile(
    rf"(?:q\s*(?P<period>\d+)\s*(?P<period_unit>[hHdD])|"
    rf"every\s+(?P<every_amount>{_NUMBER})\s+"
    rf"(?P<every_unit>hours?|days?))\s+"
    rf"(?:x\s*|for\s+)(?P<duration>{_NUMBER})\s+"
    r"(?P<duration_unit>hours?|days?|weeks?)",
    re.IGNORECASE,
)
_INTERVAL_SET_RE = re.compile(
    rf"(?:q\s*(?P<period>\d+)\s*(?P<period_unit>[hHdD])|"
    rf"every\s+(?P<every_amount>{_NUMBER})\s+"
    rf"(?P<every_unit>hours?|days?))",
    re.IGNORECASE,
)
_NAMED_SET_VALUES = {
    "daily": ("R/P1D", "day"),
    "every day": ("R/P1D", "day"),
    "weekly": ("R/P1W", "week"),
    "every week": ("R/P1W", "week"),
    "monthly": ("R/P1M", "month"),
    "every month": ("R/P1M", "month"),
    "yearly": ("R/P1Y", "year"),
    "annually": ("R/P1Y", "year"),
    "bid": ("R/PT12H", "hour"),
    "tid": ("R/PT8H", "hour"),
    "qid": ("R/PT6H", "hour"),
}
_NAMED_SET_RE = re.compile(
    "|".join(
        re.escape(value) for value in sorted(_NAMED_SET_VALUES, key=len, reverse=True)
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NormalizedTimex:
    """One normalized temporal expression with source-span provenance.

    ``granularity_flags`` always includes the narrowest supported precision
    (for example ``"day"`` or ``"month"``). Qualifiers such as
    ``"ambiguous"`` and ``"unanchored"`` explain why ``value`` is partial or
    absent instead of silently inventing precision.
    """

    text: str
    start: int
    end: int
    timex_type: TimexType
    value: str | None
    anchor: str | None
    granularity_flags: tuple[str, ...]

    @property
    def span(self) -> tuple[int, int]:
        """Return the original inclusive/exclusive source offsets."""

        return (self.start, self.end)

    @property
    def type(self) -> TimexType:
        """Return the TIMEX3 type using the field name used in JSON output."""

        return self.timex_type

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        return {
            "text": self.text,
            "span": [self.start, self.end],
            "start": self.start,
            "end": self.end,
            "type": self.timex_type,
            "value": self.value,
            "anchor": self.anchor,
            "granularity_flags": list(self.granularity_flags),
        }


@dataclass(frozen=True)
class _ReferenceTime:
    value: datetime
    iso_value: str


def normalize_temporal(
    text: str,
    spans: Iterable[Mapping[str, object] | Sequence[int]],
    reference_time: str | date | datetime | None,
) -> list[NormalizedTimex]:
    """Normalize caller-supplied temporal spans to TIMEX3-style ISO values.

    Args:
        text: Original document text.
        spans: Mappings with integer ``start``/``end`` offsets or two-item
            ``(start, end)`` sequences. Each span is normalized independently.
        reference_time: Document reference time used for relative expressions.
            An ISO string, :class:`date`, :class:`datetime`, or ``None`` is
            accepted. ``None`` never falls back to the current wall clock.

    Returns:
        Normalized records in the same order as the supplied spans.

    Raises:
        TypeError: If ``text`` or ``reference_time`` has an unsupported type.
        ValueError: If a reference string or source span is invalid.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    reference = _coerce_reference_time(reference_time)
    records: list[NormalizedTimex] = []
    for raw_span in spans:
        start, end = _coerce_span(raw_span, text_length=len(text))
        phrase = text[start:end]
        normalized = _normalize_phrase(phrase, reference)
        records.append(
            NormalizedTimex(
                text=phrase,
                start=start,
                end=end,
                timex_type=normalized.timex_type,
                value=normalized.value,
                anchor=normalized.anchor,
                granularity_flags=normalized.granularity_flags,
            )
        )
    return records


def _normalize_phrase(
    phrase: str,
    reference: _ReferenceTime | None,
) -> NormalizedTimex:
    stripped = phrase.strip()
    lowered = " ".join(stripped.casefold().split())
    leading = len(phrase) - len(phrase.lstrip())
    start = leading
    end = leading + len(stripped)

    if lowered.startswith("since "):
        return _normalize_since(stripped, start, end, reference)

    if match := _BOUNDED_SET_RE.fullmatch(stripped):
        return _normalize_interval_set(stripped, start, end, match, bounded=True)
    if match := _INTERVAL_SET_RE.fullmatch(stripped):
        return _normalize_interval_set(stripped, start, end, match, bounded=False)
    if _NAMED_SET_RE.fullmatch(stripped):
        value, granularity = _NAMED_SET_VALUES[lowered]
        return _result(stripped, start, end, "SET", value, None, granularity)

    if match := _POSTOP_RE.fullmatch(stripped):
        amount = _parse_number(match.group("amount"))
        if reference is None:
            return _result(
                stripped, start, end, "DATE", None, None, "day", "unanchored"
            )
        shifted = reference.value.date() + timedelta(days=amount)
        return _result(
            stripped,
            start,
            end,
            "DATE",
            shifted.isoformat(),
            reference.iso_value,
            "day",
        )

    if lowered in {"today", "yesterday", "tomorrow"}:
        return _normalize_relative_keyword(stripped, start, end, lowered, reference)

    if match := _VAGUE_RELATIVE_RE.fullmatch(stripped):
        unit = _canonical_unit(match.group("unit"))
        vague_flags = [_unit_granularity(unit), "ambiguous"]
        if reference is None:
            vague_flags.append("unanchored")
        return _result(
            stripped,
            start,
            end,
            "DATE",
            None,
            reference.iso_value if reference else None,
            *vague_flags,
        )

    if match := _RELATIVE_OFFSET_RE.fullmatch(stripped):
        direction = (match.group("direction") or "").casefold()
        is_future = bool(match.group("in")) or direction in {"from now", "later"}
        is_relative = bool(match.group("in")) or bool(direction)
        if is_relative:
            return _normalize_relative_offset(
                stripped, start, end, match, reference, is_future=is_future
            )

    if match := _LAST_NEXT_RE.fullmatch(stripped):
        return _normalize_last_next(stripped, start, end, match, reference)

    if _ISO_DATETIME_RE.fullmatch(stripped):
        try:
            parsed_datetime = datetime.fromisoformat(_replace_z(stripped))
        except ValueError:
            return _ambiguous_date(stripped, start, end, "second")
        granularity = "second" if parsed_datetime.second else "minute"
        return _result(
            stripped,
            start,
            end,
            "DATE",
            parsed_datetime.isoformat(),
            None,
            granularity,
        )

    if _ISO_DATE_RE.fullmatch(stripped):
        try:
            parsed_date = date.fromisoformat(stripped)
        except ValueError:
            return _ambiguous_date(stripped, start, end, "day")
        return _result(
            stripped, start, end, "DATE", parsed_date.isoformat(), None, "day"
        )

    if match := _MONTH_DAY_YEAR_RE.fullmatch(stripped):
        return _normalize_named_date(stripped, start, end, match)
    if match := _DAY_MONTH_YEAR_RE.fullmatch(stripped):
        return _normalize_named_date(stripped, start, end, match)
    if match := _MONTH_YEAR_RE.fullmatch(stripped):
        month = _MONTHS[match.group("month").casefold()]
        year = int(match.group("year"))
        return _result(
            stripped, start, end, "DATE", f"{year:04d}-{month:02d}", None, "month"
        )
    if match := _NUMERIC_DATE_RE.fullmatch(stripped):
        return _normalize_numeric_date(stripped, start, end, match)
    if _YEAR_RE.fullmatch(stripped):
        return _result(stripped, start, end, "DATE", stripped, None, "year")
    if match := _MONTH_ONLY_RE.fullmatch(stripped):
        month_flags = ["month", "ambiguous", "unanchored"]
        if match.group("part"):
            month_flags.insert(1, "part_of_month")
        return _result(stripped, start, end, "DATE", None, None, *month_flags)

    if match := _TIME_24_RE.fullmatch(stripped):
        parsed_time = time(
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second") or 0),
        )
        granularity = "second" if match.group("second") else "minute"
        return _normalize_time(
            stripped, start, end, parsed_time, reference, granularity
        )
    if match := _TIME_12_RE.fullmatch(stripped):
        hour = int(match.group("hour")) % 12
        if match.group("period").casefold().startswith("p"):
            hour += 12
        parsed_time = time(hour, int(match.group("minute") or 0))
        granularity = "minute" if match.group("minute") else "hour"
        return _normalize_time(
            stripped, start, end, parsed_time, reference, granularity
        )
    if _PART_OF_DAY_RE.fullmatch(stripped):
        return _result(
            stripped,
            start,
            end,
            "TIME",
            None,
            reference.iso_value if reference else None,
            "part_of_day",
            "ambiguous",
            *("unanchored",) if reference is None else (),
        )

    if match := _DURATION_RE.fullmatch(stripped):
        amount = _parse_number(match.group("amount"))
        unit = _canonical_unit(match.group("unit"))
        return _result(
            stripped,
            start,
            end,
            "DURATION",
            _duration_value(amount, unit),
            None,
            _unit_granularity(unit),
        )

    unknown_flags = ["unknown", "ambiguous"]
    if reference is None:
        unknown_flags.append("unanchored")
    return _result(
        stripped,
        start,
        end,
        _infer_timex_type(lowered),
        None,
        reference.iso_value if reference else None,
        *unknown_flags,
    )


def _normalize_since(
    phrase: str,
    start: int,
    end: int,
    reference: _ReferenceTime | None,
) -> NormalizedTimex:
    inner_text = phrase[6:].strip()
    if inner_text.casefold() in {"admission", "encounter", "visit", "now", "today"}:
        if reference is None:
            return _result(
                phrase,
                start,
                end,
                "DATE",
                None,
                None,
                "day",
                "since",
                "unanchored",
            )
        return _result(
            phrase,
            start,
            end,
            "DATE",
            reference.value.date().isoformat(),
            reference.iso_value,
            "day",
            "since",
        )

    inner = _normalize_phrase(inner_text, reference)
    flags = list(inner.granularity_flags)
    if "since" not in flags:
        flags.append("since")
    if reference is None and "unanchored" not in flags:
        flags.append("unanchored")
    return _result(
        phrase,
        start,
        end,
        "DATE",
        inner.value,
        reference.iso_value if reference else None,
        *flags,
    )


def _normalize_relative_keyword(
    phrase: str,
    start: int,
    end: int,
    keyword: str,
    reference: _ReferenceTime | None,
) -> NormalizedTimex:
    if reference is None:
        return _result(phrase, start, end, "DATE", None, None, "day", "unanchored")
    delta = {"yesterday": -1, "today": 0, "tomorrow": 1}[keyword]
    value = reference.value.date() + timedelta(days=delta)
    return _result(
        phrase,
        start,
        end,
        "DATE",
        value.isoformat(),
        reference.iso_value,
        "day",
    )


def _normalize_relative_offset(
    phrase: str,
    start: int,
    end: int,
    match: re.Match[str],
    reference: _ReferenceTime | None,
    *,
    is_future: bool,
) -> NormalizedTimex:
    amount = _parse_number(match.group("amount"))
    unit = _canonical_unit(match.group("unit"))
    flags = [_resolved_granularity(unit)]
    if match.group("approx"):
        flags.append("approximate")
    if reference is None:
        flags.append("unanchored")
        return _result(phrase, start, end, "DATE", None, None, *flags)
    signed_amount = amount if is_future else -amount
    shifted = _shift_reference(reference.value, signed_amount, unit)
    return _result(
        phrase,
        start,
        end,
        "DATE",
        _shifted_iso_value(shifted, unit),
        reference.iso_value,
        *flags,
    )


def _normalize_last_next(
    phrase: str,
    start: int,
    end: int,
    match: re.Match[str],
    reference: _ReferenceTime | None,
) -> NormalizedTimex:
    target = match.group("target").casefold()
    direction = -1 if match.group("direction").casefold() == "last" else 1
    if reference is None:
        return _result(
            phrase,
            start,
            end,
            "DATE",
            None,
            None,
            _last_next_granularity(target),
            "unanchored",
        )

    reference_date = reference.value.date()
    if target in _MONTHS:
        month = _MONTHS[target]
        year = reference_date.year
        if direction < 0 and month >= reference_date.month:
            year -= 1
        elif direction > 0 and month <= reference_date.month:
            year += 1
        value = f"{year:04d}-{month:02d}"
        granularity = "month"
    elif target in _WEEKDAYS:
        target_weekday = _WEEKDAYS[target]
        delta = (target_weekday - reference_date.weekday()) % 7
        if direction < 0:
            delta = delta - 7 if delta else -7
        elif delta == 0:
            delta = 7
        value = (reference_date + timedelta(days=delta)).isoformat()
        granularity = "day"
    elif target == "day":
        value = (reference_date + timedelta(days=direction)).isoformat()
        granularity = "day"
    elif target == "week":
        shifted = reference_date + timedelta(weeks=direction)
        iso_year, iso_week, _ = shifted.isocalendar()
        value = f"{iso_year:04d}-W{iso_week:02d}"
        granularity = "week"
    elif target == "month":
        shifted = _add_months(reference.value, direction)
        value = f"{shifted.year:04d}-{shifted.month:02d}"
        granularity = "month"
    else:
        value = f"{reference_date.year + direction:04d}"
        granularity = "year"
    return _result(
        phrase,
        start,
        end,
        "DATE",
        value,
        reference.iso_value,
        granularity,
    )


def _normalize_named_date(
    phrase: str,
    start: int,
    end: int,
    match: re.Match[str],
) -> NormalizedTimex:
    year = int(match.group("year"))
    month = _MONTHS[match.group("month").casefold()]
    day = int(match.group("day"))
    try:
        parsed = date(year, month, day)
    except ValueError:
        return _ambiguous_date(phrase, start, end, "day")
    return _result(phrase, start, end, "DATE", parsed.isoformat(), None, "day")


def _normalize_numeric_date(
    phrase: str,
    start: int,
    end: int,
    match: re.Match[str],
) -> NormalizedTimex:
    first = int(match.group("first"))
    second = int(match.group("second"))
    year = int(match.group("year"))
    if year < 100:
        year += 2000
    if first <= 12 and second <= 12:
        return _ambiguous_date(phrase, start, end, "day")
    month, day = (first, second) if first <= 12 else (second, first)
    try:
        parsed = date(year, month, day)
    except ValueError:
        return _ambiguous_date(phrase, start, end, "day")
    return _result(phrase, start, end, "DATE", parsed.isoformat(), None, "day")


def _normalize_time(
    phrase: str,
    start: int,
    end: int,
    parsed_time: time,
    reference: _ReferenceTime | None,
    granularity: str,
) -> NormalizedTimex:
    if reference is None:
        return _result(
            phrase,
            start,
            end,
            "TIME",
            parsed_time.isoformat(),
            None,
            granularity,
            "unanchored",
        )
    value = datetime(
        reference.value.year,
        reference.value.month,
        reference.value.day,
        parsed_time.hour,
        parsed_time.minute,
        parsed_time.second,
        tzinfo=reference.value.tzinfo,
    )
    return _result(
        phrase,
        start,
        end,
        "TIME",
        value.isoformat(),
        reference.iso_value,
        granularity,
    )


def _normalize_interval_set(
    phrase: str,
    start: int,
    end: int,
    match: re.Match[str],
    *,
    bounded: bool,
) -> NormalizedTimex:
    if match.group("period"):
        period = int(match.group("period"))
        unit = "hour" if match.group("period_unit").casefold() == "h" else "day"
    else:
        period = _parse_number(match.group("every_amount"))
        unit = _canonical_unit(match.group("every_unit"))
    interval = _duration_value(period, unit)
    value = f"R/{interval}"
    flags = [_unit_granularity(unit)]
    if bounded:
        duration = _parse_number(match.group("duration"))
        duration_unit = _canonical_unit(match.group("duration_unit"))
        repeat_count = _repeat_count(period, unit, duration, duration_unit)
        flags.append("bounded")
        if repeat_count is None:
            flags.append("ambiguous")
        else:
            value = f"R{repeat_count}/{interval}"
    return _result(phrase, start, end, "SET", value, None, *flags)


def _repeat_count(
    period: int,
    period_unit: str,
    duration: int,
    duration_unit: str,
) -> int | None:
    unit_hours = {"hour": 1, "day": 24, "week": 168}
    if period <= 0 or duration <= 0:
        return None
    if period_unit not in unit_hours or duration_unit not in unit_hours:
        return None
    total_hours = duration * unit_hours[duration_unit]
    interval_hours = period * unit_hours[period_unit]
    count, remainder = divmod(total_hours, interval_hours)
    return count if count > 0 and remainder == 0 else None


def _coerce_reference_time(
    value: str | date | datetime | None,
) -> _ReferenceTime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ReferenceTime(value=value, iso_value=value.isoformat())
    if isinstance(value, date):
        combined = datetime.combine(value, time.min)
        return _ReferenceTime(value=combined, iso_value=value.isoformat())
    if not isinstance(value, str):
        raise TypeError("reference_time must be an ISO string, date, datetime, or None")
    normalized = value.strip()
    if not normalized:
        raise ValueError("reference_time must not be empty")
    if _ISO_DATE_RE.fullmatch(normalized):
        try:
            parsed_date = date.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                "reference_time must be a valid ISO date or datetime"
            ) from exc
        return _ReferenceTime(
            value=datetime.combine(parsed_date, time.min),
            iso_value=parsed_date.isoformat(),
        )
    try:
        parsed_datetime = datetime.fromisoformat(_replace_z(normalized))
    except ValueError:
        try:
            parsed_date = date.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                "reference_time must be a valid ISO date or datetime"
            ) from exc
        return _ReferenceTime(
            value=datetime.combine(parsed_date, time.min),
            iso_value=parsed_date.isoformat(),
        )
    return _ReferenceTime(
        value=parsed_datetime,
        iso_value=parsed_datetime.isoformat(),
    )


def _coerce_span(
    raw_span: Mapping[str, object] | Sequence[int],
    *,
    text_length: int,
) -> tuple[int, int]:
    if isinstance(raw_span, Mapping):
        try:
            raw_start = raw_span["start"]
            raw_end = raw_span["end"]
        except KeyError as exc:
            raise ValueError("spans require integer 'start' and 'end'") from exc
    elif isinstance(raw_span, Sequence) and not isinstance(raw_span, (str, bytes)):
        if len(raw_span) != 2:
            raise ValueError("span sequences must contain exactly start and end")
        raw_start, raw_end = raw_span
    else:
        raise TypeError("spans must be mappings or two-item sequences")

    if isinstance(raw_start, bool) or isinstance(raw_end, bool):
        raise ValueError("spans require integer 'start' and 'end'")
    try:
        start = int(str(raw_start))
        end = int(str(raw_end))
    except (TypeError, ValueError) as exc:
        raise ValueError("spans require integer 'start' and 'end'") from exc
    if start < 0 or end <= start or end > text_length:
        raise ValueError("span offsets must satisfy 0 <= start < end <= len(text)")
    return start, end


def _parse_number(value: str) -> int:
    normalized = value.casefold().replace("-", " ").strip()
    if normalized.isdigit():
        return int(normalized)
    if normalized in _NUMBER_WORDS:
        return _NUMBER_WORDS[normalized]
    parts = normalized.split()
    if len(parts) == 2 and parts[0] in {"twenty", "thirty"}:
        return _NUMBER_WORDS[parts[0]] + _NUMBER_WORDS[parts[1]]
    raise ValueError(f"unsupported temporal number: {value!r}")


def _canonical_unit(value: str) -> str:
    normalized = value.casefold()
    for unit in ("second", "minute", "hour", "day", "week", "month", "year"):
        if normalized.startswith(unit):
            return unit
    raise ValueError(f"unsupported temporal unit: {value!r}")


def _duration_value(amount: int, unit: str) -> str:
    designator = {
        "second": "S",
        "minute": "M",
        "hour": "H",
        "day": "D",
        "week": "W",
        "month": "M",
        "year": "Y",
    }[unit]
    prefix = "PT" if unit in {"second", "minute", "hour"} else "P"
    return f"{prefix}{amount}{designator}"


def _unit_granularity(unit: str) -> str:
    return unit


def _resolved_granularity(unit: str) -> str:
    return unit if unit in {"second", "minute", "hour"} else "day"


def _last_next_granularity(target: str) -> str:
    if target in _MONTHS or target == "month":
        return "month"
    if target == "year":
        return "year"
    if target == "week":
        return "week"
    return "day"


def _shift_reference(value: datetime, amount: int, unit: str) -> datetime:
    if unit == "second":
        return value + timedelta(seconds=amount)
    if unit == "minute":
        return value + timedelta(minutes=amount)
    if unit == "hour":
        return value + timedelta(hours=amount)
    if unit == "day":
        return value + timedelta(days=amount)
    if unit == "week":
        return value + timedelta(weeks=amount)
    if unit == "month":
        return _add_months(value, amount)
    if unit == "year":
        return _add_months(value, amount * 12)
    raise ValueError(f"unsupported temporal unit: {unit!r}")


def _add_months(value: datetime, amount: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + amount
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _shifted_iso_value(value: datetime, unit: str) -> str:
    if unit in {"second", "minute", "hour"}:
        return value.isoformat()
    return value.date().isoformat()


def _ambiguous_date(
    phrase: str,
    start: int,
    end: int,
    granularity: str,
) -> NormalizedTimex:
    return _result(phrase, start, end, "DATE", None, None, granularity, "ambiguous")


def _infer_timex_type(value: str) -> TimexType:
    if re.search(r"\b(?:q\d|every|daily|weekly|monthly|bid|tid|qid)\b", value):
        return "SET"
    if re.search(r"\b(?:for|x)\s*\d", value):
        return "DURATION"
    if ":" in value or re.search(r"\b(?:am|pm|morning|evening|night)\b", value):
        return "TIME"
    return "DATE"


def _result(
    text: str,
    start: int,
    end: int,
    timex_type: TimexType,
    value: str | None,
    anchor: str | None,
    *flags: str,
) -> NormalizedTimex:
    return NormalizedTimex(
        text=text,
        start=start,
        end=end,
        timex_type=timex_type,
        value=value,
        anchor=anchor,
        granularity_flags=tuple(dict.fromkeys(flags)),
    )


def _replace_z(value: str) -> str:
    return f"{value[:-1]}+00:00" if value.upper().endswith("Z") else value


__all__ = ["NormalizedTimex", "TimexType", "normalize_temporal"]
