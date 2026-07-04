"""Deterministic TIMEX-style temporal expression detection.

The matcher is intentionally lexicon- and regex-driven.  It recognizes the
date, duration, set, and relative-offset forms needed by the timeline resolver
without using wall-clock state or remote services.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

TimexType = Literal["DATE", "DURATION", "SET"]
RelativeDirection = Literal[
    "none",
    "same",
    "past",
    "future",
    "after_previous",
    "before_previous",
    "since",
    "after_anchor",
    "before_anchor",
    "postop_day",
]

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
_NUMBER_RE = (
    r"\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty"
)
_UNIT_RE = r"days?|weeks?|months?|years?"
_APPROX_RE = r"(?:(?P<approx>about|approximately|around|circa)\s+)?"
_ANCHOR_RE = (
    r"last admission|admission|surgery|operation|procedure|discharge|visit|onset"
)
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
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ORDINAL_SUFFIX_RE = r"(?:st|nd|rd|th)?"


@dataclass(frozen=True)
class TemporalExpression:
    """A recognized temporal expression with provenance offsets.

    Attributes:
        text: Exact surface form from the source note.
        timex_type: TIMEX-style class: ``DATE``, ``DURATION``, or ``SET``.
        start: Inclusive character offset in the source note.
        end: Exclusive character offset in the source note.
        value: Normalized value when it can be expressed without a document
            reference date, such as ``YYYY-MM-DD`` or ``P3W``.
        amount: Numeric amount for relative offsets and durations.
        unit: Canonical unit: ``day``, ``week``, ``month``, or ``year``.
        direction: How a relative expression should be anchored.
        anchor: Optional intra-document anchor label, such as ``admission``.
        uncertainty_days: Symmetric day-level uncertainty for approximate or
            coarse expressions.
        metadata: Stable structured details for downstream provenance.
    """

    text: str
    timex_type: TimexType
    start: int
    end: int
    value: str | None = None
    amount: int | None = None
    unit: str | None = None
    direction: RelativeDirection = "none"
    anchor: str | None = None
    uncertainty_days: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        return {
            "text": self.text,
            "type": self.timex_type,
            "start": self.start,
            "end": self.end,
            "value": self.value,
            "amount": self.amount,
            "unit": self.unit,
            "direction": self.direction,
            "anchor": self.anchor,
            "uncertainty_days": self.uncertainty_days,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _TimexPattern:
    regex: re.Pattern[str]
    handler: Callable[[re.Match[str]], TemporalExpression | None]


def detect_timexes(text: str) -> tuple[TemporalExpression, ...]:
    """Recognize temporal expressions in ``text``.

    Matching is deterministic and overlap-safe: when patterns overlap, the
    longest surface span wins and lower-priority nested matches are discarded.
    """

    candidates: list[TemporalExpression] = []
    for pattern in _PATTERNS:
        for match in pattern.regex.finditer(text):
            expression = pattern.handler(match)
            if expression is not None:
                candidates.append(expression)
    return tuple(_select_non_overlapping(candidates))


def parse_number(value: str) -> int | None:
    """Parse a small integer written as digits or a clinical prose word."""

    normalized = value.casefold().strip()
    if normalized.isdigit():
        return int(normalized)
    return _NUMBER_WORDS.get(normalized)


def normalize_unit(value: str) -> str:
    """Return the canonical singular date unit for ``value``."""

    unit = value.casefold().strip()
    if unit.startswith("day"):
        return "day"
    if unit.startswith("week"):
        return "week"
    if unit.startswith("month"):
        return "month"
    if unit.startswith("year"):
        return "year"
    raise ValueError(f"unsupported temporal unit: {value!r}")


def duration_value(amount: int, unit: str) -> str:
    """Return an ISO-8601 duration value at day granularity or coarser."""

    designator = {
        "day": "D",
        "week": "W",
        "month": "M",
        "year": "Y",
    }[unit]
    return f"P{amount}{designator}"


def _date_expression(
    match: re.Match[str],
    *,
    value: str | None,
    direction: RelativeDirection = "none",
    amount: int | None = None,
    unit: str | None = None,
    anchor: str | None = None,
    uncertainty_days: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> TemporalExpression:
    return TemporalExpression(
        text=match.group(0),
        timex_type="DATE",
        start=match.start(),
        end=match.end(),
        value=value,
        amount=amount,
        unit=unit,
        direction=direction,
        anchor=anchor,
        uncertainty_days=uncertainty_days,
        metadata=metadata or {},
    )


def _relative_expression(
    match: re.Match[str],
    *,
    direction: RelativeDirection,
    metadata: Mapping[str, Any] | None = None,
) -> TemporalExpression | None:
    amount = parse_number(match.group("amount"))
    if amount is None:
        return None
    unit = normalize_unit(match.group("unit"))
    uncertainty_days = _unit_uncertainty_days(unit)
    if match.groupdict().get("approx"):
        uncertainty_days = max(uncertainty_days, 1)
    return _date_expression(
        match,
        value=duration_value(amount, unit),
        direction=direction,
        amount=amount,
        unit=unit,
        uncertainty_days=uncertainty_days,
        metadata={
            "relative": True,
            "approximate": bool(match.groupdict().get("approx")),
            **(metadata or {}),
        },
    )


def _duration_expression(
    match: re.Match[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> TemporalExpression | None:
    amount = parse_number(match.group("amount"))
    if amount is None:
        return None
    unit = normalize_unit(match.group("unit"))
    return TemporalExpression(
        text=match.group(0),
        timex_type="DURATION",
        start=match.start(),
        end=match.end(),
        value=duration_value(amount, unit),
        amount=amount,
        unit=unit,
        uncertainty_days=_unit_uncertainty_days(unit),
        metadata=metadata or {},
    )


def _parse_iso_date(match: re.Match[str]) -> TemporalExpression | None:
    value = match.group("date")
    try:
        date.fromisoformat(value)
    except ValueError:
        return None
    return _date_expression(match, value=value, metadata={"precision": "day"})


def _parse_slash_date(match: re.Match[str]) -> TemporalExpression | None:
    month = int(match.group("month"))
    day = int(match.group("day"))
    year = int(match.group("year"))
    if year < 100:
        year += 2000
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    return _date_expression(
        match,
        value=parsed.isoformat(),
        metadata={"precision": "day", "format": "numeric"},
    )


def _parse_month_day_year(match: re.Match[str]) -> TemporalExpression | None:
    month = _MONTHS[match.group("month").casefold()]
    day = int(match.group("day"))
    year = int(match.group("year"))
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    return _date_expression(
        match,
        value=parsed.isoformat(),
        metadata={"precision": "day", "format": "month_name"},
    )


def _parse_month_year(match: re.Match[str]) -> TemporalExpression | None:
    month = _MONTHS[match.group("month").casefold()]
    year = int(match.group("year"))
    if not 1 <= month <= 12:
        return None
    last_day = calendar.monthrange(year, month)[1]
    return _date_expression(
        match,
        value=f"{year:04d}-{month:02d}",
        uncertainty_days=max(1, last_day // 2),
        metadata={"precision": "month", "last_day": last_day},
    )


def _parse_relative_keyword(match: re.Match[str]) -> TemporalExpression:
    keyword = match.group(0).casefold()
    if keyword == "yesterday":
        return _date_expression(
            match,
            value="P1D",
            direction="past",
            amount=1,
            unit="day",
            metadata={"relative": True, "keyword": keyword},
        )
    if keyword == "tomorrow":
        return _date_expression(
            match,
            value="P1D",
            direction="future",
            amount=1,
            unit="day",
            metadata={"relative": True, "keyword": keyword},
        )
    return _date_expression(
        match,
        value="P0D",
        direction="same",
        amount=0,
        unit="day",
        metadata={"relative": True, "keyword": keyword},
    )


def _parse_since_anchor(match: re.Match[str]) -> TemporalExpression:
    anchor = _canonical_anchor(match.group("anchor"))
    return _date_expression(
        match,
        value=None,
        direction="since",
        anchor=anchor,
        metadata={"relative": True, "anchor_required": True},
    )


def _parse_anchor_relation(match: re.Match[str]) -> TemporalExpression:
    relation = match.group("relation").casefold()
    direction: RelativeDirection = (
        "before_anchor" if relation == "before" else "after_anchor"
    )
    return _date_expression(
        match,
        value=None,
        direction=direction,
        anchor=_canonical_anchor(match.group("anchor")),
        metadata={"relative": True, "anchor_required": True},
    )


def _parse_postop_day(match: re.Match[str]) -> TemporalExpression | None:
    amount = parse_number(match.group("amount"))
    if amount is None:
        return None
    return _date_expression(
        match,
        value=duration_value(amount, "day"),
        direction="postop_day",
        amount=amount,
        unit="day",
        anchor="surgery",
        metadata={"relative": True, "anchor_required": True},
    )


def _parse_set(match: re.Match[str]) -> TemporalExpression:
    text = " ".join(match.group(0).casefold().split())
    return TemporalExpression(
        text=match.group(0),
        timex_type="SET",
        start=match.start(),
        end=match.end(),
        value=_SET_VALUES[text],
        metadata={"frequency": text},
    )


def _canonical_anchor(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    if normalized == "last admission":
        return "admission"
    if normalized == "operation":
        return "surgery"
    if normalized == "procedure":
        return "procedure"
    return normalized


def _unit_uncertainty_days(unit: str) -> int:
    return {
        "day": 0,
        "week": 3,
        "month": 15,
        "year": 183,
    }[unit]


def _select_non_overlapping(
    candidates: Iterable[TemporalExpression],
) -> list[TemporalExpression]:
    selected: list[TemporalExpression] = []
    occupied: list[range] = []
    ordered = sorted(
        candidates, key=lambda item: (item.start, -(item.end - item.start))
    )
    for candidate in ordered:
        candidate_range = range(candidate.start, candidate.end)
        if any(_ranges_overlap(candidate_range, seen) for seen in occupied):
            continue
        selected.append(candidate)
        occupied.append(candidate_range)
    return sorted(selected, key=lambda item: (item.start, item.end))


def _ranges_overlap(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


_SET_VALUES = {
    "daily": "P1D",
    "every day": "P1D",
    "weekly": "P1W",
    "every week": "P1W",
    "monthly": "P1M",
    "every month": "P1M",
    "annually": "P1Y",
    "yearly": "P1Y",
    "every year": "P1Y",
    "bid": "P0.5D",
    "tid": "P0.333D",
    "qid": "P0.25D",
    "q6h": "PT6H",
}

_PATTERNS = (
    _TimexPattern(
        re.compile(r"(?<!\d)(?P<date>\d{4}-\d{2}-\d{2})(?!\d)"),
        _parse_iso_date,
    ),
    _TimexPattern(
        re.compile(
            rf"\b(?P<month>{_MONTH_RE})\s+"
            rf"(?P<day>\d{{1,2}}){_ORDINAL_SUFFIX_RE},?\s+"
            r"(?P<year>\d{4})\b",
            re.IGNORECASE,
        ),
        _parse_month_day_year,
    ),
    _TimexPattern(
        re.compile(
            rf"\b(?P<month>{_MONTH_RE})\s+(?P<year>\d{{4}})\b",
            re.IGNORECASE,
        ),
        _parse_month_year,
    ),
    _TimexPattern(
        re.compile(
            r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/"
            r"(?P<year>\d{2,4})\b"
        ),
        _parse_slash_date,
    ),
    _TimexPattern(
        re.compile(
            rf"\b{_APPROX_RE}(?P<amount>{_NUMBER_RE})\s+"
            rf"(?P<unit>{_UNIT_RE})\s+(?:ago|prior|earlier|before)\b",
            re.IGNORECASE,
        ),
        lambda match: _relative_expression(match, direction="past"),
    ),
    _TimexPattern(
        re.compile(
            rf"\b{_APPROX_RE}(?:in|after)\s+"
            rf"(?P<amount>{_NUMBER_RE})\s+(?P<unit>{_UNIT_RE})\b",
            re.IGNORECASE,
        ),
        lambda match: _relative_expression(match, direction="future"),
    ),
    _TimexPattern(
        re.compile(
            rf"\b{_APPROX_RE}(?P<amount>{_NUMBER_RE})\s+"
            rf"(?P<unit>{_UNIT_RE})\s+(?:later|after)\b",
            re.IGNORECASE,
        ),
        lambda match: _relative_expression(
            match,
            direction="after_previous",
            metadata={"relative_to_previous_anchor": True},
        ),
    ),
    _TimexPattern(
        re.compile(
            rf"\b{_APPROX_RE}(?P<amount>{_NUMBER_RE})\s+"
            rf"(?P<unit>{_UNIT_RE})\s+history\b",
            re.IGNORECASE,
        ),
        lambda match: _duration_expression(match, metadata={"history_duration": True}),
    ),
    _TimexPattern(
        re.compile(
            rf"\bfor\s+(?P<amount>{_NUMBER_RE})\s+"
            rf"(?P<unit>{_UNIT_RE})\b",
            re.IGNORECASE,
        ),
        _duration_expression,
    ),
    _TimexPattern(
        re.compile(
            rf"\b(?:post[-\s]*op(?:erative)?\s+day|pod\s*#?)\s*"
            rf"(?P<amount>{_NUMBER_RE})\b",
            re.IGNORECASE,
        ),
        _parse_postop_day,
    ),
    _TimexPattern(
        re.compile(
            rf"\bsince\s+(?:the\s+)?(?P<anchor>{_ANCHOR_RE})\b",
            re.IGNORECASE,
        ),
        _parse_since_anchor,
    ),
    _TimexPattern(
        re.compile(
            rf"\b(?P<relation>after|following|before)\s+(?:the\s+)?"
            rf"(?P<anchor>{_ANCHOR_RE})\b",
            re.IGNORECASE,
        ),
        _parse_anchor_relation,
    ),
    _TimexPattern(
        re.compile(r"\b(?:today|yesterday|tomorrow|now|currently)\b", re.IGNORECASE),
        _parse_relative_keyword,
    ),
    _TimexPattern(
        re.compile(
            r"\b(?:daily|weekly|monthly|annually|yearly|every\s+day|"
            r"every\s+week|every\s+month|every\s+year|bid|tid|qid|q6h)\b",
            re.IGNORECASE,
        ),
        _parse_set,
    ),
)


__all__ = [
    "RelativeDirection",
    "TemporalExpression",
    "TimexType",
    "detect_timexes",
    "duration_value",
    "normalize_unit",
    "parse_number",
]
