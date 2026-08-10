"""Conservative, offline normalization of clinical temporal values.

This module is intentionally narrower than the TIMEX detector.  It accepts
caller-supplied spans and returns structured values without consulting the
wall clock, a locale, a timezone database, or a remote service.  A missing
component is represented as unknown and competing interpretations are
represented as conflicts; neither is filled by a best-effort guess.

The returned records retain source offsets but do not retain source text in
their serialized form.  This keeps reports useful for review while avoiding a
second copy of potentially sensitive note content.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

TemporalKind = Literal["date", "time", "duration", "interval", "unknown"]
TemporalPrecision = Literal[
    "unknown",
    "year",
    "month",
    "day",
    "week",
    "hour",
    "minute",
    "second",
    "mixed",
]
TimezoneState = Literal["not_applicable", "explicit", "unknown", "conflicting"]
NormalizationStatus = Literal["normalized", "unknown", "conflicting"]

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
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO_DATE_RE = re.compile(r"(?P<year>\d{4})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?")
_NUMERIC_DATE_RE = re.compile(
    r"(?P<first>\d{1,2})/(?P<second>\d{1,2})/(?P<year>\d{2}|\d{4})"
)
_NAMED_DATE_RE = re.compile(
    rf"(?P<month>{_MONTH_PATTERN})\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+"
    r"(?P<year>\d{4})",
    re.IGNORECASE,
)
_REVERSE_NAMED_DATE_RE = re.compile(
    rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
    rf"(?P<month>{_MONTH_PATTERN})\s+(?P<year>\d{{4}})",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    rf"(?P<month>{_MONTH_PATTERN})\s+(?P<year>\d{{4}})",
    re.IGNORECASE,
)
_MONTH_ONLY_RE = re.compile(rf"(?P<month>{_MONTH_PATTERN})", re.IGNORECASE)

_ISO_DATETIME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})[T ]"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2})(?:\.(?P<fraction>\d+))?)?"
    r"(?P<zone>Z|UTC|[+-]\d{2}:?\d{2})?",
    re.IGNORECASE,
)
_TIME_24_RE = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2})(?:\.(?P<fraction>\d+))?)?"
)
_TIME_12_RE = re.compile(
    r"(?P<hour>0?[1-9]|1[0-2])"
    r"(?::(?P<minute>\d{2})(?::(?P<second>\d{2})"
    r"(?:\.(?P<fraction>\d+))?)?)?\s*"
    r"(?P<period>a\.?m\.?|p\.?m\.?)",
    re.IGNORECASE,
)
_TIMEZONE_SUFFIX_RE = re.compile(
    r"(?P<base>.+?)(?:\s*)(?P<zone>Z|UTC|[+-]\d{2}:?\d{2})$",
    re.IGNORECASE,
)

_ISO_DURATION_RE = re.compile(
    r"P"
    r"(?:(?P<years>\d+(?:\.\d+)?)Y)?"
    r"(?:(?P<months>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<weeks>\d+(?:\.\d+)?)W)?"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?"
)
_PROSE_DURATION_RE = re.compile(
    r"(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>years?|months?|weeks?|days?|hours?|minutes?|seconds?)",
    re.IGNORECASE,
)

_DURATION_UNIT_MAP = {
    "year": "Y",
    "month": "M",
    "week": "W",
    "day": "D",
    "hour": "H",
    "minute": "M",
    "second": "S",
}
_DURATION_PRECISION = {
    "years": "year",
    "year": "year",
    "months": "month",
    "month": "month",
    "weeks": "week",
    "week": "week",
    "days": "day",
    "day": "day",
    "hours": "hour",
    "hour": "hour",
    "minutes": "minute",
    "minute": "minute",
    "seconds": "second",
    "second": "second",
}

_OPEN_MARKERS = frozenset(
    {"..", "...", "current", "indefinite", "now", "ongoing", "open", "present"}
)
_UNKNOWN_MARKERS = frozenset({"unknown", "unspecified", "undated", "n/a"})


@dataclass(frozen=True)
class TemporalEndpoint:
    """One endpoint or scalar component of a temporal value.

    ``value`` is ``None`` when one or more components cannot be normalized.
    ``unknown_components`` and ``conflicts`` explain that state without
    copying the source surface into an audit record.
    """

    kind: TemporalKind
    value: str | None
    precision: TemporalPrecision
    timezone_state: TimezoneState
    unknown_components: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "unknown_components", _unique(self.unknown_components))
        object.__setattr__(self, "conflicts", _unique(self.conflicts))

    @property
    def timezone(self) -> TimezoneState:
        """Return the explicit timezone state under a short field name."""

        return self.timezone_state

    @property
    def status(self) -> NormalizationStatus:
        """Return whether this endpoint is complete, unknown, or conflicting."""

        if self.conflicts:
            return "conflicting"
        if self.value is None or self.unknown_components:
            return "unknown"
        return "normalized"

    @property
    def is_unknown(self) -> bool:
        """Whether this endpoint contains an unresolved component."""

        return self.status == "unknown"

    @property
    def is_conflicting(self) -> bool:
        """Whether this endpoint has competing interpretations."""

        return self.status == "conflicting"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation without source text."""

        return {
            "kind": self.kind,
            "value": self.value,
            "precision": self.precision,
            "timezone_state": self.timezone_state,
            "status": self.status,
            "unknown_components": list(self.unknown_components),
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True)
class TemporalInterval:
    """A normalized scalar or interval with source-span provenance.

    Scalar date, time, and duration values use ``start`` as their component
    and leave ``end`` unset.  Intervals use both endpoints; an unset endpoint
    is an intentional open bound, not an inferred current date or time.
    """

    kind: TemporalKind
    source_start: int
    source_end: int
    value: str | None
    precision: TemporalPrecision
    timezone_state: TimezoneState
    start: TemporalEndpoint | None = None
    end: TemporalEndpoint | None = None
    open_start: bool = False
    open_end: bool = False
    start_inclusive: bool = True
    end_inclusive: bool = True
    unknown_components: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "unknown_components", _unique(self.unknown_components))
        object.__setattr__(self, "conflicts", _unique(self.conflicts))

    @property
    def span(self) -> tuple[int, int]:
        """Return the half-open source offsets."""

        return (self.source_start, self.source_end)

    @property
    def source_span(self) -> tuple[int, int]:
        """Return the source offsets under an explicit provenance name."""

        return self.span

    @property
    def start_offset(self) -> int:
        """Return the inclusive source start offset."""

        return self.source_start

    @property
    def end_offset(self) -> int:
        """Return the exclusive source end offset."""

        return self.source_end

    @property
    def normalized_value(self) -> str | None:
        """Return the canonical value when it is not conflicting."""

        return self.value

    @property
    def type(self) -> TemporalKind:
        """Return the kind under the conventional serialized field name."""

        return self.kind

    @property
    def timezone(self) -> TimezoneState:
        """Return the interval timezone state under a short field name."""

        return self.timezone_state

    @property
    def status(self) -> NormalizationStatus:
        """Return whether this value is complete, unknown, or conflicting."""

        if self.conflicts:
            return "conflicting"
        if self.value is None or self.unknown_components:
            return "unknown"
        return "normalized"

    @property
    def is_open_ended(self) -> bool:
        """Whether either side of this value is intentionally unbounded."""

        return self.open_start or self.open_end

    @property
    def is_unknown(self) -> bool:
        """Whether this value contains an unresolved component."""

        return self.status == "unknown"

    @property
    def is_conflicting(self) -> bool:
        """Whether this value has competing interpretations."""

        return self.status == "conflicting"

    @property
    def duration(self) -> str | None:
        """Return the normalized duration for duration records."""

        return self.value if self.kind == "duration" else None

    def to_dict(self) -> dict[str, Any]:
        """Return a privacy-safe JSON representation.

        The output intentionally includes offsets and hashes-free structured
        values, but never includes the raw source surface.
        """

        return {
            "source_span": [self.source_start, self.source_end],
            "start_offset": self.source_start,
            "end_offset": self.source_end,
            "type": self.kind,
            "kind": self.kind,
            "value": self.value,
            "normalized_value": self.value,
            "precision": self.precision,
            "timezone_state": self.timezone_state,
            "status": self.status,
            "start": self.start.to_dict() if self.start is not None else None,
            "end": self.end.to_dict() if self.end is not None else None,
            "open_start": self.open_start,
            "open_end": self.open_end,
            "start_inclusive": self.start_inclusive,
            "end_inclusive": self.end_inclusive,
            "unknown_components": list(self.unknown_components),
            "conflicts": list(self.conflicts),
        }


SpanInput = Mapping[str, object] | Sequence[int]


def normalize_temporal_intervals(
    text: str,
    spans: Iterable[SpanInput] | None = None,
) -> list[TemporalInterval]:
    """Normalize caller-supplied temporal spans in source order.

    Args:
        text: Source document text. It is used transiently to read spans and
            is never copied into serialized results.
        spans: Mappings with integer ``start``/``end`` offsets or two-item
            sequences. When omitted, the complete text is normalized as one
            span.

    Returns:
        One deterministic record per input span, preserving input order.

    Raises:
        TypeError: If ``text`` is not a string or a span is not a supported
            mapping/sequence.
        ValueError: If a source span is outside the text or malformed.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    supplied_spans = spans if spans is not None else ((0, len(text)),)
    return [normalize_temporal_interval(text, span) for span in supplied_spans]


def normalize_temporal_interval(
    text: str,
    span: SpanInput | int | None = None,
    end: int | None = None,
    *,
    start: int | None = None,
) -> TemporalInterval:
    """Normalize one temporal span while preserving its source offsets.

    ``span`` may be a mapping, a two-item sequence, or an integer start offset
    when ``end`` is supplied. With neither ``span`` nor ``end``, the complete
    input text is normalized.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if start is not None:
        if span is not None:
            raise ValueError("start cannot be combined with a span object")
        span = start
    if span is None:
        if end is not None:
            raise ValueError("end requires a start offset")
        start, stop = 0, len(text)
    elif isinstance(span, int) and not isinstance(span, bool):
        if end is None:
            raise ValueError("an end offset is required")
        start, stop = _coerce_span((span, end), text_length=len(text))
    else:
        if end is not None:
            raise ValueError("end cannot be combined with a span object")
        start, stop = _coerce_span(span, text_length=len(text))  # type: ignore[arg-type]
    return _normalize_surface(text[start:stop], start, stop)


def parse_temporal_value(
    value: str,
    *,
    source_start: int = 0,
    source_end: int | None = None,
) -> TemporalInterval:
    """Normalize one value directly, defaulting its source span to the value.

    This helper is useful when a tokenizer already owns the source offsets.
    It performs the same conservative parsing as
    :func:`normalize_temporal_interval` and has no wall-clock fallback.
    """

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if isinstance(source_start, bool) or not isinstance(source_start, int):
        raise ValueError("source_start must be an integer")
    stop = len(value) + source_start if source_end is None else source_end
    if isinstance(stop, bool) or not isinstance(stop, int) or stop < source_start:
        raise ValueError("source offsets are invalid")
    return _normalize_surface(value, source_start, stop)


def parse_temporal_interval(
    value: str,
    *,
    source_start: int = 0,
    source_end: int | None = None,
) -> TemporalInterval:
    """Alias for :func:`parse_temporal_value` with interval-oriented naming."""

    return parse_temporal_value(
        value,
        source_start=source_start,
        source_end=source_end,
    )


def normalize_interval(
    text: str,
    span: SpanInput | int | None = None,
    end: int | None = None,
) -> TemporalInterval:
    """Compatibility alias for :func:`normalize_temporal_interval`."""

    return normalize_temporal_interval(text, span, end)


class TemporalIntervalNormalizer:
    """Stateless convenience wrapper around the module-level normalizer."""

    def normalize(
        self,
        text: str,
        spans: Iterable[SpanInput] | None = None,
    ) -> list[TemporalInterval]:
        """Normalize zero or more spans without adding runtime state."""

        return normalize_temporal_intervals(text, spans)

    def __call__(
        self,
        text: str,
        spans: Iterable[SpanInput] | None = None,
    ) -> list[TemporalInterval]:
        """Call :meth:`normalize` directly."""

        return self.normalize(text, spans)


def _normalize_surface(surface: str, start: int, end: int) -> TemporalInterval:
    value = _clean(surface)
    parts = _interval_parts(value)
    if parts is not None:
        return _build_interval(
            parts[0],
            parts[1],
            start,
            end,
            open_start=parts[2],
            open_end=parts[3],
            start_inclusive=parts[4],
            end_inclusive=parts[5],
        )

    endpoint = _parse_endpoint(value)
    return TemporalInterval(
        kind=endpoint.kind,
        source_start=start,
        source_end=end,
        value=endpoint.value,
        precision=endpoint.precision,
        timezone_state=endpoint.timezone_state,
        start=endpoint,
        unknown_components=endpoint.unknown_components,
        conflicts=endpoint.conflicts,
    )


def _parse_endpoint(value: str) -> TemporalEndpoint:
    if not value:
        return _unknown_endpoint("unknown", ("value",))

    if value.casefold() in _UNKNOWN_MARKERS:
        return _unknown_endpoint("unknown", ("value",))

    if _ISO_DURATION_RE.fullmatch(value.upper()) or _looks_like_duration(value):
        return _parse_duration(value)

    if _ISO_DATETIME_RE.fullmatch(value):
        return _parse_datetime(value)

    date_endpoint = _parse_date(value)
    if date_endpoint is not None:
        return date_endpoint

    time_endpoint = _parse_time(value)
    if time_endpoint is not None:
        return time_endpoint

    if re.search(r"\b(?:year|month|week|day|hour|minute|second)s?\b", value, re.I):
        return _parse_duration(value)
    return _unknown_endpoint("unknown", ("value",))


def _parse_date(value: str) -> TemporalEndpoint | None:
    match = _ISO_DATE_RE.fullmatch(value)
    if match is not None:
        year = int(match.group("year"))
        month_text = match.group("month")
        day_text = match.group("day")
        if month_text is None:
            try:
                date(year, 1, 1)
            except ValueError:
                return _unknown_endpoint("date", "year")
            return _endpoint("date", value, "year", "not_applicable")
        month = int(month_text)
        if not 1 <= month <= 12:
            return _unknown_endpoint("date", "month")
        if day_text is None:
            return _endpoint(
                "date", f"{year:04d}-{month:02d}", "month", "not_applicable"
            )
        day = int(day_text)
        try:
            parsed = date(year, month, day)
        except ValueError:
            component = "day" if 1 <= day <= 31 else "month_or_day"
            return _unknown_endpoint("date", component, precision="day")
        return _endpoint("date", parsed.isoformat(), "day", "not_applicable")

    match = _NAMED_DATE_RE.fullmatch(value) or _REVERSE_NAMED_DATE_RE.fullmatch(value)
    if match is not None:
        month = _MONTHS[match.group("month").casefold()]
        day = int(match.group("day"))
        year = int(match.group("year"))
        try:
            parsed = date(year, month, day)
        except ValueError:
            return _unknown_endpoint("date", "day", precision="day")
        return _endpoint("date", parsed.isoformat(), "day", "not_applicable")

    match = _MONTH_YEAR_RE.fullmatch(value)
    if match is not None:
        month = _MONTHS[match.group("month").casefold()]
        year = int(match.group("year"))
        try:
            date(year, month, 1)
        except ValueError:
            return _unknown_endpoint("date", "year", precision="month")
        return _endpoint("date", f"{year:04d}-{month:02d}", "month", "not_applicable")

    match = _NUMERIC_DATE_RE.fullmatch(value)
    if match is not None:
        first = int(match.group("first"))
        second = int(match.group("second"))
        year_text = match.group("year")
        if len(year_text) != 4:
            return _conflicting_endpoint("date", "day", "date_year")
        if first <= 12 and second <= 12:
            return _conflicting_endpoint("date", "day", "date_order")
        if first > 12 and second > 12:
            return _unknown_endpoint("date", "month_or_day", precision="day")
        month, day = (second, first) if first > 12 else (first, second)
        try:
            parsed = date(int(year_text), month, day)
        except ValueError:
            return _unknown_endpoint("date", "day", precision="day")
        return _endpoint("date", parsed.isoformat(), "day", "not_applicable")

    if _MONTH_ONLY_RE.fullmatch(value) is not None:
        return _unknown_endpoint("date", "year", precision="month")
    return None


def _parse_datetime(value: str) -> TemporalEndpoint:
    match = _ISO_DATETIME_RE.fullmatch(value)
    if match is None:
        return _unknown_endpoint("time", ("date_or_time",), precision="unknown")

    unknown: list[str] = []
    date_value: date | None = None
    try:
        date_value = date.fromisoformat(match.group("date"))
    except ValueError:
        unknown.append("date")

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second_text = match.group("second")
    second = int(second_text or 0)
    fraction = match.group("fraction") or ""
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
        unknown.append("time")
    if len(fraction) > 6:
        unknown.append("second")

    zone_token = match.group("zone")
    zone_info = _parse_timezone(zone_token)
    if zone_token and zone_info is None:
        unknown.append("timezone")
    elif zone_token is None:
        unknown.append("timezone")

    precision: TemporalPrecision = "second" if second_text is not None else "minute"
    fatal_unknown = [item for item in unknown if item != "timezone"]
    if fatal_unknown or date_value is None or zone_token and zone_info is None:
        return _unknown_endpoint(
            "time", tuple(unknown) or ("date_or_time",), precision=precision
        )

    tzinfo, zone_value = zone_info if zone_info is not None else (None, None)
    microsecond = int((fraction + "000000")[:6]) if fraction else 0
    parsed = datetime(
        date_value.year,
        date_value.month,
        date_value.day,
        hour,
        minute,
        second,
        microsecond,
        tzinfo=tzinfo,
    )
    if second_text is not None:
        canonical = parsed.isoformat(timespec="microseconds" if fraction else "seconds")
    else:
        canonical = parsed.isoformat(timespec="minutes")
    if zone_value and canonical.endswith("+00:00"):
        canonical = canonical[:-6] + "+00:00"
    return _endpoint(
        "time",
        canonical,
        precision,
        "explicit" if zone_token else "unknown",
        unknown_components=("timezone",) if zone_token is None else (),
    )


def _parse_time(value: str) -> TemporalEndpoint | None:
    base, zone_token = _split_timezone(value)
    match = _TIME_24_RE.fullmatch(base) or _TIME_12_RE.fullmatch(base)
    if match is None:
        return None

    period = match.groupdict().get("period")
    hour = int(match.group("hour"))
    minute_text = match.groupdict().get("minute")
    second_text = match.groupdict().get("second")
    fraction = match.groupdict().get("fraction") or ""
    minute = int(minute_text or 0)
    second = int(second_text or 0)
    unknown: list[str] = []
    if period:
        hour %= 12
        if period.casefold().startswith("p"):
            hour += 12
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
        unknown.append("time")
    if len(fraction) > 6:
        unknown.append("second")

    zone_info = _parse_timezone(zone_token)
    if zone_token and zone_info is None:
        unknown.append("timezone")
    elif zone_token is None:
        unknown.append("timezone")

    if second_text is not None:
        precision: TemporalPrecision = "second"
    elif minute_text is not None:
        precision = "minute"
    else:
        precision = "hour"
    fatal_unknown = [item for item in unknown if item != "timezone"]
    if fatal_unknown or zone_token and zone_info is None:
        return _unknown_endpoint("time", tuple(unknown), precision=precision)

    _, zone_value = zone_info if zone_info is not None else (None, None)
    canonical = f"{hour:02d}"
    if minute_text is not None:
        canonical += f":{minute:02d}"
    if second_text is not None:
        canonical += f":{second:02d}"
        if fraction:
            canonical += f".{fraction.rstrip('0') or '0'}"
    if zone_value:
        canonical += zone_value
    return _endpoint(
        "time",
        canonical,
        precision,
        "explicit" if zone_token else "unknown",
        unknown_components=("timezone",) if zone_token is None else (),
    )


def _parse_duration(value: str) -> TemporalEndpoint:
    if value.upper().startswith("P"):
        match = _ISO_DURATION_RE.fullmatch(value.upper())
        if match is None:
            return _unknown_endpoint("duration", "duration")
        fields = {
            key: match.group(key)
            for key in (
                "years",
                "months",
                "weeks",
                "days",
                "hours",
                "minutes",
                "seconds",
            )
        }
        if not any(fields.values()):
            return _unknown_endpoint("duration", "duration")
        if fields["weeks"] and any(fields[key] for key in fields if key != "weeks"):
            return _conflicting_endpoint("duration", "mixed", "duration_components")
        canonical = _format_duration(fields)
        precision = _duration_precision(fields)
        return _endpoint("duration", canonical, precision, "not_applicable")

    prose = value
    if prose.casefold().startswith("for "):
        prose = prose[4:].strip()
    matches = list(_PROSE_DURATION_RE.finditer(prose))
    if not matches or _duration_gaps_are_nonempty(prose, matches):
        precision = _duration_hint_precision(value)
        return _unknown_endpoint("duration", "amount", precision=precision)

    fields: dict[str, str | None] = {
        "years": None,
        "months": None,
        "weeks": None,
        "days": None,
        "hours": None,
        "minutes": None,
        "seconds": None,
    }
    for match in matches:
        unit = match.group("unit").casefold()
        key = unit if unit.endswith("s") else f"{unit}s"
        if fields[key] is not None:
            return _conflicting_endpoint("duration", "mixed", "duration_components")
        fields[key] = match.group("amount")
    if fields["weeks"] and any(fields[key] for key in fields if key != "weeks"):
        return _conflicting_endpoint("duration", "mixed", "duration_components")
    return _endpoint(
        "duration",
        _format_duration(fields),
        _duration_precision(fields),
        "not_applicable",
    )


def _interval_parts(
    value: str,
) -> tuple[str | None, str | None, bool, bool, bool, bool] | None:
    lowered = value.casefold()
    if not value:
        return None

    from_marker = "from "
    if lowered.startswith(from_marker):
        remainder = value[len(from_marker) :].strip()
        nested = _interval_parts(remainder)
        if nested is not None:
            return nested
        if remainder:
            return remainder, None, False, True, True, False
        return None

    for prefix in ("since", "after", "as of"):
        marker = f"{prefix} "
        if lowered.startswith(marker):
            remainder = value[len(marker) :].strip()
            if remainder:
                return remainder, None, False, True, prefix != "after", False
            return None
    for prefix in ("until", "before"):
        marker = f"{prefix} "
        if lowered.startswith(marker):
            remainder = value[len(marker) :].strip()
            if remainder:
                return None, remainder, True, False, False, prefix != "before"
            return None

    between = lowered.startswith("between ")
    if between:
        value = value[8:].strip()

    onward = re.fullmatch(r"(.+?)\s+(?:onward|onwards|indefinitely)", value, re.I)
    if onward is not None:
        return onward.group(1).strip(), None, False, True, True, False

    if value.count("/") == 1:
        left, right = (part.strip() for part in value.split("/"))
        if left and right:
            return (
                None if _is_open_marker(left) else left,
                None if _is_open_marker(right) else right,
                not left or _is_open_marker(left),
                not right or _is_open_marker(right),
                True,
                True,
            )

    if ".." in value:
        left, right = (part.strip() for part in value.split("..", 1))
        if left or right:
            return (
                None if not left or _is_open_marker(left) else left,
                None if not right or _is_open_marker(right) else right,
                not left or _is_open_marker(left),
                not right or _is_open_marker(right),
                True,
                True,
            )

    delimiter = "to|through|until|and" if between else "to|through|until"
    word_range = re.fullmatch(
        rf"(?P<left>.+?)\s+(?:{delimiter})\s+(?P<right>.+)",
        value,
        re.IGNORECASE,
    )
    if word_range is not None:
        left = word_range.group("left").strip()
        right = word_range.group("right").strip()
        return (
            None if _is_open_marker(left) else left,
            None if _is_open_marker(right) else right,
            _is_open_marker(left),
            _is_open_marker(right),
            True,
            not re.search(r"\buntil\b", value, re.I),
        )

    dash_range = re.fullmatch(r"(?P<left>.+?)\s+(?:-|–|—)\s+(?P<right>.+)", value)
    if dash_range is not None:
        left = dash_range.group("left").strip()
        right = dash_range.group("right").strip()
        return (
            None if _is_open_marker(left) else left,
            None if _is_open_marker(right) else right,
            _is_open_marker(left),
            _is_open_marker(right),
            True,
            True,
        )
    return None


def _build_interval(
    left_value: str | None,
    right_value: str | None,
    source_start: int,
    source_end: int,
    *,
    open_start: bool,
    open_end: bool,
    start_inclusive: bool,
    end_inclusive: bool,
) -> TemporalInterval:
    if open_start:
        start_inclusive = False
    if open_end:
        end_inclusive = False
    left = _parse_endpoint(left_value) if left_value is not None else None
    right = _parse_endpoint(right_value) if right_value is not None else None
    endpoints = tuple(endpoint for endpoint in (left, right) if endpoint is not None)
    known_kinds = {
        endpoint.kind for endpoint in endpoints if endpoint.kind != "unknown"
    }
    conflicts: list[str] = []
    unknown: list[str] = []
    for endpoint in endpoints:
        unknown.extend(endpoint.unknown_components)
        conflicts.extend(endpoint.conflicts)
    if len(known_kinds) > 1:
        conflicts.append("endpoint_kind")
    if left is not None and right is not None and not conflicts:
        if _obvious_order_conflict(left, right):
            conflicts.append("interval_order")

    value: str | None = None
    if not conflicts and all(
        endpoint is None or endpoint.value is not None for endpoint in (left, right)
    ):
        left_text = left.value if left is not None else ".."
        right_text = right.value if right is not None else ".."
        value = f"{left_text}/{right_text}"

    precision = _interval_precision(left, right)
    timezone_state = _interval_timezone_state(left, right)
    if (
        timezone_state == "unknown"
        and endpoints
        and all(endpoint.timezone_state != "not_applicable" for endpoint in endpoints)
    ):
        unknown.append("timezone")
    if timezone_state == "conflicting":
        conflicts.append("timezone")

    return TemporalInterval(
        kind="interval",
        source_start=source_start,
        source_end=source_end,
        value=value,
        precision=precision,
        timezone_state=timezone_state,
        start=left,
        end=right,
        open_start=open_start,
        open_end=open_end,
        start_inclusive=start_inclusive,
        end_inclusive=end_inclusive,
        unknown_components=_unique(unknown),
        conflicts=_unique(conflicts),
    )


def _obvious_order_conflict(left: TemporalEndpoint, right: TemporalEndpoint) -> bool:
    if left.kind != right.kind or left.value is None or right.value is None:
        return False
    if left.kind == "date":
        return _date_lower_bound(left) > _date_upper_bound(right)
    if left.kind == "time" and left.precision == right.precision:
        if "T" not in left.value and "T" not in right.value:
            return left.value > right.value
        if left.timezone_state != right.timezone_state:
            return False
        try:
            return datetime.fromisoformat(left.value) > datetime.fromisoformat(
                right.value
            )
        except ValueError:
            return False
    return False


def _date_lower_bound(endpoint: TemporalEndpoint) -> date:
    assert endpoint.value is not None
    parts = [int(part) for part in endpoint.value.split("-")]
    if len(parts) == 1:
        return date(parts[0], 1, 1)
    if len(parts) == 2:
        return date(parts[0], parts[1], 1)
    return date(*parts)


def _date_upper_bound(endpoint: TemporalEndpoint) -> date:
    assert endpoint.value is not None
    parts = [int(part) for part in endpoint.value.split("-")]
    if len(parts) == 1:
        return date(parts[0], 12, 31)
    if len(parts) == 2:
        return date(parts[0], parts[1], calendar.monthrange(parts[0], parts[1])[1])
    return date(*parts)


def _interval_precision(
    left: TemporalEndpoint | None,
    right: TemporalEndpoint | None,
) -> TemporalPrecision:
    precisions = [
        endpoint.precision
        for endpoint in (left, right)
        if endpoint is not None and endpoint.precision != "unknown"
    ]
    if not precisions:
        return "unknown"
    return (
        precisions[0] if all(item == precisions[0] for item in precisions) else "mixed"
    )


def _interval_timezone_state(
    left: TemporalEndpoint | None,
    right: TemporalEndpoint | None,
) -> TimezoneState:
    states = [
        endpoint.timezone_state for endpoint in (left, right) if endpoint is not None
    ]
    if not states:
        return "unknown"
    if "conflicting" in states:
        return "conflicting"
    if all(state == "not_applicable" for state in states):
        return "not_applicable"
    if all(state == "explicit" for state in states):
        return "explicit"
    if all(state == states[0] for state in states):
        return states[0]
    return "unknown"


def _format_duration(fields: Mapping[str, str | None]) -> str:
    date_parts = []
    for key in ("years", "months", "weeks", "days"):
        amount = fields[key]
        if amount is not None:
            date_parts.append(
                f"{_canonical_number(amount)}{_DURATION_UNIT_MAP[key.rstrip('s')]}"
            )
    time_parts = []
    for key in ("hours", "minutes", "seconds"):
        amount = fields[key]
        if amount is not None:
            time_parts.append(
                f"{_canonical_number(amount)}{_DURATION_UNIT_MAP[key.rstrip('s')]}"
            )
    if time_parts:
        return f"P{''.join(date_parts)}T{''.join(time_parts)}"
    return f"P{''.join(date_parts)}"


def _duration_precision(fields: Mapping[str, str | None]) -> TemporalPrecision:
    for key, precision in (
        ("seconds", "second"),
        ("minutes", "minute"),
        ("hours", "hour"),
        ("days", "day"),
        ("weeks", "week"),
        ("months", "month"),
        ("years", "year"),
    ):
        if fields[key] is not None:
            return precision  # type: ignore[return-value]
    return "unknown"


def _duration_gaps_are_nonempty(
    value: str,
    matches: Sequence[re.Match[str]],
) -> bool:
    cursor = 0
    for match in matches:
        if not _is_duration_separator(value[cursor : match.start()]):
            return True
        cursor = match.end()
    return not _is_duration_separator(value[cursor:])


def _is_duration_separator(value: str) -> bool:
    return bool(re.fullmatch(r"[\s,]*(?:and[\s,]*)?", value, re.IGNORECASE))


def _duration_hint_precision(value: str) -> TemporalPrecision:
    match = re.search(
        r"(years?|months?|weeks?|days?|hours?|minutes?|seconds?)", value, re.I
    )
    if match is None:
        return "unknown"
    return _DURATION_PRECISION[match.group(1).casefold()]  # type: ignore[return-value]


def _canonical_number(value: str) -> str:
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return value
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def _looks_like_duration(value: str) -> bool:
    return bool(
        re.search(
            r"\b(?:for\s+)?(?:\d+(?:\.\d+)?|a\s+few|several)\s+"
            r"(?:years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b",
            value,
            re.I,
        )
    )


def _split_timezone(value: str) -> tuple[str, str | None]:
    match = _TIMEZONE_SUFFIX_RE.fullmatch(value)
    if match is None:
        return value.strip(), None
    return match.group("base").strip(), match.group("zone")


def _parse_timezone(
    token: str | None,
) -> tuple[timezone, str] | tuple[None, None] | None:
    if token is None:
        return (None, None)
    if token.casefold() in {"z", "utc"}:
        return timezone.utc, "+00:00"
    match = re.fullmatch(r"(?P<sign>[+-])(?P<hour>\d{2}):?(?P<minute>\d{2})", token)
    if match is None:
        return None
    hours = int(match.group("hour"))
    minutes = int(match.group("minute"))
    if hours > 23 or minutes > 59:
        return None
    delta = timedelta(hours=hours, minutes=minutes)
    if match.group("sign") == "-":
        delta = -delta
    canonical = f"{match.group('sign')}{hours:02d}:{minutes:02d}"
    return timezone(delta), canonical


def _endpoint(
    kind: TemporalKind,
    value: str,
    precision: str,
    timezone_state: TimezoneState,
    *,
    unknown_components: Sequence[str] = (),
    conflicts: Sequence[str] = (),
) -> TemporalEndpoint:
    return TemporalEndpoint(
        kind=kind,
        value=value,
        precision=precision,  # type: ignore[arg-type]
        timezone_state=timezone_state,
        unknown_components=tuple(unknown_components),
        conflicts=tuple(conflicts),
    )


def _unknown_endpoint(
    kind: TemporalKind,
    components: str | Sequence[str],
    *,
    precision: str = "unknown",
    timezone_state: TimezoneState | None = None,
) -> TemporalEndpoint:
    if isinstance(components, str):
        components = (components,)
    if timezone_state is None:
        timezone_state = "not_applicable" if kind in {"date", "duration"} else "unknown"
    return TemporalEndpoint(
        kind=kind,
        value=None,
        precision=precision,  # type: ignore[arg-type]
        timezone_state=timezone_state,
        unknown_components=tuple(components),
    )


def _conflicting_endpoint(
    kind: TemporalKind,
    precision: str,
    conflict: str,
) -> TemporalEndpoint:
    return TemporalEndpoint(
        kind=kind,
        value=None,
        precision=precision,  # type: ignore[arg-type]
        timezone_state="not_applicable" if kind in {"date", "duration"} else "unknown",
        conflicts=(conflict,),
    )


def _is_open_marker(value: str) -> bool:
    return value.casefold().strip() in _OPEN_MARKERS


def _clean(value: str) -> str:
    return " ".join(value.strip().split())


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _coerce_span(raw_span: SpanInput, *, text_length: int) -> tuple[int, int]:
    if isinstance(raw_span, Mapping):
        if "start" not in raw_span or "end" not in raw_span:
            raise ValueError("spans require start and end offsets")
        raw_start = raw_span["start"]
        raw_end = raw_span["end"]
    elif isinstance(raw_span, Sequence) and not isinstance(raw_span, (str, bytes)):
        if len(raw_span) != 2:
            raise ValueError("span sequences must contain two offsets")
        raw_start, raw_end = raw_span
    else:
        raise TypeError("spans must be mappings or two-item sequences")

    if (
        isinstance(raw_start, bool)
        or isinstance(raw_end, bool)
        or not isinstance(raw_start, int)
        or not isinstance(raw_end, int)
    ):
        raise ValueError("spans require integer offsets")
    if raw_start < 0 or raw_end <= raw_start or raw_end > text_length:
        raise ValueError("span offsets must satisfy source bounds")
    return raw_start, raw_end


__all__ = [
    "NormalizationStatus",
    "TemporalEndpoint",
    "TemporalInterval",
    "TemporalIntervalNormalizer",
    "TemporalKind",
    "TemporalPrecision",
    "TimezoneState",
    "normalize_interval",
    "normalize_temporal_interval",
    "normalize_temporal_intervals",
    "parse_temporal_interval",
    "parse_temporal_value",
]
