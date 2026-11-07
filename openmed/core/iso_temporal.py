"""Interpreter-independent extended-calendar ISO date and time parsing.

Only ASCII extended calendar dates and colon-separated times/offsets are
accepted. Parsing components directly avoids the changing ``fromisoformat``
grammar across supported Python releases. Caller contracts still own whether
a time or timezone is mandatory.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

__all__ = ["parse_iso_date", "parse_iso_datetime", "parse_iso_time"]

_DATE = r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
_TIME = (
    r"(?P<hour>[0-9]{2})"
    r"(?::(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2})(?:\.(?P<fraction>[0-9]{1,6}))?)?)?"
)
_OFFSET = r"(?P<offset>[Zz]|[+-][0-9]{2}:[0-9]{2})?"
_DATE_RE = re.compile(_DATE)
_DATETIME_RE = re.compile(_DATE + r"(?:[Tt ]" + _TIME + _OFFSET + r")?")
_TIME_RE = re.compile(_TIME + _OFFSET)


def _match(pattern: re.Pattern[str], value: Any) -> re.Match[str]:
    if type(value) is not str or len(value) > 40:
        raise ValueError("invalid_iso_temporal")
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError("invalid_iso_temporal")
    return match


def _clock(match: re.Match[str]) -> tuple[int, int, int, int, timezone | None]:
    hour = int(match.group("hour") or 0)
    minute = int(match.group("minute") or 0)
    second = int(match.group("second") or 0)
    fraction = match.group("fraction") or ""
    microsecond = int(fraction.ljust(6, "0"))
    offset = match.group("offset")
    zone = None
    if offset in ("Z", "z"):
        zone = timezone.utc
    elif offset is not None:
        hours, minutes = int(offset[1:3]), int(offset[4:6])
        if hours > 23 or minutes > 59:
            raise ValueError("invalid_iso_temporal")
        delta = timedelta(hours=hours, minutes=minutes)
        zone = timezone(delta if offset[0] == "+" else -delta)
    return hour, minute, second, microsecond, zone


def parse_iso_date(value: str) -> date:
    """Parse an extended calendar date with a stable, value-free failure.

    Args:
        value: An ASCII ``YYYY-MM-DD`` calendar date.

    Returns:
        The validated calendar date.

    Raises:
        ValueError: ``invalid_iso_temporal`` for a malformed or impossible date.
    """
    match = _match(_DATE_RE, value)
    result = None
    try:
        result = date(*(int(match.group(name)) for name in ("year", "month", "day")))
    except ValueError:
        pass
    if result is None:
        raise ValueError("invalid_iso_temporal")
    return result


def parse_iso_datetime(value: str) -> datetime:
    """Parse an extended calendar date and optional colon-separated time.

    Args:
        value: ``YYYY-MM-DD`` optionally followed by ``T``, ``t`` or a space,
            then ``HH[:MM[:SS[.ffffff]]]`` and an optional ``Z``/``z`` or
            ``+HH:MM``/``-HH:MM`` offset. Fractions contain one to six digits.

    Returns:
        A datetime; a date-only input has midnight and no timezone. Caller
        contracts retain their own requirements for time and timezone.

    Raises:
        ValueError: ``invalid_iso_temporal`` for an unsupported or invalid value.
    """
    match = _match(_DATETIME_RE, value)
    result = None
    try:
        hour, minute, second, microsecond, zone = _clock(match)
        result = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            hour,
            minute,
            second,
            microsecond,
            tzinfo=zone,
        )
    except ValueError:
        pass
    if result is None:
        raise ValueError("invalid_iso_temporal")
    return result


def parse_iso_time(value: str) -> time:
    """Parse a colon-separated clock time using the same explicit profile.

    Args:
        value: ``HH[:MM[:SS[.ffffff]]]`` with an optional ``Z``/``z`` or
            ``+HH:MM``/``-HH:MM`` offset. Fractions contain one to six digits.

    Returns:
        The clock time, retaining any supplied offset.

    Raises:
        ValueError: ``invalid_iso_temporal`` for an unsupported or invalid value.
    """
    match = _match(_TIME_RE, value)
    result = None
    try:
        result = time(*_clock(match))
    except ValueError:
        pass
    if result is None:
        raise ValueError("invalid_iso_temporal")
    return result
