"""Keep-alive duration parsing for the REST service."""

from __future__ import annotations

import re
from typing import Any, Optional


_DURATION_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h|d)")
_UNIT_SECONDS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def parse_keep_alive(value: Any) -> Optional[float]:
    """Parse a keep-alive value into seconds.

    Accepted values are numbers in seconds, strings like ``"30s"``, ``"5m"``,
    ``"1h30m"``, and opt-out strings such as ``"off"`` or ``"forever"``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("keep_alive must be a duration, not a boolean")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError("keep_alive must be greater than or equal to 0")
        return seconds
    if not isinstance(value, str):
        raise ValueError("keep_alive must be a duration string or number of seconds")

    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"off", "none", "never", "forever", "infinite", "infinity"}:
        return None

    try:
        seconds = float(normalized)
    except ValueError:
        seconds = None

    if seconds is not None:
        if seconds < 0:
            raise ValueError("keep_alive must be greater than or equal to 0")
        return seconds

    total = 0.0
    position = 0
    for match in _DURATION_PATTERN.finditer(normalized):
        if match.start() != position:
            raise ValueError("keep_alive must use duration units like '30s', '5m', or '1h30m'")
        total += float(match.group("value")) * _UNIT_SECONDS[match.group("unit")]
        position = match.end()

    if position != len(normalized) or position == 0:
        raise ValueError("keep_alive must use duration units like '30s', '5m', or '1h30m'")
    return total
