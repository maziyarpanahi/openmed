"""Utilities for parsing and interpreting clinical lab value ranges."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

ReferenceRange = dict[str, float | bool | None]
AbnormalFlag = Literal["low", "normal", "high", "critical", "unknown"]

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUMBER_RE = re.compile(_NUMBER)
_RANGE_RE = re.compile(
    rf"^\s*(?P<low>{_NUMBER})\s*(?:-|–|—|to)\s*(?P<high>{_NUMBER})\s*$",
    re.IGNORECASE,
)
_BOUND_RE = re.compile(
    rf"^\s*(?P<op><=|>=|<|>|≤|≥)\s*(?P<value>{_NUMBER})\s*$",
    re.IGNORECASE,
)

_EXPLICIT_FLAG_ALIASES: dict[str, AbnormalFlag] = {
    "critical": "critical",
    "l": "low",
    "h": "high",
}


def parse_reference_range(text: str | None) -> ReferenceRange:
    """Parse a lab reference range string into comparable numeric bounds.

    Supported forms include closed ranges such as ``"135-145"`` and
    one-sided bounds such as ``"<5"``, ``"<=5"``, ``">10"``, and ``">=10"``.
    Unknown or unsupported input returns an unbounded range with ``None`` for
    both bounds and inclusivity fields.
    """
    result: ReferenceRange = {
        "low": None,
        "high": None,
        "low_inclusive": None,
        "high_inclusive": None,
    }
    if text is None:
        return result

    normalized = str(text).strip()
    if not normalized:
        return result

    range_match = _RANGE_RE.match(normalized)
    if range_match:
        low = float(range_match.group("low"))
        high = float(range_match.group("high"))
        if low <= high:
            result.update(
                {
                    "low": low,
                    "high": high,
                    "low_inclusive": True,
                    "high_inclusive": True,
                }
            )
        return result

    bound_match = _BOUND_RE.match(normalized)
    if not bound_match:
        return result

    operator = bound_match.group("op")
    value = float(bound_match.group("value"))
    if operator in {"<", "<=", "≤"}:
        result["high"] = value
        result["high_inclusive"] = operator in {"<=", "≤"}
    elif operator in {">", ">=", "≥"}:
        result["low"] = value
        result["low_inclusive"] = operator in {">=", "≥"}
    return result


def derive_abnormal_flag(
    value: float | int | str | None,
    reference_range: Mapping[str, Any] | None,
    explicit_flag: str | None = None,
) -> AbnormalFlag:
    """Derive an abnormality flag from a value and parsed reference range."""
    normalized_flag = _normalize_explicit_flag(explicit_flag)
    if normalized_flag is not None:
        return normalized_flag

    numeric_value = _coerce_float(value)
    if numeric_value is None or not isinstance(reference_range, Mapping):
        return "unknown"

    low = _coerce_float(reference_range.get("low"))
    high = _coerce_float(reference_range.get("high"))
    low_inclusive = reference_range.get("low_inclusive")
    high_inclusive = reference_range.get("high_inclusive")

    if low is not None and _below_low(numeric_value, low, low_inclusive):
        return "low"
    if high is not None and _above_high(numeric_value, high, high_inclusive):
        return "high"
    if low is not None or high is not None:
        return "normal"
    return "unknown"


def _normalize_explicit_flag(explicit_flag: str | None) -> AbnormalFlag | None:
    if explicit_flag is None:
        return None
    normalized = str(explicit_flag).strip().lower()
    if not normalized:
        return None
    return _EXPLICIT_FLAG_ALIASES.get(normalized, "unknown")


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    match = _NUMBER_RE.search(str(value).strip())
    if match is None:
        return None
    return float(match.group(0))


def _below_low(value: float, low: float, low_inclusive: Any) -> bool:
    if low_inclusive is False:
        return value <= low
    return value < low


def _above_high(value: float, high: float, high_inclusive: Any) -> bool:
    if high_inclusive is False:
        return value >= high
    return value > high


__all__ = [
    "AbnormalFlag",
    "ReferenceRange",
    "derive_abnormal_flag",
    "parse_reference_range",
]
