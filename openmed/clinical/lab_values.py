"""Deterministic laboratory reference-range helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Literal, TypedDict

AbnormalFlag = Literal["low", "normal", "high", "critical", "unknown"]


class ReferenceRange(TypedDict):
    """Parsed numeric reference-range bounds."""

    low: float | None
    high: float | None
    low_inclusive: bool
    high_inclusive: bool


LAB_FLAG_ADVISORY = (
    "Derived lab abnormal flags are heuristic and are not a substitute for the "
    "originating laboratory's own formal diagnostic flagging."
)

_EN_DASH = "\u2013"
_EM_DASH = "\u2014"
_LESS_THAN_OR_EQUAL = "\u2264"
_GREATER_THAN_OR_EQUAL = "\u2265"
_NUMERIC = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_RANGE_RE = re.compile(
    rf"^(?P<low>{_NUMERIC})\s*(?:-|to|{_EN_DASH}|{_EM_DASH})\s*"
    rf"(?P<high>{_NUMERIC})$",
    re.IGNORECASE,
)
_ONE_SIDED_RE = re.compile(
    rf"^(?P<operator><=|>=|<|>|{_LESS_THAN_OR_EQUAL}|"
    rf"{_GREATER_THAN_OR_EQUAL})\s*(?P<bound>{_NUMERIC})$"
)

_EXPLICIT_FLAGS: Mapping[str, AbnormalFlag] = {
    "H": "high",
    "HIGH": "high",
    "L": "low",
    "LOW": "low",
    "C": "critical",
    "CRIT": "critical",
    "CRITICAL": "critical",
    "CRITICAL HIGH": "critical",
    "CRITICAL LOW": "critical",
    "HH": "critical",
    "LL": "critical",
    "N": "normal",
    "NORMAL": "normal",
}


def _empty_reference_range() -> ReferenceRange:
    return {
        "low": None,
        "high": None,
        "low_inclusive": True,
        "high_inclusive": True,
    }


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_reference_range(text: object) -> ReferenceRange:
    """Parse a laboratory reference range into numeric bounds.

    Supported forms are closed ranges such as ``"135-145"``, ``"0.5 - 1.2"``,
    and ``"135 to 145"``, plus one-sided bounds such as ``"<5"``, ``"<=5"``,
    ``">10"``, and ``">=10"``. Units are intentionally ignored rather than
    parsed or converted.

    Args:
        text: Raw reference-range text.

    Returns:
        A mapping with ``low``, ``high``, ``low_inclusive``, and
        ``high_inclusive`` keys. Unparseable or contradictory ranges return
        empty bounds instead of guessing.
    """

    result = _empty_reference_range()
    if not isinstance(text, str):
        return result

    normalized = text.strip()
    if not normalized:
        return result

    if range_match := _RANGE_RE.fullmatch(normalized):
        low = _finite_float(range_match.group("low"))
        high = _finite_float(range_match.group("high"))
        if low is None or high is None or low > high:
            return result
        result["low"] = low
        result["high"] = high
        return result

    if one_sided_match := _ONE_SIDED_RE.fullmatch(normalized):
        bound = _finite_float(one_sided_match.group("bound"))
        if bound is None:
            return result

        operator = one_sided_match.group("operator")
        if operator in {"<", "<=", _LESS_THAN_OR_EQUAL}:
            result["high"] = bound
            result["high_inclusive"] = operator in {"<=", _LESS_THAN_OR_EQUAL}
        else:
            result["low"] = bound
            result["low_inclusive"] = operator in {">=", _GREATER_THAN_OR_EQUAL}
        return result

    return result


def _bool_or_default(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _normalize_reference_range(
    reference_range: Mapping[str, object] | str | None,
) -> ReferenceRange:
    if isinstance(reference_range, str):
        return parse_reference_range(reference_range)
    if not isinstance(reference_range, Mapping):
        return _empty_reference_range()

    raw_low = reference_range.get("low")
    raw_high = reference_range.get("high")
    low = _finite_float(raw_low)
    high = _finite_float(raw_high)

    if raw_low is not None and low is None:
        return _empty_reference_range()
    if raw_high is not None and high is None:
        return _empty_reference_range()
    if low is not None and high is not None and low > high:
        return _empty_reference_range()

    return {
        "low": low,
        "high": high,
        "low_inclusive": _bool_or_default(
            reference_range.get("low_inclusive"),
            True,
        ),
        "high_inclusive": _bool_or_default(
            reference_range.get("high_inclusive"),
            True,
        ),
    }


def derive_abnormal_flag(
    value: object,
    reference_range: Mapping[str, object] | str | None,
    explicit_flag: str | None = None,
) -> AbnormalFlag:
    """Derive a laboratory abnormal flag from a value and reference range.

    Explicit laboratory flags for high, low, normal, or critical values are
    honored before derived comparisons. Unknown explicit flags return
    ``"unknown"`` instead of being ignored. Critical thresholds beyond the
    reference range are out of scope unless the explicit flag marks the value
    critical. This helper is unit-agnostic and does not perform conversion.

    Args:
        value: Numeric lab result value.
        reference_range: Parsed range mapping or raw range text.
        explicit_flag: Optional originating-lab flag such as ``"H"``, ``"L"``,
            or ``"critical"``.

    Returns:
        ``"low"``, ``"normal"``, ``"high"``, ``"critical"``, or ``"unknown"``.
        Non-numeric values and unparseable ranges return ``"unknown"``.
    """

    if explicit_flag is not None:
        normalized_flag = explicit_flag.strip().upper()
        if normalized_flag:
            return _EXPLICIT_FLAGS.get(normalized_flag, "unknown")

    numeric_value = _finite_float(value)
    if numeric_value is None:
        return "unknown"

    parsed_range = _normalize_reference_range(reference_range)
    low = parsed_range["low"]
    high = parsed_range["high"]
    if low is None and high is None:
        return "unknown"

    if low is not None:
        if parsed_range["low_inclusive"] and numeric_value < low:
            return "low"
        if not parsed_range["low_inclusive"] and numeric_value <= low:
            return "low"

    if high is not None:
        if parsed_range["high_inclusive"] and numeric_value > high:
            return "high"
        if not parsed_range["high_inclusive"] and numeric_value >= high:
            return "high"

    return "normal"


__all__ = [
    "AbnormalFlag",
    "LAB_FLAG_ADVISORY",
    "ReferenceRange",
    "derive_abnormal_flag",
    "parse_reference_range",
]
