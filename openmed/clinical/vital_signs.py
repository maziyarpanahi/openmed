"""Deterministic vital-sign structuring helpers."""

from __future__ import annotations

import math
import re
from typing import Literal, TypedDict

from .units import MeasurementNormalization, normalize_to

VitalKind = Literal[
    "blood_pressure",
    "heart_rate",
    "body_temperature",
    "respiratory_rate",
    "oxygen_saturation",
    "unknown",
]
VitalSignComponentKind = Literal["systolic", "diastolic"]
VitalSignNumber = int | float


class VitalSignComponent(TypedDict):
    """A blood-pressure component captured from narrative text."""

    kind: VitalSignComponentKind
    value: VitalSignNumber
    unit: str


class VitalSignResult(TypedDict):
    """Structured vital-sign value captured from narrative text."""

    kind: VitalKind
    value: VitalSignNumber | None
    unit: str
    components: list[VitalSignComponent]


VITAL_SIGNS_ADVISORY = (
    "Vital-sign outputs are heuristic structuring of narrative text and are "
    "not a clinical measurement device or a substitute for the originating "
    "device's recorded values."
)

_NUMERIC = r"(?:\d+(?:\.\d*)?|\.\d+)"
_SEPARATOR = r"\s*(?:(?:is|was|=|:)\s*)?"
_TRAILING_BOUNDARY = r"(?=$|\s|[.,;)])"

_BLOOD_PRESSURE_RE = re.compile(
    rf"(?:\b(?:bp|b/p|blood pressure)\b{_SEPARATOR})?"
    rf"(?P<systolic>\d{{2,3}})\s*/\s*(?P<diastolic>\d{{2,3}})"
    rf"(?:\s*(?P<unit>mm\s*hg))?{_TRAILING_BOUNDARY}",
    re.IGNORECASE,
)
_OXYGEN_SATURATION_RE = re.compile(
    rf"\b(?:spo2|sp\s*o2|o2 sat(?:uration)?|oxygen saturation|sat)\b"
    rf"\.?{_SEPARATOR}(?P<value>{_NUMERIC})(?:\s*(?P<unit>%))?"
    rf"{_TRAILING_BOUNDARY}",
    re.IGNORECASE,
)
_TEMPERATURE_RE = re.compile(
    rf"\b(?:temp(?:erature)?|t)\b\.?{_SEPARATOR}(?P<value>{_NUMERIC})"
    rf"(?:\s*(?P<unit>c|f|celsius|fahrenheit))?{_TRAILING_BOUNDARY}",
    re.IGNORECASE,
)
_RESPIRATORY_RATE_RE = re.compile(
    rf"\b(?:rr|resp(?:iratory)? rate|respirations?|resp)\b\.?"
    rf"{_SEPARATOR}(?P<value>{_NUMERIC})"
    rf"(?:\s*(?P<unit>/min|breaths?\s*/\s*min(?:ute)?|"
    rf"breaths?\s+per\s+min(?:ute)?|resp(?:irations?)?\s*/\s*min(?:ute)?))?"
    rf"{_TRAILING_BOUNDARY}",
    re.IGNORECASE,
)
_HEART_RATE_RE = re.compile(
    rf"\b(?:hr|heart rate|pulse)\b\.?{_SEPARATOR}(?P<value>{_NUMERIC})"
    rf"(?:\s*(?P<unit>bpm|beats?\s*/\s*min(?:ute)?|"
    rf"beats?\s+per\s+min(?:ute)?|/min))?{_TRAILING_BOUNDARY}",
    re.IGNORECASE,
)
_BPM_ONLY_RE = re.compile(
    rf"(?<!\w)(?P<value>{_NUMERIC})\s*(?P<unit>bpm){_TRAILING_BOUNDARY}",
    re.IGNORECASE,
)


def empty_vital_sign_result() -> VitalSignResult:
    """Return an explicit empty result for unknown or unparseable text."""

    return {
        "kind": "unknown",
        "value": None,
        "unit": "",
        "components": [],
    }


def _parse_number(text: str) -> VitalSignNumber | None:
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if "." not in text:
        return int(text)
    return value


def _unit(match: re.Match[str]) -> str:
    return match.groupdict().get("unit") or ""


def _single_value_result(kind: VitalKind, match: re.Match[str]) -> VitalSignResult:
    value = _parse_number(match.group("value"))
    if value is None:
        return empty_vital_sign_result()
    return {
        "kind": kind,
        "value": value,
        "unit": _unit(match),
        "components": [],
    }


def _blood_pressure_result(match: re.Match[str]) -> VitalSignResult:
    systolic = _parse_number(match.group("systolic"))
    diastolic = _parse_number(match.group("diastolic"))
    if systolic is None or diastolic is None:
        return empty_vital_sign_result()

    unit = _unit(match)
    return {
        "kind": "blood_pressure",
        "value": None,
        "unit": unit,
        "components": [
            {"kind": "systolic", "value": systolic, "unit": unit},
            {"kind": "diastolic", "value": diastolic, "unit": unit},
        ],
    }


def structure_vital_sign(text: object) -> VitalSignResult:
    """Structure a short narrative vital-sign phrase.

    The parser recognizes blood pressure, heart rate, body temperature,
    respiratory rate, and oxygen saturation. Units are captured exactly as
    written and are never converted or normalized. Unknown or unparseable input
    returns an explicit ``"unknown"`` result instead of raising.

    Args:
        text: Short vital-sign phrase or already-located narrative span.

    Returns:
        A structured vital-sign mapping with a kind, value, captured unit, and
        blood-pressure components when applicable.
    """

    if not isinstance(text, str):
        return empty_vital_sign_result()

    phrase = text.strip()
    if not phrase:
        return empty_vital_sign_result()

    if match := _BLOOD_PRESSURE_RE.search(phrase):
        return _blood_pressure_result(match)
    if match := _OXYGEN_SATURATION_RE.search(phrase):
        return _single_value_result("oxygen_saturation", match)
    if match := _TEMPERATURE_RE.search(phrase):
        return _single_value_result("body_temperature", match)
    if match := _RESPIRATORY_RATE_RE.search(phrase):
        return _single_value_result("respiratory_rate", match)
    if match := _HEART_RATE_RE.search(phrase):
        return _single_value_result("heart_rate", match)
    if match := _BPM_ONLY_RE.search(phrase):
        return _single_value_result("heart_rate", match)

    return empty_vital_sign_result()


def normalize_vital_measurement(
    value: object,
    unit: object,
    target_unit: object,
) -> MeasurementNormalization:
    """Normalize a captured vital-sign value with explicit unit provenance."""

    return normalize_to(value, unit, target_unit)


__all__ = [
    "VITAL_SIGNS_ADVISORY",
    "VitalKind",
    "VitalSignComponent",
    "VitalSignComponentKind",
    "VitalSignNumber",
    "VitalSignResult",
    "empty_vital_sign_result",
    "normalize_vital_measurement",
    "structure_vital_sign",
]
