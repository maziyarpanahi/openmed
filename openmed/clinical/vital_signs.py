"""Deterministic vital-sign structuring helpers."""

from __future__ import annotations

import math
import re
from typing import Literal, TypedDict

from .lexicons.clinical_norm import (
    abbreviation_surfaces,
    canonical_unit_alias,
    get_clinical_norm_lexicon,
    parse_locale_number,
)
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
_LOCALIZED_NUMERIC = (
    r"(?:"
    r"(?:\d{1,3}(?:[ \u00a0\u202f'’.,]\d{3})+|\d+)(?:[.,\u066b]\d+)?"
    r"|[.,\u066b]\d+)"
)
_SEPARATOR = r"\s*(?:(?:is|was|=|:)\s*)?"
_TRAILING_BOUNDARY = r"(?=$|\s|[.,;)])"
_LOCALIZED_SEPARATOR = r"\s*(?:(?:is|was|=|:|：)\s*)?"

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
_LOCALIZED_BLOOD_PRESSURE_VALUE_RE = re.compile(
    rf"^(?P<systolic>{_LOCALIZED_NUMERIC})\s*/\s*"
    rf"(?P<diastolic>{_LOCALIZED_NUMERIC})(?:\s*(?P<unit>\S.*?))?\s*$",
    re.IGNORECASE,
)
_LOCALIZED_SINGLE_VALUE_RE = re.compile(
    rf"^(?P<value>{_LOCALIZED_NUMERIC})(?:\s*(?P<unit>\S.*?))?\s*$",
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


def structure_vital_sign(
    text: object,
    *,
    language: object | None = None,
) -> VitalSignResult:
    """Structure a short narrative vital-sign phrase.

    The parser recognizes blood pressure, heart rate, body temperature,
    respiratory rate, and oxygen saturation. Units are captured exactly as
    written and are never converted or normalized. Unknown or unparseable input
    returns an explicit ``"unknown"`` result instead of raising.

    Args:
        text: Short vital-sign phrase or already-located narrative span.
        language: Optional source-language code for localized vital
            abbreviations, decimal punctuation, and unit aliases.

    Returns:
        A structured vital-sign mapping with a kind, value, captured unit, and
        blood-pressure components when applicable.
    """

    if not isinstance(text, str):
        return empty_vital_sign_result()

    phrase = text.strip()
    if not phrase:
        return empty_vital_sign_result()

    if language is not None:
        localized = _localized_vital_sign(phrase, language=language)
        if localized["kind"] != "unknown":
            return localized
        if _starts_with_localized_abbreviation(phrase, language=language):
            return localized

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
    *,
    language: object | None = None,
    target_language: object | None = None,
) -> MeasurementNormalization:
    """Normalize a captured vital-sign value with explicit unit provenance."""

    return normalize_to(
        value,
        unit,
        target_unit,
        from_language=language,
        to_language=target_language,
    )


def _localized_vital_sign(
    phrase: str,
    *,
    language: object,
) -> VitalSignResult:
    for kind in (
        "blood_pressure",
        "oxygen_saturation",
        "body_temperature",
        "respiratory_rate",
        "heart_rate",
    ):
        if (tail := _strip_localized_prefix(phrase, kind, language=language)) is None:
            continue
        if kind == "blood_pressure":
            return _localized_blood_pressure_result(tail, language=language)
        return _localized_single_value_result(kind, tail, language=language)
    return empty_vital_sign_result()


def _strip_localized_prefix(
    phrase: str,
    kind: str,
    *,
    language: object,
) -> str | None:
    lexicon = get_clinical_norm_lexicon(language)
    for surface in sorted(
        abbreviation_surfaces(kind, language=language),
        key=len,
        reverse=True,
    ):
        boundary = "" if not lexicon.token_boundaries else r"(?!\w)"
        pattern = re.compile(
            rf"^\s*{re.escape(surface)}\.?{boundary}{_LOCALIZED_SEPARATOR}",
            re.IGNORECASE,
        )
        if match := pattern.match(phrase):
            return phrase[match.end() :].strip()
    return None


def _starts_with_localized_abbreviation(
    phrase: str,
    *,
    language: object,
) -> bool:
    for kind in (
        "blood_pressure",
        "oxygen_saturation",
        "body_temperature",
        "respiratory_rate",
        "heart_rate",
    ):
        if _strip_localized_prefix(phrase, kind, language=language) is not None:
            return True
    return False


def _localized_blood_pressure_result(
    tail: str,
    *,
    language: object,
) -> VitalSignResult:
    match = _LOCALIZED_BLOOD_PRESSURE_VALUE_RE.fullmatch(tail)
    if match is None:
        return empty_vital_sign_result()
    systolic = _parse_localized_number(match.group("systolic"), language=language)
    diastolic = _parse_localized_number(match.group("diastolic"), language=language)
    if systolic is None or diastolic is None:
        return empty_vital_sign_result()

    unit = _localized_unit(match.groupdict().get("unit"), language=language)
    return {
        "kind": "blood_pressure",
        "value": None,
        "unit": unit,
        "components": [
            {"kind": "systolic", "value": systolic, "unit": unit},
            {"kind": "diastolic", "value": diastolic, "unit": unit},
        ],
    }


def _localized_single_value_result(
    kind: str,
    tail: str,
    *,
    language: object,
) -> VitalSignResult:
    match = _LOCALIZED_SINGLE_VALUE_RE.fullmatch(tail)
    if match is None:
        return empty_vital_sign_result()
    value = _parse_localized_number(match.group("value"), language=language)
    if value is None:
        return empty_vital_sign_result()
    return {
        "kind": kind,  # type: ignore[typeddict-item]
        "value": value,
        "unit": _localized_unit(match.groupdict().get("unit"), language=language),
        "components": [],
    }


def _parse_localized_number(
    text: str,
    *,
    language: object,
) -> VitalSignNumber | None:
    value = parse_locale_number(text, language=language)
    if value is None:
        return None
    if value.is_integer():
        return int(value)
    return value


def _localized_unit(
    unit: object,
    *,
    language: object,
) -> str:
    if not isinstance(unit, str):
        return ""
    cleaned = unit.strip()
    if not cleaned:
        return ""
    return canonical_unit_alias(cleaned, language=language) or cleaned


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
