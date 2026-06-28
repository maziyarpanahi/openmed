"""Deterministic medication sig frequency and duration normalization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal, TypedDict

FrequencyPeriodUnit = Literal["h", "d", "wk"]
DurationUnit = Literal["d", "wk"]
Number = int | float


class FrequencyNormalization(TypedDict):
    """Structured medication frequency normalization result."""

    raw: object
    recognized: bool
    confidence: float
    period: Number | None
    period_unit: FrequencyPeriodUnit | None
    frequency_per_day: float | None
    as_needed: bool
    cue: str | None


class DurationNormalization(TypedDict):
    """Structured medication duration normalization result."""

    raw: object
    recognized: bool
    confidence: float
    value: Number | None
    unit: DurationUnit | None
    days: Number | None
    cue: str | None


class _FrequencyCue(TypedDict, total=False):
    frequency_per_day: float
    period: Number
    period_unit: FrequencyPeriodUnit
    confidence: float


MEDICATION_SIG_ADVISORY = (
    "Medication sig normalization is deterministic support tooling and is not "
    "a substitute for clinician review; PRN/as-needed cues are flagged without "
    "implying a scheduled numeric frequency."
)

# Provenance: these are common Latin prescription sig abbreviations and plain
# English equivalents used in medication instructions. The table is restricted
# to deterministic one-to-one frequency cues and intentionally excludes dose,
# route, and drug-name interpretation.
_FREQUENCY_CUES: Mapping[str, _FrequencyCue] = {
    "qd": {"frequency_per_day": 1.0, "period": 1, "period_unit": "d"},
    "daily": {"frequency_per_day": 1.0, "period": 1, "period_unit": "d"},
    "once daily": {"frequency_per_day": 1.0, "period": 1, "period_unit": "d"},
    "once a day": {"frequency_per_day": 1.0, "period": 1, "period_unit": "d"},
    "every day": {"frequency_per_day": 1.0, "period": 1, "period_unit": "d"},
    "bid": {"frequency_per_day": 2.0},
    "twice daily": {"frequency_per_day": 2.0},
    "twice a day": {"frequency_per_day": 2.0},
    "two times daily": {"frequency_per_day": 2.0},
    "tid": {"frequency_per_day": 3.0},
    "three times daily": {"frequency_per_day": 3.0},
    "three times a day": {"frequency_per_day": 3.0},
    "qid": {"frequency_per_day": 4.0},
    "four times daily": {"frequency_per_day": 4.0},
    "four times a day": {"frequency_per_day": 4.0},
    "qhs": {
        "frequency_per_day": 1.0,
        "period": 1,
        "period_unit": "d",
        "confidence": 0.9,
    },
    "at bedtime": {
        "frequency_per_day": 1.0,
        "period": 1,
        "period_unit": "d",
        "confidence": 0.9,
    },
    "bedtime": {
        "frequency_per_day": 1.0,
        "period": 1,
        "period_unit": "d",
        "confidence": 0.9,
    },
    "nightly": {
        "frequency_per_day": 1.0,
        "period": 1,
        "period_unit": "d",
        "confidence": 0.9,
    },
    "weekly": {
        "frequency_per_day": 1.0 / 7.0,
        "period": 1,
        "period_unit": "wk",
    },
    "once weekly": {
        "frequency_per_day": 1.0 / 7.0,
        "period": 1,
        "period_unit": "wk",
    },
    "once a week": {
        "frequency_per_day": 1.0 / 7.0,
        "period": 1,
        "period_unit": "wk",
    },
    "every week": {
        "frequency_per_day": 1.0 / 7.0,
        "period": 1,
        "period_unit": "wk",
    },
    "qwk": {
        "frequency_per_day": 1.0 / 7.0,
        "period": 1,
        "period_unit": "wk",
    },
    "qweek": {
        "frequency_per_day": 1.0 / 7.0,
        "period": 1,
        "period_unit": "wk",
    },
}

# Provenance: duration units mirror common written sig durations. The slash
# form below covers compact duration notation such as 10/7, where /7 denotes
# days and /52 denotes weeks.
_DURATION_UNITS: Mapping[str, tuple[DurationUnit, int]] = {
    "d": ("d", 1),
    "day": ("d", 1),
    "days": ("d", 1),
    "w": ("wk", 7),
    "wk": ("wk", 7),
    "wks": ("wk", 7),
    "week": ("wk", 7),
    "weeks": ("wk", 7),
}
_SLASH_DURATION_UNITS: Mapping[str, tuple[DurationUnit, int]] = {
    "7": ("d", 1),
    "52": ("wk", 7),
}

_NUMERIC = r"(?:\d+(?:\.\d*)?|\.\d+)"
_INTERVAL_RE = re.compile(
    rf"^(?:q|every)\s*(?P<period>{_NUMERIC})\s*"
    r"(?P<unit>hours?|hrs?|hr|h|days?|d|weeks?|wks?|wk|w)$"
)
_DURATION_RE = re.compile(
    rf"^(?:for\s+|x\s*)?(?P<value>{_NUMERIC})\s*"
    r"(?P<unit>days?|d|weeks?|wks?|wk|w)$"
)
_SLASH_DURATION_RE = re.compile(
    rf"^(?:for\s+|x\s*)?(?P<value>{_NUMERIC})\s*/\s*(?P<unit>7|52)$"
)
_AS_NEEDED_RE = re.compile(
    r"(?<![a-z0-9])(?:p\s*r\s*n|prn|as needed|when needed|if needed)"
    r"(?![a-z0-9])"
)
_SIG_PUNCTUATION_RE = re.compile(r"[.,;:()\[\]{}_-]+")
_DURATION_PUNCTUATION_RE = re.compile(r"[,;:()\[\]{}_-]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_frequency(text: object) -> FrequencyNormalization:
    """Normalize a medication sig frequency cue.

    Supported forms include common Latin abbreviations and English
    equivalents such as ``"qd"``, ``"daily"``, ``"BID"``, ``"q8h"``,
    ``"qhs"``, ``"weekly"``, and PRN/as-needed cues. PRN is represented by
    ``as_needed=True`` and is not converted to a scheduled numeric frequency
    unless a separate scheduled cue is also present, for example ``"BID PRN"``.

    Args:
        text: Raw medication sig frequency text.

    Returns:
        A structured mapping containing the raw input, recognition status,
        confidence, interval fields, times-per-day when deterministic, and the
        PRN/as-needed flag. Unrecognized input preserves ``raw`` and returns
        ``recognized=False``.
    """

    if not isinstance(text, str):
        return _empty_frequency(text)

    normalized = _phrase_text(text)
    if not normalized:
        return _empty_frequency(text)

    as_needed, scheduled_text = _split_as_needed(normalized)
    if not scheduled_text:
        if as_needed:
            return _frequency_result(
                text,
                recognized=True,
                confidence=1.0,
                as_needed=True,
                cue="prn",
            )
        return _empty_frequency(text)

    cue = _find_frequency_cue(scheduled_text)
    if cue is not None:
        cue_text, cue_values = cue
        return _frequency_result(
            text,
            recognized=True,
            confidence=cue_values.get("confidence", 1.0),
            period=cue_values.get("period"),
            period_unit=cue_values.get("period_unit"),
            frequency_per_day=cue_values.get("frequency_per_day"),
            as_needed=as_needed,
            cue=cue_text,
        )

    if interval := _INTERVAL_RE.fullmatch(scheduled_text):
        period = _positive_number(interval.group("period"))
        if period is not None:
            unit = _canonical_frequency_period_unit(interval.group("unit"))
            return _frequency_result(
                text,
                recognized=True,
                confidence=1.0,
                period=period,
                period_unit=unit,
                frequency_per_day=_frequency_per_day(period, unit),
                as_needed=as_needed,
                cue=scheduled_text,
            )

    if as_needed:
        return _frequency_result(
            text,
            recognized=True,
            confidence=1.0,
            as_needed=True,
            cue="prn",
        )

    return _empty_frequency(text)


def normalize_duration(text: object) -> DurationNormalization:
    """Normalize a medication sig duration cue.

    Supported deterministic forms include ``"x 7 days"``, ``"x7d"``,
    ``"for 2 weeks"``, and compact slash notation such as ``"10/7"``.
    Month-based durations are intentionally not converted because their day
    length is context-dependent.

    Args:
        text: Raw medication sig duration text.

    Returns:
        A structured mapping containing the raw input, recognition status,
        confidence, canonical value and unit, and duration in days. Unrecognized
        input preserves ``raw`` and returns ``recognized=False``.
    """

    if not isinstance(text, str):
        return _empty_duration(text)

    normalized = _duration_text(text)
    if not normalized:
        return _empty_duration(text)

    if slash_match := _SLASH_DURATION_RE.fullmatch(normalized):
        value = _positive_number(slash_match.group("value"))
        if value is None:
            return _empty_duration(text)
        unit, multiplier = _SLASH_DURATION_UNITS[slash_match.group("unit")]
        return _duration_result(
            text,
            recognized=True,
            confidence=1.0,
            value=value,
            unit=unit,
            days=_scaled_number(value, multiplier),
            cue=normalized,
        )

    if duration_match := _DURATION_RE.fullmatch(normalized):
        value = _positive_number(duration_match.group("value"))
        if value is None:
            return _empty_duration(text)
        unit, multiplier = _DURATION_UNITS[duration_match.group("unit")]
        return _duration_result(
            text,
            recognized=True,
            confidence=1.0,
            value=value,
            unit=unit,
            days=_scaled_number(value, multiplier),
            cue=normalized,
        )

    return _empty_duration(text)


def _empty_frequency(raw: object) -> FrequencyNormalization:
    return _frequency_result(raw, recognized=False, confidence=0.0)


def _frequency_result(
    raw: object,
    *,
    recognized: bool,
    confidence: float,
    period: Number | None = None,
    period_unit: FrequencyPeriodUnit | None = None,
    frequency_per_day: float | None = None,
    as_needed: bool = False,
    cue: str | None = None,
) -> FrequencyNormalization:
    return {
        "raw": raw,
        "recognized": recognized,
        "confidence": confidence,
        "period": period,
        "period_unit": period_unit,
        "frequency_per_day": frequency_per_day,
        "as_needed": as_needed,
        "cue": cue,
    }


def _empty_duration(raw: object) -> DurationNormalization:
    return _duration_result(raw, recognized=False, confidence=0.0)


def _duration_result(
    raw: object,
    *,
    recognized: bool,
    confidence: float,
    value: Number | None = None,
    unit: DurationUnit | None = None,
    days: Number | None = None,
    cue: str | None = None,
) -> DurationNormalization:
    return {
        "raw": raw,
        "recognized": recognized,
        "confidence": confidence,
        "value": value,
        "unit": unit,
        "days": days,
        "cue": cue,
    }


def _phrase_text(text: str) -> str:
    normalized = _SIG_PUNCTUATION_RE.sub(" ", text.casefold())
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _duration_text(text: str) -> str:
    normalized = _DURATION_PUNCTUATION_RE.sub(" ", text.casefold())
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _split_as_needed(text: str) -> tuple[bool, str]:
    as_needed = _AS_NEEDED_RE.search(text) is not None
    scheduled_text = _AS_NEEDED_RE.sub(" ", text)
    return as_needed, _WHITESPACE_RE.sub(" ", scheduled_text).strip()


def _find_frequency_cue(text: str) -> tuple[str, _FrequencyCue] | None:
    if text in _FREQUENCY_CUES:
        return text, _FREQUENCY_CUES[text]

    compact = _compact_text(text)
    if compact in _FREQUENCY_CUES:
        return compact, _FREQUENCY_CUES[compact]

    return None


def _positive_number(text: str) -> Number | None:
    value = float(text)
    if value <= 0:
        return None
    return _clean_number(value)


def _clean_number(value: float) -> Number:
    if value.is_integer():
        return int(value)
    return value


def _scaled_number(value: Number, multiplier: int) -> Number:
    return _clean_number(float(value) * multiplier)


def _canonical_frequency_period_unit(unit: str) -> FrequencyPeriodUnit:
    normalized = unit.casefold()
    if normalized in {"h", "hr", "hrs", "hour", "hours"}:
        return "h"
    if normalized in {"d", "day", "days"}:
        return "d"
    return "wk"


def _frequency_per_day(period: Number, unit: FrequencyPeriodUnit) -> float:
    period_float = float(period)
    if unit == "h":
        return 24.0 / period_float
    if unit == "d":
        return 1.0 / period_float
    return 1.0 / (period_float * 7.0)


__all__ = [
    "DurationNormalization",
    "DurationUnit",
    "FrequencyNormalization",
    "FrequencyPeriodUnit",
    "MEDICATION_SIG_ADVISORY",
    "normalize_duration",
    "normalize_frequency",
]
