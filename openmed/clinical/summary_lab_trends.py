"""Strict evidence-bound laboratory value sequences for summary views.

A sequence renders only when every observation has a normalized analyte,
finite value, compatible normalized unit, compatible observation time, and an
opaque evidence identifier. Any insufficiency suppresses the complete sequence
and yields controlled value-free codes. No clinical trend is interpreted.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Final, Literal

from .units import parse_measurement

SUMMARY_LAB_TRENDS_SCHEMA_VERSION: Final = 1
SUMMARY_LAB_TRENDS_ADVISORY: Final = (
    "Laboratory sequences are evidence-bound review views. They make no clinical "
    "interpretation and do not replace the originating report or clinician review."
)

LabTrendInsufficiencyCode = Literal[
    "conflicting_observation_time",
    "incompatible_analyte",
    "incompatible_observation_time",
    "incompatible_unit",
    "insufficient_observations",
    "invalid_evidence_identifier",
    "invalid_observation_time",
    "invalid_unit",
    "invalid_value",
    "missing_analyte",
    "missing_evidence",
    "missing_observation_time",
    "missing_unit",
    "missing_value",
]

_OPAQUE_EVIDENCE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class LabTrendObservation:
    """One caller-normalized laboratory observation.

    Optional fields let incomplete extractor output reach controlled review
    routing. A missing field is never inferred from another observation.
    """

    analyte: str | None
    value: int | float | None
    unit: str | None
    observed_at: str | None
    evidence_id: str | None

    def __post_init__(self) -> None:
        for value in (self.analyte, self.unit, self.observed_at, self.evidence_id):
            if value is not None and type(value) is not str:
                raise TypeError("laboratory observation text fields must be strings")
        if self.value is not None and (
            isinstance(self.value, bool) or not isinstance(self.value, (int, float))
        ):
            raise TypeError("laboratory observation value must be numeric")


@dataclass(frozen=True, slots=True)
class LabSequencePoint:
    """One normalized, evidence-bound value in a rendered sequence."""

    value: float
    observed_at: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("sequence value must be finite")
        if not self.observed_at:
            raise ValueError("sequence observation time must be non-empty")
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if not evidence_ids or any(
            _OPAQUE_EVIDENCE_ID_RE.fullmatch(identifier) is None
            for identifier in evidence_ids
        ):
            raise ValueError("sequence evidence identifiers must be opaque")
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def to_dict(self) -> dict[str, object]:
        """Return one normalized sequence point."""

        return {
            "value": self.value,
            "observed_at": self.observed_at,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class LabValueSequence:
    """Compatible normalized analyte values in observation-time order."""

    analyte: str
    unit: str
    points: tuple[LabSequencePoint, ...]

    def __post_init__(self) -> None:
        if not self.analyte or not self.unit:
            raise ValueError("sequence analyte and unit must be non-empty")
        points = tuple(self.points)
        if len(points) < 2:
            raise ValueError("sequence requires at least two observations")
        if any(not isinstance(point, LabSequencePoint) for point in points):
            raise TypeError("points must contain LabSequencePoint values")
        object.__setattr__(self, "points", points)

    def to_dict(self) -> dict[str, object]:
        """Return the compatible value sequence without an interpretation."""

        return {
            "analyte": self.analyte,
            "unit": self.unit,
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class LabTrendSummary:
    """Rendered sequence or a value-free insufficiency result."""

    observation_count: int
    sequence: LabValueSequence | None
    insufficiency_codes: tuple[LabTrendInsufficiencyCode, ...] = ()
    schema_version: int = SUMMARY_LAB_TRENDS_SCHEMA_VERSION
    advisory: str = SUMMARY_LAB_TRENDS_ADVISORY

    def __post_init__(self) -> None:
        if type(self.observation_count) is not int or self.observation_count < 0:
            raise ValueError("observation_count must be a non-negative integer")
        if self.schema_version != SUMMARY_LAB_TRENDS_SCHEMA_VERSION:
            raise ValueError("unsupported laboratory-sequence schema version")
        codes = tuple(sorted(set(self.insufficiency_codes)))
        if self.sequence is not None and codes:
            raise ValueError("an insufficient summary cannot contain a sequence")
        if self.sequence is None and not codes:
            raise ValueError("a missing sequence requires insufficiency codes")
        object.__setattr__(self, "insufficiency_codes", codes)

    @property
    def status(self) -> Literal["rendered", "insufficient"]:
        """Return whether a compatible sequence was rendered."""

        return "rendered" if self.sequence is not None else "insufficient"

    @property
    def review_required(self) -> bool:
        """Return whether insufficiency requires human review."""

        return self.sequence is None

    def to_dict(self) -> dict[str, object]:
        """Return a sequence or a strictly value-free insufficiency artifact."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "review_required": self.review_required,
            "observation_count": self.observation_count,
            "advisory": self.advisory,
        }
        if self.sequence is None:
            payload["insufficiency_codes"] = list(self.insufficiency_codes)
        else:
            payload["sequence"] = self.sequence.to_dict()
        return payload

    def to_json(self) -> str:
        """Return byte-stable JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class _PreparedObservation:
    analyte: str
    value: float
    canonical_unit: str
    observed_at: str
    time_kind: str
    sort_value: datetime
    evidence_id: str


def render_lab_trend_summary(
    observations: Iterable[LabTrendObservation],
) -> LabTrendSummary:
    """Render a compatible value sequence or value-free insufficiency codes.

    The function is intentionally all-or-nothing. It never drops an incomplete
    or incompatible point to make the remaining observations look trendable,
    and it never labels the sequence as increasing, decreasing, or otherwise
    clinically meaningful.

    Args:
        observations: Typed normalized laboratory observations.

    Returns:
        A chronological value sequence only when every required field and
        compatibility check passes; otherwise a value-free review result.
    """

    if isinstance(observations, (str, bytes, bytearray)):
        raise TypeError("observations must contain LabTrendObservation values")
    try:
        records = tuple(observations)
    except TypeError:
        raise TypeError(
            "observations must contain LabTrendObservation values"
        ) from None
    if any(not isinstance(record, LabTrendObservation) for record in records):
        raise TypeError("observations must contain LabTrendObservation values")

    codes: list[LabTrendInsufficiencyCode] = []
    prepared: list[_PreparedObservation] = []
    for record in records:
        record_codes, item = _prepare_observation(record)
        codes.extend(record_codes)
        if item is not None:
            prepared.append(item)

    if len(records) < 2:
        codes.append("insufficient_observations")
    if not codes and prepared:
        if len({item.analyte.casefold() for item in prepared}) != 1:
            codes.append("incompatible_analyte")
        if len({item.canonical_unit for item in prepared}) != 1:
            codes.append("incompatible_unit")
        if len({item.time_kind for item in prepared}) != 1:
            codes.append("incompatible_observation_time")
        if _has_conflicting_time_values(prepared):
            codes.append("conflicting_observation_time")
        if len({item.sort_value for item in prepared}) < 2:
            codes.append("insufficient_observations")

    if codes:
        return LabTrendSummary(
            observation_count=len(records),
            sequence=None,
            insufficiency_codes=tuple(codes),
        )

    merged = _merge_points(prepared)
    return LabTrendSummary(
        observation_count=len(records),
        sequence=LabValueSequence(
            analyte=prepared[0].analyte,
            unit=prepared[0].canonical_unit,
            points=merged,
        ),
    )


def _prepare_observation(
    record: LabTrendObservation,
) -> tuple[tuple[LabTrendInsufficiencyCode, ...], _PreparedObservation | None]:
    codes: list[LabTrendInsufficiencyCode] = []
    analyte = _clean(record.analyte)
    unit = _clean(record.unit)
    observed_at = _clean(record.observed_at)
    evidence_id = _clean(record.evidence_id)

    if analyte is None:
        codes.append("missing_analyte")
    if record.value is None:
        codes.append("missing_value")
    elif not math.isfinite(float(record.value)):
        codes.append("invalid_value")
    if unit is None:
        codes.append("missing_unit")
    if observed_at is None:
        codes.append("missing_observation_time")
    if evidence_id is None:
        codes.append("missing_evidence")
    elif _OPAQUE_EVIDENCE_ID_RE.fullmatch(evidence_id) is None:
        codes.append("invalid_evidence_identifier")

    parsed_measurement = None
    if record.value is not None and math.isfinite(float(record.value)) and unit:
        parsed_measurement = parse_measurement(record.value, unit)
        if (
            parsed_measurement.get("status") != "ok"
            or parsed_measurement.get("canonical_magnitude") is None
            or parsed_measurement.get("canonical_unit") is None
        ):
            codes.append("invalid_unit")

    parsed_time = _parse_observation_time(observed_at) if observed_at else None
    if observed_at is not None and parsed_time is None:
        codes.append("invalid_observation_time")

    if codes:
        return tuple(codes), None
    if (
        analyte is None
        or evidence_id is None
        or parsed_measurement is None
        or parsed_time is None
    ):
        raise RuntimeError("complete laboratory observation was not prepared")
    return (), _PreparedObservation(
        analyte=analyte,
        value=float(parsed_measurement["canonical_magnitude"]),
        canonical_unit=str(parsed_measurement["canonical_unit"]),
        observed_at=parsed_time[0],
        time_kind=parsed_time[1],
        sort_value=parsed_time[2],
        evidence_id=evidence_id,
    )


def _parse_observation_time(value: str) -> tuple[str, str, datetime] | None:
    try:
        if "T" not in value and " " not in value:
            parsed_date = date.fromisoformat(value)
            return value, "date", datetime.combine(parsed_date, time.min)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.isoformat(timespec="seconds"), "local_datetime", parsed
    utc_value = parsed.astimezone(timezone.utc)
    normalized = utc_value.isoformat(timespec="seconds").replace("+00:00", "Z")
    return normalized, "absolute_datetime", utc_value.replace(tzinfo=None)


def _has_conflicting_time_values(records: list[_PreparedObservation]) -> bool:
    values_by_time: dict[datetime, set[float]] = defaultdict(set)
    for record in records:
        values_by_time[record.sort_value].add(record.value)
    return any(len(values) > 1 for values in values_by_time.values())


def _merge_points(
    records: list[_PreparedObservation],
) -> tuple[LabSequencePoint, ...]:
    grouped: dict[tuple[datetime, float], list[_PreparedObservation]] = defaultdict(
        list
    )
    for record in records:
        grouped[(record.sort_value, record.value)].append(record)
    return tuple(
        LabSequencePoint(
            value=group[0].value,
            observed_at=group[0].observed_at,
            evidence_ids=tuple(item.evidence_id for item in group),
        )
        for _, group in sorted(grouped.items(), key=lambda item: item[0])
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


__all__ = [
    "SUMMARY_LAB_TRENDS_ADVISORY",
    "SUMMARY_LAB_TRENDS_SCHEMA_VERSION",
    "LabSequencePoint",
    "LabTrendInsufficiencyCode",
    "LabTrendObservation",
    "LabTrendSummary",
    "LabValueSequence",
    "render_lab_trend_summary",
]
