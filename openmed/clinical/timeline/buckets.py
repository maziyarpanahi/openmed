"""Deterministic, source-linked views of explicit ConText temporality axes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from openmed.clinical.context import (
    CERTAINTY_VALUES,
    HISTORICAL,
    HYPOTHETICAL,
    RECENT,
    TEMPORALITY_VALUES,
    UNCERTAIN,
    ClinicalAssertion,
)
from openmed.clinical.temporal_intervals import TemporalInterval
from openmed.clinical.timeline.assembler import ClinicalEvent, ClinicalEventTimeline
from openmed.clinical.timeline.timex import TimeExpr
from openmed.core.audit import hash_text

BUCKETED_TIMELINE_SCHEMA_VERSION = 2
BUCKETED_TIMELINE_ADVISORY = (
    "This timeline is an advisory ordering of supplied spans, not a clinical "
    "chronology of record. Unanchored and hypothetical events require review."
)
_LANES = (HISTORICAL, RECENT, HYPOTHETICAL)
_SAFE_EVENT_KINDS = frozenset(
    {
        "condition",
        "diagnosis",
        "event",
        "finding",
        "medication",
        "observation",
        "procedure",
        "symptom",
    }
)
_DATE_RE = re.compile(
    r"\d{4}(?:-\d{2}(?:-\d{2})?)?"
    r"(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
_RELATIVE_RE = re.compile(r"P(?=\d)(?:\d+Y)?(?:\d+M)?(?:\d+W)?(?:\d+D)?\Z")
_PRECISIONS = frozenset(
    {"unknown", "year", "month", "day", "week", "hour", "minute", "second", "mixed"}
)
_DIRECTIONS = frozenset(
    {
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
        "calendar_past",
        "calendar_future",
    }
)


@dataclass(frozen=True)
class TimelineTimeEvidence:
    """A source time expression represented by offsets and a fingerprint."""

    start: int
    end: int
    fingerprint: str | None
    normalized_value: str | None
    state: str
    relative_value: str | None = None
    direction: str | None = None
    precision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return source provenance without the original expression text."""

        return {
            "source_span": [self.start, self.end],
            "fingerprint": self.fingerprint,
            "normalized_value": self.normalized_value,
            "relative_value": self.relative_value,
            "direction": self.direction,
            "precision": self.precision,
            "state": self.state,
        }


@dataclass(frozen=True)
class BucketedClinicalTimeline(ClinicalEventTimeline):
    """A clinical timeline grouped by explicit temporality axes."""

    time_evidence: tuple[TimelineTimeEvidence | None, ...] = ()
    disclaimer: str = BUCKETED_TIMELINE_ADVISORY
    schema_version: int = BUCKETED_TIMELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.time_evidence) != len(self.events):
            raise ValueError("timeline evidence must align with events")

    @property
    def lanes(self) -> dict[str, tuple[ClinicalEvent, ...]]:
        """Return historical, recent, and hypothetical events separately."""

        return {
            lane: tuple(
                event for event in self.events if event.assertion.temporality == lane
            )
            for lane in _LANES
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, source-text-free bucketed timeline payload."""

        payload = super().to_dict()
        payload["lanes"] = {
            lane: [
                index
                for index, event in enumerate(self.events)
                if event.assertion.temporality == lane
            ]
            for lane in _LANES
        }
        payload["time_evidence"] = [
            evidence.to_dict() if evidence is not None else None
            for evidence in self.time_evidence
        ]
        return payload

    def to_jsonl(self) -> str:
        """Return byte-stable JSON lines with the schema and source offsets."""

        lines = [
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "disclaimer": self.disclaimer,
                    "kind": "timeline",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ]
        for event, evidence in zip(self.events, self.time_evidence, strict=True):
            lines.append(
                json.dumps(
                    {
                        "kind": "event",
                        "lane": event.assertion.temporality,
                        "event": event.to_dict(),
                        "time_evidence": (
                            evidence.to_dict() if evidence is not None else None
                        ),
                        "schema_version": self.schema_version,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return "\n".join(lines) + "\n"


def build_timeline(
    spans: Iterable[Mapping[str, Any] | Any],
) -> BucketedClinicalTimeline:
    """Bucket already-tagged spans without inferring an assertion or date.

    Each span must supply half-open ``start``/``end`` offsets and a ConText
    ``temporality`` (directly or through ``assertion``). Optional anchored
    ``TimeExpr`` or conservative ``TemporalInterval`` values determine ordering
    within historical and recent lanes. Unknown times retain document order.
    """

    rows: list[tuple[ClinicalEvent, TimelineTimeEvidence | None]] = []
    for span in spans:
        start = _field(span, "start")
        end = _field(span, "end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError("timeline spans require valid half-open offsets")

        assertion = _field(span, "assertion")
        temporality = _field(span, "temporality")
        if temporality is None:
            temporality = _field(assertion, "temporality")
        if temporality not in TEMPORALITY_VALUES:
            raise ValueError("timeline spans require an explicit temporality")
        certainty = _field(span, "certainty")
        if certainty is None:
            certainty = _field(assertion, "certainty")
        if certainty is None:
            certainty = UNCERTAIN
        if certainty not in CERTAINTY_VALUES:
            raise ValueError("timeline certainty must use a ConText axis value")

        label = _field(span, "label", "event")
        event_kind = str(label).casefold() if isinstance(label, str) else "event"
        if event_kind not in _SAFE_EVENT_KINDS:
            event_kind = "event"
        source_text = _field(span, "text")
        fingerprint = hash_text(
            source_text
            if isinstance(source_text, str)
            else f"{start}:{end}:{event_kind}"
        )
        normalized_time, evidence = _time_parts(span)
        rows.append(
            (
                ClinicalEvent(
                    entity=fingerprint,
                    event_kind=event_kind,
                    normalized_time=normalized_time,
                    section=None,
                    assertion=ClinicalAssertion(
                        temporality=temporality,
                        certainty=certainty,
                    ),
                    source_span=(start, end),
                ),
                evidence,
            )
        )

    ordered: list[tuple[ClinicalEvent, TimelineTimeEvidence | None]] = []
    for lane in _LANES:
        lane_rows = [row for row in rows if row[0].assertion.temporality == lane]
        lane_rows.sort(key=lambda row: _sort_key(row[0], lane))
        ordered.extend(lane_rows)
    return BucketedClinicalTimeline(
        events=tuple(row[0] for row in ordered),
        time_evidence=tuple(row[1] for row in ordered),
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError("timeline time must be a normalized date")
    candidate = value.strip()
    if not _DATE_RE.fullmatch(candidate):
        raise ValueError("timeline time must be a normalized date")
    try:
        if "T" in candidate:
            datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        elif len(candidate) == 10:
            date.fromisoformat(candidate)
        elif len(candidate) == 7:
            month = int(candidate[5:7])
            if not 1 <= month <= 12:
                raise ValueError
        elif int(candidate) < 1:
            raise ValueError
    except ValueError as exc:
        raise ValueError("timeline time must be a normalized date") from exc
    return candidate


def _time_parts(
    span: Mapping[str, Any] | Any,
) -> tuple[str | None, TimelineTimeEvidence | None]:
    timex = _field(span, "time_expr")
    if timex is None:
        timex = _field(span, "timex")
    if isinstance(timex, TimeExpr):
        normalized = (
            _canonical_date(timex.value)
            if timex.kind == "DATE" and timex.value is not None
            else None
        )
        return normalized, TimelineTimeEvidence(
            start=timex.start,
            end=timex.end,
            fingerprint=hash_text(timex.text),
            normalized_value=normalized,
            state="anchored" if normalized is not None else "relative_or_unknown",
            relative_value=(
                timex.relative_value
                if isinstance(timex.relative_value, str)
                and _RELATIVE_RE.fullmatch(timex.relative_value)
                else None
            ),
            direction=timex.direction if timex.direction in _DIRECTIONS else None,
        )

    interval = _field(span, "temporal_interval")
    if isinstance(interval, TemporalInterval):
        context = _field(span, "context")
        source_fingerprint = (
            hash_text(context[interval.source_start : interval.source_end])
            if isinstance(context, str) and interval.source_end <= len(context)
            else None
        )
        normalized = (
            _canonical_date(interval.value)
            if interval.kind == "date"
            and interval.status == "normalized"
            and interval.value is not None
            else None
        )
        return normalized, TimelineTimeEvidence(
            start=interval.source_start,
            end=interval.source_end,
            fingerprint=source_fingerprint,
            normalized_value=normalized,
            state=interval.status,
            precision=(
                interval.precision if interval.precision in _PRECISIONS else "unknown"
            ),
        )

    return _canonical_date(_field(span, "normalized_time")), None


def _sort_key(event: ClinicalEvent, lane: str) -> tuple[Any, ...]:
    if lane == HYPOTHETICAL:
        return event.start, event.end, event.entity
    value = event.normalized_time
    if value is None:
        return 1, "", event.start, event.end, event.entity
    return 0, value, event.start, event.end, event.entity


__all__ = [
    "BUCKETED_TIMELINE_ADVISORY",
    "BUCKETED_TIMELINE_SCHEMA_VERSION",
    "BucketedClinicalTimeline",
    "TimelineTimeEvidence",
    "build_timeline",
]
