"""Clinical timeline resolution with relative-date normalization."""

from __future__ import annotations

import calendar
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from openmed.clinical.context import (
    reconcile_temporality_with_interval,
    resolve_temporality,
)
from openmed.clinical.timeline.timex import (
    TemporalExpression,
    detect_timexes,
)

TIMELINE_ASSISTIVE_DISCLAIMER = (
    "Clinical timeline normalization is assistive and is not a clinical "
    "decision, diagnosis, treatment recommendation, or substitute for "
    "clinician review."
)

TimelineRelationKind = Literal["before", "after", "overlap", "unknown"]

_ANCHOR_TERMS = {
    "admission": ("admission", "admitted", "hospitalization"),
    "surgery": ("surgery", "operation", "operative", "post-op", "postoperative"),
    "procedure": ("procedure",),
    "discharge": ("discharge", "discharged"),
    "visit": ("visit", "clinic"),
    "onset": ("onset", "started", "began"),
}
_REFERENCE_DEPENDENT_DIRECTIONS = {
    "same",
    "past",
    "future",
    "since",
}


@dataclass(frozen=True)
class NormalizedInterval:
    """A normalized ISO day-level interval with explicit uncertainty bounds."""

    start: date
    end: date
    lower_bound: date
    upper_bound: date
    precision: str = "day"
    uncertainty_days: int = 0

    @property
    def iso_value(self) -> str:
        """Return ``YYYY-MM-DD/YYYY-MM-DD`` interval text."""

        return f"{self.start.isoformat()}/{self.end.isoformat()}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready interval payload."""

        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "value": self.iso_value,
            "lower_bound": self.lower_bound.isoformat(),
            "upper_bound": self.upper_bound.isoformat(),
            "precision": self.precision,
            "uncertainty_days": self.uncertainty_days,
        }


@dataclass(frozen=True)
class TimelineEvent:
    """One timeline event derived from a temporal expression."""

    event_id: str
    text: str
    start: int
    end: int
    timex: TemporalExpression
    interval: NormalizedInterval | None
    temporality: str
    reference_date_dependent: bool
    reference_date_provenance: Mapping[str, Any]
    relation_anchor: str | None = None
    relative_offset_days: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_value(self) -> str | None:
        """Return the normalized interval value when resolved."""

        return self.interval.iso_value if self.interval is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready event payload."""

        return {
            "id": self.event_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "timex": self.timex.to_dict(),
            "interval": self.interval.to_dict() if self.interval else None,
            "normalized_value": self.normalized_value,
            "temporality": self.temporality,
            "reference_date_dependent": self.reference_date_dependent,
            "reference_date_provenance": dict(self.reference_date_provenance),
            "relation_anchor": self.relation_anchor,
            "relative_offset_days": self.relative_offset_days,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class TimelineRelation:
    """A partial-order relation between two events or anchors."""

    source_id: str
    target_id: str
    relation: TimelineRelationKind
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready relation payload."""

        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ResolvedTimeline:
    """Resolved timeline with ordered events and partial-order relations."""

    events: tuple[TimelineEvent, ...]
    relations: tuple[TimelineRelation, ...]
    reference_date: date | None
    reference_date_provenance: Mapping[str, Any]
    disclaimer: str = TIMELINE_ASSISTIVE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready timeline payload."""

        return {
            "events": [event.to_dict() for event in self.events],
            "relations": [relation.to_dict() for relation in self.relations],
            "reference_date": (
                self.reference_date.isoformat()
                if self.reference_date is not None
                else None
            ),
            "reference_date_provenance": dict(self.reference_date_provenance),
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class TimelineEvaluationResult:
    """Value-accuracy and ordering-consistency metrics for gold timelines."""

    value_accuracy: float
    ordering_consistency: float
    value_correct: int
    value_total: int
    ordering_correct: int
    ordering_total: int
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready evaluation payload."""

        return {
            "value_accuracy": self.value_accuracy,
            "ordering_consistency": self.ordering_consistency,
            "value_correct": self.value_correct,
            "value_total": self.value_total,
            "ordering_correct": self.ordering_correct,
            "ordering_total": self.ordering_total,
            "failures": list(self.failures),
        }


def resolve_timeline(
    text: str,
    *,
    reference_date: str | date | datetime | None = None,
) -> ResolvedTimeline:
    """Resolve temporal expressions into a normalized clinical timeline.

    Args:
        text: Source clinical note text.
        reference_date: Optional document reference date.  Relative
            expressions that depend on this anchor remain relative-only when it
            is absent; the resolver never substitutes the current wall clock.

    Returns:
        A ``ResolvedTimeline`` carrying normalized events, partial-order
        relations, reference-date provenance, and the assistive disclaimer.
    """

    document_reference = _coerce_date(reference_date)
    reference_provenance = _reference_date_provenance(
        document_reference=document_reference,
        required=False,
    )
    timexes = detect_timexes(text)
    sentence_windows = _sentence_windows(text)
    anchor_dates: dict[str, date] = {}
    previous_interval_start: date | None = None
    events: list[TimelineEvent] = []

    for index, timex in enumerate(timexes, start=1):
        sentence_start, sentence_end = _sentence_for_offsets(
            sentence_windows,
            timex.start,
            timex.end,
        )
        sentence_text = text[sentence_start:sentence_end].strip()
        interval = _resolve_interval(
            timex,
            document_reference=document_reference,
            anchor_dates=anchor_dates,
            previous_interval_start=previous_interval_start,
        )
        reference_required = _reference_date_required(timex)
        event_reference_provenance = _reference_date_provenance(
            document_reference=document_reference,
            required=reference_required,
        )
        temporality = resolve_temporality(
            {
                "text": sentence_text,
                "context": text,
                "start": sentence_start,
                "end": sentence_end,
            }
        )
        if interval is not None and document_reference is not None:
            temporality = reconcile_temporality_with_interval(
                temporality=temporality,
                interval_start=interval.start,
                interval_end=interval.end,
                reference_date=document_reference,
            )
        event = TimelineEvent(
            event_id=f"t{index}",
            text=sentence_text,
            start=sentence_start,
            end=sentence_end,
            timex=timex,
            interval=interval,
            temporality=temporality,
            reference_date_dependent=reference_required,
            reference_date_provenance=event_reference_provenance,
            relation_anchor=timex.anchor,
            relative_offset_days=_relative_offset_days(timex),
            provenance={
                "source": "openmed.clinical.timeline",
                "timex_start": timex.start,
                "timex_end": timex.end,
                "timex_text": timex.text,
            },
        )
        events.append(event)

        if interval is not None:
            previous_interval_start = interval.start
            _update_anchor_dates(anchor_dates, sentence_text, interval.start)
        elif timex.anchor and timex.anchor in anchor_dates:
            previous_interval_start = anchor_dates[timex.anchor]

    relations = _timeline_relations(events, document_reference)
    ordered_events = tuple(sorted(events, key=_event_sort_key))
    return ResolvedTimeline(
        events=ordered_events,
        relations=tuple(relations),
        reference_date=document_reference,
        reference_date_provenance={
            **reference_provenance,
            "required": any(event.reference_date_dependent for event in events),
        },
    )


def evaluate_timeline_gold(
    cases_or_path: str | Path | Iterable[Mapping[str, Any]],
) -> TimelineEvaluationResult:
    """Evaluate ``resolve_timeline`` against synthetic gold cases."""

    cases = _load_gold_cases(cases_or_path)
    value_correct = 0
    value_total = 0
    ordering_correct = 0
    ordering_total = 0
    failures: list[str] = []

    for case in cases:
        timeline = resolve_timeline(
            str(case["text"]),
            reference_date=case.get("reference_date"),
        )
        events_by_timex = {
            _norm(event.timex.text): event
            for event in timeline.events
            if event.timex.text
        }

        for expected in case.get("expected_events", ()):
            timex_text = str(expected["timex_text"])
            expected_value = expected.get("normalized_value")
            event = events_by_timex.get(_norm(timex_text))
            if expected_value is None:
                continue
            value_total += 1
            if event is not None and event.normalized_value == expected_value:
                value_correct += 1
            else:
                actual = event.normalized_value if event is not None else None
                failures.append(
                    f"{case['id']} value {timex_text!r}: "
                    f"expected {expected_value!r}, got {actual!r}"
                )

        for expected in case.get("expected_relations", ()):
            source = events_by_timex.get(_norm(str(expected["source"])))
            target = events_by_timex.get(_norm(str(expected["target"])))
            ordering_total += 1
            actual_relation = (
                _event_pair_relation(source, target)
                if source is not None and target is not None
                else "unknown"
            )
            if actual_relation == expected["relation"]:
                ordering_correct += 1
            else:
                failures.append(
                    f"{case['id']} relation {expected['source']!r} -> "
                    f"{expected['target']!r}: expected {expected['relation']!r}, "
                    f"got {actual_relation!r}"
                )

    return TimelineEvaluationResult(
        value_accuracy=_rate(value_correct, value_total),
        ordering_consistency=_rate(ordering_correct, ordering_total),
        value_correct=value_correct,
        value_total=value_total,
        ordering_correct=ordering_correct,
        ordering_total=ordering_total,
        failures=tuple(failures),
    )


def _resolve_interval(
    timex: TemporalExpression,
    *,
    document_reference: date | None,
    anchor_dates: Mapping[str, date],
    previous_interval_start: date | None,
) -> NormalizedInterval | None:
    if timex.timex_type == "SET":
        return None
    if timex.timex_type == "DURATION":
        if timex.metadata.get("history_duration") and document_reference is not None:
            start = _add_duration(
                document_reference,
                amount=-(timex.amount or 0),
                unit=timex.unit or "day",
            )
            return _interval(
                start,
                document_reference,
                uncertainty_days=timex.uncertainty_days,
            )
        return None
    if timex.direction == "none":
        return _absolute_interval(timex)
    if timex.direction == "same" and document_reference is not None:
        return _interval(document_reference, document_reference)
    if timex.direction in {"past", "future"} and document_reference is not None:
        amount = timex.amount or 0
        if timex.direction == "past":
            amount = -amount
        target = _add_duration(document_reference, amount=amount, unit=timex.unit)
        return _interval(target, target, uncertainty_days=timex.uncertainty_days)
    if timex.direction in {"after_previous", "before_previous"}:
        if previous_interval_start is None:
            return None
        amount = timex.amount or 0
        if timex.direction == "before_previous":
            amount = -amount
        target = _add_duration(previous_interval_start, amount=amount, unit=timex.unit)
        return _interval(target, target, uncertainty_days=timex.uncertainty_days)
    if timex.direction == "postop_day":
        anchor = _lookup_anchor(anchor_dates, "surgery")
        if anchor is None:
            return None
        target = _add_duration(anchor, amount=timex.amount or 0, unit="day")
        return _interval(target, target, uncertainty_days=timex.uncertainty_days)
    if timex.direction == "since":
        anchor = _lookup_anchor(anchor_dates, timex.anchor)
        if anchor is None or document_reference is None:
            return None
        return _interval(
            min(anchor, document_reference),
            max(anchor, document_reference),
            uncertainty_days=timex.uncertainty_days,
        )
    return None


def _absolute_interval(timex: TemporalExpression) -> NormalizedInterval | None:
    if timex.value is None:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", timex.value):
        value = date.fromisoformat(timex.value)
        return _interval(
            value,
            value,
            precision=str(timex.metadata.get("precision", "day")),
            uncertainty_days=timex.uncertainty_days,
        )
    if re.fullmatch(r"\d{4}-\d{2}", timex.value):
        year, month = (int(part) for part in timex.value.split("-"))
        start = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
        return _interval(
            start,
            end,
            precision="month",
            uncertainty_days=timex.uncertainty_days,
        )
    return None


def _interval(
    start: date,
    end: date,
    *,
    precision: str = "day",
    uncertainty_days: int = 0,
) -> NormalizedInterval:
    if end < start:
        start, end = end, start
    return NormalizedInterval(
        start=start,
        end=end,
        lower_bound=start - timedelta(days=uncertainty_days),
        upper_bound=end + timedelta(days=uncertainty_days),
        precision=precision,
        uncertainty_days=uncertainty_days,
    )


def _add_duration(value: date, *, amount: int, unit: str | None) -> date:
    if unit == "day":
        return value + timedelta(days=amount)
    if unit == "week":
        return value + timedelta(weeks=amount)
    if unit == "month":
        return _add_months(value, amount)
    if unit == "year":
        return _add_months(value, amount * 12)
    return value


def _add_months(value: date, amount: int) -> date:
    month_index = value.month - 1 + amount
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _relative_offset_days(timex: TemporalExpression) -> int | None:
    if timex.amount is None or timex.unit is None:
        return None
    days_per_unit = {
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365,
    }.get(timex.unit)
    if days_per_unit is None:
        return None
    amount = timex.amount * days_per_unit
    if timex.direction in {"past", "before_previous", "before_anchor"}:
        return -amount
    if timex.direction in {"future", "after_previous", "after_anchor", "postop_day"}:
        return amount
    if timex.direction == "same":
        return 0
    return None


def _reference_date_required(timex: TemporalExpression) -> bool:
    if timex.direction in _REFERENCE_DEPENDENT_DIRECTIONS:
        return True
    return bool(timex.metadata.get("history_duration"))


def _reference_date_provenance(
    *,
    document_reference: date | None,
    required: bool,
) -> dict[str, Any]:
    return {
        "required": required,
        "provided": document_reference is not None,
        "source": "user_supplied" if document_reference is not None else "not_supplied",
        "value": document_reference.isoformat() if document_reference else None,
    }


def _timeline_relations(
    events: Sequence[TimelineEvent],
    document_reference: date | None,
) -> list[TimelineRelation]:
    relations: list[TimelineRelation] = []
    for event in events:
        if event.reference_date_dependent:
            relations.extend(_document_reference_relations(event, document_reference))
        if event.interval is None and event.relation_anchor is not None:
            relations.append(
                TimelineRelation(
                    source_id=event.event_id,
                    target_id=f"anchor:{event.relation_anchor}",
                    relation=_anchor_relation(event),
                    evidence=event.timex.text,
                )
            )

    for left, right in combinations(events, 2):
        relation = _event_pair_relation(left, right)
        if relation != "unknown":
            relations.append(
                TimelineRelation(
                    source_id=left.event_id,
                    target_id=right.event_id,
                    relation=relation,
                    evidence=f"{left.timex.text} | {right.timex.text}",
                )
            )
    return relations


def _document_reference_relations(
    event: TimelineEvent,
    document_reference: date | None,
) -> list[TimelineRelation]:
    if event.interval is not None and document_reference is not None:
        relation = _interval_relation(
            event.interval.start,
            event.interval.end,
            document_reference,
            document_reference,
        )
    elif event.relative_offset_days is not None:
        if event.relative_offset_days < 0:
            relation = "before"
        elif event.relative_offset_days > 0:
            relation = "after"
        else:
            relation = "overlap"
    else:
        relation = "unknown"
    if relation == "unknown":
        return []
    return [
        TimelineRelation(
            source_id=event.event_id,
            target_id="document_reference",
            relation=relation,
            evidence=event.timex.text,
        )
    ]


def _event_pair_relation(
    left: TimelineEvent | None,
    right: TimelineEvent | None,
) -> TimelineRelationKind:
    if left is None or right is None:
        return "unknown"
    if left.interval is not None and right.interval is not None:
        return _interval_relation(
            left.interval.start,
            left.interval.end,
            right.interval.start,
            right.interval.end,
        )
    if left.relative_offset_days is not None and right.relative_offset_days is not None:
        if left.relative_offset_days < right.relative_offset_days:
            return "before"
        if left.relative_offset_days > right.relative_offset_days:
            return "after"
        return "overlap"
    return "unknown"


def _interval_relation(
    left_start: date,
    left_end: date,
    right_start: date,
    right_end: date,
) -> TimelineRelationKind:
    if left_end < right_start:
        return "before"
    if right_end < left_start:
        return "after"
    return "overlap"


def _anchor_relation(event: TimelineEvent) -> TimelineRelationKind:
    if event.timex.direction in {"before_anchor", "before_previous"}:
        return "before"
    if event.timex.direction in {"after_anchor", "after_previous", "postop_day"}:
        return "after"
    if event.timex.direction == "since":
        return "after"
    return "unknown"


def _update_anchor_dates(
    anchor_dates: dict[str, date],
    sentence_text: str,
    anchor_date: date,
) -> None:
    normalized = sentence_text.casefold()
    for anchor, terms in _ANCHOR_TERMS.items():
        if any(term in normalized for term in terms):
            anchor_dates[anchor] = anchor_date
            if anchor == "surgery":
                anchor_dates["operation"] = anchor_date
            if anchor == "admission":
                anchor_dates["last admission"] = anchor_date


def _lookup_anchor(anchor_dates: Mapping[str, date], anchor: str | None) -> date | None:
    if anchor is None:
        return None
    if anchor in anchor_dates:
        return anchor_dates[anchor]
    if anchor == "operation":
        return anchor_dates.get("surgery")
    if anchor == "last admission":
        return anchor_dates.get("admission")
    return None


def _event_sort_key(event: TimelineEvent) -> tuple[int, Any, int]:
    if event.interval is not None:
        return (0, event.interval.start, event.start)
    if event.relative_offset_days is not None:
        return (1, event.relative_offset_days, event.start)
    return (2, event.start, event.start)


def _coerce_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _sentence_windows(text: str) -> tuple[tuple[int, int], ...]:
    if not text:
        return ()
    windows: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)|\n+", text):
        _append_sentence_window(windows, text, start, match.end())
        start = match.end()
    _append_sentence_window(windows, text, start, len(text))
    return tuple(windows)


def _append_sentence_window(
    windows: list[tuple[int, int]],
    text: str,
    start: int,
    end: int,
) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        windows.append((start, end))


def _sentence_for_offsets(
    windows: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    for sentence_start, sentence_end in windows:
        if start >= sentence_start and end <= sentence_end:
            return sentence_start, sentence_end
    return 0, end


def _load_gold_cases(
    cases_or_path: str | Path | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(cases_or_path, (str, Path)):
        raw = json.loads(Path(cases_or_path).read_text(encoding="utf-8"))
        if raw.get("synthetic") is not True:
            raise ValueError("timeline gold corpus must be marked synthetic")
        cases = raw.get("cases")
        if not isinstance(cases, list):
            raise ValueError("timeline gold corpus must contain a cases list")
        return cases
    return list(cases_or_path)


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


__all__ = [
    "NormalizedInterval",
    "ResolvedTimeline",
    "TIMELINE_ASSISTIVE_DISCLAIMER",
    "TimelineEvaluationResult",
    "TimelineEvent",
    "TimelineRelation",
    "TimelineRelationKind",
    "evaluate_timeline_gold",
    "resolve_timeline",
]
