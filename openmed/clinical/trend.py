"""Deterministic serial-measurement grouping and trend derivation.

Follow-up notes report the same measurement repeatedly over time -- a tumor
diameter, a laboratory analyte -- but a clinician wants an organized trend
rather than scattered data points. This module groups repeated measurements of
the same entity, orders them chronologically by *reusing* the clinical timeline
resolver, converts values to a common unit by *reusing* the UCUM-subset
measurement normalizer, and derives a descriptive trend direction
(``increasing`` / ``decreasing`` / ``stable`` / ``mixed`` / ``unknown``)
deterministically from the normalized value sequence.

No new timeline or unit engine is built here: timepoint ordering comes from
:func:`openmed.clinical.timeline.resolve_timeline` and unit conversion from
:func:`openmed.clinical.units.parse_measurement`. A derived direction is a
descriptive summary only; see :data:`TREND_ADVISORY`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Literal, Mapping, Optional, TypedDict

from .timeline import resolve_timeline
from .units import parse_measurement

TrendDirection = Literal["increasing", "decreasing", "stable", "mixed", "unknown"]

TREND_ADVISORY = (
    "Serial measurement trends are descriptive summaries for clinician review, "
    "not a clinical judgment. Direction is derived deterministically from "
    "unit-normalized values and never implies a diagnosis, response assessment, "
    "or treatment recommendation."
)

#: Tolerances that absorb floating-point normalization noise when deciding
#: whether two normalized values are unchanged. These are deliberately *not*
#: clinical significance thresholds: ``stable`` means numerically unchanged
#: after unit normalization, not "within a clinically meaningful band".
TREND_REL_EPSILON = 1e-9
TREND_ABS_EPSILON = 1e-12


class SerialMeasurementPoint(TypedDict):
    """One measurement occurrence within a serial group.

    ``entity``/``value``/``unit``/``timepoint`` echo the caller's input. The
    remaining fields are derived: ``canonical_magnitude``/``canonical_unit`` are
    the value expressed in the dimension's canonical unit (``None`` when the
    unit could not be parsed), ``normalized_timepoint`` is the timeline-resolved
    ISO value (``None`` when the timepoint could not be resolved), and
    ``comparable`` records whether the point joined its entity's primary
    comparable series.
    """

    entity: str
    value: Any
    unit: Optional[str]
    timepoint: Optional[str]
    canonical_magnitude: Optional[float]
    canonical_unit: Optional[str]
    normalized_timepoint: Optional[str]
    comparable: bool


class MeasurementTrend(TypedDict):
    """A grouped measurement series and its derived trend.

    ``points`` are the comparable points in chronological order when the series
    is orderable (``ordered`` is ``True``); when it is not orderable they are
    returned in input order and ``direction`` is ``unknown``. Points of the same
    entity that could not be compared (unparseable unit, or a unit whose
    dimension differs from the group's) are listed separately in
    ``incomparable_points`` rather than silently dropped. ``direction`` is
    ``unknown`` whenever the series cannot be totally ordered by timepoint or has
    fewer than two comparable points; in that case ``first_value``,
    ``last_value`` and ``delta`` are all ``None``. Otherwise ``delta`` is
    ``last_value - first_value`` in the canonical unit.
    """

    entity: str
    canonical_unit: Optional[str]
    points: list[SerialMeasurementPoint]
    direction: TrendDirection
    delta: Optional[float]
    first_value: Optional[float]
    last_value: Optional[float]
    comparable_count: int
    incomparable_points: list[SerialMeasurementPoint]
    ordered: bool
    advisory: str


@dataclass
class _PreparedPoint:
    """Internal per-point working state (never part of the public surface)."""

    input_index: int
    entity: str
    norm_entity: str
    value: Any
    unit: Optional[str]
    timepoint: Optional[str]
    canonical_magnitude: Optional[float]
    canonical_unit: Optional[str]
    normalized_timepoint: Optional[str]
    order_kind: Optional[str]
    order_value: Optional[int]
    parse_ok: bool


def _normalize_entity(entity: Any) -> str:
    """Case- and whitespace-insensitive grouping key for an entity name."""
    return " ".join(str(entity or "").split()).casefold()


def _resolve_timepoint(
    timepoint: Any,
    reference_date: str | date | datetime | None,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Resolve a timepoint expression into a sortable key via the timeline.

    Returns ``(order_kind, order_value, normalized_iso)`` where ``order_kind`` is
    ``"absolute"`` when the resolver produces an ISO date, ``"relative"`` when
    only a signed day offset is known, and ``None`` when the timepoint cannot be
    resolved. Each timepoint is expected to be a single temporal expression; if
    the resolver detects more than one, the chronologically earliest is used.
    Reuses :func:`resolve_timeline`; never consults the wall clock.
    """
    text = "" if timepoint is None else str(timepoint).strip()
    if not text:
        return None, None, None
    timeline = resolve_timeline(text, reference_date=reference_date)
    if not timeline.events:
        return None, None, None
    event = timeline.events[0]
    if event.interval is not None:
        return "absolute", event.interval.start.toordinal(), event.normalized_value
    if event.relative_offset_days is not None:
        return "relative", event.relative_offset_days, event.normalized_value
    return None, None, event.normalized_value


def _prepare_point(
    index: int,
    point: Mapping[str, Any],
    reference_date: str | date | datetime | None,
) -> _PreparedPoint:
    """Normalize a raw point: parse its measurement and resolve its timepoint."""
    entity = point.get("entity", "")
    value = point.get("value")
    unit = point.get("unit")
    timepoint = point.get("timepoint")

    parsed = parse_measurement(value, unit)
    canonical_magnitude = parsed.get("canonical_magnitude")
    parse_ok = parsed.get("status") == "ok" and canonical_magnitude is not None
    canonical_unit = parsed.get("canonical_unit") if parse_ok else None
    if not parse_ok:
        canonical_magnitude = None

    order_kind, order_value, normalized_tp = _resolve_timepoint(
        timepoint, reference_date
    )
    return _PreparedPoint(
        input_index=index,
        entity=str(entity),
        norm_entity=_normalize_entity(entity),
        value=value,
        unit=unit,
        timepoint=timepoint,
        canonical_magnitude=canonical_magnitude,
        canonical_unit=canonical_unit,
        normalized_timepoint=normalized_tp,
        order_kind=order_kind,
        order_value=order_value,
        parse_ok=parse_ok,
    )


def _point_view(
    prepared: _PreparedPoint, *, comparable: bool
) -> SerialMeasurementPoint:
    """Project internal working state onto the public point view."""
    return SerialMeasurementPoint(
        entity=prepared.entity,
        value=prepared.value,
        unit=prepared.unit,
        timepoint=prepared.timepoint,
        canonical_magnitude=prepared.canonical_magnitude,
        canonical_unit=prepared.canonical_unit,
        normalized_timepoint=prepared.normalized_timepoint,
        comparable=comparable,
    )


def _changed(previous: float, current: float) -> bool:
    """True when two normalized values differ beyond floating-point noise."""
    tolerance = max(
        TREND_ABS_EPSILON, TREND_REL_EPSILON * max(abs(previous), abs(current))
    )
    return abs(current - previous) > tolerance


def _classify(magnitudes: list[float]) -> TrendDirection:
    """Derive a direction from an ordered sequence of >=2 normalized values."""
    ups = downs = 0
    for previous, current in zip(magnitudes, magnitudes[1:]):
        if _changed(previous, current):
            if current > previous:
                ups += 1
            else:
                downs += 1
    if ups and downs:
        return "mixed"
    if ups:
        return "increasing"
    if downs:
        return "decreasing"
    return "stable"


def _primary_unit(ok_members: list[_PreparedPoint]) -> Optional[str]:
    """Pick the comparable unit: most frequent, breaking ties by unit name.

    Ties are broken by the canonical unit string rather than input order, so the
    selection is invariant to how the caller happens to order the points.
    """
    if not ok_members:
        return None
    counts = Counter(member.canonical_unit for member in ok_members)
    return min(counts, key=lambda unit: (-counts[unit], str(unit)))


def _build_trend(members: list[_PreparedPoint]) -> MeasurementTrend:
    """Build one trend for a single normalized-entity group."""
    ok_members = [member for member in members if member.parse_ok]
    primary_unit = _primary_unit(ok_members)

    comparable = [
        member for member in ok_members if member.canonical_unit == primary_unit
    ]
    comparable_ids = {member.input_index for member in comparable}
    incomparable = [
        member for member in members if member.input_index not in comparable_ids
    ]

    # A series is orderable only when every comparable point resolves to a
    # timepoint of one consistent kind (all absolute dates, or all relative
    # offsets). Otherwise -- including a series with no timepoints -- the
    # direction is left unknown rather than inferred from input order.
    orderable = bool(comparable) and all(
        member.order_value is not None for member in comparable
    )
    if orderable:
        orderable = len({member.order_kind for member in comparable}) == 1

    if orderable:
        ordered = sorted(
            comparable, key=lambda member: (member.order_value, member.input_index)
        )
    else:
        ordered = list(comparable)

    magnitudes = [member.canonical_magnitude for member in ordered]
    if orderable and len(magnitudes) >= 2:
        direction: TrendDirection = _classify(magnitudes)
        delta: Optional[float] = magnitudes[-1] - magnitudes[0]
        first_value: Optional[float] = magnitudes[0]
        last_value: Optional[float] = magnitudes[-1]
    else:
        # No derivable direction: leave the summary values unset so that
        # first_value / last_value / delta are populated only for a real trend.
        direction = "unknown"
        delta = None
        first_value = None
        last_value = None

    return MeasurementTrend(
        entity=members[0].entity,
        canonical_unit=primary_unit,
        points=[_point_view(member, comparable=True) for member in ordered],
        direction=direction,
        delta=delta,
        first_value=first_value,
        last_value=last_value,
        comparable_count=len(comparable),
        incomparable_points=[
            _point_view(member, comparable=False) for member in incomparable
        ],
        ordered=orderable,
        advisory=TREND_ADVISORY,
    )


def extract_measurement_trends(
    points: Iterable[Mapping[str, Any]],
    *,
    reference_date: str | date | datetime | None = None,
) -> list[MeasurementTrend]:
    """Group serial measurements by entity and derive a trend for each.

    Args:
        points: An iterable of measurement mappings. Each carries an ``entity``
            name, a ``value`` (a numeric magnitude, or a combined string such as
            ``"12 mm"``), an optional ``unit`` string, and an optional
            ``timepoint`` temporal expression (an ISO date such as
            ``"2026-01-05"`` or a relative phrase such as ``"3 weeks ago"``).
        reference_date: Optional document reference date used to resolve relative
            timepoints. Absent, relative expressions remain unresolved and their
            series are labeled ``unknown``; the wall clock is never substituted.

    Returns:
        One :class:`MeasurementTrend` per distinct entity (grouped case- and
        whitespace-insensitively), in first-seen order. Within each trend,
        comparable points are unit-normalized and chronologically ordered, and
        every trend carries :data:`TREND_ADVISORY`.
    """
    prepared = [
        _prepare_point(index, point, reference_date)
        for index, point in enumerate(points)
    ]

    groups: dict[str, list[_PreparedPoint]] = {}
    order: list[str] = []
    for member in prepared:
        if member.norm_entity not in groups:
            groups[member.norm_entity] = []
            order.append(member.norm_entity)
        groups[member.norm_entity].append(member)

    return [_build_trend(groups[key]) for key in order]


__all__ = [
    "TREND_ADVISORY",
    "TREND_ABS_EPSILON",
    "TREND_REL_EPSILON",
    "TrendDirection",
    "SerialMeasurementPoint",
    "MeasurementTrend",
    "extract_measurement_trends",
]
