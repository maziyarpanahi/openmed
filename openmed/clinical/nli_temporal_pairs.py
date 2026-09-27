"""Temporality-aware clinical NLI pair construction.

Clinical NLI can overstate a claim when a historical event is compared with a
present-tense hypothesis.  This module keeps the temporal evidence beside both
pair sides and applies a conservative compatibility gate before an NLI result
is treated as entailment.  The implementation is intentionally dependency-free
and local-first: temporal expressions use the existing rules-based normalizer,
and no model, network, clock, or filesystem state is consulted.

Raw pair text is retained only for the explicit local model-input boundary.
Audit dictionaries, JSON, and representations contain offsets, lengths,
hashes, normalized interval metadata, and controlled review reasons instead of
source text.  The output is an assistive review input, not a clinical decision.
"""

from __future__ import annotations

import json
import re
from calendar import monthrange
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias, cast

from openmed.core.audit import hash_text, stable_hash

TemporalStatus: TypeAlias = Literal[
    "recent",
    "historical",
    "hypothetical",
    "future",
    "unknown",
]
SpanOffset: TypeAlias = tuple[int, int]

NLI_TEMPORAL_PAIR_SCHEMA_VERSION: Final = 1
NLI_TEMPORAL_PAIR_ADVISORY: Final = (
    "Temporality-aware clinical NLI pairs are deterministic assistive inputs "
    "for qualified human review. Incompatible or unresolved temporal evidence "
    "is never treated as entailment or an autonomous clinical decision."
)
CLINICAL_NLI_TEMPORAL_PAIR_ADVISORY: Final = NLI_TEMPORAL_PAIR_ADVISORY

TEMPORAL_STATUS_VALUES: Final = (
    "recent",
    "historical",
    "hypothetical",
    "future",
    "unknown",
)
TEMPORAL_COMPATIBILITY_VALUES: Final = (
    "compatible",
    "incompatible",
    "unresolved",
)
TEMPORAL_NLI_LABELS: Final = (
    "entailment",
    "contradiction",
    "neutral",
    "abstention",
    "review_required",
)
_SAFE_TEMPORAL_PRECISIONS = frozenset(
    {
        "unknown",
        "year",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "part_of_day",
        "part_of_month",
        "interval",
    }
)
_SAFE_TEMPORAL_SOURCES = frozenset(
    {"supplied", "interval", "context", "timeline", "reference", "inferred"}
)

_MISSING = object()
_TEXT_KEYS = (
    "text",
    "surface",
    "content",
    "claim",
    "evidence",
    "premise",
    "hypothesis",
)
_OFFSET_KEYS = ("offset", "span", "source_offset", "source_span")
_START_KEYS = ("start", "source_start", "start_offset", "begin")
_END_KEYS = ("end", "source_end", "end_offset", "stop")
_INTERVAL_KEYS = (
    "interval",
    "temporal_interval",
    "event_interval",
    "normalized_interval",
    "time_interval",
)
_TIME_KEYS = (
    "normalized_time",
    "normalized_value",
    "event_time",
    "time",
    "timex",
    "temporal_expression",
)
_STATUS_KEYS = (
    "temporal_status",
    "temporality",
    "temporal_state",
    "status",
)
_TEMPORAL_CONTAINER_KEYS = (
    "temporal",
    "temporal_metadata",
    "temporal_context",
    "temporal_assertion",
    "clinical_assertion",
    "clinical_context",
    "context",
    "metadata",
    "assertion_metadata",
    "assertion",
    "timeline_event",
)
_REFERENCE_KEYS = ("reference_date", "reference_time", "document_date")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_ISO_YEAR_RE = re.compile(r"^\d{4}$")

_STATUS_ALIASES: Mapping[str, TemporalStatus] = MappingProxyType(
    {
        "recent": "recent",
        "current": "recent",
        "present": "recent",
        "active": "recent",
        "now": "recent",
        "historical": "historical",
        "history": "historical",
        "past": "historical",
        "previous": "historical",
        "resolved": "historical",
        "hypothetical": "hypothetical",
        "conditional": "hypothetical",
        "possible": "hypothetical",
        "future": "future",
        "planned": "future",
        "anticipated": "future",
        "scheduled": "future",
        "unknown": "unknown",
        "unresolved": "unknown",
        "indeterminate": "unknown",
        "unspecified": "unknown",
    }
)


class TemporalCompatibility(str, Enum):
    """Controlled result of comparing two temporal assertions."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNRESOLVED = "unresolved"


class NliTemporalPairError(ValueError):
    """Base error for malformed temporality-aware pair inputs."""


class NliTemporalPairValidationError(NliTemporalPairError):
    """Raised when pair text, offsets, labels, or interval data is invalid."""


class InconsistentTemporalMetadataError(NliTemporalPairError):
    """Raised when redundant temporal fields disagree."""


def _invalid(field_name: str) -> NliTemporalPairValidationError:
    """Build a value-free validation error."""

    return NliTemporalPairValidationError(f"{field_name} is invalid")


def _missing(field_name: str) -> NliTemporalPairValidationError:
    """Build a value-free missing-field error."""

    return NliTemporalPairValidationError(f"{field_name} is required")


def _inconsistent(field_name: str) -> InconsistentTemporalMetadataError:
    """Build a value-free inconsistency error."""

    return InconsistentTemporalMetadataError(f"{field_name} is inconsistent")


def _mapping_from_object(value: object) -> Mapping[str, object] | None:
    """Return a mapping view without invoking arbitrary string conversion."""

    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None
    fields: dict[str, object] = {}
    for name in (
        *_TEXT_KEYS,
        *_OFFSET_KEYS,
        *_START_KEYS,
        *_END_KEYS,
        *_INTERVAL_KEYS,
        *_TIME_KEYS,
        *_STATUS_KEYS,
        *_TEMPORAL_CONTAINER_KEYS,
        "resolved",
        "is_resolved",
        "lower_bound",
        "upper_bound",
        "precision",
        "uncertainty_days",
        "reference_date",
        "reference_time",
        "document_date",
        "value",
        "granularity_flags",
        "timex_type",
    ):
        try:
            candidate = getattr(value, name, _MISSING)
        except Exception:
            continue
        if candidate is not _MISSING:
            fields[name] = candidate
    return fields or None


def _first_value(source: Mapping[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        if key in source:
            return source[key]
    return _MISSING


def _coerce_date(value: object, *, field_name: str) -> date:
    """Parse an ISO date or datetime without exposing the supplied value."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if type(value) is not str:
        raise _invalid(field_name)
    candidate = value.strip()
    if not candidate:
        raise _invalid(field_name)
    try:
        if _ISO_DATE_RE.fullmatch(candidate):
            return date.fromisoformat(candidate)
        normalized = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
        return datetime.fromisoformat(normalized).date()
    except (TypeError, ValueError, OverflowError):
        raise _invalid(field_name) from None


def _coerce_reference_date(value: object) -> date | None:
    if value is None:
        return None
    return _coerce_date(value, field_name="reference date")


def _mapping_reference_date(
    mapping: Mapping[str, object],
    fallback: date | None,
    *,
    field_name: str,
) -> date | None:
    """Resolve an optional record-level reference date without using a clock."""

    values = [mapping[key] for key in _REFERENCE_KEYS if key in mapping]
    if not values:
        return fallback
    normalized = tuple(_coerce_reference_date(value) for value in values)
    if len(set(normalized)) > 1:
        raise _inconsistent(f"{field_name} reference date")
    local = normalized[0]
    if local is not None and fallback is not None and local != fallback:
        raise _inconsistent(f"{field_name} reference date")
    return fallback if local is None else local


def _resolve_reference_alias(
    reference_time: object,
    reference_date: object,
) -> object:
    """Resolve equivalent reference-time aliases without silent disagreement."""

    if reference_date is _MISSING:
        return reference_time
    if reference_time is None:
        return reference_date
    normalized_time = _coerce_reference_date(reference_time)
    normalized_date = _coerce_reference_date(reference_date)
    if normalized_time != normalized_date:
        raise _inconsistent("reference date")
    return reference_date


def _coerce_offset(value: object, *, field_name: str) -> SpanOffset:
    """Validate a non-empty half-open source span."""

    if isinstance(value, Mapping):
        start = _first_value(cast(Mapping[str, object], value), _START_KEYS)
        end = _first_value(cast(Mapping[str, object], value), _END_KEYS)
        if start is _MISSING or end is _MISSING:
            raise _invalid(field_name)
        value = (start, end)
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise _invalid(field_name)
    start, end = value
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise _invalid(field_name)
    return start, end


def _coerce_text(
    value: object, *, side_name: str
) -> tuple[str, Mapping[str, object] | None]:
    if type(value) is str:
        if not value.strip():
            raise _missing(f"{side_name} text")
        return value, None
    mapping = _mapping_from_object(value)
    if mapping is None:
        raise _invalid(f"{side_name} text")
    raw_text = _first_value(mapping, _TEXT_KEYS)
    if raw_text is _MISSING:
        raise _missing(f"{side_name} text")
    if type(raw_text) is not str or not raw_text.strip():
        raise _invalid(f"{side_name} text")
    return raw_text, mapping


def _offset_from_mapping(
    mapping: Mapping[str, object] | None,
    *,
    side_name: str,
) -> SpanOffset | None:
    if mapping is None:
        return None
    candidates: list[SpanOffset] = []
    for key in _OFFSET_KEYS:
        value = mapping.get(key, _MISSING)
        if value is not _MISSING:
            candidates.append(_coerce_offset(value, field_name=f"{side_name} offset"))
    start = _first_value(mapping, _START_KEYS)
    end = _first_value(mapping, _END_KEYS)
    if start is not _MISSING or end is not _MISSING:
        if start is _MISSING or end is _MISSING:
            raise _invalid(f"{side_name} offset")
        # A temporal mapping may use date-like ``start``/``end`` endpoints.
        # Those belong to the interval, not to the source-text span.
        if not (_looks_like_date_endpoint(start) and _looks_like_date_endpoint(end)):
            candidates.append(
                _coerce_offset((start, end), field_name=f"{side_name} offset")
            )
    if not candidates:
        return None
    if len(set(candidates)) != 1:
        raise _inconsistent(f"{side_name} offset")
    return candidates[0]


def _normalize_status(
    value: object, *, field_name: str = "temporal status"
) -> TemporalStatus:
    if value is None:
        return "unknown"
    if isinstance(value, Enum):
        value = value.value
    if type(value) is not str:
        raise _invalid(field_name)
    normalized = " ".join(value.strip().casefold().split())
    status = _STATUS_ALIASES.get(normalized)
    if status is None:
        raise _invalid(field_name)
    return status


def _normalize_label(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid("NLI label")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = {"entail": "entailment", "abstain": "abstention"}.get(
        normalized,
        normalized,
    )
    if normalized not in TEMPORAL_NLI_LABELS:
        raise _invalid("NLI label")
    return normalized


def _normalize_score(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid("NLI score")
    if not 0.0 <= value <= 1.0:
        raise _invalid("NLI score")
    normalized = float(value)
    if normalized != normalized or normalized in (float("inf"), float("-inf")):
        raise _invalid("NLI score")
    if not 0.0 <= normalized <= 1.0:
        raise _invalid("NLI score")
    return normalized


def _days_in_month(year: int, month: int) -> int:
    return monthrange(year, month)[1]


def _interval_from_value(
    value: str,
    *,
    field_name: str,
    flags: Iterable[object] = (),
) -> tuple[date, date] | None:
    """Parse a normalized ISO value into inclusive date endpoints."""

    candidate = value.strip()
    if not candidate:
        raise _invalid(field_name)
    if "/" in candidate:
        parts = candidate.split("/")
        if len(parts) != 2 or not all(parts):
            return None
        left, right = parts
        if left.casefold() in {
            "present",
            "now",
            "unknown",
            "unresolved",
        } or right.casefold() in {
            "present",
            "now",
            "unknown",
            "unresolved",
        }:
            return None
        if left.startswith("R") or right.startswith("P"):
            return None
        start = _coerce_date(left, field_name=field_name)
        end = _coerce_date(right, field_name=field_name)
        return start, end
    try:
        if _ISO_DATE_RE.fullmatch(candidate):
            point = date.fromisoformat(candidate)
            return point, point
        if _ISO_MONTH_RE.fullmatch(candidate):
            year, month = (int(part) for part in candidate.split("-"))
            start = date(year, month, 1)
            return start, date(year, month, _days_in_month(year, month))
        if _ISO_YEAR_RE.fullmatch(candidate):
            year = int(candidate)
            return date(year, 1, 1), date(year, 12, 31)
        if re.match(r"^\d{4}-\d{2}-\d{2}[T ]", candidate):
            point = _coerce_date(candidate, field_name=field_name)
            return point, point
    except (TypeError, ValueError, OverflowError):
        raise _invalid(field_name) from None
    # TIMEX duration/set values do not describe a single event interval.
    if candidate.startswith(("P", "R")):
        return None
    if _flags_unresolved(flags):
        return None
    return None


def _precision_for_value(value: str) -> str:
    """Return the coarse precision implied by a normalized value."""

    candidate = value.strip()
    if _ISO_YEAR_RE.fullmatch(candidate):
        return "year"
    if _ISO_MONTH_RE.fullmatch(candidate):
        return "month"
    if "/" in candidate:
        return "day"
    return "day"


def _timex_value_text(value: object, *, field_name: str) -> str:
    try:
        return value if isinstance(value, str) else str(value)
    except Exception:
        raise _invalid(field_name) from None


def _normalized_timex_interval(
    value: object,
    *,
    reference_date: date | None,
    field_name: str,
) -> "TemporalInterval":
    """Convert an existing TIMEX-like object or expression to an interval."""

    if isinstance(value, TemporalInterval):
        return value
    mapping = _mapping_from_object(value)
    if mapping is not None:
        raw_interval = _first_value(mapping, _INTERVAL_KEYS)
        if raw_interval is not _MISSING and raw_interval is not value:
            return _coerce_interval(
                raw_interval, reference_date=reference_date, field_name=field_name
            )

        raw_start = mapping.get("start", _MISSING)
        raw_end = mapping.get("end", _MISSING)
        # NormalizedTimex and timeline records also expose source offsets as
        # ``start``/``end``.  Treat those as interval endpoints only when they
        # are date-like; otherwise the normalized ``value`` below is primary.
        date_endpoints = (
            raw_start is not _MISSING
            and raw_end is not _MISSING
            and _looks_like_date_endpoint(raw_start)
            and _looks_like_date_endpoint(raw_end)
        )
        if date_endpoints:
            lower = mapping.get("lower_bound")
            upper = mapping.get("upper_bound")
            resolved_value = mapping.get("resolved", mapping.get("is_resolved", True))
            flags = mapping.get("granularity_flags", ())
            if type(resolved_value) is not bool:
                raise _invalid(field_name)
            return TemporalInterval(
                start=raw_start,
                end=raw_end,
                lower_bound=lower,
                upper_bound=upper,
                precision=mapping.get("precision", "day"),
                resolved=resolved_value and not _flags_unresolved(flags),
            )
        if (raw_start is not _MISSING or raw_end is not _MISSING) and (
            raw_start is None
            or raw_end is None
            or raw_start is _MISSING
            or raw_end is _MISSING
        ):
            return TemporalInterval.unresolved()
        raw_value = _first_value(mapping, ("value", "normalized_value"))
        if raw_value is not _MISSING:
            if raw_value is None:
                return TemporalInterval.unresolved(
                    precision=(
                        mapping.get("precision", "unknown")
                        if isinstance(mapping.get("precision", "unknown"), str)
                        else "unknown"
                    )
                )
            flags = mapping.get("granularity_flags", ())
            resolved_value = mapping.get("resolved", mapping.get("is_resolved", True))
            if type(resolved_value) is not bool:
                raise _invalid(field_name)
            parsed = _interval_from_value(
                _timex_value_text(raw_value, field_name=field_name),
                field_name=field_name,
                flags=flags
                if isinstance(flags, Iterable) and not isinstance(flags, (str, bytes))
                else (),
            )
            if parsed is None:
                if isinstance(raw_value, str) and not _flags_unresolved(flags):
                    return _normalized_timex_interval(
                        raw_value,
                        reference_date=reference_date,
                        field_name=field_name,
                    )
                return TemporalInterval.unresolved()
            return TemporalInterval(
                start=parsed[0],
                end=parsed[1],
                lower_bound=mapping.get("lower_bound"),
                upper_bound=mapping.get("upper_bound"),
                precision=mapping.get(
                    "precision",
                    _precision_for_value(
                        _timex_value_text(raw_value, field_name=field_name)
                    ),
                ),
                resolved=(resolved_value and not _flags_unresolved(flags)),
            )
    if isinstance(value, (date, datetime)):
        point = _coerce_date(value, field_name=field_name)
        return TemporalInterval(start=point, end=point)
    if type(value) is not str:
        raise _invalid(field_name)
    expression = value.strip()
    parsed = _interval_from_value(expression, field_name=field_name)
    if parsed is not None:
        return TemporalInterval(
            start=parsed[0],
            end=parsed[1],
            precision=_precision_for_value(expression),
        )

    # Relative expressions are normalized through the existing deterministic
    # rules engine.  With no reference date they remain explicitly unresolved.
    from openmed.clinical.temporal_normalizer import normalize_temporal

    try:
        record = normalize_temporal(
            expression,
            [(0, len(expression))],
            reference_date,
        )[0]
    except Exception:
        raise _invalid(field_name) from None
    if record.value is None:
        return TemporalInterval.unresolved(
            precision=(
                record.granularity_flags[0] if record.granularity_flags else "unknown"
            )
        )
    parsed = _interval_from_value(
        record.value,
        field_name=field_name,
        flags=record.granularity_flags,
    )
    if parsed is None:
        return TemporalInterval.unresolved(
            precision=record.granularity_flags[0]
            if record.granularity_flags
            else "unknown"
        )
    return TemporalInterval(
        start=parsed[0],
        end=parsed[1],
        precision=record.granularity_flags[0] if record.granularity_flags else "day",
        resolved=not _flags_unresolved(record.granularity_flags),
    )


def _flags_unresolved(flags: object) -> bool:
    if isinstance(flags, str):
        flags = (flags,)
    if not isinstance(flags, Iterable):
        return False
    try:
        return any(
            str(flag).casefold()
            in {"ambiguous", "unanchored", "approximate", "uncertain"}
            for flag in flags
        )
    except Exception:
        raise _invalid("interval flags") from None


def _coerce_interval(
    value: object,
    *,
    reference_date: date | None,
    field_name: str,
) -> "TemporalInterval":
    if value is None:
        return TemporalInterval.unresolved()
    if isinstance(value, TemporalInterval):
        return value
    mapping = _mapping_from_object(value)
    if mapping is not None:
        reference_date = _mapping_reference_date(
            mapping,
            reference_date,
            field_name=field_name,
        )
        return _normalized_timex_interval(
            value,
            reference_date=reference_date,
            field_name=field_name,
        )
    return _normalized_timex_interval(
        value,
        reference_date=reference_date,
        field_name=field_name,
    )


@dataclass(frozen=True, slots=True, repr=False)
class TemporalInterval:
    """A normalized inclusive event interval with explicit resolution state.

    ``start`` and ``end`` are normalized ISO-compatible dates.  ``lower_bound``
    and ``upper_bound`` retain conservative uncertainty bounds when an upstream
    timeline resolver supplies them.  An interval with missing endpoints or
    ``resolved=False`` is intentionally unresolved and cannot support
    entailment.
    """

    start: date | str | datetime | None = None
    end: date | str | datetime | None = None
    lower_bound: date | str | datetime | None = None
    upper_bound: date | str | datetime | None = None
    precision: str = "day"
    resolved: bool = True

    def __post_init__(self) -> None:
        start = (
            None
            if self.start is None
            else _coerce_date(self.start, field_name="interval start")
        )
        end = (
            None
            if self.end is None
            else _coerce_date(self.end, field_name="interval end")
        )
        if start is not None and end is not None and end < start:
            raise _invalid("temporal interval")
        if type(self.resolved) is not bool:
            raise _invalid("interval resolved")
        precision = self.precision
        if type(precision) is not str or not precision.strip():
            raise _invalid("interval precision")
        precision = precision.strip().casefold()
        if precision not in _SAFE_TEMPORAL_PRECISIONS:
            raise _invalid("interval precision")
        lower = (
            start
            if self.lower_bound is None and start is not None
            else None
            if self.lower_bound is None
            else _coerce_date(self.lower_bound, field_name="interval lower bound")
        )
        upper = (
            end
            if self.upper_bound is None and end is not None
            else None
            if self.upper_bound is None
            else _coerce_date(self.upper_bound, field_name="interval upper bound")
        )
        if lower is not None and upper is not None and upper < lower:
            raise _invalid("temporal interval bounds")
        if start is not None and lower is not None and lower > start:
            raise _invalid("temporal interval lower bound")
        if end is not None and upper is not None and upper < end:
            raise _invalid("temporal interval upper bound")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "precision", precision)
        if start is None or end is None:
            object.__setattr__(self, "resolved", False)

    @classmethod
    def unresolved(cls, *, precision: str = "unknown") -> "TemporalInterval":
        """Return an explicit unresolved interval marker."""

        return cls(None, None, precision=precision, resolved=False)

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        reference_time: object = None,
        reference_date: object = _MISSING,
    ) -> "TemporalInterval":
        """Normalize an interval, date, TIMEX expression, or timeline record."""

        anchor = _resolve_reference_alias(reference_time, reference_date)
        return _coerce_interval(
            value,
            reference_date=_coerce_reference_date(anchor),
            field_name="temporal interval",
        )

    @property
    def is_resolved(self) -> bool:
        """Return whether both endpoints are usable for compatibility checks."""

        return self.resolved and self.start is not None and self.end is not None

    @property
    def value(self) -> str | None:
        """Return the normalized date or date-range value."""

        if self.start is None or self.end is None:
            return None
        if self.start == self.end:
            return self.start.isoformat()
        return f"{self.start.isoformat()}/{self.end.isoformat()}"

    @property
    def normalized_value(self) -> str | None:
        """Return :attr:`value` under the timeline normalizer's name."""

        return self.value

    @property
    def iso_value(self) -> str | None:
        """Return the normalized interval under the timeline resolver's name."""

        return self.value

    @property
    def uncertainty_days(self) -> int:
        """Return the widest day uncertainty represented by the bounds."""

        if not self.is_resolved or self.start is None or self.end is None:
            return 0
        lower_delta = (
            (self.start - self.lower_bound).days if self.lower_bound is not None else 0
        )
        upper_delta = (
            (self.upper_bound - self.end).days if self.upper_bound is not None else 0
        )
        return max(lower_delta, upper_delta)

    @property
    def possible_start(self) -> date | None:
        """Return the earliest possible start under interval uncertainty."""

        return self.lower_bound if self.is_resolved else None

    @property
    def possible_end(self) -> date | None:
        """Return the latest possible end under interval uncertainty."""

        return self.upper_bound if self.is_resolved else None

    def to_dict(self) -> dict[str, Any]:
        """Return normalized interval metadata without source text."""

        return {
            "start": self.start.isoformat() if self.start is not None else None,
            "end": self.end.isoformat() if self.end is not None else None,
            "value": self.value,
            "lower_bound": (
                self.lower_bound.isoformat() if self.lower_bound is not None else None
            ),
            "upper_bound": (
                self.upper_bound.isoformat() if self.upper_bound is not None else None
            ),
            "precision": self.precision,
            "resolved": self.is_resolved,
        }

    def __repr__(self) -> str:
        return (
            "TemporalInterval("
            f"resolved={self.is_resolved!r}, precision={self.precision!r}, "
            f"value_hash={hash_text(self.value or '')!r})"
        )


NormalizedEventInterval = TemporalInterval
EventTemporalInterval = TemporalInterval


def _infer_status(
    interval: TemporalInterval, reference_date: date | None
) -> TemporalStatus:
    if not interval.is_resolved or reference_date is None:
        return "unknown"
    if interval.end is not None and interval.end < reference_date:
        return "historical"
    if interval.start is not None and interval.start > reference_date:
        return "future"
    return "recent"


def _temporal_fields(mapping: Mapping[str, object]) -> bool:
    return bool(
        any(key in mapping for key in (*_INTERVAL_KEYS, *_TIME_KEYS, *_STATUS_KEYS))
        or any(key in mapping for key in ("value", "normalized_value"))
    )


def _looks_like_date_endpoint(value: object) -> bool:
    """Return whether a start/end value can be a normalized date endpoint."""

    if isinstance(value, (date, datetime)):
        return True
    if type(value) is not str:
        return False
    candidate = value.strip()
    return bool(
        _ISO_YEAR_RE.fullmatch(candidate)
        or _ISO_MONTH_RE.fullmatch(candidate)
        or _ISO_DATE_RE.fullmatch(candidate)
        or re.match(r"^\d{4}-\d{2}-\d{2}[T ]", candidate)
    )


def _metadata_sources(value: object) -> tuple[object, ...]:
    mapping = _mapping_from_object(value)
    if mapping is None:
        return (value,)
    sources: list[object] = []
    for key in _TEMPORAL_CONTAINER_KEYS:
        candidate = mapping.get(key, _MISSING)
        if candidate is not _MISSING:
            sources.append(candidate)
    if _temporal_fields(mapping):
        sources.append(mapping)
    return tuple(sources) or (mapping,)


def _contains_explicit_status(value: object) -> bool:
    """Return whether a record explicitly supplied a temporal status."""

    mapping = _mapping_from_object(value)
    if mapping is None:
        return False
    if any(key in mapping for key in _STATUS_KEYS):
        return True
    return any(
        _contains_explicit_status(mapping[key])
        for key in _TEMPORAL_CONTAINER_KEYS
        if key in mapping
    )


def _has_direct_status(value: object) -> bool:
    """Return whether a mapping carries a status at its current level."""

    mapping = _mapping_from_object(value)
    return mapping is not None and any(key in mapping for key in _STATUS_KEYS)


def _explicit_status_values(
    value: object,
    *,
    field_name: str,
) -> tuple[TemporalStatus, ...]:
    """Collect direct status values from a record and its temporal containers."""

    statuses: list[TemporalStatus] = []
    for source in _metadata_sources(value):
        statuses.extend(_direct_status_values(source, field_name=field_name))
    return tuple(statuses)


def _direct_status_values(
    value: object,
    *,
    field_name: str,
) -> tuple[TemporalStatus, ...]:
    """Normalize all status aliases present on one mapping level."""

    mapping = _mapping_from_object(value)
    if mapping is None:
        return ()
    return tuple(
        _normalize_status(mapping[key], field_name=f"{field_name} status")
        for key in _STATUS_KEYS
        if key in mapping
    )


def _metadata_from_source(
    value: object,
    *,
    reference_date: date | None,
    field_name: str,
    infer_status: bool = True,
) -> tuple[TemporalInterval, TemporalStatus, str, bool] | None:
    if isinstance(value, TemporalMetadata):
        return value.interval, value.status, value.source, True
    mapping = _mapping_from_object(value)
    if mapping is None:
        if isinstance(value, (TemporalInterval, date, datetime, str)):
            interval = _coerce_interval(
                value,
                reference_date=reference_date,
                field_name=field_name,
            )
            return (
                interval,
                _infer_status(interval, reference_date) if infer_status else "unknown",
                "interval",
                True,
            )
        return None
    reference_date = _mapping_reference_date(
        mapping,
        reference_date,
        field_name=field_name,
    )

    raw_interval_values = [
        mapping[key]
        for key in (*_INTERVAL_KEYS, *_TIME_KEYS, "value", "normalized_value")
        if key in mapping
    ]
    if not raw_interval_values and (
        "start" in mapping
        and "end" in mapping
        and _looks_like_date_endpoint(mapping["start"])
        and _looks_like_date_endpoint(mapping["end"])
    ):
        raw_interval_values.append(mapping)
    interval = TemporalInterval.unresolved()
    resolution_states: set[bool] = set()
    for raw_interval in raw_interval_values:
        candidate_interval = _coerce_interval(
            raw_interval,
            reference_date=reference_date,
            field_name=field_name,
        )
        if raw_interval is not None:
            resolution_states.add(candidate_interval.is_resolved)
        if len(resolution_states) > 1:
            raise _inconsistent(field_name)
        if not candidate_interval.is_resolved:
            continue
        if interval.is_resolved and interval != candidate_interval:
            raise _inconsistent(field_name)
        interval = candidate_interval
    status_values = _direct_status_values(mapping, field_name=field_name)
    if len(set(status_values)) > 1:
        raise _inconsistent(f"{field_name} status")
    status = (
        status_values[0]
        if status_values
        else _infer_status(interval, reference_date)
        if infer_status
        else "unknown"
    )
    source = mapping.get("source", "supplied")
    if type(source) is not str or not source.strip():
        source = "supplied"
    normalized_source = source.strip().casefold()
    if normalized_source not in _SAFE_TEMPORAL_SOURCES:
        normalized_source = "supplied"
    return (
        interval,
        status,
        normalized_source,
        any(value is not None for value in raw_interval_values),
    )


@dataclass(frozen=True, slots=True, repr=False)
class TemporalMetadata:
    """Validated temporal evidence attached to one NLI pair side."""

    interval: TemporalInterval | None = None
    status: TemporalStatus | str = "unknown"
    source: str = "supplied"

    def __post_init__(self) -> None:
        interval = (
            TemporalInterval.unresolved()
            if self.interval is None
            else _coerce_interval(
                self.interval,
                reference_date=None,
                field_name="temporal interval",
            )
        )
        status = _normalize_status(self.status)
        source = self.source
        if type(source) is not str or not source.strip():
            raise _invalid("temporal metadata source")
        source = source.strip().casefold()
        if source not in _SAFE_TEMPORAL_SOURCES:
            raise _invalid("temporal metadata source")
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "source", source)

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        reference_time: object = None,
        reference_date: object = _MISSING,
        field_name: str = "temporal metadata",
    ) -> "TemporalMetadata":
        """Coerce a temporal record, assertion, interval, or expression."""

        if isinstance(value, cls):
            return value
        anchor = _resolve_reference_alias(reference_time, reference_date)
        reference_date_value = _coerce_reference_date(anchor)
        sources = _metadata_sources(value)
        status_explicit = any(_contains_explicit_status(source) for source in sources)
        parsed = [
            result
            for source in sources
            if (
                result := _metadata_from_source(
                    source,
                    reference_date=reference_date_value,
                    field_name=field_name,
                    infer_status=(not status_explicit or _has_direct_status(source)),
                )
            )
            is not None
        ]
        if not parsed:
            return cls()
        interval = TemporalInterval.unresolved()
        status: TemporalStatus = "unknown"
        source = "supplied"
        explicit_statuses = tuple(
            status_value
            for source in sources
            for status_value in _explicit_status_values(
                source,
                field_name=field_name,
            )
        )
        if len(set(explicit_statuses)) > 1:
            raise _inconsistent(f"{field_name} status")
        explicit_status: TemporalStatus | None = (
            explicit_statuses[0] if explicit_statuses else None
        )
        resolution_states: set[bool] = set()
        for (
            candidate_interval,
            candidate_status,
            candidate_source,
            has_interval,
        ) in parsed:
            if has_interval:
                resolution_states.add(candidate_interval.is_resolved)
            if len(resolution_states) > 1:
                raise _inconsistent(field_name)
            if candidate_interval.is_resolved:
                if interval.is_resolved and interval != candidate_interval:
                    raise _inconsistent(field_name)
                interval = candidate_interval
            if candidate_status != "unknown":
                if status != "unknown" and status != candidate_status:
                    raise _inconsistent(f"{field_name} status")
                status = candidate_status
            if candidate_source != "supplied":
                source = candidate_source
        if explicit_status is not None:
            status = explicit_status
        elif status == "unknown" and not status_explicit:
            status = _infer_status(interval, reference_date_value)
        return cls(interval=interval, status=status, source=source)

    @property
    def temporal_status(self) -> TemporalStatus:
        """Return the status under the explicit NLI field name."""

        return cast(TemporalStatus, self.status)

    @property
    def temporality(self) -> TemporalStatus:
        """Return the status under the existing ConText vocabulary."""

        return self.temporal_status

    @property
    def resolved(self) -> bool:
        """Return whether the attached interval is resolved."""

        return self.interval.is_resolved

    @property
    def normalized_interval(self) -> TemporalInterval:
        """Return the normalized interval under an explicit alias."""

        return self.interval

    @property
    def temporal_interval(self) -> TemporalInterval:
        """Return the interval under the field name used by event records."""

        return self.interval

    @property
    def event_interval(self) -> TemporalInterval:
        """Return the normalized event interval."""

        return self.interval

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic normalized temporal metadata."""

        return {
            "temporal_status": self.temporal_status,
            "temporality": self.temporal_status,
            "interval": self.interval.to_dict(),
            "resolved": self.resolved,
            "source": self.source,
        }

    def __repr__(self) -> str:
        return (
            "TemporalMetadata("
            f"status={self.temporal_status!r}, resolved={self.resolved!r}, "
            f"source={self.source!r})"
        )


NliTemporalMetadata = TemporalMetadata
ClinicalNliTemporalMetadata = TemporalMetadata
EventTemporalMetadata = TemporalMetadata


@dataclass(frozen=True, slots=True)
class TemporalComparison:
    """Safe explanation of temporal compatibility between two pair sides."""

    status: TemporalCompatibility
    reason: str
    nominal_overlap: bool | None = None
    possible_overlap: bool | None = None

    @property
    def compatible(self) -> bool:
        """Return whether temporal evidence permits an entailment label."""

        return self.status is TemporalCompatibility.COMPATIBLE

    @property
    def review_required(self) -> bool:
        """Return whether a reviewer must inspect temporal evidence."""

        return not self.compatible

    @property
    def requires_review(self) -> bool:
        """Return :attr:`review_required` under a concise alias."""

        return self.review_required

    @property
    def classification(self) -> str:
        """Return the compatibility state under a classifier naming alias."""

        return self.status.value

    def to_dict(self) -> dict[str, Any]:
        """Return controlled comparison metadata."""

        return {
            "status": self.status.value,
            "reason": self.reason,
            "nominal_overlap": self.nominal_overlap,
            "possible_overlap": self.possible_overlap,
        }


TemporalIntervalComparison = TemporalComparison


def compare_temporal_metadata(
    premise: TemporalMetadata | object,
    hypothesis: TemporalMetadata | object,
    *,
    reference_time: object = None,
    reference_date: object = _MISSING,
) -> TemporalComparison:
    """Compare status and interval evidence conservatively.

    Unknown statuses, missing endpoints, and unanchored/ambiguous intervals are
    unresolved.  Different statuses, hypothetical assertions, and disjoint
    intervals are incompatible.  Only same-status intervals with certain
    nominal overlap are compatible with entailment.
    """

    anchor = _resolve_reference_alias(reference_time, reference_date)
    premise_metadata = (
        premise
        if isinstance(premise, TemporalMetadata)
        else TemporalMetadata.from_value(premise, reference_time=anchor)
    )
    hypothesis_metadata = (
        hypothesis
        if isinstance(hypothesis, TemporalMetadata)
        else TemporalMetadata.from_value(hypothesis, reference_time=anchor)
    )
    premise_status = premise_metadata.temporal_status
    hypothesis_status = hypothesis_metadata.temporal_status
    if "unknown" in {premise_status, hypothesis_status}:
        return TemporalComparison(TemporalCompatibility.UNRESOLVED, "status_unresolved")
    if "hypothetical" in {premise_status, hypothesis_status}:
        return TemporalComparison(
            TemporalCompatibility.INCOMPATIBLE, "hypothetical_status"
        )
    if premise_status != hypothesis_status:
        return TemporalComparison(TemporalCompatibility.INCOMPATIBLE, "status_mismatch")

    left = premise_metadata.interval
    right = hypothesis_metadata.interval
    if not left.is_resolved or not right.is_resolved:
        return TemporalComparison(
            TemporalCompatibility.UNRESOLVED, "interval_unresolved"
        )
    assert left.start is not None and left.end is not None
    assert right.start is not None and right.end is not None
    nominal_overlap = left.start <= right.end and right.start <= left.end
    left_start = left.possible_start or left.start
    left_end = left.possible_end or left.end
    right_start = right.possible_start or right.start
    right_end = right.possible_end or right.end
    possible_overlap = left_start <= right_end and right_start <= left_end
    if nominal_overlap:
        return TemporalComparison(
            TemporalCompatibility.COMPATIBLE,
            "interval_overlap",
            nominal_overlap=True,
            possible_overlap=possible_overlap,
        )
    if possible_overlap:
        return TemporalComparison(
            TemporalCompatibility.UNRESOLVED,
            "interval_uncertainty_overlap",
            nominal_overlap=False,
            possible_overlap=True,
        )
    return TemporalComparison(
        TemporalCompatibility.INCOMPATIBLE,
        "intervals_disjoint",
        nominal_overlap=False,
        possible_overlap=False,
    )


def classify_temporal_compatibility(
    premise: TemporalMetadata | object,
    hypothesis: TemporalMetadata | object,
    *,
    reference_time: object = None,
    reference_date: object = _MISSING,
) -> str:
    """Return ``compatible``, ``incompatible``, or ``unresolved``."""

    return compare_temporal_metadata(
        premise,
        hypothesis,
        reference_time=reference_time,
        reference_date=reference_date,
    ).status.value


def compare_temporal_intervals(
    premise: TemporalInterval | object,
    hypothesis: TemporalInterval | object,
) -> str:
    """Classify two intervals without making a status assumption.

    This lower-level helper reports interval overlap only.  Pair construction
    still requires explicit, matching temporal statuses through
    :func:`classify_temporal_compatibility`.
    """

    left = TemporalInterval.from_value(premise)
    right = TemporalInterval.from_value(hypothesis)
    if not left.is_resolved or not right.is_resolved:
        return TemporalCompatibility.UNRESOLVED.value
    assert left.start is not None and left.end is not None
    assert right.start is not None and right.end is not None
    nominal_overlap = left.start <= right.end and right.start <= left.end
    if nominal_overlap:
        return TemporalCompatibility.COMPATIBLE.value
    left_start = left.possible_start or left.start
    left_end = left.possible_end or left.end
    right_start = right.possible_start or right.start
    right_end = right.possible_end or right.end
    if left_start <= right_end and right_start <= left_end:
        return TemporalCompatibility.UNRESOLVED.value
    return TemporalCompatibility.INCOMPATIBLE.value


classify_temporal_intervals = compare_temporal_intervals


def temporal_intervals_compatible(
    premise: TemporalMetadata | object,
    hypothesis: TemporalMetadata | object,
) -> bool:
    """Return true only when temporal evidence is sufficiently compatible."""

    return classify_temporal_compatibility(premise, hypothesis) == "compatible"


def _merge_temporal_metadata(
    values: Iterable[object],
    *,
    reference_date: date | None,
    side_name: str,
) -> TemporalMetadata:
    values = tuple(values)
    status_explicit = any(_contains_explicit_status(value) for value in values)
    explicit_statuses = tuple(
        status_value
        for value in values
        if value is not _MISSING
        for status_value in _explicit_status_values(
            value,
            field_name=f"{side_name} temporal metadata",
        )
    )
    if len(set(explicit_statuses)) > 1:
        raise _inconsistent(f"{side_name} temporal status")
    parsed: list[TemporalMetadata] = []
    for value in values:
        if value is _MISSING:
            continue
        metadata = TemporalMetadata.from_value(
            value,
            reference_time=reference_date,
            field_name=f"{side_name} temporal metadata",
        )
        if status_explicit and not _has_direct_status(value):
            metadata = TemporalMetadata(
                interval=metadata.interval,
                status="unknown",
                source=metadata.source,
            )
        parsed.append(metadata)
    if not parsed:
        return TemporalMetadata()
    interval = TemporalInterval.unresolved()
    status: TemporalStatus = "unknown"
    source = "supplied"
    for metadata in parsed:
        if metadata.interval.is_resolved:
            if interval.is_resolved and interval != metadata.interval:
                raise _inconsistent(f"{side_name} temporal interval")
            interval = metadata.interval
        if metadata.status != "unknown":
            if status != "unknown" and status != metadata.status:
                raise _inconsistent(f"{side_name} temporal status")
            status = metadata.status
        if metadata.source != "supplied":
            source = metadata.source
    if explicit_statuses:
        status = explicit_statuses[0]
    elif status == "unknown" and not status_explicit:
        status = _infer_status(interval, reference_date)
    return TemporalMetadata(interval=interval, status=status, source=source)


def _side_temporal_metadata(
    side_value: object,
    side_mapping: Mapping[str, object] | None,
    *,
    explicit_metadata: object,
    explicit_interval: object,
    explicit_status: object,
    explicit_time: object,
    reference_date: date | None,
    side_name: str,
) -> TemporalMetadata:
    values: list[object] = []
    if explicit_metadata is not _MISSING:
        values.append(explicit_metadata)
    if explicit_interval is not _MISSING:
        values.append({"interval": explicit_interval})
    if explicit_time is not _MISSING:
        values.append({"interval": explicit_time})
    if explicit_status is not _MISSING:
        values.append({"temporal_status": explicit_status})
    if side_mapping is not None:
        values.extend(_metadata_sources(side_value))
    if not values and side_mapping is not None:
        values.append(side_value)
    metadata = (
        _merge_temporal_metadata(
            values,
            reference_date=reference_date,
            side_name=side_name,
        )
        if values
        else TemporalMetadata()
    )
    if (
        explicit_status is _MISSING
        and metadata.status == "unknown"
        and not _contains_explicit_status(side_value)
    ):
        # Reuse the committed ConText resolver when a caller supplied an
        # event surface but no explicit status.  Historical and hypothetical
        # cues are meaningful evidence; an unqualified span remains recent
        # only when no reference-date inference is available.
        try:
            from openmed.clinical.context import resolve_temporality

            inferred_status = _normalize_status(resolve_temporality(side_value))
        except Exception:
            inferred_status = "unknown"
        if inferred_status != "unknown":
            metadata = TemporalMetadata(
                interval=metadata.interval,
                status=inferred_status,
                source="context",
            )
    return metadata


def _alias_value(values: Sequence[object], *, field_name: str) -> object:
    present = [value for value in values if value is not _MISSING]
    if not present:
        return _MISSING
    first = present[0]
    if any(value != first for value in present[1:]):
        raise _inconsistent(field_name)
    return first


@dataclass(frozen=True, slots=True, repr=False)
class NliTemporalPair:
    """One NLI pair with temporal metadata attached to both sides.

    ``premise`` and ``hypothesis`` are available only through the explicit
    local model-input methods.  The regular serialized representation is
    value-free and records hashes, offsets, temporal intervals, and the
    conservative compatibility decision.
    """

    premise: str
    hypothesis: str
    premise_temporal: TemporalMetadata
    hypothesis_temporal: TemporalMetadata
    premise_offset: SpanOffset | None = None
    hypothesis_offset: SpanOffset | None = None
    predicted_label: str | None = None
    predicted_score: float | None = None
    schema_version: int = NLI_TEMPORAL_PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.premise) is not str or not self.premise.strip():
            raise _missing("premise text")
        if type(self.hypothesis) is not str or not self.hypothesis.strip():
            raise _missing("hypothesis text")
        if not isinstance(self.premise_temporal, TemporalMetadata):
            raise _invalid("premise temporal metadata")
        if not isinstance(self.hypothesis_temporal, TemporalMetadata):
            raise _invalid("hypothesis temporal metadata")
        if self.premise_offset is not None:
            object.__setattr__(
                self,
                "premise_offset",
                _coerce_offset(self.premise_offset, field_name="premise offset"),
            )
        if self.hypothesis_offset is not None:
            object.__setattr__(
                self,
                "hypothesis_offset",
                _coerce_offset(self.hypothesis_offset, field_name="hypothesis offset"),
            )
        object.__setattr__(
            self, "predicted_label", _normalize_label(self.predicted_label)
        )
        object.__setattr__(
            self, "predicted_score", _normalize_score(self.predicted_score)
        )
        if (
            type(self.schema_version) is not int
            or self.schema_version != NLI_TEMPORAL_PAIR_SCHEMA_VERSION
        ):
            raise _invalid("schema version")

    @property
    def premise_interval(self) -> TemporalInterval:
        """Return the premise event interval."""

        return self.premise_temporal.interval

    @property
    def hypothesis_interval(self) -> TemporalInterval:
        """Return the hypothesis event interval."""

        return self.hypothesis_temporal.interval

    @property
    def premise_temporal_interval(self) -> TemporalInterval:
        """Return the premise interval under its explicit temporal alias."""

        return self.premise_interval

    @property
    def hypothesis_temporal_interval(self) -> TemporalInterval:
        """Return the hypothesis interval under its explicit temporal alias."""

        return self.hypothesis_interval

    @property
    def premise_event_interval(self) -> TemporalInterval:
        """Return the premise interval under the event-record alias."""

        return self.premise_interval

    @property
    def hypothesis_event_interval(self) -> TemporalInterval:
        """Return the hypothesis interval under the event-record alias."""

        return self.hypothesis_interval

    @property
    def premise_temporal_metadata(self) -> TemporalMetadata:
        """Return the premise temporal sidecar."""

        return self.premise_temporal

    @property
    def hypothesis_temporal_metadata(self) -> TemporalMetadata:
        """Return the hypothesis temporal sidecar."""

        return self.hypothesis_temporal

    @property
    def premise_temporal_status(self) -> TemporalStatus:
        """Return the premise temporal status."""

        return self.premise_temporal.temporal_status

    @property
    def hypothesis_temporal_status(self) -> TemporalStatus:
        """Return the hypothesis temporal status."""

        return self.hypothesis_temporal.temporal_status

    @property
    def premise_temporality(self) -> TemporalStatus:
        """Return the premise status under the existing ConText name."""

        return self.premise_temporal_status

    @property
    def hypothesis_temporality(self) -> TemporalStatus:
        """Return the hypothesis status under the existing ConText name."""

        return self.hypothesis_temporal_status

    @property
    def premise_status(self) -> TemporalStatus:
        """Return the premise status under a concise field name."""

        return self.premise_temporal_status

    @property
    def hypothesis_status(self) -> TemporalStatus:
        """Return the hypothesis status under a concise field name."""

        return self.hypothesis_temporal_status

    @property
    def temporal_metadata(self) -> dict[str, dict[str, Any]]:
        """Return both sidecars without source text."""

        return {
            "premise": self.premise_temporal.to_dict(),
            "hypothesis": self.hypothesis_temporal.to_dict(),
        }

    @property
    def temporal_comparison(self) -> TemporalComparison:
        """Return the conservative temporal compatibility decision."""

        return compare_temporal_metadata(
            self.premise_temporal,
            self.hypothesis_temporal,
        )

    @property
    def temporal_compatibility(self) -> str:
        """Return the compatibility state as a stable string."""

        return self.temporal_comparison.status.value

    @property
    def temporal_relation(self) -> str:
        """Return an alias for :attr:`temporal_compatibility`."""

        return self.temporal_compatibility

    @property
    def review_required(self) -> bool:
        """Return whether temporal evidence requires clinician review."""

        return self.temporal_comparison.review_required

    @property
    def requires_clinician_review(self) -> bool:
        """Return the review flag used by clinical downstream consumers."""

        return self.review_required

    @property
    def requires_review(self) -> bool:
        """Return :attr:`review_required` under a concise alias."""

        return self.review_required

    @property
    def label(self) -> str | None:
        """Return the effective label after the temporal safety gate."""

        if self.review_required:
            return "review_required"
        return self.predicted_label

    @property
    def effective_label(self) -> str | None:
        """Return :attr:`label` under an explicit classifier naming alias."""

        return self.label

    @property
    def pair_id(self) -> str:
        """Return a deterministic identifier derived only from safe metadata."""

        return stable_hash(self._fingerprint_payload())

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "premise_hash": hash_text(self.premise),
            "hypothesis_hash": hash_text(self.hypothesis),
            "premise_temporal": self.premise_temporal.to_dict(),
            "hypothesis_temporal": self.hypothesis_temporal.to_dict(),
            "premise_offset": list(self.premise_offset)
            if self.premise_offset
            else None,
            "hypothesis_offset": list(self.hypothesis_offset)
            if self.hypothesis_offset
            else None,
            "predicted_label": self.predicted_label,
            "predicted_score": self.predicted_score,
        }

    def to_model_input(self) -> dict[str, Any]:
        """Return raw local model text with temporal sidecars."""

        payload: dict[str, Any] = {
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "premise_temporal": self.premise_temporal.to_dict(),
            "hypothesis_temporal": self.hypothesis_temporal.to_dict(),
        }
        if self.predicted_label is not None:
            payload["predicted_label"] = self.predicted_label
        if self.predicted_score is not None:
            payload["predicted_score"] = self.predicted_score
        return payload

    def to_text_pair(self) -> tuple[str, str]:
        """Return only the raw text tuple for a local backend adapter."""

        return self.premise, self.hypothesis

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic PHI-safe pair metadata."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "premise": {
                "text_hash": hash_text(self.premise),
                "text_length": len(self.premise),
                "offset": list(self.premise_offset) if self.premise_offset else None,
                "temporal": self.premise_temporal.to_dict(),
                "interval": self.premise_interval.to_dict(),
                "temporal_status": self.premise_temporal_status,
            },
            "hypothesis": {
                "text_hash": hash_text(self.hypothesis),
                "text_length": len(self.hypothesis),
                "offset": (
                    list(self.hypothesis_offset) if self.hypothesis_offset else None
                ),
                "temporal": self.hypothesis_temporal.to_dict(),
                "interval": self.hypothesis_interval.to_dict(),
                "temporal_status": self.hypothesis_temporal_status,
            },
            "temporal_metadata": self.temporal_metadata,
            "premise_interval": self.premise_interval.to_dict(),
            "hypothesis_interval": self.hypothesis_interval.to_dict(),
            "premise_temporal_status": self.premise_temporal_status,
            "hypothesis_temporal_status": self.hypothesis_temporal_status,
            "temporal_compatibility": self.temporal_compatibility,
            "temporal_reason": self.temporal_comparison.reason,
            "review_required": self.review_required,
            "requires_clinician_review": self.requires_clinician_review,
            "label": self.label,
            "predicted_label": self.predicted_label,
            "predicted_score": self.predicted_score,
            "advisory": NLI_TEMPORAL_PAIR_ADVISORY,
        }
        return payload

    def to_audit_dict(self) -> dict[str, Any]:
        """Return the PHI-safe representation under an audit-specific name."""

        return self.to_dict()

    def to_json(self) -> str:
        """Serialize the PHI-safe pair representation deterministically."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def __repr__(self) -> str:
        return (
            "NliTemporalPair("
            f"pair_id={self.pair_id!r}, "
            f"premise_hash={hash_text(self.premise)!r}, "
            f"hypothesis_hash={hash_text(self.hypothesis)!r}, "
            f"temporal_compatibility={self.temporal_compatibility!r}, "
            f"label={self.label!r})"
        )


TemporalNliPair = NliTemporalPair
TemporalityAwareNliPair = NliTemporalPair
TemporalAwareNliPair = NliTemporalPair
ClinicalNliTemporalPair = NliTemporalPair
ClinicalTemporalNliPair = NliTemporalPair


def build_temporal_nli_pair(
    premise: object,
    hypothesis: object,
    *,
    premise_temporal: object = _MISSING,
    hypothesis_temporal: object = _MISSING,
    premise_metadata: object = _MISSING,
    hypothesis_metadata: object = _MISSING,
    premise_interval: object = _MISSING,
    hypothesis_interval: object = _MISSING,
    premise_temporality: object = _MISSING,
    hypothesis_temporality: object = _MISSING,
    premise_temporal_status: object = _MISSING,
    hypothesis_temporal_status: object = _MISSING,
    premise_status: object = _MISSING,
    hypothesis_status: object = _MISSING,
    premise_temporal_interval: object = _MISSING,
    hypothesis_temporal_interval: object = _MISSING,
    premise_event_interval: object = _MISSING,
    hypothesis_event_interval: object = _MISSING,
    premise_time: object = _MISSING,
    hypothesis_time: object = _MISSING,
    premise_timex: object = _MISSING,
    hypothesis_timex: object = _MISSING,
    reference_time: object = None,
    reference_date: object = _MISSING,
    premise_offset: object = _MISSING,
    hypothesis_offset: object = _MISSING,
    predicted_label: object = _MISSING,
    nli_label: object = _MISSING,
    label: object = _MISSING,
    predicted_score: object = _MISSING,
    nli_score: object = _MISSING,
    score: object = _MISSING,
) -> NliTemporalPair:
    """Construct one temporality-aware clinical NLI pair.

    Pair sides may be strings plus explicit temporal keyword arguments, or
    span/event mappings carrying ``interval``/``normalized_time`` and
    ``temporal_status``/``temporality`` fields.  Relative expressions are
    normalized only when ``reference_time`` is supplied; otherwise they remain
    unresolved and the pair is marked for review.  Plain pair text is never
    interpreted as an interval implicitly.  A supplied ``entailment``
    label is downgraded to ``review_required`` whenever statuses differ,
    intervals are disjoint, or either side is unresolved.
    """

    premise_text, premise_mapping = _coerce_text(premise, side_name="premise")
    hypothesis_text, hypothesis_mapping = _coerce_text(
        hypothesis, side_name="hypothesis"
    )
    reference_value = _resolve_reference_alias(reference_time, reference_date)
    reference_date_value = _coerce_reference_date(reference_value)

    premise_metadata_value = _alias_value(
        (premise_temporal, premise_metadata),
        field_name="premise temporal metadata",
    )
    hypothesis_metadata_value = _alias_value(
        (hypothesis_temporal, hypothesis_metadata),
        field_name="hypothesis temporal metadata",
    )
    premise_status = _alias_value(
        (premise_temporality, premise_temporal_status, premise_status),
        field_name="premise temporal status",
    )
    hypothesis_status = _alias_value(
        (hypothesis_temporality, hypothesis_temporal_status, hypothesis_status),
        field_name="hypothesis temporal status",
    )
    premise_interval = _alias_value(
        (premise_interval, premise_temporal_interval, premise_event_interval),
        field_name="premise temporal interval",
    )
    hypothesis_interval = _alias_value(
        (hypothesis_interval, hypothesis_temporal_interval, hypothesis_event_interval),
        field_name="hypothesis temporal interval",
    )
    premise_time = _alias_value(
        (premise_time, premise_timex), field_name="premise time"
    )
    hypothesis_time = _alias_value(
        (hypothesis_time, hypothesis_timex), field_name="hypothesis time"
    )
    premise_metadata = _side_temporal_metadata(
        premise,
        premise_mapping,
        explicit_metadata=premise_metadata_value,
        explicit_interval=premise_interval,
        explicit_status=premise_status,
        explicit_time=premise_time,
        reference_date=reference_date_value,
        side_name="premise",
    )
    hypothesis_metadata = _side_temporal_metadata(
        hypothesis,
        hypothesis_mapping,
        explicit_metadata=hypothesis_metadata_value,
        explicit_interval=hypothesis_interval,
        explicit_status=hypothesis_status,
        explicit_time=hypothesis_time,
        reference_date=reference_date_value,
        side_name="hypothesis",
    )

    resolved_premise_offset = (
        _coerce_offset(premise_offset, field_name="premise offset")
        if premise_offset is not _MISSING
        else _offset_from_mapping(premise_mapping, side_name="premise")
    )
    resolved_hypothesis_offset = (
        _coerce_offset(hypothesis_offset, field_name="hypothesis offset")
        if hypothesis_offset is not _MISSING
        else _offset_from_mapping(hypothesis_mapping, side_name="hypothesis")
    )
    resolved_label = _alias_value(
        (predicted_label, nli_label, label),
        field_name="NLI label",
    )
    resolved_score = _alias_value(
        (predicted_score, nli_score, score),
        field_name="NLI score",
    )
    return NliTemporalPair(
        premise=premise_text,
        hypothesis=hypothesis_text,
        premise_temporal=premise_metadata,
        hypothesis_temporal=hypothesis_metadata,
        premise_offset=resolved_premise_offset
        if resolved_premise_offset is not _MISSING
        else None,
        hypothesis_offset=(
            resolved_hypothesis_offset
            if resolved_hypothesis_offset is not _MISSING
            else None
        ),
        predicted_label=None
        if resolved_label is _MISSING
        else cast(str | None, resolved_label),
        predicted_score=None
        if resolved_score is _MISSING
        else cast(float | None, resolved_score),
    )


def build_nli_temporal_pair(*args: object, **kwargs: object) -> NliTemporalPair:
    """Alias for :func:`build_temporal_nli_pair`."""

    return build_temporal_nli_pair(*args, **kwargs)


def build_temporality_aware_nli_pair(
    *args: object, **kwargs: object
) -> NliTemporalPair:
    """Alias using the issue's temporality-aware vocabulary."""

    return build_temporal_nli_pair(*args, **kwargs)


def construct_temporal_nli_pair(*args: object, **kwargs: object) -> NliTemporalPair:
    """Construction alias for callers using a verb-oriented API."""

    return build_temporal_nli_pair(*args, **kwargs)


def build_temporal_aware_pair(*args: object, **kwargs: object) -> NliTemporalPair:
    """Alias using the concise temporal-aware pair vocabulary."""

    return build_temporal_nli_pair(*args, **kwargs)


def build_temporal_aware_nli_pair(*args: object, **kwargs: object) -> NliTemporalPair:
    """Alias for temporal-aware NLI pair construction."""

    return build_temporal_nli_pair(*args, **kwargs)


def build_temporal_pair(*args: object, **kwargs: object) -> NliTemporalPair:
    """Alias for callers that omit the NLI suffix."""

    return build_temporal_nli_pair(*args, **kwargs)


def construct_temporal_aware_pair(*args: object, **kwargs: object) -> NliTemporalPair:
    """Construction alias using the issue's temporal-aware vocabulary."""

    return build_temporal_nli_pair(*args, **kwargs)


def build_nli_pair(*args: object, **kwargs: object) -> NliTemporalPair:
    """Module-local compatibility alias for temporal pair construction."""

    return build_temporal_nli_pair(*args, **kwargs)


def build_temporal_nli_pairs(
    pair_inputs: Iterable[object],
    **defaults: object,
) -> tuple[NliTemporalPair, ...]:
    """Build temporal pairs in input order without deduplication or I/O."""

    pairs: list[NliTemporalPair] = []
    for item in pair_inputs:
        if isinstance(item, NliTemporalPair):
            pairs.append(item)
            continue
        if isinstance(item, Mapping):
            premise = item.get("premise", _MISSING)
            hypothesis = item.get("hypothesis", _MISSING)
            if premise is _MISSING or hypothesis is _MISSING:
                raise _missing("pair premise and hypothesis")
            parameters = dict(defaults)
            for key in (
                "premise_temporal",
                "hypothesis_temporal",
                "premise_metadata",
                "hypothesis_metadata",
                "premise_interval",
                "hypothesis_interval",
                "premise_temporality",
                "hypothesis_temporality",
                "premise_temporal_status",
                "hypothesis_temporal_status",
                "premise_status",
                "hypothesis_status",
                "premise_temporal_interval",
                "hypothesis_temporal_interval",
                "premise_event_interval",
                "hypothesis_event_interval",
                "premise_time",
                "hypothesis_time",
                "premise_timex",
                "hypothesis_timex",
                "reference_time",
                "reference_date",
                "premise_offset",
                "hypothesis_offset",
                "predicted_label",
                "nli_label",
                "label",
                "predicted_score",
                "nli_score",
                "score",
            ):
                if key in item:
                    parameters[key] = item[key]
            pairs.append(build_temporal_nli_pair(premise, hypothesis, **parameters))
            continue
        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
            and len(item) == 2
        ):
            pairs.append(build_temporal_nli_pair(item[0], item[1], **defaults))
            continue
        raise _invalid("pair input")
    return tuple(pairs)


build_nli_temporal_pairs = build_temporal_nli_pairs
build_nli_pairs = build_temporal_nli_pairs
build_temporality_aware_nli_pairs = build_temporal_nli_pairs
construct_temporal_nli_pairs = build_temporal_nli_pairs
build_temporal_aware_pairs = build_temporal_nli_pairs
build_temporal_aware_nli_pairs = build_temporal_nli_pairs
construct_temporal_aware_pairs = build_temporal_nli_pairs
build_temporal_pairs = build_temporal_nli_pairs


def validate_temporal_nli_pair(value: object) -> NliTemporalPair:
    """Revalidate and return a temporal pair without exposing source text."""

    if not isinstance(value, NliTemporalPair):
        raise _invalid("NLI temporal pair")
    return NliTemporalPair(
        premise=value.premise,
        hypothesis=value.hypothesis,
        premise_temporal=TemporalMetadata.from_value(value.premise_temporal),
        hypothesis_temporal=TemporalMetadata.from_value(value.hypothesis_temporal),
        premise_offset=value.premise_offset,
        hypothesis_offset=value.hypothesis_offset,
        predicted_label=value.predicted_label,
        predicted_score=value.predicted_score,
        schema_version=value.schema_version,
    )


validate_nli_temporal_pair = validate_temporal_nli_pair
validate_temporal_pair = validate_temporal_nli_pair
validate_nli_pair = validate_temporal_nli_pair


def normalize_temporal_interval(
    value: object,
    *,
    reference_time: object = None,
    reference_date: object = _MISSING,
) -> TemporalInterval:
    """Normalize one event interval using an optional explicit anchor."""

    return TemporalInterval.from_value(
        value,
        reference_time=reference_time,
        reference_date=reference_date,
    )


def normalize_event_interval(
    value: object,
    *,
    reference_time: object = None,
    reference_date: object = _MISSING,
) -> TemporalInterval:
    """Alias for :func:`normalize_temporal_interval`."""

    return normalize_temporal_interval(
        value,
        reference_time=reference_time,
        reference_date=reference_date,
    )


def normalize_event_time_interval(
    value: object,
    *,
    reference_time: object = None,
    reference_date: object = _MISSING,
) -> TemporalInterval:
    """Alias for :func:`normalize_temporal_interval`."""

    return normalize_temporal_interval(
        value,
        reference_time=reference_time,
        reference_date=reference_date,
    )


__all__ = [
    "NLI_TEMPORAL_PAIR_SCHEMA_VERSION",
    "NLI_TEMPORAL_PAIR_ADVISORY",
    "CLINICAL_NLI_TEMPORAL_PAIR_ADVISORY",
    "TEMPORAL_STATUS_VALUES",
    "TEMPORAL_COMPATIBILITY_VALUES",
    "TEMPORAL_NLI_LABELS",
    "TemporalStatus",
    "SpanOffset",
    "TemporalCompatibility",
    "NliTemporalPairError",
    "NliTemporalPairValidationError",
    "InconsistentTemporalMetadataError",
    "TemporalInterval",
    "NormalizedEventInterval",
    "EventTemporalInterval",
    "TemporalMetadata",
    "NliTemporalMetadata",
    "ClinicalNliTemporalMetadata",
    "EventTemporalMetadata",
    "TemporalComparison",
    "TemporalIntervalComparison",
    "NliTemporalPair",
    "TemporalNliPair",
    "TemporalityAwareNliPair",
    "TemporalAwareNliPair",
    "ClinicalNliTemporalPair",
    "ClinicalTemporalNliPair",
    "compare_temporal_metadata",
    "classify_temporal_compatibility",
    "compare_temporal_intervals",
    "classify_temporal_intervals",
    "temporal_intervals_compatible",
    "build_temporal_nli_pair",
    "build_nli_temporal_pair",
    "build_temporality_aware_nli_pair",
    "construct_temporal_nli_pair",
    "build_temporal_aware_pair",
    "build_temporal_aware_nli_pair",
    "build_temporal_pair",
    "construct_temporal_aware_pair",
    "build_nli_pair",
    "build_temporal_nli_pairs",
    "build_nli_temporal_pairs",
    "build_nli_pairs",
    "build_temporality_aware_nli_pairs",
    "construct_temporal_nli_pairs",
    "build_temporal_aware_pairs",
    "build_temporal_aware_nli_pairs",
    "construct_temporal_aware_pairs",
    "build_temporal_pairs",
    "validate_temporal_nli_pair",
    "validate_nli_temporal_pair",
    "validate_temporal_pair",
    "validate_nli_pair",
    "normalize_temporal_interval",
    "normalize_event_interval",
    "normalize_event_time_interval",
]
