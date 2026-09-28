"""Deterministic, value-free recency labels for guarded clinical evidence.

Evidence recency is a disclosure signal, not a clinical conclusion.  This
module compares a caller-supplied evidence timestamp with an explicit
reference time and returns one of four controlled labels: ``current``,
``stale``, ``future-dated``, or ``unknown``.  Missing or malformed timestamps
fail closed to ``unknown``; the machine clock and network are never consulted.

The input timestamp is used only for the comparison.  Assessments and reports
retain labels, policy metadata, offsets-free counts, and fixed review flags;
they never retain or render the source timestamp, claim text, identifiers, or
arbitrary input metadata.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Final, cast

__all__ = [
    "CURRENT_RECENCY_LABEL",
    "EVIDENCE_RECENCY_DISCLAIMER",
    "EVIDENCE_RECENCY_CURRENT_LABEL",
    "EVIDENCE_RECENCY_FUTURE_DATED_LABEL",
    "EVIDENCE_RECENCY_SCHEMA_VERSION",
    "EVIDENCE_RECENCY_STALE_LABEL",
    "EVIDENCE_RECENCY_UNKNOWN_LABEL",
    "EvidenceRecencyError",
    "EvidenceRecencyLabel",
    "EvidenceRecencyPolicy",
    "EvidenceRecencyReport",
    "EvidenceRecencyResult",
    "RECENCY_LABELS",
    "assess_evidence_recency",
    "build_evidence_recency_report",
    "classify_evidence_recency",
    "label_evidence_recency",
    "render_evidence_recency_report",
]


EVIDENCE_RECENCY_SCHEMA_VERSION: Final[int] = 1
EVIDENCE_RECENCY_DISCLAIMER: Final[str] = (
    "Evidence recency labels are assistive disclosure for clinician review; "
    "they are not a clinical decision, a freshness guarantee, or a compliance "
    "certification."
)


class EvidenceRecencyLabel(str, Enum):
    """Controlled recency labels for one guarded evidence item."""

    CURRENT = "current"
    STALE = "stale"
    FUTURE_DATED = "future-dated"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: object) -> "EvidenceRecencyLabel":
        """Normalize a label or reject it without echoing caller data."""

        if isinstance(value, cls):
            return value
        if type(value) is not str:
            raise EvidenceRecencyError("recency label is invalid")
        normalized = value.strip().lower().replace("_", "-")
        try:
            return cls(normalized)
        except ValueError:
            raise EvidenceRecencyError("recency label is invalid") from None


EVIDENCE_RECENCY_CURRENT_LABEL: Final[str] = EvidenceRecencyLabel.CURRENT.value
CURRENT_RECENCY_LABEL: Final[str] = EVIDENCE_RECENCY_CURRENT_LABEL
EVIDENCE_RECENCY_STALE_LABEL: Final[str] = EvidenceRecencyLabel.STALE.value
EVIDENCE_RECENCY_FUTURE_DATED_LABEL: Final[str] = (
    EvidenceRecencyLabel.FUTURE_DATED.value
)
EVIDENCE_RECENCY_UNKNOWN_LABEL: Final[str] = EvidenceRecencyLabel.UNKNOWN.value
RECENCY_LABELS: Final[tuple[EvidenceRecencyLabel, ...]] = (
    EvidenceRecencyLabel.CURRENT,
    EvidenceRecencyLabel.STALE,
    EvidenceRecencyLabel.FUTURE_DATED,
    EvidenceRecencyLabel.UNKNOWN,
)

_UTC = timezone.utc
_DEFAULT_CURRENT_WINDOW = timedelta(days=30)
_TIMESTAMP_FIELDS = (
    "evidence_timestamp",
    "timestamp",
    "observed_at",
    "occurred_at",
    "evidence_time",
    "recorded_at",
)
_POLICY_KEYS = frozenset(
    {
        "current_window",
        "current_window_days",
        "current_window_seconds",
        "future_tolerance",
        "future_tolerance_days",
        "future_tolerance_seconds",
        "schema_version",
        "stale_after",
        "stale_after_days",
        "stale_after_seconds",
    }
)


class EvidenceRecencyError(ValueError):
    """Raised when recency policy or serialized metadata is malformed."""


def _invalid(field_name: str) -> EvidenceRecencyError:
    """Return a fixed-category error that does not include submitted values."""

    return EvidenceRecencyError(f"{field_name} is invalid")


def _normalize_duration(value: object, field_name: str) -> timedelta:
    if isinstance(value, timedelta):
        normalized = value
    elif type(value) is int or type(value) is float:
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise _invalid(field_name)
        try:
            normalized = timedelta(days=float(value))
        except (OverflowError, TypeError, ValueError):
            raise _invalid(field_name) from None
    else:
        raise _invalid(field_name)
    if normalized < timedelta(0):
        raise _invalid(field_name)
    return normalized


def _normalize_mapping_duration(
    value: Mapping[str, object],
    keys: tuple[str, ...],
    field_name: str,
) -> timedelta:
    present = tuple(key for key in keys if key in value)
    if len(present) > 1:
        raise _invalid(field_name)
    if not present:
        raise _invalid(field_name)
    key = present[0]
    raw_value = value[key]
    if key.endswith("_days"):
        return _normalize_duration(raw_value, field_name)
    if key.endswith("_seconds"):
        if type(raw_value) not in (int, float):
            raise _invalid(field_name)
        numeric_value = cast(int | float, raw_value)
        if not math.isfinite(float(numeric_value)):
            raise _invalid(field_name)
        try:
            normalized = timedelta(seconds=float(numeric_value))
        except (OverflowError, TypeError, ValueError):
            raise _invalid(field_name) from None
        if normalized < timedelta(0):
            raise _invalid(field_name)
        return normalized
    return _normalize_duration(raw_value, field_name)


@dataclass(frozen=True, slots=True)
class EvidenceRecencyPolicy:
    """Configurable, local-only thresholds for evidence recency.

    ``current_window`` is the maximum age of evidence that receives the
    ``current`` label.  Older evidence receives ``stale``.  Evidence later
    than the reference time by more than ``future_tolerance`` receives
    ``future-dated``.  Durations may be supplied as :class:`datetime.timedelta`
    instances or non-negative numeric day counts.
    """

    current_window: timedelta = _DEFAULT_CURRENT_WINDOW
    future_tolerance: timedelta = timedelta(0)
    schema_version: int = EVIDENCE_RECENCY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != EVIDENCE_RECENCY_SCHEMA_VERSION
        ):
            raise _invalid("schema_version")
        object.__setattr__(
            self,
            "current_window",
            _normalize_duration(self.current_window, "current_window"),
        )
        object.__setattr__(
            self,
            "future_tolerance",
            _normalize_duration(self.future_tolerance, "future_tolerance"),
        )

    @property
    def stale_after(self) -> timedelta:
        """Return the age boundary after which evidence is labelled stale."""

        return self.current_window

    @classmethod
    def from_value(cls, value: object) -> "EvidenceRecencyPolicy":
        """Normalize a policy instance or a JSON-compatible mapping."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or any(
            key not in _POLICY_KEYS for key in value
        ):
            raise _invalid("policy")

        current_keys = (
            "current_window",
            "current_window_days",
            "current_window_seconds",
            "stale_after",
            "stale_after_days",
            "stale_after_seconds",
        )
        future_keys = (
            "future_tolerance",
            "future_tolerance_days",
            "future_tolerance_seconds",
        )
        if any(key in value for key in current_keys):
            current_window = _normalize_mapping_duration(
                value, current_keys, "current_window"
            )
        else:
            current_window = _DEFAULT_CURRENT_WINDOW
        if any(key in value for key in future_keys):
            future_tolerance = _normalize_mapping_duration(
                value, future_keys, "future_tolerance"
            )
        else:
            future_tolerance = timedelta(0)
        schema_version = value.get("schema_version", EVIDENCE_RECENCY_SCHEMA_VERSION)
        return cls(
            current_window=current_window,
            future_tolerance=future_tolerance,
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, object]:
        """Return threshold metadata without any evidence timestamps."""

        return {
            "current_window_seconds": _duration_seconds(self.current_window),
            "future_tolerance_seconds": _duration_seconds(self.future_tolerance),
            "schema_version": self.schema_version,
        }


def _duration_seconds(value: timedelta) -> int | float:
    seconds = value.total_seconds()
    return int(seconds) if seconds.is_integer() else seconds


@dataclass(frozen=True, slots=True)
class EvidenceRecencyResult:
    """Value-free recency assessment for one evidence item.

    The source timestamp is deliberately not a field on this record.  A
    non-current label requests additional human review, while a current label
    never removes the broader guarded-clinical review requirement.
    """

    label: EvidenceRecencyLabel
    schema_version: int = EVIDENCE_RECENCY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", EvidenceRecencyLabel.from_value(self.label))
        if type(self.schema_version) is not int or (
            self.schema_version != EVIDENCE_RECENCY_SCHEMA_VERSION
        ):
            raise _invalid("schema_version")

    @property
    def known(self) -> bool:
        """Return whether a trustworthy timestamp produced this label."""

        return self.label is not EvidenceRecencyLabel.UNKNOWN

    @property
    def review_required(self) -> bool:
        """Return whether recency should trigger additional human review."""

        return self.label is not EvidenceRecencyLabel.CURRENT

    @property
    def requires_human_review(self) -> bool:
        """Alias for :attr:`review_required` used by review workflows."""

        return self.review_required

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready record containing only controlled metadata."""

        return {
            "label": self.label.value,
            "review_required": self.review_required,
            "schema_version": self.schema_version,
        }


def _coerce_timestamp(value: object) -> datetime | None:
    """Parse a timestamp without exposing invalid input in an exception."""

    try:
        if value is None:
            return None
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, datetime.min.time())
        elif type(value) is str:
            normalized = value.strip()
            if not normalized or any(ord(character) < 32 for character in normalized):
                return None
            try:
                parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            except (TypeError, ValueError, OverflowError):
                return None
        else:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_UTC)
        return parsed.astimezone(_UTC)
    except Exception:
        return None


def _unknown_result() -> EvidenceRecencyResult:
    return EvidenceRecencyResult(EvidenceRecencyLabel.UNKNOWN)


def assess_evidence_recency(
    evidence_timestamp: object,
    as_of: object | None = None,
    *,
    policy: EvidenceRecencyPolicy | Mapping[str, object] | None = None,
) -> EvidenceRecencyResult:
    """Assess one timestamp against an explicit reference time.

    Args:
        evidence_timestamp: An ISO-8601 string, :class:`datetime`, or
            :class:`date`. Missing, malformed, and unsupported values produce
            the explicit ``unknown`` label.
        as_of: Fixed reference timestamp. ``None`` or an invalid value produces
            ``unknown``; the current machine time is never substituted.
        policy: Optional :class:`EvidenceRecencyPolicy` or compatible mapping.

    Returns:
        A value-free :class:`EvidenceRecencyResult`.

    Raises:
        EvidenceRecencyError: If the policy is malformed.
    """

    normalized_policy = EvidenceRecencyPolicy.from_value(policy)
    evidence_time = _coerce_timestamp(evidence_timestamp)
    reference_time = _coerce_timestamp(as_of)
    if evidence_time is None or reference_time is None:
        return _unknown_result()

    age = reference_time - evidence_time
    if age < -normalized_policy.future_tolerance:
        label = EvidenceRecencyLabel.FUTURE_DATED
    elif age <= normalized_policy.current_window:
        label = EvidenceRecencyLabel.CURRENT
    else:
        label = EvidenceRecencyLabel.STALE
    return EvidenceRecencyResult(label)


def classify_evidence_recency(
    evidence_timestamp: object,
    as_of: object | None = None,
    *,
    policy: EvidenceRecencyPolicy | Mapping[str, object] | None = None,
) -> EvidenceRecencyLabel:
    """Return only the controlled label for one evidence timestamp."""

    return assess_evidence_recency(
        evidence_timestamp,
        as_of,
        policy=policy,
    ).label


def label_evidence_recency(
    evidence_timestamp: object,
    as_of: object | None = None,
    *,
    policy: EvidenceRecencyPolicy | Mapping[str, object] | None = None,
) -> EvidenceRecencyLabel:
    """Return a deterministic recency label for one evidence timestamp."""

    return classify_evidence_recency(
        evidence_timestamp,
        as_of,
        policy=policy,
    )


@dataclass(frozen=True, slots=True)
class EvidenceRecencyReport:
    """Deterministic value-free recency labels for an evidence collection."""

    records: tuple[EvidenceRecencyResult, ...]
    policy: EvidenceRecencyPolicy = EvidenceRecencyPolicy()
    schema_version: int = EVIDENCE_RECENCY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != EVIDENCE_RECENCY_SCHEMA_VERSION
        ):
            raise _invalid("schema_version")
        try:
            records = tuple(self.records)
        except Exception:
            raise _invalid("records") from None
        if any(type(record) is not EvidenceRecencyResult for record in records):
            raise _invalid("records")
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self, "policy", EvidenceRecencyPolicy.from_value(self.policy)
        )

    @property
    def assessments(self) -> tuple[EvidenceRecencyResult, ...]:
        """Return the ordered, value-free assessments."""

        return self.records

    @property
    def labels(self) -> tuple[EvidenceRecencyLabel, ...]:
        """Return the controlled labels in caller-supplied evidence order."""

        return tuple(record.label for record in self.records)

    @property
    def label_counts(self) -> dict[str, int]:
        """Return counts for every controlled label, including zeroes."""

        counts = Counter(record.label for record in self.records)
        return {label.value: counts[label] for label in RECENCY_LABELS}

    @property
    def review_required_count(self) -> int:
        """Return the number of non-current assessments requiring review."""

        return sum(record.review_required for record in self.records)

    @property
    def unknown_count(self) -> int:
        """Return the number of evidence items without a trustworthy timestamp."""

        return self.label_counts[EVIDENCE_RECENCY_UNKNOWN_LABEL]

    def to_dict(self) -> dict[str, object]:
        """Return deterministic report metadata without raw evidence values."""

        return {
            "disclaimer": EVIDENCE_RECENCY_DISCLAIMER,
            "label_counts": self.label_counts,
            "policy": self.policy.to_dict(),
            "record_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
            "review_required_count": self.review_required_count,
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report with stable JSON ordering and no raw values."""

        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=indent,
                separators=(",", ":") if indent is None else None,
                sort_keys=True,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Render a value-free review table for local human inspection."""

        lines = [
            "# Evidence Recency Report",
            "",
            EVIDENCE_RECENCY_DISCLAIMER,
            "",
            "| Evidence index | Label | Review required |",
            "| ---: | --- | --- |",
        ]
        lines.extend(
            f"| {index} | {record.label.value} | "
            f"{'yes' if record.review_required else 'no'} |"
            for index, record in enumerate(self.records, start=1)
        )
        lines.extend(
            (
                "",
                f"Records: {len(self.records)}",
                f"Review required: {self.review_required_count}",
                f"Unknown timestamps: {self.unknown_count}",
            )
        )
        return "\n".join(lines) + "\n"


def _timestamp_from_item(item: object) -> object | None:
    if item is None:
        return None
    if isinstance(item, Mapping):
        for field_name in _TIMESTAMP_FIELDS:
            try:
                if field_name in item and item[field_name] is not None:
                    return item[field_name]
            except Exception:
                return None
        return None
    if isinstance(item, (str, date, datetime)):
        return item
    for field_name in _TIMESTAMP_FIELDS:
        try:
            candidate = getattr(item, field_name)
        except Exception:
            continue
        if candidate is not None:
            return candidate
    return None


def _evidence_items(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, date, datetime, Mapping)):
        return (value,)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except Exception:
        raise _invalid("evidence collection") from None


def build_evidence_recency_report(
    evidence: Iterable[object] | object,
    as_of: object | None = None,
    *,
    policy: EvidenceRecencyPolicy | Mapping[str, object] | None = None,
) -> EvidenceRecencyReport:
    """Label evidence timestamps and return a value-free deterministic report.

    Each item may be a timestamp itself, a mapping/object with one of the
    supported timestamp fields, or a missing/invalid value.  The latter cases
    are retained only as explicit ``unknown`` records; all other input fields
    are ignored and never serialized.
    """

    normalized_policy = EvidenceRecencyPolicy.from_value(policy)
    records = tuple(
        assess_evidence_recency(
            _timestamp_from_item(item),
            as_of,
            policy=normalized_policy,
        )
        for item in _evidence_items(evidence)
    )
    return EvidenceRecencyReport(records=records, policy=normalized_policy)


def render_evidence_recency_report(
    evidence: Iterable[object] | object,
    as_of: object | None = None,
    *,
    policy: EvidenceRecencyPolicy | Mapping[str, object] | None = None,
    format: str = "markdown",
) -> str:
    """Render a deterministic JSON or Markdown recency report."""

    if type(format) is not str or format.strip().lower() not in {"json", "markdown"}:
        raise _invalid("format")
    report = build_evidence_recency_report(evidence, as_of, policy=policy)
    return (
        report.to_json() if format.strip().lower() == "json" else report.to_markdown()
    )
