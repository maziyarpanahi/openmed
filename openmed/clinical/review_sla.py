"""Counts-only SLA summaries for value-free clinical review packets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from .review_transitions import (
    CLINICAL_REVIEW_COMPATIBILITY_POLICY,
    CLINICAL_REVIEW_PACKET_SCHEMA_VERSION,
    CLINICAL_REVIEW_PRIORITIES,
    CLINICAL_REVIEW_STATES,
    ClinicalReviewError,
    ClinicalReviewPacket,
)

REVIEW_QUEUE_AGE_BUCKETS = ("under_4h", "4h_to_24h", "1d_to_3d", "over_3d")
DEFAULT_REVIEW_SLA_HOURS: Mapping[str, float] = MappingProxyType(
    {"critical": 1.0, "high": 4.0, "normal": 24.0, "low": 72.0}
)


@dataclass(frozen=True, slots=True)
class ClinicalReviewQueueSummary:
    """Aggregate queue state without packet keys or reviewer identities."""

    generated_at: str
    total: int
    state_counts: Mapping[str, int]
    priority_counts: Mapping[str, int]
    age_bucket_counts: Mapping[str, int]
    expired_count: int
    overdue_count: int
    schema_version: str = CLINICAL_REVIEW_PACKET_SCHEMA_VERSION
    compatibility_policy: str = CLINICAL_REVIEW_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        try:
            _parse(self.generated_at)
        except ValueError:
            raise ClinicalReviewError(
                "generated_at must be a timezone-aware timestamp"
            ) from None
        if self.schema_version != CLINICAL_REVIEW_PACKET_SCHEMA_VERSION:
            raise ClinicalReviewError("unsupported queue summary schema version")
        if self.compatibility_policy != CLINICAL_REVIEW_COMPATIBILITY_POLICY:
            raise ClinicalReviewError("unsupported compatibility policy")
        if self.total < 0 or self.expired_count < 0 or self.overdue_count < 0:
            raise ClinicalReviewError("review queue counts must be non-negative")
        state_counts = _complete_counts(self.state_counts, CLINICAL_REVIEW_STATES)
        priority_counts = _complete_counts(
            self.priority_counts, CLINICAL_REVIEW_PRIORITIES
        )
        age_counts = _complete_counts(self.age_bucket_counts, REVIEW_QUEUE_AGE_BUCKETS)
        if sum(state_counts.values()) != self.total:
            raise ClinicalReviewError("review state counts must equal total")
        active_total = sum(
            state_counts[state] for state in ("queued", "in_review", "reopened")
        )
        if sum(priority_counts.values()) != active_total:
            raise ClinicalReviewError("priority counts must equal active queue total")
        if sum(age_counts.values()) != active_total:
            raise ClinicalReviewError("age counts must equal active queue total")
        if self.overdue_count > active_total or self.expired_count > self.total:
            raise ClinicalReviewError("review queue summary counts are inconsistent")
        object.__setattr__(self, "state_counts", MappingProxyType(state_counts))
        object.__setattr__(self, "priority_counts", MappingProxyType(priority_counts))
        object.__setattr__(self, "age_bucket_counts", MappingProxyType(age_counts))

    def to_dict(self) -> dict[str, Any]:
        """Return counts only."""

        return {
            "age_bucket_counts": dict(self.age_bucket_counts),
            "compatibility_policy": self.compatibility_policy,
            "expired_count": self.expired_count,
            "generated_at": self.generated_at,
            "overdue_count": self.overdue_count,
            "priority_counts": dict(self.priority_counts),
            "schema_version": self.schema_version,
            "state_counts": dict(self.state_counts),
            "total": self.total,
        }


def summarize_review_queue(
    packets: Iterable[ClinicalReviewPacket],
    *,
    now: datetime,
    sla_hours: Mapping[str, float] = DEFAULT_REVIEW_SLA_HOURS,
) -> ClinicalReviewQueueSummary:
    """Build a deterministic counts-only queue report from an injected clock."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ClinicalReviewError("now must be timezone-aware")
    normalized_now = now.astimezone(timezone.utc)
    normalized_sla = _sla_hours(sla_hours)
    materialized = tuple(packets)
    states: Counter[str] = Counter()
    priorities: Counter[str] = Counter()
    ages: Counter[str] = Counter()
    expired = 0
    overdue = 0
    for packet in materialized:
        states[packet.state] += 1
        created = _parse(packet.created_at)
        if created > normalized_now:
            raise ClinicalReviewError("packet creation cannot be in the future")
        if packet.state == "expired":
            expired += 1
        if packet.state not in {"queued", "in_review", "reopened"}:
            continue
        age_hours = (normalized_now - created).total_seconds() / 3600.0
        priorities[packet.priority] += 1
        ages[_age_bucket(age_hours)] += 1
        deadline_hours = normalized_sla[packet.priority]
        explicit_expiry = _parse(packet.expires_at) if packet.expires_at else None
        if age_hours > deadline_hours or (
            explicit_expiry is not None and normalized_now > explicit_expiry
        ):
            overdue += 1
    return ClinicalReviewQueueSummary(
        generated_at=normalized_now.isoformat(),
        total=len(materialized),
        state_counts=states,
        priority_counts=priorities,
        age_bucket_counts=ages,
        expired_count=expired,
        overdue_count=overdue,
    )


def _complete_counts(
    values: Mapping[str, int], allowed: Iterable[str]
) -> dict[str, int]:
    allowed_keys = tuple(sorted(allowed))
    if set(values) - set(allowed_keys):
        raise ClinicalReviewError("queue summary contains unsupported buckets")
    result = {key: int(values.get(key, 0)) for key in allowed_keys}
    if any(value < 0 for value in result.values()):
        raise ClinicalReviewError("queue summary counts must be non-negative")
    return result


def _sla_hours(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(CLINICAL_REVIEW_PRIORITIES):
        raise ClinicalReviewError("SLA hours must cover every priority")
    result = {key: float(value) for key, value in values.items()}
    if any(value <= 0 for value in result.values()):
        raise ClinicalReviewError("SLA hours must be positive")
    return result


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _age_bucket(hours: float) -> str:
    if hours < 4:
        return "under_4h"
    if hours < 24:
        return "4h_to_24h"
    if hours < 72:
        return "1d_to_3d"
    return "over_3d"


__all__ = [
    "DEFAULT_REVIEW_SLA_HOURS",
    "REVIEW_QUEUE_AGE_BUCKETS",
    "ClinicalReviewQueueSummary",
    "summarize_review_queue",
]
