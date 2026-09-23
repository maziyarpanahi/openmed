"""Versioned, privacy-safe reasons for human review overrides.

The structured record keeps an optional free-text note available to the local
review experience while deliberately omitting that note from its default,
telemetry, and aggregate representations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

REVIEW_OVERRIDE_REASON_SCHEMA_VERSION: Final = 1


class ReviewOverrideReasonCode(str, Enum):
    """Stable reason codes for a human review override."""

    ACCEPT = "accept"
    CORRECT = "correct"
    REJECT = "reject"
    DEFER = "defer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Short alias for callers that already use ``Override`` as the surrounding type.
OverrideReasonCode = ReviewOverrideReasonCode


@dataclass(frozen=True)
class ReviewOverride:
    """One structured override with an optional local-only note.

    ``local_note`` is excluded from ``repr`` and every safe/default reporting
    method. Callers must opt into :meth:`to_local_dict` when persisting the note
    to an appropriately protected local store.
    """

    reason_code: ReviewOverrideReasonCode | str
    local_note: str | None = field(default=None, repr=False)
    schema_version: int = REVIEW_OVERRIDE_REASON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_OVERRIDE_REASON_SCHEMA_VERSION:
            raise ValueError("unsupported review override reason schema version")
        object.__setattr__(self, "reason_code", _coerce_reason_code(self.reason_code))
        if self.local_note is not None and not isinstance(self.local_note, str):
            raise TypeError("local_note must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        """Return the safe default representation without local note content."""

        return self.to_telemetry_dict()

    def to_telemetry_dict(self) -> dict[str, Any]:
        """Return the allow-listed telemetry representation."""

        return {
            "schema_version": self.schema_version,
            "reason_code": self.reason_code.value,
        }

    def to_aggregate_dict(self) -> dict[str, Any]:
        """Return the value-free representation used for aggregation."""

        return self.to_telemetry_dict()

    def to_local_dict(self) -> dict[str, Any]:
        """Return the explicit local representation, including the note."""

        return {**self.to_telemetry_dict(), "local_note": self.local_note}

    @classmethod
    def from_local_dict(cls, value: Mapping[str, Any]) -> "ReviewOverride":
        """Restore an override from an explicit local representation."""

        if not isinstance(value, Mapping):
            raise TypeError("review override payload must be a mapping")
        return cls(
            reason_code=value.get("reason_code"),
            local_note=value.get("local_note"),
            schema_version=value.get(
                "schema_version", REVIEW_OVERRIDE_REASON_SCHEMA_VERSION
            ),
        )


@dataclass(frozen=True)
class ReviewOverrideAggregate:
    """Deterministic counts of structured override reasons."""

    counts: Mapping[str, int]
    total: int
    schema_version: int = REVIEW_OVERRIDE_REASON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_OVERRIDE_REASON_SCHEMA_VERSION:
            raise ValueError("unsupported review override reason schema version")
        if not isinstance(self.counts, Mapping):
            raise TypeError("review override aggregate counts must be a mapping")
        normalized: dict[str, int] = {}
        for code in ReviewOverrideReasonCode:
            count = self.counts.get(code.value, 0)
            if type(count) is not int:
                raise TypeError("review override aggregate counts must be integers")
            normalized[code.value] = count
        if any(count < 0 for count in normalized.values()):
            raise ValueError("review override aggregate counts must be non-negative")
        if type(self.total) is not int or self.total != sum(normalized.values()):
            raise ValueError("review override aggregate total must equal its counts")
        object.__setattr__(self, "counts", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        """Return a note-free, JSON-compatible aggregate report."""

        return {
            "schema_version": self.schema_version,
            "total": self.total,
            "counts": dict(self.counts),
        }


def create_review_override(
    reason_code: ReviewOverrideReasonCode | str,
    *,
    local_note: str | None = None,
) -> ReviewOverride:
    """Create a validated structured review override locally."""

    return ReviewOverride(reason_code=reason_code, local_note=local_note)


def aggregate_review_overrides(
    overrides: Iterable[ReviewOverride | Mapping[str, Any]],
) -> ReviewOverrideAggregate:
    """Count reason codes without reading or retaining local note content."""

    if isinstance(overrides, (str, bytes, bytearray)):
        raise TypeError("overrides must be an iterable of review overrides")

    counts: Counter[str] = Counter()
    try:
        for override in overrides:
            if isinstance(override, ReviewOverride):
                code = override.reason_code
            elif isinstance(override, Mapping):
                code = _coerce_reason_code(override.get("reason_code"))
            else:
                raise TypeError
            counts[code.value] += 1
    except TypeError:
        raise TypeError("overrides must contain review override records") from None

    return ReviewOverrideAggregate(counts=counts, total=sum(counts.values()))


def _coerce_reason_code(value: Any) -> ReviewOverrideReasonCode:
    if isinstance(value, ReviewOverrideReasonCode):
        return value
    try:
        return ReviewOverrideReasonCode(value)
    except (TypeError, ValueError):
        raise ValueError("unsupported review override reason code") from None


__all__ = [
    "REVIEW_OVERRIDE_REASON_SCHEMA_VERSION",
    "OverrideReasonCode",
    "ReviewOverride",
    "ReviewOverrideAggregate",
    "ReviewOverrideReasonCode",
    "aggregate_review_overrides",
    "create_review_override",
]
