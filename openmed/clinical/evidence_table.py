"""Deterministic, value-free clinical evidence tables for review.

This module retains only source offsets, controlled assertion states,
confidence, review flags, and optional SHA-256 digests. Raw extracted values
never enter an :class:`EvidenceRecord`, so JSON, Markdown, representations, and
validation errors cannot disclose them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterable

__all__ = [
    "EVIDENCE_TABLE_DISCLAIMER",
    "EVIDENCE_TABLE_SCHEMA_VERSION",
    "AssertionStatus",
    "EvidenceRecord",
    "EvidenceTable",
]


EVIDENCE_TABLE_SCHEMA_VERSION: Final = 1
EVIDENCE_TABLE_DISCLAIMER: Final = (
    "Clinical evidence tables are value-free review aids, not clinical "
    "decisions or compliance certifications."
)

_VALUE_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class AssertionStatus(str, Enum):
    """Controlled assertion states retained in evidence tables."""

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Value-free metadata for one extracted clinical finding.

    Use :meth:`from_extraction` when a protected source value is available.
    The value is omitted by default; opt-in hashing computes a digest before
    the immutable record is constructed, so the raw value is never stored.
    """

    source_start: int
    source_end: int
    assertion_status: AssertionStatus
    confidence: float
    review_required: bool
    value_hash: str | None = None

    def __post_init__(self) -> None:
        start, end = _source_offsets(self.source_start, self.source_end)
        object.__setattr__(self, "source_start", start)
        object.__setattr__(self, "source_end", end)
        object.__setattr__(
            self,
            "assertion_status",
            _assertion_status(self.assertion_status),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if type(self.review_required) is not bool:
            raise TypeError("review_required must be a boolean")
        if self.value_hash is not None:
            object.__setattr__(self, "value_hash", _value_hash(self.value_hash))

    @classmethod
    def from_extraction(
        cls,
        *,
        source_start: object,
        source_end: object,
        assertion_status: object,
        confidence: object,
        review_required: object,
        protected_value: str | bytes | None = None,
        include_value_hash: bool = False,
    ) -> "EvidenceRecord":
        """Build a record while hashing or omitting a protected value.

        Args:
            source_start: Inclusive character offset in the source document.
            source_end: Exclusive character offset in the source document.
            assertion_status: One controlled :class:`AssertionStatus` value.
            confidence: Finite probability in the closed interval ``[0, 1]``.
            review_required: Whether a reviewer must inspect this finding.
            protected_value: Optional raw extracted value. It is never stored.
            include_value_hash: Hash ``protected_value`` when ``True``;
                otherwise omit it.
        """

        if type(include_value_hash) is not bool:
            raise TypeError("include_value_hash must be a boolean")
        value_hash = None
        if include_value_hash:
            if protected_value is None:
                raise ValueError("protected value is required for hashing")
            value_hash = _hash_protected_value(protected_value)
        start, end = _source_offsets(source_start, source_end)
        status = _assertion_status(assertion_status)
        normalized_confidence = _confidence(confidence)
        if not isinstance(review_required, bool):
            raise TypeError("review_required must be a boolean")
        return cls(
            source_start=start,
            source_end=end,
            assertion_status=status,
            confidence=normalized_confidence,
            review_required=review_required,
            value_hash=value_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping containing no raw clinical value."""

        payload: dict[str, Any] = {
            "source_offsets": {
                "start": self.source_start,
                "end": self.source_end,
            },
            "assertion_status": self.assertion_status.value,
            "confidence": self.confidence,
            "review_required": self.review_required,
        }
        if self.value_hash is not None:
            payload["value_hash"] = self.value_hash
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceTable:
    """Deterministically sorted evidence records with JSON and Markdown views."""

    records: tuple[EvidenceRecord, ...]
    schema_version: int = EVIDENCE_TABLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != EVIDENCE_TABLE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported evidence table schema version")
        records = tuple(self.records)
        if any(type(record) is not EvidenceRecord for record in records):
            raise TypeError("records must contain EvidenceRecord values")
        object.__setattr__(self, "records", tuple(sorted(records, key=_record_key)))

    @classmethod
    def from_records(cls, records: Iterable[EvidenceRecord]) -> "EvidenceTable":
        """Build a table from any finite iterable of typed records."""

        return cls(records=tuple(records))

    @property
    def review_required_count(self) -> int:
        """Return the number of records flagged for review."""

        return sum(record.review_required for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic counts-and-offsets representation."""

        counts = Counter(record.assertion_status for record in self.records)
        assertion_counts = {
            status.value: counts[status]
            for status in AssertionStatus
            if counts[status] > 0
        }
        return {
            "schema_version": self.schema_version,
            "record_count": len(self.records),
            "review_required_count": self.review_required_count,
            "assertion_counts": assertion_counts,
            "records": [record.to_dict() for record in self.records],
            "disclaimer": EVIDENCE_TABLE_DISCLAIMER,
        }

    def to_json(self) -> str:
        """Serialize the table as compact deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a compact table containing offsets and safe metadata only."""

        lines = [
            "# Clinical Evidence Table",
            "",
            EVIDENCE_TABLE_DISCLAIMER,
            "",
            f"Records: {len(self.records)}",
            f"Review required: {self.review_required_count}",
            "",
            "| # | Start | End | Assertion | Confidence | Review required | Value hash |",
            "| ---: | ---: | ---: | --- | ---: | --- | --- |",
        ]
        for index, record in enumerate(self.records, start=1):
            lines.append(
                "| "
                f"{index} | {record.source_start} | {record.source_end} | "
                f"{record.assertion_status.value} | {record.confidence:.6f} | "
                f"{'yes' if record.review_required else 'no'} | "
                f"{record.value_hash or 'omitted'} |"
            )
        return "\n".join(lines) + "\n"


def _source_offsets(start: object, end: object) -> tuple[int, int]:
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise TypeError("source offsets must be integers")
    try:
        normalized_start = int(start)
        normalized_end = int(end)
    except Exception:
        raise TypeError("source offsets must be integers") from None
    if normalized_start < 0 or normalized_end <= normalized_start:
        raise ValueError("source offsets must form a non-empty half-open span")
    return normalized_start, normalized_end


def _assertion_status(value: object) -> AssertionStatus:
    try:
        return AssertionStatus(value)
    except Exception:
        raise ValueError("assertion_status is unsupported") from None


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence must be a real number")
    try:
        normalized = float(value)
    except Exception:
        raise TypeError("confidence must be a real number") from None
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("confidence must be in the closed interval [0, 1]")
    return normalized


def _hash_protected_value(value: str | bytes) -> str:
    try:
        if isinstance(value, str):
            data = str(value).encode("utf-8")
        elif isinstance(value, bytes):
            data = bytes(value)
        else:
            raise TypeError
    except Exception:
        raise TypeError("protected value must be text or bytes") from None
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _value_hash(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("value_hash is invalid")
    try:
        normalized = str(value)
    except Exception:
        raise ValueError("value_hash is invalid") from None
    if _VALUE_HASH_RE.fullmatch(normalized) is None:
        raise ValueError("value_hash is invalid")
    return normalized


def _record_key(record: EvidenceRecord) -> tuple[Any, ...]:
    return (
        record.source_start,
        record.source_end,
        record.assertion_status.value,
        record.confidence,
        record.review_required,
        record.value_hash or "",
    )
