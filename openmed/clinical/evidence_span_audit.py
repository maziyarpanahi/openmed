"""Deterministic, privacy-conscious audits for overlapping evidence spans.

The audit compares caller-supplied half-open offset ranges and retains only
opaque evidence/source identifiers, offsets, classifications, counts, and
fingerprints. It does not read source text, call a network service, or choose a
winning span. Resolution is intentionally left to a human reviewer or an
explicit downstream policy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

EVIDENCE_SPAN_AUDIT_SCHEMA_VERSION = "1.0"
EVIDENCE_SPAN_AUDIT_ADVISORY = (
    "Evidence-span overlap findings are assistive review signals, not a "
    "clinical decision or automatic provenance resolution."
)


class OverlapKind(str, Enum):
    """Classification for one overlapping pair of evidence spans."""

    EXACT = "exact"
    NESTED = "nested"
    PARTIAL = "partial"
    CROSS_SOURCE = "cross_source"


OVERLAP_KINDS: tuple[str, ...] = tuple(kind.value for kind in OverlapKind)
_OVERLAP_KIND_ORDER = {kind: index for index, kind in enumerate(OVERLAP_KINDS)}


def _canonical_json(value: Any) -> str:
    """Serialize a PHI-free payload with stable key and separator ordering."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(value: Any) -> str:
    """Return a deterministic, non-keyed fingerprint for an audit payload."""

    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a non-empty string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _offset(value: object, field_name: str) -> int:
    if type(value) is not int:  # bool is an int subclass, but not an offset.
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _mapping_value(
    item: Mapping[str, Any],
    names: tuple[str, ...],
    field_name: str,
) -> object:
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    raise ValueError(f"evidence span {field_name} is required")


@dataclass(frozen=True)
class EvidenceSpan:
    """Opaque identity and half-open source range for one evidence record.

    ``source_id`` identifies the coordinate space in which ``start`` and
    ``end`` are meaningful. ``evidence_id`` identifies the record within the
    caller's evidence set. Neither identifier is interpreted as source text.
    """

    source_id: str
    evidence_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "evidence_id",
            _identifier(self.evidence_id, "evidence_id"),
        )
        normalized_start = _offset(self.start, "start")
        normalized_end = _offset(self.end, "end")
        if normalized_end <= normalized_start:
            raise ValueError("evidence span end must be greater than start")
        object.__setattr__(self, "start", normalized_start)
        object.__setattr__(self, "end", normalized_end)

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "EvidenceSpan":
        """Build a span from an opaque mapping without retaining extra fields.

        ``source``/``document_id``/``doc_id``, ``id``/``span_id``, and offset
        aliases are accepted for interoperability with existing span-like
        records. Any additional mapping fields, including a possible ``text``
        field, are deliberately ignored.
        """

        return cls(
            source_id=_mapping_value(
                item,
                ("source_id", "source", "document_id", "doc_id"),
                "source_id",
            ),
            evidence_id=_mapping_value(
                item,
                ("evidence_id", "span_id", "id"),
                "evidence_id",
            ),
            start=_mapping_value(
                item,
                ("start", "start_offset", "start_char"),
                "start",
            ),
            end=_mapping_value(item, ("end", "end_offset", "end_char"), "end"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return only opaque identity and numeric range metadata."""

        return {
            "source_id": self.source_id,
            "evidence_id": self.evidence_id,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class EvidenceSpanOverlap:
    """One deterministic overlap finding between two evidence records."""

    first: EvidenceSpan
    second: EvidenceSpan
    kind: OverlapKind
    overlap_start: int
    overlap_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OverlapKind):
            object.__setattr__(self, "kind", OverlapKind(self.kind))
        if self.overlap_start < 0 or self.overlap_end <= self.overlap_start:
            raise ValueError("overlap range must be a non-empty non-negative range")

    @property
    def classification(self) -> str:
        """Return the string classification used in serialized reports."""

        return self.kind.value

    @property
    def left(self) -> EvidenceSpan:
        """Alias for the first canonical member of the pair."""

        return self.first

    @property
    def right(self) -> EvidenceSpan:
        """Alias for the second canonical member of the pair."""

        return self.second

    @property
    def overlap_length(self) -> int:
        """Return the numeric length of the overlapping interval."""

        return self.overlap_end - self.overlap_start

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "kind": self.kind.value,
            "overlap": [self.overlap_start, self.overlap_end],
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable fingerprint for this pair finding."""

        return _fingerprint(self._fingerprint_payload())

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-free overlap finding."""

        return {
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "kind": self.kind.value,
            "overlap": [self.overlap_start, self.overlap_end],
            "overlap_length": self.overlap_length,
            "fingerprint": self.fingerprint,
        }


def _span_sort_key(span: EvidenceSpan) -> tuple[str, int, int, str]:
    return span.source_id, span.start, span.end, span.evidence_id


def _overlap_sort_key(
    overlap: EvidenceSpanOverlap,
) -> tuple[int, tuple[str, int, int, str], tuple[str, int, int, str]]:
    return (
        _OVERLAP_KIND_ORDER[overlap.kind.value],
        _span_sort_key(overlap.first),
        _span_sort_key(overlap.second),
    )


@dataclass(frozen=True)
class EvidenceSpanAudit:
    """Complete deterministic overlap audit with no automatic resolution."""

    spans: tuple[EvidenceSpan, ...]
    overlaps: tuple[EvidenceSpanOverlap, ...]
    counts: Mapping[str, int] = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        normalized_spans = tuple(sorted(self.spans, key=_span_sort_key))
        normalized_overlaps = tuple(sorted(self.overlaps, key=_overlap_sort_key))
        counts = {kind: 0 for kind in OVERLAP_KINDS}
        for overlap in normalized_overlaps:
            counts[overlap.kind.value] += 1

        object.__setattr__(self, "spans", normalized_spans)
        object.__setattr__(self, "overlaps", normalized_overlaps)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(
            self, "fingerprint", _fingerprint(self._fingerprint_payload())
        )

    @property
    def overlap_count(self) -> int:
        """Return the total number of pair findings."""

        return len(self.overlaps)

    @property
    def source_count(self) -> int:
        """Return the number of distinct opaque source identifiers."""

        return len({span.source_id for span in self.spans})

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SPAN_AUDIT_SCHEMA_VERSION,
            "spans": [span.to_dict() for span in self.spans],
            "overlaps": [overlap._fingerprint_payload() for overlap in self.overlaps],
            "counts": dict(self.counts),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable audit report containing no source text."""

        return {
            "schema_version": EVIDENCE_SPAN_AUDIT_SCHEMA_VERSION,
            "advisory": EVIDENCE_SPAN_AUDIT_ADVISORY,
            "span_count": len(self.spans),
            "source_count": self.source_count,
            "counts": dict(self.counts),
            "overlap_count": self.overlap_count,
            "spans": [span.to_dict() for span in self.spans],
            "overlaps": [overlap.to_dict() for overlap in self.overlaps],
            "fingerprint": self.fingerprint,
        }

    def to_json(self) -> str:
        """Return the report in deterministic JSON form."""

        return _canonical_json(self.to_dict())


def _coerce_span(item: EvidenceSpan | Mapping[str, Any]) -> EvidenceSpan:
    if isinstance(item, EvidenceSpan):
        return item
    if isinstance(item, Mapping):
        return EvidenceSpan.from_mapping(item)
    raise TypeError("evidence spans must be EvidenceSpan values or mappings")


def _classify_overlap(
    first: EvidenceSpan,
    second: EvidenceSpan,
) -> EvidenceSpanOverlap | None:
    overlap_start = max(first.start, second.start)
    overlap_end = min(first.end, second.end)
    if overlap_start >= overlap_end:
        return None

    if first.source_id != second.source_id:
        kind = OverlapKind.CROSS_SOURCE
    elif first.start == second.start and first.end == second.end:
        kind = OverlapKind.EXACT
    elif (first.start <= second.start and second.end <= first.end) or (
        second.start <= first.start and first.end <= second.end
    ):
        kind = OverlapKind.NESTED
    else:
        kind = OverlapKind.PARTIAL

    return EvidenceSpanOverlap(
        first=first,
        second=second,
        kind=kind,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
    )


def audit_evidence_spans(
    spans: Iterable[EvidenceSpan | Mapping[str, Any]],
) -> EvidenceSpanAudit:
    """Audit all pairwise overlaps in an evidence-span collection.

    Ranges use half-open offsets ``[start, end)``. Same-source pairs are
    classified as ``exact``, ``nested``, or ``partial``. A numerically
    overlapping pair with different ``source_id`` values is classified as
    ``cross_source``; the audit does not assume that those coordinates refer to
    identical source text.

    Input order does not affect the report or its fingerprint. Mapping inputs
    may contain source text or other fields, but only the opaque identifiers and
    offsets described by :class:`EvidenceSpan` are read and returned.
    """

    if isinstance(spans, (str, bytes, Mapping)):
        raise TypeError("spans must be an iterable of evidence span records")

    normalized: list[EvidenceSpan] = []
    for index, item in enumerate(spans):
        try:
            normalized.append(_coerce_span(item))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"invalid evidence span at index {index}: {exc}") from exc

    ordered_spans = tuple(sorted(normalized, key=_span_sort_key))
    overlaps: list[EvidenceSpanOverlap] = []
    for index, first in enumerate(ordered_spans):
        for second in ordered_spans[index + 1 :]:
            overlap = _classify_overlap(first, second)
            if overlap is not None:
                overlaps.append(overlap)

    return EvidenceSpanAudit(spans=ordered_spans, overlaps=tuple(overlaps))


def audit_evidence_span_overlaps(
    spans: Iterable[EvidenceSpan | Mapping[str, Any]],
) -> EvidenceSpanAudit:
    """Alias for :func:`audit_evidence_spans` with an explicit audit name."""

    return audit_evidence_spans(spans)


__all__ = [
    "EVIDENCE_SPAN_AUDIT_ADVISORY",
    "EVIDENCE_SPAN_AUDIT_SCHEMA_VERSION",
    "OVERLAP_KINDS",
    "EvidenceSpan",
    "EvidenceSpanAudit",
    "EvidenceSpanOverlap",
    "OverlapKind",
    "audit_evidence_span_overlaps",
    "audit_evidence_spans",
]
