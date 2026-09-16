"""Deterministic, privacy-safe review of summary-claim evidence.

Summary claims are only as reviewable as the evidence they cite.  This module
joins claim references to caller-supplied assertion, temporal, and
source-integrity records and reports incompatible evidence without selecting a
clinical truth.  Inputs are accepted as small mappings or typed records so the
review boundary can sit after any local extractor.

The public report deliberately contains claim and record identifiers, counts,
and hashes only.  It does not retain source text, assertion excerpts, or
interval values.  The implementation is pure and local-first: it performs no
network access, reads no environment state, and emits no logs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from openmed.core.audit import hash_text

CLAIM_CONFLICT_SCHEMA_VERSION = 1
CLAIM_CONFLICT_ADVISORY = (
    "Evidence-claim contradiction review is deterministic assistive output for "
    "human review, not a clinical decision or a substitute for qualified "
    "clinical judgment."
)

ClaimConflictType = Literal[
    "assertion",
    "temporal",
    "source_integrity",
    "missing_evidence",
]
ClaimReviewState = Literal["clear", "review_required"]

CLAIM_REVIEW_CLEAR: ClaimReviewState = "clear"
CLAIM_REVIEW_REQUIRED: ClaimReviewState = "review_required"
CLAIM_REVIEW_STATES: tuple[ClaimReviewState, ...] = (
    CLAIM_REVIEW_CLEAR,
    CLAIM_REVIEW_REQUIRED,
)

ASSERTION_CONFLICT_ROUTE = "assertion_conflict"
TEMPORAL_CONFLICT_ROUTE = "temporal_conflict"
SOURCE_INTEGRITY_CONFLICT_ROUTE = "source_integrity_conflict"
MISSING_EVIDENCE_ROUTE = "missing_evidence"
CLAIM_REVIEW_ROUTES = (
    ASSERTION_CONFLICT_ROUTE,
    TEMPORAL_CONFLICT_ROUTE,
    SOURCE_INTEGRITY_CONFLICT_ROUTE,
    MISSING_EVIDENCE_ROUTE,
)

_CONFLICT_TYPES: tuple[ClaimConflictType, ...] = (
    "assertion",
    "temporal",
    "source_integrity",
    "missing_evidence",
)
CLAIM_CONFLICT_TYPES = _CONFLICT_TYPES
_ROUTE_BY_TYPE: dict[ClaimConflictType, str] = {
    "assertion": ASSERTION_CONFLICT_ROUTE,
    "temporal": TEMPORAL_CONFLICT_ROUTE,
    "source_integrity": SOURCE_INTEGRITY_CONFLICT_ROUTE,
    "missing_evidence": MISSING_EVIDENCE_ROUTE,
}
_HASH_RE = re.compile(r"^(?:hmac-)?sha256:[0-9a-f]{64}$")
_DATE_INTERVAL_RE = re.compile(
    r"^(?P<start>\d{4}-\d{2}-\d{2})/(?P<end>\d{4}-\d{2}-\d{2})$"
)
_GOOD_INTEGRITY_STATUSES = frozenset(
    {"verified", "valid", "ok", "passed", "trusted", "intact", "match", "matched"}
)
_BAD_INTEGRITY_STATUSES = frozenset(
    {
        "unverified",
        "invalid",
        "failed",
        "failure",
        "mismatch",
        "mismatched",
        "hash_mismatch",
        "verification_failed",
        "not_verified",
        "tampered",
        "revoked",
        "unknown",
        "missing",
        "conflict",
    }
)
_AFFIRMED_STATES = frozenset(
    {"affirmed", "active", "confirmed", "present", "positive", "true", "yes"}
)
_NEGATED_STATES = frozenset(
    {"negated", "refuted", "absent", "negative", "denied", "false", "no"}
)
_UNCERTAIN_STATES = frozenset(
    {"uncertain", "possible", "suspected", "provisional", "maybe"}
)
_HYPOTHETICAL_STATES = frozenset({"hypothetical", "conditional", "future", "planned"})
_UNKNOWN_STATES = frozenset({"", "unknown", "unset", "none", "not_applicable"})
_ID_KEYS = (
    "record_id",
    "assertion_id",
    "temporal_id",
    "source_integrity_id",
    "source_id",
    "evidence_id",
    "span_id",
    "entity_key",
    "id",
)


@dataclass(frozen=True)
class _IntervalBounds:
    """Internal inclusive date envelope used for temporal comparison."""

    lower: date
    upper: date

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError("temporal interval bounds are inverted")


@dataclass(frozen=True)
class ClaimReference:
    """One privacy-safe evidence reference cited by a summary claim.

    The reference stores identifiers and an optional source hash, never the
    cited surface text.  ``assertion_id``, ``temporal_id``, and
    ``source_integrity_id`` default to ``evidence_id`` so one shared evidence
    record can satisfy all three joins.
    """

    evidence_id: str
    assertion_id: str | None = None
    temporal_id: str | None = None
    source_id: str | None = None
    source_integrity_id: str | None = None
    text_hash: str | None = None
    expected_assertion: str | None = None
    expected_interval: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _required_id(self.evidence_id))
        for field_name in (
            "assertion_id",
            "temporal_id",
            "source_id",
            "source_integrity_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _required_id(value))
        if self.text_hash is not None:
            object.__setattr__(
                self,
                "text_hash",
                _normalize_hash(self.text_hash, namespace="claim-reference"),
            )
        if self.expected_assertion is not None:
            object.__setattr__(
                self,
                "expected_assertion",
                _normalize_assertion_state(self.expected_assertion),
            )
        if self.expected_interval is not None:
            object.__setattr__(
                self,
                "expected_interval",
                _parse_interval(self.expected_interval),
            )

    @property
    def assertion_record_id(self) -> str:
        """Return the assertion record key used for this reference."""

        return self.assertion_id or self.evidence_id

    @property
    def temporal_record_id(self) -> str:
        """Return the temporal record key used for this reference."""

        return self.temporal_id or self.evidence_id

    @property
    def integrity_record_id(self) -> str:
        """Return the source-integrity record key used for this reference."""

        return self.source_integrity_id or self.source_id or self.evidence_id


@dataclass(frozen=True)
class ClaimRecord:
    """A summary claim and the evidence references it asks a reviewer to check."""

    claim_id: str
    references: tuple[ClaimReference | str, ...] = ()
    expected_assertion: str | None = None
    expected_interval: object | None = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _required_id(self.claim_id))
        raw_references: list[ClaimReference | str] = list(self.references)
        raw_references.extend(self.evidence_ids)
        normalized: list[ClaimReference] = []
        seen: dict[str, ClaimReference] = {}
        for raw_reference in raw_references:
            reference = _coerce_reference(raw_reference)
            previous = seen.get(reference.evidence_id)
            if previous is not None and previous != reference:
                raise ValueError(
                    "claim contains conflicting duplicate evidence references"
                )
            if previous is None:
                seen[reference.evidence_id] = reference
                normalized.append(reference)
        normalized.sort(key=_reference_key)
        object.__setattr__(self, "references", tuple(normalized))
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(reference.evidence_id for reference in normalized),
        )
        if self.expected_assertion is not None:
            object.__setattr__(
                self,
                "expected_assertion",
                _normalize_assertion_state(self.expected_assertion),
            )
        if self.expected_interval is not None:
            object.__setattr__(
                self,
                "expected_interval",
                _parse_interval(self.expected_interval),
            )

    @property
    def assertion(self) -> str | None:
        """Alias for the expected assertion state."""

        return self.expected_assertion

    @property
    def interval(self) -> _IntervalBounds | None:
        """Return the private normalized claim interval, when supplied."""

        return self.expected_interval  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        """Return a safe claim shape with no evidence content or dates."""

        return {
            "claim_id": self.claim_id,
            "evidence_ids": list(self.evidence_ids),
            "evidence_count": len(self.references),
            "expected_assertion": self.expected_assertion,
        }


@dataclass(frozen=True)
class AssertionRecord:
    """One normalized assertion record joined to a claim reference."""

    record_id: str
    assertion: object = "unknown"
    text_hash: str | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _required_id(self.record_id))
        object.__setattr__(
            self,
            "assertion",
            _normalize_assertion_state(self.assertion),
        )
        if self.text_hash is not None:
            object.__setattr__(
                self,
                "text_hash",
                _normalize_hash(self.text_hash, namespace="assertion"),
            )
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _required_id(self.source_id))

    @property
    def state(self) -> str:
        """Return the canonical assertion state."""

        return str(self.assertion)

    @property
    def assertion_state(self) -> str:
        """Return ``state`` under the explicit review-facing name."""

        return self.state

    def to_dict(self) -> dict[str, Any]:
        """Return identifier, state, and hash metadata only."""

        return {
            "record_id": self.record_id,
            "assertion": self.state,
            "text_hash": self.text_hash,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class TemporalRecord:
    """One normalized temporal record with private date bounds."""

    record_id: str
    interval: object | None = None
    text_hash: str | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _required_id(self.record_id))
        if self.interval is not None:
            object.__setattr__(self, "interval", _parse_interval(self.interval))
        if self.text_hash is not None:
            object.__setattr__(
                self,
                "text_hash",
                _normalize_hash(self.text_hash, namespace="temporal"),
            )
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _required_id(self.source_id))

    @property
    def interval_hash(self) -> str:
        """Return a stable hash of the interval without exposing its dates."""

        interval = self.interval
        if not isinstance(interval, _IntervalBounds):
            return _normalize_hash(self.record_id, namespace="temporal-record")
        return _hash_payload(
            "temporal-interval",
            {"lower": interval.lower.isoformat(), "upper": interval.upper.isoformat()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return temporal provenance without interval values."""

        return {
            "record_id": self.record_id,
            "interval_hash": self.interval_hash,
            "text_hash": self.text_hash,
            "source_id": self.source_id,
        }


TemporalEvidenceRecord = TemporalRecord
AssertionEvidenceRecord = AssertionRecord


@dataclass(frozen=True)
class SourceIntegrityRecord:
    """One local source-integrity decision for cited evidence."""

    record_id: str
    status: str | None = None
    verified: bool | None = None
    expected_hash: str | None = None
    actual_hash: str | None = None
    source_hash: str | None = None
    text_hash: str | None = None
    source_id: str | None = None
    integrity_status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _required_id(self.record_id))
        status = self.status if self.status is not None else self.integrity_status
        if status is not None:
            if not isinstance(status, str):
                raise TypeError("source integrity status must be a string")
            object.__setattr__(self, "status", _normalize_token(status))
        if self.verified is not None and not isinstance(self.verified, bool):
            raise TypeError("source integrity verified must be a boolean")
        for field_name in (
            "expected_hash",
            "actual_hash",
            "source_hash",
            "text_hash",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _normalize_hash(value, namespace="source-integrity"),
                )
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _required_id(self.source_id))

    @property
    def integrity_ok(self) -> bool:
        """Return whether the record proves an intact, verified source."""

        status = self.status
        if status in _BAD_INTEGRITY_STATUSES:
            return False
        if (
            self.expected_hash is not None
            and self.actual_hash is not None
            and self.expected_hash != self.actual_hash
        ):
            return False
        if self.verified is False:
            return False
        if self.verified is True or status in _GOOD_INTEGRITY_STATUSES:
            return True
        return (
            self.expected_hash is not None
            and self.actual_hash is not None
            and self.expected_hash == self.actual_hash
        )

    @property
    def hashes(self) -> tuple[str, ...]:
        """Return stable hashes in fixed field order."""

        return tuple(
            value
            for value in (
                self.expected_hash,
                self.actual_hash,
                self.source_hash,
                self.text_hash,
            )
            if value is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return integrity status and hashes without source content."""

        return {
            "record_id": self.record_id,
            "status": self.status,
            "verified": self.verified,
            "integrity_ok": self.integrity_ok,
            "hashes": list(self.hashes),
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class ClaimConflict:
    """One contradiction route containing only safe evidence references."""

    claim_id: str
    conflict_type: ClaimConflictType
    evidence_ids: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    hashes: tuple[str, ...] = ()
    review_route: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _required_id(self.claim_id))
        if self.conflict_type not in _CONFLICT_TYPES:
            raise ValueError("unsupported claim conflict type")
        object.__setattr__(
            self,
            "evidence_ids",
            _sorted_unique_ids(self.evidence_ids),
        )
        object.__setattr__(self, "record_ids", _sorted_unique_ids(self.record_ids))
        object.__setattr__(
            self,
            "hashes",
            tuple(
                sorted(
                    set(
                        _normalize_hash(value, namespace="conflict")
                        for value in self.hashes
                    )
                )
            ),
        )
        route = self.review_route or _ROUTE_BY_TYPE[self.conflict_type]
        if route not in CLAIM_REVIEW_ROUTES:
            raise ValueError("unsupported claim review route")
        object.__setattr__(self, "review_route", route)

    @property
    def kind(self) -> ClaimConflictType:
        """Return the conflict type under the shorter compatibility name."""

        return self.conflict_type

    @property
    def reason(self) -> str:
        """Return the explicit review route."""

        return str(self.review_route)

    def to_dict(self) -> dict[str, Any]:
        """Return the PHI-safe serialized contradiction."""

        return {
            "claim_id": self.claim_id,
            "conflict_type": self.conflict_type,
            "review_route": self.review_route,
            "evidence_ids": list(self.evidence_ids),
            "record_ids": list(self.record_ids),
            "hashes": list(self.hashes),
        }


@dataclass(frozen=True)
class ClaimReview:
    """Safe per-claim routing result."""

    claim_id: str
    review_state: ClaimReviewState
    review_routes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_hashes: tuple[str, ...]
    conflicts: tuple[ClaimConflict, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _required_id(self.claim_id))
        if self.review_state not in CLAIM_REVIEW_STATES:
            raise ValueError("unsupported claim review state")
        object.__setattr__(
            self,
            "review_routes",
            tuple(sorted(set(self.review_routes))),
        )
        if any(route not in CLAIM_REVIEW_ROUTES for route in self.review_routes):
            raise ValueError("unsupported claim review route")
        object.__setattr__(self, "evidence_ids", _sorted_unique_ids(self.evidence_ids))
        object.__setattr__(
            self,
            "evidence_hashes",
            tuple(
                sorted(
                    set(
                        _normalize_hash(value, namespace="claim-evidence")
                        for value in self.evidence_hashes
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "conflicts",
            tuple(sorted(self.conflicts, key=_conflict_key)),
        )
        if bool(self.conflicts) != (self.review_state == CLAIM_REVIEW_REQUIRED):
            raise ValueError("claim review state must match its conflict set")

    @property
    def requires_review(self) -> bool:
        """Return whether a human review route was emitted."""

        return self.review_state == CLAIM_REVIEW_REQUIRED

    @property
    def conflict_types(self) -> tuple[ClaimConflictType, ...]:
        """Return distinct conflict types in deterministic order."""

        return tuple(sorted({conflict.conflict_type for conflict in self.conflicts}))

    def to_dict(self) -> dict[str, Any]:
        """Return counts, identifiers, and hashes only."""

        return {
            "claim_id": self.claim_id,
            "review_state": self.review_state,
            "review_routes": list(self.review_routes),
            "evidence_count": len(self.evidence_ids),
            "evidence_ids": list(self.evidence_ids),
            "evidence_hashes": list(self.evidence_hashes),
            "conflict_count": len(self.conflicts),
            "conflict_types": list(self.conflict_types),
        }


@dataclass(frozen=True)
class ClaimConflictReport:
    """Deterministic aggregate of claim reviews and contradiction routes."""

    reviews: tuple[ClaimReview, ...]
    conflicts: tuple[ClaimConflict, ...]
    disclaimer: str = CLAIM_CONFLICT_ADVISORY
    schema_version: int = CLAIM_CONFLICT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reviews", tuple(sorted(self.reviews, key=_review_key))
        )
        object.__setattr__(
            self,
            "conflicts",
            tuple(sorted(self.conflicts, key=_conflict_key)),
        )
        expected_conflicts = tuple(
            conflict for review in self.reviews for conflict in review.conflicts
        )
        if expected_conflicts != self.conflicts:
            raise ValueError("claim conflict report conflicts must match claim reviews")

    @property
    def claims(self) -> tuple[ClaimReview, ...]:
        """Return reviews under the issue-facing plural alias."""

        return self.reviews

    @property
    def review_state(self) -> ClaimReviewState:
        """Return aggregate clear/review-required state."""

        return CLAIM_REVIEW_REQUIRED if self.conflicts else CLAIM_REVIEW_CLEAR

    @property
    def requires_review(self) -> bool:
        """Return whether any claim requires human review."""

        return bool(self.conflicts)

    @property
    def conflict_counts(self) -> dict[str, int]:
        """Return fixed-key contradiction counts."""

        counts = Counter(conflict.conflict_type for conflict in self.conflicts)
        return {
            conflict_type: counts.get(conflict_type, 0)
            for conflict_type in _CONFLICT_TYPES
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report with no raw evidence values."""

        payload = {
            "schema_version": self.schema_version,
            "review_state": self.review_state,
            "review_required": self.requires_review,
            "claims": [review.to_dict() for review in self.reviews],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "summary": {
                "claim_count": len(self.reviews),
                "review_required_count": sum(
                    review.requires_review for review in self.reviews
                ),
                "clear_count": sum(
                    not review.requires_review for review in self.reviews
                ),
                "conflict_count": len(self.conflicts),
                "conflict_counts": self.conflict_counts,
            },
            "disclaimer": self.disclaimer,
        }
        return payload

    def to_json(self) -> str:
        """Serialize the report with stable key ordering and no raw text."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


ClaimEvidenceReference = ClaimReference
SummaryClaim = ClaimRecord
ClaimReviewResult = ClaimReview
ClaimConflictReview = ClaimConflictReport


@dataclass(frozen=True)
class _EvidenceBundle:
    reference: ClaimReference
    evidence_hash: str
    assertion: AssertionRecord | None
    temporal: TemporalRecord | None
    integrity: SourceIntegrityRecord | None


def review_claim_conflicts(
    claims: Iterable[ClaimRecord | Mapping[str, Any]] | Mapping[str, Any],
    assertion_records: Iterable[AssertionRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    temporal_records: Iterable[TemporalRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    source_integrity_records: Iterable[SourceIntegrityRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    *,
    assertions: Iterable[AssertionRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    temporal: Iterable[TemporalRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    source_integrity: Iterable[SourceIntegrityRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
) -> ClaimConflictReport:
    """Review claim references against local assertion, temporal, and integrity records.

    Args:
        claims: Claim records or mappings.  A mapping may use ``evidence_ids``
            or ``references``; each reference may be a string evidence ID or a
            mapping with explicit record IDs.
        assertion_records: Optional records keyed by assertion/evidence ID.
            Distinct assertion states cited by one claim route it to review.
        temporal_records: Optional records keyed by temporal/evidence ID.
            Disjoint intervals, including intervals disjoint from a claim's
            expected interval, route it to review.
        source_integrity_records: Optional records keyed by source/evidence ID.
            Missing, unverified, failed, or hash-mismatched records route it to
            review.
        assertions, temporal, source_integrity: Keyword aliases for the three
            record collections.  They cannot be combined with their explicit
            counterparts.

    Returns:
        A stable :class:`ClaimConflictReport` containing only identifiers,
        counts, hashes, and explicit review routes.

    Raises:
        TypeError: If a collection or typed record has an unsupported shape.
        ValueError: If a claim or record has no stable identifier or contains
            conflicting duplicate identifiers.
    """

    assertion_records = _choose_alias(
        assertion_records,
        assertions,
        field_name="assertion records",
    )
    temporal_records = _choose_alias(
        temporal_records,
        temporal,
        field_name="temporal records",
    )
    source_integrity_records = _choose_alias(
        source_integrity_records,
        source_integrity,
        field_name="source integrity records",
    )

    normalized_claims = _normalize_claims(claims)
    assertion_index = _index_records(
        assertion_records,
        kind="assertion",
        coerce=_coerce_assertion_record,
    )
    temporal_index = _index_records(
        temporal_records,
        kind="temporal",
        coerce=_coerce_temporal_record,
    )
    integrity_index = _index_records(
        source_integrity_records,
        kind="source_integrity",
        coerce=_coerce_integrity_record,
    )

    reviews: list[ClaimReview] = []
    for claim in normalized_claims:
        review = _review_claim(
            claim,
            assertion_index=assertion_index,
            temporal_index=temporal_index,
            integrity_index=integrity_index,
            assertions_enabled=assertion_records is not None,
            temporal_enabled=temporal_records is not None,
            integrity_enabled=source_integrity_records is not None,
        )
        reviews.append(review)

    conflicts = tuple(conflict for review in reviews for conflict in review.conflicts)
    return ClaimConflictReport(reviews=tuple(reviews), conflicts=conflicts)


def review_claims(
    claims: Iterable[ClaimRecord | Mapping[str, Any]] | Mapping[str, Any],
    assertion_records: Iterable[AssertionRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    temporal_records: Iterable[TemporalRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    source_integrity_records: Iterable[SourceIntegrityRecord | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
    **aliases: Any,
) -> ClaimConflictReport:
    """Compatibility wrapper for :func:`review_claim_conflicts`."""

    supported = {"assertions", "temporal", "source_integrity"}
    unexpected = tuple(sorted(set(aliases) - supported))
    if unexpected:
        raise TypeError("unsupported claim review keyword")

    return review_claim_conflicts(
        claims,
        assertion_records=assertion_records,
        temporal_records=temporal_records,
        source_integrity_records=source_integrity_records,
        assertions=aliases.get("assertions"),
        temporal=aliases.get("temporal"),
        source_integrity=aliases.get("source_integrity"),
    )


detect_claim_conflicts = review_claim_conflicts
review_evidence_claims = review_claim_conflicts


def _review_claim(
    claim: ClaimRecord,
    *,
    assertion_index: Mapping[str, AssertionRecord],
    temporal_index: Mapping[str, TemporalRecord],
    integrity_index: Mapping[str, SourceIntegrityRecord],
    assertions_enabled: bool,
    temporal_enabled: bool,
    integrity_enabled: bool,
) -> ClaimReview:
    bundles: list[_EvidenceBundle] = []
    missing: dict[str, set[str]] = {
        "assertion": set(),
        "temporal": set(),
        "source_integrity": set(),
    }
    missing_records: set[str] = set()

    if not claim.references:
        conflict = ClaimConflict(
            claim_id=claim.claim_id,
            conflict_type="missing_evidence",
        )
        return ClaimReview(
            claim_id=claim.claim_id,
            review_state=CLAIM_REVIEW_REQUIRED,
            review_routes=(MISSING_EVIDENCE_ROUTE,),
            evidence_ids=(),
            evidence_hashes=(),
            conflicts=(conflict,),
        )

    for reference in claim.references:
        assertion = (
            _lookup_record(assertion_index, reference.assertion_record_id)
            if assertions_enabled
            else None
        )
        temporal = (
            _lookup_record(temporal_index, reference.temporal_record_id)
            if temporal_enabled
            else None
        )
        integrity = (
            _lookup_record(integrity_index, reference.integrity_record_id)
            if integrity_enabled
            else None
        )
        if assertions_enabled and assertion is None:
            missing["assertion"].add(reference.evidence_id)
            missing_records.add(reference.assertion_record_id)
        if temporal_enabled and temporal is None:
            missing["temporal"].add(reference.evidence_id)
            missing_records.add(reference.temporal_record_id)
        elif temporal_enabled and temporal.interval is None:
            missing["temporal"].add(reference.evidence_id)
            missing_records.add(temporal.record_id)
        if integrity_enabled and integrity is None:
            missing["source_integrity"].add(reference.evidence_id)
            missing_records.add(reference.integrity_record_id)
        bundles.append(
            _EvidenceBundle(
                reference=reference,
                evidence_hash=_evidence_hash(reference, assertion, temporal, integrity),
                assertion=assertion,
                temporal=temporal,
                integrity=integrity,
            )
        )

    conflicts: list[ClaimConflict] = []
    if missing_records:
        missing_evidence_ids = {
            evidence_id
            for evidence_ids in missing.values()
            for evidence_id in evidence_ids
        }
        conflicts.append(
            ClaimConflict(
                claim_id=claim.claim_id,
                conflict_type="missing_evidence",
                evidence_ids=tuple(missing_evidence_ids),
                record_ids=tuple(missing_records),
                hashes=tuple(
                    bundle.evidence_hash
                    for bundle in bundles
                    if bundle.reference.evidence_id in missing_evidence_ids
                ),
            )
        )

    assertion_conflict = _assertion_conflict(claim, bundles, assertions_enabled)
    if assertion_conflict is not None:
        conflicts.append(assertion_conflict)

    temporal_conflict = _temporal_conflict(claim, bundles, temporal_enabled)
    if temporal_conflict is not None:
        conflicts.append(temporal_conflict)

    integrity_conflict = _integrity_conflict(claim, bundles, integrity_enabled)
    if integrity_conflict is not None:
        conflicts.append(integrity_conflict)

    conflicts = sorted(conflicts, key=_conflict_key)
    routes = tuple(sorted({str(conflict.review_route) for conflict in conflicts}))
    return ClaimReview(
        claim_id=claim.claim_id,
        review_state=CLAIM_REVIEW_REQUIRED if conflicts else CLAIM_REVIEW_CLEAR,
        review_routes=routes,
        evidence_ids=claim.evidence_ids,
        evidence_hashes=tuple(bundle.evidence_hash for bundle in bundles),
        conflicts=tuple(conflicts),
    )


def _assertion_conflict(
    claim: ClaimRecord,
    bundles: Sequence[_EvidenceBundle],
    enabled: bool,
) -> ClaimConflict | None:
    if not enabled:
        return None
    rows = [
        (bundle.reference, bundle.assertion)
        for bundle in bundles
        if bundle.assertion is not None
    ]
    states = [
        (reference, record)
        for reference, record in rows
        if record.state not in _UNKNOWN_STATES
    ]
    expected_states = {
        reference.expected_assertion or claim.expected_assertion
        for reference, _record in rows
        if reference.expected_assertion or claim.expected_assertion
    }
    conflict_rows: list[tuple[ClaimReference, AssertionRecord]] = []
    if expected_states:
        for reference, record in states:
            expected = reference.expected_assertion or claim.expected_assertion
            if expected is not None and _assertions_incompatible(
                expected, record.state
            ):
                conflict_rows.append((reference, record))
        if (
            len(expected_states) > 1
            or len({record.state for _ref, record in states}) > 1
        ):
            conflict_rows = list(states)
    elif len({record.state for _reference, record in states}) > 1:
        conflict_rows = list(states)
    if not conflict_rows:
        return None
    return ClaimConflict(
        claim_id=claim.claim_id,
        conflict_type="assertion",
        evidence_ids=tuple(
            reference.evidence_id for reference, _record in conflict_rows
        ),
        record_ids=tuple(record.record_id for _reference, record in conflict_rows),
        hashes=tuple(
            bundle.evidence_hash
            for bundle in bundles
            if bundle.reference.evidence_id
            in {reference.evidence_id for reference, _record in conflict_rows}
        ),
    )


def _temporal_conflict(
    claim: ClaimRecord,
    bundles: Sequence[_EvidenceBundle],
    enabled: bool,
) -> ClaimConflict | None:
    if not enabled:
        return None
    rows = [
        (bundle.reference, bundle.temporal)
        for bundle in bundles
        if bundle.temporal is not None and bundle.temporal.interval is not None
    ]
    intervals = [
        (reference, record)
        for reference, record in rows
        if isinstance(record.interval, _IntervalBounds)
    ]
    conflict_rows: list[tuple[ClaimReference, TemporalRecord]] = []
    expected_intervals = [
        reference.expected_interval or claim.expected_interval
        for reference, _record in rows
        if reference.expected_interval is not None
        or claim.expected_interval is not None
    ]
    if expected_intervals:
        expected = expected_intervals[0]
        if isinstance(expected, _IntervalBounds):
            conflict_rows.extend(
                (reference, record)
                for reference, record in intervals
                if not _intervals_overlap(record.interval, expected)
            )
    for index, (left_reference, left_record) in enumerate(intervals):
        for right_reference, right_record in intervals[index + 1 :]:
            if not _intervals_overlap(left_record.interval, right_record.interval):
                conflict_rows.extend(
                    ((left_reference, left_record), (right_reference, right_record))
                )
    if not conflict_rows:
        return None
    unique_rows = {
        (reference.evidence_id, record.record_id): (reference, record)
        for reference, record in conflict_rows
    }
    return ClaimConflict(
        claim_id=claim.claim_id,
        conflict_type="temporal",
        evidence_ids=tuple(
            reference.evidence_id for reference, _record in unique_rows.values()
        ),
        record_ids=tuple(
            record.record_id for _reference, record in unique_rows.values()
        ),
        hashes=tuple(
            bundle.evidence_hash
            for bundle in bundles
            if bundle.reference.evidence_id
            in {reference.evidence_id for reference, _record in unique_rows.values()}
        ),
    )


def _integrity_conflict(
    claim: ClaimRecord,
    bundles: Sequence[_EvidenceBundle],
    enabled: bool,
) -> ClaimConflict | None:
    if not enabled:
        return None
    invalid = [
        bundle
        for bundle in bundles
        if bundle.integrity is not None
        and (
            not bundle.integrity.integrity_ok
            or _reference_hash_mismatch(bundle.reference, bundle.integrity)
        )
    ]
    if not invalid:
        return None
    return ClaimConflict(
        claim_id=claim.claim_id,
        conflict_type="source_integrity",
        evidence_ids=tuple(bundle.reference.evidence_id for bundle in invalid),
        record_ids=tuple(
            bundle.integrity.record_id for bundle in invalid if bundle.integrity
        ),
        hashes=tuple(
            value
            for bundle in invalid
            for value in (
                bundle.evidence_hash,
                *(bundle.integrity.hashes if bundle.integrity else ()),
            )
        ),
    )


def _reference_hash_mismatch(
    reference: ClaimReference,
    integrity: SourceIntegrityRecord,
) -> bool:
    if reference.text_hash is None:
        return False
    known = {
        value
        for value in (
            integrity.expected_hash,
            integrity.actual_hash,
            integrity.source_hash,
            integrity.text_hash,
        )
        if value is not None
    }
    return bool(known) and reference.text_hash not in known


def _assertions_incompatible(expected: str, actual: str) -> bool:
    if expected in _UNKNOWN_STATES or actual in _UNKNOWN_STATES:
        return False
    return expected != actual


def _intervals_overlap(left: object, right: object) -> bool:
    return (
        isinstance(left, _IntervalBounds)
        and isinstance(right, _IntervalBounds)
        and left.lower <= right.upper
        and right.lower <= left.upper
    )


def _normalize_claims(
    claims: Iterable[ClaimRecord | Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[ClaimRecord, ...]:
    items = _collection_items(claims, kind="claim")
    normalized: dict[str, ClaimRecord] = {}
    for fallback_id, raw in items:
        claim = raw if isinstance(raw, ClaimRecord) else _coerce_claim(raw, fallback_id)
        previous = normalized.get(claim.claim_id)
        if previous is not None and previous != claim:
            raise ValueError("claims contain conflicting duplicate identifiers")
        normalized[claim.claim_id] = claim
    return tuple(sorted(normalized.values(), key=lambda claim: claim.claim_id))


def _coerce_claim(
    raw: Mapping[str, Any] | object, fallback_id: str | None
) -> ClaimRecord:
    if fallback_id is not None and not isinstance(raw, Mapping):
        return ClaimRecord(
            claim_id=fallback_id,
            references=_reference_items(raw),
        )
    if not isinstance(raw, Mapping):
        raise TypeError("claims must contain mappings or ClaimRecord values")
    claim_id = _field(raw, "claim_id", "summary_id", "id") or fallback_id
    if claim_id is None:
        raise ValueError("claim requires a stable identifier")
    raw_references = _first_field(
        raw,
        "references",
        "evidence_references",
        "evidence_refs",
        "evidence",
        "cited_evidence",
    )
    evidence_ids = _first_field(raw, "evidence_ids", "cited_evidence_ids")
    if raw_references is None and evidence_ids is None:
        direct_evidence = _field(raw, "evidence_id", "reference_id")
        raw_references = () if direct_evidence is None else (direct_evidence,)
    references = _reference_items(raw_references)
    if evidence_ids is not None:
        references = (*references, *_reference_items(evidence_ids))
    expected_assertion = _field(
        raw,
        "expected_assertion",
        "claim_assertion",
        "assertion",
        "assertion_status",
        "status",
        "polarity",
    )
    expected_interval = _first_field(
        raw,
        "expected_interval",
        "claim_interval",
        "temporal_interval",
    )
    if expected_interval is None and (
        _field(raw, "start_date", "start_time") is not None
        or _field(raw, "end_date", "end_time") is not None
    ):
        expected_interval = {
            "start": _field(raw, "start_date", "start_time"),
            "end": _field(raw, "end_date", "end_time"),
        }
    return ClaimRecord(
        claim_id=str(claim_id),
        references=references,
        expected_assertion=expected_assertion,
        expected_interval=expected_interval,
    )


def _coerce_reference(raw: ClaimReference | str | Mapping[str, Any]) -> ClaimReference:
    if isinstance(raw, ClaimReference):
        return raw
    if isinstance(raw, str):
        return ClaimReference(evidence_id=raw)
    if not isinstance(raw, Mapping):
        raise TypeError(
            "claim references must be IDs, mappings, or ClaimReference values"
        )
    evidence_id = _field(
        raw, "evidence_id", "reference_id", "source_id", "id", "record_id"
    )
    if evidence_id is None:
        raise ValueError("claim reference requires a stable evidence identifier")
    expected_interval = _first_field(raw, "expected_interval", "claim_interval")
    return ClaimReference(
        evidence_id=str(evidence_id),
        assertion_id=_as_optional_id(
            _field(raw, "assertion_id", "assertion_record_id")
        ),
        temporal_id=_as_optional_id(_field(raw, "temporal_id", "temporal_record_id")),
        source_id=_as_optional_id(_field(raw, "source_id", "source_record_id")),
        source_integrity_id=_as_optional_id(
            _field(raw, "source_integrity_id", "integrity_id")
        ),
        text_hash=_optional_hash(raw, "text_hash", "source_hash"),
        expected_assertion=_field(
            raw,
            "expected_assertion",
            "assertion",
            "assertion_status",
            "status",
            "polarity",
        ),
        expected_interval=expected_interval,
    )


def _coerce_assertion_record(
    raw: AssertionRecord | Mapping[str, Any] | object,
    fallback_id: str | None,
) -> AssertionRecord:
    if isinstance(raw, AssertionRecord):
        return raw
    if isinstance(raw, str):
        if fallback_id is None:
            raise ValueError("assertion record requires a stable identifier")
        return AssertionRecord(record_id=fallback_id, assertion=raw)
    if not isinstance(raw, Mapping):
        record_id = _record_identifier(raw, fallback_id, kind="assertion")
        assertion = _field(
            raw,
            "assertion",
            "assertion_state",
            "assertion_status",
            "negation",
            "state",
            "status",
            "polarity",
        )
        nested = _field(raw, "context", "clinical_assertion")
        if assertion is None:
            assertion = nested if nested is not None else "unknown"
        return AssertionRecord(
            record_id=record_id,
            assertion=assertion,
            text_hash=_optional_hash(
                raw,
                "text_hash",
                "source_hash",
                "content_hash",
                "text",
                "surface",
            ),
            source_id=_as_optional_id(_field(raw, "source_id", "evidence_id")),
        )
    record_id = _record_identifier(raw, fallback_id, kind="assertion")
    assertion = _first_field(
        raw,
        "assertion",
        "assertion_state",
        "assertion_status",
        "negation",
        "state",
        "status",
        "polarity",
        "value",
    )
    nested = _field(raw, "context", "clinical_assertion")
    if assertion is None and nested is not None:
        assertion = nested
    if assertion is None:
        assertion = "unknown"
    return AssertionRecord(
        record_id=record_id,
        assertion=assertion,
        text_hash=_optional_hash(
            raw, "text_hash", "source_hash", "content_hash", "text", "surface"
        ),
        source_id=_as_optional_id(_field(raw, "source_id", "evidence_id")),
    )


def _coerce_temporal_record(
    raw: TemporalRecord | Mapping[str, Any] | object,
    fallback_id: str | None,
) -> TemporalRecord:
    if isinstance(raw, TemporalRecord):
        return raw
    if isinstance(raw, (str, date, datetime)):
        if fallback_id is None:
            raise ValueError("temporal record requires a stable identifier")
        return TemporalRecord(record_id=fallback_id, interval=raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)):
        if fallback_id is None:
            raise ValueError("temporal record requires a stable identifier")
        return TemporalRecord(record_id=fallback_id, interval=raw)
    if not isinstance(raw, Mapping):
        record_id = _record_identifier(raw, fallback_id, kind="temporal")
        interval = _field(raw, "interval", "temporal_interval")
        if interval is None:
            start = _field(raw, "start_date", "start_time", "start")
            end = _field(raw, "end_date", "end_time", "end")
            if _date_value(start) is not None or _date_value(end) is not None:
                interval = {"start": start, "end": end}
        return TemporalRecord(
            record_id=record_id,
            interval=interval,
            text_hash=_optional_hash(
                raw,
                "text_hash",
                "source_hash",
                "content_hash",
                "text",
                "surface",
            ),
            source_id=_as_optional_id(_field(raw, "source_id", "evidence_id")),
        )
    record_id = _record_identifier(raw, fallback_id, kind="temporal")
    interval = _first_field(raw, "interval", "temporal_interval", "normalized_interval")
    if interval is None:
        interval_value = _field(raw, "normalized_value", "value", "date")
        if interval_value is not None and not isinstance(interval_value, (int, float)):
            interval = interval_value
    raw_start = _field(raw, "start_date", "start_time", "start")
    raw_end = _field(raw, "end_date", "end_time", "end")
    raw_lower = _field(raw, "lower_bound", "lower")
    raw_upper = _field(raw, "upper_bound", "upper")
    if interval is None and (
        _date_value(raw_start) is not None
        or _date_value(raw_end) is not None
        or _date_value(raw_lower) is not None
        or _date_value(raw_upper) is not None
    ):
        interval = {
            "start": raw_start,
            "end": raw_end,
            "lower_bound": raw_lower,
            "upper_bound": raw_upper,
        }
    return TemporalRecord(
        record_id=record_id,
        interval=interval,
        text_hash=_optional_hash(
            raw, "text_hash", "source_hash", "content_hash", "text", "surface"
        ),
        source_id=_as_optional_id(_field(raw, "source_id", "evidence_id")),
    )


def _coerce_integrity_record(
    raw: SourceIntegrityRecord | Mapping[str, Any] | object,
    fallback_id: str | None,
) -> SourceIntegrityRecord:
    if isinstance(raw, SourceIntegrityRecord):
        return raw
    if isinstance(raw, str):
        if fallback_id is None:
            raise ValueError("source integrity record requires a stable identifier")
        return SourceIntegrityRecord(record_id=fallback_id, status=raw)
    if not isinstance(raw, Mapping):
        record_id = _record_identifier(raw, fallback_id, kind="source integrity")
        return SourceIntegrityRecord(
            record_id=record_id,
            status=_field(
                raw,
                "status",
                "integrity_status",
                "integrity_state",
                "integrity",
                "state",
            ),
            verified=_optional_bool(_field(raw, "verified", "valid", "integrity_ok")),
            expected_hash=_optional_hash(raw, "expected_hash", "expected_sha256"),
            actual_hash=_optional_hash(
                raw,
                "actual_hash",
                "observed_hash",
                "content_hash",
                "integrity_hash",
                "hash",
            ),
            source_hash=_optional_hash(raw, "source_hash", "record_hash"),
            text_hash=_optional_hash(raw, "text_hash", "text", "surface"),
            source_id=_as_optional_id(_field(raw, "source_id", "evidence_id")),
        )
    record_id = _record_identifier(raw, fallback_id, kind="source integrity")
    return SourceIntegrityRecord(
        record_id=record_id,
        status=_field(
            raw,
            "status",
            "integrity_status",
            "integrity_state",
            "integrity",
            "state",
        ),
        verified=_optional_bool(_field(raw, "verified", "valid", "integrity_ok")),
        expected_hash=_optional_hash(raw, "expected_hash", "expected_sha256"),
        actual_hash=_optional_hash(
            raw,
            "actual_hash",
            "observed_hash",
            "content_hash",
            "integrity_hash",
            "hash",
        ),
        source_hash=_optional_hash(raw, "source_hash", "record_hash"),
        text_hash=_optional_hash(raw, "text_hash", "text", "surface"),
        source_id=_as_optional_id(_field(raw, "source_id", "evidence_id")),
    )


def _index_records(
    records: Iterable[object] | Mapping[str, Any] | None,
    *,
    kind: str,
    coerce: Any,
) -> dict[str, Any]:
    if records is None:
        return {}
    index: dict[str, Any] = {}
    for fallback_id, raw in _collection_items(records, kind=kind):
        record = coerce(raw, fallback_id)
        _insert_index(index, record.record_id, record)
        source_id = getattr(record, "source_id", None)
        if source_id is not None:
            _insert_index(index, source_id, record)
    return index


def _insert_index(index: dict[str, Any], key: str, record: Any) -> None:
    previous = index.get(key)
    if previous is not None and previous != record:
        raise ValueError("records contain conflicting duplicate identifiers")
    index[key] = record


def _lookup_record(index: Mapping[str, Any], key: str) -> Any | None:
    return index.get(key)


def _evidence_hash(
    reference: ClaimReference,
    assertion: AssertionRecord | None,
    temporal: TemporalRecord | None,
    integrity: SourceIntegrityRecord | None,
) -> str:
    if reference.text_hash is not None:
        return reference.text_hash
    for record in (integrity, assertion, temporal):
        record_hash = getattr(record, "text_hash", None) if record is not None else None
        if record_hash is not None:
            return record_hash
    return _hash_payload(
        "claim-evidence",
        {
            "evidence_id": reference.evidence_id,
            "assertion_id": reference.assertion_record_id,
            "temporal_id": reference.temporal_record_id,
            "integrity_id": reference.integrity_record_id,
        },
    )


def _collection_items(
    value: object, *, kind: str
) -> tuple[tuple[str | None, object], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if _looks_like_record(value, kind=kind):
            return ((None, value),)
        return tuple(
            (str(key), item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{kind} collection must not be text")
    if isinstance(value, Iterable):
        return tuple((None, item) for item in value)
    return ((None, value),)


def _looks_like_record(value: Mapping[str, Any], *, kind: str) -> bool:
    if any(key in value for key in _ID_KEYS):
        return True
    if kind == "claim":
        return any(
            key in value
            for key in (
                "references",
                "evidence_references",
                "evidence_refs",
                "evidence_ids",
                "evidence",
                "claim_assertion",
            )
        )
    if kind == "assertion":
        return any(
            key in value
            for key in (
                "assertion",
                "assertion_state",
                "assertion_status",
                "negation",
                "state",
                "status",
                "polarity",
            )
        )
    if kind == "temporal":
        return any(
            key in value
            for key in (
                "interval",
                "start_date",
                "end_date",
                "start",
                "end",
                "normalized_value",
                "value",
                "date",
            )
        )
    return any(
        key in value
        for key in (
            "verified",
            "valid",
            "integrity_ok",
            "status",
            "integrity_status",
            "integrity_state",
            "integrity",
            "expected_hash",
            "actual_hash",
        )
    )


def _reference_items(
    value: object,
) -> tuple[ClaimReference | str | Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if _looks_like_reference(value):
            return (value,)
        return tuple(
            ({"evidence_id": str(key), **(item if isinstance(item, Mapping) else {})})
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (str, ClaimReference)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)  # type: ignore[return-value]


def _looks_like_reference(value: Mapping[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "evidence_id",
            "reference_id",
            "assertion_id",
            "temporal_id",
            "source_integrity_id",
            "id",
        )
    )


def _field(value: object, *names: str) -> object | None:
    if isinstance(value, Mapping):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _first_field(value: object, *names: str) -> object | None:
    return _field(value, *names)


def _record_identifier(value: object, fallback_id: str | None, *, kind: str) -> str:
    identifier_keys = {
        "assertion": (
            "record_id",
            "assertion_id",
            "evidence_id",
            "span_id",
            "entity_key",
            "id",
            "source_id",
        ),
        "temporal": (
            "record_id",
            "temporal_id",
            "evidence_id",
            "span_id",
            "id",
            "source_id",
        ),
        "source integrity": (
            "record_id",
            "source_integrity_id",
            "source_id",
            "evidence_id",
            "id",
        ),
    }.get(kind, _ID_KEYS)
    identifier = _field(value, *identifier_keys)
    if identifier is None:
        identifier = fallback_id
    if identifier is None:
        raise ValueError(f"{kind} record requires a stable identifier")
    return str(identifier)


def _choose_alias(primary: Any, alias: Any, *, field_name: str) -> Any:
    if primary is not None and alias is not None:
        raise ValueError(f"provide one {field_name} collection")
    return primary if primary is not None else alias


def _required_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("identifiers must be strings")
    normalized = value.strip()
    if not normalized:
        raise ValueError("identifiers must not be empty")
    return normalized


def _as_optional_id(value: object | None) -> str | None:
    return None if value is None else _required_id(value)


def _sorted_unique_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({_required_id(value) for value in values}))


def _normalize_token(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("state values must be strings")
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _normalize_assertion_state(value: object) -> str:
    if isinstance(value, bool):
        return "affirmed" if value else "negated"
    nested = _field(value, "negation", "polarity", "status", "state", "assertion")
    if nested is not None and nested is not value:
        state = _normalize_assertion_state(nested)
        if state not in _UNKNOWN_STATES:
            return state
    temporality = _field(value, "temporality")
    if (
        isinstance(temporality, str)
        and _normalize_token(temporality) in _HYPOTHETICAL_STATES
    ):
        return "hypothetical"
    certainty = _field(value, "certainty")
    if isinstance(certainty, str):
        certainty_token = _normalize_token(certainty)
        if certainty_token in _UNCERTAIN_STATES:
            return "uncertain"
        if certainty_token in {"certain", "affirmed"}:
            return "affirmed"
    if value is None:
        return "unknown"
    if isinstance(value, Mapping) or not isinstance(value, str):
        return "unknown"
    token = _normalize_token(value)
    if token in _AFFIRMED_STATES:
        return "affirmed"
    if token in _NEGATED_STATES:
        return "negated"
    if token in _UNCERTAIN_STATES:
        return "uncertain"
    if token in _HYPOTHETICAL_STATES:
        return "hypothetical"
    if token in _UNKNOWN_STATES:
        return "unknown"
    return token


def _parse_interval(value: object) -> _IntervalBounds:
    if isinstance(value, _IntervalBounds):
        return value
    if isinstance(value, Mapping):
        nested = _first_field(
            value, "interval", "temporal_interval", "normalized_interval"
        )
        if nested is not None and nested is not value:
            return _parse_interval(nested)
        lower = _date_value(_field(value, "lower_bound", "lower"))
        upper = _date_value(_field(value, "upper_bound", "upper"))
        start = _date_value(_field(value, "start_date", "start", "from"))
        end = _date_value(_field(value, "end_date", "end", "to"))
        if start is None and end is None:
            value_text = _field(value, "value", "normalized_value")
            if value_text is not None:
                return _parse_interval(value_text)
        lower = lower or start
        upper = upper or end or start
        if lower is None or upper is None:
            raise ValueError("temporal interval requires parseable bounds")
        return _IntervalBounds(lower=min(lower, upper), upper=max(lower, upper))
    if isinstance(value, (date, datetime)):
        parsed = _date_value(value)
        if parsed is None:
            raise ValueError("temporal interval requires a parseable date")
        return _IntervalBounds(parsed, parsed)
    if isinstance(value, str):
        normalized = value.strip()
        match = _DATE_INTERVAL_RE.fullmatch(normalized)
        if match:
            start = _date_value(match.group("start"))
            end = _date_value(match.group("end"))
            if start is not None and end is not None:
                return _IntervalBounds(min(start, end), max(start, end))
        parsed = _date_value(normalized)
        if parsed is not None:
            return _IntervalBounds(parsed, parsed)
        raise ValueError("temporal interval requires parseable bounds")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("temporal interval requires two bounds")
        start = _date_value(value[0])
        end = _date_value(value[1])
        if start is None or end is None:
            raise ValueError("temporal interval requires parseable bounds")
        return _IntervalBounds(min(start, end), max(start, end))
    start = _date_value(_field(value, "start", "start_date"))
    end = _date_value(_field(value, "end", "end_date"))
    if start is not None and end is not None:
        return _IntervalBounds(min(start, end), max(start, end))
    raise ValueError("temporal interval requires parseable bounds")


def _date_value(value: object | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        if "T" in normalized:
            return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def _normalize_hash(value: object, *, namespace: str) -> str:
    if not isinstance(value, str):
        raise TypeError("hash values must be strings")
    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("hash values must not be empty")
    if _HASH_RE.fullmatch(normalized):
        return normalized
    # Callers sometimes provide the source surface to a ``text_hash`` field in
    # a synthetic fixture.  Hashing the content directly keeps that value
    # comparable across the claim, assertion, and integrity records while the
    # public output still contains only the digest.
    return hash_text(value)


def _optional_hash(value: object, *names: str) -> str | None:
    for name in names:
        candidate = _field(value, name)
        if candidate is not None:
            return _normalize_hash(candidate, namespace=name)
    return None


def _optional_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("boolean integrity fields must be booleans")
    return value


def _hash_payload(namespace: str, payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        {str(key): payload[key] for key in sorted(payload, key=str)},
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hash_text(f"{namespace}\0{serialized}")


def _reference_key(reference: ClaimReference) -> tuple[str, str, str, str]:
    return (
        reference.evidence_id,
        reference.assertion_record_id,
        reference.temporal_record_id,
        reference.integrity_record_id,
    )


def _conflict_key(conflict: ClaimConflict) -> tuple[Any, ...]:
    return (
        conflict.claim_id,
        _CONFLICT_TYPES.index(conflict.conflict_type),
        conflict.evidence_ids,
        conflict.record_ids,
    )


def _review_key(review: ClaimReview) -> tuple[str]:
    return (review.claim_id,)


__all__ = [
    "ASSERTION_CONFLICT_ROUTE",
    "AssertionEvidenceRecord",
    "AssertionRecord",
    "CLAIM_CONFLICT_ADVISORY",
    "CLAIM_CONFLICT_SCHEMA_VERSION",
    "CLAIM_CONFLICT_TYPES",
    "CLAIM_REVIEW_CLEAR",
    "CLAIM_REVIEW_REQUIRED",
    "CLAIM_REVIEW_ROUTES",
    "CLAIM_REVIEW_STATES",
    "ClaimConflict",
    "ClaimConflictReport",
    "ClaimConflictReview",
    "ClaimConflictType",
    "ClaimEvidenceReference",
    "ClaimReference",
    "ClaimReview",
    "ClaimReviewResult",
    "ClaimReviewState",
    "ClaimRecord",
    "MISSING_EVIDENCE_ROUTE",
    "SOURCE_INTEGRITY_CONFLICT_ROUTE",
    "SummaryClaim",
    "TEMPORAL_CONFLICT_ROUTE",
    "TemporalEvidenceRecord",
    "TemporalRecord",
    "detect_claim_conflicts",
    "review_claim_conflicts",
    "review_claims",
    "review_evidence_claims",
]
