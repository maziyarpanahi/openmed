"""Guarded, PHI-free evidence packets for downstream clinical reasoning.

The packet boundary accepts only synthetic evidence references that have been
reviewed, explicitly verified, bound to a policy fingerprint, and anchored by
half-open source offsets.  It never stores source text.  Invalid input is
discarded with a stable, counts-only rejection report so callers can audit
shape quality without copying sensitive values into logs or reports.

This module is intentionally local-first.  Fingerprints are computed with
local canonical JSON and no network or external service is consulted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from openmed.core.audit import stable_hash

__all__ = [
    "EVIDENCE_PACKET_KIND",
    "EVIDENCE_PACKET_SCHEMA_VERSION",
    "REVIEW_STATE_REVIEWED",
    "REVIEW_STATE_VALUES",
    "REJECTION_CATEGORIES",
    "REJECTION_DUPLICATE_REFERENCE",
    "REJECTION_INVALID_POLICY_FINGERPRINT",
    "REJECTION_INVALID_REFERENCE",
    "REJECTION_INVALID_REVIEW_STATE",
    "REJECTION_INVALID_SOURCE_OFFSET",
    "REJECTION_NOT_SYNTHETIC",
    "REJECTION_POLICY_MISMATCH",
    "REJECTION_RAW_TEXT",
    "REJECTION_UNVERIFIED",
    "EvidencePacketValidationError",
    "EvidenceValidationError",
    "EvidenceReference",
    "EvidenceRejectionReport",
    "EvidencePacket",
    "fingerprint_policy",
    "compute_policy_fingerprint",
    "build_evidence_packet",
    "package_evidence",
    "create_evidence_packet",
    "validate_evidence_packet",
]

EVIDENCE_PACKET_SCHEMA_VERSION = 1
EVIDENCE_PACKET_KIND = "clinical_evidence_packet"

REVIEW_STATE_REVIEWED: Literal["reviewed"] = "reviewed"
REVIEW_STATE_VALUES: tuple[str, ...] = (REVIEW_STATE_REVIEWED,)

REJECTION_RAW_TEXT = "raw_text"
REJECTION_UNVERIFIED = "unverified"
REJECTION_NOT_SYNTHETIC = "not_synthetic"
REJECTION_INVALID_REVIEW_STATE = "invalid_review_state"
REJECTION_INVALID_POLICY_FINGERPRINT = "invalid_policy_fingerprint"
REJECTION_POLICY_MISMATCH = "policy_mismatch"
REJECTION_INVALID_SOURCE_OFFSET = "invalid_source_offset"
REJECTION_DUPLICATE_REFERENCE = "duplicate_reference"
REJECTION_INVALID_REFERENCE = "invalid_reference"

# The order is part of the report contract.  A record receives one category,
# selected by the validation order in ``EvidenceReference.from_dict``.
REJECTION_CATEGORIES: tuple[str, ...] = (
    REJECTION_RAW_TEXT,
    REJECTION_UNVERIFIED,
    REJECTION_NOT_SYNTHETIC,
    REJECTION_INVALID_REVIEW_STATE,
    REJECTION_INVALID_POLICY_FINGERPRINT,
    REJECTION_POLICY_MISMATCH,
    REJECTION_INVALID_SOURCE_OFFSET,
    REJECTION_DUPLICATE_REFERENCE,
    REJECTION_INVALID_REFERENCE,
)

_POLICY_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_RAW_TEXT_KEYS = frozenset(
    {
        "claim",
        "content",
        "excerpt",
        "raw_text",
        "source_text",
        "surface",
        "text",
        "value",
    }
)
_SYNTHETIC_PREFIXES = ("fixture:", "fixture-", "synthetic:", "synthetic-")


class EvidencePacketValidationError(ValueError):
    """Raised when a reference or packet violates the evidence boundary.

    ``category`` is deliberately the only diagnostic detail retained.  The
    offending record and its values are never interpolated into the exception
    message.
    """

    def __init__(self, category: str) -> None:
        if category not in REJECTION_CATEGORIES:
            category = REJECTION_INVALID_REFERENCE
        self.category = category
        super().__init__(f"evidence record rejected: {category}")


# Short compatibility name for callers that prefer the general validation
# terminology used by other clinical schemas.
EvidenceValidationError = EvidencePacketValidationError


def _reject(category: str) -> EvidencePacketValidationError:
    return EvidencePacketValidationError(category)


def _required_identifier(value: Any) -> str:
    if not isinstance(value, str):
        raise _reject(REJECTION_INVALID_REFERENCE)
    normalized = value.strip()
    if not _SAFE_IDENTIFIER_RE.fullmatch(normalized):
        raise _reject(REJECTION_INVALID_REFERENCE)
    return normalized


def _validate_policy_fingerprint(value: Any) -> str:
    if not isinstance(value, str):
        raise _reject(REJECTION_INVALID_POLICY_FINGERPRINT)
    normalized = value.strip().lower()
    if not _POLICY_FINGERPRINT_RE.fullmatch(normalized):
        raise _reject(REJECTION_INVALID_POLICY_FINGERPRINT)
    return normalized


def _validate_offsets(
    start: Any, end: Any, source_length: Any = None
) -> tuple[int, int]:
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise _reject(REJECTION_INVALID_SOURCE_OFFSET)
    if start < 0 or end <= start:
        raise _reject(REJECTION_INVALID_SOURCE_OFFSET)
    if source_length is not None:
        if (
            isinstance(source_length, bool)
            or not isinstance(source_length, int)
            or source_length < 0
            or end > source_length
        ):
            raise _reject(REJECTION_INVALID_SOURCE_OFFSET)
    return start, end


def _contains_raw_text(value: Any) -> bool:
    """Return whether a candidate contains a forbidden text-bearing key."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _RAW_TEXT_KEYS:
                return True
            if _contains_raw_text(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_raw_text(child) for child in value)
    return False


def _is_synthetic_marker(value: Any, reference_id: Any) -> bool:
    if value is True:
        return True
    if not isinstance(reference_id, str):
        return False
    return reference_id.strip().lower().startswith(_SYNTHETIC_PREFIXES)


def _extract_offset_values(payload: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    offset = None
    for key in ("source_offset", "source_span", "offset", "span"):
        if key in payload:
            offset = payload[key]
            break

    if isinstance(offset, Mapping):
        return (
            offset.get("start"),
            offset.get("end"),
            offset.get("source_length", offset.get("document_length")),
        )
    if isinstance(offset, (list, tuple)) and len(offset) == 2:
        return (
            offset[0],
            offset[1],
            payload.get("source_length", payload.get("document_length")),
        )
    return (
        payload.get("start"),
        payload.get("end"),
        payload.get("source_length", payload.get("document_length")),
    )


@dataclass(frozen=True)
class EvidenceReference:
    """One reviewed, verified, synthetic, offset-only evidence reference.

    ``start`` and ``end`` are half-open character offsets.  No text or opaque
    payload is accepted by this type; a caller that needs source content must
    resolve these offsets in its own controlled review surface.
    """

    reference_id: str
    start: int
    end: int
    review_state: str
    policy_fingerprint: str
    source_id: str | None = None
    synthetic: Literal[True] = True
    verified: Literal[True] = True

    def __post_init__(self) -> None:
        reference_id = _required_identifier(self.reference_id)
        source_id = (
            reference_id
            if self.source_id is None
            else _required_identifier(self.source_id)
        )
        if self.synthetic is not True:
            raise _reject(REJECTION_NOT_SYNTHETIC)
        if self.verified is not True:
            raise _reject(REJECTION_UNVERIFIED)
        if self.review_state not in REVIEW_STATE_VALUES:
            raise _reject(REJECTION_INVALID_REVIEW_STATE)
        policy_fingerprint = _validate_policy_fingerprint(self.policy_fingerprint)
        start, end = _validate_offsets(self.start, self.end)

        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "policy_fingerprint", policy_fingerprint)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def evidence_id(self) -> str:
        """Return the stable evidence identifier alias."""

        return self.reference_id

    @property
    def source_offset(self) -> tuple[int, int]:
        """Return the half-open source offset pair."""

        return self.start, self.end

    @property
    def offset(self) -> tuple[int, int]:
        """Return ``source_offset`` under the shorter compatibility name."""

        return self.source_offset

    def to_dict(self) -> dict[str, Any]:
        """Return only the safe, deterministic reference representation."""

        return {
            "reference_id": self.reference_id,
            "source_id": self.source_id,
            "start": self.start,
            "end": self.end,
            "review_state": self.review_state,
            "policy_fingerprint": self.policy_fingerprint,
            "synthetic": True,
            "verified": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceReference":
        """Validate and build a reference from a mapping.

        The parser accepts ``reference_id`` or ``evidence_id`` and common
        offset containers to make migration from upstream span records simple.
        Unknown fields are ignored unless they are text-bearing, in which case
        the record is rejected as ``raw_text``.
        """

        if not isinstance(payload, Mapping):
            raise _reject(REJECTION_INVALID_REFERENCE)
        if _contains_raw_text(payload):
            raise _reject(REJECTION_RAW_TEXT)

        reference_id = payload.get("reference_id", payload.get("evidence_id"))
        if reference_id is None:
            reference_id = payload.get("id")
        if reference_id is None:
            raise _reject(REJECTION_INVALID_REFERENCE)

        synthetic = payload.get("synthetic", payload.get("is_synthetic"))
        if synthetic is None:
            synthetic = _is_synthetic_marker(None, reference_id)
        if synthetic is not True:
            raise _reject(REJECTION_NOT_SYNTHETIC)

        verified = payload.get("verified", payload.get("is_verified"))
        if verified is None:
            verification_state = payload.get(
                "verification_state", payload.get("verification_status")
            )
            verified = verification_state == "verified"
        if verified is not True:
            raise _reject(REJECTION_UNVERIFIED)

        review_state = payload.get("review_state")
        if review_state is None and payload.get("reviewed") is True:
            review_state = REVIEW_STATE_REVIEWED
        if review_state not in REVIEW_STATE_VALUES:
            raise _reject(REJECTION_INVALID_REVIEW_STATE)

        policy_fingerprint = payload.get(
            "policy_fingerprint", payload.get("policy_digest")
        )
        if not isinstance(
            policy_fingerprint, str
        ) or not _POLICY_FINGERPRINT_RE.fullmatch(policy_fingerprint.strip().lower()):
            raise _reject(REJECTION_INVALID_POLICY_FINGERPRINT)

        start, end, source_length = _extract_offset_values(payload)
        start, end = _validate_offsets(start, end, source_length)

        source_id = payload.get("source_id", payload.get("document_id"))
        return cls(
            reference_id=reference_id,
            source_id=source_id,
            start=start,
            end=end,
            review_state=review_state,
            policy_fingerprint=policy_fingerprint,
            synthetic=True,
            verified=True,
        )


@dataclass(frozen=True)
class EvidenceRejectionReport:
    """Counts-only result of filtering candidate evidence references."""

    input_count: int
    accepted_count: int
    rejected_count: int
    counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (self.input_count, self.accepted_count, self.rejected_count)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError("evidence rejection counts must be integers")
        if any(value < 0 for value in values):
            raise ValueError("evidence rejection counts must be non-negative")
        if self.accepted_count + self.rejected_count != self.input_count:
            raise ValueError("evidence rejection counts are inconsistent")

        normalized: dict[str, int] = {}
        for category in REJECTION_CATEGORIES:
            count = self.counts.get(category, 0)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("evidence rejection category counts are invalid")
            if count:
                normalized[category] = count
        if sum(normalized.values()) != self.rejected_count:
            raise ValueError("evidence rejection categories are inconsistent")
        unknown = set(self.counts) - set(REJECTION_CATEGORIES)
        if unknown:
            raise ValueError("evidence rejection category is not supported")
        object.__setattr__(self, "counts", MappingProxyType(normalized))

    @property
    def rejection_counts(self) -> Mapping[str, int]:
        """Return observed rejection counts in stable category order."""

        return self.counts

    @property
    def categories(self) -> tuple[str, ...]:
        """Return all supported categories in their stable contract order."""

        return REJECTION_CATEGORIES

    @property
    def rejected(self) -> int:
        """Return the number of rejected records."""

        return self.rejected_count

    def to_dict(self) -> dict[str, Any]:
        """Return a counts-only JSON-compatible report."""

        return {
            "input_count": self.input_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "rejection_counts": dict(self.counts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRejectionReport":
        """Rebuild a report from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise ValueError("evidence rejection report must be a mapping")
        counts = payload.get("rejection_counts", payload.get("counts", {}))
        if not isinstance(counts, Mapping):
            raise ValueError("evidence rejection categories must be a mapping")
        return cls(
            input_count=payload.get("input_count", 0),
            accepted_count=payload.get("accepted_count", 0),
            rejected_count=payload.get("rejected_count", 0),
            counts=dict(counts),
        )


@dataclass(frozen=True)
class EvidencePacket:
    """A deterministic packet containing only validated evidence references."""

    references: tuple[EvidenceReference, ...]
    policy_fingerprint: str
    rejection_report: EvidenceRejectionReport | None = None
    packet_id: str = "synthetic-evidence-packet"
    schema_version: int = EVIDENCE_PACKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_PACKET_SCHEMA_VERSION:
            raise ValueError("unsupported evidence packet schema version")
        policy_fingerprint = _validate_policy_fingerprint(self.policy_fingerprint)
        packet_id = _required_identifier(self.packet_id)
        references = tuple(self.references)
        seen: set[str] = set()
        for reference in references:
            if not isinstance(reference, EvidenceReference):
                raise _reject(REJECTION_INVALID_REFERENCE)
            if reference.reference_id in seen:
                raise _reject(REJECTION_DUPLICATE_REFERENCE)
            if reference.policy_fingerprint != policy_fingerprint:
                raise _reject(REJECTION_POLICY_MISMATCH)
            seen.add(reference.reference_id)
        references = tuple(
            sorted(
                references, key=lambda item: (item.start, item.end, item.reference_id)
            )
        )
        report = self.rejection_report
        if report is None:
            report = EvidenceRejectionReport(
                input_count=len(references),
                accepted_count=len(references),
                rejected_count=0,
            )
        if report.accepted_count != len(references):
            raise ValueError("evidence packet report does not match references")

        object.__setattr__(self, "policy_fingerprint", policy_fingerprint)
        object.__setattr__(self, "packet_id", packet_id)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "rejection_report", report)

    @property
    def evidence_references(self) -> tuple[EvidenceReference, ...]:
        """Return references under the descriptive compatibility name."""

        return self.references

    @property
    def rejection_counts(self) -> Mapping[str, int]:
        """Return the packet's counts-only rejection categories."""

        return self.rejection_report.counts

    @property
    def accepted_count(self) -> int:
        """Return the number of accepted references."""

        return len(self.references)

    @property
    def rejected_count(self) -> int:
        """Return the number of rejected candidates."""

        return self.rejection_report.rejected_count

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, deterministic packet representation."""

        return {
            "schema_version": self.schema_version,
            "kind": EVIDENCE_PACKET_KIND,
            "packet_id": self.packet_id,
            "policy_fingerprint": self.policy_fingerprint,
            "references": [reference.to_dict() for reference in self.references],
            "rejection_report": self.rejection_report.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the packet to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidencePacket":
        """Validate a serialized packet without retaining unknown fields."""

        if not isinstance(payload, Mapping):
            raise _reject(REJECTION_INVALID_REFERENCE)
        if _contains_raw_text(payload):
            raise _reject(REJECTION_RAW_TEXT)
        references_payload = payload.get("references", payload.get("evidence", ()))
        if not isinstance(references_payload, (list, tuple)):
            raise _reject(REJECTION_INVALID_REFERENCE)
        references = tuple(
            EvidenceReference.from_dict(reference) for reference in references_payload
        )
        report_payload = payload.get("rejection_report", payload.get("report"))
        report = (
            None
            if report_payload is None
            else EvidenceRejectionReport.from_dict(report_payload)
        )
        return cls(
            references=references,
            policy_fingerprint=payload.get("policy_fingerprint"),
            rejection_report=report,
            packet_id=payload.get("packet_id", "synthetic-evidence-packet"),
            schema_version=payload.get(
                "schema_version", EVIDENCE_PACKET_SCHEMA_VERSION
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> "EvidencePacket":
        """Deserialize and validate a JSON packet."""

        if not isinstance(text, str):
            raise _reject(REJECTION_INVALID_REFERENCE)
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            raise _reject(REJECTION_INVALID_REFERENCE) from None
        return cls.from_dict(payload)


def fingerprint_policy(policy: Any) -> str:
    """Return a local deterministic ``sha256:`` fingerprint for a policy.

    A policy object exposing a safe ``fingerprint`` property is respected after
    validation.  Otherwise its JSON-compatible value is hashed with the same
    canonical hashing helper used by OpenMed's audit surfaces.
    """

    supplied = getattr(policy, "fingerprint", None)
    if isinstance(supplied, str) and _POLICY_FINGERPRINT_RE.fullmatch(
        supplied.strip().lower()
    ):
        return supplied.strip().lower()
    if isinstance(policy, str) and _POLICY_FINGERPRINT_RE.fullmatch(
        policy.strip().lower()
    ):
        return policy.strip().lower()
    try:
        return stable_hash(policy)
    except (TypeError, ValueError):
        raise ValueError("policy cannot be fingerprinted as canonical JSON") from None


compute_policy_fingerprint = fingerprint_policy


def _inferred_policy_fingerprint(records: tuple[Any, ...]) -> str | None:
    candidates: set[str] = set()
    for record in records:
        if isinstance(record, EvidenceReference):
            candidates.add(record.policy_fingerprint)
            continue
        if isinstance(record, Mapping):
            value = record.get("policy_fingerprint", record.get("policy_digest"))
            if isinstance(value, str) and _POLICY_FINGERPRINT_RE.fullmatch(
                value.strip().lower()
            ):
                candidates.add(value.strip().lower())
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _coerce_records(records: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(records, EvidenceReference) or isinstance(records, Mapping):
        return (records,)
    if isinstance(records, (str, bytes)) or records is None:
        return (records,)
    try:
        return tuple(records)
    except TypeError:
        return (records,)


def build_evidence_packet(
    records: Iterable[EvidenceReference | Mapping[str, Any]] | EvidenceReference,
    *,
    policy_fingerprint: str | None = None,
    packet_id: str = "synthetic-evidence-packet",
) -> EvidencePacket:
    """Filter candidate records into a guarded evidence packet.

    Args:
        records: One or more :class:`EvidenceReference` instances or candidate
            mappings.  Candidate mappings must be synthetic, verified,
            reviewed, fingerprinted, and offset-bearing.
        policy_fingerprint: Expected policy digest.  When omitted, it is
            inferred only if every candidate that supplies a valid digest agrees
            on exactly one value.
        packet_id: Safe synthetic packet identifier.

    Returns:
        An immutable packet.  Invalid candidates are omitted and represented by
        counts only in ``packet.rejection_report``.

    Raises:
        EvidencePacketValidationError: If the packet-level policy digest is
            invalid or no digest can be inferred from non-empty input.
    """

    candidate_records = _coerce_records(records)
    expected = (
        _validate_policy_fingerprint(policy_fingerprint)
        if policy_fingerprint is not None
        else _inferred_policy_fingerprint(candidate_records)
    )
    if expected is None:
        if candidate_records:
            raise _reject(REJECTION_INVALID_POLICY_FINGERPRINT)
        raise _reject(REJECTION_INVALID_POLICY_FINGERPRINT)

    accepted: list[EvidenceReference] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}

    def reject(category: str) -> None:
        counts[category] = counts.get(category, 0) + 1

    for candidate in candidate_records:
        if isinstance(candidate, (str, bytes)):
            reject(REJECTION_RAW_TEXT)
            continue
        try:
            reference = (
                candidate
                if isinstance(candidate, EvidenceReference)
                else EvidenceReference.from_dict(candidate)
            )
        except EvidencePacketValidationError as error:
            reject(error.category)
            continue
        if reference.policy_fingerprint != expected:
            reject(REJECTION_POLICY_MISMATCH)
            continue
        if reference.reference_id in seen:
            reject(REJECTION_DUPLICATE_REFERENCE)
            continue
        seen.add(reference.reference_id)
        accepted.append(reference)

    accepted.sort(key=lambda item: (item.start, item.end, item.reference_id))
    report = EvidenceRejectionReport(
        input_count=len(candidate_records),
        accepted_count=len(accepted),
        rejected_count=len(candidate_records) - len(accepted),
        counts=counts,
    )
    return EvidencePacket(
        references=tuple(accepted),
        policy_fingerprint=expected,
        rejection_report=report,
        packet_id=packet_id,
    )


package_evidence = build_evidence_packet
create_evidence_packet = build_evidence_packet


def validate_evidence_packet(
    candidate: EvidencePacket | Mapping[str, Any],
) -> EvidencePacket:
    """Validate an existing packet object or serialized packet mapping."""

    if isinstance(candidate, EvidencePacket):
        # Reconstructing through the constructor re-applies the immutable
        # invariants even if a caller used object.__setattr__ to tamper with it.
        return EvidencePacket(
            references=candidate.references,
            policy_fingerprint=candidate.policy_fingerprint,
            rejection_report=candidate.rejection_report,
            packet_id=candidate.packet_id,
            schema_version=candidate.schema_version,
        )
    if isinstance(candidate, Mapping):
        return EvidencePacket.from_dict(candidate)
    raise _reject(REJECTION_INVALID_REFERENCE)
