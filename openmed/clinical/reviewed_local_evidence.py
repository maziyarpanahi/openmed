"""Value-free reviewed-local evidence; serialization never grants authority.

The injected local stores, not a marker or receipt supplied by an untrusted
caller, decide whether a source and review are current. No source is persisted.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Protocol

REVIEWED_LOCAL_EVIDENCE_VERSION = 1
REVIEWED_LOCAL_EVIDENCE_KIND = "reviewed_local_evidence"
REVIEWED_LOCAL_PROVENANCE = "reviewed_local"
REVIEWED_LOCAL_OFFSETS = "unicode_scalar_half_open"


class ReviewAdmissionRefusal(str, Enum):
    """Controlled refusal vocabulary shared with the brief boundary."""

    INVALID = "invalid_reviewed_evidence"
    MISSING = "review_receipt_missing"
    EXPIRED = "review_receipt_expired"
    MISMATCHED = "review_receipt_mismatched"
    REVOKED = "review_receipt_revoked"
    AUTHORITY_UNAVAILABLE = "review_authority_unavailable"
    SOURCE_UNAVAILABLE = "review_source_unavailable"
    SOURCE_CHANGED = "review_source_changed"
    POLICY_CHANGED = "review_policy_changed"


class ReviewAdmissionError(ValueError):
    """Admission failed with a fixed code and no caller-controlled values."""

    def __init__(self, reason: ReviewAdmissionRefusal):
        self.reason = reason
        super().__init__(reason.value)


class ReviewAuthorityStatus(str, Enum):
    """The trusted authority's current registry result, never caller approval."""

    CURRENT = "current"
    REVOKED = "revoked"
    MISMATCHED = "mismatched"


class ReviewAuthorityVerifier(Protocol):
    """Application-owned local authority; check access, binding and revocation."""

    def verify(
        self, receipt: LocalReviewReceipt, *, evidence_digest: str, now: int
    ) -> ReviewAuthorityStatus:
        """Resolve the receipt in trusted custody and verify its exact record."""
        ...


class CurrentLocalSource(Protocol):
    """Application-owned authorized source lookup; never return source text."""

    def current_digest(self, source_id: str) -> str | None:
        """Return the current source's UTF-8 SHA-256 digest, or None."""
        ...


def reviewed_source_digest(text: str) -> str:
    """Hash the exact de-identified offset coordinate source without storing it."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _invalid() -> ReviewAdmissionError:
    return ReviewAdmissionError(ReviewAdmissionRefusal.INVALID)


def _opaque(value: Any, prefix: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(prefix + r":[0-9a-f]{64}", value):
        raise _invalid()


def _digest(value: Any) -> None:
    _opaque(value, "sha256")


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _keys(value: Any, required: set[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != required:
        raise _invalid()


@dataclass(frozen=True)
class ReviewedLocalReference:
    """Opaque reference and bounded half-open Unicode-scalar source offsets."""

    reference_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        _opaque(self.reference_id, "ref")
        if not _integer(self.start) or not _integer(self.end) or self.end <= self.start:
            raise _invalid()


@dataclass(frozen=True)
class LocalReviewReceipt:
    """Public receipt locator and bindings, never credentials or a signature.

    Integer times are UTC Unix seconds. The authority must resolve this locator
    and compare the complete record with its independently held review record.
    """

    receipt_id: str
    authority_id: str
    evidence_digest: str
    issued_at: int
    expires_at: int

    def __post_init__(self) -> None:
        _opaque(self.receipt_id, "receipt")
        _opaque(self.authority_id, "authority")
        _digest(self.evidence_digest)
        if (
            not _integer(self.issued_at)
            or not _integer(self.expires_at)
            or self.expires_at <= self.issued_at
        ):
            raise _invalid()


@dataclass(frozen=True)
class ReviewedLocalEvidence:
    """Separate versioned contract; construction and decoding do not admit it."""

    source_id: str
    source_digest: str
    source_length: int
    policy_digest: str
    references: tuple[ReviewedLocalReference, ...]
    review_receipt: LocalReviewReceipt | None = None
    schema_version: int = REVIEWED_LOCAL_EVIDENCE_VERSION
    kind: str = REVIEWED_LOCAL_EVIDENCE_KIND
    provenance_class: str = REVIEWED_LOCAL_PROVENANCE
    offset_convention: str = REVIEWED_LOCAL_OFFSETS

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != REVIEWED_LOCAL_EVIDENCE_VERSION
            or self.kind != REVIEWED_LOCAL_EVIDENCE_KIND
            or self.provenance_class != REVIEWED_LOCAL_PROVENANCE
            or self.offset_convention != REVIEWED_LOCAL_OFFSETS
        ):
            raise _invalid()
        _opaque(self.source_id, "source")
        _digest(self.source_digest)
        _digest(self.policy_digest)
        if not _integer(self.source_length) or type(self.references) is not tuple:
            raise _invalid()
        if not 1 <= len(self.references) <= 64:
            raise _invalid()
        refs = []
        for ref in self.references:
            if type(ref) is not ReviewedLocalReference:
                raise _invalid()
            ref = replace(ref)
            if ref.end > self.source_length:
                raise _invalid()
            refs.append(ref)
        if len({r.reference_id for r in refs}) != len(refs):
            raise _invalid()
        object.__setattr__(self, "references", tuple(refs))
        if self.review_receipt is not None:
            if type(self.review_receipt) is not LocalReviewReceipt:
                raise _invalid()
            object.__setattr__(self, "review_receipt", replace(self.review_receipt))

    @property
    def policy_fingerprint(self) -> str:
        """Expose the admitted policy binding to existing downstream guards."""
        return self.policy_digest

    @property
    def evidence_digest(self) -> str:
        """Bind all version, provenance, source, offset and policy fields."""
        record = self.to_dict()
        del record["review_receipt"]
        return reviewed_source_digest(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict value-free wire record; no admission state is stored."""
        packet = replace(self)
        return asdict(packet) | {"references": [asdict(r) for r in packet.references]}

    def to_json(self) -> str:
        """Return canonical local JSON containing only controlled metadata."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReviewedLocalEvidence:
        """Decode only this contract; reject unknown fields and text payloads."""
        _keys(payload, set(cls.__dataclass_fields__))
        rows = payload["references"]
        if type(rows) is not list or not 1 <= len(rows) <= 64:
            raise _invalid()
        refs = []
        for row in rows:
            _keys(row, set(ReviewedLocalReference.__dataclass_fields__))
            refs.append(ReviewedLocalReference(**row))
        receipt = payload["review_receipt"]
        if receipt is not None:
            _keys(receipt, set(LocalReviewReceipt.__dataclass_fields__))
            receipt = LocalReviewReceipt(**receipt)
        return cls(
            **(dict(payload) | {"references": tuple(refs), "review_receipt": receipt})
        )

    @classmethod
    def from_json(cls, value: str) -> ReviewedLocalEvidence:
        """Decode bounded JSON with value-free errors and no exception chain."""

        def unique(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise _invalid()
                result[key] = item
            return result

        try:
            if type(value) is str and len(value.encode("utf-8")) <= 65536:
                return cls.from_dict(json.loads(value, object_pairs_hook=unique))
        except (ValueError, TypeError, RecursionError):
            pass
        raise _invalid()


def admit_reviewed_local_evidence(
    packet: ReviewedLocalEvidence,
    *,
    source_digest: str,
    policy_digest: str,
    source: CurrentLocalSource,
    authority: ReviewAuthorityVerifier,
    clock: Callable[[], float] = time.time,
) -> ReviewedLocalEvidence:
    """Revalidate current local custody and review immediately before generation.

    Args:
        packet: Untrusted value-free reviewed-local record.
        source_digest: Digest of the exact de-identified generation source.
        policy_digest: Recomputed binding of source, reviewed axes and profile.
        source: Authorized local source custody lookup.
        authority: Trusted local verifier of exact receipt, access and revocation.
        clock: Trusted UTC Unix clock, injectable for offline tests.

    Returns:
        A defensively revalidated packet; never a reusable authorization token.

    Raises:
        ReviewAdmissionError: A controlled refusal with no private values.
    """
    if type(packet) is not ReviewedLocalEvidence:
        raise _invalid()
    packet = replace(packet)
    if packet.source_digest != source_digest:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.SOURCE_CHANGED)
    if packet.policy_digest != policy_digest:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.POLICY_CHANGED)
    receipt = packet.review_receipt
    if receipt is None:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.MISSING)
    if receipt.evidence_digest != packet.evidence_digest:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.MISMATCHED)
    now = None
    try:
        instant = clock()
        if type(instant) in (int, float) and instant >= 0:
            now = int(instant)
    except Exception:
        pass
    if now is None:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.AUTHORITY_UNAVAILABLE)
    if now < receipt.issued_at:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.MISMATCHED)
    if now >= receipt.expires_at:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.EXPIRED)
    current = None
    try:
        current = source.current_digest(packet.source_id)
    except Exception:
        pass
    if current is None:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.SOURCE_UNAVAILABLE)
    if current != source_digest:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.SOURCE_CHANGED)
    status = None
    try:
        status = authority.verify(
            receipt, evidence_digest=packet.evidence_digest, now=now
        )
    except Exception:
        pass
    if type(status) is not ReviewAuthorityStatus:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.AUTHORITY_UNAVAILABLE)
    if status is ReviewAuthorityStatus.REVOKED:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.REVOKED)
    if status is not ReviewAuthorityStatus.CURRENT:
        raise ReviewAdmissionError(ReviewAdmissionRefusal.MISMATCHED)
    return packet
