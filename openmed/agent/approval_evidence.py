"""Local observation of approval receipts without issuing clinical authority.

Serialized receipts and verification reports are metadata, not credentials.
Only an application-owned custody lookup can recognize a receipt. The extra
process-local replay check does not replace the original token nonce store.
This module never issues tokens, creates receipts, dispatches, or contacts EHRs.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .approvals.tokens import (
    ApprovalNonceStore,
    ApprovalReceipt,
    ApprovalTokenValidationError,
    InMemoryApprovalNonceStore,
    _validate_digest,
    _validate_reviewer_role,
)

APPROVAL_EVIDENCE_SCHEMA_VERSION: Final = "openmed.agent.approval_evidence.v1"
MAX_APPROVAL_EVIDENCE_BYTES: Final = 65_536
_FIELDS = frozenset(
    {"schema_version", "status", "reason_code", "action_digest", "receipt_digest"}
)


class ReceiptAuthority(str, Enum):
    """A trusted local custody lookup's closed response vocabulary."""

    RECOGNIZED = "recognized"
    UNRECOGNIZED = "unrecognized"
    UNSUPPORTED = "unsupported"


class ApprovalEvidenceReason(str, Enum):
    """Closed observations; none grants authority to perform an action."""

    VERIFIED = "verified"
    UNSUPPORTED_AUTHORITY = "unsupported_authority"
    UNRECOGNIZED_RECEIPT = "unrecognized_receipt"
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    EXPIRED = "expired"
    FUTURE_RECEIPT = "future_receipt"
    REPLAYED = "replayed"
    ACTION_MISMATCH = "action_mismatch"
    REVIEWER_ROLE_MISMATCH = "reviewer_role_mismatch"
    NONCE_STORE_UNAVAILABLE = "nonce_store_unavailable"
    CLOCK_UNAVAILABLE = "clock_unavailable"


class ApprovalEvidenceError(ValueError):
    """Value-free error with one controlled code.

    Args:
        code: Fixed public diagnostic code, never submitted text.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class ApprovalEvidenceResult:
    """A content-free receipt observation, never a transferable approval.

    Args:
        reason_code: Closed observation or refusal.
        action_digest: Action recorded in the original receipt.
        receipt_digest: SHA-256 of the receipt's canonical JSON.
        schema_version: Exact report version.
    """

    reason_code: ApprovalEvidenceReason
    action_digest: str
    receipt_digest: str
    schema_version: str = APPROVAL_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPROVAL_EVIDENCE_SCHEMA_VERSION:
            raise ApprovalEvidenceError("unsupported_version")
        if not isinstance(self.reason_code, ApprovalEvidenceReason):
            raise ApprovalEvidenceError("unknown_reason")
        try:
            _validate_digest(self.action_digest, "action_digest")
            _validate_digest(self.receipt_digest, "receipt_digest")
        except ApprovalTokenValidationError:
            raise ApprovalEvidenceError("invalid_digest") from None

    @property
    def authorizes_clinical_action(self) -> bool:
        """Always return False; reports cannot be presented for dispatch."""
        return False

    def to_dict(self) -> dict[str, str]:
        """Return exact versioned observation fields."""
        return {
            "schema_version": self.schema_version,
            "status": (
                "verified"
                if self.reason_code is ApprovalEvidenceReason.VERIFIED
                else "refused"
            ),
            "reason_code": self.reason_code.value,
            "action_digest": self.action_digest,
            "receipt_digest": self.receipt_digest,
        }

    def to_json(self) -> str:
        """Return compact sorted ASCII JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> ApprovalEvidenceResult:
        """Parse a bounded report without trusting its recorded result.

        Args:
            payload: Untrusted JSON observation.

        Returns:
            A report whose authority property remains False.

        Raises:
            ApprovalEvidenceError: On malformed or unsupported metadata.
        """
        if not isinstance(payload, (str, bytes, bytearray)):
            raise ApprovalEvidenceError("invalid_json")
        try:
            raw = payload.encode("utf-8") if isinstance(payload, str) else payload
            if len(raw) > MAX_APPROVAL_EVIDENCE_BYTES:
                raise ApprovalEvidenceError("json_too_large")
            values = json.loads(raw, object_pairs_hook=_object)
        except ApprovalEvidenceError:
            raise
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise ApprovalEvidenceError("invalid_json") from None
        if not isinstance(values, Mapping) or set(values) != _FIELDS:
            raise ApprovalEvidenceError("invalid_fields")
        try:
            reason = ApprovalEvidenceReason(values["reason_code"])
        except (ValueError, TypeError):
            raise ApprovalEvidenceError("unknown_reason") from None
        result = cls(
            reason_code=reason,
            action_digest=values["action_digest"],
            receipt_digest=values["receipt_digest"],
            schema_version=values["schema_version"],
        )
        if values["status"] != result.to_dict()["status"]:
            raise ApprovalEvidenceError("status_mismatch")
        return result

    def __repr__(self) -> str:
        """Render only the controlled observation code."""
        return f"ApprovalEvidenceResult(reason_code={self.reason_code.value})"


class LocalApprovalEvidenceVerifier:
    """Observe application-custodied receipts with an atomic replay check.

    Args:
        authority: Local lookup of a canonical receipt digest. Omission refuses.
        replay_store: Atomic presentation store; default is process-local only.
        clock: Local whole-second Unix clock, injectable for offline checks.

    The application must independently authenticate reviewers, consume signed
    approvals in durable custody and enforce its dispatch policy. A successful
    observation here cannot replace any of those steps.
    """

    def __init__(
        self,
        authority: Callable[[str], ReceiptAuthority] | None = None,
        *,
        replay_store: ApprovalNonceStore | None = None,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self._authority = authority
        self._store = (
            InMemoryApprovalNonceStore() if replay_store is None else replay_store
        )
        self._clock = clock

    def verify(
        self,
        receipt: ApprovalReceipt,
        *,
        action_digest: str,
        reviewer_role: str,
    ) -> ApprovalEvidenceResult:
        """Observe one receipt, burning recognized mismatched presentations.

        Args:
            receipt: Previously consumed receipt from the existing contract.
            action_digest: Exact action currently shown to the caller.
            reviewer_role: Expected policy role, never an identity.

        Returns:
            A typed observation that cannot authorize clinical execution.

        Raises:
            ApprovalEvidenceError: If caller-supplied expected metadata is invalid.
        """
        if type(receipt) is not ApprovalReceipt:
            raise ApprovalEvidenceError("invalid_receipt")
        try:
            _validate_digest(action_digest, "action_digest")
            _validate_reviewer_role(reviewer_role)
        except ApprovalTokenValidationError:
            raise ApprovalEvidenceError("invalid_expected_metadata") from None
        digest = "sha256:" + hashlib.sha256(receipt.to_json().encode()).hexdigest()

        def result(reason: ApprovalEvidenceReason) -> ApprovalEvidenceResult:
            return ApprovalEvidenceResult(reason, receipt.action_digest, digest)

        now = self._now()
        if now is None:
            return result(ApprovalEvidenceReason.CLOCK_UNAVAILABLE)
        if now >= receipt.expires_at:
            return result(ApprovalEvidenceReason.EXPIRED)
        if now < receipt.consumed_at:
            return result(ApprovalEvidenceReason.FUTURE_RECEIPT)
        if self._authority is None:
            return result(ApprovalEvidenceReason.UNSUPPORTED_AUTHORITY)
        try:
            authority = self._authority(digest)
        except Exception:
            return result(ApprovalEvidenceReason.AUTHORITY_UNAVAILABLE)
        if authority is ReceiptAuthority.UNSUPPORTED:
            return result(ApprovalEvidenceReason.UNSUPPORTED_AUTHORITY)
        if authority is ReceiptAuthority.UNRECOGNIZED:
            return result(ApprovalEvidenceReason.UNRECOGNIZED_RECEIPT)
        if authority is not ReceiptAuthority.RECOGNIZED:
            return result(ApprovalEvidenceReason.AUTHORITY_UNAVAILABLE)
        # Recheck time after the lookup; a slow authority cannot extend expiry.
        now = self._now()
        if now is None:
            return result(ApprovalEvidenceReason.CLOCK_UNAVAILABLE)
        if now >= receipt.expires_at:
            return result(ApprovalEvidenceReason.EXPIRED)
        if now < receipt.consumed_at:
            return result(ApprovalEvidenceReason.FUTURE_RECEIPT)
        try:
            claimed = self._store.claim(
                receipt.token_digest, expires_at=receipt.expires_at, now=now
            )
        except Exception:
            return result(ApprovalEvidenceReason.NONCE_STORE_UNAVAILABLE)
        if type(claimed) is not bool:
            return result(ApprovalEvidenceReason.NONCE_STORE_UNAVAILABLE)
        if not claimed:
            return result(ApprovalEvidenceReason.REPLAYED)
        if receipt.action_digest != action_digest:
            return result(ApprovalEvidenceReason.ACTION_MISMATCH)
        if receipt.reviewer_role != reviewer_role:
            return result(ApprovalEvidenceReason.REVIEWER_ROLE_MISMATCH)
        return result(ApprovalEvidenceReason.VERIFIED)

    def _now(self) -> int | None:
        try:
            now = self._clock()
        except Exception:
            return None
        return now if type(now) is int and 0 <= now <= (1 << 63) - 1 else None

    def __repr__(self) -> str:
        """Omit custody, clock and replay-store state from diagnostics."""
        return "LocalApprovalEvidenceVerifier(<local>)"


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ApprovalEvidenceError("duplicate_field")
        result[key] = value
    return result


__all__ = [
    "APPROVAL_EVIDENCE_SCHEMA_VERSION",
    "MAX_APPROVAL_EVIDENCE_BYTES",
    "ApprovalEvidenceError",
    "ApprovalEvidenceReason",
    "ApprovalEvidenceResult",
    "LocalApprovalEvidenceVerifier",
    "ReceiptAuthority",
]
