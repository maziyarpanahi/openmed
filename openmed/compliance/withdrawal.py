"""Coordinate confirmed consent withdrawal across explicitly registered local stores."""

from __future__ import annotations

import hmac
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Protocol

from openmed.core.audit import stable_hash

_DIGEST = re.compile(r"(?:hmac-)?sha256:[0-9a-f]{64}")


class ParticipantCode(str, Enum):
    """Local store families eligible for coordinated withdrawal."""

    CONSENT_CACHE = "consent_cache"
    RESULT_CACHE = "result_cache"
    SURROGATE_VAULT = "surrogate_vault"
    JOURNEY_STORE = "journey_store"
    SERVICE_JOBS = "service_jobs"
    AMBIENT_SESSIONS = "ambient_sessions"
    VOICE_EMBEDDINGS = "voice_embeddings"


@dataclass(frozen=True)
class ParticipantPlan:
    """Bind a count to a digest of the selected local records and their revisions."""

    count: int
    selection_digest: str
    unattributable_count: int = 0

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or n < 0 for n in (self.count, self.unattributable_count)
        ):
            raise ValueError("invalid withdrawal count")
        if not _DIGEST.fullmatch(self.selection_digest):
            raise ValueError("invalid selection digest")


class WithdrawalParticipant(Protocol):
    """Implement selection, conditional erasure and independent re-selection locally."""

    def plan(self, subject_ref_digest: str) -> ParticipantPlan:
        """Select matching records without mutation."""
        ...

    def apply(self, subject_ref_digest: str, plan: ParticipantPlan) -> int:
        """Check the selection under the store lock, erase matches and return a count."""
        ...

    def verify(self, subject_ref_digest: str) -> bool:
        """Re-select and return True only when erasure can be verified."""
        ...


@dataclass(frozen=True)
class WithdrawalPlan:
    """Read-only snapshot of the registered stores for one subject reference."""

    subject_ref_digest: str
    participants: tuple[tuple[ParticipantCode, ParticipantPlan], ...]

    @property
    def confirmation_token(self) -> str:
        """Return an explicit confirmation bound to the subject and every selection."""
        return "confirm:" + stable_hash(asdict(self))


@dataclass(frozen=True)
class ParticipantResult:
    """Counts and verification outcome for one store."""

    code: ParticipantCode
    planned: int
    applied: int
    unattributable: int
    verified: bool


@dataclass(frozen=True)
class WithdrawalReceipt:
    """Value-free result of one confirmed withdrawal."""

    subject_ref_digest: str
    participants: tuple[ParticipantResult, ...]

    @property
    def state(self) -> str:
        """Return success only if all registered stores confirmed their work."""
        return (
            "success"
            if all(p.verified for p in self.participants) and self.participants
            else "partial"
        )

    def to_dict(self) -> dict:
        """Return the aggregate receipt as a JSON-compatible mapping."""
        return {
            "subject_ref_digest": self.subject_ref_digest,
            "participants": [asdict(p) for p in self.participants],
            "state": self.state,
        }


class WithdrawalCoordinator:
    """Plan by default and apply only the exact selection explicitly confirmed."""

    def __init__(self, participants: Mapping[ParticipantCode, WithdrawalParticipant]):
        if not participants or any(
            not isinstance(code, ParticipantCode) for code in participants
        ):
            raise ValueError("register at least one supported local participant")
        self._participants = dict(sorted(participants.items()))

    def plan(self, subject_ref_digest: str) -> WithdrawalPlan:
        """Return each participant's current selection, without applying it."""
        if not isinstance(subject_ref_digest, str) or not _DIGEST.fullmatch(
            subject_ref_digest
        ):
            raise ValueError("invalid subject reference digest")
        try:
            plans = tuple(
                (code, participant.plan(subject_ref_digest))
                for code, participant in self._participants.items()
            )
        except Exception:
            raise ValueError("withdrawal planning failed") from None
        if any(type(plan) is not ParticipantPlan for _, plan in plans):
            raise ValueError("invalid participant plan")
        return WithdrawalPlan(subject_ref_digest, plans)

    def withdraw(
        self, subject_ref_digest: str, *, confirmation: str | None = None
    ) -> WithdrawalPlan | WithdrawalReceipt:
        """Plan, or apply and verify all stores if the fresh plan matches confirmation."""
        plan = self.plan(subject_ref_digest)
        if confirmation is None:
            return plan
        if (
            not isinstance(confirmation, str)
            or not confirmation.isascii()
            or not hmac.compare_digest(confirmation, plan.confirmation_token)
        ):
            raise ValueError("withdrawal confirmation does not match the current plan")

        results = []
        for code, selection in plan.participants:
            participant = self._participants[code]
            applied, verified = 0, False
            try:
                count = participant.apply(subject_ref_digest, selection)
                if type(count) is int and 0 <= count <= selection.count:
                    applied = count
                    verified = (
                        participant.verify(subject_ref_digest) is True
                        and applied == selection.count
                    )
            except Exception:
                # A store may fail after mutation. Only independently verified
                # work contributes to success; continue processing other stores.
                verified = False
            results.append(
                ParticipantResult(
                    code,
                    selection.count,
                    applied,
                    selection.unattributable_count,
                    verified,
                )
            )
        return WithdrawalReceipt(subject_ref_digest, tuple(results))


class AmbientWithdrawalParticipant:
    """Registry of live session owners for one coordinated withdrawal family.

    Register an instance for AMBIENT_SESSIONS and another for VOICE_EMBEDDINGS
    as applicable. Bind hosts' subject reference digests before accepting data.
    """

    def __init__(self, code: ParticipantCode):
        from threading import RLock

        if not isinstance(code, ParticipantCode) or code not in (
            ParticipantCode.AMBIENT_SESSIONS,
            ParticipantCode.VOICE_EMBEDDINGS,
        ):
            raise ValueError("invalid_ambient_participant")
        self.code = code
        self._lock = RLock()
        self._owners = {}
        self._withdrawn = set()
        self._receipts = {}

    def register(self, subject_ref_digest: str, owner) -> None:
        """Associate a local session owner with an opaque subject digest."""
        from uuid import uuid4

        from openmed.multimodal.streaming.session import AmbientSession
        from openmed.multimodal.voice_embeddings import VoiceEmbeddingSession

        kind = (
            AmbientSession
            if self.code == ParticipantCode.AMBIENT_SESSIONS
            else VoiceEmbeddingSession
        )
        if (
            type(subject_ref_digest) is not str
            or not _DIGEST.fullmatch(subject_ref_digest)
            or type(owner) is not kind
        ):
            raise ValueError("invalid_ambient_registration")
        with self._lock:
            if subject_ref_digest in self._withdrawn:
                raise ValueError("subject_withdrawn")
            owners = self._owners.setdefault(subject_ref_digest, {})
            if owner not in owners:
                owners[owner] = uuid4().hex

    def plan(self, subject_ref_digest: str) -> ParticipantPlan:
        """Bind confirmation to the full set of registered live owners."""
        with self._lock:
            ids = sorted(self._owners.get(subject_ref_digest, {}).values())
            return ParticipantPlan(
                len(ids), stable_hash([self.code.value, subject_ref_digest, ids])
            )

    def apply(self, subject_ref_digest: str, plan: ParticipantPlan) -> int:
        """Close the registered owners and retain only controlled destruction counts."""
        with self._lock:
            if plan != self.plan(subject_ref_digest):
                raise ValueError("ambient_selection_changed")
            self._withdrawn.add(subject_ref_digest)
            owners = self._owners.get(subject_ref_digest, {})
            receipts = self._receipts.setdefault(subject_ref_digest, {})
            applied = 0
            for owner, ref in tuple(owners.items()):
                try:
                    if self.code == ParticipantCode.AMBIENT_SESSIONS:
                        owner.withdraw()
                        snapshot = owner.snapshot()
                        if snapshot["unaccounted_external_copies"]:
                            owner.delete()
                            snapshot = owner.snapshot()
                        receipt = {
                            "state": snapshot["state"],
                            "verified": snapshot["deletion_verified"],
                            "external_copies": snapshot["unaccounted_external_copies"],
                        }
                    else:
                        destroyed = owner.destroy("withdrawal")
                        receipt = {
                            "state": destroyed.reason_code,
                            "destroyed_count": destroyed.destroyed_count,
                            "verified": owner.diagnostics()["handle_count"] == 0,
                        }
                except Exception:
                    # Keep this owner registered for retry and process every other owner.
                    receipts[ref] = {"state": "deletion_incomplete", "verified": False}
                    continue
                receipts[ref] = receipt
                del owners[owner]
                applied += 1
            if not owners:
                self._owners.pop(subject_ref_digest, None)
            return applied

    def verify(self, subject_ref_digest: str) -> bool:
        """Require independently observed closure and zero outstanding owners."""
        with self._lock:
            return (
                subject_ref_digest in self._withdrawn
                and not self._owners.get(subject_ref_digest)
                and all(
                    r["verified"]
                    for r in self._receipts.get(subject_ref_digest, {}).values()
                )
            )

    def receipts(self, subject_ref_digest: str) -> tuple[dict, ...]:
        """Return value-free erasure outcomes without registration or handle IDs."""
        with self._lock:
            return tuple(
                dict(r) for r in self._receipts.get(subject_ref_digest, {}).values()
            )
