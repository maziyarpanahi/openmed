"""Continuation authorization for queued service work and long-lived streams.

Admission proves only that a caller was authorized when work was accepted. A
queued job or a long-lived stream can outlive the consent, scope or policy
revision it was admitted under. This module binds admitted work to an
authorization epoch and revalidates that epoch before each protected output,
resumed execution, effect dispatch and declared checkpoint.

The guard owns continuation authorization only. Identity policy, credential
verification and durable queueing stay with their existing owners. Status
lookups reuse the injected :class:`~openmed.agent.permissions.revocation.
AuthorityRuntime` (provider, monotonic generation store and trusted clock),
effects reuse :class:`~openmed.agent.workflows.recovery.EffectRecord`, and
cooperative stops reuse :class:`~openmed.agent.workflows.scheduler.
CancellationSignal`. Persisted continuation metadata never carries a cached
authorization verdict, so a restarted worker always re-asks the provider.

Outcomes, records and errors contain only controlled codes, counts, declared
checkpoint names, opaque digests and pinned generations; never source text,
outputs, credentials, subjects or provider exception text.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, Protocol

from openmed.agent.permissions.revocation import (
    AuthorityBinding,
    AuthorityBoundary,
    AuthorityContract,
    AuthorityDeniedError,
    AuthorityKind,
    AuthorityRuntime,
)
from openmed.agent.workflows.recovery import EffectRecord, EffectState
from openmed.agent.workflows.scheduler import CancellationSignal

if TYPE_CHECKING:
    from .auth import AuthPrincipal

CONTINUATION_RECORD_VERSION: Final[str] = "openmed.service.continuation_record.v1"
CONTINUATION_OUTCOME_VERSION: Final[str] = "openmed.service.continuation_outcome.v1"
MAX_DECLARED_CHECKPOINTS: Final[int] = 64
MAX_CONTINUATION_EFFECTS: Final[int] = 128

_EPOCH_VERSION = "openmed.service.continuation_epoch.v1"
_WORK_VERSION = "openmed.service.continuation_work.v1"
_EFFECTS_VERSION = "openmed.service.continuation_effects.v1"
_PRINCIPAL_VERSION = "openmed.service.principal_authority.v1"
_CONSENT_VERSION = "openmed.service.consent_authority.v1"
_MAX_WORK_ID_LENGTH = 256
_MAX_COUNT = 2**63 - 1
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ADMISSION_RE = re.compile(r"adm_[0-9a-f]{32}\Z")
_CHECKPOINT_RE = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")
_POLICY_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,127}\Z")
_END_OF_SOURCE = object()
_RECORD_FIELDS = frozenset(
    {"schema_version", "admission_id", "work_digest", "checkpoints", "authority"}
)
_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "invalid_continuation_authority",
        "invalid_continuation_boundary",
        "invalid_continuation_cancellation",
        "invalid_continuation_checkpoint",
        "invalid_continuation_commit_evidence",
        "invalid_continuation_count",
        "invalid_continuation_effect",
        "invalid_continuation_guard",
        "invalid_continuation_outcome",
        "invalid_continuation_record",
        "invalid_continuation_stream",
        "invalid_continuation_work",
        "invalid_continuation_worker",
    }
)


class ContinuationError(ValueError):
    """Fixed-code validation error that never echoes a rejected value.

    Args:
        code: One of the module's closed ``invalid_continuation_*`` codes.
    """

    def __init__(self, code: str) -> None:
        self.code = code if code in _ERROR_CODES else "invalid_continuation_outcome"
        super().__init__(self.code)


class ContinuationBoundary(str, Enum):
    """Points where continued work must hold current authority.

    Values:
        ADMISSION: Work is accepted and its authorization epoch is pinned.
        EXECUTION_START: A queued item is picked up or a stream starts.
        RESUME: A restarted or paused worker resumes persisted work.
        CHECKPOINT: A declared long-lived checkpoint inside the worker.
        EFFECT_DISPATCH: Immediately before one external effect.
        PROTECTED_OUTPUT: Immediately before protected output is released.
    """

    ADMISSION = "admission"
    EXECUTION_START = "execution_start"
    RESUME = "resume"
    CHECKPOINT = "checkpoint"
    EFFECT_DISPATCH = "effect_dispatch"
    PROTECTED_OUTPUT = "protected_output"


_AUTHORITY_BOUNDARIES: Final[dict[ContinuationBoundary, AuthorityBoundary]] = {
    ContinuationBoundary.ADMISSION: AuthorityBoundary.PREVIEW,
    ContinuationBoundary.EXECUTION_START: AuthorityBoundary.RESUME,
    ContinuationBoundary.RESUME: AuthorityBoundary.RESUME,
    ContinuationBoundary.CHECKPOINT: AuthorityBoundary.RESUME,
    ContinuationBoundary.EFFECT_DISPATCH: AuthorityBoundary.EFFECT_DISPATCH,
    ContinuationBoundary.PROTECTED_OUTPUT: AuthorityBoundary.SENSITIVE_READ,
}


class ContinuationDisposition(str, Enum):
    """What happens to the work after one boundary decision.

    Values:
        CONTINUE: Authority is current; the work may cross this boundary.
        COMPLETED: The work finished and its outputs were released.
        STOPPED: Authority ended or the work was cancelled; it never resumes
            under this epoch and needs a fresh admission to run again.
        QUARANTINED: Integrity or availability failure; the epoch is closed
            and retained for operator reconciliation.
    """

    CONTINUE = "continue"
    COMPLETED = "completed"
    STOPPED = "stopped"
    QUARANTINED = "quarantined"


class ContinuationReason(str, Enum):
    """Closed reason codes for continuation decisions.

    ``authority_*`` values mirror
    :class:`~openmed.agent.permissions.revocation.AuthorityReason` denials.
    """

    ACTIVE = "continuation_active"
    COMPLETED = "continuation_completed"
    CANCELLED = "continuation_cancelled"
    TERMINAL = "continuation_terminal"
    WORK_MISMATCH = "continuation_work_mismatch"
    UNDECLARED_CHECKPOINT = "continuation_checkpoint_undeclared"
    WORKER_FAILED = "continuation_worker_failed"
    SOURCE_FAILED = "continuation_source_failed"
    LEDGER_UNAVAILABLE = "continuation_ledger_unavailable"
    AUTHORITY_REVOKED = "authority_revoked"
    AUTHORITY_EXPIRED = "authority_expired"
    AUTHORITY_GENERATION_CHANGED = "authority_generation_changed"
    AUTHORITY_UNAVAILABLE = "authority_status_unavailable"
    AUTHORITY_STALE = "authority_status_stale"
    AUTHORITY_ROLLBACK = "authority_status_rollback"
    AUTHORITY_STORE_UNAVAILABLE = "authority_generation_store_unavailable"
    AUTHORITY_CONTRACT_MISMATCH = "authority_contract_mismatch"
    AUTHORITY_APPROVAL_MISMATCH = "authority_approval_mismatch"


_STOPPED_REASONS: Final[frozenset[ContinuationReason]] = frozenset(
    {
        ContinuationReason.CANCELLED,
        ContinuationReason.TERMINAL,
        ContinuationReason.AUTHORITY_REVOKED,
        ContinuationReason.AUTHORITY_EXPIRED,
    }
)


def _disposition_for(reason: ContinuationReason) -> ContinuationDisposition:
    if reason is ContinuationReason.ACTIVE:
        return ContinuationDisposition.CONTINUE
    if reason is ContinuationReason.COMPLETED:
        return ContinuationDisposition.COMPLETED
    if reason in _STOPPED_REASONS:
        return ContinuationDisposition.STOPPED
    return ContinuationDisposition.QUARANTINED


class ReconciliationState(str, Enum):
    """Honest effect knowledge attached to every outcome.

    Cancellation and revocation never compensate effects. Committed effects
    stay committed and recorded; pending effects whose dispatch did not return
    commit evidence must be reconciled against their sink by idempotency key.

    Values:
        NO_EFFECTS: No effect was dispatched under this epoch.
        COMMITTED_EFFECTS_RETAINED: Every dispatched effect is committed and
            remains in effect; nothing was undone.
        PENDING_EFFECTS_UNCONFIRMED: At least one dispatched effect has no
            commit evidence and may or may not have taken place.
    """

    NO_EFFECTS = "no_effects"
    COMMITTED_EFFECTS_RETAINED = "committed_effects_retained"
    PENDING_EFFECTS_UNCONFIRMED = "pending_effects_unconfirmed"


def _hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _is_digest(value: Any) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def _require_count(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ContinuationError("invalid_continuation_count")
    return value


def work_digest(work_id: str) -> str:
    """Return the opaque, domain-separated digest of one work identifier.

    Args:
        work_id: Caller-owned job or stream identifier. It is hashed and never
            retained.

    Returns:
        A ``sha256:`` digest used to bind continuation metadata to that work.

    Raises:
        ContinuationError: If ``work_id`` is not a non-empty bounded string.
    """

    if (
        type(work_id) is not str
        or not work_id
        or len(work_id) > _MAX_WORK_ID_LENGTH
        or not work_id.isprintable()
    ):
        raise ContinuationError("invalid_continuation_work")
    encoded = (_WORK_VERSION + "\0" + work_id).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def principal_authority(
    principal: "AuthPrincipal", *, policy_revision: str, expires_at: int
) -> AuthorityContract:
    """Commit an admitted principal's scope and policy revision to a contract.

    The digest covers the subject, credential type, key id, the sorted granted
    scopes and the policy revision in force at admission. Narrowing scopes or
    changing the revision yields a different contract, so the authority
    provider must register (and later revoke) each one explicitly. Hashing a
    principal does not authenticate it.

    Args:
        principal: Authenticated :class:`~openmed.service.auth.AuthPrincipal`.
        policy_revision: Opaque policy revision label active at admission.
        expires_at: Exclusive Unix expiry, normally the credential expiry.

    Returns:
        A ``grant`` :class:`AuthorityContract` for the continuation guard.

    Raises:
        ContinuationError: If any field is malformed.
    """

    try:
        subject = principal.subject
        credential_type = principal.credential_type
        scopes = principal.scopes
        key_id = principal.key_id
    except AttributeError:
        raise ContinuationError("invalid_continuation_authority") from None
    if (
        type(subject) is not str
        or not subject
        or type(credential_type) is not str
        or not credential_type
        or not isinstance(scopes, (tuple, list))
        or any(type(scope) is not str or not scope for scope in scopes)
        or (key_id is not None and type(key_id) is not str)
        or type(policy_revision) is not str
        or _POLICY_REVISION_RE.fullmatch(policy_revision) is None
    ):
        raise ContinuationError("invalid_continuation_authority")
    digest = _hash(
        {
            "schema_version": _PRINCIPAL_VERSION,
            "subject": subject,
            "credential_type": credential_type,
            "key_id": key_id,
            "scopes": sorted(set(scopes)),
            "policy_revision": policy_revision,
        }
    )
    return _contract(AuthorityKind.GRANT, digest, expires_at)


def consent_authority(consent_digest: str, *, expires_at: int) -> AuthorityContract:
    """Reference one consent or purpose decision held by the application.

    Args:
        consent_digest: ``sha256:`` digest of the consent artifact. The artifact
            itself stays with its owner.
        expires_at: Exclusive Unix expiry of the consent.

    Returns:
        A ``purpose_ticket`` :class:`AuthorityContract` for the guard.

    Raises:
        ContinuationError: If the digest or expiry is malformed.
    """

    if not _is_digest(consent_digest):
        raise ContinuationError("invalid_continuation_authority")
    digest = _hash({"schema_version": _CONSENT_VERSION, "consent": consent_digest})
    return _contract(AuthorityKind.PURPOSE_TICKET, digest, expires_at)


def _contract(kind: AuthorityKind, digest: str, expires_at: int) -> AuthorityContract:
    try:
        return AuthorityContract(kind, digest, expires_at)
    except ValueError:
        raise ContinuationError("invalid_continuation_authority") from None


def _validate_checkpoints(checkpoints: Any) -> tuple[str, ...]:
    if not isinstance(checkpoints, (tuple, list)):
        raise ContinuationError("invalid_continuation_checkpoint")
    names = tuple(checkpoints)
    if (
        len(names) > MAX_DECLARED_CHECKPOINTS
        or any(
            type(name) is not str or _CHECKPOINT_RE.fullmatch(name) is None
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ContinuationError("invalid_continuation_checkpoint")
    return names


@dataclass(frozen=True, slots=True, repr=False)
class ContinuationRecord:
    """Persistable authorization epoch for one admitted unit of work.

    The record holds the pinned authority generations captured at admission,
    never a verdict. Restoring it grants nothing: every boundary performs fresh
    provider lookups, and a restored binding cannot be re-pinned to a newer
    generation. Store it beside the job; store the ledger and generation high
    water independently of it.

    Attributes:
        admission_id: Random ``adm_`` nonce distinguishing re-admissions.
        work_digest: :func:`work_digest` of the admitted work identifier.
        checkpoints: Declared long-lived checkpoint names, in order.
        binding: Pinned authority generations from the injected runtime.
    """

    admission_id: str
    work_digest: str
    checkpoints: tuple[str, ...]
    binding: AuthorityBinding

    def __post_init__(self) -> None:
        if (
            type(self.admission_id) is not str
            or _ADMISSION_RE.fullmatch(self.admission_id) is None
            or not _is_digest(self.work_digest)
            or type(self.binding) is not AuthorityBinding
            or type(self.checkpoints) is not tuple
        ):
            raise ContinuationError("invalid_continuation_record")
        try:
            _validate_checkpoints(self.checkpoints)
        except ContinuationError:
            raise ContinuationError("invalid_continuation_record") from None

    @property
    def epoch_digest(self) -> str:
        """Return the content-free commitment identifying this epoch."""

        return _hash({"schema_version": _EPOCH_VERSION, "record": self.to_dict()})

    @property
    def authority_count(self) -> int:
        """Return the number of authority contracts pinned by this epoch."""

        return len(self.binding.versions)

    def to_dict(self) -> dict[str, Any]:
        """Return exact persistable metadata without any authorization verdict."""

        return {
            "schema_version": CONTINUATION_RECORD_VERSION,
            "admission_id": self.admission_id,
            "work_digest": self.work_digest,
            "checkpoints": list(self.checkpoints),
            "authority": self.binding.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContinuationRecord":
        """Restore a record, rejecting missing, extra or cached-status fields.

        Args:
            payload: Mapping previously produced by :meth:`to_dict`.

        Returns:
            The restored record. It still requires a fresh boundary check.

        Raises:
            ContinuationError: If the mapping is not an exact record.
        """

        if (
            not isinstance(payload, Mapping)
            or set(payload) != _RECORD_FIELDS
            or payload["schema_version"] != CONTINUATION_RECORD_VERSION
            or type(payload["checkpoints"]) is not list
            or not isinstance(payload["authority"], Mapping)
        ):
            raise ContinuationError("invalid_continuation_record")
        try:
            binding = AuthorityBinding.from_dict(dict(payload["authority"]))
        except ValueError:
            raise ContinuationError("invalid_continuation_record") from None
        return cls(
            admission_id=payload["admission_id"],
            work_digest=payload["work_digest"],
            checkpoints=tuple(payload["checkpoints"]),
            binding=binding,
        )

    def __repr__(self) -> str:
        return f"ContinuationRecord(epoch_digest={self.epoch_digest!r})"


@dataclass(frozen=True, slots=True)
class ContinuationOutcome:
    """Content-free decision for one boundary, suitable for audit storage.

    Attributes:
        boundary: Boundary where the decision was made.
        disposition: Whether the work continues, completed, stopped or is
            quarantined.
        reason: Closed reason code.
        work_digest: Opaque digest of the work identifier.
        epoch_digest: Epoch commitment; ``None`` only for a denied admission.
        authority_count: Number of authority contracts consulted.
        checkpoint: Declared checkpoint name, when the boundary was one.
        delivered_output_count: Protected outputs already released.
        withheld_output_count: Buffered outputs discarded unreleased.
        committed_effect_count: Effects with commit evidence; never undone.
        pending_effect_count: Dispatched effects without commit evidence.
        committed_effect_digest: Commitment to committed effect keys and
            evidence, or ``None`` when nothing was committed.
    """

    boundary: ContinuationBoundary
    disposition: ContinuationDisposition
    reason: ContinuationReason
    work_digest: str
    epoch_digest: str | None
    authority_count: int
    checkpoint: str | None = None
    delivered_output_count: int = 0
    withheld_output_count: int = 0
    committed_effect_count: int = 0
    pending_effect_count: int = 0
    committed_effect_digest: str | None = None

    def __post_init__(self) -> None:
        try:
            if (
                type(self.boundary) is not ContinuationBoundary
                or type(self.disposition) is not ContinuationDisposition
                or type(self.reason) is not ContinuationReason
                or self.disposition is not _disposition_for(self.reason)
                or not _is_digest(self.work_digest)
            ):
                raise ContinuationError("invalid_continuation_outcome")
            if self.epoch_digest is None:
                if self.boundary is not ContinuationBoundary.ADMISSION:
                    raise ContinuationError("invalid_continuation_outcome")
            elif not _is_digest(self.epoch_digest):
                raise ContinuationError("invalid_continuation_outcome")
            if self.checkpoint is not None and (
                type(self.checkpoint) is not str
                or _CHECKPOINT_RE.fullmatch(self.checkpoint) is None
            ):
                raise ContinuationError("invalid_continuation_outcome")
            for value in (
                self.authority_count,
                self.delivered_output_count,
                self.withheld_output_count,
                self.committed_effect_count,
                self.pending_effect_count,
            ):
                _require_count(value)
            if (self.committed_effect_count == 0) != (
                self.committed_effect_digest is None
            ):
                raise ContinuationError("invalid_continuation_outcome")
            if self.committed_effect_digest is not None and not _is_digest(
                self.committed_effect_digest
            ):
                raise ContinuationError("invalid_continuation_outcome")
        except ContinuationError:
            raise ContinuationError("invalid_continuation_outcome") from None

    @property
    def reconciliation(self) -> ReconciliationState:
        """Return what is honestly known about effects under this epoch."""

        if self.pending_effect_count:
            return ReconciliationState.PENDING_EFFECTS_UNCONFIRMED
        if self.committed_effect_count:
            return ReconciliationState.COMMITTED_EFFECTS_RETAINED
        return ReconciliationState.NO_EFFECTS

    def to_dict(self) -> dict[str, Any]:
        """Return controlled metadata; compensation is never attempted."""

        return {
            "schema_version": CONTINUATION_OUTCOME_VERSION,
            "boundary": self.boundary.value,
            "disposition": self.disposition.value,
            "reason_code": self.reason.value,
            "reconciliation": self.reconciliation.value,
            "compensation": "not_attempted",
            "work_digest": self.work_digest,
            "epoch_digest": self.epoch_digest,
            "authority_count": self.authority_count,
            "checkpoint": self.checkpoint,
            "delivered_output_count": self.delivered_output_count,
            "withheld_output_count": self.withheld_output_count,
            "committed_effect_count": self.committed_effect_count,
            "pending_effect_count": self.pending_effect_count,
            "committed_effect_digest": self.committed_effect_digest,
        }


class ContinuationInterrupted(RuntimeError):
    """Raised when continued work must stop; carries only its outcome.

    Args:
        outcome: The terminal, content-free decision.
    """

    def __init__(self, outcome: ContinuationOutcome) -> None:
        if type(outcome) is not ContinuationOutcome:
            raise ContinuationError("invalid_continuation_outcome")
        self.outcome = outcome
        self.code = outcome.reason.value
        super().__init__(self.code)


class ContinuationLedger(Protocol):
    """Closed-epoch tombstones retained independently of job metadata.

    Production restart safety needs a durable, atomic implementation shared by
    every worker. A closed epoch can never be reopened.
    """

    def is_closed(self, epoch_digest: str) -> bool:
        """Return exactly ``True`` when the epoch was stopped or completed."""
        ...

    def close(self, epoch_digest: str) -> bool:
        """Durably and idempotently close an epoch; return ``True`` on success."""
        ...


class InMemoryContinuationLedger:
    """Thread-safe process-local ledger; restarts need a durable equivalent."""

    def __init__(self) -> None:
        self._closed: set[str] = set()
        self._lock = threading.Lock()

    def is_closed(self, epoch_digest: str) -> bool:
        """Return whether ``epoch_digest`` was closed in this process.

        Args:
            epoch_digest: :attr:`ContinuationRecord.epoch_digest`.

        Returns:
            ``True`` when the epoch is closed.
        """

        if not _is_digest(epoch_digest):
            raise ContinuationError("invalid_continuation_record")
        with self._lock:
            return epoch_digest in self._closed

    def close(self, epoch_digest: str) -> bool:
        """Close ``epoch_digest``; closing twice is a successful no-op.

        Args:
            epoch_digest: :attr:`ContinuationRecord.epoch_digest`.

        Returns:
            ``True`` once the tombstone is recorded.
        """

        if not _is_digest(epoch_digest):
            raise ContinuationError("invalid_continuation_record")
        with self._lock:
            self._closed.add(epoch_digest)
            return True


def _effect_summary(effects: Any) -> tuple[int, int, str | None]:
    if type(effects) is not tuple or len(effects) > MAX_CONTINUATION_EFFECTS:
        raise ContinuationError("invalid_continuation_effect")
    if any(type(effect) is not EffectRecord for effect in effects):
        raise ContinuationError("invalid_continuation_effect")
    committed = [e for e in effects if e.state is EffectState.COMMITTED]
    pending = len(effects) - len(committed)
    if not committed:
        return 0, pending, None
    digest = _hash(
        {
            "schema_version": _EFFECTS_VERSION,
            "effects": [
                {
                    "idempotency_key": e.idempotency_key,
                    "commit_evidence_digest": e.commit_evidence_digest,
                }
                for e in sorted(committed, key=lambda e: e.ordinal)
            ],
        }
    )
    return len(committed), pending, digest


class ContinuationGuard:
    """Revalidate admitted work against current authority at every boundary.

    Args:
        runtime: Injected :class:`AuthorityRuntime` with the application's
            status provider, generation store and trusted clock.
        ledger: Closed-epoch store; defaults to a process-local ledger.

    Raises:
        ContinuationError: If the runtime or ledger is not usable.
    """

    def __init__(
        self,
        runtime: AuthorityRuntime,
        *,
        ledger: ContinuationLedger | None = None,
    ) -> None:
        if not isinstance(runtime, AuthorityRuntime):
            raise ContinuationError("invalid_continuation_guard")
        ledger = ledger if ledger is not None else InMemoryContinuationLedger()
        if not callable(getattr(ledger, "is_closed", None)) or not callable(
            getattr(ledger, "close", None)
        ):
            raise ContinuationError("invalid_continuation_guard")
        self._runtime = runtime
        self._ledger = ledger

    def admit(
        self,
        work_id: str,
        contracts: tuple[AuthorityContract, ...],
        *,
        checkpoints: tuple[str, ...] = (),
    ) -> ContinuationRecord:
        """Pin current authority generations to newly admitted work.

        Call after the existing admission checks (authentication, route scopes
        and request limits) have passed.

        Args:
            work_id: Caller-owned job or stream identifier; only its digest is
                kept.
            contracts: Every authority the work relies on, for example from
                :func:`principal_authority` and :func:`consent_authority`.
            checkpoints: Long-lived checkpoint names the worker will declare.

        Returns:
            The persistable :class:`ContinuationRecord`.

        Raises:
            ContinuationError: If an argument is malformed.
            ContinuationInterrupted: If any authority is not currently active.
        """

        digest = work_digest(work_id)
        names = _validate_checkpoints(checkpoints)
        if (
            not isinstance(contracts, tuple)
            or not contracts
            or any(type(c) is not AuthorityContract for c in contracts)
        ):
            raise ContinuationError("invalid_continuation_authority")
        try:
            binding = self._runtime.bind(contracts)
        except AuthorityDeniedError as denied:
            reason = ContinuationReason(denied.receipt.reason.value)
            raise ContinuationInterrupted(
                ContinuationOutcome(
                    boundary=ContinuationBoundary.ADMISSION,
                    disposition=_disposition_for(reason),
                    reason=reason,
                    work_digest=digest,
                    epoch_digest=None,
                    authority_count=len(contracts),
                )
            ) from None
        except ValueError:
            raise ContinuationError("invalid_continuation_authority") from None
        return ContinuationRecord(
            admission_id="adm_" + secrets.token_hex(16),
            work_digest=digest,
            checkpoints=names,
            binding=binding,
        )

    def check(
        self,
        record: ContinuationRecord,
        work_id: str,
        boundary: ContinuationBoundary,
        *,
        checkpoint: str | None = None,
        delivered_outputs: int = 0,
        buffered_outputs: int = 0,
        effects: tuple[EffectRecord, ...] = (),
    ) -> ContinuationOutcome:
        """Decide whether work may cross ``boundary`` right now.

        Every call performs fresh provider lookups through the runtime. Any
        decision other than ``continue`` closes the epoch in the ledger, so the
        same record can never continue later, even after a restart.

        Args:
            record: Restored or in-memory continuation record.
            work_id: Identifier of the work presenting the record.
            boundary: Boundary being crossed; not ``admission``.
            checkpoint: Declared checkpoint name; required for ``checkpoint``.
            delivered_outputs: Protected outputs already released.
            buffered_outputs: Outputs held but not released; they are counted
                as withheld when the work stops.
            effects: Effects dispatched so far under this epoch.

        Returns:
            The boundary outcome. Denials are returned, not raised.

        Raises:
            ContinuationError: If an argument is malformed.
        """

        self._validate_call(record, boundary, checkpoint)
        _require_count(delivered_outputs)
        _require_count(buffered_outputs)
        summary = _effect_summary(effects)
        reason, declared = self._evaluate(record, work_id, boundary, checkpoint)
        return self._decide(
            record,
            boundary,
            reason,
            checkpoint=declared,
            delivered=delivered_outputs,
            buffered=buffered_outputs,
            summary=summary,
        )

    def require(
        self,
        record: ContinuationRecord,
        work_id: str,
        boundary: ContinuationBoundary,
        **progress: Any,
    ) -> ContinuationOutcome:
        """Like :meth:`check`, but raise when the work may not continue.

        Args:
            record: Continuation record.
            work_id: Identifier of the work presenting the record.
            boundary: Boundary being crossed.
            **progress: Keyword arguments accepted by :meth:`check`.

        Returns:
            The ``continue`` outcome.

        Raises:
            ContinuationInterrupted: If the decision is not ``continue``.
        """

        outcome = self.check(record, work_id, boundary, **progress)
        if outcome.disposition is not ContinuationDisposition.CONTINUE:
            raise ContinuationInterrupted(outcome)
        return outcome

    def cancel(
        self,
        record: ContinuationRecord,
        work_id: str,
        *,
        boundary: ContinuationBoundary = ContinuationBoundary.EXECUTION_START,
        delivered_outputs: int = 0,
        buffered_outputs: int = 0,
        effects: tuple[EffectRecord, ...] = (),
    ) -> ContinuationOutcome:
        """Record cancellation and close the epoch without undoing anything.

        Args:
            record: Continuation record of the cancelled work.
            work_id: Identifier of the work presenting the record.
            boundary: Boundary where cancellation was observed.
            delivered_outputs: Protected outputs already released.
            buffered_outputs: Outputs discarded unreleased.
            effects: Effects dispatched so far; they stay recorded as is.

        Returns:
            A ``stopped`` outcome, or ``quarantined`` when the record does not
            belong to ``work_id`` or the ledger cannot close the epoch.
        """

        self._validate_call(record, boundary, None)
        _require_count(delivered_outputs)
        _require_count(buffered_outputs)
        summary = _effect_summary(effects)
        reason = ContinuationReason.CANCELLED
        if not self._work_matches(record, work_id):
            reason = ContinuationReason.WORK_MISMATCH
        return self._decide(
            record,
            boundary,
            reason,
            checkpoint=None,
            delivered=delivered_outputs,
            buffered=buffered_outputs,
            summary=summary,
        )

    def _validate_call(
        self,
        record: Any,
        boundary: Any,
        checkpoint: Any,
    ) -> None:
        if type(record) is not ContinuationRecord:
            raise ContinuationError("invalid_continuation_record")
        if (
            type(boundary) is not ContinuationBoundary
            or boundary is ContinuationBoundary.ADMISSION
        ):
            raise ContinuationError("invalid_continuation_boundary")
        if checkpoint is not None and boundary is not ContinuationBoundary.CHECKPOINT:
            raise ContinuationError("invalid_continuation_checkpoint")

    @staticmethod
    def _work_matches(record: ContinuationRecord, work_id: str) -> bool:
        return hmac.compare_digest(work_digest(work_id), record.work_digest)

    def _evaluate(
        self,
        record: ContinuationRecord,
        work_id: str,
        boundary: ContinuationBoundary,
        checkpoint: str | None,
    ) -> tuple[ContinuationReason, str | None]:
        if not self._work_matches(record, work_id):
            return ContinuationReason.WORK_MISMATCH, None
        declared = None
        if boundary is ContinuationBoundary.CHECKPOINT:
            if checkpoint not in record.checkpoints:
                return ContinuationReason.UNDECLARED_CHECKPOINT, None
            declared = checkpoint
        try:
            closed = self._ledger.is_closed(record.epoch_digest)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            return ContinuationReason.LEDGER_UNAVAILABLE, declared
        if type(closed) is not bool:
            return ContinuationReason.LEDGER_UNAVAILABLE, declared
        if closed:
            return ContinuationReason.TERMINAL, declared
        try:
            self._runtime.check(record.binding, _AUTHORITY_BOUNDARIES[boundary])
        except AuthorityDeniedError as denied:
            return ContinuationReason(denied.receipt.reason.value), declared
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            return ContinuationReason.AUTHORITY_UNAVAILABLE, declared
        return ContinuationReason.ACTIVE, declared

    def _decide(
        self,
        record: ContinuationRecord,
        boundary: ContinuationBoundary,
        reason: ContinuationReason,
        *,
        checkpoint: str | None,
        delivered: int,
        buffered: int,
        summary: tuple[int, int, str | None],
    ) -> ContinuationOutcome:
        if reason is not ContinuationReason.ACTIVE and reason is not (
            ContinuationReason.TERMINAL
        ):
            if not self._close(record):
                reason = ContinuationReason.LEDGER_UNAVAILABLE
        disposition = _disposition_for(reason)
        released = disposition in (
            ContinuationDisposition.CONTINUE,
            ContinuationDisposition.COMPLETED,
        )
        committed, pending, digest = summary
        return ContinuationOutcome(
            boundary=boundary,
            disposition=disposition,
            reason=reason,
            work_digest=record.work_digest,
            epoch_digest=record.epoch_digest,
            authority_count=record.authority_count,
            checkpoint=checkpoint,
            delivered_output_count=delivered,
            withheld_output_count=0 if released else buffered,
            committed_effect_count=committed,
            pending_effect_count=pending,
            committed_effect_digest=digest,
        )

    def _close(self, record: ContinuationRecord) -> bool:
        try:
            return self._ledger.close(record.epoch_digest) is True
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            return False

    def _terminate(
        self,
        record: ContinuationRecord,
        boundary: ContinuationBoundary,
        reason: ContinuationReason,
        *,
        delivered: int,
        buffered: int,
        effects: tuple[EffectRecord, ...],
    ) -> ContinuationOutcome:
        return self._decide(
            record,
            boundary,
            reason,
            checkpoint=None,
            delivered=delivered,
            buffered=buffered,
            summary=_effect_summary(effects),
        )


class ContinuationContext:
    """Worker-facing handle that guards checkpoints, effects and outputs.

    Workers receive one context from :func:`run_continuation`. Outputs passed
    to :meth:`emit` are buffered and released only after the final protected
    output check; an interruption discards them.
    """

    def __init__(
        self,
        guard: ContinuationGuard,
        record: ContinuationRecord,
        work_id: str,
        cancellation: CancellationSignal | None,
    ) -> None:
        self._guard = guard
        self._record = record
        self._work_id = work_id
        self._cancellation = cancellation
        self._outputs: list[Any] = []
        self._effects: list[EffectRecord] = []
        self._boundary = ContinuationBoundary.EXECUTION_START
        self._interruption: ContinuationOutcome | None = None
        self._issued: list[ContinuationOutcome] = []

    @property
    def effects(self) -> tuple[EffectRecord, ...]:
        """Return effects dispatched so far, in dispatch order."""

        return tuple(self._effects)

    def checkpoint(self, name: str) -> None:
        """Revalidate authority at a declared long-lived checkpoint.

        Args:
            name: Checkpoint name declared at admission.

        Raises:
            ContinuationInterrupted: If the work may not continue.
        """

        self._require(ContinuationBoundary.CHECKPOINT, checkpoint=name)

    def commit_effect(
        self, effect: EffectRecord, dispatch: Callable[[], str]
    ) -> EffectRecord:
        """Revalidate, dispatch one effect and record it honestly.

        The effect is recorded as pending before ``dispatch`` runs and becomes
        committed only when ``dispatch`` returns a ``sha256:`` commit evidence
        digest. A failed or malformed dispatch leaves it pending, which the
        outcome reports as unconfirmed rather than undone.

        Args:
            effect: Pending effect with a deterministic idempotency key.
            dispatch: Callable performing the effect and returning evidence.

        Returns:
            The committed effect record.

        Raises:
            ContinuationInterrupted: If authority is not current.
            ContinuationError: If the effect or evidence is malformed.
        """

        if (
            type(effect) is not EffectRecord
            or effect.state is not EffectState.PENDING
            or not callable(dispatch)
            or len(self._effects) >= MAX_CONTINUATION_EFFECTS
            or any(e.idempotency_key == effect.idempotency_key for e in self._effects)
        ):
            raise ContinuationError("invalid_continuation_effect")
        self._require(ContinuationBoundary.EFFECT_DISPATCH)
        index = len(self._effects)
        self._effects.append(effect)
        evidence = dispatch()
        if not _is_digest(evidence):
            raise ContinuationError("invalid_continuation_commit_evidence")
        committed = replace(
            effect, state=EffectState.COMMITTED, commit_evidence_digest=evidence
        )
        self._effects[index] = committed
        return committed

    def emit(self, output: Any) -> None:
        """Buffer one protected output until the final release check.

        Args:
            output: Worker result; never inspected, logged or persisted here.
        """

        self._outputs.append(output)

    def _require(
        self,
        boundary: ContinuationBoundary,
        *,
        checkpoint: str | None = None,
    ) -> ContinuationOutcome:
        self._boundary = boundary
        if self._cancellation is not None and self._cancellation.cancelled:
            outcome = self._guard.cancel(
                self._record,
                self._work_id,
                boundary=boundary,
                buffered_outputs=len(self._outputs),
                effects=self.effects,
            )
        else:
            outcome = self._guard.check(
                self._record,
                self._work_id,
                boundary,
                checkpoint=checkpoint,
                buffered_outputs=len(self._outputs),
                effects=self.effects,
            )
        if outcome.disposition is not ContinuationDisposition.CONTINUE:
            self._issued.append(outcome)
            if self._interruption is None:
                self._interruption = outcome
            raise ContinuationInterrupted(outcome)
        return outcome


@dataclass(frozen=True, slots=True, repr=False)
class ContinuationResult:
    """Final outcome of one guarded run, its released outputs and effects.

    Attributes:
        outcome: Terminal :class:`ContinuationOutcome`.
        outputs: Released outputs; empty unless the run completed.
        effects: Every effect dispatched under the epoch, as recorded.
    """

    outcome: ContinuationOutcome
    outputs: tuple[Any, ...]
    effects: tuple[EffectRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the outcome and content-free effect records; never outputs."""

        return {
            "outcome": self.outcome.to_dict(),
            "effects": [effect.to_dict() for effect in self.effects],
        }

    def __repr__(self) -> str:
        return (
            "ContinuationResult("
            f"disposition={self.outcome.disposition.value!r}, "
            f"reason={self.outcome.reason.value!r}, "
            f"outputs={len(self.outputs)}, effects={len(self.effects)})"
        )


def _validate_runner(guard: Any, record: Any, cancellation: Any) -> None:
    if type(guard) is not ContinuationGuard:
        raise ContinuationError("invalid_continuation_guard")
    if type(record) is not ContinuationRecord:
        raise ContinuationError("invalid_continuation_record")
    if cancellation is not None and type(cancellation) is not CancellationSignal:
        raise ContinuationError("invalid_continuation_cancellation")


def run_continuation(
    guard: ContinuationGuard,
    record: ContinuationRecord,
    work_id: str,
    worker: Callable[[ContinuationContext], None],
    *,
    resumed: bool = False,
    cancellation: CancellationSignal | None = None,
) -> ContinuationResult:
    """Run one queued unit of work under continuous authorization.

    The start (or resume) boundary is checked before ``worker`` is called, each
    checkpoint and effect is checked by the context, and buffered outputs are
    released only after a final protected output check. A worker exception is
    contained as a ``continuation_worker_failed`` quarantine without keeping its
    message or traceback.

    Args:
        guard: Continuation guard built on the injected authority runtime.
        record: Admitted or restored continuation record.
        work_id: Identifier of the work being executed.
        worker: Callable receiving a :class:`ContinuationContext`.
        resumed: ``True`` when a restarted worker resumes persisted work.
        cancellation: Optional existing cooperative cancellation signal.

    Returns:
        The :class:`ContinuationResult`.

    Raises:
        ContinuationError: If an argument is malformed.
    """

    _validate_runner(guard, record, cancellation)
    if not callable(worker) or type(resumed) is not bool:
        raise ContinuationError("invalid_continuation_worker")
    work_digest(work_id)
    context = ContinuationContext(guard, record, work_id, cancellation)
    start = (
        ContinuationBoundary.RESUME if resumed else ContinuationBoundary.EXECUTION_START
    )
    outcome: ContinuationOutcome | None = None
    failed = False
    try:
        context._require(start)
        worker(context)
        context._require(ContinuationBoundary.PROTECTED_OUTPUT)
    except ContinuationInterrupted as interrupted:
        if not any(interrupted.outcome is issued for issued in context._issued):
            failed = True
    except (KeyboardInterrupt, SystemExit):
        context._outputs.clear()
        raise
    except Exception:
        failed = True
    effects = context.effects
    if context._interruption is not None:
        # The first stop is authoritative even if the worker swallowed it and
        # kept going; later boundaries could only report the closed epoch.
        outcome = replace(
            context._interruption, withheld_output_count=len(context._outputs)
        )
    elif failed:
        outcome = guard._terminate(
            record,
            context._boundary,
            ContinuationReason.WORKER_FAILED,
            delivered=0,
            buffered=len(context._outputs),
            effects=effects,
        )
    if outcome is None:
        # Completion closes the epoch; if the ledger cannot record that, the
        # buffered outputs stay withheld rather than being released unrecorded.
        outputs = tuple(context._outputs)
        outcome = guard._decide(
            record,
            ContinuationBoundary.PROTECTED_OUTPUT,
            ContinuationReason.COMPLETED,
            checkpoint=None,
            delivered=len(outputs),
            buffered=len(outputs),
            summary=_effect_summary(effects),
        )
        if outcome.disposition is ContinuationDisposition.COMPLETED:
            return ContinuationResult(outcome, outputs, effects)
        outcome = replace(outcome, delivered_output_count=0)
    context._outputs.clear()
    return ContinuationResult(outcome, (), effects)


def continuation_error_event(outcome: ContinuationOutcome) -> dict[str, Any]:
    """Return an NDJSON-style terminal error event for a stopped stream.

    The shape matches the service streaming error envelope and carries only
    controlled codes, counts and the epoch digest.

    Args:
        outcome: Terminal outcome reported by :class:`GuardedStream`.

    Returns:
        A ``{"type": "error", "error": {...}}`` mapping.
    """

    if type(outcome) is not ContinuationOutcome:
        raise ContinuationError("invalid_continuation_outcome")
    return {
        "type": "error",
        "error": {
            "code": outcome.reason.value,
            "message": "Continuation authorization ended the response",
            "details": {
                "boundary": outcome.boundary.value,
                "disposition": outcome.disposition.value,
                "delivered_output_count": outcome.delivered_output_count,
                "withheld_output_count": outcome.withheld_output_count,
                "epoch_digest": outcome.epoch_digest,
            },
        },
    }


class GuardedStream:
    """Single-use stream wrapper that rechecks authority before every chunk.

    Iterate synchronously over an iterable source or asynchronously over an
    async iterable source. The start boundary is checked before the source is
    read. Each chunk is pulled, then released only after a fresh protected
    output check; a denied chunk is discarded and counted as withheld. The
    source is always closed. Consumer disconnects and task cancellation are
    recorded as ``continuation_cancelled``.

    Args:
        guard: Continuation guard built on the injected authority runtime.
        record: Admitted or restored continuation record.
        work_id: Identifier of the stream being served.
        source: Injected chunk source (sync or async iterable).
        cancellation: Optional existing cooperative cancellation signal.
        terminal: Optional callable turning the terminal outcome into one last
            item, for example :func:`continuation_error_event`. Without it a
            stop raises :class:`ContinuationInterrupted`.

    Attributes:
        outcome: Terminal outcome once the stream ends, else ``None``.
    """

    def __init__(
        self,
        guard: ContinuationGuard,
        record: ContinuationRecord,
        work_id: str,
        source: Any,
        *,
        cancellation: CancellationSignal | None = None,
        terminal: Callable[[ContinuationOutcome], Any] | None = None,
    ) -> None:
        _validate_runner(guard, record, cancellation)
        work_digest(work_id)
        if terminal is not None and not callable(terminal):
            raise ContinuationError("invalid_continuation_stream")
        self._guard = guard
        self._record = record
        self._work_id = work_id
        self._source = source
        self._cancellation = cancellation
        self._terminal = terminal
        self._claimed = False
        self._delivered = 0
        self.outcome: ContinuationOutcome | None = None

    def __iter__(self) -> Iterator[Any]:
        """Return the guarded synchronous iterator; usable once."""

        self._claim()
        return self._iterate()

    def __aiter__(self) -> AsyncIterator[Any]:
        """Return the guarded asynchronous iterator; usable once."""

        self._claim()
        return self._aiterate()

    def _claim(self) -> None:
        if self._claimed:
            raise ContinuationError("invalid_continuation_stream")
        self._claimed = True

    def _cancelled(self) -> bool:
        return self._cancellation is not None and self._cancellation.cancelled

    def _gate(
        self, boundary: ContinuationBoundary, buffered: int
    ) -> ContinuationOutcome | None:
        if self._cancelled():
            outcome = self._guard.cancel(
                self._record,
                self._work_id,
                boundary=boundary,
                delivered_outputs=self._delivered,
                buffered_outputs=buffered,
            )
        else:
            outcome = self._guard.check(
                self._record,
                self._work_id,
                boundary,
                delivered_outputs=self._delivered,
                buffered_outputs=buffered,
            )
        if outcome.disposition is ContinuationDisposition.CONTINUE:
            return None
        return outcome

    def _stop(self, reason: ContinuationReason, buffered: int) -> ContinuationOutcome:
        return self._guard._terminate(
            self._record,
            ContinuationBoundary.PROTECTED_OUTPUT,
            reason,
            delivered=self._delivered,
            buffered=buffered,
            effects=(),
        )

    def _finish(self) -> ContinuationOutcome:
        return self._guard._decide(
            self._record,
            ContinuationBoundary.PROTECTED_OUTPUT,
            ContinuationReason.COMPLETED,
            checkpoint=None,
            delivered=self._delivered,
            buffered=0,
            summary=(0, 0, None),
        )

    def _disconnected(self) -> None:
        if self.outcome is None:
            self.outcome = self._guard.cancel(
                self._record,
                self._work_id,
                boundary=ContinuationBoundary.PROTECTED_OUTPUT,
                delivered_outputs=self._delivered,
            )

    def _iterate(self) -> Iterator[Any]:
        iterator: Any = None
        try:
            stop = self._gate(ContinuationBoundary.EXECUTION_START, 0)
            if stop is None:
                iterator = iter(self._source)
            while stop is None:
                if self._cancelled():
                    stop = self._gate(ContinuationBoundary.PROTECTED_OUTPUT, 0)
                    break
                item = _END_OF_SOURCE
                failed = False
                try:
                    item = next(iterator)
                except StopIteration:
                    pass
                except Exception:
                    failed = True
                if failed:
                    stop = self._stop(ContinuationReason.SOURCE_FAILED, 0)
                    break
                if item is _END_OF_SOURCE:
                    self.outcome = self._finish()
                    return
                stop = self._gate(ContinuationBoundary.PROTECTED_OUTPUT, 1)
                if stop is not None:
                    item = None
                    break
                self._delivered += 1
                yield item
                item = None
            self.outcome = stop
            if self._terminal is None:
                raise ContinuationInterrupted(stop)
            yield self._terminal(stop)
        except GeneratorExit:
            self._disconnected()
            raise
        finally:
            _close_sync(iterator if iterator is not None else self._source)

    async def _aiterate(self) -> AsyncIterator[Any]:
        iterator: Any = None
        try:
            stop = self._gate(ContinuationBoundary.EXECUTION_START, 0)
            if stop is None:
                iterator = self._source.__aiter__()
            while stop is None:
                if self._cancelled():
                    stop = self._gate(ContinuationBoundary.PROTECTED_OUTPUT, 0)
                    break
                item = _END_OF_SOURCE
                failed = False
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    pass
                except Exception:
                    failed = True
                if failed:
                    stop = self._stop(ContinuationReason.SOURCE_FAILED, 0)
                    break
                if item is _END_OF_SOURCE:
                    self.outcome = self._finish()
                    return
                stop = self._gate(ContinuationBoundary.PROTECTED_OUTPUT, 1)
                if stop is not None:
                    item = None
                    break
                self._delivered += 1
                yield item
                item = None
            self.outcome = stop
            if self._terminal is None:
                raise ContinuationInterrupted(stop)
            yield self._terminal(stop)
        except (GeneratorExit, asyncio.CancelledError):
            self._disconnected()
            raise
        finally:
            await _close_async(iterator if iterator is not None else self._source)


def _close_sync(source: Any) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


async def _close_async(source: Any) -> None:
    aclose = getattr(source, "aclose", None)
    try:
        if callable(aclose):
            await aclose()
        else:
            _close_sync(source)
    except Exception:
        pass


__all__ = [
    "CONTINUATION_OUTCOME_VERSION",
    "CONTINUATION_RECORD_VERSION",
    "MAX_CONTINUATION_EFFECTS",
    "MAX_DECLARED_CHECKPOINTS",
    "ContinuationBoundary",
    "ContinuationContext",
    "ContinuationDisposition",
    "ContinuationError",
    "ContinuationGuard",
    "ContinuationInterrupted",
    "ContinuationLedger",
    "ContinuationOutcome",
    "ContinuationReason",
    "ContinuationRecord",
    "ContinuationResult",
    "GuardedStream",
    "InMemoryContinuationLedger",
    "ReconciliationState",
    "consent_authority",
    "continuation_error_event",
    "principal_authority",
    "run_continuation",
    "work_digest",
]
