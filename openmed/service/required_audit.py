"""Fail-closed required audit for declared state-changing service operations.

A declared operation may change state only after its required audit intent is
durable, and it is reported as a fully evidenced success only after its
required commit record is durable and acknowledged. When required audit storage
is full, rejects an append, or loses an acknowledgment, the outcome follows the
operation's commit boundary: a fault before the effect is invoked is a denial,
and a fault after the effect is invoked is a reconcile-required receipt whose
evidence stays pending. Reconciliation never invokes the effect again. It reads
the required store, queries an injected observer by opaque effect key, and
appends the missing closing record exactly once.

Required evidence is persisted through the existing hash-chained
:class:`~openmed.agent.audit.action_ledger.ActionLedger` via
:class:`ActionLedgerAuditStore`. Optional diagnostics, such as a
:class:`~openmed.compliance.audit_chain.AuditSink`, observe outcomes on a best
effort basis and can never substitute for required evidence.

Receipts, records, diagnostics and errors carry only fixed codes, opaque
identifiers and digests. Request keys are hashed and never persisted.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from openmed.agent.audit.action_ledger import (
    ActionLedger,
    ActionLedgerError,
    ActionState,
)
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.workflows.recovery import ObservationState
from openmed.compliance.audit_chain import AuditSink

__all__ = [
    "ActionLedgerAuditStore",
    "AuditReceipt",
    "AuditStage",
    "CommitBoundary",
    "DEFAULT_MAX_OPEN_RECEIPTS",
    "EffectObserver",
    "EvidenceState",
    "MAX_OPEN_RECEIPTS_LIMIT",
    "REQUIRED_AUDIT_SCHEMA_VERSION",
    "RequiredAuditCoordinator",
    "RequiredAuditEffect",
    "RequiredAuditError",
    "RequiredAuditOutcome",
    "RequiredAuditPolicy",
    "RequiredAuditReason",
    "RequiredAuditRecord",
    "RequiredAuditStore",
    "RequiredAuditStoreError",
    "StoredAuditRecord",
    "derive_receipt_id",
]

REQUIRED_AUDIT_SCHEMA_VERSION: Final = "openmed.service.required_audit.v1"
DIAGNOSTIC_EVENT_TYPE: Final = "openmed.service.required_audit.outcome"
DEFAULT_MAX_OPEN_RECEIPTS: Final = 64
MAX_OPEN_RECEIPTS_LIMIT: Final = 4096
DEFAULT_ACTOR_ROLE: Final = "role:openmed.service/required-audit"

_OPERATION_RE = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_REQUEST_KEY_RE = re.compile(r"[A-Za-z0-9_-]{16,128}")
_RECEIPT_RE = re.compile(r"rcpt_[0-9a-f]{32}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_EFFECT_KEY_RE = re.compile(r"idem_[0-9a-f]{64}")
_ROLE_RE = re.compile(r"role:[a-z][a-z0-9.-]{0,62}/[a-z][a-z0-9-]{0,63}")

_STORE_ERROR_CODES: Final = frozenset(
    {
        "audit_storage_full",
        "audit_append_rejected",
        "audit_unavailable",
        "audit_read_failed",
    }
)
_NOT_WRITTEN_CODES: Final = frozenset({"audit_storage_full", "audit_append_rejected"})


class RequiredAuditError(ValueError):
    """Configuration or argument failure carrying only a fixed code.

    Args:
        code: Stable machine-readable failure code. Submitted values are never
            included in the message.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RequiredAuditStoreError(RuntimeError):
    """Required audit store failure carrying only a closed code.

    Stores raise ``audit_storage_full`` or ``audit_append_rejected`` only when
    the record is known not to be durable. Any failure whose durability is
    unknown, including an interrupted acknowledgment, is ``audit_unavailable``.

    Args:
        code: One of the closed store failure codes. Unknown codes collapse to
            ``audit_unavailable`` so arbitrary text never escapes.
    """

    def __init__(self, code: str) -> None:
        self.code = code if code in _STORE_ERROR_CODES else "audit_unavailable"
        super().__init__(self.code)

    @property
    def not_written(self) -> bool:
        """Return whether the store guarantees the record was not persisted."""
        return self.code in _NOT_WRITTEN_CODES


class CommitBoundary(str, Enum):
    """Declared trust in an effect adapter failure at the commit boundary.

    Every declared operation crosses its commit boundary when the effect
    adapter is invoked. ``INVOCATION`` treats any adapter failure as an
    effect whose outcome is unknown. ``ACKNOWLEDGED`` declares that the adapter
    commits atomically on return, so a raised failure proves no commit.
    """

    INVOCATION = "invocation"
    ACKNOWLEDGED = "acknowledged"


class AuditStage(str, Enum):
    """Closed lifecycle of required audit records for one receipt."""

    INTENT = "intent"
    ATTEMPT = "attempt"
    COMMIT = "commit"
    ABORT = "abort"


class RequiredAuditOutcome(str, Enum):
    """Caller-visible outcome of a declared operation."""

    COMPLETED = "completed"
    DENIED = "denied"
    RECONCILE_REQUIRED = "reconcile_required"


class EvidenceState(str, Enum):
    """Durability of the required audit evidence behind a receipt."""

    COMPLETE = "complete"
    PENDING = "pending"
    NONE = "none"


class RequiredAuditReason(str, Enum):
    """Closed reason codes for receipts."""

    COMMITTED = "committed"
    ALREADY_RECORDED = "already_recorded"
    OPERATION_UNDECLARED = "operation_undeclared"
    REQUIRED_AUDIT_UNCONFIGURED = "required_audit_unconfigured"
    AUDIT_READ_FAILED = "audit_read_failed"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    RECONCILE_BACKLOG_FULL = "reconcile_backlog_full"
    RECONCILE_PENDING = "reconcile_pending"
    RECEIPT_IN_FLIGHT = "receipt_in_flight"
    INTENT_NOT_DURABLE = "intent_not_durable"
    ATTEMPT_NOT_DURABLE = "attempt_not_durable"
    EFFECT_NOT_COMMITTED = "effect_not_committed"
    EFFECT_OUTCOME_UNKNOWN = "effect_outcome_unknown"
    COMMIT_NOT_DURABLE = "commit_not_durable"
    ABORT_NOT_DURABLE = "abort_not_durable"
    RECONCILED_EXISTING = "reconciled_existing"
    RECONCILED_COMMIT = "reconciled_commit"
    RECONCILED_ABSENT = "reconciled_absent"
    RECONCILED_NOT_ATTEMPTED = "reconciled_not_attempted"
    NO_DURABLE_INTENT = "no_durable_intent"
    EFFECT_AMBIGUOUS = "effect_ambiguous"
    OBSERVER_UNAVAILABLE = "observer_unavailable"
    EVIDENCE_MISMATCH = "evidence_mismatch"


_VALID_SEQUENCES: Final = frozenset(
    {
        (AuditStage.INTENT,),
        (AuditStage.INTENT, AuditStage.ABORT),
        (AuditStage.INTENT, AuditStage.ATTEMPT),
        (AuditStage.INTENT, AuditStage.ATTEMPT, AuditStage.COMMIT),
        (AuditStage.INTENT, AuditStage.ATTEMPT, AuditStage.ABORT),
    }
)
_TERMINAL: Final = frozenset({AuditStage.COMMIT, AuditStage.ABORT})


def _digest(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _require_operation(operation: object) -> str:
    if type(operation) is not str or _OPERATION_RE.fullmatch(operation) is None:
        raise RequiredAuditError("invalid_operation")
    return operation


def _require_receipt_id(receipt_id: object) -> str:
    if type(receipt_id) is not str or _RECEIPT_RE.fullmatch(receipt_id) is None:
        raise RequiredAuditError("invalid_receipt_id")
    return receipt_id


def _require_digest(value: object, code: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise RequiredAuditError(code)
    return value


def derive_receipt_id(operation: str, request_key: str) -> str:
    """Derive the stable opaque receipt identifier for one logical request.

    Args:
        operation: Declared operation code.
        request_key: Caller-chosen opaque idempotency token (16 to 128 ASCII
            letters, digits, ``_`` or ``-``). It must not encode identifiers;
            only its digest contributes to the receipt.

    Returns:
        A ``rcpt_`` identifier with 32 lowercase hex characters.

    Raises:
        RequiredAuditError: If the operation or request key is malformed.
    """
    _require_operation(operation)
    if type(request_key) is not str or _REQUEST_KEY_RE.fullmatch(request_key) is None:
        raise RequiredAuditError("invalid_request_key")
    token = _digest(
        {
            "schema": REQUIRED_AUDIT_SCHEMA_VERSION,
            "operation": operation,
            "request_key": request_key,
        }
    )
    return "rcpt_" + token[:32]


def _effect_key(receipt_id: str) -> str:
    token = _digest({"schema": REQUIRED_AUDIT_SCHEMA_VERSION, "receipt_id": receipt_id})
    return "idem_" + token


def _binding_digest(operation: str, operation_digest: str) -> str:
    token = _digest(
        {
            "schema": REQUIRED_AUDIT_SCHEMA_VERSION,
            "operation": operation,
            "operation_digest": operation_digest,
        }
    )
    return "sha256:" + token


@dataclass(frozen=True, slots=True)
class RequiredAuditPolicy:
    """Explicit declaration of operations whose audit evidence is required.

    Undeclared operations are denied. There is no optional mode for a declared
    operation: diagnostics never satisfy the requirement.

    Args:
        operations: Mapping from declared operation code to its commit
            boundary.
        max_open_receipts: Maximum number of receipts whose required evidence
            may remain open before new admissions are denied.

    Raises:
        RequiredAuditError: If the declaration is empty or malformed.
    """

    operations: Mapping[str, CommitBoundary]
    max_open_receipts: int = DEFAULT_MAX_OPEN_RECEIPTS

    def __post_init__(self) -> None:
        if not isinstance(self.operations, Mapping):
            raise RequiredAuditError("invalid_policy")
        if not self.operations:
            raise RequiredAuditError("empty_policy")
        declared: dict[str, CommitBoundary] = {}
        for operation, boundary in self.operations.items():
            _require_operation(operation)
            if type(boundary) is not CommitBoundary:
                raise RequiredAuditError("invalid_commit_boundary")
            declared[operation] = boundary
        if (
            type(self.max_open_receipts) is not int
            or not 1 <= self.max_open_receipts <= MAX_OPEN_RECEIPTS_LIMIT
        ):
            raise RequiredAuditError("invalid_open_receipt_bound")
        object.__setattr__(
            self, "operations", MappingProxyType(dict(sorted(declared.items())))
        )

    def boundary_for(self, operation: str) -> CommitBoundary | None:
        """Return the declared commit boundary, or ``None`` when undeclared.

        Args:
            operation: Operation code to look up.

        Returns:
            The declared boundary, or ``None``.
        """
        return self.operations.get(operation)

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free policy declaration."""
        return {
            "schema_version": REQUIRED_AUDIT_SCHEMA_VERSION,
            "operations": {op: b.value for op, b in self.operations.items()},
            "max_open_receipts": self.max_open_receipts,
        }


@dataclass(frozen=True, slots=True)
class RequiredAuditRecord:
    """One value-free required audit record for a receipt stage.

    Args:
        receipt_id: Opaque receipt identifier.
        operation: Declared operation code.
        stage: Lifecycle stage being recorded.
        binding_digest: Digest binding the operation to its payload digest.

    Raises:
        RequiredAuditError: If any field is malformed.
    """

    receipt_id: str
    operation: str
    stage: AuditStage
    binding_digest: str

    def __post_init__(self) -> None:
        _require_receipt_id(self.receipt_id)
        _require_operation(self.operation)
        if type(self.stage) is not AuditStage:
            raise RequiredAuditError("invalid_stage")
        _require_digest(self.binding_digest, "invalid_binding_digest")


@dataclass(frozen=True, slots=True)
class StoredAuditRecord:
    """A durable record together with the store's digest for it.

    Args:
        record: The persisted record.
        record_digest: Store-assigned ``sha256:`` digest of the durable entry.
    """

    record: RequiredAuditRecord
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.record) is not RequiredAuditRecord:
            raise RequiredAuditError("invalid_record")
        _require_digest(self.record_digest, "invalid_record_digest")


@runtime_checkable
class RequiredAuditStore(Protocol):
    """Durable required audit store port.

    Implementations must make ``append`` durable before returning, enforce the
    stage lifecycle, reject a second terminal record, and raise
    :class:`RequiredAuditStoreError` for every failure.
    """

    def append(self, record: RequiredAuditRecord) -> str:
        """Persist one record and return its durable digest."""
        ...

    def history(self, receipt_id: str) -> tuple[StoredAuditRecord, ...]:
        """Return durable records for one receipt in append order."""
        ...

    def open_receipt_ids(self) -> tuple[str, ...]:
        """Return receipts whose required evidence has no terminal record."""
        ...


class RequiredAuditEffect(Protocol):
    """Effect adapter invoked at most once per receipt by the coordinator."""

    def __call__(self, effect_key: str) -> None:
        """Apply the effect idempotently under the opaque effect key."""
        ...


class EffectObserver(Protocol):
    """Read-only effect sink query used during reconciliation."""

    def observe(self, operation: str, effect_key: str) -> ObservationState:
        """Report whether the effect for ``effect_key`` is committed."""
        ...


@dataclass(frozen=True, slots=True)
class AuditReceipt:
    """Value-free receipt for one declared operation request.

    Args:
        receipt_id: Opaque receipt identifier.
        operation: Declared operation code.
        boundary: Commit boundary declared for the operation, if declared.
        outcome: Caller-visible outcome.
        evidence: Durability of required evidence.
        reason: Closed reason code.
        binding_digest: Digest binding the operation to its payload digest.
        effect_key: Opaque idempotency key handed to the effect adapter.
        recorded_stages: Stages known to be durable and acknowledged.
        evidence_digest: Required-store digest of the terminal record, when
            the evidence is complete.
    """

    receipt_id: str
    operation: str
    boundary: CommitBoundary | None
    outcome: RequiredAuditOutcome
    evidence: EvidenceState
    reason: RequiredAuditReason
    binding_digest: str
    effect_key: str
    recorded_stages: tuple[AuditStage, ...] = ()
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        _require_receipt_id(self.receipt_id)
        _require_operation(self.operation)
        if self.boundary is not None and type(self.boundary) is not CommitBoundary:
            raise RequiredAuditError("invalid_commit_boundary")
        if (
            type(self.outcome) is not RequiredAuditOutcome
            or type(self.evidence) is not EvidenceState
            or type(self.reason) is not RequiredAuditReason
        ):
            raise RequiredAuditError("invalid_receipt")
        _require_digest(self.binding_digest, "invalid_binding_digest")
        if type(self.effect_key) is not str or not _EFFECT_KEY_RE.fullmatch(
            self.effect_key
        ):
            raise RequiredAuditError("invalid_effect_key")
        if type(self.recorded_stages) is not tuple or any(
            type(stage) is not AuditStage for stage in self.recorded_stages
        ):
            raise RequiredAuditError("invalid_recorded_stages")
        if self.evidence_digest is not None:
            _require_digest(self.evidence_digest, "invalid_evidence_digest")

    @property
    def fully_evidenced(self) -> bool:
        """Return whether this is a success backed by durable required audit."""
        return (
            self.outcome is RequiredAuditOutcome.COMPLETED
            and self.evidence is EvidenceState.COMPLETE
            and self.evidence_digest is not None
            and self.recorded_stages[-1:] == (AuditStage.COMMIT,)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical value-free receipt fields."""
        return {
            "schema_version": REQUIRED_AUDIT_SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "operation": self.operation,
            "boundary": None if self.boundary is None else self.boundary.value,
            "outcome": self.outcome.value,
            "evidence": self.evidence.value,
            "reason": self.reason.value,
            "binding_digest": self.binding_digest,
            "effect_key": self.effect_key,
            "recorded_stages": [stage.value for stage in self.recorded_stages],
            "evidence_digest": self.evidence_digest,
            "fully_evidenced": self.fully_evidenced,
        }


_LEDGER_STATES: Final = {
    AuditStage.INTENT: ActionState.PROPOSED,
    AuditStage.ATTEMPT: ActionState.ATTEMPTED,
    AuditStage.COMMIT: ActionState.COMMITTED,
    AuditStage.ABORT: ActionState.REJECTED,
}
_LEDGER_STAGES: Final = {state: stage for stage, state in _LEDGER_STATES.items()}
_LEDGER_REJECTIONS: Final = frozenset(
    {
        "append_conflict",
        "invalid_transition",
        "missing_proposal",
        "action_metadata_changed",
    }
)


def _operation_digest(operation: str) -> str:
    return "sha256:" + _digest(
        {"schema": REQUIRED_AUDIT_SCHEMA_VERSION, "operation": operation}
    )


class ActionLedgerAuditStore:
    """Required audit store backed by the existing hash-chained action ledger.

    Each receipt maps to one ledger action: intent, attempt, commit and abort
    map to ``proposed``, ``attempted``, ``committed`` and ``rejected``. The
    ledger enforces the transition lifecycle and rejects a second terminal
    record. The operation code is bound as the grant digest and the payload
    binding as the tool digest, so no plaintext enters the ledger.

    Args:
        ledger: Durable action ledger in a dedicated private directory.
        run_id: Opaque run identifier shared by all entries in the ledger.
        operations: Declared operation codes whose records may be read back.
        actor_role: Governed role code recorded on every entry.

    Raises:
        RequiredAuditError: If any argument is malformed.
    """

    def __init__(
        self,
        ledger: ActionLedger,
        *,
        run_id: RunId,
        operations: Iterable[str],
        actor_role: str = DEFAULT_ACTOR_ROLE,
    ) -> None:
        if not isinstance(ledger, ActionLedger):
            raise RequiredAuditError("invalid_ledger")
        if type(run_id) is not RunId:
            raise RequiredAuditError("invalid_run_id")
        if type(actor_role) is not str or _ROLE_RE.fullmatch(actor_role) is None:
            raise RequiredAuditError("invalid_actor_role")
        names = tuple(operations)
        if not names:
            raise RequiredAuditError("empty_policy")
        self._ledger = ledger
        self._run_id = run_id
        self._actor_role = actor_role
        self._operations = {
            _operation_digest(_require_operation(name)): name for name in names
        }

    def append(self, record: RequiredAuditRecord) -> str:
        """Persist one record as the next ledger transition.

        Args:
            record: Required audit record to persist.

        Returns:
            The ledger entry digest.

        Raises:
            RequiredAuditStoreError: ``audit_append_rejected`` when the ledger
                refuses the transition, otherwise ``audit_unavailable``.
        """
        if type(record) is not RequiredAuditRecord:
            raise RequiredAuditStoreError("audit_append_rejected")
        try:
            entry = self._ledger.record(
                run_id=self._run_id,
                action_id=ActionId("act_" + record.receipt_id[5:]),
                state=_LEDGER_STATES[record.stage],
                actor_role=self._actor_role,
                grant_digest=_operation_digest(record.operation),
                tool_digest=record.binding_digest,
            )
        except ActionLedgerError as error:
            code = error.code
            if code in _LEDGER_REJECTIONS:
                raise RequiredAuditStoreError("audit_append_rejected") from None
            raise RequiredAuditStoreError("audit_unavailable") from None
        except Exception:
            raise RequiredAuditStoreError("audit_unavailable") from None
        return entry.entry_digest

    def _grouped(self) -> dict[str, list[StoredAuditRecord]]:
        try:
            entries = self._ledger.load()
        except Exception:
            raise RequiredAuditStoreError("audit_read_failed") from None
        grouped: dict[str, list[StoredAuditRecord]] = {}
        for entry in entries:
            operation = self._operations.get(entry.grant_digest)
            if entry.run_id != self._run_id or operation is None:
                raise RequiredAuditStoreError("audit_read_failed")
            receipt_id = "rcpt_" + entry.action_id.serialize()[4:]
            stored = StoredAuditRecord(
                RequiredAuditRecord(
                    receipt_id,
                    operation,
                    _LEDGER_STAGES[entry.state],
                    entry.tool_digest,
                ),
                entry.entry_digest,
            )
            grouped.setdefault(receipt_id, []).append(stored)
        return grouped

    def history(self, receipt_id: str) -> tuple[StoredAuditRecord, ...]:
        """Return verified ledger records for one receipt.

        Args:
            receipt_id: Opaque receipt identifier.

        Returns:
            Durable records in append order.

        Raises:
            RequiredAuditStoreError: ``audit_read_failed`` if the ledger cannot
                be read or verified.
        """
        _require_receipt_id(receipt_id)
        return tuple(self._grouped().get(receipt_id, ()))

    def open_receipt_ids(self) -> tuple[str, ...]:
        """Return receipts without a terminal record, in first-append order.

        Returns:
            Opaque receipt identifiers.

        Raises:
            RequiredAuditStoreError: ``audit_read_failed`` if the ledger cannot
                be read or verified.
        """
        return tuple(
            receipt_id
            for receipt_id, records in self._grouped().items()
            if records[-1].record.stage not in _TERMINAL
        )


class RequiredAuditCoordinator:
    """Coordinate admission, commit and reconciliation for declared operations.

    Args:
        policy: Explicit required-audit policy.
        store: Durable required audit store. ``None`` denies every declared
            operation, even when diagnostics are configured.
        diagnostics: Optional best-effort sink for value-free outcome events.
            Its failures are ignored and its records are never evidence.

    Raises:
        RequiredAuditError: If the policy or store is invalid.
    """

    def __init__(
        self,
        policy: RequiredAuditPolicy,
        store: RequiredAuditStore | None,
        *,
        diagnostics: AuditSink | None = None,
    ) -> None:
        if type(policy) is not RequiredAuditPolicy:
            raise RequiredAuditError("invalid_policy")
        if store is not None and (
            not isinstance(store, RequiredAuditStore) or store is diagnostics
        ):
            raise RequiredAuditError("invalid_required_store")
        self._policy = policy
        self._store = store
        self._diagnostics = diagnostics
        self._lock = threading.Lock()
        self._in_flight: set[str] = set()

    @property
    def policy(self) -> RequiredAuditPolicy:
        """Return the active required-audit policy."""
        return self._policy

    def perform(
        self,
        operation: str,
        *,
        request_key: str,
        operation_digest: str,
        effect: RequiredAuditEffect,
    ) -> AuditReceipt:
        """Run one declared operation under required audit.

        The effect is invoked at most once per receipt and only after the
        intent and attempt records are durable. A request key that already
        has a receipt never invokes the effect again.

        Args:
            operation: Declared operation code.
            request_key: Caller-chosen opaque idempotency token.
            operation_digest: ``sha256:`` digest of the value-free operation
                payload, binding the request key to one payload.
            effect: Adapter that applies the effect under the effect key.

        Returns:
            A receipt whose outcome is completed only with complete evidence.

        Raises:
            RequiredAuditError: If any argument is malformed.
        """
        receipt_id = derive_receipt_id(operation, request_key)
        _require_digest(operation_digest, "invalid_operation_digest")
        if not callable(effect):
            raise RequiredAuditError("invalid_effect")
        boundary = self._policy.boundary_for(operation)
        base = AuditReceipt(
            receipt_id=receipt_id,
            operation=operation,
            boundary=boundary,
            outcome=RequiredAuditOutcome.DENIED,
            evidence=EvidenceState.NONE,
            reason=RequiredAuditReason.OPERATION_UNDECLARED,
            binding_digest=_binding_digest(operation, operation_digest),
            effect_key=_effect_key(receipt_id),
        )
        if boundary is None:
            return self._emit(base)
        if self._store is None:
            return self._emit(
                replace(base, reason=RequiredAuditReason.REQUIRED_AUDIT_UNCONFIGURED)
            )
        with self._claim(receipt_id) as claimed:
            if not claimed:
                return self._emit(
                    replace(
                        base,
                        outcome=RequiredAuditOutcome.RECONCILE_REQUIRED,
                        evidence=EvidenceState.PENDING,
                        reason=RequiredAuditReason.RECEIPT_IN_FLIGHT,
                    )
                )
            return self._emit(self._perform(base, effect))

    def _perform(self, base: AuditReceipt, effect: RequiredAuditEffect) -> AuditReceipt:
        history = self._read(base.receipt_id)
        open_count = 0
        if history is not None and not history:
            try:
                open_count = len(self._required_store().open_receipt_ids())
            except Exception:
                history = None
        if history is None:
            return replace(
                base,
                evidence=EvidenceState.PENDING,
                reason=RequiredAuditReason.AUDIT_READ_FAILED,
            )
        if history:
            if any(r.record.binding_digest != base.binding_digest for r in history):
                return replace(base, reason=RequiredAuditReason.IDEMPOTENCY_CONFLICT)
            rebuilt = self._from_history(base, history)
            if rebuilt is None:
                return self._mismatch(base)
            reason = (
                RequiredAuditReason.ALREADY_RECORDED
                if rebuilt.evidence is EvidenceState.COMPLETE
                else RequiredAuditReason.RECONCILE_PENDING
            )
            return replace(rebuilt, reason=reason)
        if open_count >= self._policy.max_open_receipts:
            return replace(base, reason=RequiredAuditReason.RECONCILE_BACKLOG_FULL)

        stages: list[AuditStage] = []
        _, error = self._append(base, AuditStage.INTENT, stages)
        if error is not None:
            return replace(
                base,
                evidence=(
                    EvidenceState.NONE if error.not_written else EvidenceState.PENDING
                ),
                reason=RequiredAuditReason.INTENT_NOT_DURABLE,
            )
        if self._append(base, AuditStage.ATTEMPT, stages)[1] is not None:
            return replace(
                base,
                evidence=EvidenceState.PENDING,
                reason=RequiredAuditReason.ATTEMPT_NOT_DURABLE,
                recorded_stages=tuple(stages),
            )

        # Commit boundary: from here on the effect may have committed.
        try:
            effect(base.effect_key)
        except Exception:
            if base.boundary is CommitBoundary.ACKNOWLEDGED:
                return self._close_denied(
                    base, stages, RequiredAuditReason.EFFECT_NOT_COMMITTED
                )
            return replace(
                base,
                outcome=RequiredAuditOutcome.RECONCILE_REQUIRED,
                evidence=EvidenceState.PENDING,
                reason=RequiredAuditReason.EFFECT_OUTCOME_UNKNOWN,
                recorded_stages=tuple(stages),
            )
        digest, _ = self._append(base, AuditStage.COMMIT, stages)
        if digest is None:
            return replace(
                base,
                outcome=RequiredAuditOutcome.RECONCILE_REQUIRED,
                evidence=EvidenceState.PENDING,
                reason=RequiredAuditReason.COMMIT_NOT_DURABLE,
                recorded_stages=tuple(stages),
            )
        return replace(
            base,
            outcome=RequiredAuditOutcome.COMPLETED,
            evidence=EvidenceState.COMPLETE,
            reason=RequiredAuditReason.COMMITTED,
            recorded_stages=tuple(stages),
            evidence_digest=digest,
        )

    def reconcile(
        self, receipt: AuditReceipt, *, observer: EffectObserver
    ) -> AuditReceipt:
        """Close a receipt's required evidence without re-invoking its effect.

        A terminal record already in the required store is returned as is, so
        repeated reconciliation completes the same receipt exactly once. An
        attempted effect is resolved only through the read-only observer;
        ambiguous or unavailable observations stay reconcile-required.

        Args:
            receipt: Receipt returned by :meth:`perform` or
                :meth:`open_receipts`.
            observer: Read-only effect sink query.

        Returns:
            The updated receipt for the same receipt identifier.

        Raises:
            RequiredAuditError: If ``receipt`` is not an :class:`AuditReceipt`.
        """
        if type(receipt) is not AuditReceipt:
            raise RequiredAuditError("invalid_receipt")
        if self._store is None:
            return self._emit(
                replace(receipt, reason=RequiredAuditReason.REQUIRED_AUDIT_UNCONFIGURED)
            )
        with self._claim(receipt.receipt_id) as claimed:
            if not claimed:
                return self._emit(
                    replace(receipt, reason=RequiredAuditReason.RECEIPT_IN_FLIGHT)
                )
            return self._emit(self._reconcile(receipt, observer))

    def _reconcile(
        self, receipt: AuditReceipt, observer: EffectObserver
    ) -> AuditReceipt:
        history = self._read(receipt.receipt_id)
        if history is None:
            return replace(receipt, reason=RequiredAuditReason.AUDIT_READ_FAILED)
        if not history:
            return replace(
                receipt,
                outcome=RequiredAuditOutcome.DENIED,
                evidence=EvidenceState.NONE,
                reason=RequiredAuditReason.NO_DURABLE_INTENT,
                recorded_stages=(),
                evidence_digest=None,
            )
        rebuilt = self._from_history(receipt, history)
        if (
            rebuilt is None
            or receipt.effect_key != _effect_key(receipt.receipt_id)
            or any(
                r.record.binding_digest != receipt.binding_digest
                or r.record.operation != receipt.operation
                for r in history
            )
        ):
            return self._mismatch(receipt)
        stages = list(rebuilt.recorded_stages)
        if stages[-1] in _TERMINAL:
            return replace(rebuilt, reason=RequiredAuditReason.RECONCILED_EXISTING)
        if stages[-1] is AuditStage.INTENT:
            return self._close_denied(
                rebuilt, stages, RequiredAuditReason.RECONCILED_NOT_ATTEMPTED
            )
        try:
            observation = observer.observe(receipt.operation, receipt.effect_key)
        except Exception:
            observation = None
        if type(observation) is not ObservationState:
            return replace(rebuilt, reason=RequiredAuditReason.OBSERVER_UNAVAILABLE)
        if observation is ObservationState.AMBIGUOUS:
            return replace(rebuilt, reason=RequiredAuditReason.EFFECT_AMBIGUOUS)
        if observation is ObservationState.ABSENT:
            return self._close_denied(
                rebuilt, stages, RequiredAuditReason.RECONCILED_ABSENT
            )
        digest, _ = self._append(rebuilt, AuditStage.COMMIT, stages)
        if digest is None:
            return self._after_failed_close(
                rebuilt, RequiredAuditReason.COMMIT_NOT_DURABLE
            )
        return replace(
            rebuilt,
            outcome=RequiredAuditOutcome.COMPLETED,
            evidence=EvidenceState.COMPLETE,
            reason=RequiredAuditReason.RECONCILED_COMMIT,
            recorded_stages=tuple(stages),
            evidence_digest=digest,
        )

    def open_receipts(self) -> tuple[AuditReceipt, ...]:
        """Rebuild receipts with open required evidence from the store.

        Returns:
            Receipts reconstructed only from the required store, for example
            after a restart. Returns an empty tuple when no store is set.

        Raises:
            RequiredAuditStoreError: If the store cannot be read.
        """
        if self._store is None:
            return ()
        receipts = []
        for receipt_id in self._store.open_receipt_ids():
            history = self._read(receipt_id)
            if history is None:
                raise RequiredAuditStoreError("audit_read_failed")
            if not history:
                continue
            first = history[0].record
            base = AuditReceipt(
                receipt_id=receipt_id,
                operation=first.operation,
                boundary=self._policy.boundary_for(first.operation),
                outcome=RequiredAuditOutcome.RECONCILE_REQUIRED,
                evidence=EvidenceState.PENDING,
                reason=RequiredAuditReason.RECONCILE_PENDING,
                binding_digest=first.binding_digest,
                effect_key=_effect_key(receipt_id),
            )
            rebuilt = self._from_history(base, history)
            receipts.append(rebuilt if rebuilt is not None else base)
        return tuple(receipts)

    def reconcile_open(self, *, observer: EffectObserver) -> tuple[AuditReceipt, ...]:
        """Reconcile every receipt whose required evidence is still open.

        Args:
            observer: Read-only effect sink query.

        Returns:
            Reconciled receipts in store order.

        Raises:
            RequiredAuditStoreError: If the store cannot be read.
        """
        return tuple(
            self.reconcile(receipt, observer=observer)
            for receipt in self.open_receipts()
        )

    # Internal helpers -------------------------------------------------------

    @contextmanager
    def _claim(self, receipt_id: str) -> Iterator[bool]:
        with self._lock:
            if receipt_id in self._in_flight:
                claimed = False
            else:
                self._in_flight.add(receipt_id)
                claimed = True
        try:
            yield claimed
        finally:
            if claimed:
                with self._lock:
                    self._in_flight.discard(receipt_id)

    def _read(self, receipt_id: str) -> tuple[StoredAuditRecord, ...] | None:
        try:
            history = tuple(self._required_store().history(receipt_id))
        except Exception:
            return None
        if any(
            type(r) is not StoredAuditRecord or r.record.receipt_id != receipt_id
            for r in history
        ):
            return None
        return history

    def _required_store(self) -> RequiredAuditStore:
        if self._store is None:
            raise RequiredAuditError("required_audit_unconfigured")
        return self._store

    def _append(
        self, receipt: AuditReceipt, stage: AuditStage, stages: list[AuditStage]
    ) -> tuple[str | None, RequiredAuditStoreError | None]:
        """Append one stage; return its digest or a closed store error."""
        record = RequiredAuditRecord(
            receipt.receipt_id, receipt.operation, stage, receipt.binding_digest
        )
        try:
            digest = self._required_store().append(record)
        except RequiredAuditStoreError as error:
            return None, error
        except Exception:
            return None, RequiredAuditStoreError("audit_unavailable")
        if type(digest) is not str or _DIGEST_RE.fullmatch(digest) is None:
            return None, RequiredAuditStoreError("audit_unavailable")
        stages.append(stage)
        return digest, None

    def _close_denied(
        self,
        receipt: AuditReceipt,
        stages: list[AuditStage],
        reason: RequiredAuditReason,
    ) -> AuditReceipt:
        digest, _ = self._append(receipt, AuditStage.ABORT, stages)
        if digest is None:
            return self._after_failed_close(
                replace(
                    receipt,
                    outcome=RequiredAuditOutcome.DENIED,
                    recorded_stages=tuple(stages),
                ),
                RequiredAuditReason.ABORT_NOT_DURABLE,
            )
        return replace(
            receipt,
            outcome=RequiredAuditOutcome.DENIED,
            evidence=EvidenceState.COMPLETE,
            reason=reason,
            recorded_stages=tuple(stages),
            evidence_digest=digest,
        )

    def _after_failed_close(
        self, receipt: AuditReceipt, reason: RequiredAuditReason
    ) -> AuditReceipt:
        """Re-read after a failed closing append; a terminal record wins."""
        history = self._read(receipt.receipt_id)
        if history:
            rebuilt = self._from_history(receipt, history)
            if (
                rebuilt is not None
                and rebuilt.recorded_stages
                and rebuilt.recorded_stages[-1] in _TERMINAL
            ):
                return replace(rebuilt, reason=RequiredAuditReason.RECONCILED_EXISTING)
        return replace(receipt, evidence=EvidenceState.PENDING, reason=reason)

    def _from_history(
        self, base: AuditReceipt, history: tuple[StoredAuditRecord, ...]
    ) -> AuditReceipt | None:
        stages = tuple(r.record.stage for r in history)
        if stages not in _VALID_SEQUENCES or any(
            r.record.operation != history[0].record.operation
            or r.record.binding_digest != history[0].record.binding_digest
            for r in history
        ):
            return None
        last = stages[-1]
        if last is AuditStage.COMMIT:
            outcome, evidence = RequiredAuditOutcome.COMPLETED, EvidenceState.COMPLETE
        elif last is AuditStage.ABORT:
            outcome, evidence = RequiredAuditOutcome.DENIED, EvidenceState.COMPLETE
        elif last is AuditStage.INTENT:
            outcome, evidence = RequiredAuditOutcome.DENIED, EvidenceState.PENDING
        else:
            outcome = RequiredAuditOutcome.RECONCILE_REQUIRED
            evidence = EvidenceState.PENDING
        return replace(
            base,
            outcome=outcome,
            evidence=evidence,
            recorded_stages=stages,
            evidence_digest=(history[-1].record_digest if last in _TERMINAL else None),
        )

    @staticmethod
    def _mismatch(receipt: AuditReceipt) -> AuditReceipt:
        return replace(
            receipt,
            outcome=RequiredAuditOutcome.RECONCILE_REQUIRED,
            evidence=EvidenceState.PENDING,
            reason=RequiredAuditReason.EVIDENCE_MISMATCH,
            evidence_digest=None,
        )

    def _emit(self, receipt: AuditReceipt) -> AuditReceipt:
        if self._diagnostics is not None:
            try:
                self._diagnostics.append(
                    DIAGNOSTIC_EVENT_TYPE,
                    {
                        "receipt_id": receipt.receipt_id,
                        "operation": receipt.operation,
                        "outcome": receipt.outcome.value,
                        "evidence": receipt.evidence.value,
                        "reason": receipt.reason.value,
                    },
                )
            except Exception:
                pass  # Diagnostics are optional and never change outcomes.
        return receipt
