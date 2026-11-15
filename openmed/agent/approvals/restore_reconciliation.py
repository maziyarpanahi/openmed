"""Anti-replay authorization reconciliation for restored local service state.

Backup integrity and schema compatibility do not establish that authority
recorded in an older snapshot is still usable. This module binds a restored
generation to a new recovery epoch, burns every pre-restore approval nonce in
the live nonce store, rechecks recorded authority against injected current
status, and reconciles resumable effects by idempotency key. It never deletes
committed effects, rewrites checkpoints, contacts a network service, or claims
that a human review was repeated. Reports carry only controlled codes, counts,
opaque effect identifiers and digests.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NoReturn, Protocol, cast

from openmed.agent.permissions.revocation import (
    AuthorityBinding,
    AuthorityBoundary,
    AuthorityDeniedError,
    AuthorityReason,
    AuthorityRuntime,
)
from openmed.agent.workflows.recovery import (
    EffectObservation,
    EffectState,
    RecoveryCheckpoint,
    RecoveryDecision,
    RecoveryDisposition,
    RecoveryError,
    RecoveryPhase,
    RecoveryReason,
    RetiredJournalRecord,
    recover_workflow,
    validate_checkpoint_lineage,
)

from .tokens import (
    MAX_APPROVAL_CLOCK_SKEW_SECONDS,
    ApprovalNonceStore,
    ApprovalToken,
)

RESTORE_RECONCILIATION_SCHEMA_VERSION: Final = "openmed.agent.restore_reconciliation.v1"
RECOVERY_EPOCH_SCHEMA_VERSION: Final = "openmed.agent.recovery_epoch.v1"
EPOCH_ACTION_SCHEMA_VERSION: Final = "openmed.agent.recovery_epoch_action.v1"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_INT: Final = 2**63 - 1


class RestoreReconciliationError(ValueError):
    """Fail-closed restore reconciliation error with a fixed code.

    Args:
        code: Stable machine-readable failure code.
        field_name: Optional fixed public field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


class ApprovalRestoreDisposition(str, Enum):
    """Post-restore state of one approval recorded in a snapshot."""

    INVALIDATED = "invalidated"
    HELD = "held"


class WorkflowRestoreDisposition(str, Enum):
    """Post-restore admission state of one restored workflow lineage."""

    RESUME = "resume"
    COMPLETE = "complete"
    RETIRED = "retired"
    FRESH_REVIEW_REQUIRED = "fresh_review_required"
    REVIEW_REQUIRED = "review_required"
    HELD = "held"


class RestoreReason(str, Enum):
    """Closed reason codes safe to retain in restore audit evidence."""

    NONCE_BURNED = "pre_restore_nonce_burned"
    NONCE_ALREADY_CLAIMED = "pre_restore_nonce_already_claimed"
    APPROVAL_EXPIRED = "approval_expired"
    NONCE_JOURNAL_UNAVAILABLE = "nonce_journal_unavailable"
    RESUME_SAFE = "restore_resume_safe"
    EFFECTS_COMPLETE = "restore_effects_complete"
    JOURNAL_RETIRED = "restore_journal_retired"
    APPROVAL_INVALIDATED = "pre_restore_approval_invalidated"
    APPROVAL_MISSING = "approval_missing"
    AUTHORITY_REVOKED = "authority_revoked"
    AUTHORITY_EXPIRED = "authority_expired"
    AUTHORITY_CHANGED = "authority_generation_changed"
    AUTHORITY_ROLLBACK = "authority_status_rollback"
    AUTHORITY_MISMATCH = "authority_contract_mismatch"
    AUTHORITY_UNAVAILABLE = "authority_status_unavailable"
    EFFECT_STATUS_UNAVAILABLE = "effect_status_unavailable"
    EFFECTS_NEED_REVIEW = "restored_effects_need_review"


_AUTHORITY_OUTCOMES: Final = {
    AuthorityReason.REVOKED: RestoreReason.AUTHORITY_REVOKED,
    AuthorityReason.EXPIRED: RestoreReason.AUTHORITY_EXPIRED,
    AuthorityReason.GENERATION_CHANGED: RestoreReason.AUTHORITY_CHANGED,
    AuthorityReason.ROLLBACK: RestoreReason.AUTHORITY_ROLLBACK,
    AuthorityReason.CONTRACT_MISMATCH: RestoreReason.AUTHORITY_MISMATCH,
    AuthorityReason.APPROVAL_MISMATCH: RestoreReason.AUTHORITY_MISMATCH,
}


class RecoveryEpochStore(Protocol):
    """Monotonic recovery-epoch high water retained outside every backup."""

    def current(self) -> int:
        """Return the latest opened epoch, or zero before any restore."""

    def advance(self, *, floor: int) -> int:
        """Atomically open and retain an epoch above ``floor`` and high water."""


class InMemoryRecoveryEpochStore:
    """Thread-safe process-local epoch store for single-process use and tests.

    Production restores need an injected durable store that is never restored
    from the same backup as run state; otherwise epochs could be rewound.
    """

    def __init__(self) -> None:
        self._current = 0
        self._lock = threading.Lock()

    def current(self) -> int:
        """Return the latest opened epoch."""
        with self._lock:
            return self._current

    def advance(self, *, floor: int) -> int:
        """Open an epoch strictly above both ``floor`` and retained high water."""
        _require_int(floor, "floor")
        with self._lock:
            self._current = max(self._current, floor) + 1
            return self._current

    def __repr__(self) -> str:
        """Return a representation with the epoch number only."""
        return f"InMemoryRecoveryEpochStore(current={self.current()})"


@dataclass(frozen=True, slots=True, repr=False)
class RecoveryEpoch:
    """Digest-bound record of the epoch opened for one restored generation.

    Args:
        epoch: Newly opened epoch, strictly above ``previous_epoch``.
        previous_epoch: Highest epoch known before this restore.
        snapshot_digest: Opaque digest of the restored backup generation.
        opened_at: Trusted integer Unix time of reconciliation.
        epoch_digest: Commitment to every other field.
        schema_version: Stable epoch schema version.
    """

    epoch: int
    previous_epoch: int
    snapshot_digest: str
    opened_at: int
    epoch_digest: str
    schema_version: str = RECOVERY_EPOCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_EPOCH_SCHEMA_VERSION:
            raise RestoreReconciliationError(
                "unsupported_schema_version", "schema_version"
            )
        _require_int(self.epoch, "epoch")
        _require_int(self.previous_epoch, "previous_epoch")
        if self.epoch <= self.previous_epoch:
            raise RestoreReconciliationError("epoch_not_monotonic", "epoch")
        _require_digest(self.snapshot_digest, "snapshot_digest")
        _require_int(self.opened_at, "opened_at")
        _require_digest(self.epoch_digest, "epoch_digest")
        if self.epoch_digest != _digest(self._unsigned_dict()):
            raise RestoreReconciliationError("epoch_digest_mismatch", "epoch_digest")

    @classmethod
    def create(
        cls,
        *,
        epoch: int,
        previous_epoch: int,
        snapshot_digest: str,
        opened_at: int,
    ) -> RecoveryEpoch:
        """Create an epoch record and bind every field into its digest."""
        values = {
            "schema_version": RECOVERY_EPOCH_SCHEMA_VERSION,
            "epoch": epoch,
            "previous_epoch": previous_epoch,
            "snapshot_digest": snapshot_digest,
            "opened_at": opened_at,
        }
        try:
            digest = _digest(values)
        except (TypeError, ValueError):
            raise RestoreReconciliationError("invalid_epoch", "epoch") from None
        return cls(epoch_digest=digest, **values)  # type: ignore[arg-type]

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "previous_epoch": self.previous_epoch,
            "snapshot_digest": self.snapshot_digest,
            "opened_at": self.opened_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the epoch, its predecessor, timestamps and digests only."""
        return {**self._unsigned_dict(), "epoch_digest": self.epoch_digest}

    def __repr__(self) -> str:
        """Return the epoch number without linkable digests."""
        return f"RecoveryEpoch(epoch={self.epoch})"


def epoch_action_digest(action_digest: str, epoch: RecoveryEpoch) -> str:
    """Bind a reviewed action digest to one recovery epoch.

    Issue and verify post-restore approval tokens against this digest, so a
    token signed before the restore, including one absent from the snapshot,
    fails as an action mismatch and is consumed rather than authorizing work.

    Args:
        action_digest: Canonical SHA-256 digest of the reviewed action.
        epoch: Epoch returned by the restore reconciliation report.

    Returns:
        A SHA-256 digest committing to the action and the complete epoch.

    Raises:
        RestoreReconciliationError: An input is malformed.
    """
    _require_digest(action_digest, "action_digest")
    if type(epoch) is not RecoveryEpoch:
        raise RestoreReconciliationError("invalid_epoch", "epoch")
    return _digest(
        {
            "schema_version": EPOCH_ACTION_SCHEMA_VERSION,
            "action_digest": action_digest,
            "epoch": epoch.epoch,
            "epoch_digest": epoch.epoch_digest,
        }
    )


def require_current_epoch(epoch: RecoveryEpoch, store: RecoveryEpochStore) -> None:
    """Refuse admission under an epoch superseded by a later restore.

    Args:
        epoch: Epoch from the reconciliation report the caller is acting on.
        store: The same independently retained epoch store.

    Raises:
        RestoreReconciliationError: The store is unavailable or the epoch is
            no longer current.
    """
    if type(epoch) is not RecoveryEpoch:
        raise RestoreReconciliationError("invalid_epoch", "epoch")
    if epoch.epoch != _read_current(store):
        raise RestoreReconciliationError("stale_recovery_epoch", "epoch")


@dataclass(frozen=True, slots=True, repr=False)
class RestoredApproval:
    """Digest-only record of an approval outstanding in a restored snapshot.

    Args:
        token_digest: SHA-256 digest of the complete signed token JSON, as in
            ``ApprovalReceipt.token_digest``.
        nonce_digest: SHA-256 digest of the token nonce, as claimed by
            ``ApprovalTokenVerifier``.
        expires_at: Exclusive verifier retention bound, including skew.
    """

    token_digest: str
    nonce_digest: str
    expires_at: int

    def __post_init__(self) -> None:
        _require_digest(self.token_digest, "token_digest")
        _require_digest(self.nonce_digest, "nonce_digest")
        _require_int(self.expires_at, "expires_at")

    @classmethod
    def from_token(
        cls, token: ApprovalToken, *, clock_skew_seconds: int = 0
    ) -> RestoredApproval:
        """Derive digests without retaining the bearer token.

        Args:
            token: Signed token recorded as outstanding in the snapshot.
            clock_skew_seconds: Skew configured on the live verifier, so the
                burn covers its full acceptance window.

        Returns:
            A digest-only restored approval record.

        Raises:
            RestoreReconciliationError: An input is malformed.
        """
        if type(token) is not ApprovalToken:
            raise RestoreReconciliationError("invalid_token", "token")
        if (
            type(clock_skew_seconds) is not int
            or not 0 <= clock_skew_seconds <= MAX_APPROVAL_CLOCK_SKEW_SECONDS
        ):
            raise RestoreReconciliationError("invalid_skew", "clock_skew_seconds")
        return cls(
            token_digest=_sha256(token.to_json().encode("utf-8")),
            nonce_digest=_sha256(token.nonce.encode("ascii")),
            expires_at=token.expires_at + clock_skew_seconds,
        )

    def __repr__(self) -> str:
        """Hide linkable token and nonce digests."""
        return "RestoredApproval(<digest-only>)"


@dataclass(frozen=True, slots=True, repr=False)
class RestoredWorkflow:
    """One restored recovery lineage and the authority binding it recorded.

    Args:
        lineage: Restored checkpoint lineage, or a sealed retired record.
        authority: Preview authority binding recorded with the run, or
            ``None`` when the run uses no runtime-revocable authority.
    """

    lineage: tuple[RecoveryCheckpoint, ...] | RetiredJournalRecord
    authority: AuthorityBinding | None = None

    def __post_init__(self) -> None:
        if type(self.lineage) is not RetiredJournalRecord:
            if type(self.lineage) is not tuple:
                raise RestoreReconciliationError("invalid_restored_lineage", "lineage")
            try:
                validate_checkpoint_lineage(self.lineage)
            except RecoveryError:
                raise RestoreReconciliationError(
                    "invalid_restored_lineage", "lineage"
                ) from None
        if self.authority is not None and type(self.authority) is not AuthorityBinding:
            raise RestoreReconciliationError("invalid_restored_authority", "authority")

    @property
    def source_digest(self) -> str:
        """Return the digest of the restored tip checkpoint or retired record."""
        if type(self.lineage) is RetiredJournalRecord:
            return self.lineage.final_checkpoint_digest
        return self.lineage[-1].checkpoint_digest

    def __repr__(self) -> str:
        """Return counts only."""
        count = (
            self.lineage.checkpoint_count
            if type(self.lineage) is RetiredJournalRecord
            else len(self.lineage)
        )
        return f"RestoredWorkflow(checkpoints={count})"


@dataclass(frozen=True, slots=True, repr=False)
class RestoredSnapshot:
    """Content-free fixture view of one restored backup generation.

    Backup format, decryption and integrity checks belong to the backup tool.
    Everything here is treated as historical, never as current authority.

    Args:
        snapshot_digest: Opaque digest identifying the restored generation.
        recorded_epoch: Recovery epoch recorded in the backup.
        approvals: Approvals outstanding when the backup was taken.
        workflows: Restored workflow lineages.
    """

    snapshot_digest: str
    recorded_epoch: int
    approvals: tuple[RestoredApproval, ...]
    workflows: tuple[RestoredWorkflow, ...]

    def __post_init__(self) -> None:
        _require_digest(self.snapshot_digest, "snapshot_digest")
        _require_int(self.recorded_epoch, "recorded_epoch")
        if type(self.approvals) is not tuple or any(
            type(item) is not RestoredApproval for item in self.approvals
        ):
            raise RestoreReconciliationError("invalid_restored_approvals", "approvals")
        if type(self.workflows) is not tuple or any(
            type(item) is not RestoredWorkflow for item in self.workflows
        ):
            raise RestoreReconciliationError("invalid_restored_workflows", "workflows")
        token_digests = [item.token_digest for item in self.approvals]
        nonce_digests = [item.nonce_digest for item in self.approvals]
        if len(set(token_digests)) != len(token_digests) or len(
            set(nonce_digests)
        ) != len(nonce_digests):
            raise RestoreReconciliationError("duplicate_restored_approval", "approvals")
        sources = [item.source_digest for item in self.workflows]
        if len(set(sources)) != len(sources):
            raise RestoreReconciliationError("duplicate_restored_workflow", "workflows")

    def __repr__(self) -> str:
        """Return counts only."""
        return (
            f"RestoredSnapshot(approvals={len(self.approvals)}, "
            f"workflows={len(self.workflows)})"
        )


@dataclass(frozen=True, slots=True)
class ApprovalRestoreOutcome:
    """Reconciled state of one pre-restore approval.

    Args:
        token_digest: Digest of the historical token, for audit linkage.
        disposition: Whether the approval is now unspendable or held.
        reason: Controlled reason code.
    """

    token_digest: str
    disposition: ApprovalRestoreDisposition
    reason: RestoreReason

    def to_dict(self) -> dict[str, str]:
        """Return the token digest and controlled codes only."""
        return {
            "token_digest": self.token_digest,
            "disposition": self.disposition.value,
            "reason_code": self.reason.value,
        }


@dataclass(frozen=True, slots=True)
class WorkflowRestoreOutcome:
    """Reconciled admission state of one restored workflow.

    Args:
        source_digest: Restored tip checkpoint or retired record digest.
        disposition: Admission state under the new epoch.
        reason: Controlled restore reason code.
        recovery_reason: Underlying recovery planner reason code, if planned.
        committed_effects: Opaque action identifiers and commit evidence
            digests known from the snapshot or from live sink observations.
        retry_idempotency_keys: Keys that may be re-dispatched; empty unless
            the disposition is ``resume``.
        recovery_evidence_digest: Digest of the underlying recovery decision.
        historical_approval_receipt_digest: Pre-restore approval receipt kept
            as historical evidence only; it never authorizes a retry.
    """

    source_digest: str
    disposition: WorkflowRestoreDisposition
    reason: RestoreReason
    recovery_reason: str | None = None
    committed_effects: tuple[tuple[str, str], ...] = ()
    retry_idempotency_keys: tuple[str, ...] = ()
    recovery_evidence_digest: str | None = None
    historical_approval_receipt_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return controlled codes, opaque identifiers and digests only."""
        return {
            "source_digest": self.source_digest,
            "disposition": self.disposition.value,
            "reason_code": self.reason.value,
            "recovery_reason_code": self.recovery_reason,
            "committed_effects": [
                {"action_id": action_id, "commit_evidence_digest": evidence}
                for action_id, evidence in self.committed_effects
            ],
            "retry_idempotency_keys": list(self.retry_idempotency_keys),
            "recovery_evidence_digest": self.recovery_evidence_digest,
            "historical_approval_receipt_digest": (
                self.historical_approval_receipt_digest
            ),
        }


@dataclass(frozen=True, slots=True, repr=False)
class RestoreReconciliationReport:
    """Deterministic, content-free evidence of one post-restore reconciliation.

    Args:
        epoch: Recovery epoch opened for the restored generation.
        approvals: One outcome per snapshot approval, in snapshot order.
        workflows: One outcome per restored workflow, in snapshot order.
    """

    epoch: RecoveryEpoch
    approvals: tuple[ApprovalRestoreOutcome, ...]
    workflows: tuple[WorkflowRestoreOutcome, ...]

    @property
    def held_count(self) -> int:
        """Return how many approvals and workflows remain held."""
        return sum(
            item.disposition is ApprovalRestoreDisposition.HELD
            for item in self.approvals
        ) + sum(
            item.disposition is WorkflowRestoreDisposition.HELD
            for item in self.workflows
        )

    @property
    def fresh_review_count(self) -> int:
        """Return how many workflows need fresh authority or human review."""
        return sum(
            item.disposition is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
            for item in self.workflows
        )

    @property
    def complete(self) -> bool:
        """Return whether every snapshot item was reconciled without a hold."""
        return self.held_count == 0

    @property
    def report_digest(self) -> str:
        """Return a commitment to the complete report body."""
        return _digest(self._unsigned_dict())

    def require_complete(self) -> None:
        """Refuse restored admission while any approval or workflow is held.

        Raises:
            RestoreReconciliationError: Reconciliation must be rerun once the
                unavailable nonce journal, authority or sink status returns.
        """
        if not self.complete:
            raise RestoreReconciliationError(
                "restore_reconciliation_incomplete", "report"
            )

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESTORE_RECONCILIATION_SCHEMA_VERSION,
            "epoch": self.epoch.to_dict(),
            "approvals": [item.to_dict() for item in self.approvals],
            "workflows": [item.to_dict() for item in self.workflows],
            "counts": {
                "approvals": len(self.approvals),
                "workflows": len(self.workflows),
                "invalidated_approvals": sum(
                    item.disposition is ApprovalRestoreDisposition.INVALIDATED
                    for item in self.approvals
                ),
                "held": self.held_count,
                "fresh_review_required": self.fresh_review_count,
            },
            "complete": self.complete,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the report body and its digest for audit storage."""
        values = self._unsigned_dict()
        values["report_digest"] = _digest(values)
        return values

    def to_json(self) -> str:
        """Serialize the report as compact canonical JSON."""
        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        """Return the epoch number and counts only."""
        return (
            f"RestoreReconciliationReport(epoch={self.epoch.epoch}, "
            f"approvals={len(self.approvals)}, workflows={len(self.workflows)}, "
            f"held={self.held_count})"
        )


EffectObserver = Callable[[RecoveryCheckpoint], Iterable[EffectObservation]]


class RestoreReconciler:
    """Reconcile a restored generation against authoritative current state.

    Args:
        nonce_store: Live approval nonce store shared with every verifier.
            Pre-restore nonces are claimed here so they can never be spent.
        epochs: Independently retained monotonic recovery-epoch store.
        authority: Runtime with the current status provider and generation
            high water; required for workflows that recorded a binding.
        clock: Trusted integer Unix clock.
    """

    def __init__(
        self,
        nonce_store: ApprovalNonceStore,
        epochs: RecoveryEpochStore,
        *,
        authority: AuthorityRuntime | None = None,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        if not callable(getattr(nonce_store, "claim", None)):
            raise RestoreReconciliationError("invalid_nonce_store", "nonce_store")
        if not callable(getattr(epochs, "advance", None)) or not callable(
            getattr(epochs, "current", None)
        ):
            raise RestoreReconciliationError("invalid_epoch_store", "epochs")
        if authority is not None and type(authority) is not AuthorityRuntime:
            raise RestoreReconciliationError("invalid_authority_runtime", "authority")
        if not callable(clock):
            raise RestoreReconciliationError("invalid_clock", "clock")
        self._nonce_store = nonce_store
        self._epochs = epochs
        self._authority = authority
        self._clock = clock

    def reconcile(
        self, snapshot: RestoredSnapshot, *, observe: EffectObserver
    ) -> RestoreReconciliationReport:
        """Open a new epoch, burn restored approvals and plan restored work.

        Steps run in a fixed order: open the epoch, burn every unexpired
        snapshot nonce in the live store, then for each workflow check current
        authority before reading effect status by idempotency key.

        Args:
            snapshot: Restored generation; its contents are historical only.
            observe: Read-only effect status lookup for one restored tip.

        Returns:
            Content-free report; act only on ``resume`` retry keys, and only
            after ``require_complete`` succeeds.

        Raises:
            RestoreReconciliationError: The snapshot, observer, clock or epoch
                store is invalid or unavailable. Nothing is admitted.
        """
        if type(snapshot) is not RestoredSnapshot:
            raise RestoreReconciliationError("invalid_snapshot", "snapshot")
        if not callable(observe):
            raise RestoreReconciliationError("invalid_effect_observer", "observe")
        now = self._now()
        epoch = self._open_epoch(snapshot, now)
        approvals = tuple(self._burn(item, now) for item in snapshot.approvals)
        workflows = tuple(
            self._workflow(item, observe, now) for item in snapshot.workflows
        )
        return RestoreReconciliationReport(epoch, approvals, workflows)

    def _now(self) -> int:
        try:
            value = self._clock()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise RestoreReconciliationError("clock_unavailable", "clock") from None
        if type(value) is not int or not 0 <= value <= _MAX_INT:
            raise RestoreReconciliationError("clock_unavailable", "clock")
        return value

    def _open_epoch(self, snapshot: RestoredSnapshot, now: int) -> RecoveryEpoch:
        previous = max(_read_current(self._epochs), snapshot.recorded_epoch)
        try:
            opened = self._epochs.advance(floor=previous)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise RestoreReconciliationError(
                "recovery_epoch_unavailable", "epochs"
            ) from None
        if type(opened) is not int or not previous < opened <= _MAX_INT:
            raise RestoreReconciliationError("epoch_not_monotonic", "epochs")
        return RecoveryEpoch.create(
            epoch=opened,
            previous_epoch=previous,
            snapshot_digest=snapshot.snapshot_digest,
            opened_at=now,
        )

    def _burn(self, approval: RestoredApproval, now: int) -> ApprovalRestoreOutcome:
        def outcome(
            disposition: ApprovalRestoreDisposition, reason: RestoreReason
        ) -> ApprovalRestoreOutcome:
            return ApprovalRestoreOutcome(approval.token_digest, disposition, reason)

        if now >= approval.expires_at:
            return outcome(
                ApprovalRestoreDisposition.INVALIDATED, RestoreReason.APPROVAL_EXPIRED
            )
        try:
            claimed = self._nonce_store.claim(
                approval.nonce_digest, expires_at=approval.expires_at, now=now
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            claimed = None
        if type(claimed) is not bool:
            return outcome(
                ApprovalRestoreDisposition.HELD,
                RestoreReason.NONCE_JOURNAL_UNAVAILABLE,
            )
        return outcome(
            ApprovalRestoreDisposition.INVALIDATED,
            RestoreReason.NONCE_BURNED
            if claimed
            else RestoreReason.NONCE_ALREADY_CLAIMED,
        )

    def _workflow(
        self, workflow: RestoredWorkflow, observe: EffectObserver, now: int
    ) -> WorkflowRestoreOutcome:
        source = workflow.source_digest
        if type(workflow.lineage) is RetiredJournalRecord:
            return WorkflowRestoreOutcome(
                source,
                WorkflowRestoreDisposition.RETIRED,
                RestoreReason.JOURNAL_RETIRED,
            )
        lineage = cast(tuple[RecoveryCheckpoint, ...], workflow.lineage)
        tip = lineage[-1]
        recorded = tuple(
            (effect.action_id.serialize(), str(effect.commit_evidence_digest))
            for effect in tip.effects
            if effect.state is EffectState.COMMITTED
        )
        history = tip.approval_receipt_digest

        def held_or_review(
            disposition: WorkflowRestoreDisposition, reason: RestoreReason
        ) -> WorkflowRestoreOutcome:
            return WorkflowRestoreOutcome(
                source,
                disposition,
                reason,
                committed_effects=recorded,
                historical_approval_receipt_digest=history,
            )

        if tip.phase in {RecoveryPhase.COMPLETED, RecoveryPhase.ABORTED}:
            return _plan(source, lineage, (), now, recorded)
        if workflow.authority is not None:
            if self._authority is None:
                return held_or_review(
                    WorkflowRestoreDisposition.HELD,
                    RestoreReason.AUTHORITY_UNAVAILABLE,
                )
            try:
                self._authority.check(workflow.authority, AuthorityBoundary.RESUME)
            except AuthorityDeniedError as denied:
                reason = _AUTHORITY_OUTCOMES.get(denied.receipt.reason)
                if reason is None:
                    return held_or_review(
                        WorkflowRestoreDisposition.HELD,
                        RestoreReason.AUTHORITY_UNAVAILABLE,
                    )
                return held_or_review(
                    WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED, reason
                )
        try:
            observations = tuple(observe(tip))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            return held_or_review(
                WorkflowRestoreDisposition.HELD,
                RestoreReason.EFFECT_STATUS_UNAVAILABLE,
            )
        return _plan(source, lineage, observations, now, recorded)

    def __repr__(self) -> str:
        """Hide injected stores and providers."""
        return "RestoreReconciler(<redacted>)"


def _plan(
    source: str,
    lineage: tuple[RecoveryCheckpoint, ...],
    observations: tuple[Any, ...],
    now: int,
    recorded: tuple[tuple[str, str], ...],
) -> WorkflowRestoreOutcome:
    tip = lineage[-1]
    history = tip.approval_receipt_digest
    try:
        decision = recover_workflow(lineage, observations, now=now)
    except RecoveryError:
        return WorkflowRestoreOutcome(
            source,
            WorkflowRestoreDisposition.HELD,
            RestoreReason.EFFECT_STATUS_UNAVAILABLE,
            committed_effects=recorded,
            historical_approval_receipt_digest=history,
        )
    disposition, reason = _map_decision(decision, tip)
    resume = disposition is WorkflowRestoreDisposition.RESUME
    return WorkflowRestoreOutcome(
        source,
        disposition,
        reason,
        recovery_reason=decision.reason.value,
        committed_effects=_merge_committed(tip, recorded, decision),
        retry_idempotency_keys=decision.retry_idempotency_keys if resume else (),
        recovery_evidence_digest=decision.evidence_digest,
        historical_approval_receipt_digest=history,
    )


def _map_decision(
    decision: RecoveryDecision, tip: RecoveryCheckpoint
) -> tuple[WorkflowRestoreDisposition, RestoreReason]:
    if decision.disposition is RecoveryDisposition.COMPLETE:
        return WorkflowRestoreDisposition.COMPLETE, RestoreReason.EFFECTS_COMPLETE
    if decision.disposition is RecoveryDisposition.RESUME:
        retry = set(decision.retry_effect_ids)
        gated = any(
            effect.approval_required
            for effect in tip.effects
            if effect.action_id.serialize() in retry
        )
        if gated:
            # A pre-restore receipt cannot be verified against current state.
            return (
                WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED,
                RestoreReason.APPROVAL_INVALIDATED,
            )
        return WorkflowRestoreDisposition.RESUME, RestoreReason.RESUME_SAFE
    if decision.reason is RecoveryReason.APPROVAL_EXPIRED:
        return (
            WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED,
            RestoreReason.APPROVAL_EXPIRED,
        )
    if decision.reason is RecoveryReason.APPROVAL_MISSING:
        return (
            WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED,
            RestoreReason.APPROVAL_MISSING,
        )
    return WorkflowRestoreDisposition.REVIEW_REQUIRED, RestoreReason.EFFECTS_NEED_REVIEW


def _merge_committed(
    tip: RecoveryCheckpoint,
    recorded: tuple[tuple[str, str], ...],
    decision: RecoveryDecision,
) -> tuple[tuple[str, str], ...]:
    known = dict(recorded)
    for action_id, evidence in decision.committed_effects:
        known.setdefault(action_id, evidence)
    order = [effect.action_id.serialize() for effect in tip.effects]
    return tuple(
        (action_id, known[action_id]) for action_id in order if action_id in known
    )


def _read_current(store: RecoveryEpochStore) -> int:
    try:
        value = store.current()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise RestoreReconciliationError(
            "recovery_epoch_unavailable", "epochs"
        ) from None
    if type(value) is not int or not 0 <= value <= _MAX_INT:
        raise RestoreReconciliationError("recovery_epoch_unavailable", "epochs")
    return value


def _require_int(value: object, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_INT:
        _fail("invalid_integer", field_name)
    return value  # type: ignore[return-value]


def _require_digest(value: object, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        _fail("invalid_digest", field_name)
    return value  # type: ignore[return-value]


def _fail(code: str, field_name: str) -> NoReturn:
    raise RestoreReconciliationError(code, field_name)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(payload).encode("ascii"))


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


__all__ = [
    "EPOCH_ACTION_SCHEMA_VERSION",
    "RECOVERY_EPOCH_SCHEMA_VERSION",
    "RESTORE_RECONCILIATION_SCHEMA_VERSION",
    "ApprovalRestoreDisposition",
    "ApprovalRestoreOutcome",
    "EffectObserver",
    "InMemoryRecoveryEpochStore",
    "RecoveryEpoch",
    "RecoveryEpochStore",
    "RestoreReason",
    "RestoreReconciler",
    "RestoreReconciliationError",
    "RestoreReconciliationReport",
    "RestoredApproval",
    "RestoredSnapshot",
    "RestoredWorkflow",
    "WorkflowRestoreDisposition",
    "WorkflowRestoreOutcome",
    "epoch_action_digest",
    "require_current_epoch",
]
