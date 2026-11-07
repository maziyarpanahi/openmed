"""Durable recovery contracts for asynchronous service jobs.

The public ``/jobs`` queue reuses the versioned lease, checkpoint, retry,
cancellation, quarantine, and promotion records that Journey ingestion defines
in :mod:`openmed.interop.ingest.contracts` instead of introducing a second
vocabulary.  This module adapts those records to the service job record and
adds the service-specific pieces:

* a bounded :class:`RetryPolicy` with exponential backoff;
* failure classification that separates retryable work from deterministic
  document failures and from poison or structurally invalid work;
* value-free :class:`JobDecisionReceipt` evidence for quarantine, lease
  reclaim, cancellation, recovery failure, and authorized replay;
* :func:`public_recovery_view`, the public projection of recovery metadata,
  which never exposes worker identity, lease identifiers, payload content,
  storage paths, or secrets.

Nothing in this module reads or stores payload content.  Every identifier it
builds is either an opaque random job identifier or a digest derived from
other opaque identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from openmed.clinical.journey_contracts import canonical_digest
from openmed.interop.ingest.contracts import (
    Cancellation,
    Checkpoint,
    IngestionContractError,
    Lease,
    QuarantinePromotion,
    QuarantineResult,
    Retry,
)
from openmed.interop.ingest.runner import classify_failure

from .backpressure import BackpressureError
from .resilience import CircuitBreakerOpenError
from .warm_pool import WarmPoolBackpressureError

__all__ = [
    "CHECKPOINT_STEP",
    "DECISION_ACTIONS",
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_RECONCILE_SECONDS",
    "DEFAULT_RETRY_BASE_SECONDS",
    "DEFAULT_RETRY_MAX_SECONDS",
    "EXPIRING_STATUSES",
    "FAULT_BOUNDARIES",
    "JOB_RECOVERY_SCHEMA_VERSION",
    "JOB_STATUSES",
    "SYSTEM_ACTOR_DIGEST",
    "TERMINAL_STATUSES",
    "FailureDecision",
    "IdempotencyConflictError",
    "IdempotencyKeyError",
    "JobDecisionReceipt",
    "JobRecoveryError",
    "JobReplayAuthorization",
    "ReplayRefusedError",
    "RetryPolicy",
    "StaleLeaseError",
    "classify_document_failure",
    "classify_input_failure",
    "idempotency_key_digest",
    "public_recovery_view",
]

JOB_RECOVERY_SCHEMA_VERSION = 1
CHECKPOINT_STEP = "deidentify_document"

JOB_STATUSES = frozenset(
    {"queued", "running", "done", "failed", "cancelled", "quarantined"}
)
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
EXPIRING_STATUSES = TERMINAL_STATUSES | {"quarantined"}

DEFAULT_LEASE_SECONDS = 60
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_SECONDS = 5
DEFAULT_RETRY_MAX_SECONDS = 300
DEFAULT_RECONCILE_SECONDS = 5.0

FAULT_BOUNDARIES = (
    "submit.persisted",
    "claim.committed",
    "document.processed",
    "checkpoint.committed",
    "terminal.sealed",
    "terminal.committed",
    "webhook.sent",
    "webhook.recorded",
)

DECISION_ACTIONS = frozenset(
    {
        "cancelled",
        "lease_reclaimed",
        "quarantined",
        "recovery_failed",
        "replay_authorized",
    }
)

_RETRYABLE_CLASSIFICATIONS = frozenset(
    {"contention", "dependency", "resource", "transient"}
)
_INPUT_FAIL_CODES = frozenset({"payload_deleted", "payload_expired", "payload_missing"})
_INPUT_RETRY_CODES = frozenset(
    {"key_provider_failed", "key_unavailable", "payload_read_failed"}
)
_INPUT_POLICY_CODES = frozenset({"key_retired", "key_revoked"})
_CONTENTION_ERRORS = (
    BackpressureError,
    CircuitBreakerOpenError,
    WarmPoolBackpressureError,
)

_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SERVICE_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{16,128}\Z")
_IDEMPOTENCY_KEY_RE = re.compile(r"[A-Za-z0-9._~:-]{16,128}\Z")
_IDEMPOTENCY_DOMAIN = "openmed.service.jobs.idempotency.v1"

SYSTEM_ACTOR_DIGEST = canonical_digest({"actor": "openmed.service.jobs"})

_ERROR_MESSAGES = {
    "job_recovery_failed": "job recovery operation failed",
    "job_not_found": "job was not found",
    "job_state_conflict": "job is not in a state that allows this operation",
    "lease_fenced": "job lease is no longer held by this worker",
    "idempotency_key_invalid": (
        "idempotency key must be 16-128 characters of letters, digits, or ._~:-"
    ),
    "idempotency_key_conflict": (
        "idempotency key was already used for a different request"
    ),
    "job_not_quarantined": "job is not quarantined",
    "replay_unauthorized": "job replay was not authorized",
    "replay_input_unavailable": "quarantined job input is no longer available",
    "recovery_state_invalid": "job recovery state is invalid",
}


class JobRecoveryError(RuntimeError):
    """Value-free failure of a durable job operation.

    Attributes:
        code: Controlled identifier naming the failure. Messages are fixed per
            code and never contain content, identifiers, or paths.
    """

    def __init__(self, code: str = "job_recovery_failed") -> None:
        self.code = code if code in _ERROR_MESSAGES else "job_recovery_failed"
        super().__init__(_ERROR_MESSAGES[self.code])


class StaleLeaseError(JobRecoveryError):
    """Raised when a worker writes with a lease epoch it no longer holds."""

    def __init__(self) -> None:
        super().__init__("lease_fenced")


class IdempotencyKeyError(JobRecoveryError, ValueError):
    """Raised for an idempotency key outside the accepted format."""

    def __init__(self) -> None:
        super().__init__("idempotency_key_invalid")


class IdempotencyConflictError(JobRecoveryError):
    """Raised when an idempotency key is reused for a different request."""

    def __init__(self) -> None:
        super().__init__("idempotency_key_conflict")


class ReplayRefusedError(JobRecoveryError):
    """Raised when a replay is refused; ``code`` names the reason."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry budget with capped exponential backoff.

    Attributes:
        max_attempts: Attempts allowed per execution before the job is
            quarantined as poison work.
        base_delay_seconds: Delay before the second attempt.
        max_delay_seconds: Upper bound for any delay.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_seconds: int = DEFAULT_RETRY_BASE_SECONDS
    max_delay_seconds: int = DEFAULT_RETRY_MAX_SECONDS

    def __post_init__(self) -> None:
        for name in ("max_attempts", "base_delay_seconds", "max_delay_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must not be negative")

    def delay_seconds(self, attempt: int) -> int:
        """Return the backoff delay after the failed attempt ``attempt``."""

        exponent = max(0, int(attempt) - 1)
        return min(self.max_delay_seconds, self.base_delay_seconds * (2**exponent))


@dataclass(frozen=True)
class FailureDecision:
    """How the queue handles one failure.

    Attributes:
        action: ``"document"`` (record a failed document and continue),
            ``"retry"`` (release the job for a bounded, backed-off retry),
            ``"quarantine"`` (hold the job until an authorized replay), or
            ``"fail"`` (terminal failure; the work can never succeed).
        classification: Retry or quarantine classification from the
            ingestion contracts.
        reason_code: Controlled, value-free reason code.
    """

    action: str
    classification: str
    reason_code: str


def classify_document_failure(error: Exception) -> FailureDecision:
    """Classify a document-level exception without reading its message.

    Transient, resource, contention, and dependency failures (including
    service backpressure and an open circuit breaker) release the job for a
    retry. Every other class is a deterministic document failure that the job
    records and moves past, as before.

    Args:
        error: Exception raised while processing one document.

    Returns:
        The decision for the failure.
    """

    if isinstance(error, _CONTENTION_ERRORS):
        return FailureDecision("retry", "contention", "service_contention")
    classified = classify_failure(error)
    if classified.classification in _RETRYABLE_CLASSIFICATIONS:
        return FailureDecision(
            "retry", classified.classification, classified.reason_code
        )
    return FailureDecision(
        "document", classified.classification, classified.reason_code
    )


def classify_input_failure(code: str) -> FailureDecision:
    """Classify a failure to read a job's protected input payload.

    Args:
        code: Controlled error code from the payload store or key provider.

    Returns:
        ``fail`` for inputs that no longer exist, ``retry`` for an
        unavailable key or storage, and ``quarantine`` for damaged,
        unauthenticated, or policy-blocked inputs.
    """

    safe_code = code if isinstance(code, str) and _CODE_RE.fullmatch(code) else ""
    if safe_code in _INPUT_FAIL_CODES:
        return FailureDecision("fail", "permanent", safe_code)
    if safe_code in _INPUT_RETRY_CODES:
        return FailureDecision("retry", "dependency", safe_code)
    if safe_code in _INPUT_POLICY_CODES:
        return FailureDecision("quarantine", "policy_denied", safe_code)
    return FailureDecision("quarantine", "malformed", safe_code or "payload_invalid")


@dataclass(frozen=True)
class JobReplayAuthorization:
    """Explicit, value-free authorization to replay one quarantined job.

    Attributes:
        reviewer_digest: ``sha256:`` digest that attributes the decision to a
            reviewer without naming them.
        evidence_digest: ``sha256:`` digest of the review evidence.
        reason_code: Controlled reason code for the replay.
    """

    reviewer_digest: str
    evidence_digest: str
    reason_code: str = "operator_replay"

    def __post_init__(self) -> None:
        for name in ("reviewer_digest", "evidence_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
                raise ReplayRefusedError("replay_unauthorized")
        if (
            not isinstance(self.reason_code, str)
            or _CODE_RE.fullmatch(self.reason_code) is None
        ):
            raise ReplayRefusedError("replay_unauthorized")


@dataclass(frozen=True)
class JobDecisionReceipt:
    """Append-only, value-free evidence of one recovery decision.

    Attributes:
        receipt_id: Deterministic opaque receipt identifier.
        job_id: Opaque job identifier.
        action: One of :data:`DECISION_ACTIONS`.
        reason_code: Controlled reason code.
        from_status: Job status before the decision, or ``"unknown"`` when
            the stored status could not be trusted.
        to_status: Job status after the decision.
        execution: Execution number; a replay starts a new execution.
        attempt: Attempt number within the execution.
        lease_epoch: Highest lease epoch issued for the job.
        recorded_at: UTC timestamp of the decision.
        actor_digest: Digest attributing the decision (a fixed system digest
            for automatic decisions).
        evidence_digest: Review evidence digest for an authorized replay.
        schema_version: Recovery schema version.
    """

    receipt_id: str
    job_id: str
    action: str
    reason_code: str
    from_status: str
    to_status: str
    execution: int
    attempt: int
    lease_epoch: int
    recorded_at: str
    actor_digest: str = SYSTEM_ACTOR_DIGEST
    evidence_digest: Optional[str] = None
    schema_version: int = JOB_RECOVERY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.action not in DECISION_ACTIONS:
            raise JobRecoveryError("recovery_state_invalid")
        if _CODE_RE.fullmatch(self.reason_code) is None:
            raise JobRecoveryError("recovery_state_invalid")
        if self.from_status not in JOB_STATUSES | {"unknown"}:
            raise JobRecoveryError("recovery_state_invalid")
        if self.to_status not in JOB_STATUSES:
            raise JobRecoveryError("recovery_state_invalid")
        for digest in (self.actor_digest, self.evidence_digest):
            if digest is not None and _DIGEST_RE.fullmatch(digest) is None:
                raise JobRecoveryError("recovery_state_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return the receipt as a JSON-safe mapping."""

        return {
            "receipt_id": self.receipt_id,
            "job_id": self.job_id,
            "action": self.action,
            "reason_code": self.reason_code,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "execution": self.execution,
            "attempt": self.attempt,
            "lease_epoch": self.lease_epoch,
            "recorded_at": self.recorded_at,
            "actor_digest": self.actor_digest,
            "evidence_digest": self.evidence_digest,
            "schema_version": self.schema_version,
        }


def idempotency_key_digest(key: str, *, scope: Optional[str] = None) -> str:
    """Return the scoped, value-free digest of a client idempotency key.

    The key is never stored. Its digest is domain-separated and bound to the
    caller's scope (for example the authenticated principal), so the same key
    from two principals names two logical jobs.

    Args:
        key: Client-generated key; use a random UUID.
        scope: Opaque caller scope; ``None`` for unauthenticated callers.

    Returns:
        A ``sha256:`` digest.

    Raises:
        IdempotencyKeyError: If the key is outside the accepted format.
    """

    if not isinstance(key, str) or _IDEMPOTENCY_KEY_RE.fullmatch(key) is None:
        raise IdempotencyKeyError()
    return canonical_digest(
        {
            "domain": _IDEMPOTENCY_DOMAIN,
            "key": key,
            "scope": scope if isinstance(scope, str) and scope else "anonymous",
        }
    )


def isoformat(value: datetime) -> str:
    """Return a UTC timestamp in the job record format."""

    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def parse_time(raw_value: Any) -> Optional[datetime]:
    """Parse a job record timestamp, returning ``None`` when invalid."""

    if not isinstance(raw_value, str) or not raw_value:
        return None
    candidate = raw_value[:-1] + "+00:00" if raw_value.endswith("Z") else raw_value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def is_due(raw_value: Any, now: datetime) -> bool:
    """Return whether an optional deadline has passed (missing means due)."""

    if raw_value is None:
        return True
    moment = parse_time(raw_value)
    return moment is None or moment <= now


def derived_id(prefix: str, *values: str) -> str:
    """Return a deterministic opaque identifier for contract records."""

    digest = canonical_digest({"prefix": prefix, "values": list(values)})
    return f"{prefix}_{digest.removeprefix('sha256:')[:32]}"


def contract_job_id(job_id: str) -> str:
    """Map a service job identifier to the ingestion contract identifier."""

    if not isinstance(job_id, str) or _SERVICE_JOB_ID_RE.fullmatch(job_id) is None:
        raise JobRecoveryError("recovery_state_invalid")
    return f"job_{job_id}"


def new_recovery(*, max_attempts: int, request_digest: str) -> dict[str, Any]:
    """Return the initial recovery block for a newly admitted job."""

    return {
        "schema_version": JOB_RECOVERY_SCHEMA_VERSION,
        "request_digest": request_digest,
        "execution": 1,
        "attempt": 0,
        "max_attempts": int(max_attempts),
        "lease_epoch": 0,
        "lease": None,
        "lease_draining": False,
        "completed_documents": 0,
        "document_error": None,
        "checkpoint": None,
        "retry": None,
        "quarantine": None,
        "quarantines": [],
        "promotions": [],
        "cancellation": None,
        "webhook": None,
        "transitions": [],
    }


def build_lease(
    job_id: str,
    *,
    worker_id: str,
    epoch: int,
    acquired_at: datetime,
    lease_seconds: int,
) -> dict[str, Any]:
    """Return a contract-valid lease mapping for one claim."""

    cid = contract_job_id(job_id)
    return Lease(
        lease_id=derived_id("lease", cid, worker_id, str(epoch)),
        job_id=cid,
        worker_id=worker_id,
        epoch=epoch,
        acquired_at=isoformat(acquired_at),
        expires_at=isoformat(acquired_at + timedelta(seconds=lease_seconds)),
    ).to_dict()


def renew_lease(
    lease: Mapping[str, Any], *, now: datetime, lease_seconds: int
) -> dict[str, Any]:
    """Return ``lease`` with its deadline extended from ``now``."""

    parsed = Lease.from_dict(lease)
    return replace(
        parsed, expires_at=isoformat(now + timedelta(seconds=lease_seconds))
    ).to_dict()


def build_checkpoint(
    job_id: str,
    *,
    request_digest: str,
    input_ref: str,
    sequence: int,
    output_digest: str,
    completed_at: datetime,
    committed_revision: int,
) -> dict[str, Any]:
    """Return a contract-valid checkpoint for ``sequence`` committed documents.

    The input identity is derived from the opaque input payload reference and
    the document position, never from document content.
    """

    cid = contract_job_id(job_id)
    return Checkpoint(
        checkpoint_id=derived_id("checkpoint", cid, input_ref, str(sequence)),
        job_id=cid,
        manifest_digest=request_digest,
        step=CHECKPOINT_STEP,
        sequence=sequence,
        input_digest=canonical_digest({"input_ref": input_ref, "sequence": sequence}),
        output_digest=output_digest,
        completed_at=isoformat(completed_at),
        committed_revision=committed_revision,
    ).to_dict()


def build_retry(
    job_id: str,
    *,
    execution: int,
    attempt: int,
    classification: str,
    reason_code: str,
    recorded_at: datetime,
    retry_after: datetime,
) -> dict[str, Any]:
    """Return a contract-valid retry record carrying the backoff deadline."""

    cid = contract_job_id(job_id)
    return Retry(
        retry_id=derived_id("retry", cid, str(execution), str(attempt), reason_code),
        job_id=cid,
        classification=classification,
        reason_code=reason_code,
        attempt=max(1, int(attempt)),
        recorded_at=isoformat(recorded_at),
        retry_after=isoformat(retry_after),
    ).to_dict()


def build_quarantine(
    job_id: str,
    *,
    request_digest: str,
    execution: int,
    classification: str,
    reason_code: str,
    candidate_count: int,
    failure_count: int,
    created_at: datetime,
    output_digest: Optional[str],
) -> dict[str, Any]:
    """Return a contract-valid quarantine result for one execution."""

    cid = contract_job_id(job_id)
    return QuarantineResult(
        quarantine_id=derived_id("quarantine", cid, str(execution), reason_code),
        job_id=cid,
        manifest_digest=request_digest,
        classification=classification,
        reason_code=reason_code,
        candidate_count=max(0, int(candidate_count)),
        failure_count=max(1, int(failure_count)),
        created_at=isoformat(created_at),
        output_digest=output_digest,
    ).to_dict()


def build_promotion(
    *,
    quarantine_id: str,
    authorization: JobReplayAuthorization,
    promoted_at: datetime,
) -> dict[str, Any]:
    """Return a contract-valid promotion record for an authorized replay."""

    return QuarantinePromotion(
        promotion_id=derived_id(
            "promotion",
            quarantine_id,
            authorization.reviewer_digest,
            authorization.evidence_digest,
        ),
        quarantine_id=quarantine_id,
        reviewer_digest=authorization.reviewer_digest,
        evidence_digest=authorization.evidence_digest,
        promoted_at=isoformat(promoted_at),
    ).to_dict()


def build_cancellation(
    job_id: str,
    *,
    actor_digest: str,
    reason_code: str,
    requested_at: datetime,
) -> dict[str, Any]:
    """Return a contract-valid cancellation record."""

    cid = contract_job_id(job_id)
    return Cancellation(
        cancellation_id=derived_id("cancellation", cid, reason_code),
        job_id=cid,
        actor_digest=actor_digest,
        reason_code=reason_code,
        requested_at=isoformat(requested_at),
    ).to_dict()


def transition_entry(
    *,
    state_version: int,
    from_status: Optional[str],
    to_status: str,
    reason_code: str,
    at: datetime,
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one value-free state transition entry."""

    return {
        "state_version": int(state_version),
        "from": from_status,
        "to": to_status,
        "reason": reason_code,
        "at": isoformat(at),
        "execution": int(recovery.get("execution", 1)),
        "attempt": int(recovery.get("attempt", 0)),
        "lease_epoch": int(recovery.get("lease_epoch", 0)),
    }


def decision_receipt(
    job_id: str,
    *,
    action: str,
    reason_code: str,
    from_status: str,
    to_status: str,
    recovery: Mapping[str, Any],
    state_version: int,
    at: datetime,
    actor_digest: str = SYSTEM_ACTOR_DIGEST,
    evidence_digest: Optional[str] = None,
) -> dict[str, Any]:
    """Return one validated decision receipt as a mapping."""

    execution = int(recovery.get("execution", 1))
    attempt = int(recovery.get("attempt", 0))
    epoch = int(recovery.get("lease_epoch", 0))
    return JobDecisionReceipt(
        receipt_id=derived_id(
            "receipt", job_id, action, str(execution), str(state_version)
        ),
        job_id=job_id,
        action=action,
        reason_code=reason_code,
        from_status=from_status,
        to_status=to_status,
        execution=execution,
        attempt=attempt,
        lease_epoch=epoch,
        recorded_at=isoformat(at),
        actor_digest=actor_digest,
        evidence_digest=evidence_digest,
    ).to_dict()


def validate_recovery(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a job record's recovery block and return it.

    Every embedded lease, checkpoint, retry, quarantine, promotion, and
    cancellation record must parse under its ingestion contract, and the
    counters must agree with each other and with the record.

    Args:
        record: Job record.

    Returns:
        The recovery block (the record's own mapping).

    Raises:
        JobRecoveryError: ``recovery_state_invalid`` for missing, corrupt, or
            ambiguous recovery state.
    """

    try:
        _validate_recovery(record)
    except (IngestionContractError, TypeError, ValueError, KeyError):
        raise JobRecoveryError("recovery_state_invalid") from None
    recovery = record["recovery"]
    assert isinstance(recovery, dict)
    return recovery


def _validate_recovery(record: Mapping[str, Any]) -> None:
    status = record.get("status")
    recovery = record.get("recovery")
    if status not in JOB_STATUSES or not isinstance(recovery, dict):
        raise ValueError("recovery state missing")
    if recovery.get("schema_version") != JOB_RECOVERY_SCHEMA_VERSION:
        raise ValueError("recovery schema unsupported")
    digest = recovery.get("request_digest")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ValueError("request digest invalid")
    execution = _int(recovery, "execution", minimum=1)
    _int(recovery, "attempt", minimum=0)
    _int(recovery, "max_attempts", minimum=1)
    epoch = _int(recovery, "lease_epoch", minimum=0)
    completed = _int(recovery, "completed_documents", minimum=0)
    document_count = record.get("document_count")
    if not isinstance(document_count, int) or completed > document_count:
        raise ValueError("checkpoint beyond document count")
    cid = contract_job_id(str(record.get("id")))
    lease = recovery.get("lease")
    if lease is not None:
        parsed_lease = Lease.from_dict(lease)
        if parsed_lease.job_id != cid or parsed_lease.epoch != epoch:
            raise ValueError("lease does not match job")
    if status == "running" and lease is None:
        raise ValueError("running job without lease")
    if status not in {"running"} and lease is not None:
        raise ValueError("lease outside running state")
    checkpoint = recovery.get("checkpoint")
    if checkpoint is not None:
        parsed_checkpoint = Checkpoint.from_dict(checkpoint)
        if parsed_checkpoint.job_id != cid or parsed_checkpoint.sequence != completed:
            raise ValueError("checkpoint does not match progress")
    elif completed:
        raise ValueError("progress without checkpoint")
    retry = recovery.get("retry")
    if retry is not None:
        Retry.from_dict(retry)
    quarantine = recovery.get("quarantine")
    if quarantine is not None:
        QuarantineResult.from_dict(quarantine)
    if (status == "quarantined") != (quarantine is not None):
        raise ValueError("quarantine does not match status")
    for entry in _list(recovery, "quarantines"):
        QuarantineResult.from_dict(entry)
    promotions = _list(recovery, "promotions")
    for entry in promotions:
        QuarantinePromotion.from_dict(entry)
    if len(promotions) != execution - 1:
        raise ValueError("executions do not match promotions")
    cancellation = recovery.get("cancellation")
    if cancellation is not None:
        Cancellation.from_dict(cancellation)
    webhook = recovery.get("webhook")
    if webhook is not None and not isinstance(webhook, dict):
        raise ValueError("webhook intent invalid")
    _list(recovery, "transitions")


def _int(recovery: Mapping[str, Any], name: str, *, minimum: int) -> int:
    value = recovery.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("recovery counter invalid")
    return value


def _list(recovery: Mapping[str, Any], name: str) -> Sequence[Any]:
    value = recovery.get(name)
    if not isinstance(value, list):
        raise ValueError("recovery list invalid")
    return value


def public_recovery_view(recovery: Any) -> Optional[dict[str, Any]]:
    """Return the public, value-free projection of a recovery block.

    Lease identifiers, worker identifiers, payload references, request
    digests, and promotion reviewer digests are omitted.

    Args:
        recovery: The record's recovery block, or ``None``.

    Returns:
        Versioned recovery metadata, or ``None`` for records without one.
    """

    if not isinstance(recovery, Mapping):
        return None
    try:
        return _public_view(recovery)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _public_view(recovery: Mapping[str, Any]) -> dict[str, Any]:
    lease = recovery.get("lease")
    checkpoint = recovery.get("checkpoint")
    retry = recovery.get("retry")
    quarantine = recovery.get("quarantine")
    webhook = recovery.get("webhook")
    return {
        "schema_version": recovery.get("schema_version"),
        "execution": recovery.get("execution"),
        "attempt": recovery.get("attempt"),
        "max_attempts": recovery.get("max_attempts"),
        "lease_epoch": recovery.get("lease_epoch"),
        "lease": (
            None
            if not isinstance(lease, Mapping)
            else {
                "epoch": lease["epoch"],
                "acquired_at": lease["acquired_at"],
                "expires_at": lease["expires_at"],
            }
        ),
        "checkpoint": (
            None
            if not isinstance(checkpoint, Mapping)
            else {
                "sequence": checkpoint["sequence"],
                "completed_at": checkpoint["completed_at"],
                "output_digest": checkpoint["output_digest"],
            }
        ),
        "completed_documents": recovery.get("completed_documents"),
        "retry": (
            None
            if not isinstance(retry, Mapping)
            else {
                "classification": retry["classification"],
                "reason_code": retry["reason_code"],
                "attempt": retry["attempt"],
                "retry_after": retry["retry_after"],
            }
        ),
        "quarantine": (
            None
            if not isinstance(quarantine, Mapping)
            else {
                "quarantine_id": quarantine["quarantine_id"],
                "classification": quarantine["classification"],
                "reason_code": quarantine["reason_code"],
                "created_at": quarantine["created_at"],
            }
        ),
        "webhook": (
            None if not isinstance(webhook, Mapping) else {"state": webhook["state"]}
        ),
        "transitions": [dict(entry) for entry in recovery.get("transitions") or []],
    }
