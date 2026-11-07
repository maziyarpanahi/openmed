"""Async de-identification jobs for the OpenMed REST service.

Job metadata (status, counts, hashes, offsets, opaque payload references, and
value-free recovery state) is persisted by :class:`LocalJobStore`.  PHI-bearing
content never enters that metadata: each submission is snapshotted once into
an immutable input payload, and each terminal webhook event is sealed once
into an immutable terminal-event payload, both held by a
:class:`~openmed.service.job_payloads.JobPayloadStore` under an explicit,
bounded retention deadline.

Jobs are durable.  The input payload reference, the immutable request digest,
and an optional idempotency key digest are persisted before a job is admitted
to the worker pool.  Workers claim a job with an expiring lease whose epoch
fences every state write, commit a checkpoint after every document, and commit
the terminal state together with the webhook intent.  Failures are separated
into document failures, bounded retries with backoff, and poison quarantine;
a quarantined job runs again only after an authorized replay.  A reconcile
pass at startup (and periodically while work is pending) reclaims expired
leases, resumes queued work, quarantines poison work, and re-drives pending
webhook intents.  The recovery records are the Journey ingestion contracts,
adapted in :mod:`openmed.service.job_recovery`.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import math
import os
import secrets
import tempfile
import threading
import uuid
import weakref
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

import openmed
from openmed.clinical.journey_contracts import canonical_digest
from openmed.core.audit import hash_text
from openmed.core.key_provider import EncryptionError, KeyProvider

from .job_payloads import (
    DEFAULT_PAYLOAD_RETENTION_SECONDS,
    JOB_PAYLOAD_SCHEMA_VERSION,
    PAYLOAD_KIND_INPUT,
    PAYLOAD_KIND_TERMINAL_EVENT,
    PAYLOAD_KINDS,
    PAYLOAD_MODE_ENCRYPTED,
    PAYLOAD_MODE_MEMORY,
    PAYLOAD_STATE_DELETED,
    PAYLOAD_STATE_EXPIRED,
    PAYLOAD_STATE_RETAINED,
    EncryptedJobPayloadStore,
    JobPayloadError,
    JobPayloadStore,
    MemoryJobPayloadStore,
    PayloadRef,
)
from .job_recovery import (
    DEFAULT_LEASE_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RECONCILE_SECONDS,
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS,
    EXPIRING_STATUSES,
    FAULT_BOUNDARIES,
    JOB_RECOVERY_SCHEMA_VERSION,
    JOB_STATUSES,
    SYSTEM_ACTOR_DIGEST,
    TERMINAL_STATUSES,
    FailureDecision,
    IdempotencyConflictError,
    JobRecoveryError,
    JobReplayAuthorization,
    ReplayRefusedError,
    RetryPolicy,
    StaleLeaseError,
    build_cancellation,
    build_checkpoint,
    build_lease,
    build_promotion,
    build_quarantine,
    build_retry,
    classify_document_failure,
    classify_input_failure,
    decision_receipt,
    idempotency_key_digest,
    is_due,
    new_recovery,
    parse_time,
    public_recovery_view,
    renew_lease,
    transition_entry,
    validate_recovery,
)
from .runtime import ServiceRuntime
from .schemas import DeidentifyJobDocument, DeidentifyJobRequest, JobWebhookRequest
from .webhooks import (
    DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    WebhookDeliveryResult,
    deliver_webhook,
)

logger = logging.getLogger(__name__)

SERVICE_JOBS_STORE_ENV_VAR = "OPENMED_SERVICE_JOBS_STORE_PATH"
SERVICE_JOBS_TTL_ENV_VAR = "OPENMED_SERVICE_JOBS_TTL_SECONDS"
SERVICE_JOBS_WORKERS_ENV_VAR = "OPENMED_SERVICE_JOBS_WORKERS"
SERVICE_JOBS_PAYLOAD_MODE_ENV_VAR = "OPENMED_SERVICE_JOBS_PAYLOAD_MODE"
SERVICE_JOBS_PAYLOAD_DIR_ENV_VAR = "OPENMED_SERVICE_JOBS_PAYLOAD_DIR"
SERVICE_JOBS_PAYLOAD_RETENTION_ENV_VAR = (
    "OPENMED_SERVICE_JOBS_PAYLOAD_RETENTION_SECONDS"
)
SERVICE_JOBS_LEASE_ENV_VAR = "OPENMED_SERVICE_JOBS_LEASE_SECONDS"
SERVICE_JOBS_MAX_ATTEMPTS_ENV_VAR = "OPENMED_SERVICE_JOBS_MAX_ATTEMPTS"
SERVICE_JOBS_RETRY_BASE_ENV_VAR = "OPENMED_SERVICE_JOBS_RETRY_BASE_SECONDS"
SERVICE_JOBS_RETRY_MAX_ENV_VAR = "OPENMED_SERVICE_JOBS_RETRY_MAX_SECONDS"
SERVICE_JOBS_RECONCILE_ENV_VAR = "OPENMED_SERVICE_JOBS_RECONCILE_SECONDS"
DEFAULT_JOBS_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOBS_WORKERS = 2
PAYLOAD_MODES = frozenset({PAYLOAD_MODE_MEMORY, PAYLOAD_MODE_ENCRYPTED})

WEBHOOK_SENDER_FAILED = "webhook_sender_failed"
WEBHOOK_EVENT_UNAVAILABLE = "terminal_event_unavailable"

_PAYLOAD_UNREADABLE_MESSAGE = "Job input payload could not be read"


WebhookSender = Callable[..., WebhookDeliveryResult]
FaultHook = Callable[[str, str], None]
ReplayAuthorizer = Callable[[str, JobReplayAuthorization], bool]


@dataclass(frozen=True)
class _JobWorkItem:
    """Queued work: the job and the opaque reference of its input snapshot."""

    job_id: str
    input_ref: str


@dataclass(frozen=True)
class _DeliveryWorkItem:
    """Re-driven webhook intent: the job and its terminal-event reference."""

    job_id: str
    payload_ref: str


@dataclass(frozen=True)
class _JobLease:
    """In-memory handle for one claimed lease; ``epoch`` is the fencing token."""

    job_id: str
    epoch: int
    execution: int
    input_ref: str


class JobSubmission(NamedTuple):
    """Result of :meth:`DeidentifyJobQueue.submit_job`.

    Attributes:
        record: The job record.
        created: ``False`` when an idempotent resubmission returned the
            existing logical job.
    """

    record: dict[str, Any]
    created: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_positive_int(
    raw_value: Optional[str],
    *,
    env_var: str,
    default: int,
) -> int:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{env_var} must be a positive integer")
    return parsed


def _parse_non_negative_int(
    raw_value: Optional[str],
    *,
    env_var: str,
    default: int,
) -> int:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{env_var} must be a non-negative integer")
    return parsed


def _default_store_path() -> Path:
    return Path.home() / ".cache" / "openmed" / "service-jobs.json"


def parse_jobs_store_path(raw_value: Optional[str]) -> Path:
    """Parse the async job metadata store path."""
    if raw_value is None or not raw_value.strip():
        return _default_store_path()
    return Path(raw_value).expanduser()


def parse_jobs_ttl_seconds(raw_value: Optional[str]) -> int:
    """Parse terminal job metadata retention in seconds."""
    return _parse_positive_int(
        raw_value,
        env_var=SERVICE_JOBS_TTL_ENV_VAR,
        default=DEFAULT_JOBS_TTL_SECONDS,
    )


def parse_jobs_workers(raw_value: Optional[str]) -> int:
    """Parse the bounded async job worker count."""
    return _parse_positive_int(
        raw_value,
        env_var=SERVICE_JOBS_WORKERS_ENV_VAR,
        default=DEFAULT_JOBS_WORKERS,
    )


def parse_jobs_lease_seconds(raw_value: Optional[str]) -> int:
    """Parse the async job lease duration in seconds."""
    return _parse_positive_int(
        raw_value,
        env_var=SERVICE_JOBS_LEASE_ENV_VAR,
        default=DEFAULT_LEASE_SECONDS,
    )


def parse_jobs_retry_policy(
    environ: Optional[Mapping[str, str]] = None,
) -> RetryPolicy:
    """Parse the async job retry budget and backoff from the environment.

    Args:
        environ: Mapping to read; defaults to :data:`os.environ`.

    Returns:
        The retry policy.

    Raises:
        ValueError: If a value is not an integer in range.
    """
    env = os.environ if environ is None else environ
    return RetryPolicy(
        max_attempts=_parse_positive_int(
            env.get(SERVICE_JOBS_MAX_ATTEMPTS_ENV_VAR),
            env_var=SERVICE_JOBS_MAX_ATTEMPTS_ENV_VAR,
            default=DEFAULT_MAX_ATTEMPTS,
        ),
        base_delay_seconds=_parse_non_negative_int(
            env.get(SERVICE_JOBS_RETRY_BASE_ENV_VAR),
            env_var=SERVICE_JOBS_RETRY_BASE_ENV_VAR,
            default=DEFAULT_RETRY_BASE_SECONDS,
        ),
        max_delay_seconds=_parse_non_negative_int(
            env.get(SERVICE_JOBS_RETRY_MAX_ENV_VAR),
            env_var=SERVICE_JOBS_RETRY_MAX_ENV_VAR,
            default=DEFAULT_RETRY_MAX_SECONDS,
        ),
    )


def parse_jobs_reconcile_seconds(raw_value: Optional[str]) -> Optional[float]:
    """Parse the periodic reconcile interval; ``0`` disables the background pass.

    Args:
        raw_value: Seconds as a decimal string, or empty for the default.

    Returns:
        The interval in seconds, or ``None`` when disabled.

    Raises:
        ValueError: If the value is negative or not a finite number.
    """
    if raw_value is None or not raw_value.strip():
        return DEFAULT_RECONCILE_SECONDS
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{SERVICE_JOBS_RECONCILE_ENV_VAR} must be a non-negative number"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"{SERVICE_JOBS_RECONCILE_ENV_VAR} must be a non-negative number"
        )
    return parsed or None


def parse_jobs_payload_mode(raw_value: Optional[str]) -> Optional[str]:
    """Parse the async job payload retention mode.

    Args:
        raw_value: ``"memory"``, ``"encrypted"``, or empty for the default.

    Returns:
        The mode, or ``None`` when unset.

    Raises:
        ValueError: If the value names another mode.
    """
    if raw_value is None or not raw_value.strip():
        return None
    mode = raw_value.strip().lower()
    if mode not in PAYLOAD_MODES:
        raise ValueError(
            f"{SERVICE_JOBS_PAYLOAD_MODE_ENV_VAR} must be 'memory' or 'encrypted'"
        )
    return mode


def parse_jobs_payload_dir(raw_value: Optional[str]) -> Path:
    """Parse the encrypted async job payload directory."""
    if raw_value is None or not raw_value.strip():
        return Path.home() / ".cache" / "openmed" / "service-job-payloads"
    return Path(raw_value).expanduser()


def parse_jobs_payload_retention_seconds(raw_value: Optional[str]) -> int:
    """Parse the maximum async job payload retention in seconds."""
    return _parse_positive_int(
        raw_value,
        env_var=SERVICE_JOBS_PAYLOAD_RETENTION_ENV_VAR,
        default=DEFAULT_PAYLOAD_RETENTION_SECONDS,
    )


def job_payload_store_from_env(
    *,
    key_provider: Optional[KeyProvider] = None,
) -> JobPayloadStore:
    """Build the async job payload store from service environment variables.

    Durable retention is opt-in. Without a mode or key provider, payloads stay
    in memory only. ``OPENMED_SERVICE_JOBS_PAYLOAD_MODE=encrypted`` (or passing
    ``key_provider`` without a mode) retains payloads on disk, sealed with the
    caller-supplied key provider.

    Args:
        key_provider: Caller-owned key provider for encrypted retention.

    Returns:
        A memory-only or encrypted payload store.

    Raises:
        ValueError: If encrypted retention is requested without a key
            provider, or an environment value is invalid.
    """
    mode = parse_jobs_payload_mode(os.getenv(SERVICE_JOBS_PAYLOAD_MODE_ENV_VAR))
    if mode is None:
        mode = PAYLOAD_MODE_MEMORY if key_provider is None else PAYLOAD_MODE_ENCRYPTED
    retention = parse_jobs_payload_retention_seconds(
        os.getenv(SERVICE_JOBS_PAYLOAD_RETENTION_ENV_VAR)
    )
    if mode == PAYLOAD_MODE_MEMORY:
        return MemoryJobPayloadStore(max_retention_seconds=retention)
    if key_provider is None:
        raise ValueError(
            "Encrypted async job payload retention requires a caller-supplied "
            "key provider"
        )
    return EncryptedJobPayloadStore(
        parse_jobs_payload_dir(os.getenv(SERVICE_JOBS_PAYLOAD_DIR_ENV_VAR)),
        key_provider=key_provider,
        max_retention_seconds=retention,
    )


def _copy_record(record: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(record)


def _state_version(record: Mapping[str, Any]) -> int:
    value = record.get("state_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


_PATH_LOCKS: dict[str, Any] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> Any:
    """Return the process-wide lock shared by every store on one path."""
    key = os.path.abspath(os.fspath(path))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


class _DuplicateIdempotencyKey(Exception):
    """Internal: another live record already holds the idempotency key."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("duplicate idempotency key")
        self.record = record


StoreTransition = Callable[[dict[str, Any]], Optional[Sequence[Mapping[str, Any]]]]


class LocalJobStore:
    """Durable local metadata store for async jobs.

    The store intentionally persists only metadata and no raw submitted text or
    redacted output. Job payloads are held by a separate
    :class:`~openmed.service.job_payloads.JobPayloadStore` and appear here only
    as opaque references with their lifecycle state and retention deadline.

    Every write replaces the whole file atomically, so one write commits a
    state transition and its decision receipts together. Stores opened on the
    same path in one process share a lock and reload the file when another
    store has replaced it, so a guarded :meth:`transition` always sees the
    latest committed state. The store serves one service process; it does not
    coordinate writers across processes.
    """

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: int = DEFAULT_JOBS_TTL_SECONDS,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = path
        self.ttl_seconds = int(ttl_seconds)
        self.clock = clock
        self._lock = _path_lock(path)
        with self._lock:
            self._records, self._decisions = self._load()
            self._signature = self._file_signature()

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist a new job record.

        Raises:
            _DuplicateIdempotencyKey: Internal signal that a live record
                already holds the same idempotency key digest.
        """
        with self._lock:
            self._refresh_locked()
            key_digest = _record_idempotency_digest(record)
            if key_digest is not None:
                existing = self._find_idempotent_locked(key_digest)
                if existing is not None:
                    raise _DuplicateIdempotencyKey(_copy_record(existing))
            previous = _copy_record(self._records)
            self.cleanup_expired_locked()
            job_id = str(record["id"])
            self._records[job_id] = _copy_record(record)
            self._publish_locked(previous)
            return _copy_record(self._records[job_id])

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return one job record, if present."""
        with self._lock:
            self._refresh_locked()
            self.cleanup_expired_locked()
            record = self._records.get(job_id)
            return None if record is None else _copy_record(record)

    def records(self) -> list[dict[str, Any]]:
        """Return copies of every unexpired job record."""
        with self._lock:
            self._refresh_locked()
            self.cleanup_expired_locked()
            return [_copy_record(record) for record in self._records.values()]

    def find_idempotent(self, key_digest: str) -> Optional[dict[str, Any]]:
        """Return the live record holding an idempotency key digest, if any."""
        with self._lock:
            self._refresh_locked()
            record = self._find_idempotent_locked(key_digest)
            return None if record is None else _copy_record(record)

    def decisions(self) -> list[dict[str, Any]]:
        """Return the append-only decision receipts, oldest first."""
        with self._lock:
            self._refresh_locked()
            return [_copy_record(receipt) for receipt in self._decisions]

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        """Apply changes to a job record and persist them."""
        with self._lock:
            self._refresh_locked()
            previous = _copy_record(self._records)
            record = self._records[job_id]
            record.update(_copy_record(changes))
            record["updated_at"] = _isoformat(self.clock())
            record["state_version"] = _state_version(record) + 1
            self._publish_locked(previous)
            return _copy_record(record)

    def transition(
        self,
        job_id: str,
        apply: StoreTransition,
    ) -> Optional[dict[str, Any]]:
        """Apply one guarded change to a job record atomically.

        ``apply`` receives a private copy of the current record. It checks its
        preconditions and mutates the copy, then returns the decision receipts
        to append in the same write, or ``None`` to leave the record unchanged.
        Exceptions it raises propagate and nothing is written.

        Args:
            job_id: Job to change.
            apply: Guard-and-mutate callback.

        Returns:
            The committed record, or ``None`` when the job is missing or
            ``apply`` declined the change.
        """
        with self._lock:
            self._refresh_locked()
            current = self._records.get(job_id)
            if current is None:
                return None
            candidate = _copy_record(current)
            receipts = apply(candidate)
            if receipts is None:
                return None
            candidate["state_version"] = _state_version(current) + 1
            candidate["updated_at"] = _isoformat(self.clock())
            previous_decisions = list(self._decisions)
            self._records[job_id] = candidate
            self._decisions.extend(dict(receipt) for receipt in receipts)
            try:
                self._persist_locked()
            except BaseException:
                self._records[job_id] = current
                self._decisions = previous_decisions
                raise
            return _copy_record(candidate)

    def cleanup_expired(self) -> None:
        """Remove terminal and quarantined records whose TTL has elapsed."""
        with self._lock:
            self._refresh_locked()
            previous = _copy_record(self._records)
            if self.cleanup_expired_locked():
                self._publish_locked(previous)

    def cleanup_expired_locked(self) -> bool:
        now = self.clock()
        expired = [
            job_id
            for job_id, record in self._records.items()
            if record.get("status") in EXPIRING_STATUSES
            and _parse_timestamp(record.get("expires_at")) <= now
        ]
        for job_id in expired:
            self._records.pop(job_id, None)
        return bool(expired)

    def _find_idempotent_locked(self, key_digest: str) -> Optional[dict[str, Any]]:
        now = self.clock()
        for record in self._records.values():
            if _record_idempotency_digest(record) != key_digest:
                continue
            if (
                record.get("status") in EXPIRING_STATUSES
                and _parse_timestamp(record.get("expires_at")) <= now
            ):
                continue
            return record
        return None

    def _file_signature(self) -> Optional[tuple[int, int, int]]:
        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _refresh_locked(self) -> None:
        signature = self._file_signature()
        if signature != self._signature:
            self._records, self._decisions = self._load()
            self._signature = signature

    def _load(self) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        if not self.path.exists():
            return {}, []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError("Job metadata store contains invalid JSON") from None
        if not isinstance(raw, dict):
            raise ValueError("Job metadata store must contain a JSON object")
        jobs = raw.get("jobs")
        if not isinstance(jobs, dict):
            raise ValueError("Job metadata store must contain a jobs object")
        if any(not isinstance(record, dict) for record in jobs.values()):
            raise ValueError("Job metadata store contains an invalid job record")
        decisions = raw.get("decisions", [])
        if not isinstance(decisions, list) or any(
            not isinstance(receipt, dict) for receipt in decisions
        ):
            raise ValueError("Job metadata store contains an invalid decision receipt")
        return (
            {str(job_id): dict(record) for job_id, record in jobs.items()},
            [dict(receipt) for receipt in decisions],
        )

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"decisions": self._decisions, "jobs": self._records}
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                delete=False,
            ) as handle:
                temp_name = handle.name
                json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_name, self.path)
        except BaseException:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            raise
        self._signature = self._file_signature()

    def _publish_locked(self, previous: dict[str, dict[str, Any]]) -> None:
        try:
            self._persist_locked()
        except BaseException:
            self._records = previous
            raise


def _record_idempotency_digest(record: Mapping[str, Any]) -> Optional[str]:
    idempotency = record.get("idempotency")
    if not isinstance(idempotency, dict):
        return None
    digest = idempotency.get("key_digest")
    return digest if isinstance(digest, str) else None


def _parse_timestamp(raw_value: Any) -> datetime:
    if not isinstance(raw_value, str) or not raw_value:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class DeidentifyJobQueue:
    """Durable, bounded local worker pool for async de-identification jobs.

    Each submission is snapshotted once into an immutable input payload before
    any metadata is derived, so later changes to the caller's request object
    never reach the queued work. The payload reference, the request digest,
    and an optional idempotency key digest are persisted before the job is
    handed to a worker.

    A worker claims a job with an expiring lease. The lease epoch is the
    fencing token: every checkpoint, retry, quarantine, and terminal write
    checks it, so a worker that lost its lease can never commit. Progress is
    checkpointed after every document, and a restarted job resumes after the
    last committed document. The terminal state and the webhook intent are
    committed in one write; delivery reads the immutable terminal-event
    payload instead of rebuilding the body from mutable job state.

    Args:
        runtime: Service runtime used to run model requests.
        store: Job metadata store.
        max_workers: Bounded worker count.
        webhook_sender: Callable that delivers one signed webhook.
        clock: Source of timezone-aware UTC timestamps.
        payload_store: Payload store; defaults to a memory-only store.
        retry_policy: Retry budget and backoff; defaults to
            :class:`~openmed.service.job_recovery.RetryPolicy`.
        lease_seconds: Lease duration. Workers renew the lease at every
            checkpoint and the periodic reconcile pass renews leases held by
            live workers.
        reconcile_interval_seconds: Interval of the background reconcile pass,
            or ``None`` to reconcile only at startup and on explicit
            :meth:`reconcile` calls.
        replay_authorizer: Optional policy hook that must approve every
            replay; it receives the job id and the authorization record.
        fault_hook: Test seam called with ``(boundary, job_id)`` at each
            durability boundary in
            :data:`~openmed.service.job_recovery.FAULT_BOUNDARIES`.
    """

    def __init__(
        self,
        runtime: ServiceRuntime,
        *,
        store: LocalJobStore,
        max_workers: int = DEFAULT_JOBS_WORKERS,
        webhook_sender: WebhookSender = deliver_webhook,
        clock: Callable[[], datetime] = _utc_now,
        payload_store: Optional[JobPayloadStore] = None,
        retry_policy: Optional[RetryPolicy] = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        reconcile_interval_seconds: Optional[float] = None,
        replay_authorizer: Optional[ReplayAuthorizer] = None,
        fault_hook: Optional[FaultHook] = None,
    ) -> None:
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds < 1
        ):
            raise ValueError("lease_seconds must be a positive integer")
        if reconcile_interval_seconds is not None and not (
            math.isfinite(reconcile_interval_seconds) and reconcile_interval_seconds > 0
        ):
            raise ValueError("reconcile_interval_seconds must be positive")
        self.runtime = runtime
        self.store = store
        self.webhook_sender = webhook_sender
        self.clock = clock
        self.payloads = (
            MemoryJobPayloadStore(clock=clock)
            if payload_store is None
            else payload_store
        )
        self.retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self.lease_seconds = lease_seconds
        self.reconcile_interval_seconds = reconcile_interval_seconds
        self.replay_authorizer = replay_authorizer
        self.fault_hook = fault_hook
        self.worker_id = "worker_" + secrets.token_hex(16)
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="openmed-job",
        )
        self._shutdown = False
        self._lock = threading.Lock()
        self._futures_lock = threading.Lock()
        self._futures: dict[str, Future[None]] = {}
        self._active: dict[str, int] = {}
        self._deliveries: set[str] = set()
        self._ticker_stop = threading.Event()
        self._ticker: Optional[threading.Thread] = None
        self.payloads.recover(self._live_payload_refs())
        counts = self.reconcile()
        if any(counts.get(name) for name in ("dispatched", "waiting", "leased")):
            self._ensure_ticker()

    @classmethod
    def from_env(
        cls,
        runtime: ServiceRuntime,
        *,
        key_provider: Optional[KeyProvider] = None,
    ) -> "DeidentifyJobQueue":
        """Create a queue from service environment variables.

        Args:
            runtime: Service runtime used to run model requests.
            key_provider: Optional caller-owned key provider that enables
                encrypted payload retention, and with it restart-safe resume.

        Returns:
            A configured job queue.
        """
        store = LocalJobStore(
            parse_jobs_store_path(os.getenv(SERVICE_JOBS_STORE_ENV_VAR)),
            ttl_seconds=parse_jobs_ttl_seconds(os.getenv(SERVICE_JOBS_TTL_ENV_VAR)),
        )
        return cls(
            runtime,
            store=store,
            max_workers=parse_jobs_workers(os.getenv(SERVICE_JOBS_WORKERS_ENV_VAR)),
            payload_store=job_payload_store_from_env(key_provider=key_provider),
            retry_policy=parse_jobs_retry_policy(),
            lease_seconds=parse_jobs_lease_seconds(
                os.getenv(SERVICE_JOBS_LEASE_ENV_VAR)
            ),
            reconcile_interval_seconds=parse_jobs_reconcile_seconds(
                os.getenv(SERVICE_JOBS_RECONCILE_ENV_VAR)
            ),
        )

    # Submission -------------------------------------------------------

    def submit(
        self,
        payload: DeidentifyJobRequest,
        *,
        idempotency_key: Optional[str] = None,
        idempotency_scope: Optional[str] = None,
    ) -> dict[str, Any]:
        """Submit one de-identification job and return its initial metadata.

        Args:
            payload: The validated request. It is snapshotted immediately; the
                caller may reuse or mutate it afterwards.
            idempotency_key: Optional client key; a resubmission with the same
                key and request returns the existing job.
            idempotency_scope: Opaque caller scope the key is bound to.

        Returns:
            The job record (the existing one for an idempotent resubmission).

        Raises:
            ValueError: If the requested payload retention exceeds the
                configured maximum, the model is not served, or the
                idempotency key is malformed.
            IdempotencyConflictError: If the key was used for another request.
            RuntimeError: If the queue is shutting down.
            EncryptionError: If the input payload cannot be retained.
        """
        return self.submit_job(
            payload,
            idempotency_key=idempotency_key,
            idempotency_scope=idempotency_scope,
        ).record

    def submit_job(
        self,
        payload: DeidentifyJobRequest,
        *,
        idempotency_key: Optional[str] = None,
        idempotency_scope: Optional[str] = None,
    ) -> JobSubmission:
        """Submit one job and report whether it created a new logical job.

        The input payload, its reference, the request digest, and the
        idempotency key digest are persisted before the job is admitted.

        Args:
            payload: The validated request; it is snapshotted immediately.
            idempotency_key: Optional client key (16-128 characters; use a
                random UUID). It is stored only as a scoped digest.
            idempotency_scope: Opaque caller scope, for example the
                authenticated principal, that the key is bound to.

        Returns:
            The record and ``created``; ``created`` is ``False`` when the same
            authorized key and request returned the existing job.

        Raises:
            ValueError: For an out-of-range retention, an unserved model, or a
                malformed idempotency key.
            IdempotencyConflictError: If the key was used for another request.
            RuntimeError: If the queue is shutting down.
            EncryptionError: If the input payload cannot be retained.
        """
        snapshot = DeidentifyJobRequest(**_request_body(payload))
        self.runtime.validate_served_model(snapshot.model_name)
        retention = self._payload_retention_seconds(snapshot)
        body = _request_body(snapshot)
        key_digest = (
            None
            if idempotency_key is None
            else idempotency_key_digest(idempotency_key, scope=idempotency_scope)
        )
        request_digest = _request_digest(snapshot)
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Job queue is shutting down")
            self._sweep_payloads()
            if key_digest is not None:
                existing = self.store.find_idempotent(key_digest)
                if existing is not None:
                    return self._idempotent_existing(existing, request_digest, body)
            now = self.clock()
            job_id = uuid.uuid4().hex
            input_ref = self.payloads.create(
                PAYLOAD_KIND_INPUT,
                job_id,
                body,
                retention_seconds=retention,
                now=now,
            )
            record = self._new_record(
                snapshot,
                job_id=job_id,
                now=now,
                input_ref=input_ref,
                retention_seconds=retention,
                request_digest=request_digest,
                key_digest=key_digest,
            )
            try:
                stored = self.store.create(record)
            except _DuplicateIdempotencyKey as duplicate:
                self._delete_payload(input_ref.ref, reason="duplicate_submission")
                return self._idempotent_existing(duplicate.record, request_digest, body)
            except BaseException:
                self._delete_payload(input_ref.ref, reason="submission_failed")
                raise
            self._fault("submit.persisted", job_id)
            self._dispatch(job_id, input_ref.ref)
        self._ensure_ticker()
        return JobSubmission(stored, True)

    def _idempotent_existing(
        self,
        existing: dict[str, Any],
        request_digest: str,
        body: Mapping[str, Any],
    ) -> JobSubmission:
        recovery = existing.get("recovery")
        stored_digest = (
            recovery.get("request_digest") if isinstance(recovery, dict) else None
        )
        if stored_digest != request_digest:
            raise IdempotencyConflictError()
        # The digest covers values the record already holds. While the input
        # snapshot is retained, compare the full request too, which includes
        # fields such as the date shift that never enter metadata.
        entry = _payload_entry(existing, PAYLOAD_KIND_INPUT)
        if (
            entry is not None
            and entry.get("state") == PAYLOAD_STATE_RETAINED
            and self.payloads.contains(entry["ref"])
        ):
            try:
                original = self.payloads.read(
                    entry["ref"], job_id=str(existing["id"]), kind=PAYLOAD_KIND_INPUT
                )
            except EncryptionError:
                original = None
            if original is not None and _canonical(original) != _canonical(body):
                raise IdempotencyConflictError()
        return JobSubmission(existing, False)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return persisted job metadata."""
        self._sweep_payloads()
        return self.store.get(job_id)

    def decision_receipts(self) -> list[dict[str, Any]]:
        """Return the append-only, value-free recovery decision receipts."""
        return self.store.decisions()

    def recovery_evidence(self) -> dict[str, Any]:
        """Return value-free recovery evidence for this queue.

        Returns:
            Schema versions, counts of jobs per status, counts of decision
            receipts per action, the highest lease epoch issued, and the
            receipts themselves. No payload content, worker identity, or
            storage path is included.
        """
        records = self.store.records()
        receipts = self.store.decisions()
        statuses = Counter(str(record.get("status")) for record in records)
        actions = Counter(str(receipt.get("action")) for receipt in receipts)
        epochs = [
            int(record["recovery"].get("lease_epoch", 0))
            for record in records
            if isinstance(record.get("recovery"), dict)
        ]
        return {
            "schema_version": JOB_RECOVERY_SCHEMA_VERSION,
            "payload_schema_version": JOB_PAYLOAD_SCHEMA_VERSION,
            "status_counts": dict(sorted(statuses.items())),
            "decision_counts": dict(sorted(actions.items())),
            "max_lease_epoch": max(epochs, default=0),
            "receipts": receipts,
        }

    # Operator decisions ----------------------------------------------

    def cancel(
        self,
        job_id: str,
        *,
        actor_digest: str = SYSTEM_ACTOR_DIGEST,
        reason_code: str = "cancel_requested",
    ) -> dict[str, Any]:
        """Cancel a queued, running, or quarantined job durably.

        The cancellation is committed before anything else happens; a running
        worker is fenced and can no longer commit progress or a terminal
        state. No webhook is sent for a cancelled job.

        Args:
            job_id: Job to cancel.
            actor_digest: ``sha256:`` digest attributing the cancellation.
            reason_code: Controlled reason code.

        Returns:
            The cancelled record.

        Raises:
            JobRecoveryError: ``job_not_found`` or ``job_state_conflict`` for a
                job that already finished.
        """
        record = self._cancel(
            job_id,
            reason_code=reason_code,
            actor_digest=actor_digest,
            message="Job was cancelled",
            allowed=frozenset({"queued", "running", "quarantined"}),
        )
        if record is None:
            current = self.store.get(job_id)
            if current is None:
                raise JobRecoveryError("job_not_found")
            if current.get("status") != "cancelled":
                raise JobRecoveryError("job_state_conflict")
            return current
        return record

    def replay(
        self,
        job_id: str,
        authorization: JobReplayAuthorization,
    ) -> dict[str, Any]:
        """Replay a quarantined job as a new, linked execution.

        The quarantine result and every earlier transition are preserved; the
        replay appends a promotion record, a value-free decision receipt, and
        a new execution that resumes after the last committed document with a
        fresh attempt budget.

        Args:
            job_id: Quarantined job.
            authorization: Explicit reviewer and evidence digests.

        Returns:
            The re-queued record.

        Raises:
            ReplayRefusedError: ``replay_unauthorized``,
                ``job_not_quarantined``, or ``replay_input_unavailable``.
            JobRecoveryError: ``job_not_found``.
            RuntimeError: If the queue is shutting down.
        """
        if not isinstance(authorization, JobReplayAuthorization):
            raise ReplayRefusedError("replay_unauthorized")
        if self.replay_authorizer is not None:
            try:
                allowed = bool(self.replay_authorizer(job_id, authorization))
            except Exception:
                allowed = False
            if not allowed:
                raise ReplayRefusedError("replay_unauthorized")
        if self._shutdown:
            raise RuntimeError("Job queue is shutting down")
        self._sweep_payloads()
        record = self.store.get(job_id)
        if record is None:
            raise JobRecoveryError("job_not_found")
        if record.get("status") != "quarantined":
            raise ReplayRefusedError("job_not_quarantined")
        entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
        if (
            entry is None
            or entry.get("state") != PAYLOAD_STATE_RETAINED
            or not self.payloads.contains(entry["ref"])
        ):
            raise ReplayRefusedError("replay_input_unavailable")
        now = self.clock()

        def apply(candidate: dict[str, Any]) -> list[dict[str, Any]]:
            if candidate.get("status") != "quarantined":
                raise ReplayRefusedError("job_not_quarantined")
            recovery = validate_recovery(candidate)
            quarantine = recovery["quarantine"]
            recovery["promotions"].append(
                build_promotion(
                    quarantine_id=quarantine["quarantine_id"],
                    authorization=authorization,
                    promoted_at=now,
                )
            )
            recovery["quarantine"] = None
            recovery["execution"] += 1
            recovery["attempt"] = 0
            recovery["retry"] = None
            recovery["lease_draining"] = False
            candidate["status"] = "queued"
            candidate["error"] = None
            candidate["completed_at"] = None
            self._append_transition(
                candidate, "quarantined", "queued", authorization.reason_code, now
            )
            return [
                decision_receipt(
                    job_id,
                    action="replay_authorized",
                    reason_code=authorization.reason_code,
                    from_status="quarantined",
                    to_status="queued",
                    recovery=recovery,
                    state_version=_state_version(candidate) + 1,
                    at=now,
                    actor_digest=authorization.reviewer_digest,
                    evidence_digest=authorization.evidence_digest,
                )
            ]

        stored = self.store.transition(job_id, apply)
        if stored is None:
            raise JobRecoveryError("job_not_found")
        self._dispatch(job_id, entry["ref"])
        self._ensure_ticker()
        return stored

    # Lifecycle --------------------------------------------------------

    def shutdown(self) -> None:
        """Stop accepting work and settle jobs that never started.

        With memory-only payloads, a job whose work never started is recorded
        as ``cancelled`` and its input payload is deleted, because the payload
        cannot survive the process. With encrypted payloads the job stays
        ``queued`` and resumes at the next start; jobs still running are
        marked as draining, so an interruption by this shutdown is not counted
        against their retry budget. Running and finished jobs keep their own
        states.

        Raises:
            RuntimeError: If a cancelled job's state could not be persisted.
                Every other cancelled job is still recorded first.
        """
        with self._lock:
            self._shutdown = True
        self._ticker_stop.set()
        with self._futures_lock:
            pending = dict(self._futures)
            active = dict(self._active)
        self._executor.shutdown(wait=False, cancel_futures=True)
        durable = self.payloads.mode == PAYLOAD_MODE_ENCRYPTED
        failed = False
        for job_id, future in pending.items():
            if not future.cancelled() or durable:
                continue
            try:
                self._mark_cancelled(job_id)
            except Exception:
                failed = True
        if durable:
            for job_id, epoch in active.items():
                try:
                    self._mark_draining(job_id, epoch)
                except Exception:
                    # Draining only refunds an attempt; the lease still expires.
                    continue
        if failed:
            # Storage errors can name paths; surface a fixed message instead.
            raise RuntimeError("Cancelled job state could not be persisted")

    def reconcile(self, now: Optional[datetime] = None) -> dict[str, int]:
        """Reconcile every job record with leases, payloads, and intents.

        Runs at startup and periodically. It reclaims expired leases (and
        quarantines jobs that exhausted their attempts), renews leases held by
        this queue's live workers, fails queued jobs whose input can no longer
        be read, dispatches queued jobs whose backoff has elapsed, re-drives
        webhook intents whose delivery claim expired, and fails jobs whose
        recovery state is corrupt or ambiguous.

        Args:
            now: Reconcile time; defaults to the queue clock.

        Returns:
            Value-free counts of the actions taken.
        """
        moment = self.clock() if now is None else now
        counts: Counter[str] = Counter()
        self._sweep_payloads()
        for record in self.store.records():
            job_id = str(record.get("id"))
            status = record.get("status")
            if status in TERMINAL_STATUSES:
                self._reconcile_delivery(record, moment, counts)
                continue
            try:
                validate_recovery(record)
            except JobRecoveryError:
                code = (
                    "legacy_job_not_resumable"
                    if "recovery" not in record and status in JOB_STATUSES
                    else "recovery_state_invalid"
                )
                if self._fail_unrecoverable(job_id, code, moment):
                    counts["failed"] += 1
                continue
            if status == "running":
                self._reconcile_running(record, moment, counts)
        for record in self.store.records():
            if record.get("status") == "queued":
                self._reconcile_queued(record, moment, counts)
        return dict(sorted(counts.items()))

    # Worker -------------------------------------------------------------

    def _run_job(self, item: _JobWorkItem) -> None:
        claimed = self._claim(item.job_id)
        if claimed is None:
            return
        lease, record = claimed
        with self._futures_lock:
            self._active[lease.job_id] = lease.epoch
        try:
            self._process(lease, record)
        except StaleLeaseError:
            # Another worker owns the job now; this worker commits nothing.
            return
        finally:
            with self._futures_lock:
                if self._active.get(lease.job_id) == lease.epoch:
                    self._active.pop(lease.job_id, None)

    def _claim(self, job_id: str) -> Optional[tuple[_JobLease, dict[str, Any]]]:
        if self._shutdown:
            return None
        now = self.clock()

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            if record.get("status") != "queued":
                return None
            recovery = validate_recovery(record)
            retry = recovery.get("retry")
            if retry is not None and not is_due(retry.get("retry_after"), now):
                return None
            epoch = recovery["lease_epoch"] + 1
            recovery["attempt"] += 1
            recovery["lease_epoch"] = epoch
            recovery["lease"] = build_lease(
                job_id,
                worker_id=self.worker_id,
                epoch=epoch,
                acquired_at=now,
                lease_seconds=self.lease_seconds,
            )
            recovery["lease_draining"] = False
            record["status"] = "running"
            if record.get("started_at") is None:
                record["started_at"] = _isoformat(now)
            self._append_transition(record, "queued", "running", "claimed", now)
            return []

        try:
            stored = self.store.transition(job_id, apply)
        except JobRecoveryError as exc:
            self._fail_unrecoverable(job_id, exc.code, now)
            return None
        if stored is None:
            return None
        recovery = stored["recovery"]
        entry = _payload_entry(stored, PAYLOAD_KIND_INPUT)
        lease = _JobLease(
            job_id=job_id,
            epoch=int(recovery["lease_epoch"]),
            execution=int(recovery["execution"]),
            input_ref="" if entry is None else str(entry["ref"]),
        )
        return lease, stored

    def _process(self, lease: _JobLease, record: dict[str, Any]) -> None:
        self._fault("claim.committed", lease.job_id)
        entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
        if entry is None or entry.get("state") != PAYLOAD_STATE_RETAINED:
            self._handle_input_failure(lease, _payload_state_code(entry))
            return
        try:
            request = _request_from_body(
                self.payloads.read(
                    lease.input_ref,
                    job_id=lease.job_id,
                    kind=PAYLOAD_KIND_INPUT,
                )
            )
        except EncryptionError as exc:
            self._handle_input_failure(lease, exc.code)
            return
        except (TypeError, ValueError):
            self._handle_input_failure(lease, "payload_invalid")
            return
        if not _request_matches_record(request, record):
            self._handle_input_failure(lease, "input_mismatch")
            return

        recovery = record["recovery"]
        summary = _JobSummary.restore(record)
        error: Optional[dict[str, str]] = recovery.get("document_error")
        checkpoint = recovery.get("checkpoint") or {}
        output_digest: Optional[str] = checkpoint.get("output_digest")
        current = record
        start = int(recovery["completed_documents"])

        for index in range(start, len(request.documents)):
            document = request.documents[index]
            document_id = _document_id(index, document)
            try:
                result = self._deidentify_document(request, document)
                spans = summary.add_result(document_id, result)
                outcome: dict[str, Any] = {
                    "document_id": document_id,
                    "failed": False,
                    "spans": spans,
                }
            except Exception as exc:
                decision = classify_document_failure(exc)
                if decision.action == "retry":
                    self._release_for_retry(lease, decision)
                    return
                summary.failed_count += 1
                error = _safe_error(exc)
                outcome = {"document_id": document_id, "failed": True, "error": error}
            self._fault("document.processed", lease.job_id)
            output_digest = canonical_digest(
                {"previous": output_digest, "sequence": index + 1, "outcome": outcome}
            )
            current = self._commit_checkpoint(
                lease,
                summary,
                sequence=index + 1,
                error=error,
                output_digest=output_digest,
            )
            self._fault("checkpoint.committed", lease.job_id)

        self._finish_job(lease, request, current, summary, error)

    def _fenced(
        self,
        lease: _JobLease,
        mutate: Callable[
            [dict[str, Any], dict[str, Any]], Optional[list[dict[str, Any]]]
        ],
        now: datetime,
    ) -> dict[str, Any]:
        """Commit ``mutate`` only while ``lease`` is the job's live lease."""

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            recovery = record.get("recovery")
            current = recovery.get("lease") if isinstance(recovery, dict) else None
            if (
                record.get("status") != "running"
                or not isinstance(recovery, dict)
                or not isinstance(current, dict)
                or current.get("epoch") != lease.epoch
                or recovery.get("execution") != lease.execution
                or is_due(current.get("expires_at"), now)
            ):
                raise StaleLeaseError()
            return mutate(record, recovery)

        stored = self.store.transition(lease.job_id, apply)
        if stored is None:
            raise StaleLeaseError()
        return stored

    def _commit_checkpoint(
        self,
        lease: _JobLease,
        summary: "_JobSummary",
        *,
        sequence: int,
        error: Optional[dict[str, str]],
        output_digest: str,
    ) -> dict[str, Any]:
        now = self.clock()

        def mutate(
            record: dict[str, Any], recovery: dict[str, Any]
        ) -> list[dict[str, Any]]:
            record.update(summary.to_progress_record())
            recovery["completed_documents"] = sequence
            recovery["document_error"] = error
            recovery["checkpoint"] = build_checkpoint(
                lease.job_id,
                request_digest=recovery["request_digest"],
                input_ref=lease.input_ref,
                sequence=sequence,
                output_digest=output_digest,
                completed_at=now,
                committed_revision=_state_version(record) + 1,
            )
            recovery["lease"] = renew_lease(
                recovery["lease"], now=now, lease_seconds=self.lease_seconds
            )
            return []

        return self._fenced(lease, mutate, now)

    def _release_for_retry(self, lease: _JobLease, decision: FailureDecision) -> None:
        now = self.clock()

        def mutate(
            record: dict[str, Any], recovery: dict[str, Any]
        ) -> list[dict[str, Any]]:
            if recovery["attempt"] >= recovery["max_attempts"]:
                return self._quarantine_locked(
                    record,
                    recovery,
                    now,
                    classification="unsafe",
                    reason_code="attempts_exhausted",
                    from_status="running",
                )
            delay = self.retry_policy.delay_seconds(recovery["attempt"])
            recovery["retry"] = build_retry(
                lease.job_id,
                execution=recovery["execution"],
                attempt=recovery["attempt"],
                classification=decision.classification,
                reason_code=decision.reason_code,
                recorded_at=now,
                retry_after=now + timedelta(seconds=delay),
            )
            recovery["lease"] = None
            record["status"] = "queued"
            self._append_transition(
                record, "running", "queued", decision.reason_code, now
            )
            return []

        self._fenced(lease, mutate, now)

    def _handle_input_failure(self, lease: _JobLease, code: str) -> None:
        decision = classify_input_failure(code)
        if decision.action == "retry":
            self._release_for_retry(lease, decision)
            return
        now = self.clock()
        error = {
            "code": decision.reason_code,
            "type": "JobPayloadError",
            "message": _PAYLOAD_UNREADABLE_MESSAGE,
        }

        def mutate(
            record: dict[str, Any], recovery: dict[str, Any]
        ) -> list[dict[str, Any]]:
            if decision.action == "quarantine":
                return self._quarantine_locked(
                    record,
                    recovery,
                    now,
                    classification=decision.classification,
                    reason_code=decision.reason_code,
                    from_status="running",
                    error=error,
                )
            # The payload is left untouched so the last verified state
            # survives; it still expires on its own deadline.
            record["status"] = "failed"
            record["error"] = error
            record["completed_at"] = _isoformat(now)
            record["expires_at"] = self._expiry_timestamp(now)
            recovery["lease"] = None
            self._append_transition(
                record, "running", "failed", decision.reason_code, now
            )
            return []

        self._fenced(lease, mutate, now)

    def _quarantine_locked(
        self,
        record: dict[str, Any],
        recovery: dict[str, Any],
        now: datetime,
        *,
        classification: str,
        reason_code: str,
        from_status: str,
        error: Optional[dict[str, str]] = None,
    ) -> list[dict[str, Any]]:
        checkpoint = recovery.get("checkpoint") or {}
        quarantine = build_quarantine(
            str(record["id"]),
            request_digest=recovery["request_digest"],
            execution=recovery["execution"],
            classification=classification,
            reason_code=reason_code,
            candidate_count=recovery["completed_documents"],
            failure_count=recovery["attempt"],
            created_at=now,
            output_digest=checkpoint.get("output_digest"),
        )
        recovery["quarantine"] = quarantine
        recovery["quarantines"].append(quarantine)
        recovery["lease"] = None
        recovery["lease_draining"] = False
        record["status"] = "quarantined"
        record["error"] = error or {
            "code": reason_code,
            "type": "JobQuarantined",
            "message": "Job was quarantined until an authorized replay",
        }
        record["expires_at"] = self._expiry_timestamp(now)
        self._append_transition(record, from_status, "quarantined", reason_code, now)
        return [
            decision_receipt(
                str(record["id"]),
                action="quarantined",
                reason_code=reason_code,
                from_status=from_status,
                to_status="quarantined",
                recovery=recovery,
                state_version=_state_version(record) + 1,
                at=now,
            )
        ]

    def _finish_job(
        self,
        lease: _JobLease,
        request: DeidentifyJobRequest,
        current: dict[str, Any],
        summary: "_JobSummary",
        error: Optional[dict[str, str]],
    ) -> None:
        now = self.clock()
        status = "failed" if summary.failed_count else "done"
        final_values = {
            **summary.to_progress_record(),
            "status": status,
            "progress_percent": 100.0,
            "error": error,
            "completed_at": _isoformat(now),
            "expires_at": self._expiry_timestamp(now),
        }
        webhook = request.webhook
        terminal_ref: Optional[PayloadRef] = None
        unavailable_code = WEBHOOK_EVENT_UNAVAILABLE
        if webhook is not None:
            event = {
                "delivery": {
                    "url": webhook.url,
                    "secret": webhook.secret,
                    "max_attempts": webhook.max_attempts,
                    "backoff_seconds": webhook.backoff_seconds,
                },
                "body": webhook_payload({**current, **final_values}),
            }
            try:
                terminal_ref = self.payloads.create(
                    PAYLOAD_KIND_TERMINAL_EVENT,
                    lease.job_id,
                    event,
                    retention_seconds=_record_retention_seconds(
                        current, self.payloads.max_retention_seconds
                    ),
                )
            except EncryptionError as exc:
                unavailable_code = exc.code
        self._fault("terminal.sealed", lease.job_id)

        def mutate(
            record: dict[str, Any], recovery: dict[str, Any]
        ) -> list[dict[str, Any]]:
            record.update(final_values)
            payloads = dict(record.get("payloads") or {})
            input_entry = payloads.get(PAYLOAD_KIND_INPUT)
            if isinstance(input_entry, dict):
                payloads[PAYLOAD_KIND_INPUT] = {
                    **input_entry,
                    "state": PAYLOAD_STATE_DELETED,
                    "reason": "consumed",
                }
            if terminal_ref is not None:
                payloads[PAYLOAD_KIND_TERMINAL_EVENT] = terminal_ref.to_record()
                recovery["webhook"] = {
                    "state": "pending",
                    "payload_ref": terminal_ref.ref,
                    "claim_expires_at": _isoformat(
                        now + timedelta(seconds=self._delivery_window(record))
                    ),
                }
            elif webhook is not None:
                delivery = WebhookDeliveryResult(
                    success=False,
                    attempts=0,
                    status_code=None,
                    error=unavailable_code,
                ).to_dict()
                delivery["payload_ref"] = None
                record["webhook_delivery"] = delivery
                recovery["webhook"] = {"state": "unavailable", "payload_ref": None}
            record["payloads"] = payloads
            recovery["lease"] = None
            self._append_transition(record, "running", status, "completed", now)
            return []

        try:
            self._fenced(lease, mutate, now)
        except Exception:
            # The sealed event was never referenced by a committed record.
            if terminal_ref is not None:
                self._delete_payload(terminal_ref.ref, reason="not_committed")
            raise
        self._fault("terminal.committed", lease.job_id)
        self._delete_payload(lease.input_ref, reason="consumed")
        if terminal_ref is not None:
            self._deliver_terminal_event(lease.job_id, terminal_ref)

    def _deliver_terminal_event(
        self, job_id: str, terminal_ref: PayloadRef | str
    ) -> None:
        ref = terminal_ref.ref if isinstance(terminal_ref, PayloadRef) else terminal_ref
        try:
            with self.payloads.pinned(ref):
                event = self.payloads.read(
                    ref,
                    job_id=job_id,
                    kind=PAYLOAD_KIND_TERMINAL_EVENT,
                )
                result = self._send_webhook(event)
        except EncryptionError as exc:
            result = WebhookDeliveryResult(
                success=False,
                attempts=0,
                status_code=None,
                error=exc.code,
            )
        self._fault("webhook.sent", job_id)
        self._record_delivery(job_id, ref, result)
        self._fault("webhook.recorded", job_id)

    def _record_delivery(
        self,
        job_id: str,
        ref: str,
        result: WebhookDeliveryResult,
        *,
        state: Optional[str] = None,
    ) -> None:
        delivery = result.to_dict()
        delivery["payload_ref"] = ref
        discard = result.success or not self.payloads.retains_failed_deliveries
        reason = "delivered" if result.success else "delivery_failed"
        intent_state = state or ("delivered" if result.success else "failed")

        def apply(record: dict[str, Any]) -> list[dict[str, Any]]:
            record["webhook_delivery"] = delivery
            recovery = record.get("recovery")
            if isinstance(recovery, dict):
                recovery["webhook"] = {"state": intent_state, "payload_ref": ref}
            entry = _payload_entry(record, PAYLOAD_KIND_TERMINAL_EVENT)
            if discard and entry is not None and entry.get("ref") == ref:
                payloads = dict(record["payloads"])
                payloads[PAYLOAD_KIND_TERMINAL_EVENT] = {
                    **entry,
                    "state": PAYLOAD_STATE_DELETED,
                    "reason": reason,
                }
                record["payloads"] = payloads
            return []

        self.store.transition(job_id, apply)
        if discard:
            self._delete_payload(ref, reason=reason)

    def _run_delivery(self, item: _DeliveryWorkItem) -> None:
        try:
            self._deliver_terminal_event(item.job_id, item.payload_ref)
        finally:
            with self._futures_lock:
                self._deliveries.discard(item.job_id)

    def _send_webhook(self, event: Mapping[str, Any]) -> WebhookDeliveryResult:
        delivery = event.get("delivery")
        body = event.get("body")
        if not isinstance(delivery, dict) or not isinstance(body, dict):
            raise JobPayloadError("payload_damaged")
        try:
            url = str(delivery["url"])
            secret = str(delivery["secret"])
            max_attempts = int(delivery["max_attempts"])
            backoff_seconds = float(delivery["backoff_seconds"])
        except (KeyError, TypeError, ValueError):
            raise JobPayloadError("payload_damaged") from None
        try:
            return self.webhook_sender(
                url,
                body,
                secret=secret,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
        except Exception:
            # Sender exceptions can carry endpoints or response content; only a
            # controlled code is recorded.
            return WebhookDeliveryResult(
                success=False,
                attempts=0,
                status_code=None,
                error=WEBHOOK_SENDER_FAILED,
            )

    # Reconcile helpers ------------------------------------------------

    def _reconcile_running(
        self, record: dict[str, Any], now: datetime, counts: Counter[str]
    ) -> None:
        job_id = str(record["id"])
        recovery = record["recovery"]
        lease = recovery["lease"]
        epoch = int(lease["epoch"])
        if not is_due(lease.get("expires_at"), now):
            with self._futures_lock:
                held = self._active.get(job_id) == epoch
            counts["leased"] += 1
            if held and self._renewal_due(lease, now):
                self._renew_lease(job_id, epoch, int(recovery["execution"]), now)
                counts["renewed"] += 1
            return

        def apply(candidate: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            if candidate.get("status") != "running":
                return None
            current_recovery = validate_recovery(candidate)
            current = current_recovery["lease"]
            if current["epoch"] != epoch or not is_due(current["expires_at"], now):
                return None
            draining = bool(current_recovery.get("lease_draining"))
            reason = "interrupted_by_shutdown" if draining else "lease_expired"
            if draining:
                current_recovery["attempt"] = max(0, current_recovery["attempt"] - 1)
            exhausted = (
                not draining
                and current_recovery["attempt"] >= current_recovery["max_attempts"]
            )
            receipts = [
                decision_receipt(
                    job_id,
                    action="lease_reclaimed",
                    reason_code=reason,
                    from_status="running",
                    to_status="quarantined" if exhausted else "queued",
                    recovery=current_recovery,
                    state_version=_state_version(candidate) + 1,
                    at=now,
                )
            ]
            if exhausted:
                return receipts + self._quarantine_locked(
                    candidate,
                    current_recovery,
                    now,
                    classification="unsafe",
                    reason_code="attempts_exhausted",
                    from_status="running",
                )
            delay = (
                0
                if draining
                else self.retry_policy.delay_seconds(current_recovery["attempt"])
            )
            current_recovery["retry"] = build_retry(
                job_id,
                execution=current_recovery["execution"],
                attempt=max(1, current_recovery["attempt"]),
                classification="cancelled" if draining else "unknown",
                reason_code=reason,
                recorded_at=now,
                retry_after=now + timedelta(seconds=delay),
            )
            current_recovery["lease"] = None
            current_recovery["lease_draining"] = False
            candidate["status"] = "queued"
            self._append_transition(candidate, "running", "queued", reason, now)
            return receipts

        try:
            stored = self.store.transition(job_id, apply)
        except JobRecoveryError as exc:
            if self._fail_unrecoverable(job_id, exc.code, now):
                counts["failed"] += 1
            return
        if stored is None:
            return
        counts["reclaimed"] += 1
        if stored.get("status") == "quarantined":
            counts["quarantined"] += 1

    def _reconcile_queued(
        self, record: dict[str, Any], now: datetime, counts: Counter[str]
    ) -> None:
        job_id = str(record["id"])
        entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
        if entry is None or entry.get("state") != PAYLOAD_STATE_RETAINED:
            if self._fail_queued(job_id, _payload_state_code(entry), now):
                counts["failed"] += 1
            return
        if is_due(entry.get("expires_at"), now):
            if self._fail_queued(
                job_id,
                "payload_expired",
                now,
                entry_state=PAYLOAD_STATE_EXPIRED,
                entry_reason="retention_elapsed",
            ):
                counts["failed"] += 1
            return
        if self.payloads.mode == PAYLOAD_MODE_MEMORY and not self.payloads.contains(
            entry["ref"]
        ):
            # Memory-only payloads do not survive a restart.
            if self._fail_queued(
                job_id,
                "input_unavailable",
                now,
                entry_state=PAYLOAD_STATE_DELETED,
                entry_reason="process_restarted",
            ):
                counts["failed"] += 1
            return
        retry = record["recovery"].get("retry")
        if retry is not None and not is_due(retry.get("retry_after"), now):
            counts["waiting"] += 1
            return
        if self._dispatch(job_id, entry["ref"]):
            counts["dispatched"] += 1

    def _reconcile_delivery(
        self, record: dict[str, Any], now: datetime, counts: Counter[str]
    ) -> None:
        recovery = record.get("recovery")
        intent = recovery.get("webhook") if isinstance(recovery, dict) else None
        if not isinstance(intent, dict) or intent.get("state") != "pending":
            return
        if not is_due(intent.get("claim_expires_at"), now):
            return
        job_id = str(record["id"])
        ref = intent.get("payload_ref")
        with self._futures_lock:
            if job_id in self._deliveries:
                return
        window = self._delivery_window(record)

        def apply(candidate: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            current_recovery = candidate.get("recovery")
            current = (
                current_recovery.get("webhook")
                if isinstance(current_recovery, dict)
                else None
            )
            if (
                not isinstance(current, dict)
                or current.get("state") != "pending"
                or current.get("payload_ref") != ref
                or not is_due(current.get("claim_expires_at"), now)
            ):
                return None
            current["claim_expires_at"] = _isoformat(now + timedelta(seconds=window))
            return []

        if not isinstance(ref, str) or self.store.transition(job_id, apply) is None:
            return
        if not self.payloads.contains(ref):
            self._record_delivery(
                job_id,
                ref,
                WebhookDeliveryResult(
                    success=False,
                    attempts=0,
                    status_code=None,
                    error=WEBHOOK_EVENT_UNAVAILABLE,
                ),
                state="unavailable",
            )
            counts["deliveries_unavailable"] += 1
            return
        with self._futures_lock:
            if self._shutdown or job_id in self._deliveries:
                return
            self._deliveries.add(job_id)
            try:
                self._executor.submit(
                    self._run_delivery, _DeliveryWorkItem(job_id, ref)
                )
            except RuntimeError:
                self._deliveries.discard(job_id)
                return
        counts["deliveries_redriven"] += 1

    def _fail_queued(
        self,
        job_id: str,
        code: str,
        now: datetime,
        *,
        entry_state: Optional[str] = None,
        entry_reason: Optional[str] = None,
    ) -> bool:
        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            if record.get("status") != "queued":
                return None
            recovery = validate_recovery(record)
            record["status"] = "failed"
            record["error"] = {
                "code": code,
                "type": "JobPayloadError",
                "message": _PAYLOAD_UNREADABLE_MESSAGE,
            }
            record["completed_at"] = _isoformat(now)
            record["expires_at"] = self._expiry_timestamp(now)
            entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
            if entry is not None and entry_state is not None:
                payloads = dict(record["payloads"])
                payloads[PAYLOAD_KIND_INPUT] = {
                    **entry,
                    "state": entry_state,
                    "reason": entry_reason,
                }
                record["payloads"] = payloads
            self._append_transition(record, "queued", "failed", code, now)
            return [
                decision_receipt(
                    job_id,
                    action="recovery_failed",
                    reason_code=code,
                    from_status="queued",
                    to_status="failed",
                    recovery=recovery,
                    state_version=_state_version(record) + 1,
                    at=now,
                )
            ]

        try:
            return self.store.transition(job_id, apply) is not None
        except JobRecoveryError:
            return self._fail_unrecoverable(job_id, "recovery_state_invalid", now)

    def _fail_unrecoverable(self, job_id: str, code: str, now: datetime) -> bool:
        """Fail a job whose recovery state cannot be trusted, closed."""
        input_ref: list[str] = []

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            status = record.get("status")
            if status in TERMINAL_STATUSES:
                return None
            from_status = status if status in JOB_STATUSES else "unknown"
            record["status"] = "failed"
            record["error"] = {
                "code": code,
                "type": "JobRecoveryError",
                "message": "Job recovery state could not be trusted",
            }
            record["completed_at"] = _isoformat(now)
            record["expires_at"] = self._expiry_timestamp(now)
            entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
            if entry is not None and entry.get("state") == PAYLOAD_STATE_RETAINED:
                input_ref.append(str(entry["ref"]))
                payloads = dict(record["payloads"])
                payloads[PAYLOAD_KIND_INPUT] = {
                    **entry,
                    "state": PAYLOAD_STATE_DELETED,
                    "reason": "recovery_failed",
                }
                record["payloads"] = payloads
            recovery = record.get("recovery")
            if isinstance(recovery, dict):
                recovery["lease"] = None
            return [
                decision_receipt(
                    job_id,
                    action="recovery_failed",
                    reason_code=code,
                    from_status=str(from_status),
                    to_status="failed",
                    recovery={},
                    state_version=_state_version(record) + 1,
                    at=now,
                )
            ]

        stored = self.store.transition(job_id, apply)
        for ref in input_ref:
            self._delete_payload(ref, reason="recovery_failed")
        return stored is not None

    def _renewal_due(self, lease: Mapping[str, Any], now: datetime) -> bool:
        expires = parse_time(lease.get("expires_at"))
        if expires is None:
            return True
        return (expires - now).total_seconds() < self.lease_seconds / 2

    def _renew_lease(
        self, job_id: str, epoch: int, execution: int, now: datetime
    ) -> None:
        lease = _JobLease(job_id=job_id, epoch=epoch, execution=execution, input_ref="")

        def mutate(
            record: dict[str, Any], recovery: dict[str, Any]
        ) -> list[dict[str, Any]]:
            recovery["lease"] = renew_lease(
                recovery["lease"], now=now, lease_seconds=self.lease_seconds
            )
            return []

        try:
            self._fenced(lease, mutate, now)
        except StaleLeaseError:
            return

    def _mark_draining(self, job_id: str, epoch: int) -> None:
        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            recovery = record.get("recovery")
            lease = recovery.get("lease") if isinstance(recovery, dict) else None
            if (
                record.get("status") != "running"
                or not isinstance(lease, dict)
                or lease.get("epoch") != epoch
            ):
                return None
            assert isinstance(recovery, dict)
            recovery["lease_draining"] = True
            return []

        self.store.transition(job_id, apply)

    def _mark_cancelled(self, job_id: str) -> None:
        self._cancel(
            job_id,
            reason_code="cancelled_on_shutdown",
            actor_digest=SYSTEM_ACTOR_DIGEST,
            message="Job was cancelled before it started",
            allowed=frozenset({"queued"}),
        )

    def _cancel(
        self,
        job_id: str,
        *,
        reason_code: str,
        actor_digest: str,
        message: str,
        allowed: frozenset[str],
    ) -> Optional[dict[str, Any]]:
        now = self.clock()
        cancellation = build_cancellation(
            job_id,
            actor_digest=actor_digest,
            reason_code=reason_code,
            requested_at=now,
        )
        input_ref: list[str] = []

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            status = record.get("status")
            if status not in allowed:
                return None
            record["status"] = "cancelled"
            record["error"] = {
                "code": reason_code,
                "type": "CancelledError",
                "message": message,
            }
            record["completed_at"] = _isoformat(now)
            record["expires_at"] = self._expiry_timestamp(now)
            entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
            if entry is not None and entry.get("state") == PAYLOAD_STATE_RETAINED:
                input_ref.append(str(entry["ref"]))
                payloads = dict(record["payloads"])
                payloads[PAYLOAD_KIND_INPUT] = {
                    **entry,
                    "state": PAYLOAD_STATE_DELETED,
                    "reason": "cancelled",
                }
                record["payloads"] = payloads
            recovery = record.get("recovery")
            if not isinstance(recovery, dict):
                return []
            recovery["lease"] = None
            recovery["lease_draining"] = False
            recovery["quarantine"] = None
            recovery["cancellation"] = cancellation
            self._append_transition(record, str(status), "cancelled", reason_code, now)
            return [
                decision_receipt(
                    job_id,
                    action="cancelled",
                    reason_code=reason_code,
                    from_status=str(status),
                    to_status="cancelled",
                    recovery=recovery,
                    state_version=_state_version(record) + 1,
                    at=now,
                    actor_digest=actor_digest,
                )
            ]

        stored = self.store.transition(job_id, apply)
        for ref in input_ref:
            self._delete_payload(ref, reason="cancelled")
        return stored

    # Shared helpers ---------------------------------------------------

    def _dispatch(self, job_id: str, input_ref: str) -> bool:
        with self._futures_lock:
            if self._shutdown or job_id in self._futures:
                return False
            try:
                future = self._executor.submit(
                    self._run_job, _JobWorkItem(job_id, input_ref)
                )
            except RuntimeError:
                return False
            self._futures[job_id] = future
        future.add_done_callback(functools.partial(self._forget_future, job_id))
        return True

    def _forget_future(self, job_id: str, future: Future[None]) -> None:
        with self._futures_lock:
            if self._futures.get(job_id) is future:
                self._futures.pop(job_id, None)

    def _ensure_ticker(self) -> None:
        interval = self.reconcile_interval_seconds
        if interval is None:
            return
        with self._futures_lock:
            if self._ticker is not None or self._shutdown:
                return
            thread = threading.Thread(
                target=_reconcile_loop,
                args=(weakref.ref(self), self._ticker_stop, float(interval)),
                name="openmed-job-reconcile",
                daemon=True,
            )
            self._ticker = thread
        thread.start()

    def _fault(self, boundary: str, job_id: str) -> None:
        if self.fault_hook is not None:
            assert boundary in FAULT_BOUNDARIES
            self.fault_hook(boundary, job_id)

    def _append_transition(
        self,
        record: dict[str, Any],
        from_status: Optional[str],
        to_status: str,
        reason_code: str,
        now: datetime,
    ) -> None:
        recovery = record["recovery"]
        recovery["transitions"].append(
            transition_entry(
                state_version=_state_version(record) + 1,
                from_status=from_status,
                to_status=to_status,
                reason_code=reason_code,
                at=now,
                recovery=recovery,
            )
        )

    def _delivery_window(self, record: Mapping[str, Any]) -> int:
        webhook = record.get("webhook")
        attempts = 1
        backoff = 0.0
        if isinstance(webhook, dict):
            try:
                attempts = max(1, int(webhook.get("max_attempts") or 1))
                backoff = max(0.0, float(webhook.get("backoff_seconds") or 0.0))
            except (TypeError, ValueError):
                pass
        waits = backoff * ((2 ** (attempts - 1)) - 1)
        return self.lease_seconds + math.ceil(
            attempts * DEFAULT_WEBHOOK_TIMEOUT_SECONDS + waits
        )

    def _payload_retention_seconds(self, payload: DeidentifyJobRequest) -> int:
        maximum = self.payloads.max_retention_seconds
        requested = getattr(payload, "payload_retention_seconds", None)
        if requested is None:
            return maximum
        if requested > maximum:
            raise ValueError(
                "payload_retention_seconds exceeds the configured service maximum"
            )
        return int(requested)

    def _expiry_timestamp(self, moment: datetime) -> str:
        """Return the metadata expiry for a record that reached *moment*."""
        expires_at = datetime.fromtimestamp(
            moment.timestamp() + self.store.ttl_seconds,
            tz=timezone.utc,
        )
        return _isoformat(expires_at)

    def _new_record(
        self,
        payload: DeidentifyJobRequest,
        *,
        job_id: str,
        now: datetime,
        input_ref: PayloadRef,
        retention_seconds: int,
        request_digest: str,
        key_digest: Optional[str],
    ) -> dict[str, Any]:
        documents = [
            _document_metadata(index, document)
            for index, document in enumerate(payload.documents)
        ]
        recovery = new_recovery(
            max_attempts=self.retry_policy.max_attempts,
            request_digest=request_digest,
        )
        recovery["transitions"].append(
            transition_entry(
                state_version=1,
                from_status=None,
                to_status="queued",
                reason_code="admitted",
                at=now,
                recovery=recovery,
            )
        )
        return {
            "id": job_id,
            "status": "queued",
            "state_version": 1,
            "progress_percent": 0.0,
            "document_count": len(payload.documents),
            "processed_count": 0,
            "failed_count": 0,
            "label_histogram": {},
            "spans": [],
            "documents": documents,
            "error": None,
            "webhook": _webhook_metadata(payload.webhook),
            "webhook_delivery": None,
            "payloads": {
                "schema_version": JOB_PAYLOAD_SCHEMA_VERSION,
                "mode": self.payloads.mode,
                "retention_seconds": retention_seconds,
                PAYLOAD_KIND_INPUT: input_ref.to_record(),
                PAYLOAD_KIND_TERMINAL_EVENT: None,
            },
            "recovery": recovery,
            "idempotency": (None if key_digest is None else {"key_digest": key_digest}),
            "created_at": _isoformat(now),
            "updated_at": _isoformat(now),
            "started_at": None,
            "completed_at": None,
            "expires_at": self._expiry_timestamp(now),
        }

    def _delete_payload(self, ref: str, *, reason: str) -> None:
        if not ref:
            return
        self.payloads.delete(ref, reason=reason)

    def _set_payload_entry(
        self,
        job_id: str,
        kind: str,
        **changes: Any,
    ) -> None:
        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            payloads = record.get("payloads")
            if not isinstance(payloads, dict):
                return None
            current = payloads.get(kind)
            if not isinstance(current, dict) or not current:
                return None
            if all(current.get(name) == value for name, value in changes.items()):
                return None
            updated = dict(payloads)
            updated[kind] = {**current, **changes}
            record["payloads"] = updated
            return []

        self.store.transition(job_id, apply)

    def _sweep_payloads(self) -> None:
        for expired in self.payloads.sweep_expired():
            self._set_payload_entry(
                expired.job_id,
                expired.kind,
                state=PAYLOAD_STATE_EXPIRED,
                reason="retention_elapsed",
            )

    def _live_payload_refs(self) -> set[str]:
        live: set[str] = set()
        for record in self.store.records():
            for kind in PAYLOAD_KINDS:
                entry = _payload_entry(record, kind)
                if entry is not None and entry.get("state") == PAYLOAD_STATE_RETAINED:
                    live.add(str(entry.get("ref")))
        return live

    def _deidentify_document(
        self,
        payload: DeidentifyJobRequest,
        document: DeidentifyJobDocument,
    ) -> Any:
        def _operation() -> Any:
            return openmed.deidentify(
                document.text,
                method=payload.method,
                model_name=payload.model_name,
                confidence_threshold=payload.confidence_threshold,
                keep_year=payload.keep_year,
                shift_dates=payload.shift_dates,
                date_shift_days=payload.date_shift_days,
                keep_mapping=payload.keep_mapping,
                config=self.runtime.config,
                use_smart_merging=payload.use_smart_merging,
                use_safety_sweep=payload.use_safety_sweep,
                lang=payload.lang,
                normalize_accents=payload.normalize_accents,
                loader=self.runtime.get_loader(),
                policy=payload.policy,
            )

        return self.runtime.run_model_request(
            payload.model_name,
            payload.keep_alive,
            _operation,
        )


def _reconcile_loop(
    queue_ref: "weakref.ReferenceType[DeidentifyJobQueue]",
    stop: threading.Event,
    interval: float,
) -> None:
    while not stop.wait(interval):
        queue = queue_ref()
        if queue is None:
            return
        try:
            queue.reconcile()
        except Exception:
            # Store and payload errors can name paths; log a fixed message.
            logger.warning("Async job reconcile pass failed")
        del queue


@dataclass
class _JobSummary:
    total: int
    processed_count: int = 0
    failed_count: int = 0
    label_histogram: dict[str, int] = field(default_factory=dict)
    spans: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def restore(cls, record: Mapping[str, Any]) -> "_JobSummary":
        """Rebuild the committed summary from a checkpointed record."""
        return cls(
            total=int(record.get("document_count") or 0),
            processed_count=int(record.get("processed_count") or 0),
            failed_count=int(record.get("failed_count") or 0),
            label_histogram=dict(record.get("label_histogram") or {}),
            spans=list(record.get("spans") or []),
        )

    def add_result(self, document_id: str, result: Any) -> list[dict[str, Any]]:
        """Add one de-identification result without retaining raw text."""
        labels: dict[str, int] = {}
        spans: list[dict[str, Any]] = []
        for entity in getattr(result, "pii_entities", []) or []:
            label = _entity_label(entity)
            spans.append(_entity_span(document_id, entity, label))
            labels[label] = labels.get(label, 0) + 1
        self.processed_count += 1
        for label, count in labels.items():
            self.label_histogram[label] = self.label_histogram.get(label, 0) + count
        self.spans.extend(spans)
        return spans

    def to_progress_record(self) -> dict[str, Any]:
        attempted = self.processed_count + self.failed_count
        progress = (
            100.0 if self.total == 0 else round((attempted / self.total) * 100, 2)
        )
        return {
            "progress_percent": progress,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "label_histogram": dict(sorted(self.label_histogram.items())),
            "spans": list(self.spans),
        }


def _document_id(index: int, document: DeidentifyJobDocument) -> str:
    return document.id or str(index)


def _document_metadata(index: int, document: DeidentifyJobDocument) -> dict[str, Any]:
    return {
        "id": _document_id(index, document),
        "length": len(document.text),
        "text_hash": hash_text(document.text),
    }


def _webhook_metadata(webhook: Optional[JobWebhookRequest]) -> Optional[dict[str, Any]]:
    if webhook is None:
        return None
    return {
        "configured": True,
        "url_hash": hash_text(webhook.url),
        "max_attempts": webhook.max_attempts,
        "backoff_seconds": webhook.backoff_seconds,
    }


def _request_digest(payload: DeidentifyJobRequest) -> str:
    """Return the immutable request digest over values metadata already holds.

    The digest covers the execution parameters, the document metadata, and the
    webhook metadata. The date shift is reduced to whether it is set, so the
    digest never encodes a value that metadata does not already expose.
    """
    body = _request_body(payload)
    params = {
        name: value
        for name, value in body.items()
        if name not in {"documents", "webhook", "date_shift_days"}
    }
    params["date_shift_days_set"] = body.get("date_shift_days") is not None
    return canonical_digest(
        {
            "schema_version": JOB_RECOVERY_SCHEMA_VERSION,
            "params": params,
            "documents": [
                _document_metadata(index, document)
                for index, document in enumerate(payload.documents)
            ],
            "webhook": _webhook_metadata(payload.webhook),
        }
    )


def _request_matches_record(
    request: DeidentifyJobRequest, record: Mapping[str, Any]
) -> bool:
    documents = [
        _document_metadata(index, document)
        for index, document in enumerate(request.documents)
    ]
    return (
        record.get("document_count") == len(request.documents)
        and record.get("documents") == documents
    )


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_state_code(entry: Optional[Mapping[str, Any]]) -> str:
    state = None if entry is None else entry.get("state")
    if state == PAYLOAD_STATE_EXPIRED:
        return "payload_expired"
    if state == PAYLOAD_STATE_DELETED:
        return "payload_deleted"
    return "payload_missing"


def _entity_label(entity: Any) -> str:
    label = getattr(entity, "canonical_label", None) or getattr(entity, "label", None)
    return str(label or "UNKNOWN")


def _entity_span(document_id: str, entity: Any, label: str) -> dict[str, Any]:
    text = str(getattr(entity, "text", "") or "")
    return {
        "document_id": document_id,
        "start": int(getattr(entity, "start", 0) or 0),
        "end": int(getattr(entity, "end", 0) or 0),
        "label": label,
        "text_hash": hash_text(text),
        "confidence": _optional_float(getattr(entity, "confidence", None)),
    }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_error(exc: Exception) -> dict[str, str]:
    return {
        "code": "document_failed",
        "type": exc.__class__.__name__,
        "message": "Document failed during de-identification",
    }


def _request_body(payload: DeidentifyJobRequest) -> dict[str, Any]:
    """Return a detached JSON-mode copy of a job request."""
    dump = getattr(payload, "model_dump", None)
    if callable(dump):
        body = dump(mode="json")
    else:  # pragma: no cover - pydantic v1 fallback
        body = json.loads(payload.json())
    return json.loads(json.dumps(body, allow_nan=False))


def _request_from_body(body: Mapping[str, Any]) -> DeidentifyJobRequest:
    return DeidentifyJobRequest(**dict(body))


def _payload_entry(record: Mapping[str, Any], kind: str) -> Optional[dict[str, Any]]:
    payloads = record.get("payloads")
    if not isinstance(payloads, dict):
        return None
    entry = payloads.get(kind)
    if not isinstance(entry, dict) or not isinstance(entry.get("ref"), str):
        return None
    return entry


def _record_retention_seconds(record: Mapping[str, Any], maximum: int) -> int:
    payloads = record.get("payloads")
    retention = (
        payloads.get("retention_seconds") if isinstance(payloads, dict) else None
    )
    if isinstance(retention, bool) or not isinstance(retention, int):
        return maximum
    return max(1, min(retention, maximum))


def webhook_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build a no-PHI terminal webhook payload from a persisted record."""
    return {
        "event": f"job.{record['status']}",
        "job_id": record["id"],
        "status": record["status"],
        "progress_percent": record["progress_percent"],
        "document_count": record["document_count"],
        "processed_count": record["processed_count"],
        "failed_count": record["failed_count"],
        "label_histogram": record["label_histogram"],
        "spans": record["spans"],
        "documents": record["documents"],
        "error": record["error"],
        "completed_at": record["completed_at"],
    }


def job_response_payload(
    record: dict[str, Any],
    *,
    status_url: Optional[str] = None,
) -> dict[str, Any]:
    """Return the public REST representation of a job record.

    Recovery metadata is reduced to its public, value-free projection: lease
    and worker identifiers, the request digest, and the idempotency key digest
    are omitted.
    """
    payload = _copy_record(record)
    payload.pop("idempotency", None)
    payload["recovery"] = public_recovery_view(payload.pop("recovery", None))
    if status_url is not None:
        payload["status_url"] = status_url
    return payload
