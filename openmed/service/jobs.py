"""Async de-identification jobs for the OpenMed REST service.

Job metadata (status, counts, keyed digests, offsets, opaque payload
references, and value-free recovery state) is persisted by
:class:`LocalJobStore`.  PHI-bearing content never enters that metadata: each
submission is snapshotted once into an immutable input payload, and each
terminal webhook event is sealed once into an immutable terminal-event
payload, both held by a :class:`~openmed.service.job_payloads.JobPayloadStore`
under an explicit, bounded retention deadline.

Job records and terminal webhook payloads never carry submitted text. Span,
document and webhook URL references are keyed, domain-separated HMAC-SHA256
digests under the deployment digest key (see :mod:`.keyed_digest`), so a
low-entropy identifier cannot be recovered by hashing candidate values. Outside
the ``dev`` and ``test`` profiles, a service without a configured key omits
these digests and records an explicit marker instead of hashing without a key.

Jobs are durable.  The input payload reference, an immutable request digest,
and an optional keyed idempotency digest are persisted before a job reaches
the worker pool.  Workers claim a job with an expiring lease whose epoch
fences every state write, commit checkpoints as documents complete, and
commit the terminal state together with the webhook intent.  Failures are
separated into document failures, bounded retries with backoff, and poison
quarantine; a quarantined job runs again only after an authorized replay.  A
reconcile pass at startup (and periodically while work is pending) reclaims
expired leases, resumes queued work, quarantines poison work, and re-drives
pending webhook intents through the bounded webhook dispatcher.  The recovery
records are the Journey ingestion contracts, adapted in
:mod:`openmed.service.job_recovery`.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hmac
import inspect
import json
import logging
import math
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
import weakref
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Final, NamedTuple, Optional, Sequence

import openmed
from openmed.clinical.journey_contracts import canonical_digest
from openmed.core.config import PROFILE_ENV_VAR
from openmed.core.key_provider import EncryptionError, KeyProvider

from .compatibility import (
    CURRENT_CONTRACT,
    JOB_DEIDENTIFY_CAPABILITY,
    JOB_STORE_COMPONENT,
    JOB_STORE_SCHEMA_VERSION,
    JOB_WEBHOOK_CAPABILITY,
    CompatibilityDescriptor,
    CompatibilityState,
    ComponentCompatibility,
    IncompatibleStateError,
    SchemaObservation,
    WorkEnvelope,
    check_work,
    evaluate_component,
    job_store_observation,
    local_descriptor,
    model_generation_ref,
)
from .drain import SHUTTING_DOWN, ServiceShuttingDownError
from .idempotency import (
    IdempotencyError,
    IdempotentStartRequest,
    parse_idempotency_key,
)
from .job_admission import (
    DEFAULT_MAX_DOCUMENTS_PER_JOB,
    DEFAULT_MAX_PENDING_JOBS,
    DEFAULT_MAX_QUEUED_TEXT_BYTES,
    JobAdmission,
    JobAdmissionLimits,
    JobCost,
    JobReservation,
)
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
    is_due,
    new_recovery,
    parse_time,
    public_recovery_view,
    renew_lease,
    transition_entry,
    validate_recovery,
)
from .keyed_digest import (
    DEVELOPMENT_PROFILES,
    KEYED_DIGEST_ALGORITHM,
    ServiceDigestKey,
    encode_digest_fields,
    resolve_service_digest_key,
)
from .runtime import ServiceRuntime
from .schemas import DeidentifyJobDocument, DeidentifyJobRequest, JobWebhookRequest
from .signing import new_delivery_id, normalize_delivery_id
from .webhooks import (
    DEFAULT_WEBHOOK_DRAIN_SECONDS,
    DEFAULT_WEBHOOK_MAX_DELIVERY_SECONDS,
    DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    DEFAULT_WEBHOOK_WORKERS,
    WEBHOOK_ERROR_CANCELLED,
    WEBHOOK_ERROR_SENDER,
    WebhookDeliveryResult,
    WebhookDispatcher,
    authorize_webhook_destination,
    deliver_webhook,
    pending_delivery_metadata,
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
SERVICE_JOBS_MAX_PENDING_ENV_VAR = "OPENMED_SERVICE_JOBS_MAX_PENDING"
SERVICE_JOBS_MAX_QUEUED_BYTES_ENV_VAR = "OPENMED_SERVICE_JOBS_MAX_QUEUED_BYTES"
SERVICE_JOBS_MAX_DOCUMENTS_ENV_VAR = "OPENMED_SERVICE_JOBS_MAX_DOCUMENTS"
SERVICE_JOBS_LEASE_ENV_VAR = "OPENMED_SERVICE_JOBS_LEASE_SECONDS"
SERVICE_JOBS_MAX_ATTEMPTS_ENV_VAR = "OPENMED_SERVICE_JOBS_MAX_ATTEMPTS"
SERVICE_JOBS_RETRY_BASE_ENV_VAR = "OPENMED_SERVICE_JOBS_RETRY_BASE_SECONDS"
SERVICE_JOBS_RETRY_MAX_ENV_VAR = "OPENMED_SERVICE_JOBS_RETRY_MAX_SECONDS"
SERVICE_JOBS_RECONCILE_ENV_VAR = "OPENMED_SERVICE_JOBS_RECONCILE_SECONDS"
DEFAULT_JOBS_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOBS_WORKERS = 2
DEFAULT_PROGRESS_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_PROGRESS_WRITES = 100
PAYLOAD_MODES = frozenset({PAYLOAD_MODE_MEMORY, PAYLOAD_MODE_ENCRYPTED})

# Sender exceptions are recorded with the dispatcher's controlled code.
WEBHOOK_SENDER_FAILED = WEBHOOK_ERROR_SENDER
WEBHOOK_EVENT_UNAVAILABLE = "terminal_event_unavailable"
# JOB_STORE_SCHEMA_VERSION comes from openmed.service.compatibility, which
# documents each version and the window of versions this build reads. The
# optional ``decisions`` array holds append-only, value-free recovery
# receipts; it is written only once a receipt exists.
DECISIONS_FIELD: Final = "decisions"
_JOB_STORE_FIELDS = frozenset({DECISIONS_FIELD, "jobs", "schema_version"})
OWNER_DIGEST_FIELD = "owner_digest"
DIGESTS_FIELD: Final = "digests"
TEXT_DIGEST_FIELD: Final = "text_digest"
URL_DIGEST_FIELD: Final = "url_digest"
SPAN_TEXT_DIGEST_DOMAIN: Final = "job-span-text"
DOCUMENT_TEXT_DIGEST_DOMAIN: Final = "job-document-text"
WEBHOOK_URL_DIGEST_DOMAIN: Final = "job-webhook-url"
DIGEST_MODE_KEYED: Final = "keyed"
DIGEST_MODE_OMITTED: Final = "omitted"
DIGEST_KEY_UNCONFIGURED: Final = "digest_key_unconfigured"
LEGACY_DIGESTS_REMOVED: Final = "legacy_digests_removed"
RECOVERY_FIELD: Final = "recovery"
IDEMPOTENCY_FIELD: Final = "idempotency"
#: Fixed route label bound into ``Idempotency-Key`` scopes for ``POST /jobs``.
JOBS_IDEMPOTENCY_ROUTE: Final = "jobs-deidentify"
_LEGACY_TEXT_HASH_FIELD: Final = "text_hash"
_LEGACY_URL_HASH_FIELD: Final = "url_hash"
ACTIVE_STATUSES = frozenset({"queued", "running"})
_SETTLED_STATUSES = TERMINAL_STATUSES | {"quarantined"}
_PAYLOAD_UNREADABLE_MESSAGE = "Job input payload could not be read"
_REASON_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")

# Phases a job can hold in this process while it is outstanding.
_PHASE_QUEUED: Final = "queued"
_PHASE_RUNNING: Final = "running"
_PHASE_CALLBACK: Final = "callback"

WebhookSender = Callable[..., WebhookDeliveryResult]
FaultHook = Callable[[str, str], None]
ReplayAuthorizer = Callable[[str, JobReplayAuthorization], bool]
#: Guard-and-mutate callback for :meth:`LocalJobStore.update_if`. It receives a
#: private copy of the current record, may edit nested state, and returns the
#: decision receipts to append, or ``None`` to leave the record unchanged.
StoreMutation = Callable[[dict[str, Any]], Optional[Sequence[Mapping[str, Any]]]]
# Maps each job id touched by one publication to its prior record, or None
# when the id did not exist before. Used to roll back a failed publication.
_RecordSnapshots = dict[str, Optional[dict[str, Any]]]


class JobStoreSchemaError(ValueError):
    """Value-free refusal of a job metadata document this binary cannot read.

    Attributes:
        code: ``job_store_schema_unsupported`` for a newer or invalid version,
            or ``job_store_unknown_fields`` for fields this binary would
            otherwise drop on its next write.
        compatibility_reason: Value-free compatibility-window reason, such as
            ``schema_ahead_of_window``, when the version itself was readable.
    """

    def __init__(self, code: str, *, compatibility_reason: Optional[str] = None):
        self.code = code
        self.compatibility_reason = compatibility_reason
        super().__init__(f"Job metadata store cannot be read ({code})")


class JobQueueClosedError(ServiceShuttingDownError):
    """Raised when a job is submitted after the queue stopped admitting work."""

    def __init__(self) -> None:
        super().__init__("Job queue is shutting down")


@dataclass(frozen=True)
class _JobWorkItem:
    """Queued work: the job and the opaque reference of its input snapshot."""

    job_id: str
    input_ref: str


@dataclass(frozen=True)
class _JobLease:
    """In-memory handle for one claimed lease; ``epoch`` is the fencing token.

    ``attempt`` and ``max_attempts`` are the values committed with the claim;
    they cannot change while the lease is held.
    """

    job_id: str
    epoch: int
    execution: int
    input_ref: str
    attempt: int = 1
    max_attempts: int = DEFAULT_MAX_ATTEMPTS


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


def parse_job_admission_limits(
    *,
    max_pending: Optional[str] = None,
    max_queued_bytes: Optional[str] = None,
    max_documents: Optional[str] = None,
) -> JobAdmissionLimits:
    """Parse async job submission ceilings from raw environment values.

    Args:
        max_pending: Raw ``OPENMED_SERVICE_JOBS_MAX_PENDING`` value.
        max_queued_bytes: Raw ``OPENMED_SERVICE_JOBS_MAX_QUEUED_BYTES`` value.
        max_documents: Raw ``OPENMED_SERVICE_JOBS_MAX_DOCUMENTS`` value.

    Returns:
        Admission limits, using safe defaults for unset or blank values.

    Raises:
        ValueError: If a set value is not a positive integer.
    """
    return JobAdmissionLimits(
        max_pending_jobs=_parse_positive_int(
            max_pending,
            env_var=SERVICE_JOBS_MAX_PENDING_ENV_VAR,
            default=DEFAULT_MAX_PENDING_JOBS,
        ),
        max_queued_text_bytes=_parse_positive_int(
            max_queued_bytes,
            env_var=SERVICE_JOBS_MAX_QUEUED_BYTES_ENV_VAR,
            default=DEFAULT_MAX_QUEUED_TEXT_BYTES,
        ),
        max_documents_per_job=_parse_positive_int(
            max_documents,
            env_var=SERVICE_JOBS_MAX_DOCUMENTS_ENV_VAR,
            default=DEFAULT_MAX_DOCUMENTS_PER_JOB,
        ),
    )


def job_admission_limits_from_env() -> JobAdmissionLimits:
    """Return async job submission ceilings from the environment."""
    return parse_job_admission_limits(
        max_pending=os.getenv(SERVICE_JOBS_MAX_PENDING_ENV_VAR),
        max_queued_bytes=os.getenv(SERVICE_JOBS_MAX_QUEUED_BYTES_ENV_VAR),
        max_documents=os.getenv(SERVICE_JOBS_MAX_DOCUMENTS_ENV_VAR),
    )


def _copy_record(record: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(record)


def _state_version(record: Mapping[str, Any]) -> int:
    value = record.get("state_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _digest_descriptor(
    mode: str,
    *,
    key_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> dict[str, Optional[str]]:
    return {
        "mode": mode,
        "alg": KEYED_DIGEST_ALGORITHM if mode == DIGEST_MODE_KEYED else None,
        "key_id": key_id,
        "reason": reason,
    }


@dataclass(frozen=True, eq=False)
class JobDigestPolicy:
    """Keyed digests for span, document and webhook URL references.

    Each reference is an HMAC-SHA256 digest of the value under the deployment
    digest key, in a purpose-specific domain, so a span digest never equals a
    document or URL digest of the same text. Digests are stable for one key,
    change when the key rotates, and carry ``alg`` and ``key_id`` markers.

    Attributes:
        key: Deployment digest key, or ``None`` when digests are omitted
            because no durable key is configured outside ``dev`` and ``test``.
    """

    key: Optional[ServiceDigestKey] = None

    @classmethod
    def for_key(cls, key: ServiceDigestKey, *, profile: str) -> "JobDigestPolicy":
        """Return a policy for *key*, omitting digests where that is required.

        Args:
            key: Resolved service digest key.
            profile: Active service profile name.

        Returns:
            A keyed policy, or an omitting policy when *key* is a per-process
            ephemeral key outside the ``dev`` and ``test`` profiles.
        """
        if key.ephemeral and profile not in DEVELOPMENT_PROFILES:
            return cls(key=None)
        return cls(key=key)

    @classmethod
    def from_env(
        cls,
        *,
        key: Optional[ServiceDigestKey] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "JobDigestPolicy":
        """Build the policy from service environment settings.

        Args:
            key: Already-resolved service digest key, such as the job
                ownership key, so one deployment key serves every digest.
                When omitted it is resolved from ``OPENMED_SERVICE_DIGEST_KEY``.
            environ: Environment mapping; defaults to the process environment.

        Returns:
            The configured digest policy.

        Raises:
            DigestKeyError: If a configured digest key is invalid.
        """
        env = os.environ if environ is None else environ
        profile = (env.get(PROFILE_ENV_VAR) or "prod").strip()
        if key is None:
            key = resolve_service_digest_key(
                profile=profile,
                required=False,
                environ=env,
            )
        return cls.for_key(key, profile=profile)

    @property
    def keyed(self) -> bool:
        """Whether this policy emits digests."""
        return self.key is not None

    def descriptor(self) -> dict[str, Optional[str]]:
        """Return the record-level marker describing how digests were made.

        Returns:
            ``mode`` is ``keyed`` with ``alg`` and ``key_id``, or ``omitted``
            with the value-free ``reason`` ``digest_key_unconfigured``.
        """
        if self.key is None:
            return _digest_descriptor(
                DIGEST_MODE_OMITTED, reason=DIGEST_KEY_UNCONFIGURED
            )
        return _digest_descriptor(DIGEST_MODE_KEYED, key_id=self.key.key_id)

    def digest(self, domain: str, value: str) -> Optional[dict[str, str]]:
        """Return the keyed digest of *value* within *domain*, if keyed.

        Args:
            domain: Purpose label such as ``job-span-text``.
            value: Text to digest; it is never stored.

        Returns:
            A JSON-safe ``{"alg", "key_id", "value"}`` mapping, or ``None``
            when digests are omitted.
        """
        if self.key is None:
            return None
        return self.key.digest(domain, encode_digest_fields(value)).to_dict()


def _migrate_record(record: dict[str, Any]) -> bool:
    """Bring a stored record to the current schema in place.

    Records written before schema version 3 carry unkeyed SHA-256 hashes that
    can be reversed by enumeration. They cannot be re-keyed without the source
    text, so they are removed and the record is marked accordingly. Records
    written before durable recovery carry no ``recovery`` block; one is never
    invented, so such a record that is still in flight fails closed at the
    next reconcile instead of resuming from unverified state.

    Returns:
        Whether the record changed.
    """
    changed = False
    if OWNER_DIGEST_FIELD not in record:
        # Version 1 records predate ownership; mark them explicitly unowned.
        record[OWNER_DIGEST_FIELD] = None
        changed = True
    for collection in ("documents", "spans"):
        items = record.get(collection)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                changed |= _replace_legacy_hash(
                    item, _LEGACY_TEXT_HASH_FIELD, TEXT_DIGEST_FIELD
                )
    webhook = record.get("webhook")
    if isinstance(webhook, dict):
        changed |= _replace_legacy_hash(
            webhook, _LEGACY_URL_HASH_FIELD, URL_DIGEST_FIELD
        )
    if DIGESTS_FIELD not in record:
        record[DIGESTS_FIELD] = _digest_descriptor(
            DIGEST_MODE_OMITTED, reason=LEGACY_DIGESTS_REMOVED
        )
        changed = True
    if record.get("schema_version") != JOB_STORE_SCHEMA_VERSION:
        record["schema_version"] = JOB_STORE_SCHEMA_VERSION
        changed = True
    return changed


def _replace_legacy_hash(item: dict[str, Any], legacy: str, current: str) -> bool:
    changed = False
    if legacy in item:
        del item[legacy]
        changed = True
    if current not in item:
        item[current] = None
        changed = True
    return changed


_FileSignature = tuple[int, int, int]


def _file_signature(path: Path) -> Optional[_FileSignature]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


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
    """Internal: another live record already holds the idempotency scope."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("duplicate idempotency key")
        self.record = record


class LocalJobStore:
    """Durable local metadata store for async jobs.

    The store intentionally persists only metadata and no raw submitted text or
    redacted output. Job payloads are held by a separate
    :class:`~openmed.service.job_payloads.JobPayloadStore` and appear here only
    as opaque references with their lifecycle state and retention deadline.
    Stores written by older schema versions are migrated when loaded and
    rewritten at once, so removed unkeyed hashes do not linger on disk.

    Every write replaces the whole file atomically, so one write commits a
    state transition and its decision receipts together. Lifecycle writes go
    through :meth:`update_if`, the store's single fenced write path. Stores
    opened on the same path in one process share a lock and reload the file
    when another store replaced it, so a fenced write always sees the latest
    committed state. The store serves one service process; it does not
    coordinate writers across processes.

    The file records its schema version. A store outside this build's
    compatibility window is refused when it is opened, and if another process
    replaces the file with one outside the window while this store is open,
    every later operation raises :class:`IncompatibleStateError` before any
    mutation.
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
        self._schema = CURRENT_CONTRACT.durable_schemas[JOB_STORE_COMPONENT]
        self._blocked: Optional[ComponentCompatibility] = None
        self._published_signature: Optional[_FileSignature] = None
        with self._lock:
            records, decisions, migrated = self._load()
            self._records: dict[str, dict[str, Any]] = records
            self._decisions: list[dict[str, Any]] = decisions
            if migrated:
                self._persist_locked()

    @property
    def compatibility(self) -> ComponentCompatibility:
        """Return the value-free compatibility of the persisted store."""
        with self._lock:
            if self._blocked is not None:
                return self._blocked
            if self._published_signature is None:
                return evaluate_component(self._schema, SchemaObservation.absent())
            return evaluate_component(
                self._schema, SchemaObservation.at(JOB_STORE_SCHEMA_VERSION)
            )

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist a new job record.

        Raises:
            _DuplicateIdempotencyKey: Internal signal that a live record
                already holds the same idempotency scope digest.
        """
        with self._lock:
            self._begin_locked()
            scope = _record_idempotency_scope(record)
            if scope is not None:
                existing = self._find_idempotent_locked(scope)
                if existing is not None:
                    raise _DuplicateIdempotencyKey(_copy_record(existing))
            previous = self._remove_expired_locked()
            job_id = str(record["id"])
            previous.setdefault(job_id, self._records.get(job_id))
            self._records[job_id] = _copy_record(record)
            self._publish_locked(previous)
            return _copy_record(self._records[job_id])

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return one job record, if present."""
        with self._lock:
            self._begin_locked()
            self.cleanup_expired_locked()
            record = self._records.get(job_id)
            return None if record is None else _copy_record(record)

    def records(self) -> list[dict[str, Any]]:
        """Return copies of every unexpired job record."""
        with self._lock:
            self._begin_locked()
            self.cleanup_expired_locked()
            return [_copy_record(record) for record in self._records.values()]

    def find_idempotent(self, scope_digest: str) -> Optional[dict[str, Any]]:
        """Return the live record holding an idempotency scope digest, if any."""
        with self._lock:
            self._begin_locked()
            record = self._find_idempotent_locked(scope_digest)
            return None if record is None else _copy_record(record)

    def decisions(self) -> list[dict[str, Any]]:
        """Return the append-only decision receipts, oldest first."""
        with self._lock:
            self._begin_locked()
            return [_copy_record(receipt) for receipt in self._decisions]

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        """Apply changes to a job record and persist them."""
        with self._lock:
            self._begin_locked()
            current = self._records[job_id]
            # Build the new record from a shallow copy so the current record
            # stays untouched and serves as the rollback snapshot. Changes only
            # replace top-level values, and stored records are never mutated in
            # place, so no other record needs a snapshot.
            record = dict(current)
            record.update(_copy_record(changes))
            record["updated_at"] = _isoformat(self.clock())
            record["state_version"] = _state_version(current) + 1
            self._records[job_id] = record
            self._publish_locked({job_id: current})
            return _copy_record(record)

    def update_if(
        self,
        job_id: str,
        *,
        expected_statuses: Optional[Collection[str]],
        mutate: Optional[StoreMutation] = None,
        **changes: Any,
    ) -> Optional[dict[str, Any]]:
        """Apply changes only while the record passes its fence.

        This is the commit fence for every job lifecycle transition: once a
        drain marks a job ``cancelled``, or a reclaim moves a job to a new
        lease epoch, a stale worker's later write is refused.

        ``mutate`` receives a private copy of the current record before
        ``changes`` are applied. It checks further preconditions (such as the
        lease epoch), edits nested state such as the recovery block, and
        returns the decision receipts to append in the same atomic write, or
        ``None`` to decline the change. Exceptions it raises propagate and
        nothing is written. ``changes`` then replace top-level fields.

        Args:
            job_id: Record identifier.
            expected_statuses: Statuses the record must currently have, or
                ``None`` to leave the status unfenced.
            mutate: Optional guard-and-mutate callback.
            **changes: Top-level fields to replace.

        Returns:
            The updated record, or ``None`` when a fence refused the change.
        """
        with self._lock:
            self._begin_locked()
            current = self._records.get(job_id)
            if current is None or (
                expected_statuses is not None
                and current.get("status") not in expected_statuses
            ):
                return None
            # Copy-on-write: the current record is the rollback snapshot and is
            # never mutated in place.
            if mutate is None:
                record = dict(current)
                receipts: Sequence[Mapping[str, Any]] = ()
            else:
                record = _copy_record(current)
                outcome = mutate(record)
                if outcome is None:
                    return None
                receipts = outcome
            record.update(_copy_record(changes))
            record["updated_at"] = _isoformat(self.clock())
            record["state_version"] = _state_version(current) + 1
            self._records[job_id] = record
            previous_decisions = len(self._decisions)
            self._decisions.extend(_copy_record(dict(item)) for item in receipts)
            try:
                self._publish_locked({job_id: current})
            except BaseException:
                del self._decisions[previous_decisions:]
                raise
            return _copy_record(record)

    def cleanup_expired(self) -> None:
        """Remove terminal and quarantined records whose TTL has elapsed."""
        with self._lock:
            self._begin_locked()
            previous = self._remove_expired_locked()
            if previous:
                self._publish_locked(previous)

    def cleanup_expired_locked(self) -> bool:
        """Drop expired records from memory without persisting."""
        return bool(self._remove_expired_locked())

    def _remove_expired_locked(self) -> _RecordSnapshots:
        now = self.clock()
        expired = [
            job_id
            for job_id, record in self._records.items()
            if record.get("status") in EXPIRING_STATUSES
            and _parse_timestamp(record.get("expires_at")) <= now
        ]
        return {job_id: self._records.pop(job_id) for job_id in expired}

    def _find_idempotent_locked(self, scope_digest: str) -> Optional[dict[str, Any]]:
        now = self.clock()
        for record in self._records.values():
            held = _record_idempotency_scope(record)
            if held is None or not hmac.compare_digest(held, scope_digest):
                continue
            if (
                record.get("status") in EXPIRING_STATUSES
                and _parse_timestamp(record.get("expires_at")) <= now
            ):
                continue
            return record
        return None

    def _load(
        self,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], bool]:
        if not self.path.exists():
            return {}, [], False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError("Job metadata store contains invalid JSON") from None
        if not isinstance(raw, dict):
            raise ValueError("Job metadata store must contain a JSON object")
        # Refuse before any write: a newer document, or one with fields this
        # binary does not know, would otherwise be silently rewritten.
        version = raw.get("schema_version", 1)
        if type(version) is not int:
            raise JobStoreSchemaError("job_store_schema_unsupported")
        if not 1 <= version <= JOB_STORE_SCHEMA_VERSION:
            window = evaluate_component(self._schema, SchemaObservation.at(version))
            raise JobStoreSchemaError(
                "job_store_schema_unsupported", compatibility_reason=window.code
            )
        if set(raw) - _JOB_STORE_FIELDS:
            raise JobStoreSchemaError("job_store_unknown_fields")
        jobs = raw.get("jobs")
        if not isinstance(jobs, dict):
            raise ValueError("Job metadata store must contain a jobs object")
        if any(not isinstance(record, dict) for record in jobs.values()):
            raise ValueError("Job metadata store contains an invalid job record")
        decisions = raw.get(DECISIONS_FIELD, [])
        if not isinstance(decisions, list) or any(
            not isinstance(receipt, dict) for receipt in decisions
        ):
            raise ValueError("Job metadata store contains an invalid decision receipt")
        records = {str(job_id): dict(record) for job_id, record in jobs.items()}
        migrated = version != JOB_STORE_SCHEMA_VERSION
        for record in records.values():
            migrated |= _migrate_record(record)
        self._published_signature = _file_signature(self.path)
        return records, [dict(receipt) for receipt in decisions], migrated

    def _begin_locked(self) -> None:
        """Refuse a blocked store, then pick up another store's newer write."""
        self._require_compatible_locked()
        self._refresh_locked()

    def _require_compatible_locked(self) -> None:
        if self._blocked is not None:
            raise IncompatibleStateError(JOB_STORE_COMPONENT, self._blocked.code)

    def _refresh_locked(self) -> None:
        """Reload the file when another store or build replaced it.

        A replacement outside the compatibility window blocks this store
        before anything is loaded. A compatible replacement is loaded and
        migrated in memory, so the next fenced write sees the latest committed
        state rather than overwriting it.
        """
        signature = _file_signature(self.path)
        if signature is None or signature == self._published_signature:
            return
        self._verify_on_disk_schema_locked()
        self._records, self._decisions, _ = self._load()

    def _verify_on_disk_schema_locked(self) -> None:
        """Refuse to replace a file another build rewrote outside the window."""
        signature = _file_signature(self.path)
        if signature is None or signature == self._published_signature:
            return
        compatibility = evaluate_component(
            self._schema, job_store_observation(self.path)
        )
        if compatibility.usable:
            return
        self._blocked = compatibility
        raise IncompatibleStateError(JOB_STORE_COMPONENT, compatibility.code)

    def _persist_locked(self) -> None:
        self._verify_on_disk_schema_locked()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "jobs": self._records,
            "schema_version": JOB_STORE_SCHEMA_VERSION,
        }
        if self._decisions:
            payload[DECISIONS_FIELD] = self._decisions
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
            self._published_signature = _file_signature(self.path)
        except BaseException:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            raise

    def _publish_locked(self, previous: _RecordSnapshots) -> None:
        try:
            self._persist_locked()
        except BaseException:
            for job_id, record in previous.items():
                if record is None:
                    self._records.pop(job_id, None)
                else:
                    self._records[job_id] = record
            raise


def _record_idempotency_scope(record: Mapping[str, Any]) -> Optional[str]:
    idempotency = record.get(IDEMPOTENCY_FIELD)
    if not isinstance(idempotency, dict):
        return None
    digest = idempotency.get("scope_digest")
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
    and an optional keyed idempotency digest are persisted before the job is
    handed to a worker. Submissions pass
    :class:`~openmed.service.job_admission.JobAdmission` before any payload or
    metadata is written; the reservation is released when the job reaches a
    terminal or quarantined state in this process.

    A worker claims a job with an expiring lease. The lease epoch is the
    fencing token: every checkpoint, retry, quarantine, and terminal write
    goes through :meth:`LocalJobStore.update_if` and checks it, so a worker
    that lost its lease, or whose job a drain or an operator cancelled, can
    never commit.

    Per-document progress is coalesced into checkpoints: a checkpoint is
    written only after ``progress_interval_seconds`` have elapsed since the
    previous write and after at least ``1 / max_progress_writes`` of the job's
    documents have completed since then, or sooner when the lease needs
    renewing. A document's output counts as done only once a checkpoint
    covering it is committed, so a restarted job resumes after the last
    committed checkpoint and re-runs any document after it. The terminal
    state commits the final checkpoint together with the webhook intent.

    Terminal webhooks are delivered on a separate bounded
    :class:`~openmed.service.webhooks.WebhookDispatcher`, so job workers return
    as soon as the terminal record is committed. Delivery reads the immutable
    terminal-event payload instead of rebuilding the body from mutable job
    state, and the reconcile pass re-drives an intent whose delivery never
    settled.

    Every record is stamped with a :class:`WorkEnvelope` naming the contract
    and model generation it was admitted under, and a worker checks that
    envelope with :func:`check_work` before it claims the job.

    Args:
        runtime: Service runtime that executes model requests.
        store: Local metadata store for job records.
        max_workers: Maximum number of concurrently running jobs.
        webhook_sender: Callable that delivers one signed webhook. It is called
            like :func:`~openmed.service.webhooks.deliver_webhook` with ``url``,
            ``payload``, ``secret``, ``max_attempts``, ``backoff_seconds``, and
            the stable ``delivery_id`` sealed with the terminal event.
        clock: Wall-clock source for record timestamps and lease deadlines.
        progress_interval_seconds: Minimum seconds between two progress
            checkpoints of one job. ``0`` disables the time limit.
        max_progress_writes: Maximum number of coalesced progress checkpoints
            per job; lease renewals may add one write per half lease.
        monotonic: Monotonic seconds source used to space checkpoints.
        webhook_workers: Maximum number of concurrent webhook deliveries.
        webhook_max_delivery_seconds: Server-side ceiling on the total wall
            time of one webhook delivery, including retries and backoff.
        webhook_drain_seconds: Time :meth:`shutdown` waits for in-flight
            deliveries before recording them as abandoned.
        payload_store: Payload store; defaults to a memory-only store.
        digests: Digest policy for job metadata; defaults to the environment
            policy.
        admission: Submission admission control; defaults to the built-in
            ceilings.
        descriptor: Compatibility descriptor of this node.
        retry_policy: Retry budget and backoff; defaults to
            :class:`~openmed.service.job_recovery.RetryPolicy`.
        lease_seconds: Lease duration. Workers renew the lease as they
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

    Raises:
        ValueError: If ``progress_interval_seconds`` is negative,
            ``max_progress_writes`` is less than one, ``lease_seconds`` is not
            a positive integer, or ``reconcile_interval_seconds`` is not
            positive.
    """

    def __init__(
        self,
        runtime: ServiceRuntime,
        *,
        store: LocalJobStore,
        max_workers: int = DEFAULT_JOBS_WORKERS,
        webhook_sender: WebhookSender = deliver_webhook,
        clock: Callable[[], datetime] = _utc_now,
        progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
        max_progress_writes: int = DEFAULT_MAX_PROGRESS_WRITES,
        monotonic: Callable[[], float] = time.monotonic,
        webhook_workers: int = DEFAULT_WEBHOOK_WORKERS,
        webhook_max_delivery_seconds: float = DEFAULT_WEBHOOK_MAX_DELIVERY_SECONDS,
        webhook_drain_seconds: float = DEFAULT_WEBHOOK_DRAIN_SECONDS,
        payload_store: Optional[JobPayloadStore] = None,
        digests: Optional[JobDigestPolicy] = None,
        admission: Optional[JobAdmission] = None,
        descriptor: Optional[CompatibilityDescriptor] = None,
        retry_policy: Optional[RetryPolicy] = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        reconcile_interval_seconds: Optional[float] = None,
        replay_authorizer: Optional[ReplayAuthorizer] = None,
        fault_hook: Optional[FaultHook] = None,
    ) -> None:
        if not progress_interval_seconds >= 0:
            raise ValueError("progress_interval_seconds must be non-negative")
        if int(max_progress_writes) < 1:
            raise ValueError("max_progress_writes must be a positive integer")
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
        self.progress_interval_seconds = float(progress_interval_seconds)
        self.max_progress_writes = int(max_progress_writes)
        self.monotonic = monotonic
        self._webhooks = WebhookDispatcher(
            max_workers=webhook_workers,
            max_delivery_seconds=webhook_max_delivery_seconds,
            drain_seconds=webhook_drain_seconds,
        )
        self.payloads = (
            MemoryJobPayloadStore(clock=clock)
            if payload_store is None
            else payload_store
        )
        self.digests = JobDigestPolicy.from_env() if digests is None else digests
        self.admission = (
            JobAdmission(metrics=getattr(runtime, "metrics", None))
            if admission is None
            else admission
        )
        self.descriptor = (
            descriptor if descriptor is not None else local_descriptor(runtime)
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
        # Re-entrant: submission holds it while it dispatches work.
        self._lock = threading.RLock()
        self._futures_lock = threading.Lock()
        self._futures: dict[str, Future[None]] = {}
        self._admission_closed = False
        self._cancel_requested = threading.Event()
        self._callback_timeout_seconds: Optional[float] = None
        # Jobs outstanding in this process, by phase.
        self._active: dict[str, str] = {}
        # Lease epochs held by this process's live workers.
        self._leases: dict[str, int] = {}
        self._reservations: dict[str, JobReservation] = {}
        self._deliveries: set[str] = set()
        self._idempotency_key: Optional[ServiceDigestKey] = None
        self._drain_completed = 0
        self._drain_cancelled = 0
        self._drain_checkpointed = 0
        self._ticker_stop = threading.Event()
        self._ticker: Optional[threading.Thread] = None
        self.payloads.recover(self._live_payload_refs())
        counts = self.reconcile()
        if any(
            counts.get(name)
            for name in ("dispatched", "waiting", "leased", "deliveries_redriven")
        ):
            self._ensure_ticker()

    @classmethod
    def from_env(
        cls,
        runtime: ServiceRuntime,
        *,
        key_provider: Optional[KeyProvider] = None,
        digests: Optional[JobDigestPolicy] = None,
    ) -> "DeidentifyJobQueue":
        """Create a queue from service environment variables.

        Args:
            runtime: Service runtime used to run model requests.
            key_provider: Optional caller-owned key provider that enables
                encrypted payload retention, and with it restart-safe resume.
            digests: Keyed digest policy shared with the rest of the service;
                resolved from the environment when omitted.

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
            digests=digests,
            admission=JobAdmission(
                job_admission_limits_from_env(),
                metrics=getattr(runtime, "metrics", None),
            ),
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
        owner_digest: Optional[dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Submit one de-identification job and return its initial metadata.

        See :meth:`submit_job`, which also reports whether an idempotent
        resubmission returned an existing job.

        Args:
            payload: The validated request. It is snapshotted immediately; the
                caller may reuse or mutate it afterwards.
            owner_digest: Keyed digest of the submitting principal, or ``None``
                when authentication is disabled.
            idempotency_key: Optional client ``Idempotency-Key``.

        Returns:
            The job record (the existing one for an idempotent resubmission).
        """
        return self.submit_job(
            payload,
            owner_digest=owner_digest,
            idempotency_key=idempotency_key,
        ).record

    def submit_job(
        self,
        payload: DeidentifyJobRequest,
        *,
        owner_digest: Optional[dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> JobSubmission:
        """Submit one job and report whether it created a new logical job.

        A repeated idempotency key is resolved before admission, so a replay
        never consumes capacity. Otherwise capacity is reserved before the
        input payload or record is created; a rejected submission persists no
        record and queues no work. The input payload, its reference, the
        request digest, and the keyed idempotency digest are persisted before
        the job is handed to a worker.

        A webhook URL is checked against the egress policy before the job is
        accepted, so a refused callback is rejected at submission.

        Args:
            payload: The validated request; it is snapshotted immediately.
            owner_digest: Keyed digest of the submitting principal, or ``None``
                when authentication is disabled. Idempotency keys are scoped
                to it.
            idempotency_key: Optional client key in the ``Idempotency-Key``
                format. It is stored only as a keyed scope digest; across a
                restart it is honoured only under a durable digest key.

        Returns:
            The record and ``created``; ``created`` is ``False`` when the same
            key and request returned the existing job.

        Raises:
            ValueError: If the requested payload retention exceeds the
                configured maximum or the model is not served.
            IdempotencyError: ``idempotency_key_invalid`` for a malformed key
                or ``idempotency_key_conflict`` when the key was used for a
                different request.
            JobLimitExceededError: If the job exceeds a per-job ceiling.
            JobQueueFullError: If the pending-job queue is at capacity.
            JobQueueClosedError: If the queue is shutting down.
            EncryptionError: If the input payload cannot be retained.
            EgressDenied: When the egress policy refuses the webhook URL.
        """
        snapshot = DeidentifyJobRequest(**_request_body(payload))
        self.runtime.validate_served_model(snapshot.model_name)
        if snapshot.webhook is not None:
            authorize_webhook_destination(snapshot.webhook.url)
        self._payload_retention_seconds(snapshot)
        identity = (
            None
            if idempotency_key is None
            else self._idempotency_identity(
                snapshot, idempotency_key, owner_digest=owner_digest
            )
        )
        cost = JobCost.for_texts(document.text for document in snapshot.documents)
        with self._lock:
            if self._shutdown or self._admission_closed:
                raise JobQueueClosedError()
            self._sweep_payloads()
            if identity is not None:
                existing = self.store.find_idempotent(identity.scope_digest)
                if existing is not None:
                    return self._idempotent_existing(existing, identity)
            reservation = self.admission.reserve(cost)
            try:
                record, item = self._stage(
                    snapshot, owner_digest=owner_digest, idempotency=identity
                )
                try:
                    stored = self.store.create(record)
                except _DuplicateIdempotencyKey as duplicate:
                    self._delete_payload(item.input_ref, reason="duplicate_submission")
                    assert identity is not None
                    reservation.release()
                    return self._idempotent_existing(duplicate.record, identity)
                except BaseException:
                    self.payloads.delete(item.input_ref, reason="submission_failed")
                    raise
            except BaseException:
                reservation.release()
                raise
            # The durable record now owns the reservation: it is released on
            # the job's terminal or quarantine transition in this process.
            self._reservations[item.job_id] = reservation
            self._active[item.job_id] = _PHASE_QUEUED
            self._fault("submit.persisted", item.job_id)
            self._dispatch(item.job_id, item.input_ref)
        self._ensure_ticker()
        return JobSubmission(_copy_record(stored), True)

    def _stage(
        self,
        payload: DeidentifyJobRequest,
        *,
        owner_digest: Optional[dict[str, str]] = None,
        idempotency: Optional[IdempotentStartRequest] = None,
    ) -> tuple[dict[str, Any], _JobWorkItem]:
        """Snapshot *payload* into an input payload and build its queued record.

        The record is not stored and no worker is scheduled; :meth:`submit`
        does both.
        """
        snapshot = DeidentifyJobRequest(**_request_body(payload))
        retention = self._payload_retention_seconds(snapshot)
        now = self.clock()
        job_id = uuid.uuid4().hex
        input_ref = self.payloads.create(
            PAYLOAD_KIND_INPUT,
            job_id,
            _request_body(snapshot),
            retention_seconds=retention,
            now=now,
        )
        record = self._new_record(
            snapshot,
            job_id=job_id,
            now=now,
            input_ref=input_ref,
            retention_seconds=retention,
            owner_digest=owner_digest,
            idempotency=idempotency,
        )
        return record, _JobWorkItem(job_id, input_ref.ref)

    def _idempotency_identity(
        self,
        snapshot: DeidentifyJobRequest,
        idempotency_key: str,
        *,
        owner_digest: Optional[Mapping[str, str]],
    ) -> IdempotentStartRequest:
        """Derive the keyed scope and request digests of a keyed submission.

        The scope binds the ``/jobs`` route label, the submitter's owner digest
        and the key; the request digest covers every request field except the
        webhook secret, so a retry with a rotated secret still matches. Both
        are keyed HMAC digests, so neither the key nor any field is retained.
        """
        key = parse_idempotency_key([idempotency_key])
        assert key is not None
        return IdempotentStartRequest.derive(
            self._idempotency_digest_key(),
            route=JOBS_IDEMPOTENCY_ROUTE,
            owner_digest=owner_digest,
            idempotency_key=key,
            request_fields=_idempotency_fields(snapshot),
        )

    def _idempotency_digest_key(self) -> ServiceDigestKey:
        if self.digests.key is not None:
            return self.digests.key
        # Without a durable digest key, idempotency holds for this process only.
        with self._lock:
            if self._idempotency_key is None:
                self._idempotency_key = ServiceDigestKey.generate()
            return self._idempotency_key

    def _idempotent_existing(
        self,
        existing: dict[str, Any],
        identity: IdempotentStartRequest,
    ) -> JobSubmission:
        stored = existing.get(IDEMPOTENCY_FIELD)
        digest = stored.get("request_digest") if isinstance(stored, dict) else None
        if not isinstance(digest, str) or not hmac.compare_digest(
            digest, identity.request_digest
        ):
            raise IdempotencyError("idempotency_key_conflict")
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
            int(record[RECOVERY_FIELD].get("lease_epoch", 0))
            for record in records
            if isinstance(record.get(RECOVERY_FIELD), dict)
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
        record = self._cancel_record(
            job_id,
            expected=frozenset({"queued", "running", "quarantined"}),
            error={
                "code": reason_code,
                "type": "CancelledError",
                "message": "Job was cancelled",
            },
            actor_digest=actor_digest,
        )
        if record is not None:
            return record
        current = self.store.get(job_id)
        if current is None:
            raise JobRecoveryError("job_not_found")
        if current.get("status") != "cancelled":
            raise JobRecoveryError("job_state_conflict")
        return current

    def replay(
        self,
        job_id: str,
        authorization: JobReplayAuthorization,
    ) -> dict[str, Any]:
        """Replay a quarantined job as a new, linked execution.

        The quarantine result and every earlier transition are preserved; the
        replay appends a promotion record, a value-free decision receipt, and
        a new execution that resumes after the last committed checkpoint with
        a fresh attempt budget. An operator replay is not subject to
        submission admission limits.

        Args:
            job_id: Quarantined job.
            authorization: Explicit reviewer and evidence digests.

        Returns:
            The re-queued record.

        Raises:
            ReplayRefusedError: ``replay_unauthorized``,
                ``job_not_quarantined``, or ``replay_input_unavailable``.
            JobRecoveryError: ``job_not_found``.
            JobQueueClosedError: If the queue is shutting down.
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
        with self._lock:
            if self._shutdown or self._admission_closed:
                raise JobQueueClosedError()
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

        stored = self.store.update_if(
            job_id,
            expected_statuses=("quarantined",),
            mutate=apply,
            status="queued",
            error=None,
            completed_at=None,
        )
        if stored is None:
            raise ReplayRefusedError("job_not_quarantined")
        self._dispatch(job_id, entry["ref"])
        self._ensure_ticker()
        return stored

    # Lifecycle and drain ---------------------------------------------

    def shutdown(self, drain_seconds: Optional[float] = None) -> None:
        """Stop accepting work, settle jobs that never started, and webhooks.

        With memory-only payloads, a job whose work never started is recorded
        as ``cancelled`` and its input payload is deleted, because the payload
        cannot survive the process. With encrypted payloads the job stays
        ``queued`` and resumes at the next start, and jobs still running are
        marked as draining, so an interruption by this shutdown is not counted
        against their retry budget. Running and finished jobs keep their own
        states. Job workers are released without waiting. Webhook backoff
        waits are cancelled, and deliveries that do not settle within the
        drain bound are recorded as ``abandoned`` with error ``cancelled``;
        with encrypted payloads their intent stays pending for the next start.

        Args:
            drain_seconds: Override for the configured webhook drain bound.

        Raises:
            RuntimeError: If a cancelled job's state could not be persisted.
                Every other cancelled job is still recorded first, and webhook
                deliveries are still settled.
        """
        with self._lock:
            self._shutdown = True
            active = dict(self._active)
            leases = dict(self._leases)
        self._ticker_stop.set()
        with self._futures_lock:
            pending = dict(self._futures)
        self._executor.shutdown(wait=False, cancel_futures=True)
        durable = self._durable_payloads
        unstarted = {
            job_id for job_id, future in pending.items() if future.cancelled()
        } | {
            job_id
            for job_id, phase in active.items()
            if phase == _PHASE_QUEUED and job_id not in pending
        }
        failed = False
        for job_id in sorted(unstarted):
            if durable:
                self._hand_off(job_id)
                continue
            try:
                self._mark_cancelled(job_id)
            except Exception:
                failed = True
            # The job leaves this process either way.
            self._release_reservation(job_id)
        if durable:
            for job_id, epoch in leases.items():
                try:
                    self._mark_draining(job_id, epoch)
                except Exception:
                    # Draining only refunds an attempt; the lease still expires.
                    continue
        self._webhooks.shutdown(drain_seconds)
        if failed:
            # Storage errors can name paths; surface a fixed message instead.
            raise RuntimeError("Cancelled job state could not be persisted")

    def close_admission(
        self, *, callback_timeout_seconds: Optional[float] = None
    ) -> None:
        """Refuse new jobs and bound terminal callbacks sent while draining.

        Args:
            callback_timeout_seconds: Per-attempt timeout for callbacks; while
                draining each terminal callback makes a single attempt.
        """
        with self._lock:
            self._admission_closed = True
            self._callback_timeout_seconds = callback_timeout_seconds

    def request_cancel(self) -> None:
        """Ask workers to stop at their next document (checkpoint) boundary.

        With memory-only payloads a stopped job is ``cancelled`` with a
        retryable ``shutting_down`` error. With encrypted payloads it is
        checkpointed and left ``queued``, so the next start resumes it.
        """
        self._cancel_requested.set()

    def cancel_outstanding(self) -> int:
        """Fence every queued or running job so no stale worker can commit.

        With memory-only payloads each job is moved to ``cancelled``. With
        encrypted payloads, running jobs lose their lease and return to
        ``queued`` with their last checkpoint, and queued jobs stay queued for
        the next start; either way a worker still inside a model call can no
        longer commit its result.

        Returns:
            Jobs moved to ``cancelled`` by this call.
        """
        return self._cancel_jobs(phases=(_PHASE_QUEUED, _PHASE_RUNNING))

    def outstanding_count(self) -> int:
        """Return jobs that are queued, running, or sending their callback."""
        with self._lock:
            return len(self._active)

    def drain_stats(self) -> dict[str, int]:
        """Return value-free terminal counts recorded since admission closed."""
        with self._lock:
            return {
                "completed": self._drain_completed,
                "cancelled": self._drain_cancelled,
                "checkpointed": self._drain_checkpointed,
            }

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
                    if RECOVERY_FIELD not in record and status in JOB_STATUSES
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

    def _cancel_jobs(self, *, phases: tuple[str, ...]) -> int:
        with self._lock:
            targets = [
                (job_id, phase)
                for job_id, phase in self._active.items()
                if phase in phases
            ]
        cancelled = 0
        for job_id, phase in targets:
            if self._durable_payloads:
                if phase == _PHASE_RUNNING:
                    self._fence_for_handoff(job_id)
                self._hand_off(job_id)
                continue
            if self._mark_cancelled(
                job_id, expected=ACTIVE_STATUSES, error=_shutdown_error()
            ):
                cancelled += 1
            if phase == _PHASE_QUEUED:
                with self._lock:
                    if self._active.get(job_id) == _PHASE_QUEUED:
                        self._active.pop(job_id, None)
        return cancelled

    def _hand_off(self, job_id: str) -> None:
        """Leave a durable job for the next start and stop tracking it here."""
        with self._lock:
            phase = self._active.pop(job_id, None)
            if phase is not None and (self._admission_closed or self._shutdown):
                self._drain_checkpointed += 1
        self._release_reservation(job_id)

    # Worker -------------------------------------------------------------

    def _run_job(self, item: _JobWorkItem) -> None:
        keep: Optional[str] = None
        try:
            keep = self._run_job_fenced(item)
        except StaleLeaseError:
            # Another worker, a drain, or an operator owns the job now; this
            # worker commits nothing.
            keep = self._pending_phase(item.job_id)
        except Exception:
            # An unexpected failure (for example a store write error) leaves
            # the job to lease expiry and reconcile; it leaves this process.
            self._release_reservation(item.job_id)
            raise
        finally:
            self._leave_worker(item.job_id, keep)

    def _run_job_fenced(self, item: _JobWorkItem) -> Optional[str]:
        """Run one job and return the phase it keeps in this process, if any."""
        if self._cancel_requested.is_set():
            self._stop_unstarted(item.job_id)
            return None
        refusal = self._incompatible_work_refusal(item)
        if refusal is not None:
            return _PHASE_CALLBACK if refusal else None
        claimed = self._claim(item.job_id)
        if claimed is None:
            return self._pending_phase(item.job_id)
        lease, record = claimed
        with self._lock:
            self._leases[lease.job_id] = lease.epoch
        self._set_phase(lease.job_id, _PHASE_RUNNING)
        return self._process(lease, record)

    def _leave_worker(self, job_id: str, keep: Optional[str]) -> None:
        with self._lock:
            self._leases.pop(job_id, None)
            phase = self._active.get(job_id)
            if phase is None or phase == _PHASE_CALLBACK:
                # A delivery that already settled released the job itself.
                return
            if keep is None:
                self._active.pop(job_id, None)
            elif keep != _PHASE_CALLBACK:
                self._active[job_id] = keep

    def _pending_phase(self, job_id: str) -> Optional[str]:
        record = self.store.get(job_id)
        if record is not None and record.get("status") == "queued":
            return _PHASE_QUEUED
        return None

    def _incompatible_work_refusal(self, item: _JobWorkItem) -> Optional[bool]:
        """Fail a job this worker cannot handle before claiming it.

        Returns:
            ``None`` when this worker may claim the job. Otherwise the job is
            gone or was refused, and the value says whether its terminal
            callback went to the webhook dispatcher.
        """
        record = self.store.get(item.job_id)
        if record is None:
            return False
        decision = check_work(self.descriptor, record.get("compatibility"))
        if decision.compatible:
            return None
        try:
            request: Optional[DeidentifyJobRequest] = _request_from_body(
                self.payloads.read(
                    item.input_ref,
                    job_id=item.job_id,
                    kind=PAYLOAD_KIND_INPUT,
                )
            )
        except (EncryptionError, TypeError, ValueError):
            request = None
        completed_at = self.clock()
        final_values = {
            "status": "failed",
            "error": {
                "type": "IncompatibleWork",
                "message": "Job was refused before processing",
                "code": decision.code,
            },
            "completed_at": _isoformat(completed_at),
            "expires_at": self._expiry_timestamp(completed_at),
        }

        def refuse(candidate: dict[str, Any]) -> list[dict[str, Any]]:
            if isinstance(candidate.get(RECOVERY_FIELD), dict):
                self._append_transition(
                    candidate, "queued", "failed", "incompatible_work", completed_at
                )
            return []

        final_record, terminal_ref = self._commit_terminal(
            item.job_id,
            webhook=None if request is None else request.webhook,
            base=record,
            final_values=final_values,
            expected=("queued",),
            mutate=refuse,
            now=completed_at,
        )
        if final_record is None:
            return False
        return (
            self._hand_off_terminal(item.job_id, item.input_ref, terminal_ref)
            == _PHASE_CALLBACK
        )

    def _claim(self, job_id: str) -> Optional[tuple[_JobLease, dict[str, Any]]]:
        if self._shutdown and self._durable_payloads:
            # The next start resumes durable work; never take a new lease here.
            return None
        now = self.clock()

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
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
            if record.get("started_at") is None:
                record["started_at"] = _isoformat(now)
            self._append_transition(record, "queued", "running", "claimed", now)
            return []

        try:
            stored = self.store.update_if(
                job_id, expected_statuses=("queued",), mutate=apply, status="running"
            )
        except JobRecoveryError as exc:
            self._fail_unrecoverable(job_id, exc.code, now)
            return None
        if stored is None:
            return None
        recovery = stored[RECOVERY_FIELD]
        entry = _payload_entry(stored, PAYLOAD_KIND_INPUT)
        lease = _JobLease(
            job_id=job_id,
            epoch=int(recovery["lease_epoch"]),
            execution=int(recovery["execution"]),
            input_ref="" if entry is None else str(entry["ref"]),
            attempt=int(recovery["attempt"]),
            max_attempts=int(recovery["max_attempts"]),
        )
        return lease, stored

    def _process(self, lease: _JobLease, record: dict[str, Any]) -> Optional[str]:
        self._fault("claim.committed", lease.job_id)
        entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
        if entry is None or entry.get("state") != PAYLOAD_STATE_RETAINED:
            return self._handle_input_failure(lease, _payload_state_code(entry))
        try:
            request = _request_from_body(
                self.payloads.read(
                    lease.input_ref,
                    job_id=lease.job_id,
                    kind=PAYLOAD_KIND_INPUT,
                )
            )
        except EncryptionError as exc:
            return self._handle_input_failure(lease, exc.code)
        except (TypeError, ValueError):
            return self._handle_input_failure(lease, "payload_invalid")
        if not _request_matches_record(request, record, self.digests):
            return self._handle_input_failure(lease, "input_mismatch")

        recovery = record[RECOVERY_FIELD]
        summary = _JobSummary.restore(record, digests=self.digests)
        error: Optional[dict[str, str]] = recovery.get("document_error")
        checkpoint = recovery.get("checkpoint") or {}
        output_digest: Optional[str] = checkpoint.get("output_digest")
        current = record
        start = int(recovery["completed_documents"])
        total = len(request.documents)
        # Each coalesced checkpoint covers at least this many newly attempted
        # documents, which caps progress writes at max_progress_writes.
        checkpoint_step = max(1, math.ceil(total / self.max_progress_writes))
        next_checkpoint = start + checkpoint_step
        last_write = self.monotonic()

        for index in range(start, total):
            if self._cancel_requested.is_set():
                # Document boundary: committed work stays committed.
                return self._stop_running(
                    lease,
                    summary,
                    sequence=index,
                    error=error,
                    output_digest=output_digest,
                )
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
                    return self._release_for_retry(
                        lease,
                        decision,
                        summary=summary,
                        sequence=index,
                        error=error,
                        output_digest=output_digest,
                    )
                summary.failed_count += 1
                error = _safe_error(exc)
                outcome = {"document_id": document_id, "failed": True, "error": error}
            self._fault("document.processed", lease.job_id)
            output_digest = canonical_digest(
                {"previous": output_digest, "sequence": index + 1, "outcome": outcome}
            )
            attempted = index + 1
            if attempted >= total:
                # The terminal write commits the final checkpoint.
                break
            now = self.monotonic()
            if (
                attempted >= next_checkpoint
                and now - last_write >= self.progress_interval_seconds
            ):
                current = self._commit_checkpoint(
                    lease,
                    summary,
                    sequence=attempted,
                    error=error,
                    output_digest=output_digest,
                )
                last_write = now
                next_checkpoint = attempted + checkpoint_step
                self._fault("checkpoint.committed", lease.job_id)

        return self._finish_job(
            lease, request, current, summary, error, output_digest=output_digest
        )

    def _fenced(
        self,
        lease: _JobLease,
        mutate: Callable[
            [dict[str, Any], dict[str, Any]], Optional[list[dict[str, Any]]]
        ],
        now: datetime,
        **changes: Any,
    ) -> dict[str, Any]:
        """Commit through the store fence only while ``lease`` is held.

        A reclaim moves the job back to ``queued`` and the next claim issues a
        new epoch, so the status and epoch checks fence every stale writer. A
        lease that merely expired, with no reclaim yet, still belongs to its
        worker.

        Raises:
            StaleLeaseError: When the job is no longer running under this
                lease epoch and execution.
        """

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            recovery = record.get(RECOVERY_FIELD)
            if not _holds_lease(recovery, lease):
                raise StaleLeaseError()
            assert isinstance(recovery, dict)
            return mutate(record, recovery)

        stored = self.store.update_if(
            lease.job_id, expected_statuses=("running",), mutate=apply, **changes
        )
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
            self._record_checkpoint(
                record,
                recovery,
                lease,
                now,
                sequence=sequence,
                error=error,
                output_digest=output_digest,
            )
            recovery["lease"] = renew_lease(
                recovery["lease"], now=now, lease_seconds=self.lease_seconds
            )
            return []

        return self._fenced(lease, mutate, now, **summary.to_progress_record())

    def _record_checkpoint(
        self,
        record: Mapping[str, Any],
        recovery: dict[str, Any],
        lease: _JobLease,
        now: datetime,
        *,
        sequence: int,
        error: Optional[dict[str, str]],
        output_digest: Optional[str],
    ) -> None:
        if output_digest is None or sequence <= recovery["completed_documents"]:
            return
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

    def _release_for_retry(
        self,
        lease: _JobLease,
        decision: FailureDecision,
        *,
        summary: Optional["_JobSummary"] = None,
        sequence: int = 0,
        error: Optional[dict[str, str]] = None,
        output_digest: Optional[str] = None,
    ) -> Optional[str]:
        """Release a failed attempt for a backed-off retry, or quarantine it.

        Documents finished since the last coalesced checkpoint are committed
        in the same fenced write, so the retry resumes after them.
        """
        now = self.clock()
        progress = (
            {}
            if summary is None or output_digest is None
            else summary.to_progress_record()
        )

        def checkpoint(record: dict[str, Any], recovery: dict[str, Any]) -> None:
            self._record_checkpoint(
                record,
                recovery,
                lease,
                now,
                sequence=sequence,
                error=error,
                output_digest=output_digest,
            )

        if lease.attempt >= lease.max_attempts:

            def exhaust(
                record: dict[str, Any], recovery: dict[str, Any]
            ) -> list[dict[str, Any]]:
                checkpoint(record, recovery)
                return self._quarantine_recovery(
                    record,
                    recovery,
                    now,
                    classification="unsafe",
                    reason_code="attempts_exhausted",
                    from_status="running",
                )

            stored = self._fenced(
                lease,
                exhaust,
                now,
                **progress,
                **self._quarantine_changes("attempts_exhausted", now),
            )
            self._settle_local(lease.job_id, stored)
            return None
        delay = self.retry_policy.delay_seconds(lease.attempt)

        def requeue(
            record: dict[str, Any], recovery: dict[str, Any]
        ) -> list[dict[str, Any]]:
            checkpoint(record, recovery)
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
            self._append_transition(
                record, "running", "queued", decision.reason_code, now
            )
            return []

        self._fenced(lease, requeue, now, status="queued", **progress)
        return _PHASE_QUEUED

    def _handle_input_failure(self, lease: _JobLease, code: str) -> Optional[str]:
        decision = classify_input_failure(code)
        if decision.action == "retry":
            return self._release_for_retry(lease, decision)
        now = self.clock()
        error = {
            "code": decision.reason_code,
            "type": "JobPayloadError",
            "message": _PAYLOAD_UNREADABLE_MESSAGE,
        }
        if decision.action == "quarantine":

            def quarantine(
                record: dict[str, Any], recovery: dict[str, Any]
            ) -> list[dict[str, Any]]:
                return self._quarantine_recovery(
                    record,
                    recovery,
                    now,
                    classification=decision.classification,
                    reason_code=decision.reason_code,
                    from_status="running",
                )

            stored = self._fenced(
                lease,
                quarantine,
                now,
                **self._quarantine_changes(decision.reason_code, now, error=error),
            )
        else:

            def fail(
                record: dict[str, Any], recovery: dict[str, Any]
            ) -> list[dict[str, Any]]:
                # The payload is left untouched so the last verified state
                # survives; it still expires on its own deadline.
                recovery["lease"] = None
                self._append_transition(
                    record, "running", "failed", decision.reason_code, now
                )
                return []

            stored = self._fenced(
                lease,
                fail,
                now,
                status="failed",
                error=error,
                completed_at=_isoformat(now),
                expires_at=self._expiry_timestamp(now),
            )
        self._settle_local(lease.job_id, stored)
        return None

    def _quarantine_changes(
        self,
        reason_code: str,
        now: datetime,
        *,
        error: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        return {
            "status": "quarantined",
            "error": error
            or {
                "code": reason_code,
                "type": "JobQuarantined",
                "message": "Job was quarantined until an authorized replay",
            },
            "expires_at": self._expiry_timestamp(now),
        }

    def _quarantine_recovery(
        self,
        record: dict[str, Any],
        recovery: dict[str, Any],
        now: datetime,
        *,
        classification: str,
        reason_code: str,
        from_status: str,
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
        *,
        output_digest: Optional[str] = None,
    ) -> Optional[str]:
        """Commit the terminal state, final checkpoint and webhook intent.

        Returns:
            ``callback`` when the terminal callback went to the webhook
            dispatcher, so the job stays outstanding until it settles.

        Raises:
            StaleLeaseError: When the lease was lost before the commit; the
                sealed terminal event is deleted again.
        """
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

        def finalize(record: dict[str, Any]) -> list[dict[str, Any]]:
            recovery = record.get(RECOVERY_FIELD)
            if not _holds_lease(recovery, lease):
                raise StaleLeaseError()
            assert isinstance(recovery, dict)
            self._record_checkpoint(
                record,
                recovery,
                lease,
                now,
                sequence=summary.total,
                error=error,
                output_digest=output_digest,
            )
            recovery["lease"] = None
            self._append_transition(record, "running", status, "completed", now)
            return []

        final_record, terminal_ref = self._commit_terminal(
            lease.job_id,
            webhook=request.webhook,
            base=current,
            final_values=final_values,
            expected=("running",),
            mutate=finalize,
            now=now,
        )
        if final_record is None:
            raise StaleLeaseError()
        with self._lock:
            if self._admission_closed or self._shutdown:
                self._drain_completed += 1
        return self._hand_off_terminal(lease.job_id, lease.input_ref, terminal_ref)

    def _commit_terminal(
        self,
        job_id: str,
        *,
        webhook: Optional[JobWebhookRequest],
        base: Mapping[str, Any],
        final_values: dict[str, Any],
        expected: Collection[str],
        mutate: StoreMutation,
        now: datetime,
    ) -> tuple[Optional[dict[str, Any]], Optional[PayloadRef]]:
        """Seal the terminal event, then commit it with the terminal state.

        The terminal state, the consumed input, the terminal-event reference
        and the webhook intent are one fenced write. A sealed event that no
        committed record references is deleted again.
        """
        terminal_ref: Optional[PayloadRef] = None
        unavailable_code = WEBHOOK_EVENT_UNAVAILABLE
        if webhook is not None:
            event = {
                "delivery": {
                    "url": webhook.url,
                    "secret": webhook.secret,
                    "max_attempts": webhook.max_attempts,
                    "backoff_seconds": webhook.backoff_seconds,
                    # Sealed with the event so every retry, including one
                    # after a restart, reuses the same delivery identity.
                    "delivery_id": new_delivery_id(),
                },
                "body": webhook_payload({**base, **final_values}),
            }
            try:
                terminal_ref = self.payloads.create(
                    PAYLOAD_KIND_TERMINAL_EVENT,
                    job_id,
                    event,
                    retention_seconds=_record_retention_seconds(
                        base, self.payloads.max_retention_seconds
                    ),
                )
            except EncryptionError as exc:
                unavailable_code = exc.code
        self._fault("terminal.sealed", job_id)
        changes = dict(final_values)
        if terminal_ref is not None:
            changes["webhook_delivery"] = pending_delivery_metadata()
        elif webhook is not None:
            delivery = _payload_delivery_failure(unavailable_code)
            delivery["payload_ref"] = None
            changes["webhook_delivery"] = delivery
        window = self._delivery_window(base)

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            receipts = mutate(record)
            if receipts is None:
                return None
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
            if payloads:
                record["payloads"] = payloads
            recovery = record.get(RECOVERY_FIELD)
            if isinstance(recovery, dict):
                if terminal_ref is not None:
                    recovery["webhook"] = {
                        "state": "pending",
                        "payload_ref": terminal_ref.ref,
                        "claim_expires_at": _isoformat(now + timedelta(seconds=window)),
                    }
                elif webhook is not None:
                    recovery["webhook"] = {"state": "unavailable", "payload_ref": None}
            return list(receipts)

        try:
            final_record = self.store.update_if(
                job_id, expected_statuses=expected, mutate=apply, **changes
            )
        except BaseException:
            if terminal_ref is not None:
                self._delete_payload(terminal_ref.ref, reason="not_committed")
            raise
        if final_record is None:
            if terminal_ref is not None:
                self._delete_payload(terminal_ref.ref, reason="not_committed")
            return None, None
        self._settle_local(job_id, final_record)
        return final_record, terminal_ref

    def _hand_off_terminal(
        self,
        job_id: str,
        input_ref: str,
        terminal_ref: Optional[PayloadRef],
    ) -> Optional[str]:
        self._fault("terminal.committed", job_id)
        self._delete_payload(input_ref, reason="consumed")
        if terminal_ref is None:
            return None
        self._set_phase(job_id, _PHASE_CALLBACK)
        self._deliver_terminal_event(job_id, terminal_ref)
        return _PHASE_CALLBACK

    def _deliver_terminal_event(
        self, job_id: str, terminal_ref: PayloadRef | str
    ) -> None:
        """Deliver the sealed terminal event on the bounded webhook dispatcher.

        The body is read from the immutable terminal-event payload, never
        rebuilt from job state. Payload read failures keep their controlled
        code; sender outcomes are recorded through the dispatcher together
        with the durable webhook intent.
        """
        ref = terminal_ref.ref if isinstance(terminal_ref, PayloadRef) else terminal_ref
        payload_failure: list[str] = []

        def _deliver(
            cancel_event: threading.Event,
            remaining_seconds: float,
        ) -> WebhookDeliveryResult:
            try:
                with self.payloads.pinned(ref):
                    event = self.payloads.read(
                        ref,
                        job_id=job_id,
                        kind=PAYLOAD_KIND_TERMINAL_EVENT,
                    )
                    result = self._send_webhook(
                        event,
                        cancel_event=cancel_event,
                        max_total_seconds=remaining_seconds,
                    )
            except EncryptionError as exc:
                payload_failure.append(exc.code)
                return _unsent_delivery(exc.code)
            self._fault("webhook.sent", job_id)
            return result

        def _settle(result: WebhookDeliveryResult) -> None:
            try:
                self._record_delivery(
                    job_id,
                    ref,
                    result,
                    payload_failure=payload_failure[0] if payload_failure else None,
                )
                self._fault("webhook.recorded", job_id)
            finally:
                with self._lock:
                    self._deliveries.discard(job_id)
                self._release_callback(job_id)

        with self._lock:
            self._deliveries.add(job_id)
        try:
            self._webhooks.submit(_deliver, _settle)
        except BaseException:
            with self._lock:
                self._deliveries.discard(job_id)
            self._release_callback(job_id)
            raise

    def _record_delivery(
        self,
        job_id: str,
        ref: str,
        result: WebhookDeliveryResult,
        *,
        payload_failure: Optional[str] = None,
        state: Optional[str] = None,
    ) -> None:
        if payload_failure is not None and not result.success:
            delivery = _payload_delivery_failure(payload_failure)
        else:
            delivery = result.to_dict()
        delivery["payload_ref"] = ref
        now = self.clock()
        # A delivery the shutdown abandoned stays pending when its sealed
        # event survives the process, so the next start re-drives it.
        resume = (
            state is None
            and not result.success
            and result.error == WEBHOOK_ERROR_CANCELLED
            and self._durable_payloads
        )
        discard = not resume and (
            result.success or not self.payloads.retains_failed_deliveries
        )
        reason = "delivered" if result.success else "delivery_failed"
        if resume:
            intent: dict[str, Any] = {
                "state": "pending",
                "payload_ref": ref,
                "claim_expires_at": _isoformat(now),
            }
        else:
            intent = {
                "state": state or ("delivered" if result.success else "failed"),
                "payload_ref": ref,
            }

        def apply(record: dict[str, Any]) -> list[dict[str, Any]]:
            recovery = record.get(RECOVERY_FIELD)
            if isinstance(recovery, dict):
                recovery["webhook"] = dict(intent)
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

        self.store.update_if(
            job_id, expected_statuses=None, mutate=apply, webhook_delivery=delivery
        )
        if discard:
            self._delete_payload(ref, reason=reason)

    def _release_callback(self, job_id: str) -> None:
        with self._lock:
            if self._active.get(job_id) == _PHASE_CALLBACK:
                self._active.pop(job_id, None)

    def _send_webhook(
        self,
        event: Mapping[str, Any],
        *,
        cancel_event: Optional[threading.Event] = None,
        max_total_seconds: Optional[float] = None,
    ) -> WebhookDeliveryResult:
        delivery = event.get("delivery")
        body = event.get("body")
        if not isinstance(delivery, dict) or not isinstance(body, dict):
            raise JobPayloadError("payload_damaged")
        try:
            url = str(delivery["url"])
            secret = str(delivery["secret"])
            max_attempts = int(delivery["max_attempts"])
            backoff_seconds = float(delivery["backoff_seconds"])
            delivery_id = normalize_delivery_id(delivery["delivery_id"])
        except (KeyError, TypeError, ValueError):
            raise JobPayloadError("payload_damaged") from None
        sender = self.webhook_sender
        # Optional keywords reach senders that accept them; deliver_webhook
        # takes all three. The sealed delivery id keeps retries and restarts
        # on one stable id.
        options: dict[str, Any] = {"delivery_id": delivery_id}
        if cancel_event is not None:
            options["cancel_event"] = cancel_event
        if max_total_seconds is not None:
            options["max_total_seconds"] = max_total_seconds
        with self._lock:
            draining = self._admission_closed or self._shutdown
            callback_timeout = self._callback_timeout_seconds
        if draining:
            # One bounded attempt: a drain never waits on retry backoff.
            max_attempts = 1
            backoff_seconds = 0.0
            if callback_timeout is not None:
                options["timeout_seconds"] = callback_timeout
        try:
            result = sender(
                url,
                body,
                secret=secret,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
                **_sender_options(sender, **options),
            )
        except Exception:
            # Sender exceptions can carry endpoints or response content; only a
            # controlled code is recorded.
            result = _unsent_delivery(WEBHOOK_SENDER_FAILED)
        # The sealed delivery id is authoritative even for custom senders.
        return dataclasses.replace(result, delivery_id=delivery_id)

    def _stop_unstarted(self, job_id: str) -> None:
        """Stop a job a drain reached before its worker claimed it."""
        if self._durable_payloads:
            self._hand_off(job_id)
            return
        self._mark_cancelled(job_id, expected=("queued",), error=_shutdown_error())

    def _stop_running(
        self,
        lease: _JobLease,
        summary: "_JobSummary",
        *,
        sequence: int,
        error: Optional[dict[str, str]],
        output_digest: Optional[str],
    ) -> None:
        """Stop a running job at a document boundary for a drain."""
        if not self._durable_payloads:
            self._mark_cancelled(
                lease.job_id,
                expected=("running",),
                error=_shutdown_error(),
                lease=lease,
            )
            return None
        now = self.clock()

        def release(
            record: dict[str, Any], recovery: dict[str, Any]
        ) -> list[dict[str, Any]]:
            self._record_checkpoint(
                record,
                recovery,
                lease,
                now,
                sequence=sequence,
                error=error,
                output_digest=output_digest,
            )
            self._requeue_interrupted(record, recovery, now)
            return []

        changes = summary.to_progress_record() if sequence else {}
        self._fenced(lease, release, now, status="queued", **changes)
        self._hand_off(lease.job_id)
        return None

    def _fence_for_handoff(self, job_id: str) -> None:
        """Take a running job's lease away so its worker can never commit."""
        now = self.clock()

        def release(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            recovery = record.get(RECOVERY_FIELD)
            if not isinstance(recovery, dict):
                return None
            self._requeue_interrupted(record, recovery, now)
            return []

        try:
            self.store.update_if(
                job_id, expected_statuses=("running",), mutate=release, status="queued"
            )
        except Exception:
            # The lease still expires and reconcile reclaims the job.
            return

    def _requeue_interrupted(
        self, record: dict[str, Any], recovery: dict[str, Any], now: datetime
    ) -> None:
        recovery["attempt"] = max(0, recovery["attempt"] - 1)
        recovery["retry"] = build_retry(
            str(record["id"]),
            execution=recovery["execution"],
            attempt=max(1, recovery["attempt"]),
            classification="cancelled",
            reason_code="interrupted_by_shutdown",
            recorded_at=now,
            retry_after=now,
        )
        recovery["lease"] = None
        recovery["lease_draining"] = False
        self._append_transition(
            record, "running", "queued", "interrupted_by_shutdown", now
        )

    def _mark_cancelled(
        self,
        job_id: str,
        *,
        expected: Collection[str] = ("queued",),
        error: Optional[dict[str, Any]] = None,
        actor_digest: str = SYSTEM_ACTOR_DIGEST,
        lease: Optional[_JobLease] = None,
    ) -> bool:
        """Fence one job as ``cancelled`` and delete its retained input payload.

        The store refuses the change unless the record is still in an
        ``expected`` status (and, with ``lease``, still under that lease), so
        a stale worker can never overwrite it and a job is cancelled at most
        once.

        Args:
            job_id: Job to cancel.
            expected: Statuses the job may currently have.
            error: Value-free error to record; defaults to the
                ``cancelled_on_shutdown`` error for jobs that never started.
            actor_digest: ``sha256:`` digest attributing the cancellation.
            lease: Lease the cancelling worker holds, if any.

        Returns:
            Whether this call moved the job to ``cancelled``.
        """
        return (
            self._cancel_record(
                job_id,
                expected=expected,
                error=error,
                actor_digest=actor_digest,
                lease=lease,
            )
            is not None
        )

    def _cancel_record(
        self,
        job_id: str,
        *,
        expected: Collection[str],
        error: Optional[dict[str, Any]],
        actor_digest: str,
        lease: Optional[_JobLease] = None,
    ) -> Optional[dict[str, Any]]:
        now = self.clock()
        error = error if error is not None else _cancelled_on_shutdown_error()
        code = error.get("code")
        reason_code = (
            code
            if isinstance(code, str) and _REASON_CODE_RE.fullmatch(code)
            else "cancelled"
        )
        input_refs: list[str] = []

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            recovery = record.get(RECOVERY_FIELD)
            if lease is not None and not _holds_lease(recovery, lease):
                return None
            from_status = str(record.get("status"))
            entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
            if entry is not None and entry.get("state") != PAYLOAD_STATE_DELETED:
                if entry.get("state") == PAYLOAD_STATE_RETAINED:
                    input_refs.append(str(entry["ref"]))
                payloads = dict(record["payloads"])
                payloads[PAYLOAD_KIND_INPUT] = {
                    **entry,
                    "state": PAYLOAD_STATE_DELETED,
                    "reason": "cancelled",
                }
                record["payloads"] = payloads
            if not isinstance(recovery, dict):
                return []
            recovery["lease"] = None
            recovery["lease_draining"] = False
            recovery["quarantine"] = None
            recovery["cancellation"] = build_cancellation(
                job_id,
                actor_digest=actor_digest,
                reason_code=reason_code,
                requested_at=now,
            )
            self._append_transition(record, from_status, "cancelled", reason_code, now)
            return [
                decision_receipt(
                    job_id,
                    action="cancelled",
                    reason_code=reason_code,
                    from_status=from_status,
                    to_status="cancelled",
                    recovery=recovery,
                    state_version=_state_version(record) + 1,
                    at=now,
                    actor_digest=actor_digest,
                )
            ]

        record = self.store.update_if(
            job_id,
            expected_statuses=expected,
            mutate=apply,
            status="cancelled",
            error=error,
            completed_at=_isoformat(now),
            expires_at=self._expiry_timestamp(now),
        )
        if record is None:
            return None
        with self._lock:
            if self._admission_closed or self._shutdown:
                self._drain_cancelled += 1
        self._settle_local(job_id, record)
        for ref in input_refs:
            self._delete_payload(ref, reason="cancelled")
        return record

    def _mark_draining(self, job_id: str, epoch: int) -> None:
        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            recovery = record.get(RECOVERY_FIELD)
            lease = recovery.get("lease") if isinstance(recovery, dict) else None
            if not isinstance(lease, dict) or lease.get("epoch") != epoch:
                return None
            assert isinstance(recovery, dict)
            recovery["lease_draining"] = True
            return []

        self.store.update_if(job_id, expected_statuses=("running",), mutate=apply)

    # Reconcile helpers ------------------------------------------------

    def _reconcile_running(
        self, record: dict[str, Any], now: datetime, counts: Counter[str]
    ) -> None:
        job_id = str(record["id"])
        recovery = record[RECOVERY_FIELD]
        lease = recovery["lease"]
        epoch = int(lease["epoch"])
        if not is_due(lease.get("expires_at"), now):
            with self._lock:
                held = self._leases.get(job_id) == epoch
            counts["leased"] += 1
            if held and self._renewal_due(lease, now):
                self._renew_lease(job_id, epoch, int(recovery["execution"]), now)
                counts["renewed"] += 1
            return
        draining = bool(recovery.get("lease_draining"))
        reason = "interrupted_by_shutdown" if draining else "lease_expired"
        exhausted = not draining and recovery["attempt"] >= recovery["max_attempts"]

        def apply(candidate: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            current_recovery = validate_recovery(candidate)
            current = current_recovery["lease"]
            if (
                current["epoch"] != epoch
                or bool(current_recovery.get("lease_draining")) != draining
                or not is_due(current["expires_at"], now)
            ):
                return None
            if draining:
                current_recovery["attempt"] = max(0, current_recovery["attempt"] - 1)
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
                return receipts + self._quarantine_recovery(
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
            self._append_transition(candidate, "running", "queued", reason, now)
            return receipts

        changes = (
            self._quarantine_changes("attempts_exhausted", now)
            if exhausted
            else {"status": "queued"}
        )
        try:
            stored = self.store.update_if(
                job_id, expected_statuses=("running",), mutate=apply, **changes
            )
        except JobRecoveryError as exc:
            if self._fail_unrecoverable(job_id, exc.code, now):
                counts["failed"] += 1
            return
        if stored is None:
            return
        counts["reclaimed"] += 1
        if stored.get("status") == "quarantined":
            counts["quarantined"] += 1
            self._settle_local(job_id, stored)

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
        retry = record[RECOVERY_FIELD].get("retry")
        if retry is not None and not is_due(retry.get("retry_after"), now):
            counts["waiting"] += 1
            with self._lock:
                if not self._shutdown and not self._cancel_requested.is_set():
                    self._active.setdefault(job_id, _PHASE_QUEUED)
            return
        if self._dispatch(job_id, entry["ref"]):
            counts["dispatched"] += 1

    def _reconcile_delivery(
        self, record: dict[str, Any], now: datetime, counts: Counter[str]
    ) -> None:
        recovery = record.get(RECOVERY_FIELD)
        intent = recovery.get("webhook") if isinstance(recovery, dict) else None
        if not isinstance(intent, dict) or intent.get("state") != "pending":
            return
        if not is_due(intent.get("claim_expires_at"), now):
            return
        job_id = str(record["id"])
        ref = intent.get("payload_ref")
        with self._lock:
            if self._shutdown or job_id in self._deliveries:
                return
        window = self._delivery_window(record)

        def apply(candidate: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            current_recovery = candidate.get(RECOVERY_FIELD)
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

        if (
            not isinstance(ref, str)
            or self.store.update_if(
                job_id, expected_statuses=TERMINAL_STATUSES, mutate=apply
            )
            is None
        ):
            return
        if not self.payloads.contains(ref):
            self._record_delivery(
                job_id,
                ref,
                _unsent_delivery(WEBHOOK_EVENT_UNAVAILABLE),
                payload_failure=WEBHOOK_EVENT_UNAVAILABLE,
                state="unavailable",
            )
            counts["deliveries_unavailable"] += 1
            return
        with self._lock:
            if self._shutdown or job_id in self._deliveries:
                return
            self._active[job_id] = _PHASE_CALLBACK
        try:
            self._deliver_terminal_event(job_id, ref)
        except RuntimeError:
            self._release_callback(job_id)
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
        def apply(record: dict[str, Any]) -> list[dict[str, Any]]:
            recovery = validate_recovery(record)
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
            stored = self.store.update_if(
                job_id,
                expected_statuses=("queued",),
                mutate=apply,
                status="failed",
                error={
                    "code": code,
                    "type": "JobPayloadError",
                    "message": _PAYLOAD_UNREADABLE_MESSAGE,
                },
                completed_at=_isoformat(now),
                expires_at=self._expiry_timestamp(now),
            )
        except JobRecoveryError:
            return self._fail_unrecoverable(job_id, "recovery_state_invalid", now)
        if stored is None:
            return False
        self._settle_local(job_id, stored)
        return True

    def _fail_unrecoverable(self, job_id: str, code: str, now: datetime) -> bool:
        """Fail a job whose recovery state cannot be trusted, closed."""
        input_refs: list[str] = []

        def apply(record: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
            status = record.get("status")
            if status in TERMINAL_STATUSES:
                return None
            from_status = status if status in JOB_STATUSES else "unknown"
            entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
            if entry is not None and entry.get("state") == PAYLOAD_STATE_RETAINED:
                input_refs.append(str(entry["ref"]))
                payloads = dict(record["payloads"])
                payloads[PAYLOAD_KIND_INPUT] = {
                    **entry,
                    "state": PAYLOAD_STATE_DELETED,
                    "reason": "recovery_failed",
                }
                record["payloads"] = payloads
            recovery = record.get(RECOVERY_FIELD)
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

        stored = self.store.update_if(
            job_id,
            expected_statuses=None,
            mutate=apply,
            status="failed",
            error={
                "code": code,
                "type": "JobRecoveryError",
                "message": "Job recovery state could not be trusted",
            },
            completed_at=_isoformat(now),
            expires_at=self._expiry_timestamp(now),
        )
        for ref in input_refs:
            self._delete_payload(ref, reason="recovery_failed")
        if stored is None:
            return False
        self._settle_local(job_id, stored)
        return True

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
        ) -> Optional[list[dict[str, Any]]]:
            if is_due(recovery["lease"].get("expires_at"), now):
                # An expired lease is left for reclaim, never extended.
                return None
            recovery["lease"] = renew_lease(
                recovery["lease"], now=now, lease_seconds=self.lease_seconds
            )
            return []

        try:
            self._fenced(lease, mutate, now)
        except StaleLeaseError:
            return

    # Shared helpers ---------------------------------------------------

    @property
    def _durable_payloads(self) -> bool:
        """Whether job payloads survive this process."""
        return self.payloads.mode == PAYLOAD_MODE_ENCRYPTED

    def _dispatch(self, job_id: str, input_ref: str) -> bool:
        # Lock order: the queue lock, then the futures lock.
        with self._lock:
            with self._futures_lock:
                if (
                    self._shutdown
                    or self._cancel_requested.is_set()
                    or job_id in self._futures
                ):
                    return False
                try:
                    future = self._executor.submit(
                        self._run_job, _JobWorkItem(job_id, input_ref)
                    )
                except RuntimeError:
                    return False
                self._futures[job_id] = future
            self._active.setdefault(job_id, _PHASE_QUEUED)
        future.add_done_callback(functools.partial(self._forget_future, job_id))
        return True

    def _forget_future(self, job_id: str, future: Future[None]) -> None:
        with self._futures_lock:
            if self._futures.get(job_id) is future:
                self._futures.pop(job_id, None)
        if future.cancelled():
            # A cancelled future never runs ``_run_job``, so it is released
            # here instead.
            with self._lock:
                if self._active.get(job_id) == _PHASE_QUEUED:
                    self._active.pop(job_id, None)

    def _settle_local(self, job_id: str, record: Optional[Mapping[str, Any]]) -> None:
        """Return capacity once a job reaches a terminal or quarantined state."""
        if record is None or record.get("status") not in _SETTLED_STATUSES:
            return
        self._release_reservation(job_id)
        with self._lock:
            if self._active.get(job_id) == _PHASE_QUEUED:
                self._active.pop(job_id, None)

    def _release_reservation(self, job_id: str) -> None:
        with self._lock:
            reservation = self._reservations.pop(job_id, None)
        if reservation is not None:
            reservation.release()

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
        recovery = record[RECOVERY_FIELD]
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
        """Return how long one delivery may hold its claim before a re-drive."""
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
        bounded = min(
            attempts * DEFAULT_WEBHOOK_TIMEOUT_SECONDS + waits,
            self._webhooks.max_delivery_seconds,
        )
        return self.lease_seconds + math.ceil(bounded)

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
        owner_digest: Optional[dict[str, str]] = None,
        idempotency: Optional[IdempotentStartRequest] = None,
    ) -> dict[str, Any]:
        documents = [
            _document_metadata(index, document, self.digests)
            for index, document in enumerate(payload.documents)
        ]
        recovery = new_recovery(
            max_attempts=self.retry_policy.max_attempts,
            request_digest=_request_digest(payload, self.digests),
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
            "schema_version": JOB_STORE_SCHEMA_VERSION,
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
            DIGESTS_FIELD: self.digests.descriptor(),
            "webhook": _webhook_metadata(payload.webhook, self.digests),
            "webhook_delivery": None,
            "payloads": {
                "schema_version": JOB_PAYLOAD_SCHEMA_VERSION,
                "mode": self.payloads.mode,
                "retention_seconds": retention_seconds,
                PAYLOAD_KIND_INPUT: input_ref.to_record(),
                PAYLOAD_KIND_TERMINAL_EVENT: None,
            },
            RECOVERY_FIELD: recovery,
            IDEMPOTENCY_FIELD: (
                None
                if idempotency is None
                else {
                    "scope_digest": idempotency.scope_digest,
                    "request_digest": idempotency.request_digest,
                }
            ),
            "created_at": _isoformat(now),
            "updated_at": _isoformat(now),
            "started_at": None,
            "completed_at": None,
            "expires_at": self._expiry_timestamp(now),
            OWNER_DIGEST_FIELD: None if owner_digest is None else dict(owner_digest),
            "compatibility": self._work_envelope(payload).to_dict(),
        }

    def _work_envelope(self, payload: DeidentifyJobRequest) -> WorkEnvelope:
        capabilities = {JOB_DEIDENTIFY_CAPABILITY}
        if payload.webhook is not None:
            capabilities.add(JOB_WEBHOOK_CAPABILITY)
        # Pin the model generation whenever this node publishes the generations
        # it serves, so a claimant that does not serve it refuses the job.
        model_generation = (
            model_generation_ref(payload.model_name)
            if self.descriptor.model_generations
            else None
        )
        return WorkEnvelope.for_work(
            self.descriptor,
            components=(JOB_STORE_COMPONENT,),
            required_capabilities=capabilities,
            model_generation=model_generation,
        )

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

        self.store.update_if(job_id, expected_statuses=None, mutate=apply)

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

    def _set_phase(self, job_id: str, phase: str) -> None:
        with self._lock:
            if job_id in self._active or phase == _PHASE_RUNNING:
                self._active[job_id] = phase

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


def _holds_lease(recovery: Any, lease: _JobLease) -> bool:
    """Return whether ``lease`` is still the job's current lease."""
    current = recovery.get("lease") if isinstance(recovery, dict) else None
    return (
        isinstance(current, dict)
        and current.get("epoch") == lease.epoch
        and recovery.get("execution") == lease.execution
    )


def _sender_options(sender: WebhookSender, **options: Any) -> dict[str, Any]:
    """Return the cancellation/budget options *sender* accepts.

    Custom senders with the original keyword set keep working; senders that
    accept ``cancel_event`` / ``max_total_seconds`` (or ``**kwargs``) receive
    them.
    """
    try:
        parameters = inspect.signature(sender).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return options
    return {name: value for name, value in options.items() if name in parameters}


def _unsent_delivery(code: str) -> WebhookDeliveryResult:
    return WebhookDeliveryResult(
        success=False,
        attempts=0,
        status_code=None,
        error=code,
    )


def _payload_delivery_failure(code: str) -> dict[str, Any]:
    """Return delivery metadata for a webhook that never reached the sender.

    ``code`` is a controlled payload or encryption code, so it is kept rather
    than collapsed into the generic ``sender_error``.
    """
    delivery = _unsent_delivery(code).to_dict()
    delivery["error"] = code
    return delivery


@dataclass
class _JobSummary:
    total: int
    digests: JobDigestPolicy = field(default_factory=JobDigestPolicy)
    processed_count: int = 0
    failed_count: int = 0
    label_histogram: dict[str, int] = field(default_factory=dict)
    spans: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def restore(
        cls,
        record: Mapping[str, Any],
        *,
        digests: Optional[JobDigestPolicy] = None,
    ) -> "_JobSummary":
        """Rebuild the committed summary from a checkpointed record."""
        return cls(
            total=int(record.get("document_count") or 0),
            digests=JobDigestPolicy() if digests is None else digests,
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
            spans.append(_entity_span(document_id, entity, label, self.digests))
            labels[label] = labels.get(label, 0) + 1
        self.processed_count += 1
        for label, count in labels.items():
            self.label_histogram[label] = self.label_histogram.get(label, 0) + count
        self.spans.extend(spans)
        return spans

    @property
    def attempted(self) -> int:
        """Return the number of documents processed or failed so far."""
        return self.processed_count + self.failed_count

    def to_progress_record(self) -> dict[str, Any]:
        attempted = self.attempted
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


def _document_metadata(
    index: int,
    document: DeidentifyJobDocument,
    digests: JobDigestPolicy,
) -> dict[str, Any]:
    return {
        "id": _document_id(index, document),
        "length": len(document.text),
        TEXT_DIGEST_FIELD: digests.digest(DOCUMENT_TEXT_DIGEST_DOMAIN, document.text),
    }


def _webhook_metadata(
    webhook: Optional[JobWebhookRequest],
    digests: JobDigestPolicy,
) -> Optional[dict[str, Any]]:
    if webhook is None:
        return None
    return {
        "configured": True,
        URL_DIGEST_FIELD: digests.digest(WEBHOOK_URL_DIGEST_DOMAIN, webhook.url),
        "max_attempts": webhook.max_attempts,
        "backoff_seconds": webhook.backoff_seconds,
    }


def _request_digest(payload: DeidentifyJobRequest, digests: JobDigestPolicy) -> str:
    """Return the immutable request digest over values metadata already holds.

    The digest covers the execution parameters, the document metadata, and the
    webhook metadata, whose text and URL references are keyed digests (or
    omitted). The date shift is reduced to whether it is set, so the digest
    never encodes a value that metadata does not already expose.
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
                _document_metadata(index, document, digests)
                for index, document in enumerate(payload.documents)
            ],
            "webhook": _webhook_metadata(payload.webhook, digests),
        }
    )


def _idempotency_fields(payload: DeidentifyJobRequest) -> dict[str, Any]:
    """Return the request fields an idempotency request digest covers."""
    fields = _request_body(payload)
    webhook = fields.get("webhook")
    if isinstance(webhook, dict):
        # A credential, like the bulk routes' private key: rotating it on a
        # retry still names the same request.
        fields["webhook"] = {
            name: value for name, value in webhook.items() if name != "secret"
        }
    return fields


def _request_matches_record(
    request: DeidentifyJobRequest,
    record: Mapping[str, Any],
    digests: JobDigestPolicy,
) -> bool:
    """Return whether the input snapshot is the one the record was built from.

    Document identifiers and lengths always match. Text digests are compared
    only while the record was keyed under the same digest key, so a key
    rotation between admission and execution does not quarantine the job.
    """
    stored = record.get("documents")
    if record.get("document_count") != len(request.documents) or not isinstance(
        stored, list
    ):
        return False
    if len(stored) != len(request.documents):
        return False
    descriptor = record.get(DIGESTS_FIELD)
    same_key = (
        digests.key is not None
        and isinstance(descriptor, dict)
        and descriptor.get("mode") == DIGEST_MODE_KEYED
        and descriptor.get("key_id") == digests.key.key_id
    )
    for index, document in enumerate(request.documents):
        expected = _document_metadata(index, document, digests)
        item = stored[index]
        if not isinstance(item, dict):
            return False
        if item.get("id") != expected["id"] or item.get("length") != expected["length"]:
            return False
        if same_key and item.get(TEXT_DIGEST_FIELD) != expected[TEXT_DIGEST_FIELD]:
            return False
    return True


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


def _entity_span(
    document_id: str,
    entity: Any,
    label: str,
    digests: JobDigestPolicy,
) -> dict[str, Any]:
    text = str(getattr(entity, "text", "") or "")
    return {
        "document_id": document_id,
        "start": int(getattr(entity, "start", 0) or 0),
        "end": int(getattr(entity, "end", 0) or 0),
        "label": label,
        TEXT_DIGEST_FIELD: digests.digest(SPAN_TEXT_DIGEST_DOMAIN, text),
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


def _cancelled_on_shutdown_error() -> dict[str, Any]:
    return {
        "code": "cancelled_on_shutdown",
        "type": "CancelledError",
        "message": "Job was cancelled before it started",
    }


def _shutdown_error() -> dict[str, Any]:
    return {
        "type": "ServiceShuttingDown",
        "code": SHUTTING_DOWN,
        "message": "Job was cancelled because the service shut down; resubmit it",
        "retryable": True,
    }


def webhook_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build a no-PHI terminal webhook payload from a persisted record.

    Args:
        record: Persisted job record.

    Returns:
        The payload: ``digests`` describes how span and document digests were
        made, and each span and document carries a keyed ``text_digest`` (or
        ``None`` when omitted) instead of an unkeyed hash. Like the other
        published contracts it carries no ``schema_version`` field; the
        contract version is the registered ``service_webhook_payload`` schema.
    """
    current = _copy_record(record)
    _migrate_record(current)
    return {
        "event": f"job.{current['status']}",
        "job_id": current["id"],
        "status": current["status"],
        "progress_percent": current["progress_percent"],
        "document_count": current["document_count"],
        "processed_count": current["processed_count"],
        "failed_count": current["failed_count"],
        "label_histogram": current["label_histogram"],
        DIGESTS_FIELD: current[DIGESTS_FIELD],
        "spans": current["spans"],
        "documents": current["documents"],
        "error": current["error"],
        "completed_at": current["completed_at"],
    }


def job_response_payload(
    record: dict[str, Any],
    *,
    status_url: Optional[str] = None,
) -> dict[str, Any]:
    """Return the public REST representation of a job record.

    The stored owner digest, the idempotency digests and the mixed-version
    work envelope are internal state and are never returned to callers.
    Recovery metadata is reduced to its public, value-free projection: lease
    and worker identifiers and the request digest are omitted. Records are
    returned in the current schema, so a legacy unkeyed hash is never echoed.
    """
    payload = _copy_record(record)
    _migrate_record(payload)
    payload.pop(OWNER_DIGEST_FIELD, None)
    # The storage schema version drives migration only; the published
    # service_job_record contract is versioned by its registered schema.
    payload.pop("schema_version", None)
    # The work envelope is internal routing state for mixed-version workers.
    payload.pop("compatibility", None)
    payload.pop(IDEMPOTENCY_FIELD, None)
    payload[RECOVERY_FIELD] = public_recovery_view(payload.pop(RECOVERY_FIELD, None))
    if status_url is not None:
        payload["status_url"] = status_url
    return payload
