"""Async de-identification jobs for the OpenMed REST service.

Job metadata (status, counts, hashes, offsets, and opaque payload references)
is persisted by :class:`LocalJobStore`.  PHI-bearing content never enters that
metadata: each submission is snapshotted once into an immutable input payload,
and each terminal webhook event is sealed once into an immutable terminal-event
payload, both held by a :class:`~openmed.service.job_payloads.JobPayloadStore`
under an explicit, bounded retention deadline.

Job records and terminal webhook payloads never carry submitted text. Span,
document and webhook URL references are keyed, domain-separated HMAC-SHA256
digests under the deployment digest key (see :mod:`.keyed_digest`), so a
low-entropy identifier cannot be recovered by hashing candidate values. Outside
the ``dev`` and ``test`` profiles, a service without a configured key omits
these digests and records an explicit marker instead of hashing without a key.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Final, Optional

import openmed
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
from .job_admission import (
    DEFAULT_MAX_DOCUMENTS_PER_JOB,
    DEFAULT_MAX_PENDING_JOBS,
    DEFAULT_MAX_QUEUED_TEXT_BYTES,
    JobAdmission,
    JobAdmissionLimits,
    JobCost,
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
    DEFAULT_WEBHOOK_WORKERS,
    WEBHOOK_ERROR_SENDER,
    WebhookDeliveryResult,
    WebhookDispatcher,
    authorize_webhook_destination,
    deliver_webhook,
    pending_delivery_metadata,
)

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
DEFAULT_JOBS_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOBS_WORKERS = 2
DEFAULT_PROGRESS_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_PROGRESS_WRITES = 100
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
PAYLOAD_MODES = frozenset({PAYLOAD_MODE_MEMORY, PAYLOAD_MODE_ENCRYPTED})

# Sender exceptions are recorded with the dispatcher's controlled code.
WEBHOOK_SENDER_FAILED = WEBHOOK_ERROR_SENDER
WEBHOOK_EVENT_UNAVAILABLE = "terminal_event_unavailable"
# JOB_STORE_SCHEMA_VERSION comes from openmed.service.compatibility, which
# documents each version and the window of versions this build reads.
_JOB_STORE_FIELDS = frozenset({"jobs", "schema_version"})
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
_LEGACY_TEXT_HASH_FIELD: Final = "text_hash"
_LEGACY_URL_HASH_FIELD: Final = "url_hash"
ACTIVE_STATUSES = frozenset({"queued", "running"})


WebhookSender = Callable[..., WebhookDeliveryResult]
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
    text, so they are removed and the record is marked accordingly.

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


class LocalJobStore:
    """Durable local metadata store for async jobs.

    The store intentionally persists only metadata and no raw submitted text or
    redacted output. Job payloads are held by a separate
    :class:`~openmed.service.job_payloads.JobPayloadStore` and appear here only
    as opaque references with their lifecycle state and retention deadline. Stores written by older schema versions are migrated when loaded and
    rewritten at once, so removed unkeyed hashes do not linger on disk.

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
        self._lock = threading.RLock()
        self._schema = CURRENT_CONTRACT.durable_schemas[JOB_STORE_COMPONENT]
        self._blocked: Optional[ComponentCompatibility] = None
        self._published_signature: Optional[_FileSignature] = None
        records, migrated = self._load()
        self._records: dict[str, dict[str, Any]] = records
        if migrated:
            with self._lock:
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
        """Persist a new job record."""
        with self._lock:
            self._require_compatible_locked()
            previous = self._remove_expired_locked()
            job_id = str(record["id"])
            previous.setdefault(job_id, self._records.get(job_id))
            self._records[job_id] = _copy_record(record)
            self._publish_locked(previous)
            return _copy_record(self._records[job_id])

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return one job record, if present."""
        with self._lock:
            self._require_compatible_locked()
            self.cleanup_expired_locked()
            record = self._records.get(job_id)
            return None if record is None else _copy_record(record)

    def records(self) -> list[dict[str, Any]]:
        """Return copies of every unexpired job record."""
        with self._lock:
            self._require_compatible_locked()
            self.cleanup_expired_locked()
            return [_copy_record(record) for record in self._records.values()]

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        """Apply changes to a job record and persist them."""
        with self._lock:
            self._require_compatible_locked()
            current = self._records[job_id]
            # Build the new record from a shallow copy so the current record
            # stays untouched and serves as the rollback snapshot. Changes only
            # replace top-level values, and stored records are never mutated in
            # place, so no other record needs a snapshot.
            record = dict(current)
            record.update(_copy_record(changes))
            record["updated_at"] = _isoformat(self.clock())
            self._records[job_id] = record
            self._publish_locked({job_id: current})
            return _copy_record(record)

    def update_if(
        self,
        job_id: str,
        *,
        expected_statuses: Collection[str],
        **changes: Any,
    ) -> Optional[dict[str, Any]]:
        """Apply changes only while the record is in an expected status.

        This is the commit fence for job lifecycle transitions: once a drain
        marks a job ``cancelled``, a stale worker's later update is refused.

        Args:
            job_id: Record identifier.
            expected_statuses: Statuses the record must currently have.
            **changes: Fields to update.

        Returns:
            The updated record, or ``None`` when the fence refused the change.
        """
        with self._lock:
            self._require_compatible_locked()
            current = self._records.get(job_id)
            if current is None or current.get("status") not in expected_statuses:
                return None
            # Same copy-on-write discipline as :meth:`update`: the current
            # record is the rollback snapshot and is never mutated in place.
            record = dict(current)
            record.update(_copy_record(changes))
            record["updated_at"] = _isoformat(self.clock())
            self._records[job_id] = record
            self._publish_locked({job_id: current})
            return _copy_record(record)

    def cleanup_expired(self) -> None:
        """Remove terminal records whose TTL has elapsed."""
        with self._lock:
            self._require_compatible_locked()
            previous = self._remove_expired_locked()
            if previous:
                self._publish_locked(previous)

    def cleanup_expired_locked(self) -> bool:
        """Drop expired terminal records from memory without persisting."""
        return bool(self._remove_expired_locked())

    def _remove_expired_locked(self) -> _RecordSnapshots:
        now = self.clock()
        expired = [
            job_id
            for job_id, record in self._records.items()
            if record.get("status") in TERMINAL_STATUSES
            and _parse_timestamp(record.get("expires_at")) <= now
        ]
        return {job_id: self._records.pop(job_id) for job_id in expired}

    def _load(self) -> tuple[dict[str, dict[str, Any]], bool]:
        if not self.path.exists():
            return {}, False
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
        records = {str(job_id): dict(record) for job_id, record in jobs.items()}
        migrated = version != JOB_STORE_SCHEMA_VERSION
        for record in records.values():
            migrated |= _migrate_record(record)
        self._published_signature = _file_signature(self.path)
        return records, migrated

    def _require_compatible_locked(self) -> None:
        if self._blocked is not None:
            raise IncompatibleStateError(JOB_STORE_COMPONENT, self._blocked.code)

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
        payload = {"jobs": self._records, "schema_version": JOB_STORE_SCHEMA_VERSION}
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
    """Bounded local worker pool for async de-identification jobs.

    Status transitions and terminal results are persisted immediately.
    Per-document progress is coalesced into checkpoints: a checkpoint is
    written only after ``progress_interval_seconds`` have elapsed since the
    previous write and after at least ``1 / max_progress_writes`` of the job's
    documents have completed since then. Each job therefore rewrites the store
    a bounded number of times regardless of its document count.

    Terminal webhooks are delivered on a separate bounded
    :class:`~openmed.service.webhooks.WebhookDispatcher`, so job workers return
    as soon as the terminal record is committed.

    Each submission is snapshotted once into an immutable input payload before
    any metadata is derived, so later changes to the caller's request object
    never reach the queued work. Workers read the snapshot back through its
    opaque reference. A terminal webhook event is sealed once into an immutable
    terminal-event payload, and delivery reads that payload instead of
    rebuilding the body from mutable job state.

    Submissions pass :class:`~openmed.service.job_admission.JobAdmission`
    before any metadata is persisted, so pending jobs, their queued text bytes
    and documents per job stay within configured ceilings.

    Every record is stamped with a :class:`WorkEnvelope` naming the contract
    and model generation it was admitted under, and a worker checks that
    envelope with :func:`check_work` before it marks the job running.

    Args:
        runtime: Service runtime that executes model requests.
        store: Local metadata store for job records.
        max_workers: Maximum number of concurrently running jobs.
        webhook_sender: Callable that delivers one signed webhook. It is called
            like :func:`~openmed.service.webhooks.deliver_webhook` with ``url``,
            ``payload``, ``secret``, ``max_attempts``, ``backoff_seconds``, and
            the stable ``delivery_id`` sealed with the terminal event.
        clock: Wall-clock source for record timestamps.
        progress_interval_seconds: Minimum seconds between two progress
            checkpoints of one job. ``0`` disables the time limit.
        max_progress_writes: Maximum number of progress checkpoints per job.
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

    Raises:
        ValueError: If ``progress_interval_seconds`` is negative or
            ``max_progress_writes`` is less than one.
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
    ) -> None:
        if not progress_interval_seconds >= 0:
            raise ValueError("progress_interval_seconds must be non-negative")
        if int(max_progress_writes) < 1:
            raise ValueError("max_progress_writes must be a positive integer")
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
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="openmed-job",
        )
        self._shutdown = False
        self._lock = threading.Lock()
        self._record_lock = threading.RLock()
        self._futures_lock = threading.Lock()
        self._futures: dict[str, Future[None]] = {}
        self.payloads.recover(self._live_payload_refs())
        self._admission_closed = False
        self._cancel_requested = threading.Event()
        self._callback_timeout_seconds: Optional[float] = None
        self._active: dict[str, str] = {}
        self._drain_completed = 0
        self._drain_cancelled = 0

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
                encrypted payload retention.
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
        )

    def submit(
        self,
        payload: DeidentifyJobRequest,
        *,
        owner_digest: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Submit one de-identification job and return its initial metadata.

        Capacity is reserved before the record is created. A rejected
        submission persists no record and queues no work. The reservation is
        released when the job's work finishes, fails or is cancelled.

        A webhook URL is checked against the egress policy before the job is
        accepted, so a refused callback is rejected at submission.

        Args:
            payload: The validated request. It is snapshotted immediately; the
                caller may reuse or mutate it afterwards.
            owner_digest: Keyed digest of the submitting principal, or ``None``
                when authentication is disabled.

        Returns:
            The initial job record.

        Raises:
            ValueError: If the requested payload retention exceeds the
                configured maximum or the model is not served.
            JobLimitExceededError: If the job exceeds a per-job ceiling.
            JobQueueFullError: If the pending-job queue is at capacity.
            RuntimeError: If the queue is shutting down.
            EncryptionError: If the input payload cannot be retained.
            EgressDenied: When the egress policy refuses the webhook URL.
        """
        snapshot = DeidentifyJobRequest(**_request_body(payload))
        self.runtime.validate_served_model(snapshot.model_name)
        if snapshot.webhook is not None:
            authorize_webhook_destination(snapshot.webhook.url)
        self._payload_retention_seconds(snapshot)
        cost = JobCost.for_texts(document.text for document in snapshot.documents)
        with self._lock:
            if self._shutdown or self._admission_closed:
                raise JobQueueClosedError()
            self._sweep_payloads()
            reservation = self.admission.reserve(cost)
            try:
                record, item = self._stage(snapshot, owner_digest=owner_digest)
                try:
                    self.store.create(record)
                except BaseException:
                    self.payloads.delete(item.input_ref, reason="submission_failed")
                    raise
                self._active[item.job_id] = "queued"
                try:
                    future = self._executor.submit(self._run_job, item)
                except BaseException:
                    self._active.pop(item.job_id, None)
                    raise
            except BaseException:
                reservation.release()
                raise
            with self._futures_lock:
                self._futures[item.job_id] = future
            future.add_done_callback(
                functools.partial(self._forget_future, item.job_id)
            )
            # Runs on completion, failure or cancellation, including shutdown.
            future.add_done_callback(reservation.release)
            return _copy_record(record)

    def _stage(
        self,
        payload: DeidentifyJobRequest,
        *,
        owner_digest: Optional[dict[str, str]] = None,
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
        )
        return record, _JobWorkItem(job_id, input_ref.ref)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return persisted job metadata."""
        self._sweep_payloads()
        return self.store.get(job_id)

    def shutdown(self, drain_seconds: Optional[float] = None) -> None:
        """Stop accepting work, cancel unstarted jobs, and settle webhooks.

        Jobs whose work was cancelled before a worker picked it up are recorded
        as ``cancelled`` and their input payloads are deleted. Running and
        finished jobs keep their own states. Job workers are released without
        waiting. Webhook backoff waits are cancelled, and deliveries that do
        not settle within the drain bound are recorded as ``abandoned`` with
        error ``cancelled``.

        Args:
            drain_seconds: Override for the configured webhook drain bound.

        Raises:
            RuntimeError: If a cancelled job's state could not be persisted.
                Every other cancelled job is still recorded first, and webhook
                deliveries are still settled.
        """
        with self._lock:
            self._shutdown = True
        with self._futures_lock:
            pending = dict(self._futures)
        self._executor.shutdown(wait=False, cancel_futures=True)
        failed = False
        for job_id, future in pending.items():
            if not future.cancelled():
                continue
            try:
                self._mark_cancelled(job_id)
            except Exception:
                failed = True
        self._webhooks.shutdown(drain_seconds)
        if failed:
            # Storage errors can name paths; surface a fixed message instead.
            raise RuntimeError("Cancelled job state could not be persisted")

    def _forget_future(self, job_id: str, future: Future[None]) -> None:
        with self._futures_lock:
            self._futures.pop(job_id, None)
        if future.cancelled():
            # A cancelled future never runs ``_run_job``, so it is released
            # here instead.
            with self._lock:
                if self._active.get(job_id) == "queued":
                    self._active.pop(job_id, None)

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
        """Ask workers to stop at their next document (checkpoint) boundary."""
        self._cancel_requested.set()

    def cancel_outstanding(self) -> int:
        """Fence every queued or running job as cancelled.

        A worker that is still inside a model call can no longer commit its
        result: every later status-bearing update is refused by the store.

        Returns:
            Jobs moved to ``cancelled`` by this call.
        """
        return self._cancel_jobs(phases=("queued", "running"))

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
            }

    def _cancel_jobs(self, *, phases: tuple[str, ...]) -> int:
        with self._lock:
            targets = [
                (job_id, phase)
                for job_id, phase in self._active.items()
                if phase in phases
            ]
        cancelled = 0
        for job_id, phase in targets:
            if self._mark_cancelled(
                job_id, expected=ACTIVE_STATUSES, error=_shutdown_error()
            ):
                cancelled += 1
            if phase == "queued":
                with self._lock:
                    if self._active.get(job_id) == "queued":
                        self._active.pop(job_id, None)
        return cancelled

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
    ) -> dict[str, Any]:
        documents = [
            _document_metadata(index, document, self.digests)
            for index, document in enumerate(payload.documents)
        ]
        return {
            "schema_version": JOB_STORE_SCHEMA_VERSION,
            "id": job_id,
            "status": "queued",
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
        final_record = self.store.update_if(
            item.job_id,
            expected_statuses=("queued",),
            status="failed",
            error={
                "type": "IncompatibleWork",
                "message": "Job was refused before processing",
                "code": decision.code,
            },
            completed_at=_isoformat(completed_at),
            expires_at=self._expiry_timestamp(completed_at),
            **_pending_webhook_delivery(None if request is None else request.webhook),
        )
        if final_record is None or request is None:
            return False
        return self._finish_job(item, request, final_record)

    def _run_job(self, item: _JobWorkItem) -> None:
        handed_off = False
        try:
            handed_off = self._run_job_fenced(item)
        finally:
            # A job whose terminal callback went to the dispatcher stays
            # outstanding until that delivery settles.
            if not handed_off:
                with self._lock:
                    self._active.pop(item.job_id, None)

    def _run_job_fenced(self, item: _JobWorkItem) -> bool:
        if self._cancel_requested.is_set():
            self._mark_cancelled(
                item.job_id, expected=("queued",), error=_shutdown_error()
            )
            return False
        refusal = self._incompatible_work_refusal(item)
        if refusal is not None:
            return refusal
        started_at = self.clock()
        started = self.store.update_if(
            item.job_id,
            expected_statuses=("queued",),
            status="running",
            started_at=_isoformat(started_at),
            progress_percent=0.0,
        )
        if started is None:
            return False
        self._set_phase(item.job_id, "running")
        try:
            request = _request_from_body(
                self.payloads.read(
                    item.input_ref,
                    job_id=item.job_id,
                    kind=PAYLOAD_KIND_INPUT,
                )
            )
        except EncryptionError as exc:
            self._fail_unreadable_input(item.job_id, exc.code)
            return False
        except (TypeError, ValueError):
            self._fail_unreadable_input(item.job_id, "payload_invalid")
            return False

        summary = _JobSummary(total=len(request.documents), digests=self.digests)
        error: Optional[dict[str, str]] = None
        # Each checkpoint must cover at least this many newly attempted
        # documents, which caps progress writes at max_progress_writes.
        checkpoint_step = max(1, math.ceil(summary.total / self.max_progress_writes))
        next_checkpoint = checkpoint_step
        last_write = self.monotonic()

        for index, document in enumerate(request.documents):
            if self._cancel_requested.is_set():
                # Document boundary: the last committed checkpoint stays.
                self._mark_cancelled(
                    item.job_id, expected=("running",), error=_shutdown_error()
                )
                return False
            document_id = _document_id(index, document)
            try:
                result = self._deidentify_document(request, document)
                summary.add_result(document_id, result)
            except Exception as exc:
                summary.failed_count += 1
                error = _safe_error(exc)

            attempted = summary.attempted
            # The terminal update below records the last document.
            if attempted < summary.total and attempted >= next_checkpoint:
                now = self.monotonic()
                if now - last_write >= self.progress_interval_seconds:
                    progress = self.store.update_if(
                        item.job_id,
                        expected_statuses=("running",),
                        **summary.to_progress_record(),
                    )
                    if progress is None:
                        return False
                    last_write = now
                    next_checkpoint = attempted + checkpoint_step

        completed_at = self.clock()
        status = "failed" if summary.failed_count else "done"
        terminal = summary.to_progress_record()
        terminal.update(
            status=status,
            progress_percent=100.0,
            error=error,
            completed_at=_isoformat(completed_at),
            expires_at=self._expiry_timestamp(completed_at),
            **_pending_webhook_delivery(request.webhook),
        )
        final_record = self.store.update_if(
            item.job_id, expected_statuses=("running",), **terminal
        )
        if final_record is None:
            # A drain fenced this job while its last document ran.
            return False
        with self._lock:
            if self._admission_closed or self._shutdown:
                self._drain_completed += 1
        return self._finish_job(item, request, final_record)

    def _fail_unreadable_input(self, job_id: str, code: str) -> None:
        # The payload is left untouched so the last verified state survives; it
        # still expires on its own deadline.
        completed_at = self.clock()
        self.store.update_if(
            job_id,
            expected_statuses=("running",),
            status="failed",
            error={
                "code": code,
                "type": "JobPayloadError",
                "message": "Job input payload could not be read",
            },
            completed_at=_isoformat(completed_at),
            expires_at=self._expiry_timestamp(completed_at),
        )

    def _finish_job(
        self,
        item: _JobWorkItem,
        request: DeidentifyJobRequest,
        final_record: dict[str, Any],
    ) -> bool:
        """Seal and hand off the terminal event; return whether it is pending.

        Returns:
            ``True`` when the terminal callback went to the webhook dispatcher,
            so the job stays outstanding until that delivery settles.
        """
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
                    # Sealed with the event so every retry, including one
                    # after a restart, reuses the same delivery identity.
                    "delivery_id": new_delivery_id(),
                },
                "body": webhook_payload(final_record),
            }
            try:
                terminal_ref = self.payloads.create(
                    PAYLOAD_KIND_TERMINAL_EVENT,
                    item.job_id,
                    event,
                    retention_seconds=_record_retention_seconds(
                        final_record, self.payloads.max_retention_seconds
                    ),
                )
            except EncryptionError as exc:
                unavailable_code = exc.code
            else:
                self._set_payload_entry(
                    item.job_id,
                    PAYLOAD_KIND_TERMINAL_EVENT,
                    entry=terminal_ref.to_record(),
                )
        self._discard_payload(
            item.job_id, PAYLOAD_KIND_INPUT, item.input_ref, reason="consumed"
        )
        if webhook is None:
            return False
        if terminal_ref is None:
            delivery = _payload_delivery_failure(unavailable_code)
            delivery["payload_ref"] = None
            self.store.update(item.job_id, webhook_delivery=delivery)
            return False
        self._set_phase(item.job_id, "callback")
        self._deliver_terminal_event(item.job_id, terminal_ref)
        return True

    def _deliver_terminal_event(self, job_id: str, terminal_ref: PayloadRef) -> None:
        """Deliver the sealed terminal event on the bounded webhook dispatcher.

        The body is read from the immutable terminal-event payload, never
        rebuilt from job state. Payload read failures keep their controlled
        code; sender outcomes are recorded through the dispatcher.
        """
        payload_failure: list[str] = []

        def _deliver(
            cancel_event: threading.Event,
            remaining_seconds: float,
        ) -> WebhookDeliveryResult:
            try:
                with self.payloads.pinned(terminal_ref.ref):
                    event = self.payloads.read(
                        terminal_ref.ref,
                        job_id=job_id,
                        kind=PAYLOAD_KIND_TERMINAL_EVENT,
                    )
                    return self._send_webhook(
                        event,
                        cancel_event=cancel_event,
                        max_total_seconds=remaining_seconds,
                    )
            except EncryptionError as exc:
                payload_failure.append(exc.code)
                return _unsent_delivery(exc.code)

        def _settle(result: WebhookDeliveryResult) -> None:
            try:
                _record(result)
            finally:
                self._release_callback(job_id)

        def _record(result: WebhookDeliveryResult) -> None:
            if payload_failure and not result.success:
                delivery = _payload_delivery_failure(payload_failure[0])
            else:
                delivery = result.to_dict()
            delivery["payload_ref"] = terminal_ref.ref
            self.store.update(job_id, webhook_delivery=delivery)
            if result.success:
                self._discard_payload(
                    job_id,
                    PAYLOAD_KIND_TERMINAL_EVENT,
                    terminal_ref.ref,
                    reason="delivered",
                )
            elif not self.payloads.retains_failed_deliveries:
                self._discard_payload(
                    job_id,
                    PAYLOAD_KIND_TERMINAL_EVENT,
                    terminal_ref.ref,
                    reason="delivery_failed",
                )

        try:
            self._webhooks.submit(_deliver, _settle)
        except BaseException:
            self._release_callback(job_id)
            raise

    def _release_callback(self, job_id: str) -> None:
        with self._lock:
            if self._active.get(job_id) == "callback":
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

    def _mark_cancelled(
        self,
        job_id: str,
        *,
        expected: Collection[str] = ("queued",),
        error: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Fence one job as ``cancelled`` and delete its retained input payload.

        The store refuses the change unless the record is still in an
        ``expected`` status, so a stale worker can never overwrite it and a job
        is cancelled at most once.

        Args:
            job_id: Job to cancel.
            expected: Statuses the job may currently have.
            error: Value-free error to record; defaults to the
                ``cancelled_on_shutdown`` error for jobs that never started.

        Returns:
            Whether this call moved the job to ``cancelled``.
        """
        with self._record_lock:
            cancelled_at = self.clock()
            record = self.store.update_if(
                job_id,
                expected_statuses=expected,
                status="cancelled",
                error=error if error is not None else _cancelled_on_shutdown_error(),
                completed_at=_isoformat(cancelled_at),
                expires_at=self._expiry_timestamp(cancelled_at),
            )
        if record is None:
            return False
        with self._lock:
            if self._admission_closed or self._shutdown:
                self._drain_cancelled += 1
        input_entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
        if (
            input_entry is not None
            and input_entry.get("state") != PAYLOAD_STATE_DELETED
        ):
            self._discard_payload(
                job_id, PAYLOAD_KIND_INPUT, input_entry["ref"], reason="cancelled"
            )
        return True

    def _discard_payload(
        self, job_id: str, kind: str, ref: str, *, reason: str
    ) -> None:
        self.payloads.delete(ref, reason=reason)
        self._set_payload_entry(
            job_id, kind, state=PAYLOAD_STATE_DELETED, reason=reason
        )

    def _set_payload_entry(
        self,
        job_id: str,
        kind: str,
        *,
        entry: Optional[dict[str, Any]] = None,
        **changes: Any,
    ) -> None:
        with self._record_lock:
            record = self.store.get(job_id)
            if record is None or not isinstance(record.get("payloads"), dict):
                return
            payloads = dict(record["payloads"])
            current = dict(entry if entry is not None else payloads.get(kind) or {})
            if not current:
                return
            current.update(changes)
            payloads[kind] = current
            self.store.update(job_id, payloads=payloads)

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
            if job_id in self._active or phase == "running":
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


def _pending_webhook_delivery(
    webhook: Optional[JobWebhookRequest],
) -> dict[str, Any]:
    if webhook is None:
        return {}
    return {"webhook_delivery": pending_delivery_metadata()}


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

    def add_result(self, document_id: str, result: Any) -> None:
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

    The stored owner digest is internal authorization state and is never
    returned to callers. Records are returned in the current schema, so a
    legacy unkeyed hash is never echoed.
    """
    payload = _copy_record(record)
    _migrate_record(payload)
    payload.pop(OWNER_DIGEST_FIELD, None)
    # The storage schema version drives migration only; the published
    # service_job_record contract is versioned by its registered schema.
    payload.pop("schema_version", None)
    if status_url is not None:
        payload["status_url"] = status_url
    return payload
