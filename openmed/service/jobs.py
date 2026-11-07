"""Async de-identification jobs for the OpenMed REST service.

Job records and terminal webhook payloads never carry submitted text. Span,
document and webhook URL references are keyed, domain-separated HMAC-SHA256
digests under the deployment digest key (see :mod:`.keyed_digest`), so a
low-entropy identifier cannot be recovered by hashing candidate values. Outside
the ``dev`` and ``test`` profiles, a service without a configured key omits
these digests and records an explicit marker instead of hashing without a key.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, Optional

import openmed
from openmed.core.config import PROFILE_ENV_VAR

from .keyed_digest import (
    DEVELOPMENT_PROFILES,
    KEYED_DIGEST_ALGORITHM,
    ServiceDigestKey,
    encode_digest_fields,
    resolve_service_digest_key,
)
from .runtime import ServiceRuntime
from .schemas import DeidentifyJobDocument, DeidentifyJobRequest, JobWebhookRequest
from .webhooks import WebhookDeliveryResult, deliver_webhook

SERVICE_JOBS_STORE_ENV_VAR = "OPENMED_SERVICE_JOBS_STORE_PATH"
SERVICE_JOBS_TTL_ENV_VAR = "OPENMED_SERVICE_JOBS_TTL_SECONDS"
SERVICE_JOBS_WORKERS_ENV_VAR = "OPENMED_SERVICE_JOBS_WORKERS"
DEFAULT_JOBS_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOBS_WORKERS = 2
TERMINAL_STATUSES = frozenset({"done", "failed"})
# Version 2 adds the keyed submitting-principal digest to every record.
# Version 3 replaces unkeyed SHA-256 span, document and webhook URL hashes with
# keyed digests and records how the digests were produced.
JOB_STORE_SCHEMA_VERSION = 3
# Terminal webhook bodies were unversioned (version 1) before keyed digests.
WEBHOOK_PAYLOAD_SCHEMA_VERSION = 2
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


WebhookSender = Callable[..., WebhookDeliveryResult]


@dataclass(frozen=True)
class _JobWorkItem:
    job_id: str
    payload: DeidentifyJobRequest


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


class LocalJobStore:
    """Durable local metadata store for async jobs.

    The store intentionally persists only metadata and no raw submitted text or
    redacted output. Raw job payloads live in memory until a worker processes
    them. Stores written by older schema versions are migrated when loaded and
    rewritten at once, so removed unkeyed hashes do not linger on disk.
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
        records, migrated = self._load()
        self._records: dict[str, dict[str, Any]] = records
        if migrated:
            with self._lock:
                self._persist_locked()

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist a new job record."""
        with self._lock:
            previous = _copy_record(self._records)
            self.cleanup_expired_locked()
            job_id = str(record["id"])
            self._records[job_id] = _copy_record(record)
            self._publish_locked(previous)
            return _copy_record(self._records[job_id])

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return one job record, if present."""
        with self._lock:
            self.cleanup_expired_locked()
            record = self._records.get(job_id)
            return None if record is None else _copy_record(record)

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        """Apply changes to a job record and persist them."""
        with self._lock:
            previous = _copy_record(self._records)
            record = self._records[job_id]
            record.update(_copy_record(changes))
            record["updated_at"] = _isoformat(self.clock())
            self._publish_locked(previous)
            return _copy_record(record)

    def cleanup_expired(self) -> None:
        """Remove terminal records whose TTL has elapsed."""
        with self._lock:
            previous = _copy_record(self._records)
            if self.cleanup_expired_locked():
                self._publish_locked(previous)

    def cleanup_expired_locked(self) -> bool:
        now = self.clock()
        expired = [
            job_id
            for job_id, record in self._records.items()
            if record.get("status") in TERMINAL_STATUSES
            and _parse_timestamp(record.get("expires_at")) <= now
        ]
        for job_id in expired:
            self._records.pop(job_id, None)
        return bool(expired)

    def _load(self) -> tuple[dict[str, dict[str, Any]], bool]:
        if not self.path.exists():
            return {}, False
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
        version = raw.get("schema_version", 1)
        if type(version) is not int or not 1 <= version <= JOB_STORE_SCHEMA_VERSION:
            raise ValueError("Job metadata store has an unsupported schema version")
        records = {str(job_id): dict(record) for job_id, record in jobs.items()}
        migrated = version != JOB_STORE_SCHEMA_VERSION
        for record in records.values():
            migrated |= _migrate_record(record)
        return records, migrated

    def _persist_locked(self) -> None:
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
        except BaseException:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            raise

    def _publish_locked(self, previous: dict[str, dict[str, Any]]) -> None:
        try:
            self._persist_locked()
        except BaseException:
            self._records = previous
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
    """Bounded local worker pool for async de-identification jobs."""

    def __init__(
        self,
        runtime: ServiceRuntime,
        *,
        store: LocalJobStore,
        max_workers: int = DEFAULT_JOBS_WORKERS,
        webhook_sender: WebhookSender = deliver_webhook,
        clock: Callable[[], datetime] = _utc_now,
        digests: Optional[JobDigestPolicy] = None,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.webhook_sender = webhook_sender
        self.clock = clock
        self.digests = JobDigestPolicy.from_env() if digests is None else digests
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="openmed-job",
        )
        self._shutdown = False
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        runtime: ServiceRuntime,
        *,
        digests: Optional[JobDigestPolicy] = None,
    ) -> "DeidentifyJobQueue":
        """Create a queue from service environment variables.

        Args:
            runtime: Service runtime used to run de-identification.
            digests: Keyed digest policy shared with the rest of the service;
                resolved from the environment when omitted.

        Returns:
            The configured job queue.
        """
        store = LocalJobStore(
            parse_jobs_store_path(os.getenv(SERVICE_JOBS_STORE_ENV_VAR)),
            ttl_seconds=parse_jobs_ttl_seconds(os.getenv(SERVICE_JOBS_TTL_ENV_VAR)),
        )
        return cls(
            runtime,
            store=store,
            max_workers=parse_jobs_workers(os.getenv(SERVICE_JOBS_WORKERS_ENV_VAR)),
            digests=digests,
        )

    def submit(
        self,
        payload: DeidentifyJobRequest,
        *,
        owner_digest: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Submit one de-identification job and return its initial metadata.

        Args:
            payload: Validated job request; raw text stays in memory only.
            owner_digest: Keyed digest of the submitting principal, or ``None``
                when authentication is disabled.

        Returns:
            The persisted initial record.
        """
        self.runtime.validate_served_model(payload.model_name)
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Job queue is shutting down")
            record = self._new_record(payload, owner_digest=owner_digest)
            self.store.create(record)
            self._executor.submit(self._run_job, _JobWorkItem(record["id"], payload))
            return _copy_record(record)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return persisted job metadata."""
        return self.store.get(job_id)

    def shutdown(self) -> None:
        """Stop accepting work and request worker shutdown."""
        with self._lock:
            self._shutdown = True
        self._executor.shutdown(wait=False, cancel_futures=True)

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
        owner_digest: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        now = self.clock()
        documents = [
            _document_metadata(index, document, self.digests)
            for index, document in enumerate(payload.documents)
        ]
        return {
            "schema_version": JOB_STORE_SCHEMA_VERSION,
            "id": uuid.uuid4().hex,
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
            "created_at": _isoformat(now),
            "updated_at": _isoformat(now),
            "started_at": None,
            "completed_at": None,
            "expires_at": self._expiry_timestamp(now),
            OWNER_DIGEST_FIELD: None if owner_digest is None else dict(owner_digest),
        }

    def _run_job(self, item: _JobWorkItem) -> None:
        started_at = self.clock()
        self.store.update(
            item.job_id,
            status="running",
            started_at=_isoformat(started_at),
            progress_percent=0.0,
        )
        summary = _JobSummary(total=len(item.payload.documents), digests=self.digests)
        error: Optional[dict[str, str]] = None

        for index, document in enumerate(item.payload.documents):
            document_id = _document_id(index, document)
            try:
                result = self._deidentify_document(item.payload, document)
                summary.add_result(document_id, result)
            except Exception as exc:
                summary.failed_count += 1
                error = _safe_error(exc)

            self.store.update(item.job_id, **summary.to_progress_record())

        completed_at = self.clock()
        status = "failed" if summary.failed_count else "done"
        final_record = self.store.update(
            item.job_id,
            status=status,
            progress_percent=100.0,
            error=error,
            completed_at=_isoformat(completed_at),
            expires_at=self._expiry_timestamp(completed_at),
        )
        self._send_terminal_webhook(item.payload.webhook, final_record)

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

    def _send_terminal_webhook(
        self,
        webhook: Optional[JobWebhookRequest],
        record: dict[str, Any],
    ) -> None:
        if webhook is None:
            return
        payload = webhook_payload(record)
        try:
            result = self.webhook_sender(
                webhook.url,
                payload,
                secret=webhook.secret,
                max_attempts=webhook.max_attempts,
                backoff_seconds=webhook.backoff_seconds,
            )
        except Exception as exc:
            result = WebhookDeliveryResult(
                success=False,
                attempts=0,
                status_code=None,
                error=str(exc),
            )
        self.store.update(record["id"], webhook_delivery=result.to_dict())


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
        "type": exc.__class__.__name__,
        "message": "Document failed during de-identification",
    }


def webhook_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build a no-PHI terminal webhook payload from a persisted record.

    Args:
        record: Persisted job record.

    Returns:
        Version 2 payload: ``schema_version`` identifies it, ``digests``
        describes how span and document digests were made, and each span and
        document carries a keyed ``text_digest`` (or ``None`` when omitted)
        instead of an unkeyed hash.
    """
    current = _copy_record(record)
    _migrate_record(current)
    return {
        "schema_version": WEBHOOK_PAYLOAD_SCHEMA_VERSION,
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
    if status_url is not None:
        payload["status_url"] = status_url
    return payload
