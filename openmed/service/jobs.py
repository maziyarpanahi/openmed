"""Async de-identification jobs for the OpenMed REST service.

Job metadata (status, counts, hashes, offsets, and opaque payload references)
is persisted by :class:`LocalJobStore`.  PHI-bearing content never enters that
metadata: each submission is snapshotted once into an immutable input payload,
and each terminal webhook event is sealed once into an immutable terminal-event
payload, both held by a :class:`~openmed.service.job_payloads.JobPayloadStore`
under an explicit, bounded retention deadline.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import tempfile
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import openmed
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
from .runtime import ServiceRuntime
from .schemas import DeidentifyJobDocument, DeidentifyJobRequest, JobWebhookRequest
from .webhooks import WebhookDeliveryResult, deliver_webhook

SERVICE_JOBS_STORE_ENV_VAR = "OPENMED_SERVICE_JOBS_STORE_PATH"
SERVICE_JOBS_TTL_ENV_VAR = "OPENMED_SERVICE_JOBS_TTL_SECONDS"
SERVICE_JOBS_WORKERS_ENV_VAR = "OPENMED_SERVICE_JOBS_WORKERS"
SERVICE_JOBS_PAYLOAD_MODE_ENV_VAR = "OPENMED_SERVICE_JOBS_PAYLOAD_MODE"
SERVICE_JOBS_PAYLOAD_DIR_ENV_VAR = "OPENMED_SERVICE_JOBS_PAYLOAD_DIR"
SERVICE_JOBS_PAYLOAD_RETENTION_ENV_VAR = (
    "OPENMED_SERVICE_JOBS_PAYLOAD_RETENTION_SECONDS"
)
DEFAULT_JOBS_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOBS_WORKERS = 2
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
PAYLOAD_MODES = frozenset({PAYLOAD_MODE_MEMORY, PAYLOAD_MODE_ENCRYPTED})

WEBHOOK_SENDER_FAILED = "webhook_sender_failed"
WEBHOOK_EVENT_UNAVAILABLE = "terminal_event_unavailable"


WebhookSender = Callable[..., WebhookDeliveryResult]


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


def _copy_record(record: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(record)


class LocalJobStore:
    """Durable local metadata store for async jobs.

    The store intentionally persists only metadata and no raw submitted text or
    redacted output. Job payloads are held by a separate
    :class:`~openmed.service.job_payloads.JobPayloadStore` and appear here only
    as opaque references with their lifecycle state and retention deadline.
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
        self._records: dict[str, dict[str, Any]] = self._load()

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

    def records(self) -> list[dict[str, Any]]:
        """Return copies of every unexpired job record."""
        with self._lock:
            self.cleanup_expired_locked()
            return [_copy_record(record) for record in self._records.values()]

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

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
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
        return {str(job_id): dict(record) for job_id, record in jobs.items()}

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"jobs": self._records}
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
    """Bounded local worker pool for async de-identification jobs.

    Each submission is snapshotted once into an immutable input payload before
    any metadata is derived, so later changes to the caller's request object
    never reach the queued work. Workers read the snapshot back through its
    opaque reference. A terminal webhook event is sealed once into an immutable
    terminal-event payload, and delivery reads that payload instead of
    rebuilding the body from mutable job state.

    Args:
        runtime: Service runtime used to run model requests.
        store: Job metadata store.
        max_workers: Bounded worker count.
        webhook_sender: Callable that delivers one signed webhook.
        clock: Source of timezone-aware UTC timestamps.
        payload_store: Payload store; defaults to a memory-only store.
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
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.webhook_sender = webhook_sender
        self.clock = clock
        self.payloads = (
            MemoryJobPayloadStore(clock=clock)
            if payload_store is None
            else payload_store
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
                encrypted payload retention.

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
        )

    def submit(self, payload: DeidentifyJobRequest) -> dict[str, Any]:
        """Submit one de-identification job and return its initial metadata.

        Args:
            payload: The validated request. It is snapshotted immediately; the
                caller may reuse or mutate it afterwards.

        Returns:
            The initial job record.

        Raises:
            ValueError: If the requested payload retention exceeds the
                configured maximum or the model is not served.
            RuntimeError: If the queue is shutting down.
            EncryptionError: If the input payload cannot be retained.
        """
        snapshot = DeidentifyJobRequest(**_request_body(payload))
        self.runtime.validate_served_model(snapshot.model_name)
        retention = self._payload_retention_seconds(snapshot)
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Job queue is shutting down")
            self._sweep_payloads()
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
            )
            try:
                self.store.create(record)
            except BaseException:
                self.payloads.delete(input_ref.ref, reason="submission_failed")
                raise
            future = self._executor.submit(
                self._run_job, _JobWorkItem(job_id, input_ref.ref)
            )
            with self._futures_lock:
                self._futures[job_id] = future
            future.add_done_callback(functools.partial(self._forget_future, job_id))
            return _copy_record(record)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return persisted job metadata."""
        self._sweep_payloads()
        return self.store.get(job_id)

    def shutdown(self) -> None:
        """Stop accepting work and cancel jobs that never started.

        Jobs whose work was cancelled before a worker picked it up are recorded
        as ``cancelled`` and their input payloads are deleted. Running and
        finished jobs keep their own states.

        Raises:
            RuntimeError: If a cancelled job's state could not be persisted.
                Every other cancelled job is still recorded first.
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
        if failed:
            # Storage errors can name paths; surface a fixed message instead.
            raise RuntimeError("Cancelled job state could not be persisted")

    def _forget_future(self, job_id: str, _future: Future[None]) -> None:
        with self._futures_lock:
            self._futures.pop(job_id, None)

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
    ) -> dict[str, Any]:
        documents = [
            _document_metadata(index, document)
            for index, document in enumerate(payload.documents)
        ]
        return {
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
            "webhook": _webhook_metadata(payload.webhook),
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
        }

    def _run_job(self, item: _JobWorkItem) -> None:
        record = self.store.get(item.job_id)
        if record is None or record.get("status") != "queued":
            return
        started_at = self.clock()
        self.store.update(
            item.job_id,
            status="running",
            started_at=_isoformat(started_at),
            progress_percent=0.0,
        )
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
            return
        except (TypeError, ValueError):
            self._fail_unreadable_input(item.job_id, "payload_invalid")
            return

        summary = _JobSummary(total=len(request.documents))
        error: Optional[dict[str, str]] = None

        for index, document in enumerate(request.documents):
            document_id = _document_id(index, document)
            try:
                result = self._deidentify_document(request, document)
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
        self._finish_job(item, request, final_record)

    def _fail_unreadable_input(self, job_id: str, code: str) -> None:
        # The payload is left untouched so the last verified state survives; it
        # still expires on its own deadline.
        completed_at = self.clock()
        self.store.update(
            job_id,
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
    ) -> None:
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
            return
        if terminal_ref is None:
            delivery = WebhookDeliveryResult(
                success=False,
                attempts=0,
                status_code=None,
                error=unavailable_code,
            ).to_dict()
            delivery["payload_ref"] = None
            self.store.update(item.job_id, webhook_delivery=delivery)
            return
        self._deliver_terminal_event(item.job_id, terminal_ref)

    def _deliver_terminal_event(self, job_id: str, terminal_ref: PayloadRef) -> None:
        try:
            with self.payloads.pinned(terminal_ref.ref):
                event = self.payloads.read(
                    terminal_ref.ref,
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

    def _mark_cancelled(self, job_id: str) -> None:
        with self._record_lock:
            record = self.store.get(job_id)
            if record is None or record.get("status") != "queued":
                return
            cancelled_at = self.clock()
            self.store.update(
                job_id,
                status="cancelled",
                error={
                    "code": "cancelled_on_shutdown",
                    "type": "CancelledError",
                    "message": "Job was cancelled before it started",
                },
                completed_at=_isoformat(cancelled_at),
                expires_at=self._expiry_timestamp(cancelled_at),
            )
        input_entry = _payload_entry(record, PAYLOAD_KIND_INPUT)
        if input_entry is not None:
            self._discard_payload(
                job_id, PAYLOAD_KIND_INPUT, input_entry["ref"], reason="cancelled"
            )

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


@dataclass
class _JobSummary:
    total: int
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
            spans.append(_entity_span(document_id, entity, label))
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
    """Return the public REST representation of a job record."""
    payload = _copy_record(record)
    if status_url is not None:
        payload["status_url"] = status_url
    return payload
