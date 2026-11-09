"""Async de-identification jobs for the OpenMed REST service."""

from __future__ import annotations

import copy
import inspect
import json
import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Optional

import openmed
from openmed.core.audit import hash_text

from .drain import SHUTTING_DOWN, ServiceShuttingDownError
from .runtime import ServiceRuntime
from .schemas import DeidentifyJobDocument, DeidentifyJobRequest, JobWebhookRequest
from .webhooks import WebhookDeliveryResult, deliver_webhook

SERVICE_JOBS_STORE_ENV_VAR = "OPENMED_SERVICE_JOBS_STORE_PATH"
SERVICE_JOBS_TTL_ENV_VAR = "OPENMED_SERVICE_JOBS_TTL_SECONDS"
SERVICE_JOBS_WORKERS_ENV_VAR = "OPENMED_SERVICE_JOBS_WORKERS"
DEFAULT_JOBS_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOBS_WORKERS = 2
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
ACTIVE_STATUSES = frozenset({"queued", "running"})


WebhookSender = Callable[..., WebhookDeliveryResult]


class JobQueueClosedError(ServiceShuttingDownError):
    """Raised when a job is submitted after the queue stopped admitting work."""

    def __init__(self) -> None:
        super().__init__("Job queue is shutting down")


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


class LocalJobStore:
    """Durable local metadata store for async jobs.

    The store intentionally persists only metadata and no raw submitted text or
    redacted output. Raw job payloads live in memory until a worker processes
    them.
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

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        """Apply changes to a job record and persist them."""
        with self._lock:
            previous = _copy_record(self._records)
            record = self._records[job_id]
            record.update(_copy_record(changes))
            record["updated_at"] = _isoformat(self.clock())
            self._publish_locked(previous)
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
            record = self._records.get(job_id)
            if record is None or record.get("status") not in expected_statuses:
                return None
            previous = _copy_record(self._records)
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
    """Bounded local worker pool for async de-identification jobs."""

    def __init__(
        self,
        runtime: ServiceRuntime,
        *,
        store: LocalJobStore,
        max_workers: int = DEFAULT_JOBS_WORKERS,
        webhook_sender: WebhookSender = deliver_webhook,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.webhook_sender = webhook_sender
        self.clock = clock
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="openmed-job",
        )
        self._shutdown = False
        self._lock = threading.Lock()
        self._admission_closed = False
        self._cancel_requested = threading.Event()
        self._callback_timeout_seconds: Optional[float] = None
        self._active: dict[str, str] = {}
        self._drain_completed = 0
        self._drain_cancelled = 0

    @classmethod
    def from_env(cls, runtime: ServiceRuntime) -> "DeidentifyJobQueue":
        """Create a queue from service environment variables."""
        store = LocalJobStore(
            parse_jobs_store_path(os.getenv(SERVICE_JOBS_STORE_ENV_VAR)),
            ttl_seconds=parse_jobs_ttl_seconds(os.getenv(SERVICE_JOBS_TTL_ENV_VAR)),
        )
        return cls(
            runtime,
            store=store,
            max_workers=parse_jobs_workers(os.getenv(SERVICE_JOBS_WORKERS_ENV_VAR)),
        )

    def submit(self, payload: DeidentifyJobRequest) -> dict[str, Any]:
        """Submit one de-identification job and return its initial metadata."""
        self.runtime.validate_served_model(payload.model_name)
        with self._lock:
            if self._shutdown or self._admission_closed:
                raise JobQueueClosedError()
            record = self._new_record(payload)
            self.store.create(record)
            self._active[record["id"]] = "queued"
            self._executor.submit(self._run_job, _JobWorkItem(record["id"], payload))
            return _copy_record(record)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return persisted job metadata."""
        return self.store.get(job_id)

    def shutdown(self) -> None:
        """Stop accepting work and request worker shutdown.

        Jobs whose worker never started are marked ``cancelled`` with a
        retryable ``shutting_down`` code instead of being left ``queued``.
        """
        with self._lock:
            self._shutdown = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._cancel_jobs(phases=("queued",))

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
            if self._mark_cancelled(job_id, expected=ACTIVE_STATUSES):
                cancelled += 1
            if phase == "queued":
                with self._lock:
                    if self._active.get(job_id) == "queued":
                        self._active.pop(job_id, None)
        return cancelled

    def _mark_cancelled(self, job_id: str, *, expected: Collection[str]) -> bool:
        now = self.clock()
        updated = self.store.update_if(
            job_id,
            expected_statuses=expected,
            status="cancelled",
            error=_shutdown_error(),
            completed_at=_isoformat(now),
            expires_at=self._expiry_timestamp(now),
        )
        if updated is None:
            return False
        with self._lock:
            if self._admission_closed or self._shutdown:
                self._drain_cancelled += 1
        return True

    def _expiry_timestamp(self, moment: datetime) -> str:
        """Return the metadata expiry for a record that reached *moment*."""
        expires_at = datetime.fromtimestamp(
            moment.timestamp() + self.store.ttl_seconds,
            tz=timezone.utc,
        )
        return _isoformat(expires_at)

    def _new_record(self, payload: DeidentifyJobRequest) -> dict[str, Any]:
        now = self.clock()
        documents = [
            _document_metadata(index, document)
            for index, document in enumerate(payload.documents)
        ]
        return {
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
            "webhook": _webhook_metadata(payload.webhook),
            "webhook_delivery": None,
            "created_at": _isoformat(now),
            "updated_at": _isoformat(now),
            "started_at": None,
            "completed_at": None,
            "expires_at": self._expiry_timestamp(now),
        }

    def _run_job(self, item: _JobWorkItem) -> None:
        try:
            self._run_job_fenced(item)
        finally:
            with self._lock:
                self._active.pop(item.job_id, None)

    def _run_job_fenced(self, item: _JobWorkItem) -> None:
        if self._cancel_requested.is_set():
            self._mark_cancelled(item.job_id, expected=("queued",))
            return
        started_at = self.clock()
        started = self.store.update_if(
            item.job_id,
            expected_statuses=("queued",),
            status="running",
            started_at=_isoformat(started_at),
            progress_percent=0.0,
        )
        if started is None:
            return
        self._set_phase(item.job_id, "running")
        summary = _JobSummary(total=len(item.payload.documents))
        error: Optional[dict[str, str]] = None

        for index, document in enumerate(item.payload.documents):
            if self._cancel_requested.is_set():
                # Document boundary: the last progress update is the checkpoint.
                self._mark_cancelled(item.job_id, expected=("running",))
                return
            document_id = _document_id(index, document)
            try:
                result = self._deidentify_document(item.payload, document)
                summary.add_result(document_id, result)
            except Exception as exc:
                summary.failed_count += 1
                error = _safe_error(exc)

            progress = self.store.update_if(
                item.job_id,
                expected_statuses=("running",),
                **summary.to_progress_record(),
            )
            if progress is None:
                return

        completed_at = self.clock()
        status = "failed" if summary.failed_count else "done"
        final_record = self.store.update_if(
            item.job_id,
            expected_statuses=("running",),
            status=status,
            progress_percent=100.0,
            error=error,
            completed_at=_isoformat(completed_at),
            expires_at=self._expiry_timestamp(completed_at),
        )
        if final_record is None:
            # A drain fenced this job while its last document ran.
            return
        with self._lock:
            if self._admission_closed or self._shutdown:
                self._drain_completed += 1
        self._set_phase(item.job_id, "callback")
        self._send_terminal_webhook(item.payload.webhook, final_record)

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

    def _send_terminal_webhook(
        self,
        webhook: Optional[JobWebhookRequest],
        record: dict[str, Any],
    ) -> None:
        if webhook is None:
            return
        payload = webhook_payload(record)
        options: dict[str, Any] = {
            "secret": webhook.secret,
            "max_attempts": webhook.max_attempts,
            "backoff_seconds": webhook.backoff_seconds,
        }
        with self._lock:
            draining = self._admission_closed or self._shutdown
            callback_timeout = self._callback_timeout_seconds
        if draining:
            # One bounded attempt: a drain never waits on retry backoff.
            options["max_attempts"] = 1
            options["backoff_seconds"] = 0.0
            if callback_timeout is not None and _accepts_keyword(
                self.webhook_sender, "timeout_seconds"
            ):
                options["timeout_seconds"] = callback_timeout
        try:
            result = self.webhook_sender(webhook.url, payload, **options)
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
        "type": exc.__class__.__name__,
        "message": "Document failed during de-identification",
    }


def _shutdown_error() -> dict[str, Any]:
    return {
        "type": "ServiceShuttingDown",
        "code": SHUTTING_DOWN,
        "message": "Job was cancelled because the service shut down; resubmit it",
        "retryable": True,
    }


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


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
