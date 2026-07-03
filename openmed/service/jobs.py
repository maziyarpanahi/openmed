"""Async de-identification jobs for the OpenMed REST service."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import openmed
from openmed.core.audit import hash_text

from .runtime import ServiceRuntime
from .schemas import DeidentifyJobDocument, DeidentifyJobRequest, JobWebhookRequest
from .webhooks import WebhookDeliveryResult, deliver_webhook

SERVICE_JOBS_STORE_ENV_VAR = "OPENMED_SERVICE_JOBS_STORE_PATH"
SERVICE_JOBS_TTL_ENV_VAR = "OPENMED_SERVICE_JOBS_TTL_SECONDS"
SERVICE_JOBS_WORKERS_ENV_VAR = "OPENMED_SERVICE_JOBS_WORKERS"
DEFAULT_JOBS_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOBS_WORKERS = 2
TERMINAL_STATUSES = frozenset({"done", "failed"})


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
            self.cleanup_expired_locked()
            job_id = str(record["id"])
            self._records[job_id] = _copy_record(record)
            self._persist_locked()
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
            record = self._records[job_id]
            record.update(_copy_record(changes))
            record["updated_at"] = _isoformat(self.clock())
            self._persist_locked()
            return _copy_record(record)

    def cleanup_expired(self) -> None:
        """Remove terminal records whose TTL has elapsed."""
        with self._lock:
            if self.cleanup_expired_locked():
                self._persist_locked()

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
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        jobs = raw.get("jobs")
        if not isinstance(jobs, dict):
            return {}
        return {
            str(job_id): dict(record)
            for job_id, record in jobs.items()
            if isinstance(record, dict)
        }

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"jobs": self._records}
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            temp_name = handle.name
        os.replace(temp_name, self.path)


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
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Job queue is shutting down")
            record = self._new_record(payload)
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

    def _new_record(self, payload: DeidentifyJobRequest) -> dict[str, Any]:
        now = self.clock()
        expires_at = datetime.fromtimestamp(
            now.timestamp() + self.store.ttl_seconds,
            tz=timezone.utc,
        )
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
            "expires_at": _isoformat(expires_at),
        }

    def _run_job(self, item: _JobWorkItem) -> None:
        started_at = self.clock()
        self.store.update(
            item.job_id,
            status="running",
            started_at=_isoformat(started_at),
            progress_percent=0.0,
        )
        summary = _JobSummary(total=len(item.payload.documents))
        error: Optional[dict[str, str]] = None

        for index, document in enumerate(item.payload.documents):
            document_id = _document_id(index, document)
            try:
                result = self._deidentify_document(item.payload, document)
            except Exception as exc:
                summary.failed_count += 1
                error = _safe_error(exc)
            else:
                summary.add_result(document_id, result)

            self.store.update(item.job_id, **summary.to_progress_record())

        completed_at = self.clock()
        status = "failed" if summary.failed_count else "done"
        final_record = self.store.update(
            item.job_id,
            status=status,
            progress_percent=100.0,
            error=error,
            completed_at=_isoformat(completed_at),
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
    processed_count: int = 0
    failed_count: int = 0
    label_histogram: dict[str, int] = field(default_factory=dict)
    spans: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, document_id: str, result: Any) -> None:
        """Add one de-identification result without retaining raw text."""
        self.processed_count += 1
        for entity in getattr(result, "pii_entities", []) or []:
            label = _entity_label(entity)
            self.label_histogram[label] = self.label_histogram.get(label, 0) + 1
            self.spans.append(_entity_span(document_id, entity, label))

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
