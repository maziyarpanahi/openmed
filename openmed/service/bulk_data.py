"""Asynchronous FHIR Bulk Data job semantics for the local REST service."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

import httpx

from openmed.interop.fhir.bulk import (
    BulkDataGateway,
    BulkGatewayConfig,
    BulkJobCancelled,
    FHIRBulkJobReport,
    _safe_resource_type,
)

from .smart_backend import (
    SMARTBackendBulkIngestor,
    SMARTBackendConfig,
    SMARTBackendError,
    SMARTBackendIngestionSummary,
)


@dataclass(frozen=True)
class FHIRBulkJobConfig:
    """PHI-safe configuration for a local or SMART-backed bulk job."""

    output_dir: str | Path
    input_dir: str | Path | None = None
    checkpoint_path: str | Path | None = None
    policy: str = "hipaa_safe_harbor"
    method: str = "replace"
    max_buffered_resources: int = 1
    max_inflight_downloads: int = 2
    poll_interval_seconds: float = 1.0
    request_timeout_seconds: float = 30.0
    fhir_base_url: str | None = None
    token_url: str | None = None
    client_id: str | None = None
    private_key_pem: str | None = field(default=None, repr=False)
    key_id: str | None = None
    scope: str = "system/*.read"
    export_path: str = "$export"

    def __post_init__(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("output_dir must not be blank")
        if self.max_buffered_resources < 1:
            raise ValueError("max_buffered_resources must be at least 1")
        if self.max_inflight_downloads < 1:
            raise ValueError("max_inflight_downloads must be at least 1")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

    @property
    def is_local(self) -> bool:
        """Return whether this job processes a local synthetic export."""

        return self.input_dir is not None

    def to_smart_config(self) -> SMARTBackendConfig:
        """Build the optional SMART backend configuration on demand."""

        required = {
            "fhir_base_url": self.fhir_base_url,
            "token_url": self.token_url,
            "client_id": self.client_id,
            "private_key_pem": self.private_key_pem,
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in required.values()
        ):
            raise ValueError("SMART backend configuration is incomplete")
        return SMARTBackendConfig(
            fhir_base_url=self.fhir_base_url,
            token_url=self.token_url,
            client_id=self.client_id,
            private_key_pem=self.private_key_pem,
            output_dir=self.output_dir,
            checkpoint_path=self.checkpoint_path,
            key_id=self.key_id,
            scope=self.scope,
            export_path=self.export_path,
            max_inflight_downloads=self.max_inflight_downloads,
            poll_interval_seconds=self.poll_interval_seconds,
            request_timeout_seconds=self.request_timeout_seconds,
            policy=self.policy,
            method=self.method,
            max_buffered_resources=self.max_buffered_resources,
        )


@dataclass
class FHIRBulkJobStatus:
    """Current PHI-free state of a Bulk Data job."""

    job_id: str
    status: str
    created_at: float
    updated_at: float
    manifest: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a status payload safe to expose from polling endpoints."""

        report = self.report
        progress = None
        if isinstance(report, dict):
            progress = {
                key: report[key]
                for key in (
                    "files_total",
                    "files_completed",
                    "resources_deidentified",
                    "lines_processed",
                    "rejection_count",
                    "peak_buffered_resources",
                )
                if key in report
            }
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": progress,
            "manifest": self.manifest,
            "report": report,
            "error": self.error,
        }


@dataclass
class _JobRecord:
    status: FHIRBulkJobStatus
    task: asyncio.Task[Any]
    cancel_event: Event


class FHIRBulkJobManager:
    """Manage asynchronous local and SMART Bulk Data jobs in memory."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
        client_assertion_builder: Any = None,
    ) -> None:
        self._transport = transport
        self._client_assertion_builder = client_assertion_builder
        self._jobs: dict[str, _JobRecord] = {}

    def start(
        self,
        config: FHIRBulkJobConfig,
        *,
        deidentifier: Any = None,
        job_id: str | None = None,
    ) -> FHIRBulkJobStatus:
        """Start one asynchronous job and return its polling status."""

        self._validate_config(config)
        run_id = job_id or uuid.uuid4().hex
        if run_id in self._jobs:
            raise ValueError("job_id already exists")
        now = time.time()
        status = FHIRBulkJobStatus(
            job_id=run_id,
            status="running",
            created_at=now,
            updated_at=now,
        )
        cancel_event = Event()
        task = asyncio.create_task(self._run(run_id, config, deidentifier))
        self._jobs[run_id] = _JobRecord(
            status=status,
            task=task,
            cancel_event=cancel_event,
        )
        return status

    def get(self, job_id: str) -> FHIRBulkJobStatus:
        """Return one job's current status."""

        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record.status

    def report(self, job_id: str) -> dict[str, Any]:
        """Return the completed PHI-free report."""

        status = self.get(job_id)
        if status.report is None:
            raise ValueError("bulk job has not completed")
        return dict(status.report)

    def manifest(self, job_id: str) -> dict[str, Any]:
        """Return the completed PHI-free output manifest."""

        status = self.get(job_id)
        if status.manifest is None:
            raise ValueError("bulk job has not completed")
        return dict(status.manifest)

    async def cancel(self, job_id: str) -> FHIRBulkJobStatus:
        """Cancel one running job and remove any uncommitted local output."""

        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if not record.task.done() and record.status.status == "running":
            record.cancel_event.set()
            record.status.status = "cancelled"
            record.status.updated_at = time.time()
            record.task.cancel()
            await asyncio.gather(record.task, return_exceptions=True)
        return record.status

    async def cancel_all(self) -> None:
        """Cancel all in-flight jobs during service shutdown."""

        jobs = [
            job_id for job_id, record in self._jobs.items() if not record.task.done()
        ]
        if jobs:
            await asyncio.gather(*(self.cancel(job_id) for job_id in jobs))

    async def _run(
        self,
        job_id: str,
        config: FHIRBulkJobConfig,
        deidentifier: Any,
    ) -> None:
        record = self._jobs.get(job_id)
        if record is None:
            return
        try:
            if config.is_local:
                gateway = BulkDataGateway(
                    BulkGatewayConfig(
                        input_dir=config.input_dir,
                        output_dir=config.output_dir,
                        checkpoint_path=config.checkpoint_path,
                        policy=config.policy,
                        method=config.method,
                        max_buffered_resources=config.max_buffered_resources,
                    ),
                    deidentifier=deidentifier,
                )
                report = await asyncio.to_thread(
                    gateway.export,
                    cancel_event=record.cancel_event,
                    job_id=job_id,
                )
                result = report.to_dict()
                manifest = _manifest_from_local_report(report)
            else:
                ingestor = SMARTBackendBulkIngestor(
                    config.to_smart_config(),
                    transport=self._transport,
                    client_assertion_builder=self._client_assertion_builder,
                    deidentifier=deidentifier,
                )
                summary = await ingestor.run(job_id=job_id)
                result = _report_from_smart_summary(summary)
                manifest = _manifest_from_smart_summary(summary)
        except asyncio.CancelledError:
            record.status.status = "cancelled"
            record.status.error = None
            record.status.updated_at = time.time()
            return
        except BulkJobCancelled:
            record.status.status = "cancelled"
            record.status.error = None
            record.status.updated_at = time.time()
            return
        except Exception as exc:
            record.status.status = "failed"
            record.status.error = _safe_job_error(exc)
            record.status.updated_at = time.time()
            return

        record.status.status = "succeeded"
        record.status.report = result
        record.status.manifest = manifest
        record.status.updated_at = time.time()

    @staticmethod
    def _validate_config(config: FHIRBulkJobConfig) -> None:
        if config.is_local:
            if not str(config.input_dir).strip():
                raise ValueError("input_dir must not be blank")
            return
        config.to_smart_config()


def _manifest_from_local_report(report: FHIRBulkJobReport) -> dict[str, Any]:
    output = []
    for file in report.summary.files:
        resource_type = _resource_type_from_label(file.source)
        output.append(
            {
                "type": resource_type,
                "file": _safe_file_label(file.destination),
                "url": _safe_file_label(file.destination),
                "count": file.resources_deidentified,
                "sha256": file.output_sha256,
            }
        )
    return {
        "bulk_data_version": report.summary.to_dict()["bulk_data_version"],
        "transactionTime": _transaction_time(report.summary.started_at),
        "request": "local",
        "requiresAccessToken": False,
        "status": report.status,
        "output": output,
        "error": [
            {
                "reason": rejection.reason,
                "resource_sha256": rejection.resource_sha256,
            }
            for rejection in report.summary.rejections
        ],
    }


def _manifest_from_smart_summary(
    summary: SMARTBackendIngestionSummary,
) -> dict[str, Any]:
    return {
        "bulk_data_version": summary.bulk_data_version,
        "transactionTime": _transaction_time(summary.started_at),
        "request": "smart-backend",
        "requiresAccessToken": False,
        "status": summary.status,
        "output": [
            {
                "type": file.resource_type,
                "file": file.output_file,
                "url": file.output_file,
                "count": file.resources_deidentified,
                "sha256": file.output_sha256,
            }
            for file in summary.files
        ],
        "error": [
            {
                "reason": rejection.reason,
                "resource_sha256": rejection.resource_sha256,
            }
            for file in summary.files
            for rejection in file.rejections
        ],
    }


def _report_from_smart_summary(summary: SMARTBackendIngestionSummary) -> dict[str, Any]:
    payload = summary.to_dict()
    payload["job_id"] = summary.job_id
    return payload


def _resource_type_from_label(value: str) -> str:
    label = Path(value).stem
    return _safe_resource_type(label)


def _transaction_time(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _safe_file_label(value: str | Path) -> str:
    name = Path(str(value)).name or "output.ndjson"
    stem = Path(name).stem
    if _safe_resource_type(stem) != "unknown":
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"file-{digest}.ndjson"


def _safe_job_error(exc: BaseException) -> str:
    if isinstance(exc, SMARTBackendError):
        return str(exc)
    if isinstance(exc, (httpx.HTTPError, OSError)):
        return "remote SMART endpoint unavailable"
    if isinstance(exc, ValueError):
        message = str(exc)
        if message in {
            "input export path is not a directory",
            "input_dir must not be blank",
            "SMART backend configuration is incomplete",
        }:
            return message
    return f"{exc.__class__.__name__} during bulk processing"


__all__ = [
    "FHIRBulkJobConfig",
    "FHIRBulkJobManager",
    "FHIRBulkJobStatus",
]
