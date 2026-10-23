"""Asynchronous FHIR Bulk Data job semantics for the local REST service."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event
from typing import Any, Callable, ParamSpec, TypeVar

import httpx

from openmed.interop.fhir.bulk import (
    BulkDataGateway,
    BulkGatewayConfig,
    BulkJobCancelled,
    FHIRBulkJobReport,
    _safe_resource_type,
)

from .job_bounds import (
    DEFAULT_MAX_RETAINED_JOBS,
    DEFAULT_MAX_RUNNING_JOBS,
    DEFAULT_TERMINAL_TTL_SECONDS,
    BulkJobBounds,
    ensure_job_capacity,
    prune_job_records,
)
from .jobs import _parse_positive_int
from .smart_backend import (
    SMARTBackendBulkIngestor,
    SMARTBackendConfig,
    SMARTBackendError,
    SMARTBackendIngestionSummary,
)

_PATH_CODES = frozenset(
    {"bulk_paths_disabled", "path_invalid", "path_exists", "path_outside_root"}
)
_P = ParamSpec("_P")
_R = TypeVar("_R")
BULK_MAX_RUNNING_JOBS_ENV_VAR = "OPENMED_SERVICE_BULK_MAX_RUNNING_JOBS"
BULK_MAX_RETAINED_JOBS_ENV_VAR = "OPENMED_SERVICE_BULK_MAX_RETAINED_JOBS"
BULK_JOB_TTL_ENV_VAR = "OPENMED_SERVICE_BULK_JOB_TTL_SECONDS"


class BulkPathError(ValueError):
    """Controlled rejection of untrusted service-side filesystem authority."""

    def __init__(self, code: str) -> None:
        self.code = (
            code if type(code) is str and code in _PATH_CODES else "path_invalid"
        )
        super().__init__(self.code)


def _private_path_errors(callback: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(callback)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        code = "path_invalid"
        try:
            return callback(*args, **kwargs)
        except BulkPathError as error:
            if (
                type(error) is BulkPathError
                and type(error.code) is str
                and error.code in _PATH_CODES
            ):
                code = error.code
        except Exception:
            pass
        # Filesystem exceptions retain private filenames even with `from None`.
        raise BulkPathError(code)

    return guarded


@dataclass(frozen=True)
class BulkStoragePolicy:
    """Operator-owned local roots; request paths never grant filesystem access."""

    input_roots: tuple[Path, ...] = ()
    output_root: Path | None = None

    @classmethod
    def from_env(cls) -> "BulkStoragePolicy":
        """Read trusted root configuration once, before handling requests."""
        roots = tuple(
            Path(item.strip()).expanduser().resolve()
            for item in os.getenv("OPENMED_SERVICE_BULK_INPUT_ROOTS", "").split(",")
            if item.strip()
        )
        output = os.getenv("OPENMED_SERVICE_BULK_OUTPUT_ROOT")
        return cls(roots, Path(output).expanduser().resolve() if output else None)

    @_private_path_errors
    def prepare(
        self,
        *,
        input_dir: str | None,
        output_dir: str,
        checkpoint_path: str | None,
        job_id: str,
    ) -> tuple[Path | None, Path, Path]:
        """Validate all inputs, then reserve a new private job-owned output."""
        if self.output_root is None or (input_dir is not None and not self.input_roots):
            raise BulkPathError("bulk_paths_disabled")
        if (
            type(job_id) is not str
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", job_id)
            or ".." in job_id
        ):
            raise BulkPathError("path_invalid")
        if checkpoint_path is not None:
            raise BulkPathError("path_invalid")
        output_parts = _relative_path_parts(output_dir)
        input_parts = _relative_path_parts(input_dir) if input_dir is not None else None
        source = None
        if input_parts is not None:
            for root in self.input_roots:
                candidate = _confined_path(root, input_parts)
                if candidate.is_dir():
                    source = candidate
                    break
            if source is None:
                raise BulkPathError("path_invalid")
            # BulkDataGateway walks recursively: reject sibling file and directory
            # links too, not just a link in the submitted directory itself.
            if any(path.is_symlink() for path in source.rglob("*")):
                raise BulkPathError("path_outside_root")
        destination = _confined_path(self.output_root, output_parts)
        if destination.exists():
            raise BulkPathError("path_exists")
        if not destination.parent.is_dir():
            raise BulkPathError("path_invalid")
        try:
            destination.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            raise BulkPathError("path_exists") from None
        except OSError:
            raise BulkPathError("path_invalid") from None
        return source, destination, destination / f".checkpoint-{job_id}.json"


def bulk_job_bounds_from_env() -> BulkJobBounds:
    """Read bulk and SMART job ceilings once, before handling requests.

    Each manager applies the same ceilings independently. Values must be
    positive integers, and the running ceiling must not exceed the retained
    cap so admission can always make room for a new record.

    Returns:
        Validated admission and retention bounds.

    Raises:
        ValueError: If a value is malformed or the ceilings are inconsistent.
    """

    running = _parse_positive_int(
        os.getenv(BULK_MAX_RUNNING_JOBS_ENV_VAR),
        env_var=BULK_MAX_RUNNING_JOBS_ENV_VAR,
        default=DEFAULT_MAX_RUNNING_JOBS,
    )
    retained = _parse_positive_int(
        os.getenv(BULK_MAX_RETAINED_JOBS_ENV_VAR),
        env_var=BULK_MAX_RETAINED_JOBS_ENV_VAR,
        default=DEFAULT_MAX_RETAINED_JOBS,
    )
    ttl_seconds = _parse_positive_int(
        os.getenv(BULK_JOB_TTL_ENV_VAR),
        env_var=BULK_JOB_TTL_ENV_VAR,
        default=int(DEFAULT_TERMINAL_TTL_SECONDS),
    )
    if running > retained:
        raise ValueError(
            f"{BULK_MAX_RUNNING_JOBS_ENV_VAR} must not exceed "
            f"{BULK_MAX_RETAINED_JOBS_ENV_VAR}"
        )
    return BulkJobBounds(
        max_running_jobs=running,
        max_retained_jobs=retained,
        terminal_ttl_seconds=ttl_seconds,
    )


def _relative_path_parts(value: str) -> tuple[str, ...]:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or "~" in value
        or "\x00" in value
    ):
        raise BulkPathError("path_invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise BulkPathError("path_invalid")
    return posix.parts


def _confined_path(root: Path, parts: tuple[str, ...]) -> Path:
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise BulkPathError("path_outside_root")
    try:
        current.resolve().relative_to(root)
    except (ValueError, OSError, RuntimeError):
        raise BulkPathError("path_outside_root") from None
    return current


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
    """Manage asynchronous local and SMART Bulk Data jobs in memory.

    Admission and retention are bounded by ``BulkJobBounds``: starts beyond
    the running ceiling raise ``BulkJobsSaturatedError`` before a task is
    created, and terminal records expire or are evicted oldest first.

    Args:
        transport: Optional injected HTTP transport for SMART requests.
        client_assertion_builder: Optional injected client-assertion signer.
        bounds: Running, retained and TTL ceilings; defaults apply when omitted.
        clock: Seconds clock for job timestamps and terminal-record expiry.
    """

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
        client_assertion_builder: Any = None,
        bounds: BulkJobBounds | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._transport = transport
        self._client_assertion_builder = client_assertion_builder
        self._bounds = bounds if bounds is not None else BulkJobBounds()
        self._clock = clock
        self._jobs: dict[str, _JobRecord] = {}

    @property
    def bounds(self) -> BulkJobBounds:
        """Return the admission and retention ceilings for this manager."""

        return self._bounds

    def ensure_capacity(self) -> None:
        """Raise ``BulkJobsSaturatedError`` if no running slot is free."""

        prune_job_records(self._jobs, self._bounds, self._clock())
        ensure_job_capacity(self._jobs, self._bounds, kind="fhir_bulk")

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
        now = self._clock()
        prune_job_records(self._jobs, self._bounds, now)
        if run_id in self._jobs:
            raise ValueError("job_id already exists")
        ensure_job_capacity(self._jobs, self._bounds, kind="fhir_bulk")
        prune_job_records(self._jobs, self._bounds, now, reserve=1)
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
        """Return one job's current status; expired jobs read as unknown."""

        prune_job_records(self._jobs, self._bounds, self._clock())
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

        prune_job_records(self._jobs, self._bounds, self._clock())
        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return await self._cancel_record(record)

    async def cancel_all(self) -> None:
        """Cancel all in-flight jobs during service shutdown."""

        records = [record for record in self._jobs.values() if not record.task.done()]
        if records:
            await asyncio.gather(*(self._cancel_record(record) for record in records))

    async def _cancel_record(self, record: _JobRecord) -> FHIRBulkJobStatus:
        if not record.task.done() and record.status.status == "running":
            record.cancel_event.set()
            record.status.status = "cancelled"
            record.status.updated_at = self._clock()
            record.task.cancel()
            await asyncio.gather(record.task, return_exceptions=True)
        return record.status

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
            record.status.updated_at = self._clock()
            return
        except BulkJobCancelled:
            record.status.status = "cancelled"
            record.status.error = None
            record.status.updated_at = self._clock()
            return
        except Exception as exc:
            record.status.status = "failed"
            record.status.error = _safe_job_error(exc)
            record.status.updated_at = self._clock()
            return

        record.status.status = "succeeded"
        record.status.report = result
        record.status.manifest = manifest
        record.status.updated_at = self._clock()

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
    "BULK_JOB_TTL_ENV_VAR",
    "BULK_MAX_RETAINED_JOBS_ENV_VAR",
    "BULK_MAX_RUNNING_JOBS_ENV_VAR",
    "FHIRBulkJobConfig",
    "FHIRBulkJobManager",
    "FHIRBulkJobStatus",
    "bulk_job_bounds_from_env",
]
