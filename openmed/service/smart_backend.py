"""SMART Backend Services bulk-export ingestion for the REST service.

The implementation keeps the SMART-on-FHIR network boundary explicit: it only
calls the configured token endpoint, system export/status endpoints, and NDJSON
file URLs that resolve to the same configured origins. Raw FHIR resources are
streamed directly through the local de-identification path and are never
checkpointed or persisted before de-identification.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx

from openmed.interop.fhir_bulk import NDJSONFileSummary, deidentify_ndjson_async
from openmed.interop.fhir_operations import Deidentifier

CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
DEFAULT_BACKEND_SERVICES_SCOPE = "system/*.read"
DEFAULT_EXPORT_PATH = "$export"
DEFAULT_MAX_INFLIGHT_DOWNLOADS = 2
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_POLICY = "hipaa_safe_harbor"
DEFAULT_METHOD = "replace"

ClientAssertionBuilder = Callable[["SMARTBackendConfig"], str]
SleepCallable = Callable[[float], Awaitable[None]]


class SMARTBackendError(RuntimeError):
    """Raised when backend-services ingestion cannot proceed safely."""


@dataclass(frozen=True)
class SMARTBackendConfig:
    """Configuration for one SMART backend-services bulk ingestion run."""

    fhir_base_url: str
    token_url: str
    client_id: str
    private_key_pem: str = field(repr=False)
    output_dir: str | Path
    checkpoint_path: str | Path | None = None
    key_id: Optional[str] = None
    scope: str = DEFAULT_BACKEND_SERVICES_SCOPE
    export_path: str = DEFAULT_EXPORT_PATH
    max_inflight_downloads: int = DEFAULT_MAX_INFLIGHT_DOWNLOADS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    policy: str = DEFAULT_POLICY
    method: str = DEFAULT_METHOD

    def __post_init__(self) -> None:
        _require_nonblank("fhir_base_url", self.fhir_base_url)
        _require_nonblank("token_url", self.token_url)
        _require_nonblank("client_id", self.client_id)
        _require_nonblank("private_key_pem", self.private_key_pem)
        _require_nonblank("scope", self.scope)
        _require_nonblank("export_path", self.export_path)
        if self.max_inflight_downloads < 1:
            raise ValueError("max_inflight_downloads must be at least 1")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        _validate_http_url("fhir_base_url", self.fhir_base_url)
        _validate_http_url("token_url", self.token_url)

    @property
    def output_path(self) -> Path:
        """Return the destination directory for de-identified NDJSON."""

        return Path(self.output_dir)

    @property
    def checkpoint_file(self) -> Path:
        """Return the durable checkpoint path for this ingestion run."""

        if self.checkpoint_path is not None:
            return Path(self.checkpoint_path)
        return self.output_path / ".openmed-smart-backend-checkpoint.json"


@dataclass(frozen=True)
class BulkDataFileDescriptor:
    """One FHIR Bulk Data output file descriptor."""

    index: int
    resource_type: str
    url: str
    count: Optional[int] = None

    @property
    def key(self) -> str:
        """Return a stable checkpoint key without exposing the source URL."""

        payload = f"{self.index}\0{self.resource_type}\0{self.url}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def source_label(self) -> str:
        """Return a PHI-safe source label for error summaries."""

        return f"{self.index:05d}-{_safe_filename_piece(self.resource_type)}"

    @property
    def output_filename(self) -> str:
        """Return the deterministic de-identified output filename."""

        return f"{self.source_label}.ndjson"


@dataclass(frozen=True)
class BulkExportManifest:
    """PHI-safe summary of a completed FHIR bulk-export status response."""

    transaction_time: str
    request_url: str
    output: tuple[BulkDataFileDescriptor, ...]
    error_count: int = 0


@dataclass(frozen=True)
class SMARTBackendFileResult:
    """PHI-safe result for one de-identified output file."""

    index: int
    resource_type: str
    output_file: str
    expected_count: Optional[int]
    lines_processed: int
    resources_deidentified: int
    blank_lines: int
    error_count: int
    output_sha256: str
    resumed: bool = False

    @classmethod
    def from_summary(
        cls,
        descriptor: BulkDataFileDescriptor,
        summary: NDJSONFileSummary,
        *,
        resumed: bool = False,
    ) -> "SMARTBackendFileResult":
        """Build a result from a streaming NDJSON de-identification summary."""

        return cls(
            index=descriptor.index,
            resource_type=descriptor.resource_type,
            output_file=Path(summary.destination).name,
            expected_count=descriptor.count,
            lines_processed=summary.lines_processed,
            resources_deidentified=summary.resources_deidentified,
            blank_lines=summary.blank_lines,
            error_count=summary.error_count,
            output_sha256=summary.output_sha256,
            resumed=resumed,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable PHI-free representation."""

        return {
            "index": self.index,
            "resource_type": self.resource_type,
            "output_file": self.output_file,
            "expected_count": self.expected_count,
            "lines_processed": self.lines_processed,
            "resources_deidentified": self.resources_deidentified,
            "blank_lines": self.blank_lines,
            "error_count": self.error_count,
            "output_sha256": self.output_sha256,
            "resumed": self.resumed,
        }

    def to_checkpoint_record(self) -> dict[str, Any]:
        """Return the durable checkpoint record for this file."""

        payload = self.to_dict()
        payload["resumed"] = False
        return payload

    @classmethod
    def from_checkpoint_record(
        cls, record: dict[str, Any], *, resumed: bool
    ) -> "SMARTBackendFileResult":
        """Load a file result from a checkpoint record."""

        return cls(
            index=int(record["index"]),
            resource_type=str(record["resource_type"]),
            output_file=str(record["output_file"]),
            expected_count=record.get("expected_count"),
            lines_processed=int(record.get("lines_processed", 0)),
            resources_deidentified=int(record.get("resources_deidentified", 0)),
            blank_lines=int(record.get("blank_lines", 0)),
            error_count=int(record.get("error_count", 0)),
            output_sha256=str(record.get("output_sha256", "")),
            resumed=resumed,
        )


@dataclass(frozen=True)
class SMARTBackendIngestionSummary:
    """PHI-free summary for one SMART backend-services ingestion job."""

    job_id: str
    status: str
    files_total: int
    files_completed: int
    resources_deidentified: int
    lines_processed: int
    error_count: int
    output_sha256: str
    max_inflight_downloads_observed: int
    files: tuple[SMARTBackendFileResult, ...] = ()
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable PHI-free job summary."""

        return {
            "job_id": self.job_id,
            "status": self.status,
            "files_total": self.files_total,
            "files_completed": self.files_completed,
            "resources_deidentified": self.resources_deidentified,
            "lines_processed": self.lines_processed,
            "error_count": self.error_count,
            "output_sha256": self.output_sha256,
            "max_inflight_downloads_observed": self.max_inflight_downloads_observed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "files": [file.to_dict() for file in self.files],
        }


class SMARTBackendBulkIngestor:
    """Run SMART backend-services auth, export polling, and NDJSON ingestion."""

    def __init__(
        self,
        config: SMARTBackendConfig,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
        client_assertion_builder: ClientAssertionBuilder | None = None,
        sleep: SleepCallable = asyncio.sleep,
        deidentifier: Deidentifier | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._client_assertion_builder = (
            client_assertion_builder or build_client_assertion
        )
        self._sleep = sleep
        self._deidentifier = deidentifier
        self._active_downloads = 0
        self._max_active_downloads = 0
        self._counter_lock = asyncio.Lock()

    async def run(self, *, job_id: str | None = None) -> SMARTBackendIngestionSummary:
        """Run the ingestion job to completion and return its PHI-free summary."""

        run_id = job_id or uuid.uuid4().hex
        started_at = time.time()
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = _Checkpoint.load(self.config.checkpoint_file)

        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.config.request_timeout_seconds,
            follow_redirects=False,
        ) as client:
            token = await self._fetch_access_token(client)
            status_url = await self._kickoff_export(client, token)
            manifest = await self._poll_export_manifest(client, token, status_url)
            files = await self._download_manifest_files(
                client,
                token,
                manifest,
                checkpoint,
            )

        finished_at = time.time()
        ordered = tuple(sorted(files, key=lambda item: item.index))
        return SMARTBackendIngestionSummary(
            job_id=run_id,
            status="succeeded",
            files_total=len(ordered),
            files_completed=len(ordered),
            resources_deidentified=sum(file.resources_deidentified for file in ordered),
            lines_processed=sum(file.lines_processed for file in ordered),
            error_count=sum(file.error_count for file in ordered),
            output_sha256=_aggregate_output_digest(ordered),
            max_inflight_downloads_observed=self._max_active_downloads,
            files=ordered,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def _fetch_access_token(self, client: httpx.AsyncClient) -> str:
        self._assert_allowed_url(self.config.token_url)
        assertion = self._client_assertion_builder(self.config)
        response = await client.post(
            self.config.token_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": self.config.scope,
                "client_assertion_type": CLIENT_ASSERTION_TYPE,
                "client_assertion": assertion,
            },
        )
        _raise_for_status(response, "token endpoint")
        payload = _json_response(response, "token endpoint")
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise SMARTBackendError("token endpoint did not return an access token")
        return token

    async def _kickoff_export(
        self,
        client: httpx.AsyncClient,
        token: str,
    ) -> str:
        export_url = self._export_url()
        self._assert_allowed_url(export_url)
        response = await client.get(
            export_url,
            headers={
                "Accept": "application/fhir+json, application/json",
                "Authorization": f"Bearer {token}",
                "Prefer": "respond-async",
            },
            params={"_outputFormat": "application/fhir+ndjson"},
        )
        if response.status_code != 202:
            _raise_for_status(response, "bulk export kickoff")
            raise SMARTBackendError(
                "bulk export kickoff did not return HTTP 202 Accepted"
            )
        status_url = response.headers.get("Content-Location")
        if not status_url:
            raise SMARTBackendError(
                "bulk export kickoff did not return Content-Location"
            )
        status_url = urljoin(str(response.url), status_url)
        self._assert_allowed_url(status_url)
        return status_url

    async def _poll_export_manifest(
        self,
        client: httpx.AsyncClient,
        token: str,
        status_url: str,
    ) -> BulkExportManifest:
        while True:
            self._assert_allowed_url(status_url)
            response = await client.get(
                status_url,
                headers={
                    "Accept": "application/fhir+json, application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            if response.status_code == 202:
                await self._sleep(_retry_after_seconds(response, self.config))
                continue
            if response.status_code in {429, 503} and "Retry-After" in response.headers:
                await self._sleep(_retry_after_seconds(response, self.config))
                continue
            _raise_for_status(response, "bulk export status")
            return self._parse_manifest(_json_response(response, "bulk export status"))

    async def _download_manifest_files(
        self,
        client: httpx.AsyncClient,
        token: str,
        manifest: BulkExportManifest,
        checkpoint: "_Checkpoint",
    ) -> list[SMARTBackendFileResult]:
        results: list[SMARTBackendFileResult] = []
        pending: list[BulkDataFileDescriptor] = []
        for descriptor in manifest.output:
            destination = self._destination_for(descriptor)
            if checkpoint.is_completed(descriptor, destination):
                results.append(checkpoint.result_for(descriptor, resumed=True))
            else:
                pending.append(descriptor)

        if not pending:
            return results

        queue: asyncio.Queue[BulkDataFileDescriptor] = asyncio.Queue()
        for descriptor in pending:
            queue.put_nowait(descriptor)

        async def worker() -> None:
            while True:
                try:
                    descriptor = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    result = await self._download_one(
                        client,
                        token,
                        descriptor,
                        checkpoint,
                    )
                    results.append(result)
                finally:
                    queue.task_done()

        worker_count = min(self.config.max_inflight_downloads, len(pending))
        await asyncio.gather(*(worker() for _ in range(worker_count)))
        return results

    async def _download_one(
        self,
        client: httpx.AsyncClient,
        token: str,
        descriptor: BulkDataFileDescriptor,
        checkpoint: "_Checkpoint",
    ) -> SMARTBackendFileResult:
        self._assert_allowed_url(descriptor.url)
        destination = self._destination_for(descriptor)
        partial = destination.with_name(f"{destination.name}.part")
        partial.unlink(missing_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)

        await self._enter_download()
        try:
            async with client.stream(
                "GET",
                descriptor.url,
                headers={
                    "Accept": (
                        "application/fhir+ndjson, application/ndjson, "
                        "application/x-ndjson, application/json"
                    ),
                    "Authorization": f"Bearer {token}",
                },
            ) as response:
                _raise_for_status(response, "bulk export file")
                summary = await deidentify_ndjson_async(
                    response.aiter_lines(),
                    partial,
                    source=descriptor.source_label,
                    destination=destination,
                    policy=self.config.policy,
                    method=self.config.method,
                    deidentifier=self._deidentifier,
                )
            os.replace(partial, destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        finally:
            await self._leave_download()

        result = SMARTBackendFileResult.from_summary(descriptor, summary)
        checkpoint.mark_completed(descriptor, result)
        return result

    async def _enter_download(self) -> None:
        async with self._counter_lock:
            self._active_downloads += 1
            self._max_active_downloads = max(
                self._max_active_downloads,
                self._active_downloads,
            )

    async def _leave_download(self) -> None:
        async with self._counter_lock:
            self._active_downloads -= 1

    def _parse_manifest(self, payload: dict[str, Any]) -> BulkExportManifest:
        output = payload.get("output")
        if not isinstance(output, list):
            raise SMARTBackendError("bulk export status did not contain output files")

        descriptors: list[BulkDataFileDescriptor] = []
        for index, raw_descriptor in enumerate(output):
            if not isinstance(raw_descriptor, dict):
                raise SMARTBackendError("bulk export output descriptor is invalid")
            resource_type = raw_descriptor.get("type")
            url = raw_descriptor.get("url")
            if not isinstance(resource_type, str) or not resource_type:
                raise SMARTBackendError("bulk export output descriptor is missing type")
            if not isinstance(url, str) or not url:
                raise SMARTBackendError("bulk export output descriptor is missing url")
            absolute_url = urljoin(self.config.fhir_base_url, url)
            self._assert_allowed_url(absolute_url)
            count = raw_descriptor.get("count")
            descriptors.append(
                BulkDataFileDescriptor(
                    index=index,
                    resource_type=resource_type,
                    url=absolute_url,
                    count=count if isinstance(count, int) else None,
                )
            )

        errors = payload.get("error") or []
        error_count = len(errors) if isinstance(errors, list) else 0
        return BulkExportManifest(
            transaction_time=str(payload.get("transactionTime", "")),
            request_url=str(payload.get("request", "")),
            output=tuple(descriptors),
            error_count=error_count,
        )

    def _destination_for(self, descriptor: BulkDataFileDescriptor) -> Path:
        return self.config.output_path / descriptor.output_filename

    def _export_url(self) -> str:
        base_url = self.config.fhir_base_url.rstrip("/") + "/"
        if self.config.export_path.startswith(("http://", "https://")):
            return self.config.export_path
        return urljoin(base_url, self.config.export_path.lstrip("/"))

    def _assert_allowed_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SMARTBackendError("FHIR endpoint URL must be absolute HTTP(S)")
        if _origin(url) not in {
            _origin(self.config.fhir_base_url),
            _origin(self.config.token_url),
        }:
            raise SMARTBackendError(
                "bulk export referenced a URL outside configured FHIR origins"
            )


@dataclass
class SMARTBackendJobStatus:
    """Current PHI-free state for an async service ingestion job."""

    job_id: str
    status: str
    created_at: float
    updated_at: float
    summary: SMARTBackendIngestionSummary | None = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable PHI-free status payload."""

        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "error": self.error,
        }


@dataclass
class _JobRecord:
    status: SMARTBackendJobStatus
    task: asyncio.Task[Any]


class SMARTBackendJobManager:
    """In-memory async job manager for REST-triggered ingestion runs."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
        client_assertion_builder: ClientAssertionBuilder | None = None,
    ) -> None:
        self._transport = transport
        self._client_assertion_builder = client_assertion_builder
        self._jobs: dict[str, _JobRecord] = {}

    def start(
        self,
        config: SMARTBackendConfig,
        *,
        deidentifier: Deidentifier | None = None,
        job_id: str | None = None,
    ) -> SMARTBackendJobStatus:
        """Start one background ingestion job."""

        run_id = job_id or uuid.uuid4().hex
        if run_id in self._jobs:
            raise ValueError("job_id already exists")
        now = time.time()
        status = SMARTBackendJobStatus(
            job_id=run_id,
            status="running",
            created_at=now,
            updated_at=now,
        )
        task = asyncio.create_task(self._run(run_id, config, deidentifier))
        self._jobs[run_id] = _JobRecord(status=status, task=task)
        return status

    def get(self, job_id: str) -> SMARTBackendJobStatus:
        """Return the current job status."""

        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record.status

    def summary(self, job_id: str) -> SMARTBackendIngestionSummary:
        """Return the completed job summary."""

        status = self.get(job_id)
        if status.summary is None:
            raise ValueError("ingestion job has not completed")
        return status.summary

    async def cancel_all(self) -> None:
        """Cancel any still-running background jobs during service shutdown."""

        tasks = [
            record.task for record in self._jobs.values() if not record.task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(
        self,
        job_id: str,
        config: SMARTBackendConfig,
        deidentifier: Deidentifier | None,
    ) -> None:
        record = self._jobs[job_id]
        try:
            ingestor = SMARTBackendBulkIngestor(
                config,
                transport=self._transport,
                client_assertion_builder=self._client_assertion_builder,
                deidentifier=deidentifier,
            )
            summary = await ingestor.run(job_id=job_id)
        except Exception as exc:
            record.status.status = "failed"
            record.status.error = _sanitize_exception(exc)
            record.status.updated_at = time.time()
        else:
            record.status.status = "succeeded"
            record.status.summary = summary
            record.status.updated_at = time.time()


def build_client_assertion(config: SMARTBackendConfig) -> str:
    """Build a signed SMART Backend Services JWT client assertion."""

    now = int(time.time())
    header: dict[str, Any] = {"alg": "RS384", "typ": "JWT"}
    if config.key_id:
        header["kid"] = config.key_id
    claims = {
        "iss": config.client_id,
        "sub": config.client_id,
        "aud": config.token_url,
        "iat": now,
        "exp": now + 300,
        "jti": secrets.token_urlsafe(24),
    }
    signing_input = f"{_b64json(header)}.{_b64json(claims)}".encode("ascii")
    signature = _sign_rs384(signing_input, config.private_key_pem)
    return f"{signing_input.decode('ascii')}.{_b64url(signature)}"


def _sign_rs384(signing_input: bytes, private_key_pem: str) -> bytes:
    n, d = _load_rsa_private_numbers(private_key_pem)
    digest = hashlib.sha384(signing_input).digest()
    digest_info = bytes.fromhex("3041300d060960864801650304020205000430") + digest
    modulus_size = (n.bit_length() + 7) // 8
    if modulus_size < len(digest_info) + 11:
        raise SMARTBackendError("RSA private key is too small for RS384 signing")
    encoded = b"\x00\x01" + (b"\xff" * (modulus_size - len(digest_info) - 3))
    encoded += b"\x00" + digest_info
    signature = pow(int.from_bytes(encoded, "big"), d, n)
    return signature.to_bytes(modulus_size, "big")


def _load_rsa_private_numbers(private_key_pem: str) -> tuple[int, int]:
    der, label = _pem_to_der(private_key_pem)
    if label == "RSA PRIVATE KEY":
        return _parse_pkcs1_private_key(der)
    if label == "PRIVATE KEY":
        reader = _DERReader(der)
        seq = reader.read_constructed(0x30)
        seq.read_integer()
        seq.read_tlv()
        private_key = seq.read_octet_string()
        return _parse_pkcs1_private_key(private_key)
    raise SMARTBackendError("only unencrypted RSA private keys are supported")


def _parse_pkcs1_private_key(der: bytes) -> tuple[int, int]:
    reader = _DERReader(der)
    seq = reader.read_constructed(0x30)
    seq.read_integer()
    n = seq.read_integer()
    seq.read_integer()
    d = seq.read_integer()
    return n, d


class _DERReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def read_constructed(self, tag: int) -> "_DERReader":
        actual_tag, value = self.read_tlv()
        if actual_tag != tag:
            raise SMARTBackendError("invalid RSA private key encoding")
        return _DERReader(value)

    def read_integer(self) -> int:
        tag, value = self.read_tlv()
        if tag != 0x02:
            raise SMARTBackendError("invalid RSA private key integer")
        return int.from_bytes(value, "big", signed=False)

    def read_octet_string(self) -> bytes:
        tag, value = self.read_tlv()
        if tag != 0x04:
            raise SMARTBackendError("invalid RSA private key payload")
        return value

    def read_tlv(self) -> tuple[int, bytes]:
        if self.offset >= len(self.data):
            raise SMARTBackendError("truncated RSA private key")
        tag = self.data[self.offset]
        self.offset += 1
        length = self._read_length()
        end = self.offset + length
        if end > len(self.data):
            raise SMARTBackendError("truncated RSA private key")
        value = self.data[self.offset : end]
        self.offset = end
        return tag, value

    def _read_length(self) -> int:
        if self.offset >= len(self.data):
            raise SMARTBackendError("truncated RSA private key")
        first = self.data[self.offset]
        self.offset += 1
        if first < 0x80:
            return first
        size = first & 0x7F
        if size == 0 or size > 4 or self.offset + size > len(self.data):
            raise SMARTBackendError("invalid RSA private key length")
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        return int.from_bytes(raw, "big")


@dataclass
class _Checkpoint:
    path: Path
    completed: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "_Checkpoint":
        if not path.exists():
            return cls(path=path, completed={})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SMARTBackendError("checkpoint file is not valid JSON") from exc
        completed = payload.get("completed", {})
        if not isinstance(completed, dict):
            raise SMARTBackendError("checkpoint file has invalid completed state")
        return cls(path=path, completed=dict(completed))

    def is_completed(
        self, descriptor: BulkDataFileDescriptor, destination: Path
    ) -> bool:
        return descriptor.key in self.completed and destination.is_file()

    def result_for(
        self,
        descriptor: BulkDataFileDescriptor,
        *,
        resumed: bool,
    ) -> SMARTBackendFileResult:
        record = self.completed[descriptor.key]
        return SMARTBackendFileResult.from_checkpoint_record(
            record,
            resumed=resumed,
        )

    def mark_completed(
        self,
        descriptor: BulkDataFileDescriptor,
        result: SMARTBackendFileResult,
    ) -> None:
        self.completed[descriptor.key] = result.to_checkpoint_record()
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "completed": self.completed,
        }
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def _retry_after_seconds(
    response: httpx.Response,
    config: SMARTBackendConfig,
) -> float:
    raw_value = response.headers.get("Retry-After")
    if not raw_value:
        return config.poll_interval_seconds
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw_value)
        except (TypeError, ValueError):
            return config.poll_interval_seconds
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _raise_for_status(response: httpx.Response, endpoint_name: str) -> None:
    if 200 <= response.status_code < 300:
        return
    raise SMARTBackendError(f"{endpoint_name} returned HTTP {response.status_code}")


def _json_response(response: httpx.Response, endpoint_name: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise SMARTBackendError(f"{endpoint_name} did not return JSON") from exc
    if not isinstance(payload, dict):
        raise SMARTBackendError(f"{endpoint_name} JSON payload must be an object")
    return payload


def _aggregate_output_digest(files: tuple[SMARTBackendFileResult, ...]) -> str:
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda item: item.index):
        digest.update(str(file.index).encode("ascii"))
        digest.update(b"\0")
        digest.update(file.output_sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _b64json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _b64url(raw)


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _pem_to_der(pem: str) -> tuple[bytes, str]:
    match = re.search(
        r"-----BEGIN ([A-Z ]+)-----(.*?)-----END \1-----",
        pem,
        flags=re.DOTALL,
    )
    if match is None:
        raise SMARTBackendError("private key must be PEM encoded")
    label = match.group(1)
    body = "".join(match.group(2).strip().split())
    try:
        return base64.b64decode(body, validate=True), label
    except ValueError as exc:
        raise SMARTBackendError("private key PEM body is not valid base64") from exc


def _safe_filename_piece(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or "resource"


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _validate_http_url(name: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTP(S) URL")


def _require_nonblank(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")


def _sanitize_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__
