"""Tests for SMART backend-services bulk-export ingestion."""

from __future__ import annotations

import asyncio
import json
import resource
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient

from openmed.service.app import create_app
from openmed.service.smart_backend import (
    SMARTBackendBulkIngestor,
    SMARTBackendConfig,
    SMARTBackendError,
    SMARTBackendFileResult,
    SMARTBackendIngestionSummary,
    SMARTBackendJobStatus,
)


@dataclass
class _FakeResult:
    deidentified_text: str


def _fake_deidentify(text: str, **_: Any) -> _FakeResult:
    redacted = text.replace("Jane Roe", "[NAME]").replace("555-0100", "[PHONE]")
    return _FakeResult(deidentified_text=redacted)


async def _no_sleep(_: float) -> None:
    return None


class _GeneratedNDJSONStream(httpx.AsyncByteStream):
    def __init__(
        self,
        server: "_FakeBulkServer",
        file_index: int,
        resource_count: int,
    ) -> None:
        self.server = server
        self.file_index = file_index
        self.resource_count = resource_count

    async def __aiter__(self):
        self.server.file_inflight += 1
        self.server.max_file_inflight = max(
            self.server.max_file_inflight,
            self.server.file_inflight,
        )
        try:
            for resource_index in range(self.resource_count):
                fail_after = self.server.fail_once_after.get(self.file_index)
                if fail_after is not None and resource_index >= fail_after:
                    self.server.fail_once_after.pop(self.file_index, None)
                    raise httpx.ReadError("simulated interrupted file stream")
                if self.server.stream_delay_seconds:
                    await asyncio.sleep(self.server.stream_delay_seconds)
                resource = {
                    "resourceType": "Patient",
                    "id": f"pat-{self.file_index}-{resource_index}",
                    "name": [{"text": "Jane Roe"}],
                    "telecom": [{"system": "phone", "value": "555-0100"}],
                }
                yield (json.dumps(resource, separators=(",", ":")) + "\n").encode()
        finally:
            self.server.file_inflight -= 1


class _FakeBulkServer:
    def __init__(
        self,
        *,
        resource_counts: list[int],
        stream_delay_seconds: float = 0.0,
        external_file_url: str | None = None,
    ) -> None:
        self.resource_counts = resource_counts
        self.stream_delay_seconds = stream_delay_seconds
        self.external_file_url = external_file_url
        self.base_url = "https://fhir.example.test"
        self.token_url = "https://auth.example.test/token"
        self.status_url = f"{self.base_url}/bulk-status/1"
        self.requests: list[str] = []
        self.token_forms: list[dict[str, list[str]]] = []
        self.file_gets: Counter[int] = Counter()
        self.file_inflight = 0
        self.max_file_inflight = 0
        self.fail_once_after: dict[int, int] = {}
        self._status_polls = 0
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        if str(request.url) == self.token_url and request.method == "POST":
            body = request.content.decode("utf-8")
            self.token_forms.append(parse_qs(body))
            return httpx.Response(200, json={"access_token": "bearer-token-secret"})

        if request.url.path == "/$export":
            return httpx.Response(
                202,
                headers={"Content-Location": self.status_url, "Retry-After": "0"},
            )

        if str(request.url) == self.status_url:
            self._status_polls += 1
            if self._status_polls == 1:
                return httpx.Response(202, headers={"Retry-After": "0"})
            return httpx.Response(200, json=self._manifest())

        if request.url.path.startswith("/files/"):
            file_index = int(Path(request.url.path).stem)
            self.file_gets[file_index] += 1
            return httpx.Response(
                200,
                stream=_GeneratedNDJSONStream(
                    self,
                    file_index,
                    self.resource_counts[file_index],
                ),
            )

        return httpx.Response(404)

    def _manifest(self) -> dict[str, Any]:
        output = []
        for index, count in enumerate(self.resource_counts):
            url = f"{self.base_url}/files/{index}.ndjson"
            if self.external_file_url is not None and index == 0:
                url = self.external_file_url
            output.append({"type": "Patient", "url": url, "count": count})
        return {
            "transactionTime": "2026-06-29T00:00:00Z",
            "request": f"{self.base_url}/$export",
            "output": output,
        }


def _config(
    tmp_path: Path,
    server: _FakeBulkServer,
    *,
    output_name: str = "out",
    max_inflight_downloads: int = 2,
) -> SMARTBackendConfig:
    return SMARTBackendConfig(
        fhir_base_url=server.base_url,
        token_url=server.token_url,
        client_id="openmed-test-client",
        private_key_pem="operator-signing-key-secret",
        output_dir=tmp_path / output_name,
        max_inflight_downloads=max_inflight_downloads,
        poll_interval_seconds=0,
        request_timeout_seconds=30,
        policy="hipaa_safe_harbor",
        method="replace",
    )


def _run_ingestion(
    config: SMARTBackendConfig,
    server: _FakeBulkServer,
    *,
    job_id: str = "job",
) -> SMARTBackendIngestionSummary:
    ingestor = SMARTBackendBulkIngestor(
        config,
        transport=server.transport,
        client_assertion_builder=lambda _: "signed-client-assertion",
        sleep=_no_sleep,
        deidentifier=_fake_deidentify,
    )
    return asyncio.run(ingestor.run(job_id=job_id))


def _maxrss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value) * 1024


def test_ingests_million_resource_export_with_bounded_memory(tmp_path: Path) -> None:
    server = _FakeBulkServer(resource_counts=[100_000] * 10)
    config = _config(tmp_path, server, max_inflight_downloads=4)

    before_rss = _maxrss_bytes()
    summary = _run_ingestion(config, server, job_id="population")
    rss_growth = max(0, _maxrss_bytes() - before_rss)

    assert summary.files_total == 10
    assert summary.resources_deidentified == 1_000_000
    assert summary.lines_processed == 1_000_000
    assert summary.error_count == 0
    assert rss_growth < 512 * 1024 * 1024
    assert summary.max_inflight_downloads_observed <= 4
    assert server.max_file_inflight <= 4
    assert sum(server.file_gets.values()) == 10


def test_checkpoint_resume_redownloads_only_interrupted_file(
    tmp_path: Path,
) -> None:
    interrupted_server = _FakeBulkServer(resource_counts=[3, 3, 3])
    interrupted_server.fail_once_after[1] = 1
    config = _config(tmp_path, interrupted_server, max_inflight_downloads=1)

    with pytest.raises(httpx.ReadError):
        _run_ingestion(config, interrupted_server, job_id="interrupted")

    assert interrupted_server.file_gets == Counter({0: 1, 1: 1})
    checkpoint = config.checkpoint_file.read_text(encoding="utf-8")
    assert "00000-Patient.ndjson" in checkpoint
    assert "00001-Patient.ndjson" not in checkpoint

    resumed_summary = _run_ingestion(config, interrupted_server, job_id="resumed")
    clean_server = _FakeBulkServer(resource_counts=[3, 3, 3])
    clean_summary = _run_ingestion(
        _config(tmp_path, clean_server, output_name="clean", max_inflight_downloads=1),
        clean_server,
        job_id="clean",
    )

    assert interrupted_server.file_gets == Counter({0: 1, 1: 2, 2: 1})
    assert resumed_summary.resources_deidentified == 9
    assert resumed_summary.output_sha256 == clean_summary.output_sha256
    assert [file.resumed for file in resumed_summary.files] == [True, False, False]


def test_summary_and_logs_exclude_key_token_and_payload(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _FakeBulkServer(resource_counts=[2])
    config = _config(tmp_path, server)

    summary = _run_ingestion(config, server, job_id="secret-scan")
    rendered_summary = json.dumps(summary.to_dict(), sort_keys=True)
    rendered_logs = caplog.text
    token_form = server.token_forms[0]

    for sensitive in (
        "operator-signing-key-secret",
        "bearer-token-secret",
        "Jane Roe",
        "555-0100",
    ):
        assert sensitive not in rendered_summary
        assert sensitive not in rendered_logs
    assert token_form["client_assertion"] == ["signed-client-assertion"]
    assert "operator-signing-key-secret" not in json.dumps(token_form)


def test_backpressure_caps_concurrent_file_downloads(tmp_path: Path) -> None:
    server = _FakeBulkServer(
        resource_counts=[5] * 8,
        stream_delay_seconds=0.001,
    )
    config = _config(tmp_path, server, max_inflight_downloads=3)

    summary = _run_ingestion(config, server, job_id="backpressure")

    assert summary.resources_deidentified == 40
    assert summary.max_inflight_downloads_observed <= 3
    assert server.max_file_inflight <= 3


def test_rejects_manifest_file_urls_outside_configured_origins(
    tmp_path: Path,
) -> None:
    server = _FakeBulkServer(
        resource_counts=[1],
        external_file_url="https://storage.example.test/file.ndjson",
    )
    config = _config(tmp_path, server)

    with pytest.raises(SMARTBackendError, match="outside configured FHIR origins"):
        _run_ingestion(config, server, job_id="blocked")

    assert not server.file_gets
    assert all("storage.example.test" not in request for request in server.requests)


class _FakeManager:
    def __init__(self) -> None:
        self.started_config: SMARTBackendConfig | None = None
        self.started = False
        file_result = SMARTBackendFileResult(
            index=0,
            resource_type="Patient",
            output_file="00000-Patient.ndjson",
            expected_count=1,
            lines_processed=1,
            resources_deidentified=1,
            blank_lines=0,
            error_count=0,
            output_sha256="abc123",
        )
        self.completed_summary = SMARTBackendIngestionSummary(
            job_id="route-job",
            status="succeeded",
            files_total=1,
            files_completed=1,
            resources_deidentified=1,
            lines_processed=1,
            error_count=0,
            output_sha256="digest",
            max_inflight_downloads_observed=1,
            files=(file_result,),
        )
        self.status = SMARTBackendJobStatus(
            job_id="route-job",
            status="succeeded",
            created_at=1.0,
            updated_at=2.0,
            summary=self.completed_summary,
        )

    def start(self, config, *, deidentifier=None, job_id=None):
        self.started = True
        self.started_config = config
        return self.status

    def get(self, job_id: str):
        assert job_id == "route-job"
        return self.status

    def summary(self, job_id: str):
        assert job_id == "route-job"
        return self.completed_summary

    async def cancel_all(self) -> None:
        return None


def test_service_routes_start_status_and_summary_without_echoing_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    app = create_app()
    manager = _FakeManager()
    app.state.smart_backend_jobs = manager
    payload = {
        "fhir_base_url": "https://fhir.example.test",
        "token_url": "https://auth.example.test/token",
        "client_id": "route-client",
        "private_key_pem": "route-private-key-secret",
        "output_dir": str(tmp_path / "route-out"),
        "max_inflight_downloads": 1,
    }

    with TestClient(app, base_url="http://127.0.0.1") as client:
        start_response = client.post("/fhir/smart-backend/ingestions", json=payload)
        status_response = client.get("/fhir/smart-backend/ingestions/route-job")
        summary_response = client.get(
            "/fhir/smart-backend/ingestions/route-job/summary"
        )

    assert start_response.status_code == 200
    assert status_response.status_code == 200
    assert summary_response.status_code == 200
    rendered = json.dumps(
        {
            "start": start_response.json(),
            "status": status_response.json(),
            "summary": summary_response.json(),
        },
        sort_keys=True,
    )
    assert "route-private-key-secret" not in rendered
    assert manager.started is True
    assert manager.started_config is not None
    assert manager.started_config.client_id == "route-client"
