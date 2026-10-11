"""Tests for REST request correlation IDs and no-PHI access logs."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.logging import (
    ACCESS_LOGGER_NAME,
    REQUEST_ID_HEADER,
    SERVICE_LOG_FORMAT_ENV_VAR,
    SERVICE_LOG_LEVEL_ENV_VAR,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
PHI_TEXT = (
    "Patient Juniper Solstice, MRN JS-1188, DOB 02/03/1979, "
    "SSN 111-22-3333, phone 425-555-0199."
)
PHI_SUBSTRINGS = (
    "Juniper Solstice",
    "JS-1188",
    "02/03/1979",
    "111-22-3333",
    "425-555-0199",
)


@pytest.mark.parametrize("level", ["INFO", "off"])
def test_reference_launch_emits_only_safe_access_records(level, tmp_path):
    import os
    import socket
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request

    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    environment = os.environ.copy()
    environment.update(
        {
            "OPENMED_SERVICE_LOG_LEVEL": level,
            "OPENMED_PROFILE": "test",
            "OPENMED_OFFLINE": "1",
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openmed.service.logging",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sentinel = "SYNTHETIC-QUERY-PATH-SENTINEL-0042"
    targets = [
        "/analyze?text=" + sentinel,
        "/graphql?query=" + sentinel,
        "/" + sentinel,
    ]
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1
                ):
                    break
            except OSError:
                if time.monotonic() >= deadline or process.poll() is not None:
                    raise AssertionError(
                        "Reference service did not become ready"
                    ) from None
                time.sleep(0.05)
        for target in targets:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{target}", timeout=2
                ):
                    pass
            except urllib.error.HTTPError:
                pass
    finally:
        process.terminate()
        output, _ = process.communicate(timeout=15)
    assert sentinel not in output
    records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    if level == "off":
        assert records == []
    else:
        assert len(records) == 4
        assert [record["route"] for record in records] == [
            "/health",
            "/analyze",
            "/graphql",
            "unknown",
        ]
        assert all(
            {"method", "route", "status_code", "duration_ms", "request_id"}
            <= record.keys()
            for record in records
        )
        assert all("127.0.0.1" not in json.dumps(record) for record in records)


class FakeLoader:
    """Minimal loader double for request logging tests."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}

    def unload_model(self, model_name: str) -> dict[str, Any]:
        return {
            "model_name": model_name,
            "models": 0,
            "tokenizers": 0,
            "pipelines": 0,
        }

    def unload_all_models(self) -> dict[str, int]:
        return {"models": 0, "tokenizers": 0, "pipelines": 0}


@pytest.fixture(autouse=True)
def service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv(
        "OPENMED_SERVICE_SERVED_MODELS",
        "disease_detection_superclinical,OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1,test-pii-model",
    )
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_RPS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_BURST", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_THROTTLE_KEY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", raising=False)
    monkeypatch.delenv(SERVICE_LOG_FORMAT_ENV_VAR, raising=False)
    monkeypatch.delenv(SERVICE_LOG_LEVEL_ENV_VAR, raising=False)


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def _access_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == ACCESS_LOGGER_NAME]


def test_every_response_gets_generated_request_id(client: TestClient) -> None:
    response = client.get("/health")

    request_id = response.headers[REQUEST_ID_HEADER]
    uuid.UUID(request_id)
    assert response.status_code == 200


def test_inbound_request_id_is_preserved_and_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "req-inbound-123"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "req-inbound-123"


def test_access_log_record_is_single_line_json_with_required_fields(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze(text: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "text": text,
            "model_name": kwargs["model_name"],
            "entities": [],
        }

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)

    response = client.post(
        "/analyze",
        headers={REQUEST_ID_HEADER: "req-log-json"},
        json={"text": "sample", "model_name": "disease_detection_superclinical"},
    )

    assert response.status_code == 200
    records = _access_records(caplog)
    assert len(records) == 1
    payload = json.loads(records[0].getMessage())
    assert payload["method"] == "POST"
    assert payload["route"] == "/analyze"
    assert payload["status_code"] == 200
    assert payload["model_name"] == "disease_detection_superclinical"
    assert payload["request_id"] == "req-log-json"
    assert payload["duration_ms"] >= 0


def test_phi_bearing_request_does_not_leak_phi_substrings_to_access_log(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extract_pii(text: str, **kwargs: Any) -> dict[str, Any]:
        del text
        return {"entities": [], "model_name": kwargs["model_name"]}

    monkeypatch.setattr(openmed, "extract_pii", fake_extract_pii)
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)

    response = client.post(
        "/pii/extract",
        json={"text": PHI_TEXT, "model_name": "test-pii-model"},
    )

    assert response.status_code == 200
    rendered_logs = "\n".join(record.getMessage() for record in _access_records(caplog))
    leaked = [substring for substring in PHI_SUBSTRINGS if substring in rendered_logs]
    assert leaked == []


def test_error_envelope_includes_request_id(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        headers={REQUEST_ID_HEADER: "req-validation-error"},
        json={"text": "   "},
    )

    payload = response.json()
    assert response.status_code == 422
    assert response.headers[REQUEST_ID_HEADER] == "req-validation-error"
    assert payload["error"]["request_id"] == "req-validation-error"
    assert payload["error"]["details"][0]["field"] == "body.text"


def test_access_log_format_can_be_plain(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SERVICE_LOG_FORMAT_ENV_VAR, "plain")
    app = create_app()
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        response = test_client.get(
            "/health",
            headers={REQUEST_ID_HEADER: "req-plain-log"},
        )

    assert response.status_code == 200
    records = _access_records(caplog)
    assert len(records) == 1
    assert records[0].getMessage().startswith("GET /health 200 ")
    assert "request_id=req-plain-log" in records[0].getMessage()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/graphql", "/graphql"),
        ("/jobs/SYNTHETIC-PRIVATE-ID", "/jobs/{job_id}"),
        ("/SYNTHETIC-PRIVATE-ID", "unknown"),
    ],
)
def test_access_template_uses_registered_route_when_scope_path_is_empty(path, expected):
    from types import SimpleNamespace

    from openmed.service.logging import _route_template

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "root_path": "",
        "headers": [],
        "query_string": b"value=SYNTHETIC-PRIVATE-QUERY",
        "route": SimpleNamespace(path=""),
        "app": create_app(),
    }
    template = _route_template(scope)
    assert template == expected
    assert "SYNTHETIC-PRIVATE" not in template
