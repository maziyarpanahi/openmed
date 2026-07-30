"""Tests for async de-identification job APIs."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.webhooks import WebhookDeliveryResult

LOOPBACK_BASE_URL = "http://127.0.0.1"


class FakeLoader:
    """Minimal model loader double for warm-pool bookkeeping."""

    def __init__(self, config):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def clear_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [
        "OPENMED_PROFILE",
        "OPENMED_SERVICE_PRELOAD_MODELS",
        "OPENMED_SERVICE_KEEP_ALIVE",
        "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
        "OPENMED_SERVICE_MAX_TEXT_LENGTH",
        "OPENMED_SERVICE_BATCHING_ENABLED",
        "OPENMED_SERVICE_BATCH_MAX_SIZE",
        "OPENMED_SERVICE_BATCH_MAX_WAIT_MS",
        "OPENMED_SERVICE_CORS_ORIGINS",
        "OPENMED_SERVICE_TRUSTED_HOSTS",
        "OPENMED_SERVICE_COALESCING_ENABLED",
        "OPENMED_SERVICE_RATE_LIMIT_RPS",
        "OPENMED_SERVICE_RATE_LIMIT_BURST",
        "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
        "OPENMED_SERVICE_THROTTLE_KEY",
        "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
        "OPENMED_SERVICE_JOBS_STORE_PATH",
        "OPENMED_SERVICE_JOBS_TTL_SECONDS",
        "OPENMED_SERVICE_JOBS_WORKERS",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _sample_deid_result(text: str, *, label: str = "NAME") -> DeidentificationResult:
    entity_text = "Maria Garcia" if "Maria Garcia" in text else "555-1212"
    start = text.index(entity_text)
    end = start + len(entity_text)
    return DeidentificationResult(
        original_text=text,
        deidentified_text=text.replace(entity_text, f"[{label}]"),
        pii_entities=[
            PIIEntity(
                text=entity_text,
                label=label,
                confidence=0.99,
                start=start,
                end=end,
                entity_type=label,
                redacted_text=f"[{label}]",
                original_text=entity_text,
            )
        ],
        method="mask",
        timestamp=datetime.now(),
    )


def _wait_for_job(
    client: TestClient,
    job_id: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for _ in range(100):
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if predicate(payload):
            return payload
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach expected state: {payload}")


def test_jobs_api_runs_batch_and_posts_no_phi_webhook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store_path = tmp_path / "jobs.json"
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(store_path))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)

    def fake_deidentify(text: str, **kwargs: Any) -> DeidentificationResult:
        assert kwargs["loader"].loader.__class__ is FakeLoader
        time.sleep(0.05)
        return _sample_deid_result(text, label="NAME")

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    callbacks: list[dict[str, Any]] = []

    def webhook_sender(
        url: str,
        payload: dict[str, Any],
        *,
        secret: str,
        max_attempts: int,
        backoff_seconds: float,
    ) -> WebhookDeliveryResult:
        callbacks.append(
            {
                "url": url,
                "payload": payload,
                "secret": secret,
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
            }
        )
        return WebhookDeliveryResult(success=True, attempts=1, status_code=204)

    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        app.state.job_queue.webhook_sender = webhook_sender
        response = client.post(
            "/jobs",
            json={
                "documents": [
                    {"id": "doc-1", "text": "Paciente: Maria Garcia"},
                    {"id": "doc-2", "text": "Call Maria Garcia at 555-1212"},
                ],
                "webhook": {
                    "url": "https://callbacks.example.test/openmed",
                    "secret": "top-secret",
                    "max_attempts": 2,
                    "backoff_seconds": 0,
                },
            },
        )

        assert response.status_code == 202
        created = response.json()
        assert created["status"] == "queued"
        assert created["status_url"] == f"/jobs/{created['id']}"

        running = _wait_for_job(
            client,
            created["id"],
            lambda payload: payload["status"] in {"running", "done"},
        )
        assert running["status"] in {"running", "done"}

        done = _wait_for_job(
            client,
            created["id"],
            lambda payload: (
                payload["status"] == "done" and payload["webhook_delivery"] is not None
            ),
        )

    assert done["progress_percent"] == 100.0
    assert done["processed_count"] == 2
    assert done["failed_count"] == 0
    assert done["label_histogram"] == {"NAME": 2}
    assert done["webhook"]["url_hash"].startswith("sha256:")
    assert done["webhook_delivery"]["success"] is True
    assert callbacks[0]["payload"]["status"] == "done"
    assert callbacks[0]["payload"]["label_histogram"] == {"NAME": 2}

    callback_json = json.dumps(callbacks[0]["payload"], sort_keys=True)
    store_json = store_path.read_text(encoding="utf-8")
    for raw_phi in ("Maria Garcia", "555-1212", "Paciente"):
        assert raw_phi not in callback_json
        assert raw_phi not in store_json
    assert "top-secret" not in store_json
    assert "https://callbacks.example.test/openmed" not in store_json


def test_failed_job_surfaces_error_and_records_webhook_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setattr(
        openmed,
        "deidentify",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model offline")),
    )
    callbacks: list[dict[str, Any]] = []

    def webhook_sender(
        url: str,
        payload: dict[str, Any],
        *,
        secret: str,
        max_attempts: int,
        backoff_seconds: float,
    ) -> WebhookDeliveryResult:
        del url, secret, max_attempts, backoff_seconds
        callbacks.append(payload)
        return WebhookDeliveryResult(success=True, attempts=2, status_code=200)

    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        app.state.job_queue.webhook_sender = webhook_sender
        response = client.post(
            "/jobs",
            json={
                "documents": [{"text": "Paciente: Maria Garcia"}],
                "webhook": {
                    "url": "https://callbacks.example.test/openmed",
                    "secret": "top-secret",
                    "max_attempts": 2,
                    "backoff_seconds": 0,
                },
            },
        )
        job_id = response.json()["id"]
        failed = _wait_for_job(
            client,
            job_id,
            lambda payload: (
                payload["status"] == "failed"
                and payload["webhook_delivery"] is not None
            ),
        )

    assert failed["failed_count"] == 1
    assert failed["processed_count"] == 0
    assert failed["error"] == {
        "type": "RuntimeError",
        "message": "Document failed during de-identification",
    }
    assert failed["webhook_delivery"]["attempts"] == 2
    assert callbacks[0]["status"] == "failed"
    assert callbacks[0]["error"]["type"] == "RuntimeError"


def test_missing_job_returns_standard_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/jobs/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "bad_request"
