"""End-to-end offline test of encrypted async job payload retention.

Runs the REST app with a synthetic caller-owned key, submits a job carrying
plaintext sentinels, and scans every byte the service wrote (job metadata,
payload files, temporary files), the HTTP responses, the lifecycle evidence,
and the logs for those sentinels. A restart with the same key then confirms
that startup cleanup leaves no payload artifact behind.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.key_provider import DATA_KEY_BYTES, InMemoryKeyProvider, KeyState
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.job_payloads import JOB_PAYLOAD_KEY_PURPOSE
from openmed.service.webhooks import WebhookDeliveryResult

pytestmark = pytest.mark.integration

LOOPBACK_BASE_URL = "http://127.0.0.1"
NAME_SENTINEL = "Zqintegrationsentinel Patient"
TEXT_SENTINEL = f"Synthetic note for {NAME_SENTINEL} at ward 7."
SECRET_SENTINEL = "zq-integration-webhook-secret-31"
URL_SENTINEL = "https://zqintegration.example.test/openmed/jobs"
SENTINELS = (NAME_SENTINEL, TEXT_SENTINEL, SECRET_SENTINEL, URL_SENTINEL)


class FakeLoader:
    """Minimal model loader double for warm-pool bookkeeping."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _provider() -> InMemoryKeyProvider:
    provider = InMemoryKeyProvider()
    provider.add_key(
        "jobs-key-1",
        bytes(range(DATA_KEY_BYTES)),
        purpose=JOB_PAYLOAD_KEY_PURPOSE,
        state=KeyState.ACTIVE,
    )
    return provider


def _fake_deidentify(text: str, **_kwargs: Any) -> DeidentificationResult:
    start = text.index(NAME_SENTINEL)
    end = start + len(NAME_SENTINEL)
    return DeidentificationResult(
        original_text=text,
        deidentified_text=text.replace(NAME_SENTINEL, "[NAME]"),
        pii_entities=[
            PIIEntity(
                text=NAME_SENTINEL,
                label="NAME",
                confidence=0.99,
                start=start,
                end=end,
                entity_type="NAME",
                redacted_text="[NAME]",
                original_text=NAME_SENTINEL,
            )
        ],
        method="mask",
        timestamp=datetime.now(),
    )


def _wait_for(
    client: TestClient,
    job_id: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for _ in range(200):
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if predicate(payload):
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not reach the expected state")


def _scan(blob: bytes | str) -> None:
    data = blob.encode("utf-8") if isinstance(blob, str) else blob
    for sentinel in SENTINELS:
        encoded = sentinel.encode("utf-8")
        assert encoded not in data
        assert base64.b64encode(encoded) not in data


def _tree_bytes(root: Path) -> bytes:
    return b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())


def test_encrypted_job_payloads_leave_no_plaintext_and_no_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload_dir = tmp_path / "payloads"
    store_path = tmp_path / "jobs.json"
    for name in (
        "OPENMED_SERVICE_JOBS_PAYLOAD_MODE",
        "OPENMED_SERVICE_JOBS_PAYLOAD_RETENTION_SECONDS",
        "OPENMED_SERVICE_JOBS_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(store_path))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_PAYLOAD_DIR", str(payload_dir))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_PAYLOAD_MODE", "encrypted")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setattr(openmed, "deidentify", _fake_deidentify)
    callbacks: list[dict[str, Any]] = []
    observed_payload_files: list[bytes] = []

    def webhook_sender(
        url: str,
        payload: dict[str, Any],
        *,
        secret: str,
        max_attempts: int,
        backoff_seconds: float,
        delivery_id: str,
    ) -> WebhookDeliveryResult:
        del delivery_id
        observed_payload_files.append(_tree_bytes(payload_dir))
        callbacks.append({"url": url, "secret": secret, "payload": payload})
        return WebhookDeliveryResult(success=True, attempts=1, status_code=204)

    responses: list[str] = []
    app = create_app(job_payload_key_provider=_provider())
    with caplog.at_level(logging.DEBUG):
        with TestClient(
            app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False
        ) as client:
            app.state.job_queue.webhook_sender = webhook_sender
            created = client.post(
                "/jobs",
                json={
                    "documents": [{"id": "note-1", "text": TEXT_SENTINEL}],
                    "webhook": {
                        "url": URL_SENTINEL,
                        "secret": SECRET_SENTINEL,
                        "max_attempts": 2,
                        "backoff_seconds": 0,
                    },
                    "payload_retention_seconds": 600,
                },
            )
            assert created.status_code == 202
            responses.append(created.text)
            job = created.json()
            assert job["payloads"]["mode"] == "encrypted"
            assert job["payloads"]["input"]["state"] == "retained"
            assert job["payloads"]["input"]["digest"].startswith("sha256:")

            done = _wait_for(
                client,
                job["id"],
                lambda record: (
                    record["status"] == "done"
                    and record["webhook_delivery"] is not None
                    and (record["payloads"]["terminal_event"] or {}).get("state")
                    == "deleted"
                ),
            )
            responses.append(json.dumps(done))
            too_long = client.post(
                "/jobs",
                json={
                    "documents": [{"text": TEXT_SENTINEL}],
                    "payload_retention_seconds": 10**9,
                },
            )
            assert too_long.status_code == 400
            responses.append(too_long.text)
            evidence = app.state.job_queue.payloads.lifecycle_evidence()

    assert callbacks and callbacks[0]["secret"] == SECRET_SENTINEL
    assert callbacks[0]["url"] == URL_SENTINEL
    assert callbacks[0]["payload"]["label_histogram"] == {"NAME": 1}
    assert (
        done["webhook_delivery"]["payload_ref"]
        == (done["payloads"]["terminal_event"]["ref"])
    )
    assert done["payloads"]["input"]["reason"] == "consumed"
    assert done["payloads"]["terminal_event"]["reason"] == "delivered"
    assert evidence["event_counts"]["created"] == 2
    assert evidence["event_counts"]["deleted"] == 2
    assert evidence["retained_count"] == 0
    assert list(payload_dir.glob("*")) == []

    for blob in [
        *observed_payload_files,
        _tree_bytes(tmp_path),
        *responses,
        json.dumps(evidence),
        caplog.text,
    ]:
        _scan(blob)
    for text in [*responses, json.dumps(evidence)]:
        assert str(tmp_path) not in text
        assert "jobs-key-1" not in text

    # Restart with the same key: startup reconciliation keeps the store clean
    # and the terminal record still carries only opaque references.
    restarted = create_app(job_payload_key_provider=_provider())
    with TestClient(
        restarted, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False
    ) as client:
        reopened = client.get(f"/jobs/{job['id']}").json()
    assert reopened["status"] == "done"
    assert reopened["payloads"]["input"]["state"] == "deleted"
    assert list(payload_dir.glob("*")) == []


def test_encrypted_mode_without_a_key_provider_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_PAYLOAD_DIR", str(tmp_path / "payloads"))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_PAYLOAD_MODE", "encrypted")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)

    app = create_app()
    with pytest.raises(ValueError, match="key provider"):
        with TestClient(app, base_url=LOOPBACK_BASE_URL):
            pass
    assert not (tmp_path / "payloads").exists()
