from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app

LOOPBACK_BASE_URL = "http://127.0.0.1"


class FakeLoader:
    instances: list["FakeLoader"] = []

    def __init__(self, config):
        self.config = config
        FakeLoader.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def clear_security_env(monkeypatch):
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)


@pytest.fixture
def client(monkeypatch):
    FakeLoader.reset()
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_RPS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_BURST", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_THROTTLE_KEY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", raising=False)
    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def _deid_result(mapping: dict[str, str] | None = None) -> DeidentificationResult:
    entity = PIIEntity(
        text="Maria Garcia",
        label="NAME",
        confidence=0.98,
        start=10,
        end=22,
        entity_type="NAME",
        redacted_text="[NAME]",
        original_text="Maria Garcia",
        reversible_id="rev_test",
    )
    return DeidentificationResult(
        original_text="Paciente: Maria Garcia",
        deidentified_text="Paciente: [NAME]",
        pii_entities=[entity],
        method="mask",
        timestamp=datetime.now(),
        mapping=mapping,
    )


def test_pii_deidentify_rejects_unknown_policy(client):
    response = client.post(
        "/pii/deidentify",
        json={"text": "Paciente: Maria Garcia", "policy": "unknown_policy"},
    )

    payload = response.json()
    assert response.status_code == 422
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["details"][0]["field"] == "body.policy"


def test_pii_deidentify_policy_alias_is_routed_and_mapping_returned(
    client,
    monkeypatch,
):
    captured: dict[str, Any] = {}

    def fake_deidentify(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _deid_result(mapping={"[NAME]": "Maria Garcia"})

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)

    response = client.post(
        "/pii/deidentify",
        json={"text": "Paciente: Maria Garcia", "policy": "gdpr"},
    )

    assert response.status_code == 200
    assert captured["kwargs"]["policy"] == "gdpr_pseudonymization"
    assert response.json()["mapping"] == {"[NAME]": "Maria Garcia"}
