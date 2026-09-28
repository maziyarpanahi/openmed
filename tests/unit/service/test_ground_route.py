"""Focused tests for the shared REST grounding contract."""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openmed.clinical.grounding import VocabLoader
from openmed.service.app import create_app
from openmed.service.logging import ACCESS_LOGGER_NAME

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"


@pytest.fixture()
def grounding_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / "grounding"
    VocabLoader(cache_dir=cache_dir, local_only=True).import_snapshot(
        "icd10cm",
        FIXTURE,
        version="synthetic-fixture-1",
    )
    monkeypatch.setenv("OPENMED_GROUNDING_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    return cache_dir


def test_ground_route_matches_python_result_and_keeps_logs_phi_free(
    grounding_cache: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del grounding_cache
    source_text = "Synthetic patient Juniper Example has type 2 diabetes in assessment."
    surface = "type 2 diabetes"
    start = source_text.index(surface)
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            json={
                "text": source_text,
                "systems": ["icd10cm"],
                "lang": "en",
                "offline": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "openmed.grounding.v1"
    result = payload["results"][0]
    assert result["start"] == start
    assert result["end"] == start + len(surface)
    assert result["system"] == "icd10cm"
    assert result["code"] == "E11.9"
    assert result["system_uri"] == "http://hl7.org/fhir/sid/icd-10-cm"
    assert result["snapshot_provenance"]["icd10cm"]["version"] == (
        "synthetic-fixture-1"
    )
    assert "calibrated_confidence" not in result
    assert "confidence_band" not in result

    rendered = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == ACCESS_LOGGER_NAME
    )
    assert source_text not in rendered
    assert "Juniper Example" not in rendered
    assert surface not in rendered
    access_record = next(
        record for record in caplog.records if record.name == ACCESS_LOGGER_NAME
    )
    assert access_record.openmed_access_log["grounding_input_count"] == 1
    assert access_record.openmed_access_log["grounding_result_count"] == 1
    assert access_record.openmed_access_log["grounding_systems"] == ("icd10cm",)
    assert access_record.openmed_access_log["grounding_lang"] == "en"


def test_ground_route_returns_typed_restricted_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENMED_GROUNDING_CACHE_DIR", raising=False)
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            json={"text": "synthetic finding", "systems": ["snomed"]},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "restricted_terminology_unconfigured"
    assert "synthetic finding" not in json.dumps(body)


def test_ground_route_rejects_invalid_system_via_validation_handler() -> None:
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            json={"text": "synthetic finding", "systems": ["not-a-system"]},
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["details"][0]["field"] == "body.systems"


def test_ground_route_rejects_oversized_request_before_json_parsing() -> None:
    with TestClient(
        create_app(max_request_body_bytes=128),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            content=json.dumps({"text": "x" * 512, "systems": ["rxnorm"]}),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_ground_route_is_throttled(
    grounding_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del grounding_cache
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "1")

    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        first = client.post(
            "/ground",
            json={"text": "type 2 diabetes", "systems": ["icd10cm"]},
        )
        second = client.post(
            "/ground",
            json={"text": "type 2 diabetes", "systems": ["icd10cm"]},
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"


def test_ground_route_missing_snapshot_stays_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_network = False

    def fail_connect(*_: object, **__: object) -> None:
        nonlocal attempted_network
        attempted_network = True
        raise AssertionError("grounding attempted network access")

    monkeypatch.setenv("OPENMED_GROUNDING_CACHE_DIR", str(tmp_path / "empty"))
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        # Windows creates an asyncio loopback socketpair when TestClient starts.
        # Intercept only the request, after the test harness is initialized.
        with monkeypatch.context() as request_patch:
            request_patch.setattr(socket.socket, "connect", fail_connect)
            response = client.post(
                "/ground",
                json={
                    "text": "synthetic medication",
                    "systems": ["rxnorm"],
                    "offline": True,
                },
            )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "offline_snapshot_unavailable"
    assert attempted_network is False
