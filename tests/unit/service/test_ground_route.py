"""Focused tests for the shared REST grounding contract."""

from __future__ import annotations

import json
import logging
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
    marker = "synthetic-private-mention-1961"
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            json={
                "entities": [
                    {
                        "text": "type 2 diabetes",
                        "start": 4,
                        "end": 19,
                        "section": "synthetic assessment",
                    }
                ],
                "systems": ["icd10cm"],
                "offline": True,
            },
            headers={"X-Synthetic-Credential": marker},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "openmed.grounding.v1"
    result = payload["results"][0]
    assert result["start"] == 4
    assert result["end"] == 19
    assert result["code"] == "E11.9"
    assert result["system_uri"] == "http://hl7.org/fhir/sid/icd-10-cm"
    assert result["section_context"] == "synthetic assessment"
    assert result["snapshot_provenance"]["icd10cm"]["version"] == (
        "synthetic-fixture-1"
    )

    rendered = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == ACCESS_LOGGER_NAME
    )
    assert marker not in rendered
    assert "type 2 diabetes" not in rendered


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
