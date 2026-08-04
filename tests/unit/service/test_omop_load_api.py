"""Tests for the ``POST /omop/load`` REST surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openmed.service.app import create_app

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "interop" / "omop" / "grounded_notes.jsonl"
LOOPBACK_BASE_URL = "http://127.0.0.1"

# Synthetic note text that must never appear in the PHI-free response.
_NOTE_TEXT_MARKERS = ("Patient reports", "diabetes", "aspirin", "Glucose")


@pytest.fixture()
def client() -> TestClient:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as test_client:
        yield test_client


def _records_jsonl() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_omop_load_returns_phi_free_summary(client: TestClient) -> None:
    response = client.post(
        "/omop/load",
        json={
            "records_jsonl": _records_jsonl(),
            "vocabulary_version": "SNOMED-2026",
            "validate_constraints": True,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["row_counts"]["note"] == 2
    assert body["row_counts"]["note_nlp"] == 4
    assert body["rejection_counts"] == {"invalid_offset": 1}
    assert len(body["source_note_hashes"]) == 2
    assert body["constraint_violations"] == {"count": 0, "by_reason": {}}

    serialized = json.dumps(body)
    for marker in _NOTE_TEXT_MARKERS:
        assert marker not in serialized


def test_omop_load_omits_violations_without_validation(client: TestClient) -> None:
    response = client.post(
        "/omop/load",
        json={"records_jsonl": _records_jsonl()},
    )
    assert response.status_code == 200
    assert "constraint_violations" not in response.json()


def test_omop_load_rejects_missing_field(client: TestClient) -> None:
    response = client.post("/omop/load", json={"vocabulary_version": "x"})
    assert response.status_code == 422


def test_omop_load_rejects_blank_records(client: TestClient) -> None:
    response = client.post("/omop/load", json={"records_jsonl": "   "})
    assert response.status_code == 422


def test_omop_load_rejects_unknown_field(client: TestClient) -> None:
    response = client.post(
        "/omop/load",
        json={"records_jsonl": _records_jsonl(), "surprise": True},
    )
    assert response.status_code == 422


def test_omop_load_rejects_oversize_records(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", "64")
    response = client.post("/omop/load", json={"records_jsonl": "x" * 256})
    assert response.status_code == 422


def test_omop_load_malformed_line_is_phi_free_error(client: TestClient) -> None:
    response = client.post(
        "/omop/load",
        json={"records_jsonl": '{"note_text": "secret diabetes note"'},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "bad_request"
    message = body["error"]["message"]
    assert "line 1" in message
    assert "secret" not in message
    assert "diabetes" not in message
