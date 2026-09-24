"""Focused tests for ``POST /profile`` and its downstream quality gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openmed.interop.athena import load_athena_vocab
from openmed.service.app import create_app

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "quality" / "profiler_batch.jsonl"
ATHENA = ROOT / "tests" / "fixtures" / "quality" / "athena"
LOOPBACK_BASE_URL = "http://127.0.0.1"


@pytest.fixture()
def client() -> TestClient:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as test_client:
        yield test_client


def test_profile_endpoint_returns_phi_free_quality_report(client: TestClient) -> None:
    response = client.post(
        "/profile",
        json={
            "records_jsonl": FIXTURE.read_text(encoding="utf-8"),
            "completeness_floor": 0.9,
            "required_fields": ["condition", "drug", "measurement"],
            "athena_index": load_athena_vocab(ATHENA),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "fail"
    assert body["gate"]["passed"] is False
    assert body["grounding"]["grounded_spans"] == 5
    assert body["grounding"]["by_domain"]["condition"]["coverage"] == 0.5
    serialized = json.dumps(body, sort_keys=True)
    for raw_marker in ("diabetes", "metformin", "aspirin", "glucose"):
        assert raw_marker not in serialized


def test_omop_endpoint_rejects_a_batch_below_the_quality_floor(
    client: TestClient,
) -> None:
    records_jsonl = '{"note_id":"synthetic","person_id":"person","entities":[]}\n'
    response = client.post(
        "/omop/load",
        json={
            "records_jsonl": records_jsonl,
            "completeness_floor": 0.5,
            "required_fields": ["condition"],
        },
    )

    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "rejected"
    assert body["quality_gate"]["gate"]["passed"] is False
    assert "synthetic" not in json.dumps(body, sort_keys=True)
