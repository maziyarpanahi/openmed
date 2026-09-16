"""Focused tests for ``POST /cohort/resolve``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openmed.interop.omop import deterministic_omop_id
from openmed.service.app import create_app
from openmed.structured.cohort import PhenotypeDefinition

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cohort"
LOOPBACK_BASE_URL = "http://127.0.0.1"


@pytest.fixture()
def client() -> TestClient:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as test_client:
        yield test_client


def _request_payload() -> dict:
    definition = PhenotypeDefinition.load(
        FIXTURES / "phenotypes" / "diabetes_on_metformin.json"
    )
    return {
        "phenotype": definition.to_dict(),
        "records_jsonl": (FIXTURES / "synthetic_grounded.jsonl").read_text(
            encoding="utf-8"
        ),
        "concept_ancestors": [
            {
                "ancestor_concept_id": 201826,
                "descendant_concept_id": 443238,
            }
        ],
    }


def test_cohort_resolve_endpoint_returns_phi_free_exact_cohort(
    client: TestClient,
) -> None:
    response = client.post("/cohort/resolve", json=_request_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["patient_ids"] == sorted(
        [
            deterministic_omop_id("person", "raw-person-alpha"),
            deterministic_omop_id("person", "raw-person-epsilon"),
        ]
    )
    assert body["provenance"]["matched_patient_count"] == 2
    serialized = json.dumps(body, sort_keys=True)
    for raw_marker in (
        "raw-person-",
        "synthetic-alpha",
        "Synthetic diabetes marker",
    ):
        assert raw_marker not in serialized


def test_cohort_resolve_endpoint_rejects_unknown_fields(client: TestClient) -> None:
    payload = _request_payload()
    payload["database_path"] = "/not/an/allowed/server/path"

    response = client.post("/cohort/resolve", json=payload)

    assert response.status_code == 422
