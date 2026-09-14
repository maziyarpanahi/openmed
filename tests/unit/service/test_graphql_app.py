"""Unit tests for the OpenMed GraphQL service."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.graphql_schema import SAFE_RESOLVER_ERROR
from scripts.export_graphql_schema import (
    DEFAULT_OUTPUT_PATH,
    render_graphql_schema,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
SYNTHETIC_TEXT = "Paciente: Maria Garcia"


class FakeLoader:
    """Minimal model loader used to verify shared runtime wiring."""

    instances: list[FakeLoader] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.instances.append(self)

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def clear_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep service startup deterministic and offline."""
    for name in (
        "OPENMED_SERVICE_PRELOAD_MODELS",
        "OPENMED_SERVICE_KEEP_ALIVE",
        "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
        "OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES",
        "OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES",
        "OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS",
        "OPENMED_SERVICE_MAX_TEXT_LENGTH",
        "OPENMED_SERVICE_BATCHING_ENABLED",
        "OPENMED_SERVICE_COALESCING_ENABLED",
        "OPENMED_SERVICE_RATE_LIMIT_RPS",
        "OPENMED_SERVICE_RATE_LIMIT_BURST",
        "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
        "OPENMED_SERVICE_CORS_ORIGINS",
        "OPENMED_SERVICE_TRUSTED_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENMED_PROFILE", "test")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Create a GraphQL client backed by one shared fake loader."""
    FakeLoader.instances = []
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def _prediction_result() -> PredictionResult:
    return PredictionResult(
        text=SYNTHETIC_TEXT,
        entities=[
            EntityPrediction(
                text="Maria Garcia",
                label="NAME",
                confidence=0.97,
                start=10,
                end=22,
            )
        ],
        model_name="disease_detection_superclinical",
        timestamp="2026-07-19T00:00:00",
        processing_time=0.01,
        metadata={"sentence_detection": True},
    )


def _deidentification_result() -> DeidentificationResult:
    entity = PIIEntity(
        text="Maria Garcia",
        label="NAME",
        confidence=0.98,
        start=10,
        end=22,
        entity_type="NAME",
        redacted_text="[NAME]",
        original_text="Maria Garcia",
        canonical_label="PERSON",
        action="mask",
    )
    return DeidentificationResult(
        original_text=SYNTHETIC_TEXT,
        deidentified_text="Paciente: [NAME]",
        pii_entities=[entity],
        method="mask",
        timestamp=datetime(2026, 7, 19),
    )


def test_analyze_query_returns_only_selected_span_fields(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphQL selection prevents unrequested span PHI from over-fetching."""

    def fake_analyze(*args: Any, **kwargs: Any) -> PredictionResult:
        assert kwargs["loader"].loader is FakeLoader.instances[0]
        assert kwargs["config"].profile == "test"
        return _prediction_result()

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    response = client.post(
        "/graphql",
        json={
            "query": """
                query Analyze($input: AnalyzeInput!) {
                  analyze(input: $input) {
                    spans { label start }
                  }
                }
            """,
            "variables": {"input": {"text": SYNTHETIC_TEXT}},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "data": {"analyze": {"spans": [{"label": "NAME", "start": 10}]}}
    }
    assert "Maria Garcia" not in response.text


def test_deidentify_query_exposes_policy_and_aggregate_risk_facets(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_deidentify(*args: Any, **kwargs: Any) -> DeidentificationResult:
        captured["loader"] = kwargs["loader"]
        captured["policy"] = kwargs["policy"]
        return _deidentification_result()

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    response = client.post(
        "/graphql",
        json={
            "query": """
                query Deidentify($input: DeidentifyInput!) {
                  deidentify(input: $input) {
                    deidentifiedText
                    entities { label start }
                    policy { name defaultAction }
                    risk { leakageRate minimumK }
                  }
                }
            """,
            "variables": {
                "input": {
                    "text": SYNTHETIC_TEXT,
                    "policy": "hipaa_safe_harbor",
                }
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]["deidentify"]
    assert payload["deidentifiedText"] == "Paciente: [NAME]"
    assert payload["entities"] == [{"label": "NAME", "start": 10}]
    assert payload["policy"]["name"] == "hipaa_safe_harbor"
    assert payload["risk"] == {"leakageRate": 0.0, "minimumK": 1}
    assert captured["loader"].loader is FakeLoader.instances[0]
    assert captured["policy"] == "hipaa_safe_harbor"


def test_entity_types_and_introspection_are_available(client: TestClient) -> None:
    response = client.post(
        "/graphql",
        json={
            "query": """
                {
                  entityTypes { label policyLabel }
                  __schema { queryType { fields { name } } }
                }
            """
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]
    entity_types = {item["label"]: item for item in payload["entityTypes"]}
    assert entity_types["PERSON"]["policyLabel"] == "DIRECT_IDENTIFIER"
    field_names = {
        field["name"] for field in payload["__schema"]["queryType"]["fields"]
    }
    assert field_names == {"analyze", "deidentify", "entityTypes"}


def test_resolver_errors_and_logs_never_contain_source_text(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_text = "Patient SECRET-PHI-471"

    def fail_with_source_text(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"backend failed for {source_text}")

    monkeypatch.setattr(openmed, "analyze_text", fail_with_source_text)
    caplog.set_level(logging.DEBUG)
    response = client.post(
        "/graphql",
        json={
            "query": "query($input: AnalyzeInput!) { analyze(input: $input) { text } }",
            "variables": {"input": {"text": source_text}},
        },
    )

    assert response.status_code == 200
    assert response.json()["errors"][0]["message"] == SAFE_RESOLVER_ERROR
    assert source_text not in response.text
    assert source_text not in caplog.text


def test_committed_sdl_matches_live_schema() -> None:
    assert DEFAULT_OUTPUT_PATH.exists(), (
        "GraphQL SDL is missing. Re-run "
        ".venv/bin/python scripts/export_graphql_schema.py."
    )
    assert DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8") == render_graphql_schema()


def test_schema_is_read_only() -> None:
    sdl = render_graphql_schema()

    assert "type Query" in sdl
    assert "type Mutation" not in sdl
    assert "type Subscription" not in sdl
