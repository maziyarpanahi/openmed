"""Unit tests for the OpenMed GraphQL service."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.graphql_app import graphql_ide_for_profile
from openmed.service.graphql_schema import SAFE_RESOLVER_ERROR
from scripts.export_graphql_schema import (
    DEFAULT_OUTPUT_PATH,
    render_graphql_schema,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
SYNTHETIC_TEXT = "Paciente: Maria Garcia"


def test_graphql_authorizes_complete_document_before_resolvers(subtests, caplog):
    for selection in [
        'analyze(input: {text: "SYNTHETIC_FORBIDDEN"}) { text }',
        'alias: deidentify(input: {text: "SYNTHETIC_FORBIDDEN"}) { deidentifiedText }',
        'entityTypes { label } hidden: analyze(input: {text: "SYNTHETIC_FORBIDDEN"}) { text }',
        "...Secret",
    ]:
        with subtests.test(selection=selection):
            caplog.clear()
            with pytest.MonkeyPatch.context() as monkeypatch:
                import json

                from openmed.service.auth import hash_api_key

                monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
                monkeypatch.setenv(
                    "OPENMED_SERVICE_AUTH_API_KEYS",
                    json.dumps(
                        [
                            {
                                "key_hash": hash_api_key("synthetic-key"),
                                "scopes": ["models:read"],
                            }
                        ]
                    ),
                )
                app = create_app()
                query = "query { " + selection + " }"
                if selection == "...Secret":
                    query += ' fragment Secret on Query { alias: analyze(input: {text: "SYNTHETIC_FORBIDDEN"}) { text } }'
                with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
                    response = client.post(
                        "/graphql",
                        json={"query": query},
                        headers={"X-API-Key": "synthetic-key"},
                    )
                result = response.json()
                assert result["data"] is None
                assert any(
                    error.get("extensions", {}).get("code") == "OPENMED_FORBIDDEN"
                    for error in result["errors"]
                )
                assert app.state.runtime._loader is None
                assert "SYNTHETIC_FORBIDDEN" not in response.text + caplog.text


def test_graphql_allowed_scope_preserves_results_and_rejects_unserved_models(
    monkeypatch,
):
    import json

    from openmed.service.auth import hash_api_key

    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [{"key_hash": hash_api_key("synthetic-key"), "scopes": ["analyze:*"]}]
        ),
    )
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setattr(openmed, "analyze_text", lambda *a, **kw: _prediction_result())
    query = "query($input: AnalyzeInput!) { analyze(input: $input) { spans { label start } } entityTypes { label } }"
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        allowed = client.post(
            "/graphql",
            json={"query": query, "variables": {"input": {"text": "synthetic"}}},
            headers={"X-API-Key": "synthetic-key"},
        )
        denied = client.post(
            "/graphql",
            json={
                "query": query,
                "variables": {
                    "input": {"text": "synthetic", "modelName": "/SYNTHETIC_UNSERVED"}
                },
            },
            headers={"X-API-Key": "synthetic-key"},
        )
    assert allowed.json()["data"]["analyze"]["spans"] == [
        {"label": "NAME", "start": 10}
    ]
    assert denied.json()["errors"][0]["extensions"]["code"] == "model_not_served"
    assert "SYNTHETIC_UNSERVED" not in denied.text


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


@pytest.fixture
def development_client(monkeypatch: pytest.MonkeyPatch):
    """Create a GraphQL client for the explicit development profile."""
    FakeLoader.instances = []
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
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
    assert field_names == {
        "analyze",
        "deidentify",
        "entityTypes",
        "journeyResources",
    }


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


SPAN_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "graphql_openmed_span_snapshot.json"
)
SPAN_SNAPSHOT_QUERY = """
query Spans($analyze: AnalyzeInput!, $deidentify: DeidentifyInput!) {
  analyze(input: $analyze) { spans { ...SpanFields } }
  deidentify(input: $deidentify) { spans { ...SpanFields } }
}
fragment SpanFields on OpenMedSpan {
  label start end text schemaVersion docId textHash entityType canonicalLabel
  policyLabel regulatoryTags score detector evidence action replacement
  reversibleId section metadata
}
"""


def _sdl_span_type(sdl: str):
    from graphql import build_schema

    return build_schema(sdl).get_type("OpenMedSpan")


def test_every_span_schema_property_is_on_the_sdl_with_its_mapped_type() -> None:
    from openmed.core.schemas import load_schema
    from scripts.codegen.span_graphql import camel_case, graphql_type

    schema = load_schema("span")
    for sdl in (
        DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"),
        render_graphql_schema(),
    ):
        fields = _sdl_span_type(sdl).fields
        for name, prop in schema["properties"].items():
            field = fields[camel_case(name)]
            assert str(field.type) == graphql_type(name, prop), name
            assert field.deprecation_reason is None, name


def test_exported_sdl_deprecates_label_and_text() -> None:
    sdl = DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8")
    fields = _sdl_span_type(sdl).fields

    assert "canonicalLabel" in fields["label"].deprecation_reason
    assert "textHash" in fields["text"].deprecation_reason
    assert "5.0" in fields["label"].deprecation_reason
    assert "5.0" in fields["text"].deprecation_reason
    assert str(fields["label"].type) == "String!"
    assert str(fields["text"].type) == "String"
    assert sdl.count("@deprecated") == 2


def test_strawberry_span_type_and_generated_module_match_the_schema() -> None:
    from scripts.codegen.span_graphql import check

    assert check() == []


def test_span_codegen_check_command_passes_on_committed_files() -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "scripts/codegen/span_graphql.py", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (
            lambda schema: schema["properties"].update(
                {"synthetic_note_ref": {"type": ["string", "null"]}}
            ),
            "OpenMedSpan is missing canonical field syntheticNoteRef",
        ),
        (
            lambda schema: schema["properties"]["score"].update({"type": "string"}),
            "OpenMedSpan.score is Float, expected String",
        ),
        (
            lambda schema: schema["properties"].pop("section"),
            "OpenMedSpan.section is not in the span schema",
        ),
    ],
)
def test_synthetic_schema_change_without_regeneration_fails_the_check(
    tmp_path: Path, edit, expected: str
) -> None:
    import json

    from scripts.codegen.span_graphql import (
        DEFAULT_MODULE_PATH,
        DEFAULT_SCHEMA_PATH,
        check,
        main,
    )

    schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    edit(schema)
    if "section" not in schema["properties"]:
        schema["required"].remove("section")
    schema_path = tmp_path / "span.schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    problems = check(schema_path=schema_path, module_path=DEFAULT_MODULE_PATH)

    assert problems[0] == "graphql_span_fields.py differs from the span schema"
    assert expected in problems
    module_path = tmp_path / "graphql_span_fields.py"
    assert main(["--schema", str(schema_path), "--module", str(module_path)]) == 0
    regenerated = check(schema_path=schema_path, module_path=module_path)
    assert "graphql_span_fields.py differs from the span schema" not in regenerated
    assert expected in regenerated


def test_span_codegen_rejects_unsupported_types_and_legacy_collisions() -> None:
    from scripts.codegen.span_graphql import (
        SpanGraphQLCodegenError,
        canonical_fields,
        graphql_type,
    )

    for prop in (
        {"type": ["string", "integer"]},
        {"type": "array"},
        {"type": "array", "items": {"type": ["string", "null"]}},
        {},
    ):
        with pytest.raises(SpanGraphQLCodegenError):
            graphql_type("synthetic", prop)
    with pytest.raises(SpanGraphQLCodegenError, match="legacy"):
        canonical_fields({"properties": {"label": {"type": "string"}}})
    assert graphql_type("synthetic", {"type": "boolean"}) == "Boolean"
    assert graphql_type(
        "synthetic", {"type": "array", "items": {"type": "object"}}
    ) == ("[JSON!]!")


def test_full_span_selection_keeps_the_committed_response_shape(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    monkeypatch.setattr(openmed, "analyze_text", lambda *a, **k: _prediction_result())
    monkeypatch.setattr(
        openmed, "deidentify", lambda *a, **k: _deidentification_result()
    )
    response = client.post(
        "/graphql",
        json={
            "query": SPAN_SNAPSHOT_QUERY,
            "variables": {
                "analyze": {"text": SYNTHETIC_TEXT},
                "deidentify": {"text": SYNTHETIC_TEXT},
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == json.loads(SPAN_SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_graphql_missing_field_policy_denies_even_wildcard_principal(monkeypatch):
    import json

    from openmed.service.auth import hash_api_key
    from openmed.service.graphql_schema import GRAPHQL_FIELD_SCOPES

    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps([{"key_hash": hash_api_key("synthetic-key"), "scopes": ["*"]}]),
    )
    monkeypatch.delitem(GRAPHQL_FIELD_SCOPES, "analyze")
    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.post(
            "/graphql",
            json={
                "query": 'query { analyze(input: {text: "synthetic"}) { text } entityTypes { label } }'
            },
            headers={"X-API-Key": "synthetic-key"},
        )
    assert response.json()["data"] is None
    assert response.json()["errors"][0]["extensions"]["code"] == "OPENMED_FORBIDDEN"
    assert app.state.runtime._loader is None


def test_get_queries_are_rejected_before_any_resolver_runs(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A GET request cannot execute introspection or resolver work."""

    sentinel = "Patient SECRET-PHI-471"
    caplog.set_level(logging.DEBUG)
    response = client.get(
        "/graphql",
        params={
            "query": (
                "query {"
                "  __schema { queryType { fields { name } } }"
                f'  analyze(input: {{ text: "{sentinel}" }}) {{ text }}'
                "}"
            )
        },
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 400
    assert "entityTypes" not in response.text
    assert sentinel not in response.text
    assert sentinel not in caplog.text
    assert FakeLoader.instances == []


def test_interactive_ide_is_disabled_outside_the_development_profile(
    client: TestClient,
) -> None:
    response = client.get("/graphql", headers={"Accept": "text/html"})

    assert response.status_code == 404
    assert "graphiql" not in response.text.lower()


def test_development_profile_serves_the_ide_and_keeps_operations_on_post(
    development_client: TestClient,
) -> None:
    ide_response = development_client.get("/graphql", headers={"Accept": "text/html"})
    assert ide_response.status_code == 200
    assert "graphiql" in ide_response.text.lower()

    rejected = development_client.get(
        "/graphql",
        params={"query": "{ entityTypes { label } }"},
        headers={"Accept": "application/json"},
    )
    assert rejected.status_code == 400

    accepted = development_client.post(
        "/graphql", json={"query": "{ entityTypes { label } }"}
    )
    assert accepted.status_code == 200
    assert accepted.json()["data"]["entityTypes"]


def test_every_graphql_response_carries_the_privacy_headers(
    client: TestClient,
) -> None:
    responses = {
        "post": client.post("/graphql", json={"query": "{ entityTypes { label } }"}),
        "get_query": client.get(
            "/graphql",
            params={"query": "{ entityTypes { label } }"},
            headers={"Accept": "application/json"},
        ),
        "get_ide_disabled": client.get("/graphql", headers={"Accept": "text/html"}),
        "unsupported_method": client.request(
            "PUT", "/graphql", json={"query": "{ entityTypes { label } }"}
        ),
    }

    assert responses["post"].status_code == 200
    assert responses["get_query"].status_code == 400
    assert responses["get_ide_disabled"].status_code == 404
    assert responses["unsupported_method"].status_code == 405
    for name, response in responses.items():
        assert response.headers["cache-control"] == "no-store", name
        assert response.headers["pragma"] == "no-cache", name
        assert response.headers["referrer-policy"] == "no-referrer", name


def test_privacy_headers_are_scoped_to_the_graphql_transport(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("pragma") != "no-cache"


def test_graphql_ide_is_limited_to_the_development_profile() -> None:
    assert graphql_ide_for_profile("dev") == "graphiql"
    assert graphql_ide_for_profile("prod") is None
    assert graphql_ide_for_profile("test") is None
