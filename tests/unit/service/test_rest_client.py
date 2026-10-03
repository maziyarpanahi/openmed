"""Tests for the typed OpenMed REST client."""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import httpx
import pytest

import openmed
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.client import (
    CLIENT_ENDPOINTS,
    CLIENT_OPENAPI_EXCLUSIONS,
    AnalyzeRequest,
    BriefRequest,
    CohortResolveRequest,
    ConceptAncestorRequest,
    FixedOptionDecisionRequest,
    GroundRequest,
    ModelUnloadRequest,
    OpenMedAPIError,
    OpenMedClient,
    PIIDeidentifyRequest,
    PIIExtractRequest,
    PIIExtractStreamRequest,
    PIILanguage,
    PrivacyGatewayRequest,
    ProfileRequest,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"


class SyncASGITransport(httpx.BaseTransport):
    """Sync adapter that drives ``httpx.ASGITransport`` for client tests."""

    def __init__(self, app: Any) -> None:
        self._transport = httpx.ASGITransport(
            app=app,
            raise_app_exceptions=False,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def send() -> httpx.Response:
            response = await self._transport.handle_async_request(request)
            content = await response.aread()
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=content,
                extensions=response.extensions,
                request=request,
            )

        return asyncio.run(send())

    def close(self) -> None:
        asyncio.run(self._transport.aclose())


class FakeLoader:
    """Minimal service loader double for in-process client tests."""

    instances: list["FakeLoader"] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.pipelines: dict[str, object] = {}
        FakeLoader.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        pipeline = self.pipelines.setdefault(model_name, object())
        return pipeline

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {
            model_name: {"models": 0, "tokenizers": 0, "pipelines": 1}
            for model_name in sorted(self.pipelines)
        }

    def unload_model(self, model_name: str) -> dict[str, Any]:
        released = int(model_name in self.pipelines)
        self.pipelines.pop(model_name, None)
        return {
            "model_name": model_name,
            "models": 0,
            "tokenizers": 0,
            "pipelines": released,
        }

    def unload_all_models(self) -> dict[str, int]:
        released = len(self.pipelines)
        self.pipelines.clear()
        return {"models": 0, "tokenizers": 0, "pipelines": released}


@pytest.fixture
def rest_client(monkeypatch: pytest.MonkeyPatch):
    FakeLoader.reset()
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_RPS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_BURST", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_THROTTLE_KEY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", raising=False)

    app = create_app()
    transport = SyncASGITransport(app)
    with OpenMedClient(base_url=LOOPBACK_BASE_URL, transport=transport) as client:
        yield client


def test_client_calls_service_endpoints_with_asgi_transport(
    rest_client: OpenMedClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze(text: str, **kwargs: Any) -> dict[str, Any]:
        kwargs["loader"].create_pipeline(kwargs["model_name"])
        return {
            "text": text,
            "model_name": kwargs["model_name"],
            "entities": [{"label": "DISEASE", "text": "CML"}],
        }

    def fake_extract(text: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["lang"] == "es"
        return {
            "text": text,
            "entities": [{"label": "NAME", "text": "Maria Garcia"}],
        }

    def fake_deidentify(text: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["method"] == "mask"
        assert kwargs["keep_mapping"] is True
        return {
            "original_text": text,
            "deidentified_text": "Paciente: [NAME]",
            "method": kwargs["method"],
            "mapping": {"[NAME]": "Maria Garcia"},
        }

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    monkeypatch.setattr(openmed, "extract_pii", fake_extract)
    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)

    analyze = rest_client.analyze(
        "Patient started imatinib for CML.",
        keep_alive="forever",
    )
    extract = rest_client.extract_pii(
        "Paciente: Maria Garcia",
        lang="es",
    )
    deidentify = rest_client.deidentify(
        "Paciente: Maria Garcia",
        keep_mapping=True,
    )
    loaded = rest_client.loaded_models()
    unload_one = rest_client.unload_model("disease_detection_superclinical")
    unload_all = rest_client.unload_all_models()

    assert analyze["entities"][0]["label"] == "DISEASE"
    assert extract["entities"][0]["label"] == "NAME"
    assert deidentify["deidentified_text"] == "Paciente: [NAME]"
    assert "disease_detection_superclinical" in loaded["models"]
    assert unload_one["unloaded"] is True
    assert unload_one["released"]["pipelines"] == 1
    assert unload_all["released"]["pipelines"] == 0


def test_client_maps_asgi_error_envelope_to_typed_exception(
    rest_client: OpenMedClient,
) -> None:
    with pytest.raises(OpenMedAPIError) as exc_info:
        rest_client.analyze("   ")

    exc = exc_info.value
    assert exc.status_code == 422
    assert exc.code == "validation_error"
    assert exc.message == "Request validation failed"
    assert exc.details[0]["field"] == "body.text"


def test_client_propagates_request_id_on_error() -> None:
    seen_request_ids: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_request_ids.append(request.headers.get("x-request-id"))
        return httpx.Response(
            400,
            headers={"X-Request-ID": "req-response"},
            json={
                "error": {
                    "code": "bad_request",
                    "message": "Invalid model",
                    "details": {"reason": "Invalid model"},
                }
            },
        )

    transport = httpx.MockTransport(handler)
    with OpenMedClient(
        base_url="http://testserver",
        request_id="req-outgoing",
        transport=transport,
    ) as client:
        with pytest.raises(OpenMedAPIError) as exc_info:
            client.loaded_models()

    assert seen_request_ids == ["req-outgoing"]
    assert exc_info.value.code == "bad_request"
    assert exc_info.value.request_id == "req-response"


def test_client_endpoint_metadata_matches_committed_openapi_spec() -> None:
    spec = json.loads(open("docs/api/openapi.json", encoding="utf-8").read())
    request_types = {
        "brief": BriefRequest,
        "ground": GroundRequest,
        "profile": ProfileRequest,
        "resolve_cohort": CohortResolveRequest,
        "analyze": AnalyzeRequest,
        "extract_pii": PIIExtractRequest,
        "extract_pii_stream": PIIExtractStreamRequest,
        "deidentify": PIIDeidentifyRequest,
        "privacy_gateway": PrivacyGatewayRequest,
        "decision": FixedOptionDecisionRequest,
        "unload_model": ModelUnloadRequest,
        "unload_all_models": ModelUnloadRequest,
    }

    assert set(CLIENT_ENDPOINTS) == {
        "brief",
        "ground",
        "profile",
        "resolve_cohort",
        "analyze",
        "extract_pii",
        "extract_pii_stream",
        "deidentify",
        "privacy_gateway",
        "loaded_models",
        "journey_resources",
        "decision",
        "unload_model",
        "unload_all_models",
    }

    for method_name, endpoint in CLIENT_ENDPOINTS.items():
        assert hasattr(OpenMedClient, method_name)
        operation = spec["paths"][endpoint.path][endpoint.method.lower()]

        if not endpoint.request_fields:
            assert "requestBody" not in operation
            continue

        schema = _request_body_schema(spec, operation)
        assert endpoint.request_fields == set(schema["properties"])

        request_type = request_types[method_name]
        required_fields = {
            field.name
            for field in fields(request_type)
            if field.default is MISSING and field.default_factory is MISSING
        }
        assert required_fields == set(schema.get("required", []))

        type_hints = get_type_hints(request_type)
        for field_name, annotation in type_hints.items():
            literal_values = _literal_values(annotation)
            if not literal_values:
                continue
            openapi_values = _schema_enum_values(schema["properties"][field_name])
            assert literal_values == openapi_values


def _assert_openapi_path_coverage(spec: dict[str, Any]) -> None:
    covered = {endpoint.path for endpoint in CLIENT_ENDPOINTS.values()}
    excluded = set(CLIENT_OPENAPI_EXCLUSIONS)
    assert not covered & excluded, "Implemented paths must not remain excluded"
    assert all(reason.strip() for reason in CLIENT_OPENAPI_EXCLUSIONS.values())
    assert set(spec["paths"]) == covered | excluded, (
        "Every OpenAPI path needs a Python client method or documented exclusion"
    )


def test_client_covers_every_openapi_path_or_explicitly_excludes_it() -> None:
    spec = json.loads(Path("docs/api/openapi.json").read_text(encoding="utf-8"))
    _assert_openapi_path_coverage(spec)
    docs = Path("docs/rest-service.md").read_text(encoding="utf-8")
    for path in CLIENT_OPENAPI_EXCLUSIONS:
        assert f"`{path}`" in docs


def test_new_openapi_path_cannot_silently_escape_client_coverage() -> None:
    spec = json.loads(Path("docs/api/openapi.json").read_text(encoding="utf-8"))
    spec["paths"]["/synthetic-unimplemented-path"] = {"post": {}}
    with pytest.raises(AssertionError, match="Every OpenAPI path"):
        _assert_openapi_path_coverage(spec)


def test_client_concept_hierarchy_fields_match_openapi() -> None:
    spec = json.loads(Path("docs/api/openapi.json").read_text(encoding="utf-8"))
    schema = spec["components"]["schemas"]["ConceptAncestorRequest"]
    assert (
        {field.name for field in fields(ConceptAncestorRequest)}
        == set(schema["properties"])
        == set(schema["required"])
    )
    for request_type in (GroundRequest, ProfileRequest, CohortResolveRequest):
        properties = spec["components"]["schemas"][request_type.__name__]["properties"]
        for field in fields(request_type):
            if "default" in properties[field.name]:
                assert field.default == properties[field.name]["default"]


@pytest.mark.parametrize(
    ("method", "kwargs", "path", "payload"),
    [
        (
            "ground",
            {
                "entities": [{"text": "synthetic finding", "start": 0, "end": 17}],
                "systems": ["icd10cm"],
                "source_language": "fr",
                "top_k": 2,
            },
            "/ground",
            {
                "text": None,
                "entities": [{"text": "synthetic finding", "start": 0, "end": 17}],
                "systems": ["icd10cm"],
                "source_language": "fr",
                "top_k": 2,
                "offline": True,
            },
        ),
        (
            "profile",
            {
                "records_jsonl": "{}\n",
                "completeness_floor": 0.8,
                "required_fields": ["condition"],
                "athena_index": {"synthetic": {}},
            },
            "/profile",
            {
                "records_jsonl": "{}\n",
                "completeness_floor": 0.8,
                "required_fields": ["condition"],
                "athena_index": {"synthetic": {}},
            },
        ),
        (
            "resolve_cohort",
            {
                "phenotype": {"name": "synthetic"},
                "records_jsonl": "{}\n",
                "concept_ancestors": [ConceptAncestorRequest(1, 2)],
                "completeness_floor": 0.7,
                "required_fields": ["condition"],
            },
            "/cohort/resolve",
            {
                "phenotype": {"name": "synthetic"},
                "records_jsonl": "{}\n",
                "concept_ancestors": [
                    {"ancestor_concept_id": 1, "descendant_concept_id": 2}
                ],
                "completeness_floor": 0.7,
                "required_fields": ["condition"],
            },
        ),
    ],
)
def test_clinical_client_methods_use_injected_transport(
    monkeypatch, method, kwargs, path, payload
) -> None:
    def reject_network(*args, **kwargs):
        raise AssertionError("Test must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    seen = []

    def handler(request):
        seen.append(request)
        assert request.method == "POST"
        assert request.url.path == path
        assert json.loads(request.content) == payload
        assert request.headers["X-Request-ID"] == "opaque-request"
        return httpx.Response(200, json={"schema_version": "synthetic-v1"})

    with OpenMedClient(transport=httpx.MockTransport(handler)) as client:
        result = getattr(client, method)(**kwargs, request_id="opaque-request")
    assert result == {"schema_version": "synthetic-v1"}
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("ground", {"text": "synthetic-sensitive-marker"}),
        ("profile", {"records_jsonl": "synthetic-sensitive-marker"}),
        (
            "resolve_cohort",
            {"phenotype": {}, "records_jsonl": "synthetic-sensitive-marker"},
        ),
    ],
)
def test_clinical_client_errors_do_not_include_submitted_content(
    method, kwargs, caplog
) -> None:
    def handler(request):
        return httpx.Response(
            422,
            headers={"X-Request-ID": "opaque-response"},
            json={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": None,
                }
            },
        )

    with OpenMedClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OpenMedAPIError) as caught:
            getattr(client, method)(**kwargs)
    assert caught.value.code == "validation_error"
    assert caught.value.status_code == 422
    assert caught.value.request_id == "opaque-response"
    assert caught.value.details is None
    assert "synthetic-sensitive-marker" not in str(caught.value) + caplog.text


def test_clinical_client_methods_work_with_local_asgi_routes(
    rest_client, tmp_path, monkeypatch
) -> None:
    from openmed.clinical.grounding import VocabLoader
    from openmed.structured.cohort import PhenotypeDefinition

    def reject_network(*args, **kwargs):
        raise AssertionError("Test must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    root = Path(__file__).resolve().parents[3]
    vocabulary = root / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"
    cache = tmp_path / "grounding"
    VocabLoader(cache_dir=cache, local_only=True).import_snapshot(
        "icd10cm", vocabulary, version="synthetic-fixture-1"
    )
    monkeypatch.setenv("OPENMED_GROUNDING_CACHE_DIR", str(cache))
    ground = rest_client.ground("type 2 diabetes", systems=["icd10cm"])
    assert ground["results"][0]["code"] == "E11.9"

    records = '{"note_id":"synthetic","person_id":"synthetic","entities":[]}\n'
    profile = rest_client.profile(records)
    assert profile["gate"]["passed"] is True

    fixtures = root / "tests/fixtures/cohort"
    phenotype = PhenotypeDefinition.load(
        fixtures / "phenotypes/diabetes_on_metformin.json"
    )
    cohort = rest_client.resolve_cohort(
        phenotype.to_dict(),
        (fixtures / "synthetic_grounded.jsonl").read_text(encoding="utf-8"),
        concept_ancestors=[ConceptAncestorRequest(201826, 443238)],
    )
    assert cohort["provenance"]["matched_patient_count"] == 2


def test_client_pii_language_literal_matches_core() -> None:
    from openmed.core.pii_i18n import (
        INDIC_NER_LANGUAGES,
        SUPPORTED_LANGUAGES,
        USER_SUPPLIED_MODEL_LANGUAGES,
    )

    assert set(get_args(PIILanguage)) == (
        SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES | USER_SUPPLIED_MODEL_LANGUAGES
    )


def _request_body_schema(
    spec: dict[str, Any],
    operation: dict[str, Any],
) -> dict[str, Any]:
    schema = operation["requestBody"]["content"]["application/json"]["schema"]
    ref = schema.get("$ref")
    if ref is None:
        return schema

    _, _, schema_name = ref.rpartition("/")
    return spec["components"]["schemas"][schema_name]


def _literal_values(annotation: Any) -> set[Any]:
    if get_origin(annotation) is Literal:
        return set(get_args(annotation))

    values: set[Any] = set()
    for child in get_args(annotation):
        values.update(_literal_values(child))
    return values


def _schema_enum_values(schema: dict[str, Any]) -> set[Any]:
    values = set(schema.get("enum", []))
    for child in schema.get("anyOf", []):
        values.update(_schema_enum_values(child))
    return values
