"""Tests for the typed OpenMed REST client."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

import openmed
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.client import (
    CLIENT_ENDPOINTS,
    OpenMedAPIError,
    OpenMedClient,
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

    assert set(CLIENT_ENDPOINTS) == {
        "analyze",
        "extract_pii",
        "deidentify",
        "loaded_models",
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
        assert endpoint.request_fields <= set(schema["properties"])


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
