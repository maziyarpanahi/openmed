"""Tests for optional no-PHI OpenTelemetry tracing."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import format_trace_id
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import openmed
from openmed.service import runtime as service_runtime
from openmed.service import tracing as service_tracing
from openmed.service.app import create_app
from openmed.service.logging import REQUEST_ID_HEADER
from openmed.service.tracing import (
    OTLP_ENDPOINT_ENV_VAR,
    TRACING_ENABLED_ENV_VAR,
    ServiceTraceConfig,
    ServiceTracing,
    safe_trace_attributes,
    service_tracing_from_env,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
PHI_TEXT = "Patient Juniper Solstice, MRN JS-1188, DOB 02/03/1979, phone 425-555-0199."
PHI_SUBSTRINGS = (
    "Juniper Solstice",
    "JS-1188",
    "02/03/1979",
    "425-555-0199",
)
SERVICE_APP_MODULE = importlib.import_module("openmed.service.app")


class FakeLoader:
    """Loader double that prevents model downloads in tracing tests."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}

    def unload_model(self, model_name: str) -> dict[str, Any]:
        return {
            "model_name": model_name,
            "models": 0,
            "tokenizers": 0,
            "pipelines": 0,
        }

    def unload_all_models(self) -> dict[str, int]:
        return {"models": 0, "tokenizers": 0, "pipelines": 0}


@pytest.fixture(autouse=True)
def service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
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
    monkeypatch.delenv(TRACING_ENABLED_ENV_VAR, raising=False)
    monkeypatch.delenv(OTLP_ENDPOINT_ENV_VAR, raising=False)


@pytest.fixture
def span_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "openmed-rest"})
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = ServiceTracing(
        config=ServiceTraceConfig(enabled=True),
        tracer_provider=provider,
        tracer=provider.get_tracer("openmed.service.test"),
        propagator=TraceContextTextMapPropagator(),
    )
    monkeypatch.setattr(SERVICE_APP_MODULE, "service_tracing_from_env", lambda: tracing)
    return exporter


def _span_by_name(spans: list[Any], name: str) -> Any:
    matches = [span for span in spans if span.name == name]
    assert len(matches) == 1
    return matches[0]


def test_tracing_is_disabled_by_default_without_otlp_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExporterSentinel:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise AssertionError("OTLP exporter should not be created by default")

    monkeypatch.setattr(service_tracing, "OTLPSpanExporter", ExporterSentinel)

    tracing = service_tracing_from_env()

    assert not tracing.enabled


def test_synthetic_request_emits_parent_and_child_spans_without_phi(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extract_pii(text: str, **kwargs: Any) -> dict[str, Any]:
        del text
        return {
            "entities": [
                {
                    "text": "Juniper Solstice",
                    "label": "NAME",
                    "start": 8,
                    "end": 24,
                    "confidence": 0.99,
                }
            ],
            "model_name": kwargs["model_name"],
        }

    monkeypatch.setattr(openmed, "extract_pii", fake_extract_pii)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/pii/extract",
            headers={REQUEST_ID_HEADER: "req-trace-123"},
            json={"text": PHI_TEXT, "model_name": "test-pii-model"},
        )

    assert response.status_code == 200
    spans = list(span_exporter.get_finished_spans())
    request_span = _span_by_name(spans, "openmed.service.request")
    model_span = _span_by_name(spans, "openmed.service.model_request")
    pipeline_span = _span_by_name(spans, "openmed.service.pii_extract_pipeline")

    assert request_span.attributes["http.route"] == "/pii/extract"
    assert request_span.attributes["openmed.request_id"] == "req-trace-123"
    assert model_span.parent.span_id == request_span.context.span_id
    assert pipeline_span.parent.span_id == model_span.context.span_id
    assert pipeline_span.attributes["openmed.input.length"] == len(PHI_TEXT)
    assert pipeline_span.attributes["openmed.entity.count"] == 1
    assert pipeline_span.attributes["openmed.entity.labels"] == ("NAME",)

    rendered_attributes = "\n".join(
        f"{key}={value}" for span in spans for key, value in span.attributes.items()
    )
    leaked = [
        substring for substring in PHI_SUBSTRINGS if substring in rendered_attributes
    ]
    assert leaked == []


def test_incoming_traceparent_is_used_as_request_span_parent(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    parent_span_id = "00f067aa0ba902b7"

    def fake_analyze(text: str, **kwargs: Any) -> dict[str, Any]:
        del text
        return {
            "text": "",
            "entities": [],
            "model_name": kwargs["model_name"],
        }

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/analyze",
            headers={
                "traceparent": f"00-{trace_id}-{parent_span_id}-01",
                REQUEST_ID_HEADER: "req-upstream-trace",
            },
            json={"text": "short synthetic note", "model_name": "test-model"},
        )

    assert response.status_code == 200
    request_span = _span_by_name(
        list(span_exporter.get_finished_spans()),
        "openmed.service.request",
    )
    assert format_trace_id(request_span.context.trace_id) == trace_id
    assert request_span.parent.span_id == int(parent_span_id, 16)


def test_span_attribute_guard_drops_unapproved_text_keys() -> None:
    attributes = safe_trace_attributes(
        {
            "openmed.input.text": PHI_TEXT,
            "openmed.response.text": "Patient Juniper Solstice",
            "openmed.entity.text": "Juniper Solstice",
            "openmed.entity.count": 1,
            "openmed.entity.labels": ["NAME"],
        }
    )

    assert attributes == {
        "openmed.entity.count": 1,
        "openmed.entity.labels": ["NAME"],
    }
