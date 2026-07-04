"""Unit tests for the OpenMed REST service."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app

LOOPBACK_BASE_URL = "http://127.0.0.1"


@pytest.fixture(autouse=True)
def clear_security_env(monkeypatch):
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)


class FakeLoader:
    """Simple loader double with per-instance pipeline caching."""

    instances: list["FakeLoader"] = []

    def __init__(self, config):
        self.config = config
        self.pipelines = {}
        self.pipeline_creations = 0
        self.create_pipeline_calls = []
        FakeLoader.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def create_pipeline(self, model_name: str, **kwargs: Any):
        self.create_pipeline_calls.append((model_name, kwargs))
        key = (
            model_name,
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
        )
        if key not in self.pipelines:
            self.pipeline_creations += 1
            self.pipelines[key] = object()
        return self.pipelines[key]

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        model_names = {key[0] for key in self.pipelines}
        return {
            model_name: {
                "models": 0,
                "tokenizers": 0,
                "pipelines": sum(1 for key in self.pipelines if key[0] == model_name),
            }
            for model_name in sorted(model_names)
        }

    def unload_model(self, model_name: str) -> dict[str, Any]:
        keys = [key for key in self.pipelines if key[0] == model_name]
        for key in keys:
            self.pipelines.pop(key, None)
        return {
            "model_name": model_name,
            "models": 0,
            "tokenizers": 0,
            "pipelines": len(keys),
        }

    def unload_all_models(self) -> dict[str, int]:
        released = {"models": 0, "tokenizers": 0, "pipelines": len(self.pipelines)}
        self.pipelines.clear()
        return released


def _sample_prediction_result() -> PredictionResult:
    return PredictionResult(
        text="Paciente: Maria Garcia",
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
        timestamp=datetime.now().isoformat(),
        processing_time=0.01,
        metadata={"sentence_detection": True},
    )


def _sample_deid_result(
    mapping=None, *, method: str = "mask"
) -> DeidentificationResult:
    pii_entity = PIIEntity(
        text="Maria Garcia",
        label="NAME",
        confidence=0.98,
        start=10,
        end=22,
        entity_type="NAME",
        redacted_text="[NAME]",
        original_text="Maria Garcia",
    )
    return DeidentificationResult(
        original_text="Paciente: Maria Garcia",
        deidentified_text="Paciente: [NAME]",
        pii_entities=[pii_entity],
        method=method,
        timestamp=datetime.now(),
        mapping=mapping,
    )


def _assert_error_payload(response, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    payload = response.json()
    assert payload["error"]["code"] == code
    assert "message" in payload["error"]
    assert "details" in payload["error"]
    return payload


@pytest.fixture
def fake_loader_cls(monkeypatch):
    FakeLoader.reset()
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    return FakeLoader


@pytest.fixture
def client(monkeypatch, fake_loader_cls):
    """Create a test client with a stable profile and fake loader."""
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
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
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def test_health_returns_ok_and_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "openmed-rest"
    assert payload["version"] == openmed.__version__
    assert payload["profile"] == "test"


def test_analyze_success_returns_prediction_result_shape(
    client, monkeypatch, fake_loader_cls
):
    result = _sample_prediction_result()

    def fake_analyze(*args, **kwargs):
        assert kwargs["output_format"] == "dict"
        assert kwargs["loader"].loader is fake_loader_cls.instances[0]
        return result

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    response = client.post(
        "/analyze",
        json={
            "text": "Paciente: Maria Garcia",
            "model_name": "disease_detection_superclinical",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Paciente: Maria Garcia"
    assert payload["model_name"] == "disease_detection_superclinical"
    assert isinstance(payload["entities"], list)
    assert payload["entities"][0]["label"] == "NAME"


def test_analyze_blank_text_returns_validation_error(client):
    response = client.post("/analyze", json={"text": "   "})

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.text"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/analyze", {}),
        ("/pii/extract", {}),
        ("/pii/deidentify", {"method": "mask"}),
        ("/privacy-gateway/complete", {}),
    ],
)
def test_oversized_text_returns_validation_error(
    client,
    monkeypatch,
    path,
    payload,
):
    monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", "10")
    response = client.post(path, json={"text": "x" * 11, **payload})

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.text"


def test_invalid_max_text_length_env_falls_back_to_default(monkeypatch):
    from openmed.service.limits import (
        DEFAULT_MAX_TEXT_LENGTH,
        SERVICE_MAX_TEXT_LENGTH_ENV_VAR,
        get_max_text_length,
    )

    monkeypatch.setenv(SERVICE_MAX_TEXT_LENGTH_ENV_VAR, "not-an-int")

    assert get_max_text_length() == DEFAULT_MAX_TEXT_LENGTH


def test_analyze_invalid_confidence_threshold_returns_validation_error(client):
    response = client.post(
        "/analyze", json={"text": "sample", "confidence_threshold": 1.5}
    )

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.confidence_threshold"


def test_analyze_invalid_model_name_returns_validation_error(client):
    response = client.post(
        "/analyze", json={"text": "sample", "model_name": "bad model"}
    )

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.model_name"


def test_analyze_invalid_aggregation_strategy_returns_validation_error(client):
    response = client.post(
        "/analyze",
        json={"text": "sample", "aggregation_strategy": "median"},
    )

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.aggregation_strategy"


def test_analyze_extra_field_returns_validation_error(client):
    response = client.post("/analyze", json={"text": "sample", "unexpected": True})

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.unexpected"


def test_analyze_value_error_returns_bad_request(client, monkeypatch):
    def fake_analyze(*args, **kwargs):
        raise ValueError("Invalid model")

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    response = client.post("/analyze", json={"text": "sample"})

    payload = _assert_error_payload(response, 400, "bad_request")
    assert payload["error"]["details"] == {"reason": "Invalid model"}


def test_model_memory_backpressure_returns_service_busy(
    monkeypatch,
    fake_loader_cls,
):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.setenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", "1")
    monkeypatch.setenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", "2")
    monkeypatch.setenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", "0")

    def fake_analyze(*args, **kwargs):
        kwargs["loader"].create_pipeline(
            kwargs["model_name"],
            task="token-classification",
            aggregation_strategy=kwargs["aggregation_strategy"],
            use_fast_tokenizer=kwargs["use_fast_tokenizer"],
        )
        return _sample_prediction_result()

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        response = test_client.post(
            "/analyze",
            json={
                "text": "sample",
                "model_name": "disease_detection_superclinical",
            },
        )

    payload = _assert_error_payload(response, 503, "service_busy")
    assert payload["error"]["details"]["required_bytes"] == 2
    assert payload["error"]["details"]["budget_bytes"] == 1


def test_pii_extract_success_with_lang_es(client, monkeypatch, fake_loader_cls):
    result = _sample_prediction_result()

    def fake_extract(*args, **kwargs):
        assert kwargs["lang"] == "es"
        assert kwargs["loader"].loader is fake_loader_cls.instances[0]
        return result

    monkeypatch.setattr(openmed, "extract_pii", fake_extract)

    response = client.post(
        "/pii/extract",
        json={"text": "Paciente: Maria Garcia", "lang": "es"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["entities"][0]["label"] == "NAME"


def test_pii_extract_stream_returns_ndjson_events_without_audit_phi(
    client,
    monkeypatch,
    fake_loader_cls,
):
    def fake_extract(text, *args, **kwargs):
        entities = []
        marker = "jane.patient@example.com"
        start = text.find(marker)
        if start >= 0:
            entities.append(
                EntityPrediction(
                    text=marker,
                    label="EMAIL",
                    confidence=0.99,
                    start=start,
                    end=start + len(marker),
                )
            )
        return PredictionResult(
            text=text,
            entities=entities,
            model_name="privacy-filter",
            timestamp=datetime.now().isoformat(),
        )

    monkeypatch.setattr(openmed, "extract_pii", fake_extract)

    with client.stream(
        "POST",
        "/pii/extract/stream",
        json={
            "text": "Contact jane.patient@example.com today.",
            "chunk_size": 8,
            "window_chars": 64,
            "tokenizer_context_chars": 16,
            "max_entity_chars": 32,
            "include_text": False,
        },
    ) as response:
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert any(event["type"] == "emit" for event in events)
    assert events[-1]["type"] == "final"
    emit_event = next(event for event in events if event["type"] == "emit")
    assert "text" not in emit_event["span"]
    audit_payload = json.dumps([event["audit"] for event in events])
    assert "jane.patient" not in audit_payload
    assert fake_loader_cls.instances[0].loaded_models() == {}


@pytest.mark.parametrize("lang", ["nl", "hi", "te", "ar", "ja", "tr"])
def test_pii_extract_accepts_new_langs(client, monkeypatch, fake_loader_cls, lang):
    result = _sample_prediction_result()

    def fake_extract(*args, **kwargs):
        assert kwargs["lang"] == lang
        assert kwargs["loader"].loader is fake_loader_cls.instances[0]
        return result

    monkeypatch.setattr(openmed, "extract_pii", fake_extract)

    response = client.post(
        "/pii/extract",
        json={"text": "Paciente: Maria Garcia", "lang": lang},
    )

    assert response.status_code == 200


def test_pii_extract_invalid_lang_returns_validation_error(client):
    response = client.post(
        "/pii/extract",
        json={"text": "Paciente: Maria Garcia", "lang": "xx"},
    )

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.lang"


def test_pii_lang_literal_matches_supported_languages():
    """The REST schema must accept exactly the languages the core supports.

    Regression guard: ar/ja/tr shipped with published PII models and were in
    ``SUPPORTED_LANGUAGES`` but were missing from the schema ``Literal``, so the
    REST/MCP layer rejected them. Keep the two in lockstep.
    """
    from typing import get_args

    from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
    from openmed.service.schemas import PIILanguage

    assert set(get_args(PIILanguage)) == set(SUPPORTED_LANGUAGES)


def test_pii_deidentify_mask_success(client, monkeypatch, fake_loader_cls):
    result = _sample_deid_result()

    def fake_deidentify(*args, **kwargs):
        assert kwargs["method"] == "mask"
        assert kwargs["keep_year"] is False
        assert kwargs["loader"].loader is fake_loader_cls.instances[0]
        return result

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)

    response = client.post(
        "/pii/deidentify",
        json={"text": "Paciente: Maria Garcia", "method": "mask"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["deidentified_text"] == "Paciente: [NAME]"
    assert payload["method"] == "mask"


@pytest.mark.parametrize("lang", ["nl", "hi", "te", "ar", "ja", "tr"])
def test_pii_deidentify_accepts_new_langs(client, monkeypatch, fake_loader_cls, lang):
    result = _sample_deid_result()

    def fake_deidentify(*args, **kwargs):
        assert kwargs["lang"] == lang
        assert kwargs["loader"].loader is fake_loader_cls.instances[0]
        return result

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)

    response = client.post(
        "/pii/deidentify",
        json={"text": "Paciente: Maria Garcia", "method": "mask", "lang": lang},
    )

    assert response.status_code == 200


def test_pii_deidentify_keep_mapping_includes_mapping_key(client, monkeypatch):
    result = _sample_deid_result(mapping={"[NAME]": "Maria Garcia"})

    monkeypatch.setattr(openmed, "deidentify", lambda *args, **kwargs: result)

    response = client.post(
        "/pii/deidentify",
        json={
            "text": "Paciente: Maria Garcia",
            "method": "mask",
            "keep_mapping": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mapping"] == {"[NAME]": "Maria Garcia"}


def test_pii_deidentify_shift_dates_alias_promotes_method(client, monkeypatch):
    def fake_deidentify(*args, **kwargs):
        assert kwargs["method"] == "shift_dates"
        assert kwargs["shift_dates"] is True
        assert kwargs["date_shift_days"] == 30
        return _sample_deid_result(method="shift_dates")

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)

    response = client.post(
        "/pii/deidentify",
        json={
            "text": "Paciente: Maria Garcia",
            "shift_dates": True,
            "date_shift_days": 30,
        },
    )

    assert response.status_code == 200
    assert response.json()["method"] == "shift_dates"


def test_pii_deidentify_invalid_shift_dates_combination_returns_validation_error(
    client,
):
    response = client.post(
        "/pii/deidentify",
        json={
            "text": "Paciente: Maria Garcia",
            "method": "shift_dates",
            "shift_dates": False,
        },
    )

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body"


def test_pii_deidentify_date_shift_days_requires_shift_method(client):
    response = client.post(
        "/pii/deidentify",
        json={
            "text": "Paciente: Maria Garcia",
            "method": "mask",
            "date_shift_days": 30,
        },
    )

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body"


def test_unhandled_exception_returns_internal_error(client, monkeypatch):
    def fake_analyze(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    response = client.post("/analyze", json={"text": "sample"})

    payload = _assert_error_payload(response, 500, "internal_error")
    assert payload["error"]["details"] is None


def test_internal_payload_conversion_error_returns_internal_error(client, monkeypatch):
    monkeypatch.setattr(openmed, "analyze_text", lambda *args, **kwargs: ["invalid"])

    response = client.post("/analyze", json={"text": "sample"})

    payload = _assert_error_payload(response, 500, "internal_error")
    assert payload["error"]["details"] is None


@pytest.mark.parametrize(
    ("endpoint", "patch_target", "payload"),
    [
        ("/analyze", "analyze_text", {"text": "sample"}),
        ("/pii/extract", "extract_pii", {"text": "sample"}),
        ("/pii/deidentify", "deidentify", {"text": "sample"}),
    ],
)
def test_service_timeouts_return_gateway_timeout(
    client,
    monkeypatch,
    endpoint,
    patch_target,
    payload,
):
    def slow_call(*args, **kwargs):
        time.sleep(0.05)
        return _sample_prediction_result()

    if patch_target == "deidentify":
        slow_call = lambda *args, **kwargs: _sample_deid_result()

        def timed_deidentify(*args, **kwargs):
            time.sleep(0.05)
            return _sample_deid_result()

        monkeypatch.setattr(openmed, patch_target, timed_deidentify)
    else:
        monkeypatch.setattr(openmed, patch_target, slow_call)

    client.app.state.runtime.config.timeout = 0.01

    response = client.post(endpoint, json=payload)

    payload = _assert_error_payload(response, 504, "timeout")
    assert payload["error"]["details"] == {"timeout_seconds": 0.01}


def test_app_uses_profile_config_from_env(monkeypatch, fake_loader_cls):
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", raising=False)
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["profile"] == "dev"


def test_app_defaults_to_prod_profile_when_env_missing(monkeypatch, fake_loader_cls):
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", raising=False)
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["profile"] == "prod"


def test_preload_models_are_parsed_deduped_and_warmed(monkeypatch, fake_loader_cls):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv(
        "OPENMED_SERVICE_PRELOAD_MODELS",
        " disease_detection_superclinical , OpenMed/model-two , disease_detection_superclinical ",
    )
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL):
        runtime = app.state.runtime
        loader = fake_loader_cls.instances[0]

    assert runtime.preload_models == (
        "disease_detection_superclinical",
        "OpenMed/model-two",
    )
    assert loader.pipeline_creations == 2


def test_preload_invalid_model_name_fails_startup(monkeypatch, fake_loader_cls):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_PRELOAD_MODELS", "bad model")
    app = create_app()

    with pytest.raises(ValueError, match="Invalid characters in model name"):
        with TestClient(app, base_url=LOOPBACK_BASE_URL):
            pass


def test_preload_model_load_failure_fails_startup(monkeypatch):
    class FailingLoader(FakeLoader):
        def create_pipeline(self, model_name: str, **kwargs: Any):
            raise ValueError(f"Could not load model {model_name}")

    FakeLoader.reset()
    monkeypatch.setattr(service_runtime, "ModelLoader", FailingLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv(
        "OPENMED_SERVICE_PRELOAD_MODELS", "disease_detection_superclinical"
    )
    app = create_app()

    with pytest.raises(ValueError, match="Could not load model"):
        with TestClient(app, base_url=LOOPBACK_BASE_URL):
            pass


def test_pii_extract_rejects_attacker_controlled_privacy_filter_model_name(
    client,
    monkeypatch,
    fake_loader_cls,
):
    """CVE-2026-47117 regression: a request with an attacker-controlled
    repo name whose path contains "privacy-filter" must NOT route through
    the privacy-filter dispatcher (which would otherwise load with
    trust_remote_code=True). It should fall through to the standard PII
    loader, which never enables custom-code execution.
    """
    monkeypatch.setattr(
        openmed, "analyze_text", lambda *args, **kwargs: _sample_prediction_result()
    )

    with (
        patch(
            "openmed.torch.privacy_filter.PrivacyFilterTorchPipeline",
        ) as MockPipeline,
        patch(
            "openmed.core.backends.create_privacy_filter_pipeline",
        ) as mock_factory,
    ):
        response = client.post(
            "/pii/extract",
            json={
                "text": "John Doe called 555-1212",
                "model_name": "attacker/foo-privacy-filter-bar",
                "confidence_threshold": 0.0,
            },
        )

    assert response.status_code == 200
    MockPipeline.assert_not_called()
    mock_factory.assert_not_called()


def test_pii_deidentify_rejects_attacker_controlled_privacy_filter_model_name(
    client,
    monkeypatch,
    fake_loader_cls,
):
    """CVE-2026-47117 regression: same as the /pii/extract case but for the
    /pii/deidentify endpoint, which reaches the same vulnerable code path."""
    monkeypatch.setattr(
        openmed, "analyze_text", lambda *args, **kwargs: _sample_prediction_result()
    )

    with (
        patch(
            "openmed.torch.privacy_filter.PrivacyFilterTorchPipeline",
        ) as MockPipeline,
        patch(
            "openmed.core.backends.create_privacy_filter_pipeline",
        ) as mock_factory,
    ):
        response = client.post(
            "/pii/deidentify",
            json={
                "text": "John Doe called 555-1212",
                "method": "mask",
                "model_name": "attacker/foo-privacy-filter-bar",
                "confidence_threshold": 0.0,
            },
        )

    assert response.status_code == 200
    MockPipeline.assert_not_called()
    mock_factory.assert_not_called()


def test_second_request_reuses_shared_warmed_pipeline(monkeypatch, fake_loader_cls):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv(
        "OPENMED_SERVICE_PRELOAD_MODELS", "disease_detection_superclinical"
    )

    def fake_analyze(*args, **kwargs):
        kwargs["loader"].create_pipeline(
            kwargs["model_name"],
            task="token-classification",
            aggregation_strategy=kwargs["aggregation_strategy"],
            use_fast_tokenizer=kwargs["use_fast_tokenizer"],
        )
        return _sample_prediction_result()

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        first = test_client.post("/analyze", json={"text": "sample"})
        second = test_client.post("/analyze", json={"text": "sample"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(fake_loader_cls.instances) == 1
    assert fake_loader_cls.instances[0].pipeline_creations == 1


def test_keep_alive_zero_unloads_pipeline_after_request(
    client, monkeypatch, fake_loader_cls
):
    def fake_analyze(*args, **kwargs):
        kwargs["loader"].create_pipeline(
            kwargs["model_name"],
            task="token-classification",
            aggregation_strategy=kwargs["aggregation_strategy"],
            use_fast_tokenizer=kwargs["use_fast_tokenizer"],
        )
        return _sample_prediction_result()

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    response = client.post("/analyze", json={"text": "sample", "keep_alive": 0})

    assert response.status_code == 200
    assert fake_loader_cls.instances[0].pipelines == {}


def test_default_keep_alive_env_unloads_pipeline_after_request(
    monkeypatch,
    fake_loader_cls,
):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", raising=False)
    monkeypatch.setenv("OPENMED_SERVICE_KEEP_ALIVE", "0")

    def fake_analyze(*args, **kwargs):
        kwargs["loader"].create_pipeline(
            kwargs["model_name"],
            task="token-classification",
            aggregation_strategy=kwargs["aggregation_strategy"],
            use_fast_tokenizer=kwargs["use_fast_tokenizer"],
        )
        return _sample_prediction_result()

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        response = test_client.post("/analyze", json={"text": "sample"})
        loaded = test_client.get("/models/loaded")

    assert response.status_code == 200
    assert loaded.status_code == 200
    assert loaded.json()["default_keep_alive_seconds"] == 0.0
    assert fake_loader_cls.instances[0].pipelines == {}


def test_invalid_keep_alive_returns_validation_error(client):
    response = client.post(
        "/analyze",
        json={"text": "sample", "keep_alive": "five-ish"},
    )

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.keep_alive"


def test_manual_unload_model_endpoint_releases_cached_pipeline(
    client,
    monkeypatch,
    fake_loader_cls,
):
    def fake_analyze(*args, **kwargs):
        kwargs["loader"].create_pipeline(
            kwargs["model_name"],
            task="token-classification",
            aggregation_strategy=kwargs["aggregation_strategy"],
            use_fast_tokenizer=kwargs["use_fast_tokenizer"],
        )
        return _sample_prediction_result()

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    analyze = client.post(
        "/analyze",
        json={"text": "sample", "keep_alive": "forever"},
    )
    loaded_before = client.get("/models/loaded")
    unload = client.post(
        "/models/unload",
        json={"model_name": "disease_detection_superclinical"},
    )
    loaded_after = client.get("/models/loaded")

    assert analyze.status_code == 200
    assert loaded_before.status_code == 200
    assert "disease_detection_superclinical" in loaded_before.json()["models"]
    assert unload.status_code == 200
    assert unload.json()["unloaded"] is True
    assert unload.json()["released"]["pipelines"] == 1
    assert loaded_after.json()["models"] == {}
    assert fake_loader_cls.instances[0].pipelines == {}


def test_manual_unload_all_endpoint_releases_cached_pipelines(
    client,
    monkeypatch,
    fake_loader_cls,
):
    def fake_analyze(*args, **kwargs):
        kwargs["loader"].create_pipeline(
            kwargs["model_name"],
            task="token-classification",
            aggregation_strategy=kwargs["aggregation_strategy"],
            use_fast_tokenizer=kwargs["use_fast_tokenizer"],
        )
        return _sample_prediction_result()

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    first = client.post(
        "/analyze",
        json={
            "text": "sample",
            "model_name": "disease_detection_superclinical",
            "keep_alive": "forever",
        },
    )
    second = client.post(
        "/analyze",
        json={
            "text": "sample",
            "model_name": "OpenMed/model-two",
            "keep_alive": "forever",
        },
    )
    unload = client.post("/models/unload", json={"all": True})

    assert first.status_code == 200
    assert second.status_code == 200
    assert unload.status_code == 200
    assert unload.json()["unloaded"] is True
    assert unload.json()["released"]["pipelines"] == 2
    assert fake_loader_cls.instances[0].pipelines == {}


def test_unload_endpoint_requires_model_or_all(client):
    response = client.post("/models/unload", json={})

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body"
