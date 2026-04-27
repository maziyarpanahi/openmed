"""Unit tests for the OpenMed REST service."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import openmed
import pytest
from fastapi.testclient import TestClient

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service.app import create_app
from openmed.service import runtime as service_runtime


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


def _sample_deid_result(mapping=None, *, method: str = "mask") -> DeidentificationResult:
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
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_returns_ok_and_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "openmed-rest"
    assert payload["version"] == openmed.__version__
    assert payload["profile"] == "test"


def test_analyze_success_returns_prediction_result_shape(client, monkeypatch, fake_loader_cls):
    result = _sample_prediction_result()

    def fake_analyze(*args, **kwargs):
        assert kwargs["output_format"] == "dict"
        assert kwargs["loader"] is fake_loader_cls.instances[0]
        return result

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    response = client.post(
        "/analyze",
        json={"text": "Paciente: Maria Garcia", "model_name": "disease_detection_superclinical"},
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


def test_analyze_invalid_confidence_threshold_returns_validation_error(client):
    response = client.post("/analyze", json={"text": "sample", "confidence_threshold": 1.5})

    payload = _assert_error_payload(response, 422, "validation_error")
    assert payload["error"]["details"][0]["field"] == "body.confidence_threshold"


def test_analyze_invalid_model_name_returns_validation_error(client):
    response = client.post("/analyze", json={"text": "sample", "model_name": "bad model"})

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


def test_pii_extract_success_with_lang_es(client, monkeypatch, fake_loader_cls):
    result = _sample_prediction_result()

    def fake_extract(*args, **kwargs):
        assert kwargs["lang"] == "es"
        assert kwargs["loader"] is fake_loader_cls.instances[0]
        return result

    monkeypatch.setattr(openmed, "extract_pii", fake_extract)

    response = client.post(
        "/pii/extract",
        json={"text": "Paciente: Maria Garcia", "lang": "es"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["entities"][0]["label"] == "NAME"


@pytest.mark.parametrize("lang", ["nl", "hi", "te"])
def test_pii_extract_accepts_new_langs(client, monkeypatch, fake_loader_cls, lang):
    result = _sample_prediction_result()

    def fake_extract(*args, **kwargs):
        assert kwargs["lang"] == lang
        assert kwargs["loader"] is fake_loader_cls.instances[0]
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


def test_pii_deidentify_mask_success(client, monkeypatch, fake_loader_cls):
    result = _sample_deid_result()

    def fake_deidentify(*args, **kwargs):
        assert kwargs["method"] == "mask"
        assert kwargs["loader"] is fake_loader_cls.instances[0]
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


@pytest.mark.parametrize("lang", ["nl", "hi", "te"])
def test_pii_deidentify_accepts_new_langs(client, monkeypatch, fake_loader_cls, lang):
    result = _sample_deid_result()

    def fake_deidentify(*args, **kwargs):
        assert kwargs["lang"] == lang
        assert kwargs["loader"] is fake_loader_cls.instances[0]
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


def test_pii_deidentify_invalid_shift_dates_combination_returns_validation_error(client):
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
    app = create_app()

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["profile"] == "dev"


def test_app_defaults_to_prod_profile_when_env_missing(monkeypatch, fake_loader_cls):
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    app = create_app()

    with TestClient(app) as test_client:
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

    with TestClient(app):
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
        with TestClient(app):
            pass


def test_preload_model_load_failure_fails_startup(monkeypatch):
    class FailingLoader(FakeLoader):
        def create_pipeline(self, model_name: str, **kwargs: Any):
            raise ValueError(f"Could not load model {model_name}")

    FakeLoader.reset()
    monkeypatch.setattr(service_runtime, "ModelLoader", FailingLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_PRELOAD_MODELS", "disease_detection_superclinical")
    app = create_app()

    with pytest.raises(ValueError, match="Could not load model"):
        with TestClient(app):
            pass


def test_second_request_reuses_shared_warmed_pipeline(monkeypatch, fake_loader_cls):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_PRELOAD_MODELS", "disease_detection_superclinical")

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

    with TestClient(app, raise_server_exceptions=False) as test_client:
        first = test_client.post("/analyze", json={"text": "sample"})
        second = test_client.post("/analyze", json={"text": "sample"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(fake_loader_cls.instances) == 1
    assert fake_loader_cls.instances[0].pipeline_creations == 1
