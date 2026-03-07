"""Unit tests for the OpenMed REST MVP."""

from datetime import datetime

import openmed
import pytest
from fastapi.testclient import TestClient

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service.app import create_app


@pytest.fixture
def client(monkeypatch):
    """Create a test client with a stable profile."""
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


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


def _sample_deid_result(mapping=None) -> DeidentificationResult:
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
        method="mask",
        timestamp=datetime.now(),
        mapping=mapping,
    )


def test_health_returns_ok_and_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "openmed-rest"
    assert payload["version"] == openmed.__version__
    assert payload["profile"] == "test"


def test_analyze_success_returns_prediction_result_shape(client, monkeypatch):
    result = _sample_prediction_result()

    def fake_analyze(*args, **kwargs):
        assert kwargs["output_format"] == "dict"
        assert "config" in kwargs
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


def test_analyze_empty_text_returns_422(client):
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_value_error_returns_400(client, monkeypatch):
    def fake_analyze(*args, **kwargs):
        raise ValueError("Invalid model")

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    response = client.post("/analyze", json={"text": "sample"})

    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"] == "Invalid model"
    assert payload["error_type"] == "ValueError"


def test_pii_extract_success_with_lang_es(client, monkeypatch):
    result = _sample_prediction_result()

    def fake_extract(*args, **kwargs):
        assert kwargs["lang"] == "es"
        return result

    monkeypatch.setattr(openmed, "extract_pii", fake_extract)

    response = client.post(
        "/pii/extract",
        json={"text": "Paciente: Maria Garcia", "lang": "es"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["entities"][0]["label"] == "NAME"


def test_pii_extract_invalid_lang_returns_422(client):
    response = client.post(
        "/pii/extract",
        json={"text": "Paciente: Maria Garcia", "lang": "pt"},
    )
    assert response.status_code == 422


def test_pii_deidentify_mask_success(client, monkeypatch):
    result = _sample_deid_result()

    def fake_deidentify(*args, **kwargs):
        assert kwargs["method"] == "mask"
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


def test_pii_deidentify_invalid_method_returns_422(client):
    response = client.post(
        "/pii/deidentify",
        json={"text": "Paciente: Maria Garcia", "method": "invalid"},
    )
    assert response.status_code == 422


def test_unhandled_exception_returns_500(client, monkeypatch):
    def fake_analyze(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    response = client.post("/analyze", json={"text": "sample"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_internal_payload_conversion_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(openmed, "analyze_text", lambda *args, **kwargs: ["invalid"])

    response = client.post("/analyze", json={"text": "sample"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_app_uses_profile_config_from_env(monkeypatch):
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    app = create_app()

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["profile"] == "dev"


def test_app_defaults_to_prod_profile_when_env_missing(monkeypatch):
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)
    app = create_app()

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["profile"] == "prod"
