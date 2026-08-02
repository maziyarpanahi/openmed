"""Guard that PII and de-identification paths do not log raw PHI."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.pii import deidentify, extract_pii
from openmed.processing.batch import BatchProcessor
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.processing.text import TextProcessor
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app

LOOPBACK_BASE_URL = "http://127.0.0.1"

PHI_TEXT = (
    "Patient Evelyn Quantum, MRN ZQ-7391, DOB 04/17/1972, "
    "SSN 042-66-9001, phone 919-555-0188, email "
    "evelyn.quantum@example.test, address 9 Radiant Plaza."
)

PHI_SUBSTRINGS = (
    "Evelyn Quantum",
    "ZQ-7391",
    "04/17/1972",
    "042-66-9001",
    "919-555-0188",
    "evelyn.quantum@example.test",
    "9 Radiant Plaza",
)

PHI_LABELS = {
    "Evelyn Quantum": "NAME",
    "ZQ-7391": "MEDICAL_RECORD_NUMBER",
    "04/17/1972": "DATE",
    "042-66-9001": "SSN",
    "919-555-0188": "PHONE",
    "evelyn.quantum@example.test": "EMAIL",
    "9 Radiant Plaza": "STREET_ADDRESS",
}


class _NoopLoader:
    """Loader double used by REST service tests to avoid model loading."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, Any]:
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


def _entities_for(text: str) -> list[EntityPrediction]:
    entities = []
    for substring, label in PHI_LABELS.items():
        start = text.index(substring)
        entities.append(
            EntityPrediction(
                text=substring,
                label=label,
                start=start,
                end=start + len(substring),
                confidence=0.99,
            )
        )
    return entities


def _fake_analyze_text(text: str, *args: Any, **kwargs: Any) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=_entities_for(text),
        model_name=kwargs.get("model_name", "test-pii-model"),
        timestamp=datetime.now().isoformat(),
    )


def _fake_extract_pii(text: str, *args: Any, **kwargs: Any) -> PredictionResult:
    return extract_pii(text, *args, **kwargs)


def _fake_deidentify(text: str, *args: Any, **kwargs: Any) -> Any:
    return deidentify(text, *args, **kwargs)


def _render_logs(records: list[logging.LogRecord]) -> str:
    return "\n".join(
        f"{record.name} {record.levelname} {record.getMessage()} {record.args!r}"
        for record in records
    )


def _assert_no_phi_substrings(log_output: str) -> None:
    leaked = [substring for substring in PHI_SUBSTRINGS if substring in log_output]
    assert leaked == [], f"raw PHI leaked into logs: {leaked!r}"


def test_guard_detects_intentional_raw_phi_log_payload():
    log_output = f"openmed.test INFO processing source text {PHI_SUBSTRINGS[0]}"

    with pytest.raises(AssertionError, match="raw PHI leaked into logs"):
        _assert_no_phi_substrings(log_output)


def test_pii_processing_and_service_paths_do_not_log_raw_phi(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(openmed, "analyze_text", _fake_analyze_text)
    monkeypatch.setattr(openmed, "extract_pii", _fake_extract_pii)
    monkeypatch.setattr(openmed, "deidentify", _fake_deidentify)
    monkeypatch.setattr(service_runtime, "ModelLoader", _NoopLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)

    caplog.set_level(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        cleaned = TextProcessor().clean_text(PHI_TEXT)
        pii_result = extract_pii(PHI_TEXT, model_name="test-pii-model")
        deid_result = deidentify(PHI_TEXT, model_name="test-pii-model")
        batch_result = BatchProcessor(
            model_name="test-pii-model",
            operation="deidentify",
            batch_size=1,
        ).process_texts([PHI_TEXT], ids=["case-001"])

        app = create_app()
        with TestClient(
            app,
            base_url=LOOPBACK_BASE_URL,
            raise_server_exceptions=False,
        ) as client:
            extract_response = client.post(
                "/pii/extract",
                json={"text": PHI_TEXT, "model_name": "test-pii-model"},
            )
            deid_response = client.post(
                "/pii/deidentify",
                json={"text": PHI_TEXT, "model_name": "test-pii-model"},
            )

    assert cleaned
    assert len(pii_result.entities) >= len(PHI_SUBSTRINGS)
    assert deid_result.deidentified_text != PHI_TEXT
    assert batch_result.successful_items == 1
    assert extract_response.status_code == 200
    assert deid_response.status_code == 200
    _assert_no_phi_substrings(_render_logs(caplog.records))
