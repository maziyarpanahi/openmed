"""Tests for the privacy gateway redaction proxy."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.privacy_gateway import (
    InMemoryReidentificationStore,
    PrivacyGateway,
    PrivacyGatewayAuditTrail,
    PrivacyGatewayEntity,
    PrivacyGatewayPolicy,
    PrivacyPolicyViolation,
    PrivacyReidentificationError,
    PrivacyTripwireViolation,
    build_placeholder_token,
    redact_text,
    reidentify_placeholders,
)


class FakeLoader:
    """Minimal service loader double for gateway endpoint tests."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _span(text: str, value: str) -> tuple[int, int]:
    start = text.index(value)
    return start, start + len(value)


def _entity(label: str, text: str, value: str, confidence: float = 0.99) -> dict:
    start, end = _span(text, value)
    return {
        "label": label,
        "start": start,
        "end": end,
        "confidence": confidence,
    }


def test_privacy_gateway_round_trip_redacts_before_transport_and_reidentifies():
    note = "Patient Maria Garcia called 555-0100."
    seen_prompts: list[str] = []

    def detector(text: str, **_: Any) -> list[dict[str, Any]]:
        return [
            _entity("NAME", text, "Maria Garcia"),
            _entity("PHONE", text, "555-0100"),
        ]

    def transport(redacted_text: str, **_: Any) -> str:
        seen_prompts.append(redacted_text)
        return f"Model echoed: {redacted_text}"

    gateway = PrivacyGateway(
        transport=transport,
        extractor=detector,
        tripwire_extractor=lambda *args, **kwargs: [],
    )

    result = gateway.complete(note)

    assert result.reidentified_text == f"Model echoed: {note}"
    assert seen_prompts
    assert "Maria Garcia" not in seen_prompts[0]
    assert "555-0100" not in seen_prompts[0]
    assert "<<OPENMED_PHI_NAME_" in seen_prompts[0]
    assert result.entity_counts == {"NAME": 1, "PHONE": 1}
    assert result.audit_verified is True


def test_privacy_gateway_low_confidence_fails_closed_and_forwards_nothing():
    note = "Patient Maria Garcia called."
    transport_calls = 0
    audit_trail = PrivacyGatewayAuditTrail()

    def detector(text: str, **_: Any) -> list[dict[str, Any]]:
        return [_entity("NAME", text, "Maria Garcia", confidence=0.42)]

    def transport(redacted_text: str, **_: Any) -> str:
        nonlocal transport_calls
        transport_calls += 1
        return redacted_text

    gateway = PrivacyGateway(
        transport=transport,
        extractor=detector,
        tripwire_extractor=lambda *args, **kwargs: [],
        audit_trail=audit_trail,
    )

    with pytest.raises(PrivacyPolicyViolation):
        gateway.complete(note, policy=PrivacyGatewayPolicy(min_confidence=0.9))

    assert transport_calls == 0
    assert audit_trail.records[-1]["status"] == "blocked"
    assert audit_trail.records[-1]["reason_code"] == (
        "redaction_confidence_below_threshold"
    )


def test_placeholder_tokens_are_collision_free_across_100k_entities():
    entity_count = 100_000
    text = " ".join("x" for _ in range(entity_count))
    entities = [
        PrivacyGatewayEntity(
            label="NAME",
            start=index * 2,
            end=(index * 2) + 1,
            confidence=0.99,
        )
        for index in range(entity_count)
    ]

    session = redact_text(text, entities, request_id="stress-request")

    assert len(session.placeholder_map) == entity_count
    assert len(set(session.placeholder_map)) == entity_count
    assert len(set(session.placeholder_hashes)) == entity_count


def test_reidentifier_rejects_unknown_or_mangled_placeholders():
    known = build_placeholder_token("NAME", request_id="req-1", index=1)

    with pytest.raises(PrivacyReidentificationError):
        reidentify_placeholders(
            "Hello <<OPENMED_PHI_NAME_DEADBEEF_000001>>",
            {known: "Maria Garcia"},
        )

    with pytest.raises(PrivacyReidentificationError):
        reidentify_placeholders(
            "Hello OPENMED_PHI_NAME_DEADBEEF_000001",
            {known: "Maria Garcia"},
        )


def test_audit_trail_is_hash_chained_tamper_evident_and_phi_free():
    note = "Patient Maria Garcia called."
    audit_trail = PrivacyGatewayAuditTrail()

    def detector(text: str, **_: Any) -> list[dict[str, Any]]:
        return [_entity("NAME", text, "Maria Garcia")]

    gateway = PrivacyGateway(
        transport=lambda redacted_text, **_: f"Echo {redacted_text}",
        extractor=detector,
        tripwire_extractor=lambda *args, **kwargs: [],
        audit_trail=audit_trail,
    )

    result = gateway.complete(note)

    assert audit_trail.verify() is True
    tampered = [dict(record) for record in audit_trail.records]
    tampered[0]["entity_counts"] = {"NAME": 99}
    assert PrivacyGatewayAuditTrail.verify_records(tampered) is False
    assert (
        audit_trail.contains_plaintext(
            [note, "Maria Garcia", result.external_response, result.reidentified_text]
        )
        is False
    )


def test_outbound_tripwire_blocks_residual_phi_before_transport():
    note = "Call 555-0100."
    transport_calls = 0

    def tripwire(text: str, **_: Any) -> list[dict[str, Any]]:
        return [_entity("PHONE", text, "555-0100")]

    def transport(redacted_text: str, **_: Any) -> str:
        nonlocal transport_calls
        transport_calls += 1
        return redacted_text

    gateway = PrivacyGateway(
        transport=transport,
        extractor=lambda *args, **kwargs: [],
        tripwire_extractor=tripwire,
    )

    with pytest.raises(PrivacyTripwireViolation):
        gateway.complete(note)

    assert transport_calls == 0


def test_privacy_gateway_service_route_uses_configured_transport(
    monkeypatch: pytest.MonkeyPatch,
):
    note = "Patient Maria Garcia called."
    seen_prompts: list[str] = []

    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)

    def fake_extract(text: str, **_: Any) -> PredictionResult:
        start, end = _span(text, "Maria Garcia")
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="Maria Garcia",
                    label="NAME",
                    confidence=0.99,
                    start=start,
                    end=end,
                )
            ],
            model_name="fixture-pii-model",
            timestamp=datetime.now().isoformat(),
        )

    def transport(redacted_text: str, **_: Any) -> str:
        seen_prompts.append(redacted_text)
        return f"Route echo: {redacted_text}"

    monkeypatch.setattr(openmed, "extract_pii", fake_extract)
    app = create_app()
    app.state.privacy_gateway_transport = transport
    app.state.privacy_gateway_audit_trail = PrivacyGatewayAuditTrail()
    app.state.privacy_gateway_store = InMemoryReidentificationStore()

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/privacy-gateway/complete",
            json={"text": note, "confidence_threshold": 0.9},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reidentified_text"] == f"Route echo: {note}"
    assert "Maria Garcia" not in seen_prompts[0]
