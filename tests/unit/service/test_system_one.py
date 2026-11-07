"""Exercise the actual privacy route, admission middleware and PHI-free audit."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import openmed
from openmed.service.app import create_app
from openmed.service.auth import hash_api_key
from openmed.structured.decision import (
    DETERMINISTIC_DECISION_BACKEND,
    DecisionBackendOutput,
)

PRIVATE = "SyntheticAlice"
RAW_ID = "SyntheticAliceCase"
BODY = {
    "state": {"patient": PRIVATE},
    "questions": {
        RAW_ID: {
            "type": "choice",
            "instructions": "Route " + PRIVATE,
            "criteria": {PRIVATE: "Call " + PRIVATE, "later": "Follow up"},
        }
    },
}
HEADERS = {"X-API-Key": "example-synthetic-key"}


def configured(monkeypatch, *, scopes=("decisions:write", "pii:write"), burst=20):
    # Exercise the real service admission and warm pool without installing
    # optional Transformers weights for a fake de-identifier.
    from openmed.service import runtime as service_runtime

    monkeypatch.setattr(
        service_runtime,
        "ModelLoader",
        lambda _config: SimpleNamespace(loaded_models=lambda: {}),
    )
    monkeypatch.setenv("OPENMED_SERVICE_SYSTEM_ONE", "1")
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "1")
    # The default prod profile requires a durable job-ownership digest key.
    monkeypatch.setenv("OPENMED_SERVICE_DIGEST_KEY", "synthetic-digest-key-" + "0" * 32)
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "test",
                    "key_hash": hash_api_key(HEADERS["X-API-Key"]),
                    "principal": "synthetic",
                    "scopes": list(scopes),
                }
            ]
        ),
    )
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", str(burst))
    app = create_app()
    return app


def test_disabled_route_is_documented_and_never_loads_a_model(monkeypatch):
    monkeypatch.delenv("OPENMED_SERVICE_SYSTEM_ONE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_AUTH_ENABLED", raising=False)
    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert "/v1/systemone" in app.openapi()["paths"]
        assert client.post("/v1/systemone", json=BODY).status_code == 404


def test_protected_state_criteria_ids_and_audit(monkeypatch, caplog):
    app = configured(monkeypatch, burst=1)
    captured = []

    class Backend:
        identity = DETERMINISTIC_DECISION_BACKEND.identity

        def score_questions(self, r):
            captured.append(r)
            return {
                q.question_id: DecisionBackendOutput(option_scores=(0.97, 0.03))
                for q in r.as_questions()
            }

        def score(self, r):
            raise AssertionError()

    app.state.decision_backend = Backend()

    def redact(text, **kw):
        assert kw["method"] == "mask" and kw["keep_mapping"] is False
        start = text.find(PRIVATE)
        entities = (
            [SimpleNamespace(start=start, end=start + len(PRIVATE))]
            if start >= 0
            else []
        )
        return SimpleNamespace(
            deidentified_text=text.replace(PRIVATE, "[PERSON]"), pii_entities=entities
        )

    monkeypatch.setattr(openmed, "deidentify", redact)
    caplog.set_level("INFO", logger="openmed.service.systemone.audit")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/v1/systemone", json=BODY).status_code == 401
        r = client.post("/v1/systemone", json=BODY, headers=HEADERS)
        assert r.status_code == 200, r.text
        assert (
            client.post("/v1/systemone", json=BODY, headers=HEADERS).status_code == 429
        )
    out = r.json()
    assert out["answers"][RAW_ID]["choice"] == PRIVATE and len(out["decision_id"]) == 32
    assert len(captured) == 1
    scored = captured[0].to_dict()
    assert PRIVATE not in json.dumps(scored) and "[PERSON]" in scored["input_text"]
    assert PRIVATE not in caplog.text and "[PERSON]" not in caplog.text
    records = [
        rec for rec in caplog.records if rec.name == "openmed.service.systemone.audit"
    ]
    audit = json.loads(records[0].getMessage().removeprefix("systemone "))
    assert audit["entity_count"] == 4 and audit["decision_id"] == out["decision_id"]
    assert (
        len(audit["questions"][0]["question_id"]) == 64
        and audit["questions"][0]["state"] == "success"
    )


@pytest.mark.parametrize(
    "reason", ["failed", "abstained", "residual", "claimed_redaction"]
)
def test_refusals_never_dispatch_or_echo_sensitive_input(monkeypatch, reason):
    app = configured(monkeypatch)
    backend = Mock()
    app.state.decision_backend = backend

    def bad(text, **_):
        if reason == "failed":
            raise RuntimeError(PRIVATE)
        if reason == "abstained":
            return SimpleNamespace(
                deidentified_text=text, pii_entities=[], metadata={"state": "abstained"}
            )
        if reason == "residual":
            return SimpleNamespace(
                deidentified_text="Contact test@example.com", pii_entities=[]
            )
        return SimpleNamespace(
            deidentified_text=text, pii_entities=[SimpleNamespace(start=0, end=1)]
        )

    cleaner = Mock(side_effect=bad)
    monkeypatch.setattr(openmed, "deidentify", cleaner)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        r = client.post("/v1/systemone", json=BODY, headers=HEADERS)
    assert (
        r.status_code == 422
        and r.json()["error"]["code"] == "outbound_redaction_failed"
    )
    assert PRIVATE not in r.text
    cleaner.assert_called()
    assert not backend.mock_calls


def test_scope_and_unredacted_media_are_refused(monkeypatch):
    app = configured(monkeypatch, scopes=("decisions:write",))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert (
            client.post("/v1/systemone", json=BODY, headers=HEADERS).status_code == 403
        )
    app = configured(monkeypatch)
    fail = Mock(side_effect=AssertionError())
    monkeypatch.setattr(openmed, "deidentify", fail)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        r = client.post(
            "/v1/systemone",
            json={**BODY, "images": ["https://invalid.invalid"]},
            headers=HEADERS,
        )
    assert r.status_code in {400, 422} and PRIVATE not in r.text
    fail.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {"state": PRIVATE, "questions": {RAW_ID: []}},
        {**BODY, PRIVATE: "unexpected field"},
    ],
)
@pytest.mark.parametrize("mounted", [False, True])
def test_schema_refusals_never_echo_caller_keys_or_run_models(
    monkeypatch, caplog, body, mounted
):
    app = configured(monkeypatch)
    cleaner = Mock(side_effect=AssertionError("de-identification must not run"))
    backend = Mock()
    monkeypatch.setattr(openmed, "deidentify", cleaner)
    app.state.decision_backend = backend
    caplog.set_level("DEBUG")

    served_app = app
    prefix = ""
    if mounted:
        served_app = FastAPI()
        prefix = "/service"
        served_app.mount(prefix, app)
    with TestClient(served_app, base_url="http://127.0.0.1") as client:
        response = client.post(prefix + "/v1/systemone", json=body, headers=HEADERS)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert PRIVATE not in response.text + caplog.text
    assert RAW_ID not in response.text + caplog.text
    cleaner.assert_not_called()
    assert not backend.mock_calls
