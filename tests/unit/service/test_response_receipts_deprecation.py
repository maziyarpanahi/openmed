"""Exercise migration headers through the normal service paths."""

from fastapi.testclient import TestClient

from openmed.service.app import create_app
from openmed.structured.decision import DecisionRequest, decide
from tests.unit.service.test_decision_api import _payload


def test_receipt_notice_on_legacy_successes_only(monkeypatch):
    import importlib

    module = importlib.import_module("openmed.service.app")
    monkeypatch.setenv("OPENMED_SERVICE_SECURITY_PROFILE", "dev")

    async def fake_deidentified(request, endpoint, payload, operation, **kwargs):
        return {"deidentified_text": "synthetic", "entities": []}

    monkeypatch.setattr(module, "_run_maybe_coalesced", fake_deidentified)
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        body = _payload()
        decision = client.post("/v1/decisions", json=body)
        pii = client.post("/pii/deidentify", json={"text": "synthetic"})
        assert decision.status_code == pii.status_code == 200
        assert decision.json() == decide(DecisionRequest.from_dict(body)).to_dict()
        assert pii.json() == {"deidentified_text": "synthetic", "entities": []}
        for response in (pii, decision):
            assert response.headers["deprecation"].startswith("@")
            assert int(response.headers["deprecation"][1:]) > 0
            assert '#policy-receipts>; rel="deprecation"' in response.headers["link"]
            assert "preference-applied" not in response.headers
        assert "deprecation" not in client.get("/health").headers
        assert "deprecation" not in client.post("/pii/deidentify", json={}).headers
