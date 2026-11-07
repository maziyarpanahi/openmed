"""Offline end-to-end admission; all evidence and authorities are synthetic."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from openmed.clinical.brief import BriefRefusal
from openmed.service.app import create_app
from tests.unit.clinical.test_reviewed_local_evidence import reviewed_fixture

pytestmark = pytest.mark.integration


def test_reviewed_local_admission_through_protected_service():
    value, context = reviewed_fixture()
    app = create_app()
    app.state.brief_context_provider = lambda *_: (value, context)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        request = {
            "text": value.original_text,
            "model": "extractive",
            "review_id": "d" * 64,
        }
        response = client.post("/brief", json=request)
        assert response.status_code == 200
        assert response.json()["status"] == "needs_review"
        assert (
            response.json()["metrics"]["reviewed_evidence"] == context.packet.to_dict()
        )
        context.authority.revoked = True
        refused = client.post("/brief", json=request).json()
        assert refused["refusal_reason"] == BriefRefusal.REVIEW_RECEIPT_REVOKED.value
        assert refused["summary"] == ""
        context.authority.revoked = False
        context = replace(context, packet=replace(context.packet, review_receipt=None))
        refused = client.post("/brief", json=request).json()
        assert refused["refusal_reason"] == BriefRefusal.REVIEW_RECEIPT_MISSING.value
