"""Offline composer-to-document round trip, with no socket/model/EHR calls."""

import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openmed.clinical.brief import build_clinical_brief
from openmed.clinical.exporters.fhir import export_brief_document, import_brief_document
from tests.unit.clinical.test_brief import fixture_context


@pytest.mark.integration
def test_composer_export_and_roundtrip_are_offline_and_match_native_fixture(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("network transport called")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    value, context = fixture_context()
    brief = build_clinical_brief(value, model="extractive", context=context)
    document = export_brief_document(
        brief,
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        privacy_detector=lambda _: [],
    )
    fixture = json.loads(
        (
            Path(__file__).parents[2] / "fixtures/clinical/brief_parity/verified.json"
        ).read_text()
    )
    assert document.to_response() == fixture["fhir_document"]
    restored = import_brief_document(
        document.to_response(), privacy_detector=lambda _: []
    )
    assert restored.to_response() == document.to_response()
