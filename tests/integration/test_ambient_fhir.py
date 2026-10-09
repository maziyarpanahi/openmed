"""No providers, persistence, sockets or EHR writes in passive ambient export."""

import json
import socket
from pathlib import Path

import pytest

from openmed.clinical.exporters.fhir.ambient import (
    export_ambient_document,
    import_ambient_document,
)


@pytest.mark.integration
def test_passive_export_roundtrip_with_network_and_files_disabled(monkeypatch):
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/clinical/ambient_fhir.json").read_text()
    )
    draft = fixture["draft"]

    def forbidden(*args, **kwargs):
        pytest.fail("Passive exporter attempted I/O")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    result = export_ambient_document(
        draft,
        current_evidence_digest=draft["evidence_digest"],
        recorded_at="2026-01-02T03:04:05Z",
        confirmed_digest=fixture["digest"],
    )
    assert import_ambient_document(result.bundle) == draft
    # No transcript/audio payload exists in the input or output contract.
    rendered = json.dumps(result.bundle)
    assert "attachment" not in rendered
    assert "request" not in rendered
