from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from openmed.integrations.search_ingest_processor import create_app


def _fake_redact(text: str) -> str:
    return (
        text.replace("Jane Roe", "[NAME]")
        .replace("555-0100", "[PHONE]")
        .replace("jane.roe@example.test", "[EMAIL]")
    )


def _fake_result(texts: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        items=[
            SimpleNamespace(
                success=True,
                result=SimpleNamespace(deidentified_text=_fake_redact(text)),
            )
            for text in texts
        ]
    )


def test_http_processor_redacts_nested_fields_and_preserves_envelope(
    caplog,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append({"texts": list(texts), "kwargs": dict(kwargs)})
        return _fake_result(list(texts))

    document = {
        "_index": "synthetic-clinical-documents",
        "_id": "synthetic-001",
        "_routing": "tenant-a",
        "_source": {
            "note": "Patient Jane Roe called 555-0100.",
            "patient": {
                "summary": "Contact jane.roe@example.test for follow-up.",
                "category": "synthetic",
            },
            "status": "open",
        },
        "_ingest": {"timestamp": "2026-07-19T00:00:00Z"},
    }
    original = deepcopy(document)
    app = create_app(process_batch_fn=fake_process_batch)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/process",
            json={
                "document": document,
                "fields": ["note", "patient.summary"],
                "policy": "hipaa_safe_harbor",
            },
        )

    assert response.status_code == 200
    redacted = response.json()
    assert set(redacted) == set(document)
    assert redacted["_index"] == document["_index"]
    assert redacted["_id"] == document["_id"]
    assert redacted["_routing"] == document["_routing"]
    assert redacted["_ingest"] == document["_ingest"]
    assert redacted["_source"]["note"] == "Patient [NAME] called [PHONE]."
    assert redacted["_source"]["patient"]["summary"] == "Contact [EMAIL] for follow-up."
    assert redacted["_source"]["patient"]["category"] == "synthetic"
    assert redacted["_source"]["status"] == "open"
    assert document == original
    assert calls[0]["texts"] == [
        "Patient Jane Roe called 555-0100.",
        "Contact jane.roe@example.test for follow-up.",
    ]
    assert calls[0]["kwargs"]["operation"] == "deidentify"
    assert calls[0]["kwargs"]["policy"] == "hipaa_safe_harbor"
    assert calls[0]["kwargs"]["continue_on_error"] is False
    assert calls[0]["kwargs"]["use_safety_sweep"] is True
    assert "Jane Roe" not in caplog.text
    assert "555-0100" not in caplog.text
    assert "jane.roe@example.test" not in caplog.text


def test_unknown_and_non_string_field_paths_are_skipped_without_error() -> None:
    calls: list[list[str]] = []

    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(list(texts))
        return _fake_result(list(texts))

    document = {
        "_index": "synthetic-clinical-documents",
        "_source": {
            "note": "Patient Jane Roe called.",
            "metadata": {"attempt": 2},
        },
    }
    app = create_app(process_batch_fn=fake_process_batch)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/process",
            json={
                "document": document,
                "fields": ["missing.path", "metadata.attempt", "_source.note"],
                "policy": "strict_no_leak",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "_index": "synthetic-clinical-documents",
        "_source": {
            "note": "Patient [NAME] called.",
            "metadata": {"attempt": 2},
        },
    }
    assert calls == [["Patient Jane Roe called."]]


def test_batch_failures_return_phi_safe_error_and_do_not_log_document(
    caplog,
) -> None:
    synthetic_text = "Patient Jane Roe called 555-0100."

    def failing_process_batch(texts: list[str], **kwargs: Any) -> None:
        raise RuntimeError(f"model failure while processing {texts[0]}")

    app = create_app(process_batch_fn=failing_process_batch)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/process",
            json={
                "document": {"_source": {"note": synthetic_text}},
                "fields": ["note"],
                "policy": "hipaa_safe_harbor",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "failed to redact configured ingest document fields"
    }
    assert synthetic_text not in response.text
    assert "Jane Roe" not in caplog.text
    assert "555-0100" not in caplog.text


def test_pipeline_definition_registers_processor_and_route() -> None:
    root = Path(__file__).resolve().parents[3]
    pipeline = json.loads(
        (root / "examples/search-ingest/pipeline.json").read_text(encoding="utf-8")
    )

    processor = pipeline["processors"][0]
    assert processor["id"] == "openmed-inline-redaction"
    assert processor["type"] == "http"
    assert processor["request"]["url"].endswith("/process")
    assert processor["request"]["body"]["policy"] == "hipaa_safe_harbor"
    assert pipeline["pipeline"]["processors"] == ["openmed-inline-redaction"]
    assert pipeline["routes"][0]["pipeline"] == pipeline["pipeline"]["id"]
