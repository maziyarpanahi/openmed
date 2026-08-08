"""Focused tests for the local self-hosted redaction service."""

from __future__ import annotations

import socket
from pathlib import Path

from fastapi.testclient import TestClient

from openmed.service.redaction_app import RedactionResult, create_app

SYNTHETIC_TEXT = (
    "Patient Synthetic Person, MRN DEMO-001, DOB 02/03/1979, "
    "phone 425-555-0199, email synthetic@example.test."
)


def test_text_redaction_is_deterministic_and_reports_aggregate_counts_only() -> None:
    with TestClient(create_app()) as client:
        first = client.post(
            "/redact/text",
            json={"text": SYNTHETIC_TEXT, "policy": "strict_no_leak"},
        )
        second = client.post(
            "/redact/text",
            json={"text": SYNTHETIC_TEXT, "policy": "strict_no_leak"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload == second_payload
    assert first_payload["redacted_text"] == (
        "Patient [NAME], MRN [ID], DOB [DATE], phone [PHONE], email [EMAIL]."
    )
    assert first_payload["summary"]["counts"] == {
        "total_entities": 5,
        "by_label": {
            "DATE": 1,
            "EMAIL": 1,
            "ID": 1,
            "NAME": 1,
            "PHONE": 1,
        },
    }
    assert first_payload["artifact"] == {
        "status": "returned",
        "kind": "text",
        "sha256": None,
    }


def test_file_redaction_writes_explicit_artifact_and_review_page_is_content_free(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "synthetic-input.txt"
    output_path = tmp_path / "artifacts" / "synthetic-output.txt"
    input_path.write_text(SYNTHETIC_TEXT, encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.post(
            "/redact/file",
            json={
                "input_path": str(input_path),
                "output_path": str(output_path),
            },
        )
        review = client.get("/review")
        status = client.get("/status")

    assert response.status_code == 200
    assert output_path.read_text(encoding="utf-8") == (
        "Patient [NAME], MRN [ID], DOB [DATE], phone [PHONE], email [EMAIL]."
    )
    assert "Synthetic Person" not in response.text
    assert "synthetic@example.test" not in response.text
    assert "Synthetic Person" not in review.text
    assert "synthetic@example.test" not in review.text
    assert status.json()["artifact"]["status"] == "written"
    assert status.json()["counts"]["total_entities"] == 5


def test_default_service_does_not_require_network(monkeypatch) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("network access is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)
    with TestClient(create_app()) as client:
        response = client.post("/redact/text", json={"text": SYNTHETIC_TEXT})

    assert response.status_code == 200
    assert response.json()["summary"]["counts"]["total_entities"] == 5


def test_injected_redactor_is_normalized_without_echoing_failed_content() -> None:
    def failing_redactor(_text: str, **_kwargs):
        raise RuntimeError("synthetic detector failure")

    with TestClient(create_app(redactor=failing_redactor)) as client:
        response = client.post(
            "/redact/text",
            json={"text": "Synthetic Secret Marker"},
        )
        review = client.get("/review")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "redaction_failed",
            "message": "Redaction failed",
            "details": None,
        }
    }
    assert "Synthetic Secret Marker" not in response.text
    assert "Synthetic Secret Marker" not in review.text


def test_injected_result_keeps_only_redacted_text_and_label_counts() -> None:
    def fake_redactor(text: str, **_kwargs) -> RedactionResult:
        return RedactionResult(
            redacted_text=text.replace("Synthetic Person", "[NAME]"),
            entity_counts={"NAME": 1},
        )

    with TestClient(create_app(redactor=fake_redactor)) as client:
        response = client.post(
            "/redact/text",
            json={"text": "Patient Synthetic Person"},
        )

    assert response.status_code == 200
    assert response.json()["redacted_text"] == "Patient [NAME]"
    assert response.json()["summary"]["counts"] == {
        "total_entities": 1,
        "by_label": {"NAME": 1},
    }
