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
SYNTHETIC_SECRET = "synthetic-sensitive-label-0042"


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
    original_connect = socket.socket.connect

    def fail_external_connect(sock, address):
        # Windows implements ``socketpair()`` with a loopback connection when
        # AnyIO starts TestClient's event loop.  That is test-harness plumbing,
        # not application network access.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        raise AssertionError("network access is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", fail_external_connect)
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


def test_unknown_injected_labels_are_collapsed_without_exposing_values() -> None:
    def fake_redactor(_text: str, **_kwargs):
        return {
            "redacted_text": "Patient [UNKNOWN]",
            "entity_counts": {SYNTHETIC_SECRET: 2},
        }

    with TestClient(create_app(redactor=fake_redactor)) as client:
        response = client.post("/redact/text", json={"text": SYNTHETIC_TEXT})
        review = client.get("/review")
        status = client.get("/status")

    assert response.status_code == 200
    assert response.json()["summary"]["counts"] == {
        "total_entities": 2,
        "by_label": {"UNKNOWN": 2},
    }
    assert SYNTHETIC_SECRET not in response.text
    assert SYNTHETIC_SECRET not in review.text
    assert SYNTHETIC_SECRET not in status.text


def test_result_repr_and_hostile_count_mappings_are_content_free() -> None:
    class FailingCounts(dict):
        def items(self):
            raise RuntimeError(SYNTHETIC_SECRET)

    result = RedactionResult(
        redacted_text=SYNTHETIC_SECRET,
        entity_counts=FailingCounts({"NAME": 1}),
        input_characters=len(SYNTHETIC_SECRET),
    )

    assert result.entity_counts == {}
    assert SYNTHETIC_SECRET not in repr(result)
    assert "output_characters" in repr(result)


def test_falsey_callable_redactor_is_not_replaced_by_the_default() -> None:
    class FalseyRedactor:
        def __bool__(self) -> bool:
            return False

        def __call__(self, text: str, **_kwargs) -> RedactionResult:
            return RedactionResult(
                redacted_text=text.replace("Synthetic Person", "[CUSTOM]"),
                entity_counts={"NAME": 1},
            )

    with TestClient(create_app(redactor=FalseyRedactor())) as client:
        response = client.post(
            "/redact/text",
            json={"text": "Patient Synthetic Person"},
        )

    assert response.status_code == 200
    assert response.json()["redacted_text"] == "Patient [CUSTOM]"


def test_string_subclass_hooks_are_not_used_at_result_boundaries() -> None:
    class HostileText(str):
        def __str__(self) -> str:
            raise AssertionError(SYNTHETIC_SECRET)

        def strip(self, *_args, **_kwargs):
            raise AssertionError(SYNTHETIC_SECRET)

        def upper(self):
            raise AssertionError(SYNTHETIC_SECRET)

        def encode(self, *_args, **_kwargs):
            raise AssertionError(SYNTHETIC_SECRET)

    def fake_redactor(_text: str, **_kwargs) -> RedactionResult:
        return RedactionResult(
            redacted_text=HostileText("Patient [NAME]"),
            entity_counts={HostileText("NAME"): 1},
        )

    with TestClient(create_app(redactor=fake_redactor)) as client:
        response = client.post("/redact/text", json={"text": SYNTHETIC_TEXT})

    assert response.status_code == 200
    assert response.json()["redacted_text"] == "Patient [NAME]"
    assert response.json()["summary"]["counts"]["by_label"] == {"NAME": 1}


def test_file_reads_are_bounded_before_decoding(tmp_path: Path) -> None:
    input_path = tmp_path / "oversized.txt"
    output_path = tmp_path / "redacted.txt"
    input_path.write_bytes(b"x" * 17)

    with TestClient(create_app(max_input_characters=4)) as client:
        response = client.post(
            "/redact/file",
            json={
                "input_path": str(input_path),
                "output_path": str(output_path),
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "input_too_large"
    assert not output_path.exists()
