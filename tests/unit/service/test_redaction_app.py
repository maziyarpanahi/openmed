"""Focused tests for the local self-hosted redaction service."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openmed.service import redaction_app as redaction_app_module
from openmed.service.redaction_app import (
    RedactionResult,
    RedactionService,
    RedactionServiceError,
    create_app,
)

SYNTHETIC_TEXT = (
    "Patient Synthetic Person, MRN DEMO-001, DOB 02/03/1979, "
    "phone 425-555-0199, email synthetic@example.test."
)
SYNTHETIC_SECRET = "synthetic-sensitive-label-0042"
LOOPBACK_BASE_URL = "http://127.0.0.1"


def test_text_redaction_is_deterministic_and_reports_aggregate_counts_only() -> None:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
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

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
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
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post("/redact/text", json={"text": SYNTHETIC_TEXT})

    assert response.status_code == 200
    assert response.json()["summary"]["counts"]["total_entities"] == 5


def test_injected_redactor_is_normalized_without_echoing_failed_content() -> None:
    def failing_redactor(_text: str, **_kwargs):
        raise RuntimeError("synthetic detector failure")

    with TestClient(
        create_app(redactor=failing_redactor),
        base_url=LOOPBACK_BASE_URL,
    ) as client:
        response = client.post(
            "/redact/text",
            json={"text": "Synthetic Secret Marker"},
        )
        review = client.get("/review")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
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

    with TestClient(
        create_app(redactor=fake_redactor),
        base_url=LOOPBACK_BASE_URL,
    ) as client:
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

    with TestClient(
        create_app(redactor=fake_redactor),
        base_url=LOOPBACK_BASE_URL,
    ) as client:
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

    with TestClient(
        create_app(redactor=FalseyRedactor()),
        base_url=LOOPBACK_BASE_URL,
    ) as client:
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

    with TestClient(
        create_app(redactor=fake_redactor),
        base_url=LOOPBACK_BASE_URL,
    ) as client:
        response = client.post("/redact/text", json={"text": SYNTHETIC_TEXT})

    assert response.status_code == 200
    assert response.json()["redacted_text"] == "Patient [NAME]"
    assert response.json()["summary"]["counts"]["by_label"] == {"NAME": 1}


def test_file_reads_are_bounded_before_decoding(tmp_path: Path) -> None:
    input_path = tmp_path / "oversized.txt"
    output_path = tmp_path / "redacted.txt"
    input_path.write_bytes(b"x" * 17)

    with TestClient(
        create_app(max_input_characters=4),
        base_url=LOOPBACK_BASE_URL,
    ) as client:
        response = client.post(
            "/redact/file",
            json={
                "input_path": str(input_path),
                "output_path": str(output_path),
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "budget_exceeded"
    assert not output_path.exists()


def test_non_loopback_host_is_rejected_before_routes_run() -> None:
    with TestClient(
        create_app(),
        base_url="http://attacker.example.com",
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/health")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_request_body_limit_runs_before_json_parsing() -> None:
    encoded_secret = (SYNTHETIC_SECRET * 8).encode("utf-8")
    with TestClient(
        create_app(max_request_body_bytes=64),
        base_url=LOOPBACK_BASE_URL,
    ) as client:
        response = client.post(
            "/redact/text",
            content=b'{"text":"' + encoded_secret + b'"}',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "budget_exceeded"
    assert SYNTHETIC_SECRET not in response.text


@pytest.mark.parametrize("seed", [True, False, "1", 1.5])
def test_seed_rejects_coercion(seed: object) -> None:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post(
            "/redact/text",
            json={"text": SYNTHETIC_TEXT, "seed": seed},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert SYNTHETIC_TEXT not in response.text


def test_seed_rejects_values_outside_the_portable_integer_range() -> None:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post(
            "/redact/text",
            json={"text": SYNTHETIC_TEXT, "seed": 2**63},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_validation_details_never_echo_unknown_field_names() -> None:
    sensitive_field_name = "synthetic_patient_name_0042"
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post(
            "/redact/text",
            json={"text": SYNTHETIC_TEXT, sensitive_field_name: "ignored"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["details"] == [
        {"field": "body", "message": "invalid value"}
    ]
    assert sensitive_field_name not in response.text


def test_policy_is_validated_and_aliases_are_reported_canonically() -> None:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        accepted = client.post(
            "/redact/text",
            json={"text": SYNTHETIC_TEXT, "policy": "gdpr"},
        )
        rejected = client.post(
            "/redact/text",
            json={"text": SYNTHETIC_TEXT, "policy": "invented_policy"},
        )

    assert accepted.status_code == 200
    assert accepted.json()["summary"]["policy"] == "gdpr_pseudonymization"
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "validation_error"
    assert SYNTHETIC_TEXT not in rejected.text


def test_local_redaction_methods_have_distinct_deterministic_semantics() -> None:
    text = "DOB 01/15/2000, email synthetic@example.test"
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        replaced = client.post(
            "/redact/text",
            json={"text": text, "method": "replace"},
        )
        preserved = client.post(
            "/redact/text",
            json={"text": text, "method": "format_preserve"},
        )
        shifted_first = client.post(
            "/redact/text",
            json={"text": text, "method": "shift_dates", "seed": 42},
        )
        shifted_second = client.post(
            "/redact/text",
            json={"text": text, "method": "shift_dates", "seed": 42},
        )

    assert replaced.json()["redacted_text"] == "DOB [REDACTED], email [REDACTED]"
    assert preserved.json()["redacted_text"] == (
        "DOB 00/00/0000, email xxxxxxxxx@xxxxxxx.xxxx"
    )
    shifted_text = shifted_first.json()["redacted_text"]
    assert shifted_text == shifted_second.json()["redacted_text"]
    assert "01/15/2000" not in shifted_text
    assert "synthetic@example.test" not in shifted_text
    assert shifted_text.endswith("email [EMAIL]")


def test_format_preserving_placeholder_masks_uncased_unicode_letters() -> None:
    source = "患者-१२-مريض"

    replacement = redaction_app_module._format_preserving_placeholder(source)

    assert replacement == "xx-00-xxxx"
    assert not any(
        character in replacement for character in source if character.isalpha()
    )


def test_older_operation_cannot_overwrite_newer_review_state() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    failures: list[str] = []

    def controlled_redactor(text: str, **_kwargs) -> RedactionResult:
        if text == "first":
            first_started.set()
            assert release_first.wait(timeout=5)
            raise RuntimeError(SYNTHETIC_SECRET)
        return RedactionResult(
            redacted_text="[NAME]",
            entity_counts={"NAME": 1},
        )

    service = RedactionService(controlled_redactor)

    def run_first() -> None:
        try:
            service.process_text("first")
        except RedactionServiceError as exc:
            failures.append(exc.code)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert first_started.wait(timeout=5)
    service.process_text("second")
    release_first.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == ["redaction_failed"]
    assert service.snapshot()["status"] == "completed"
    assert service.snapshot()["counts"] == {
        "total_entities": 1,
        "by_label": {"NAME": 1},
    }
