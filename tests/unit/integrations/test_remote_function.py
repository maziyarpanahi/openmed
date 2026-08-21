from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openmed.integrations.remote_function import (
    DEFAULT_REMOTE_FUNCTION_MODEL,
    POLICY_HEADER,
    create_app,
)
from openmed.processing.batch import BatchItemResult, BatchResult

ROOT = Path(__file__).resolve().parents[3]


def _redact_synthetic(text: str) -> str:
    replacements = {
        "Jane Roe": "[PERSON]",
        "john.roe@example.test": "[EMAIL]",
        "555-0101": "[PHONE]",
        "MRN-7788": "[MEDICAL_RECORD]",
    }
    redacted = text
    for identifier, replacement in replacements.items():
        redacted = redacted.replace(identifier, replacement)
    return redacted


def _result_for(texts: list[str]) -> BatchResult:
    return BatchResult(
        items=[
            BatchItemResult(
                id=f"synthetic-{index}",
                result=SimpleNamespace(
                    deidentified_text=_redact_synthetic(text),
                ),
            )
            for index, text in enumerate(texts)
        ]
    )


def test_remote_function_redacts_synthetic_batch_in_one_vector_call(caplog) -> None:
    process_calls: list[dict[str, Any]] = []
    caplog.set_level(logging.DEBUG)

    def fake_process_batch(texts: list[str], **kwargs: Any) -> BatchResult:
        process_calls.append({"texts": list(texts), **kwargs})
        return _result_for(list(texts))

    payload = {
        "requestId": "synthetic-request-001",
        "caller": "//bigquery.googleapis.com/projects/example/jobs/synthetic-job",
        "sessionUser": "warehouse.user@example.test",
        "calls": [
            [
                "Jane Roe called 555-0101 about MRN-7788.",
                "hipaa_safe_harbor",
            ],
            [
                "Email john.roe@example.test for follow-up.",
                "hipaa_safe_harbor",
            ],
            [None, "hipaa_safe_harbor"],
            ["", "hipaa_safe_harbor"],
        ],
    }
    app = create_app(process_batch_fn=fake_process_batch)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "replies": [
            "[PERSON] called [PHONE] about [MEDICAL_RECORD].",
            "Email [EMAIL] for follow-up.",
            None,
            "",
        ]
    }
    assert len(process_calls) == 1
    assert process_calls[0]["texts"] == [
        "Jane Roe called 555-0101 about MRN-7788.",
        "Email john.roe@example.test for follow-up.",
    ]
    assert process_calls[0]["ids"] == ["remote:0", "remote:1"]
    assert process_calls[0]["model_name"] == DEFAULT_REMOTE_FUNCTION_MODEL
    assert process_calls[0]["operation"] == "deidentify"
    assert process_calls[0]["batch_size"] == 2
    assert process_calls[0]["method"] == "mask"
    assert process_calls[0]["continue_on_error"] is False
    assert process_calls[0]["use_safety_sweep"] is True
    assert process_calls[0]["policy"] == "hipaa_safe_harbor"

    for private_value in (
        "Jane Roe",
        "555-0101",
        "MRN-7788",
        "john.roe@example.test",
        "warehouse.user@example.test",
        "synthetic-request-001",
    ):
        assert private_value not in caplog.text


def test_mixed_row_policies_are_grouped_and_reply_order_is_preserved() -> None:
    process_calls: list[dict[str, Any]] = []

    def fake_process_batch(texts: list[str], **kwargs: Any) -> BatchResult:
        process_calls.append({"texts": list(texts), **kwargs})
        return _result_for(list(texts))

    app = create_app(process_batch_fn=fake_process_batch)
    payload = {
        "calls": [
            ["Jane Roe has asthma.", "hipaa"],
            ["Call 555-0101.", "gdpr"],
            ["MRN-7788 is inactive.", "safe_harbor"],
        ]
    }

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "replies": [
            "[PERSON] has asthma.",
            "Call [PHONE].",
            "[MEDICAL_RECORD] is inactive.",
        ]
    }
    assert [call["policy"] for call in process_calls] == [
        "hipaa_safe_harbor",
        "gdpr_pseudonymization",
    ]
    assert process_calls[0]["texts"] == [
        "Jane Roe has asthma.",
        "MRN-7788 is inactive.",
    ]
    assert process_calls[1]["texts"] == ["Call 555-0101."]


@pytest.mark.parametrize(
    ("request_target", "headers", "context", "expected_policy"),
    [
        ("/?policy=safe_harbor", {}, None, "hipaa_safe_harbor"),
        ("/", {POLICY_HEADER: "gdpr"}, None, "gdpr_pseudonymization"),
        (
            "/",
            {},
            {"policy": "strict-no-leak"},
            "strict_no_leak",
        ),
    ],
)
def test_request_wide_policy_sources_are_supported(
    request_target: str,
    headers: dict[str, str],
    context: dict[str, str] | None,
    expected_policy: str,
) -> None:
    policies: list[str] = []

    def fake_process_batch(texts: list[str], **kwargs: Any) -> BatchResult:
        policies.append(kwargs["policy"])
        return _result_for(list(texts))

    payload: dict[str, Any] = {"calls": [["Jane Roe called."]]}
    if context is not None:
        payload["userDefinedContext"] = context

    app = create_app(process_batch_fn=fake_process_batch)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(request_target, headers=headers, json=payload)

    assert response.status_code == 200
    assert response.json() == {"replies": ["[PERSON] called."]}
    assert policies == [expected_policy]


def test_request_policy_cannot_be_weakened_by_a_row_argument() -> None:
    invoked = False

    def unexpected_process_batch(texts: list[str], **kwargs: Any) -> None:
        nonlocal invoked
        invoked = True

    app = create_app(process_batch_fn=unexpected_process_batch)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/",
            headers={POLICY_HEADER: "strict_no_leak"},
            json={
                "calls": [["Jane Roe called 555-0101.", "clinical_minimal_redaction"]]
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "errorMessage": "calls[0] policy conflicts with the request policy"
    }
    assert invoked is False
    assert "Jane Roe" not in response.text
    assert "555-0101" not in response.text


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "calls must be a non-empty array"),
        ({"calls": []}, "calls must be a non-empty array"),
        ({"calls": "not-an-array"}, "calls must be a non-empty array"),
        ({"calls": [[]]}, "calls[0] must contain text and an optional policy"),
        (
            {"calls": [["one", "hipaa_safe_harbor", "extra"]]},
            "calls[0] must contain text and an optional policy",
        ),
        ({"calls": [[42]]}, "calls[0] text must be a string or null"),
        (
            {"calls": [["synthetic", 42]]},
            "calls[0] policy must be a non-empty string",
        ),
        (
            {"calls": [["synthetic", "not-a-policy"]]},
            "calls[0] policy is not a supported profile",
        ),
        (
            {"calls": [["synthetic"]], "userDefinedContext": []},
            "userDefinedContext must be an object",
        ),
    ],
)
def test_malformed_or_empty_batches_return_protocol_errors(
    payload: Any,
    message: str,
) -> None:
    app = create_app(
        process_batch_fn=lambda *_args, **_kwargs: pytest.fail(
            "invalid requests must not invoke process_batch"
        )
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/", json=payload)

    assert response.status_code == 400
    assert response.json() == {"errorMessage": message}
    assert "Traceback" not in response.text


def test_invalid_json_returns_protocol_error_without_a_stack_trace() -> None:
    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/",
            content=b'{"calls": [["synthetic"]]',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json() == {"errorMessage": "request body must be valid JSON"}
    assert "Traceback" not in response.text


def test_processing_failure_never_exposes_or_logs_input_text(caplog) -> None:
    synthetic_text = "Jane Roe called 555-0101 about MRN-7788."
    caplog.set_level(logging.DEBUG)

    def failing_process_batch(texts: list[str], **kwargs: Any) -> None:
        raise RuntimeError(f"model failed while processing {texts[0]}")

    app = create_app(process_batch_fn=failing_process_batch)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/", json={"calls": [[synthetic_text]]})

    assert response.status_code == 503
    assert response.json() == {
        "errorMessage": "remote-function batch could not be de-identified"
    }
    assert synthetic_text not in response.text
    assert "Jane Roe" not in caplog.text
    assert "555-0101" not in caplog.text
    assert "MRN-7788" not in caplog.text
    assert "Traceback" not in response.text


def test_invalid_batch_result_returns_phi_safe_error() -> None:
    def short_result(texts: list[str], **kwargs: Any) -> BatchResult:
        return BatchResult(items=[])

    app = create_app(process_batch_fn=short_result)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/",
            json={"calls": [["Jane Roe"], ["Call 555-0101"]]},
        )

    assert response.status_code == 503
    assert response.json() == {
        "errorMessage": "remote-function batch could not be de-identified"
    }
    assert "Jane Roe" not in response.text
    assert "555-0101" not in response.text


def test_null_and_empty_rows_do_not_load_the_batch_processor() -> None:
    app = create_app(
        process_batch_fn=lambda *_args, **_kwargs: pytest.fail(
            "null and empty rows must bypass process_batch"
        )
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/",
            json={"calls": [[None], ["", "hipaa_safe_harbor"]]},
        )

    assert response.status_code == 200
    assert response.json() == {"replies": [None, ""]}


def test_health_endpoint_contains_no_runtime_or_model_details() -> None:
    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_example_ships_container_entrypoint_and_bigquery_ddl() -> None:
    example = ROOT / "examples" / "warehouse-remote-function"
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")
    ddl = (example / "create_function.sql").read_text(encoding="utf-8")
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert "openmed.integrations.remote_function:app" in dockerfile
    assert "--no-access-log" in dockerfile
    assert 'pip install --no-cache-dir ".[hf,service]"' in dockerfile
    assert "REMOTE WITH CONNECTION" in ddl
    assert "text STRING" in ddl
    assert "policy STRING" in ddl
    assert "max_batching_rows" in ddl
    assert "ENDPOINT_URL" in ddl
    assert "--no-allow-unauthenticated" in readme
    assert "OPENMED_OFFLINE=1" in readme
    assert "synthetic" in readme.lower()
