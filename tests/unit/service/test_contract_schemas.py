"""Contract tests for the published service JSON Schemas.

The three schemas cover the async job record returned by ``GET /jobs/{job_id}``
and ``POST /jobs``, the terminal webhook payload, and the service error
envelope. They are registered in ``SCHEMA_NAMES`` so the bundled fingerprint
snapshot guard covers them as well.

Service payloads do not embed a ``schema_version`` field, so the property is
declared for registry consistency but is intentionally not required; the
contract version travels in ``schema-fingerprints.json``.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from openmed.core.schemas import (
    build_schema_snapshot,
    load_all_schemas,
    load_schema,
    load_schema_snapshot,
)
from openmed.service import jobs
from openmed.service.app import _error_response
from openmed.service.logging import _REQUEST_ID
from openmed.service.schemas import (
    DeidentifyJobDocument,
    DeidentifyJobRequest,
    JobWebhookRequest,
)
from openmed.service.webhooks import WebhookDeliveryResult

JOB_RECORD_SCHEMA = "service_job_record"
WEBHOOK_PAYLOAD_SCHEMA = "service_webhook_payload"
ERROR_ENVELOPE_SCHEMA = "service_error_envelope"
CONTRACT_SCHEMAS = (JOB_RECORD_SCHEMA, WEBHOOK_PAYLOAD_SCHEMA, ERROR_ENVELOPE_SCHEMA)

SYNTHETIC_TEXTS = ("synthetic clinical note one", "synthetic clinical note two")
STATUS_URL = "/jobs/synthetic-job"
WEBHOOK = JobWebhookRequest(
    url="https://example.invalid/openmed/hooks/synthetic",
    secret="synthetic-webhook-secret",
)


def _validator(schema_name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(schema_name))


def _recording_sender(delivered: list[dict[str, Any]]) -> Any:
    def sender(
        url: str,
        payload: dict[str, Any],
        *,
        secret: str,
        max_attempts: int,
        backoff_seconds: float,
    ) -> WebhookDeliveryResult:
        delivered.append(payload)
        return WebhookDeliveryResult(
            success=True,
            attempts=1,
            status_code=200,
            error=None,
        )

    return sender


def _synthetic_job_payload(webhook: JobWebhookRequest | None) -> DeidentifyJobRequest:
    return DeidentifyJobRequest(
        documents=[
            DeidentifyJobDocument(id="synthetic-0", text=SYNTHETIC_TEXTS[0]),
            DeidentifyJobDocument(id="synthetic-1", text=SYNTHETIC_TEXTS[1]),
        ],
        webhook=webhook,
    )


def _run_synthetic_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_second_document: bool = False,
    webhook: JobWebhookRequest | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Run one synthetic job through the real queue and return its payloads."""
    store = jobs.LocalJobStore(tmp_path / "jobs.json")
    delivered: list[dict[str, Any]] = []
    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(),
        store=store,
        webhook_sender=_recording_sender(delivered),
    )
    payload = _synthetic_job_payload(webhook)
    queued = queue._new_record(payload)
    store.create(queued)
    processed = {"count": 0}

    def process(
        _payload: DeidentifyJobRequest,
        _document: DeidentifyJobDocument,
    ) -> Any:
        processed["count"] += 1
        if fail_second_document and processed["count"] == 2:
            raise ValueError("synthetic document failure")
        return SimpleNamespace(
            pii_entities=[
                SimpleNamespace(
                    canonical_label="PERSON",
                    label="PERSON",
                    text="synthetic entity",
                    start=0,
                    end=16,
                    confidence=0.9,
                )
            ]
        )

    monkeypatch.setattr(queue, "_deidentify_document", process)
    try:
        queue._run_job(jobs._JobWorkItem(queued["id"], payload))
    finally:
        queue.shutdown()

    final = store.get(queued["id"])
    assert final is not None
    return queued, final, delivered


def test_contract_schemas_are_registered_and_versioned() -> None:
    for name in CONTRACT_SCHEMAS:
        schema = load_schema(name)

        assert schema["schema_version"] == 1
        assert schema["properties"]["schema_version"]["const"] == 1
        assert schema["additionalProperties"] is False
        assert "schema_version" not in schema["required"]


def test_contract_schemas_match_the_committed_snapshot() -> None:
    snapshot = load_schema_snapshot()
    current = build_schema_snapshot(load_all_schemas())

    for name in CONTRACT_SCHEMAS:
        assert snapshot[name] == current[name]
        assert current[name]["schema_version"] == 1


def test_job_record_schema_validates_queued_and_terminal_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queued, final, _ = _run_synthetic_job(tmp_path, monkeypatch)
    validator = _validator(JOB_RECORD_SCHEMA)

    validator.validate(jobs.job_response_payload(queued, status_url=STATUS_URL))
    validator.validate(jobs.job_response_payload(final, status_url=STATUS_URL))

    assert queued["status"] == "queued"
    assert final["status"] == "done"


def test_job_record_schema_validates_failed_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, final, _ = _run_synthetic_job(tmp_path, monkeypatch, fail_second_document=True)

    _validator(JOB_RECORD_SCHEMA).validate(
        jobs.job_response_payload(final, status_url=STATUS_URL)
    )

    assert final["status"] == "failed"
    assert final["error"]["type"] == "ValueError"


@pytest.mark.parametrize("field", ["status", "expires_at", "webhook_delivery"])
def test_job_record_schema_rejects_a_removed_required_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    _, final, _ = _run_synthetic_job(tmp_path, monkeypatch)
    payload = jobs.job_response_payload(final, status_url=STATUS_URL)
    damaged = copy.deepcopy(payload)
    del damaged[field]

    _validator(JOB_RECORD_SCHEMA).validate(payload)
    with pytest.raises(ValidationError):
        _validator(JOB_RECORD_SCHEMA).validate(damaged)


def test_job_record_schema_rejects_an_undeclared_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, final, _ = _run_synthetic_job(tmp_path, monkeypatch)
    payload = jobs.job_response_payload(final, status_url=STATUS_URL)
    payload["redacted_text"] = "synthetic output"

    with pytest.raises(ValidationError):
        _validator(JOB_RECORD_SCHEMA).validate(payload)


@pytest.mark.parametrize("fail_second_document", [False, True])
def test_webhook_payload_schema_validates_terminal_deliveries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_second_document: bool,
) -> None:
    _, final, delivered = _run_synthetic_job(
        tmp_path,
        monkeypatch,
        fail_second_document=fail_second_document,
        webhook=WEBHOOK,
    )

    assert len(delivered) == 1
    payload = delivered[0]
    _validator(WEBHOOK_PAYLOAD_SCHEMA).validate(payload)

    assert payload["event"] == f"job.{final['status']}"
    assert payload["status"] == final["status"]
    assert "status_url" not in payload


def test_webhook_payload_schema_rejects_an_undeclared_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, delivered = _run_synthetic_job(tmp_path, monkeypatch, webhook=WEBHOOK)
    payload = copy.deepcopy(delivered[0])
    payload["documents"][0]["text"] = SYNTHETIC_TEXTS[0]

    with pytest.raises(ValidationError):
        _validator(WEBHOOK_PAYLOAD_SCHEMA).validate(payload)


def test_error_envelope_schema_validates_service_errors() -> None:
    validator = _validator(ERROR_ENVELOPE_SCHEMA)

    validation_body = json.loads(
        _error_response(
            422,
            "validation_error",
            "Request validation failed",
            details={"documents": ["text exceeds the configured limit"]},
        ).body
    )
    validator.validate(validation_body)

    internal_body = json.loads(
        _error_response(
            500,
            "runtime_error",
            "Internal service error",
            details=None,
        ).body
    )
    validator.validate(internal_body)

    assert internal_body["error"]["details"] is None
    assert "request_id" not in internal_body["error"]


def test_error_envelope_schema_validates_request_scoped_errors() -> None:
    token = _REQUEST_ID.set("synthetic-request-id")
    try:
        body = json.loads(_error_response(404, "not_found", "job not found").body)
    finally:
        _REQUEST_ID.reset(token)

    _validator(ERROR_ENVELOPE_SCHEMA).validate(body)

    assert body["error"]["request_id"] == "synthetic-request-id"


def test_error_envelope_schema_rejects_drift() -> None:
    validator = _validator(ERROR_ENVELOPE_SCHEMA)
    body = json.loads(_error_response(400, "input_error", "Invalid input").body)

    without_details = copy.deepcopy(body)
    del without_details["error"]["details"]
    with pytest.raises(ValidationError):
        validator.validate(without_details)

    with_extra_field = copy.deepcopy(body)
    with_extra_field["error"]["status_code"] = 400
    with pytest.raises(ValidationError):
        validator.validate(with_extra_field)


def test_contract_samples_carry_hashes_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queued, final, delivered = _run_synthetic_job(
        tmp_path,
        monkeypatch,
        webhook=WEBHOOK,
    )
    samples = [
        jobs.job_response_payload(queued, status_url=STATUS_URL),
        jobs.job_response_payload(final, status_url=STATUS_URL),
        delivered[0],
        json.loads(_error_response(400, "input_error", "Invalid input").body),
    ]

    assert all(text.startswith("synthetic ") for text in SYNTHETIC_TEXTS)
    for sample in samples:
        rendered = json.dumps(sample, sort_keys=True)
        for text in SYNTHETIC_TEXTS:
            assert text not in rendered
        assert re.search(r"\b\d{3}-\d{2}-\d{4}\b", rendered) is None

    hashes = [document["text_hash"] for document in final["documents"]]
    assert hashes
    assert all(
        re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None for value in hashes
    )
