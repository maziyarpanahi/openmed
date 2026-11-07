"""Unit tests for durable service job recovery contracts (#3574)."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient
from jsonschema.validators import validator_for

import openmed
from openmed.interop.ingest import load_ingestion_schema
from openmed.service import job_recovery, jobs
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.job_recovery import (
    IdempotencyKeyError,
    JobDecisionReceipt,
    JobRecoveryError,
    JobReplayAuthorization,
    ReplayRefusedError,
    RetryPolicy,
    classify_document_failure,
    classify_input_failure,
    idempotency_key_digest,
    public_recovery_view,
    validate_recovery,
)
from openmed.service.resilience import CircuitBreakerOpenError
from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest
from openmed.service.webhooks import WebhookDeliveryResult

START = datetime(2026, 4, 1, tzinfo=timezone.utc)
TEXT_SENTINEL = "ZQUNITRECOVERYSENTINEL Synthetic Patient"
KEY = "6c1f0a2e-8f3b-4d7e-9a10-5b2c3d4e5f60"
LOOPBACK_BASE_URL = "http://127.0.0.1"


class Clock:
    """Mutable deterministic UTC clock."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class ManualExecutor:
    """Executor double that records work without running it."""

    def __init__(self, **_: Any) -> None:
        self.work: list[tuple[Callable[[Any], None], Any]] = []

    def submit(self, operation: Callable[[Any], None], item: Any) -> Future[None]:
        self.work.append((operation, item))
        return Future()

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait, cancel_futures


def _queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock, **kwargs: Any
) -> jobs.DeidentifyJobQueue:
    store = jobs.LocalJobStore(tmp_path / "jobs.json", clock=clock)
    with monkeypatch.context() as patched:
        patched.setattr(jobs, "ThreadPoolExecutor", ManualExecutor)
        queue = jobs.DeidentifyJobQueue(
            SimpleNamespace(validate_served_model=lambda _name: None),
            store=store,
            clock=clock,
            **kwargs,
        )
    queue._deidentify_document = lambda _request, _document: SimpleNamespace(  # type: ignore[method-assign]
        pii_entities=[]
    )
    return queue


def _request(**overrides: Any) -> DeidentifyJobRequest:
    values: dict[str, Any] = {
        "documents": [
            DeidentifyJobDocument(id="synthetic-0", text=TEXT_SENTINEL),
            DeidentifyJobDocument(id="synthetic-1", text="synthetic second note"),
        ]
    }
    values.update(overrides)
    return DeidentifyJobRequest(**values)


def test_retry_policy_backoff_is_capped_and_validated() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay_seconds=5, max_delay_seconds=12)

    assert [policy.delay_seconds(attempt) for attempt in (1, 2, 3, 4)] == [
        5,
        10,
        12,
        12,
    ]
    assert RetryPolicy(base_delay_seconds=0).delay_seconds(3) == 0
    for bad in (
        {"max_attempts": 0},
        {"base_delay_seconds": -1},
        {"max_attempts": True},
    ):
        with pytest.raises(ValueError):
            RetryPolicy(**bad)


@pytest.mark.parametrize(
    ("error", "action", "classification"),
    [
        (TimeoutError("synthetic"), "retry", "transient"),
        (ConnectionError("synthetic"), "retry", "transient"),
        (MemoryError(), "retry", "resource"),
        (OSError("synthetic"), "retry", "dependency"),
        (CircuitBreakerOpenError(5), "retry", "contention"),
        (ValueError("synthetic"), "document", "permanent"),
        (PermissionError("synthetic"), "document", "policy_denied"),
        (RuntimeError("synthetic"), "document", "unknown"),
    ],
)
def test_document_failures_separate_retries_from_document_results(
    error: Exception, action: str, classification: str
) -> None:
    decision = classify_document_failure(error)

    assert (decision.action, decision.classification) == (action, classification)
    assert "synthetic" not in decision.reason_code


@pytest.mark.parametrize(
    ("code", "action", "classification"),
    [
        ("payload_expired", "fail", "permanent"),
        ("payload_deleted", "fail", "permanent"),
        ("payload_missing", "fail", "permanent"),
        ("key_unavailable", "retry", "dependency"),
        ("payload_read_failed", "retry", "dependency"),
        ("key_revoked", "quarantine", "policy_denied"),
        ("envelope_authentication_failed", "quarantine", "malformed"),
        ("payload_damaged", "quarantine", "malformed"),
        ("input_mismatch", "quarantine", "malformed"),
        ("Free Text With Spaces", "quarantine", "malformed"),
    ],
)
def test_input_failures_are_classified_by_controlled_code(
    code: str, action: str, classification: str
) -> None:
    decision = classify_input_failure(code)

    assert (decision.action, decision.classification) == (action, classification)
    assert " " not in decision.reason_code


def test_idempotency_key_digest_is_scoped_and_value_free() -> None:
    anonymous = idempotency_key_digest(KEY)

    assert anonymous.startswith("sha256:") and KEY not in anonymous
    assert idempotency_key_digest(KEY, scope=None) == anonymous
    assert idempotency_key_digest(KEY, scope="api_key:a") != anonymous
    assert idempotency_key_digest(KEY, scope="api_key:a") != idempotency_key_digest(
        KEY, scope="api_key:b"
    )
    for bad in ("short", "x" * 129, "contains space here!!", "", 7):
        with pytest.raises(IdempotencyKeyError) as caught:
            idempotency_key_digest(bad)  # type: ignore[arg-type]
        assert caught.value.code == "idempotency_key_invalid"


def test_replay_authorization_and_receipts_reject_free_text() -> None:
    with pytest.raises(ReplayRefusedError):
        JobReplayAuthorization(
            reviewer_digest="sha256:" + "a" * 64,
            evidence_digest="sha256:" + "b" * 64,
            reason_code="Approved by Dr. Synthetic",
        )
    with pytest.raises(JobRecoveryError):
        JobDecisionReceipt(
            receipt_id="receipt_x",
            job_id="0" * 32,
            action="quarantined",
            reason_code="free text",
            from_status="running",
            to_status="quarantined",
            execution=1,
            attempt=1,
            lease_epoch=1,
            recorded_at="2026-04-01T00:00:00Z",
        )
    with pytest.raises(JobRecoveryError):
        JobDecisionReceipt(
            receipt_id="receipt_x",
            job_id="0" * 32,
            action="deleted_everything",
            reason_code="ok",
            from_status="running",
            to_status="quarantined",
            execution=1,
            attempt=1,
            lease_epoch=1,
            recorded_at="2026-04-01T00:00:00Z",
        )
    assert str(JobRecoveryError("not_a_known_code")) == "job recovery operation failed"


def test_embedded_records_conform_to_the_ingestion_schemas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The service queue reuses the #3323 contracts; validate them as such."""
    clock = Clock()
    queue = _queue(
        monkeypatch,
        tmp_path,
        clock,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=1),
    )
    job_id = queue.submit(_request())["id"]
    claimed = queue._claim(job_id)
    assert claimed is not None
    lease, record = claimed
    lease_record = queue.get(job_id)["recovery"]["lease"]
    summary = jobs._JobSummary.restore(record)
    summary.processed_count = 1
    queue._commit_checkpoint(
        lease, summary, sequence=1, error=None, output_digest="sha256:" + "e" * 64
    )
    queue._release_for_retry(lease, classify_document_failure(TimeoutError()))
    quarantined = queue.get(job_id)
    recovery = quarantined["recovery"]
    assert quarantined["status"] == "quarantined"
    queue.replay(
        job_id,
        JobReplayAuthorization("sha256:" + "a" * 64, "sha256:" + "b" * 64),
    )
    replayed = queue.get(job_id)["recovery"]
    queue.cancel(job_id)
    cancelled = queue.get(job_id)["recovery"]

    samples = {
        "lease": lease_record,
        "checkpoint": recovery["checkpoint"],
        "quarantine_result": recovery["quarantine"],
        "cancellation": cancelled["cancellation"],
    }
    for name, sample in samples.items():
        schema = load_ingestion_schema(name)
        validator = validator_for(schema)
        validator.check_schema(schema)
        assert not tuple(validator(schema).iter_errors(sample)), name
    assert (
        replayed["promotions"][0]["quarantine_id"]
        == (recovery["quarantine"]["quarantine_id"])
    )
    assert validate_recovery(queue.get(job_id)) is not None


def test_retry_records_conform_to_the_ingestion_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    queue = _queue(monkeypatch, tmp_path, clock)
    job_id = queue.submit(_request())["id"]
    claimed = queue._claim(job_id)
    assert claimed is not None
    queue._release_for_retry(claimed[0], classify_document_failure(TimeoutError()))

    retry = queue.get(job_id)["recovery"]["retry"]
    schema = load_ingestion_schema("retry")
    assert not tuple(validator_for(schema)(schema).iter_errors(retry))
    assert retry["retry_after"] == "2026-04-01T00:00:05Z"


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda record: record["recovery"].update(schema_version=2),
        lambda record: record["recovery"].update(request_digest="sha256:short"),
        lambda record: record["recovery"].update(max_attempts=0),
        lambda record: record["recovery"].update(completed_documents=1),
        lambda record: record.update(status="running"),
        lambda record: record.update(status="quarantined"),
        lambda record: record["recovery"].update(transitions="not-a-list"),
        lambda record: record.update(id="short"),
    ],
)
def test_validate_recovery_rejects_corrupt_or_ambiguous_state(
    corrupt: Callable[[dict[str, Any]], Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queue = _queue(monkeypatch, tmp_path, Clock())
    record = queue.submit(_request())
    assert validate_recovery(record) is record["recovery"]

    corrupt(record)

    with pytest.raises(JobRecoveryError) as caught:
        validate_recovery(record)
    assert caught.value.code == "recovery_state_invalid"
    assert TEXT_SENTINEL not in str(caught.value)


def test_public_view_omits_worker_lease_and_request_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    queue = _queue(monkeypatch, tmp_path, Clock())
    job_id = queue.submit(_request(), idempotency_key=KEY)["id"]
    assert queue._claim(job_id) is not None
    record = queue.get(job_id)

    public = jobs.job_response_payload(record, status_url=f"/jobs/{job_id}")
    encoded = json.dumps(public)

    assert public["recovery"]["lease"]["epoch"] == 1
    assert set(public["recovery"]["lease"]) == {"epoch", "acquired_at", "expires_at"}
    assert "idempotency" not in public
    for internal in (
        queue.worker_id,
        record["recovery"]["lease"]["lease_id"],
        record["recovery"]["request_digest"],
        record["idempotency"]["key_digest"],
        KEY,
        TEXT_SENTINEL,
    ):
        assert internal not in encoded
    assert public_recovery_view(None) is None
    assert public_recovery_view({"lease": {"epoch": 1}}) is None


def test_stores_on_one_path_share_transitions_and_receipts(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "jobs.json"
    first = jobs.LocalJobStore(path, clock=clock)
    second = jobs.LocalJobStore(path, clock=clock)
    first.create({"id": "synthetic-a", "status": "queued", "state_version": 1})

    def promote(record: dict[str, Any]) -> list[dict[str, Any]]:
        record["status"] = "running"
        return [{"action": "lease_reclaimed", "reason_code": "lease_expired"}]

    stored = second.transition("synthetic-a", promote)

    assert stored is not None and stored["state_version"] == 2
    assert first.get("synthetic-a")["status"] == "running"
    assert first.decisions() == [
        {"action": "lease_reclaimed", "reason_code": "lease_expired"}
    ]
    assert second.transition("synthetic-a", lambda _record: None) is None
    assert second.transition("missing", promote) is None
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["jobs"]["synthetic-a"]["status"] == "running"
    assert len(on_disk["decisions"]) == 1


def test_failed_transition_leaves_memory_and_disk_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "jobs.json"
    store = jobs.LocalJobStore(path, clock=Clock())
    store.create({"id": "synthetic-a", "status": "queued", "state_version": 1})
    before = path.read_bytes()

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic publication failure")

    def change(record: dict[str, Any]) -> list[dict[str, Any]]:
        record["status"] = "running"
        return [{"action": "quarantined"}]

    with monkeypatch.context() as patched:
        patched.setattr(jobs.os, "replace", fail)
        with pytest.raises(OSError, match="synthetic publication failure"):
            store.transition("synthetic-a", change)

    assert path.read_bytes() == before
    assert store.get("synthetic-a")["status"] == "queued"
    assert store.decisions() == []
    assert sorted(item.name for item in tmp_path.iterdir()) == ["jobs.json"]


def test_store_rejects_damaged_decision_receipts(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text('{"jobs": {}, "decisions": {"bad": 1}}', encoding="utf-8")

    with pytest.raises(ValueError, match="decision receipt"):
        jobs.LocalJobStore(path)


def test_memory_mode_restart_fails_unresumable_work_with_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    sender_calls: list[Any] = []

    def sender(*args: Any, **_kwargs: Any) -> WebhookDeliveryResult:
        sender_calls.append(args)
        return WebhookDeliveryResult(success=True, attempts=1, status_code=204)

    first = _queue(monkeypatch, tmp_path, clock, webhook_sender=sender)
    queued_id = first.submit(_request())["id"]
    pending_id = first.submit(
        _request(
            webhook={
                "url": "https://synthetic.example.test/hook",
                "secret": "synthetic-secret-value-123",
                "backoff_seconds": 0,
            }
        )
    )["id"]
    claimed = first._claim(pending_id)
    assert claimed is not None
    first.fault_hook = _crash_after_terminal_commit
    with pytest.raises(_Crash):
        first._process(*claimed)

    second = _queue(monkeypatch, tmp_path, clock, webhook_sender=sender)
    queued = second.get(queued_id)
    assert queued["status"] == "failed"
    assert queued["error"]["code"] == "input_unavailable"
    assert queued["payloads"]["input"]["reason"] == "process_restarted"
    clock.advance(3600)
    second.reconcile()
    pending = second.get(pending_id)
    assert pending["status"] == "done"
    assert pending["recovery"]["webhook"]["state"] == "unavailable"
    assert pending["webhook_delivery"]["error"] == "terminal_event_unavailable"
    assert sender_calls == []


class _Crash(BaseException):
    """Synthetic process termination."""


def _crash_after_terminal_commit(boundary: str, _job_id: str) -> None:
    if boundary == "terminal.committed":
        raise _Crash(boundary)


def test_environment_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert jobs.parse_jobs_reconcile_seconds(None) == 5.0
    assert jobs.parse_jobs_reconcile_seconds("0") is None
    assert jobs.parse_jobs_reconcile_seconds("1.5") == 1.5
    for bad in ("-1", "nan", "soon"):
        with pytest.raises(ValueError):
            jobs.parse_jobs_reconcile_seconds(bad)
    assert jobs.parse_jobs_lease_seconds("") == job_recovery.DEFAULT_LEASE_SECONDS
    with pytest.raises(ValueError):
        jobs.parse_jobs_lease_seconds("0")
    policy = jobs.parse_jobs_retry_policy(
        {
            "OPENMED_SERVICE_JOBS_MAX_ATTEMPTS": "5",
            "OPENMED_SERVICE_JOBS_RETRY_BASE_SECONDS": "0",
            "OPENMED_SERVICE_JOBS_RETRY_MAX_SECONDS": "9",
        }
    )
    assert policy == RetryPolicy(5, 0, 9)
    with pytest.raises(ValueError):
        jobs.parse_jobs_retry_policy({"OPENMED_SERVICE_JOBS_MAX_ATTEMPTS": "0"})
    with pytest.raises(ValueError):
        jobs.parse_jobs_retry_policy({"OPENMED_SERVICE_JOBS_RETRY_BASE_SECONDS": "-2"})


class FakeLoader:
    """Minimal model loader double for warm-pool bookkeeping."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def test_rest_idempotency_key_returns_the_existing_job_or_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "OPENMED_SERVICE_JOBS_PAYLOAD_MODE",
        "OPENMED_SERVICE_JOBS_PAYLOAD_DIR",
        "OPENMED_SERVICE_JOBS_PAYLOAD_RETENTION_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_RECONCILE_SECONDS", "0")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setattr(
        openmed,
        "deidentify",
        lambda text, **_kwargs: SimpleNamespace(pii_entities=[]),
    )
    body = {"documents": [{"id": "synthetic-0", "text": TEXT_SENTINEL}]}

    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        first = client.post("/jobs", json=body, headers={"Idempotency-Key": KEY})
        again = client.post("/jobs", json=body, headers={"Idempotency-Key": KEY})
        changed = client.post(
            "/jobs",
            json={**body, "method": "remove"},
            headers={"Idempotency-Key": KEY},
        )
        malformed = client.post(
            "/jobs", json=body, headers={"Idempotency-Key": "short"}
        )
        job_id = first.json()["id"]
        for _ in range(200):
            status = client.get(f"/jobs/{job_id}").json()
            if status["status"] == "done":
                break
            time.sleep(0.01)

    assert first.status_code == 202 and again.status_code == 202
    assert again.json()["id"] == job_id
    assert "idempotent-replayed" not in first.headers
    assert again.headers["Idempotent-Replayed"] == "true"
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "idempotency_key_conflict"
    assert malformed.status_code == 400
    assert status["status"] == "done"
    assert status["state_version"] >= 4
    assert status["recovery"]["checkpoint"]["sequence"] == 1
    transitions = [
        (item["from"], item["to"]) for item in status["recovery"]["transitions"]
    ]
    assert transitions == [(None, "queued"), ("queued", "running"), ("running", "done")]
    for response in (first, again, changed, malformed):
        encoded = response.text
        assert TEXT_SENTINEL not in encoded and KEY not in encoded
        assert "lease_id" not in encoded and "worker_" not in encoded
    stored = (tmp_path / "jobs.json").read_text(encoding="utf-8")
    assert TEXT_SENTINEL not in stored and KEY not in stored
