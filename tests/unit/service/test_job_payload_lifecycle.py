"""Queue-level tests for protected async job and webhook payloads.

Covers the immutable submission snapshot (#3468), cancellation on shutdown
(#3467), immutable terminal-event delivery, value-free failure codes, the
memory-only mode, and restart recovery of encrypted payloads.
"""

from __future__ import annotations

import base64
import errno
import json
import logging
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from openmed.core.key_provider import (
    DATA_KEY_BYTES,
    InMemoryKeyProvider,
    KeyState,
)
from openmed.service import job_payloads, jobs
from openmed.service.job_payloads import (
    JOB_PAYLOAD_KEY_PURPOSE,
    PAYLOAD_KIND_INPUT,
    PAYLOAD_KIND_TERMINAL_EVENT,
    EncryptedJobPayloadStore,
    MemoryJobPayloadStore,
)
from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest
from openmed.service.webhooks import WebhookDeliveryResult

KEY_ONE = bytes(range(DATA_KEY_BYTES))
KEY_TWO = bytes(reversed(range(DATA_KEY_BYTES)))
TEXT_SENTINEL = "ZQQUEUESENTINEL58 Synthetic Patient"
SECRET_SENTINEL = "ZQQUEUESECRETSENTINEL19"
URL_SENTINEL = "https://zqqueuecallback.example.test/hook"
SENTINELS = (TEXT_SENTINEL, SECRET_SENTINEL, URL_SENTINEL)
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Clock:
    """Mutable deterministic clock."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class ControlledExecutor:
    """Executor double that records work and cancels it on shutdown."""

    def __init__(self, **_: Any) -> None:
        self.work: list[Any] = []
        self.futures: list[Future[None]] = []

    def submit(self, _operation: Any, item: Any) -> Future[None]:
        future: Future[None] = Future()
        self.work.append(item)
        self.futures.append(future)
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait
        if cancel_futures:
            for future in self.futures:
                future.cancel()


class RecordingSender:
    """Webhook sender double that records the bodies it was given."""

    def __init__(self, result: Optional[WebhookDeliveryResult] = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result or WebhookDeliveryResult(
            success=True, attempts=1, status_code=204
        )

    def __call__(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        secret: str,
        max_attempts: int,
        backoff_seconds: float,
        delivery_id: str,
    ) -> WebhookDeliveryResult:
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "secret": secret,
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
                "delivery_id": delivery_id,
            }
        )
        return self.result


def _provider(material: bytes = KEY_ONE) -> InMemoryKeyProvider:
    provider = InMemoryKeyProvider()
    provider.add_key(
        "jobs-key-1", material, purpose=JOB_PAYLOAD_KEY_PURPOSE, state=KeyState.ACTIVE
    )
    return provider


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(validate_served_model=lambda _model_name: None)


def _queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    clock: Clock,
    payload_store: Any = None,
    sender: Any = None,
) -> jobs.DeidentifyJobQueue:
    store = jobs.LocalJobStore(tmp_path / "jobs.json", clock=clock)
    with monkeypatch.context() as patched:
        patched.setattr(jobs, "ThreadPoolExecutor", ControlledExecutor)
        queue = jobs.DeidentifyJobQueue(
            _runtime(),
            store=store,
            clock=clock,
            payload_store=payload_store,
            webhook_sender=sender or RecordingSender(),
        )
    queue._deidentify_document = lambda _request, _document: SimpleNamespace(  # type: ignore[method-assign]
        pii_entities=[]
    )
    return queue


def _encrypted_store(
    tmp_path: Path, clock: Clock, material: bytes = KEY_ONE
) -> EncryptedJobPayloadStore:
    return EncryptedJobPayloadStore(
        tmp_path / "payloads",
        key_provider=_provider(material),
        max_retention_seconds=600,
        clock=clock,
    )


def _request(**overrides: Any) -> DeidentifyJobRequest:
    values: dict[str, Any] = {
        "documents": [DeidentifyJobDocument(id="synthetic-0", text=TEXT_SENTINEL)],
        "webhook": {
            "url": URL_SENTINEL,
            "secret": SECRET_SENTINEL,
            "max_attempts": 2,
            "backoff_seconds": 0,
        },
    }
    values.update(overrides)
    return DeidentifyJobRequest(**values)


def _all_bytes(root: Path) -> bytes:
    return b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())


def _assert_no_sentinels(blob: bytes | str) -> None:
    data = blob.encode("utf-8") if isinstance(blob, str) else blob
    for sentinel in SENTINELS:
        encoded = sentinel.encode("utf-8")
        assert encoded not in data
        assert base64.b64encode(encoded) not in data


def test_submission_snapshot_ignores_later_request_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression for #3468: metadata and queued work share one snapshot."""
    queue = _queue(monkeypatch, tmp_path, clock=Clock())
    payload = DeidentifyJobRequest(
        documents=[DeidentifyJobDocument(text="original synthetic text")],
        webhook={"url": URL_SENTINEL, "secret": SECRET_SENTINEL},
    )

    metadata = queue.submit(payload)
    payload.documents[0].text = "changed synthetic text"
    payload.documents.append(DeidentifyJobDocument(text="extra synthetic text"))
    payload.method = "remove"
    assert payload.webhook is not None
    payload.webhook.url = "https://changed.example.test/hook"

    item = queue._executor.work[0]
    assert not hasattr(item, "payload")
    queued = DeidentifyJobRequest(
        **queue.payloads.read(
            item.input_ref, job_id=metadata["id"], kind=PAYLOAD_KIND_INPUT
        )
    )
    assert metadata["document_count"] == 1
    assert len(queued.documents) == 1
    assert metadata["documents"][0]["text_hash"] == jobs.hash_text(
        queued.documents[0].text
    )
    assert queued.method == "mask"
    assert queued.webhook is not None and queued.webhook.url == URL_SENTINEL
    assert metadata["webhook"]["url_hash"] == jobs.hash_text(URL_SENTINEL)
    assert metadata["payloads"]["input"]["ref"] == item.input_ref
    queue.shutdown()


def test_worker_runs_the_snapshot_not_the_mutated_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    queue = _queue(monkeypatch, tmp_path, clock=Clock())
    payload = DeidentifyJobRequest(
        documents=[DeidentifyJobDocument(id="a", text="original synthetic text")]
    )
    seen: list[tuple[str, str]] = []

    def process(request: DeidentifyJobRequest, document: DeidentifyJobDocument):
        seen.append((request.method, document.text))
        return SimpleNamespace(pii_entities=[])

    queue._deidentify_document = process  # type: ignore[method-assign]
    queue.submit(payload)
    payload.documents[0].text = "mutated synthetic text"
    payload.method = "remove"

    queue._run_job(queue._executor.work[0])

    assert seen == [("mask", "original synthetic text")]
    queue.shutdown()


def test_shutdown_records_never_started_jobs_as_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression for #3467: cancelled futures no longer stay ``queued``."""
    clock = Clock()
    queue = _queue(monkeypatch, tmp_path, clock=clock)
    job_id = queue.submit(_request())["id"]
    clock.advance(5)

    queue.shutdown()

    assert queue._executor.futures[0].cancelled()
    reopened = jobs.LocalJobStore(queue.store.path, clock=clock)
    record = reopened.get(job_id)
    assert record is not None
    assert record["status"] == "cancelled"
    assert record["error"] == {
        "code": "cancelled_on_shutdown",
        "type": "CancelledError",
        "message": "Job was cancelled before it started",
    }
    assert record["completed_at"] == "2026-01-01T00:00:05Z"
    assert record["payloads"]["input"]["state"] == "deleted"
    assert record["payloads"]["input"]["reason"] == "cancelled"
    assert queue.payloads.retained_count() == 0
    assert record["webhook_delivery"] is None
    queue.shutdown()


def test_shutdown_leaves_running_and_finished_jobs_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    queue = _queue(monkeypatch, tmp_path, clock=Clock())
    running_id = queue.submit(_request(webhook=None))["id"]
    done_id = queue.submit(_request(webhook=None))["id"]
    queued_id = queue.submit(_request(webhook=None))["id"]
    running_future, done_future, _queued_future = queue._executor.futures
    assert running_future.set_running_or_notify_cancel()
    queue.store.update(running_id, status="running")
    queue._run_job(queue._executor.work[1])
    assert done_future.set_running_or_notify_cancel()
    done_future.set_result(None)

    queue.shutdown()

    assert queue.get(running_id)["status"] == "running"
    assert queue.get(running_id)["payloads"]["input"]["state"] == "retained"
    assert queue.get(done_id)["status"] == "done"
    assert queue.get(queued_id)["status"] == "cancelled"
    with pytest.raises(RuntimeError, match="shutting down"):
        queue.submit(_request())


def test_shutdown_surfaces_persistence_failures_without_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    queue = _queue(monkeypatch, tmp_path, clock=Clock())
    broken_id = queue.submit(_request())["id"]
    other_id = queue.submit(_request())["id"]
    original_update = queue.store.update

    def update(job_id: str, **changes: Any) -> dict[str, Any]:
        if job_id == broken_id and changes.get("status") == "cancelled":
            raise OSError(f"synthetic failure writing {tmp_path}")
        return original_update(job_id, **changes)

    monkeypatch.setattr(queue.store, "update", update)
    with pytest.raises(RuntimeError) as caught:
        queue.shutdown()

    assert str(caught.value) == "Cancelled job state could not be persisted"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert queue.get(other_id)["status"] == "cancelled"
    assert queue.get(broken_id)["status"] == "queued"


def test_cancelled_job_is_not_started_by_a_late_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    queue = _queue(monkeypatch, tmp_path, clock=Clock())
    job_id = queue.submit(_request())["id"]
    queue.shutdown()

    queue._run_job(queue._executor.work[0])

    assert queue.get(job_id)["status"] == "cancelled"
    assert queue.webhook_sender.calls == []


def test_memory_mode_leaves_no_payload_artifact_after_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sender = RecordingSender()
    queue = _queue(monkeypatch, tmp_path, clock=Clock(), sender=sender)
    job_id = queue.submit(_request())["id"]

    queue._run_job(queue._executor.work[0])

    record = queue.get(job_id)
    assert record["status"] == "done"
    assert record["payloads"]["mode"] == "memory"
    assert record["payloads"]["input"]["state"] == "deleted"
    assert record["payloads"]["input"]["reason"] == "consumed"
    assert record["payloads"]["terminal_event"]["state"] == "deleted"
    assert record["payloads"]["terminal_event"]["reason"] == "delivered"
    assert queue.payloads.retained_count() == 0
    assert isinstance(queue.payloads, MemoryJobPayloadStore)
    assert queue.payloads._blobs == {}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["jobs.json"]
    assert sender.calls[0]["secret"] == SECRET_SENTINEL
    assert sender.calls[0]["url"] == URL_SENTINEL
    _assert_no_sentinels(_all_bytes(tmp_path))


def test_memory_mode_drops_failed_delivery_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sender = RecordingSender(
        WebhookDeliveryResult(
            success=False, attempts=2, status_code=500, error="HTTP 500"
        )
    )
    queue = _queue(monkeypatch, tmp_path, clock=Clock(), sender=sender)
    job_id = queue.submit(_request())["id"]

    queue._run_job(queue._executor.work[0])

    record = queue.get(job_id)
    assert record["webhook_delivery"]["success"] is False
    assert record["payloads"]["terminal_event"]["reason"] == "delivery_failed"
    assert queue.payloads.retained_count() == 0


def test_webhook_delivery_uses_the_immutable_terminal_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    sender = RecordingSender(
        WebhookDeliveryResult(success=False, attempts=2, status_code=503)
    )
    payload_store = _encrypted_store(tmp_path, clock)
    queue = _queue(
        monkeypatch, tmp_path, clock=clock, payload_store=payload_store, sender=sender
    )
    job_id = queue.submit(_request())["id"]
    queue._run_job(queue._executor.work[0])
    record = queue.get(job_id)
    terminal = record["payloads"]["terminal_event"]
    first_body = sender.calls[0]["payload"]
    assert first_body == jobs.webhook_payload(record)
    assert record["webhook_delivery"]["payload_ref"] == terminal["ref"]

    queue.store.update(job_id, label_histogram={"MUTATED": 9}, status="failed")
    retry_ref = payload_store._index[terminal["ref"]]
    queue._deliver_terminal_event(job_id, retry_ref)

    assert sender.calls[1]["payload"] == first_body
    assert sender.calls[1]["payload"]["label_histogram"] == {}
    assert sender.calls[1]["payload"]["status"] == "done"
    retry_metadata = json.dumps(queue.get(job_id)["webhook_delivery"])
    _assert_no_sentinels(retry_metadata)
    assert "label_histogram" not in retry_metadata
    queue.shutdown()


def test_retries_and_restarts_reuse_the_sealed_delivery_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    sender = RecordingSender(
        WebhookDeliveryResult(
            success=False,
            attempts=2,
            status_code=503,
            delivery_id="dlv_sender_supplied_value",
        )
    )
    first = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock),
        sender=sender,
    )
    job_id = first.submit(_request())["id"]
    first._run_job(first._executor.work[0])
    record = first.get(job_id)
    terminal_ref = record["payloads"]["terminal_event"]["ref"]
    delivery_id = record["webhook_delivery"]["delivery_id"]
    assert delivery_id.startswith("dlv_")
    assert sender.calls[0]["delivery_id"] == delivery_id

    first._deliver_terminal_event(job_id, first.payloads._index[terminal_ref])
    assert sender.calls[1]["delivery_id"] == delivery_id
    first.shutdown = lambda: None  # type: ignore[method-assign]

    # After a restart the terminal event, and so the delivery id, is reread.
    second = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock),
        sender=sender,
    )
    second._deliver_terminal_event(job_id, second.payloads._index[terminal_ref])

    assert sender.calls[2]["delivery_id"] == delivery_id
    delivery = second.get(job_id)["webhook_delivery"]
    assert delivery["delivery_id"] == delivery_id
    assert delivery["payload_ref"] == terminal_ref
    _assert_no_sentinels(json.dumps(delivery))
    _assert_no_sentinels(_all_bytes(tmp_path))
    second.shutdown()


def test_a_terminal_event_without_a_valid_delivery_id_is_damaged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    queue = _queue(monkeypatch, tmp_path, clock=Clock())
    event = {
        "delivery": {
            "url": URL_SENTINEL,
            "secret": SECRET_SENTINEL,
            "max_attempts": 1,
            "backoff_seconds": 0,
            "delivery_id": "not a delivery id",
        },
        "body": {"event": "job.done"},
    }

    with pytest.raises(job_payloads.JobPayloadError) as damaged:
        queue._send_webhook(event)
    assert damaged.value.code == "payload_damaged"
    assert queue.webhook_sender.calls == []
    queue.shutdown()


def test_sender_exception_is_recorded_as_a_value_free_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> WebhookDeliveryResult:
        raise RuntimeError(f"failed posting {URL_SENTINEL} with {SECRET_SENTINEL}")

    queue = _queue(monkeypatch, tmp_path, clock=Clock(), sender=explode)
    job_id = queue.submit(_request())["id"]
    with caplog.at_level(logging.DEBUG):
        queue._run_job(queue._executor.work[0])

    delivery = queue.get(job_id)["webhook_delivery"]
    assert delivery["success"] is False
    assert delivery["attempts"] == 0
    assert delivery["error"] == "webhook_sender_failed"
    _assert_no_sentinels(_all_bytes(tmp_path))
    _assert_no_sentinels(caplog.text)


def test_terminal_event_write_failure_skips_delivery_with_a_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    sender = RecordingSender()
    payload_store = _encrypted_store(tmp_path, clock)
    queue = _queue(
        monkeypatch, tmp_path, clock=clock, payload_store=payload_store, sender=sender
    )
    job_id = queue.submit(_request())["id"]

    def disk_full(_descriptor: int) -> None:
        raise OSError(errno.ENOSPC, "synthetic disk full")

    with monkeypatch.context() as patched:
        patched.setattr(job_payloads.os, "fsync", disk_full)
        queue._run_job(queue._executor.work[0])

    record = queue.get(job_id)
    assert record["status"] == "done"
    assert record["webhook_delivery"] == {
        "success": False,
        "attempts": 0,
        "status_code": None,
        "error": "payload_storage_full",
        "payload_ref": None,
    }
    assert record["payloads"]["terminal_event"] is None
    assert record["payloads"]["input"]["state"] == "deleted"
    assert sender.calls == []
    assert list((tmp_path / "payloads").iterdir()) == []


def test_unreadable_input_fails_closed_and_keeps_the_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    queue = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock),
    )
    job_id = queue.submit(_request())["id"]
    item = queue._executor.work[0]
    before = _all_bytes(tmp_path / "payloads")
    queue.shutdown = lambda: None  # type: ignore[method-assign]

    wrong = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock, material=KEY_TWO),
    )
    wrong._run_job(item)

    record = wrong.get(job_id)
    assert record["status"] == "failed"
    assert record["error"] == {
        "code": "envelope_authentication_failed",
        "type": "JobPayloadError",
        "message": "Job input payload could not be read",
    }
    assert record["payloads"]["input"]["state"] == "retained"
    assert _all_bytes(tmp_path / "payloads") == before
    assert wrong.webhook_sender.calls == []


def test_input_expiring_while_queued_fails_the_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    queue = _queue(monkeypatch, tmp_path, clock=clock)
    job_id = queue.submit(_request(payload_retention_seconds=30))["id"]
    clock.advance(30)

    queue._run_job(queue._executor.work[0])

    record = queue.get(job_id)
    assert record["status"] == "failed"
    assert record["error"]["code"] == "payload_expired"
    assert record["payloads"]["input"]["state"] == "expired"
    assert record["payloads"]["input"]["reason"] == "retention_elapsed"
    queue.shutdown()


def test_requested_retention_is_bounded_by_the_service_maximum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    queue = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=MemoryJobPayloadStore(max_retention_seconds=120, clock=clock),
    )

    shorter = queue.submit(_request(payload_retention_seconds=60))
    default = queue.submit(_request())
    with pytest.raises(ValueError, match="maximum"):
        queue.submit(_request(payload_retention_seconds=121))

    assert shorter["payloads"]["retention_seconds"] == 60
    assert shorter["payloads"]["input"]["expires_at"] == "2026-01-01T00:01:00Z"
    assert default["payloads"]["retention_seconds"] == 120
    assert queue.payloads.retained_count() == 2
    with pytest.raises(ValueError):
        DeidentifyJobRequest(
            documents=[{"text": "synthetic"}], payload_retention_seconds=0
        )
    queue.shutdown()


def test_submit_rolls_back_the_payload_when_the_record_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    queue = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock),
    )

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic metadata failure")

    with monkeypatch.context() as patched:
        patched.setattr(queue.store, "_persist_locked", fail)
        with pytest.raises(OSError, match="synthetic metadata failure"):
            queue.submit(_request())

    assert queue.payloads.retained_count() == 0
    assert list((tmp_path / "payloads").glob("*.payload.json")) == []
    assert queue._executor.work == []
    receipts = [(r.event, r.reason) for r in queue.payloads.receipts()]
    assert receipts[-1] == ("deleted", "submission_failed")


def test_restart_recovers_encrypted_input_and_terminal_event_until_expiry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    first = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock),
    )
    job_id = first.submit(_request(payload_retention_seconds=300))["id"]
    first.shutdown = lambda: None  # type: ignore[method-assign]

    # Restart: a new queue with the same key reopens the queued input snapshot.
    failing = RecordingSender(
        WebhookDeliveryResult(success=False, attempts=2, status_code=502)
    )
    second = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock),
        sender=failing,
    )
    record = second.get(job_id)
    input_ref = record["payloads"]["input"]["ref"]
    recovered = second.payloads.read(input_ref, job_id=job_id, kind=PAYLOAD_KIND_INPUT)
    assert recovered["documents"][0]["text"] == TEXT_SENTINEL
    second._run_job(jobs._JobWorkItem(job_id, input_ref))
    record = second.get(job_id)
    terminal = record["payloads"]["terminal_event"]
    assert record["status"] == "done"
    assert terminal["state"] == "retained"
    assert terminal["expires_at"] == "2026-01-01T00:05:00Z"
    assert record["webhook_delivery"]["payload_ref"] == terminal["ref"]
    second.shutdown = lambda: None  # type: ignore[method-assign]

    # A second restart can still reach the terminal event until its deadline.
    clock.advance(299)
    third = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock),
    )
    event = third.payloads.read(
        terminal["ref"], job_id=job_id, kind=PAYLOAD_KIND_TERMINAL_EVENT
    )
    assert event["body"] == jobs.webhook_payload(record)
    assert event["delivery"]["secret"] == SECRET_SENTINEL

    clock.advance(1)
    expired = third.get(job_id)
    assert expired["payloads"]["terminal_event"]["state"] == "expired"
    assert list((tmp_path / "payloads").iterdir()) == []

    for blob in [
        _all_bytes(tmp_path),
        json.dumps(expired),
        json.dumps(third.payloads.lifecycle_evidence()),
    ]:
        _assert_no_sentinels(blob)
    assert str(tmp_path) not in json.dumps(expired)
    third.shutdown()


def test_startup_removes_payloads_without_a_live_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    store = _encrypted_store(tmp_path, clock)
    orphan = store.create(
        PAYLOAD_KIND_INPUT, "0" * 32, {"synthetic": True}, retention_seconds=60
    )
    remnant = tmp_path / "payloads" / f".{orphan.ref}.zz9_tmp.tmp"
    remnant.write_bytes(b"synthetic ciphertext remnant")

    queue = _queue(
        monkeypatch,
        tmp_path,
        clock=clock,
        payload_store=_encrypted_store(tmp_path, clock),
    )

    assert list((tmp_path / "payloads").iterdir()) == []
    reasons = {(r.event, r.reason) for r in queue.payloads.receipts()}
    assert reasons == {("deleted", "orphan"), ("deleted", "crash_remnant")}
    queue.shutdown()


def test_queue_construction_fails_closed_on_an_unreadable_payload_dir(
    tmp_path: Path,
) -> None:
    target = tmp_path / "payloads"
    target.write_text("not a directory", encoding="utf-8")

    with pytest.raises(job_payloads.JobPayloadError) as caught:
        _encrypted_store(tmp_path, Clock())

    assert caught.value.code == "payload_read_failed"
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize(
    ("environment", "with_provider", "expected"),
    [
        ({}, False, "memory"),
        ({}, True, "encrypted"),
        ({"OPENMED_SERVICE_JOBS_PAYLOAD_MODE": "memory"}, True, "memory"),
        ({"OPENMED_SERVICE_JOBS_PAYLOAD_MODE": "Encrypted"}, True, "encrypted"),
        ({"OPENMED_SERVICE_JOBS_PAYLOAD_MODE": "encrypted"}, False, ValueError),
        ({"OPENMED_SERVICE_JOBS_PAYLOAD_MODE": "plaintext"}, True, ValueError),
        ({"OPENMED_SERVICE_JOBS_PAYLOAD_RETENTION_SECONDS": "0"}, False, ValueError),
    ],
)
def test_payload_store_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment: dict[str, str],
    with_provider: bool,
    expected: Any,
) -> None:
    for name in (
        "OPENMED_SERVICE_JOBS_PAYLOAD_MODE",
        "OPENMED_SERVICE_JOBS_PAYLOAD_DIR",
        "OPENMED_SERVICE_JOBS_PAYLOAD_RETENTION_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_PAYLOAD_DIR", str(tmp_path / "p"))
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    provider = _provider() if with_provider else None

    if expected is ValueError:
        with pytest.raises(ValueError):
            jobs.job_payload_store_from_env(key_provider=provider)
        return
    store = jobs.job_payload_store_from_env(key_provider=provider)
    assert store.mode == expected
    assert store.max_retention_seconds == job_payloads.DEFAULT_PAYLOAD_RETENTION_SECONDS
    assert not (tmp_path / "p").exists()
