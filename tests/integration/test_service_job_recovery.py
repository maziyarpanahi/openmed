"""Offline crash, restart, and fencing tests for durable service jobs (#3574).

Every test runs the public ``/jobs`` queue against real files: a
``LocalJobStore`` JSON file and an encrypted payload directory sealed with a
synthetic key. A "process" is one queue built from those files. A crash is a
``SimulatedCrash`` (a ``BaseException``) raised from the queue's fault hook at
a durability boundary, which unwinds the worker like a kill; the queue object
is then abandoned without shutdown and a fresh one is built from disk. One
test kills a real child process with ``os._exit``.

Documents, webhook targets, and secrets carry synthetic sentinels; each test
scans the metadata file, the payload directory, decision receipts, recovery
evidence, public responses, and logs for them.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openmed.core.key_provider import DATA_KEY_BYTES, InMemoryKeyProvider, KeyState
from openmed.service import jobs
from openmed.service.job_payloads import (
    JOB_PAYLOAD_KEY_PURPOSE,
    EncryptedJobPayloadStore,
)
from openmed.service.job_recovery import (
    FAULT_BOUNDARIES,
    JOB_STATUSES,
    IdempotencyConflictError,
    IdempotencyKeyError,
    JobReplayAuthorization,
    ReplayRefusedError,
    RetryPolicy,
    StaleLeaseError,
)
from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest
from openmed.service.webhooks import WebhookDeliveryResult

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
KEY = bytes(range(DATA_KEY_BYTES))
TEXT_SENTINELS = (
    "ZQRECOVERYSENTINEL01 Synthetic Patient",
    "ZQRECOVERYSENTINEL02 Synthetic Patient",
    "ZQRECOVERYSENTINEL03 Synthetic Patient",
)
SECRET_SENTINEL = "zqrecovery-webhook-secret-77"
URL_SENTINEL = "https://zqrecovery.example.test/hooks/jobs"
SENTINELS = (*TEXT_SENTINELS, SECRET_SENTINEL, URL_SENTINEL)
START = datetime(2026, 3, 1, tzinfo=timezone.utc)
LEASE_SECONDS = 30
STEP_SECONDS = 120
REVIEWER = "sha256:" + "a" * 64
EVIDENCE = "sha256:" + "b" * 64
DOCUMENT_BOUNDARIES = {"document.processed", "checkpoint.committed"}


class SimulatedCrash(BaseException):
    """Abrupt worker termination that bypasses ordinary exception handling."""


class Clock:
    """Mutable deterministic UTC clock."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class ManualExecutor:
    """Executor double: records work so a test can run it, or never run it."""

    def __init__(self, **_: Any) -> None:
        self.work: list[tuple[Callable[[Any], None], Any, Future[None]]] = []

    def submit(self, operation: Callable[[Any], None], item: Any) -> Future[None]:
        future: Future[None] = Future()
        self.work.append((operation, item, future))
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait
        if cancel_futures:
            for _operation, _item, future in self.work:
                future.cancel()


class RecordingSender:
    """Webhook sender double recording every delivery it is asked to make."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        secret: str,
        max_attempts: int,
        backoff_seconds: float,
    ) -> WebhookDeliveryResult:
        self.calls.append({"url": url, "payload": payload, "secret": secret})
        return WebhookDeliveryResult(success=True, attempts=1, status_code=204)


class Deidentifier:
    """Deterministic model double; ``behaviour`` can raise per document."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.behaviour: Callable[[str], None] = lambda _document_id: None

    def __call__(
        self, _request: DeidentifyJobRequest, document: DeidentifyJobDocument
    ) -> SimpleNamespace:
        document_id = document.id or ""
        self.calls.append(document_id)
        self.behaviour(document_id)
        return SimpleNamespace(
            pii_entities=[
                SimpleNamespace(
                    label="NAME",
                    text=document.text[:20],
                    start=0,
                    end=20,
                    confidence=0.9,
                )
            ]
        )


def crash_at(boundary: str, *, occurrence: int = 1) -> Callable[[str, str], None]:
    """Return a fault hook that crashes the process at one boundary hit."""

    seen = {"count": 0}

    def hook(hit: str, _job_id: str) -> None:
        if hit != boundary:
            return
        seen["count"] += 1
        if seen["count"] == occurrence:
            raise SimulatedCrash(boundary)

    return hook


class Harness:
    """Builds one queue "process" at a time over the same on-disk state."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = tmp_path
        self.monkeypatch = monkeypatch
        self.clock = Clock()
        self.sender = RecordingSender()
        self.model = Deidentifier()
        self.queues: list[jobs.DeidentifyJobQueue] = []

    @property
    def store_path(self) -> Path:
        return self.root / "jobs.json"

    def start(
        self,
        *,
        fault: Optional[Callable[[str, str], None]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        replay_authorizer: Any = None,
    ) -> jobs.DeidentifyJobQueue:
        provider = InMemoryKeyProvider()
        provider.add_key(
            "recovery-key-1",
            KEY,
            purpose=JOB_PAYLOAD_KEY_PURPOSE,
            state=KeyState.ACTIVE,
        )
        payloads = EncryptedJobPayloadStore(
            self.root / "payloads",
            key_provider=provider,
            max_retention_seconds=24 * 60 * 60,
            clock=self.clock,
        )
        store = jobs.LocalJobStore(self.store_path, clock=self.clock)
        with self.monkeypatch.context() as patched:
            patched.setattr(jobs, "ThreadPoolExecutor", ManualExecutor)
            queue = jobs.DeidentifyJobQueue(
                SimpleNamespace(validate_served_model=lambda _name: None),
                store=store,
                clock=self.clock,
                payload_store=payloads,
                webhook_sender=self.sender,
                retry_policy=retry_policy
                or RetryPolicy(
                    max_attempts=3, base_delay_seconds=5, max_delay_seconds=60
                ),
                lease_seconds=LEASE_SECONDS,
                replay_authorizer=replay_authorizer,
                fault_hook=fault,
            )
        queue._deidentify_document = self.model  # type: ignore[method-assign]
        self.queues.append(queue)
        return queue


def drain(queue: jobs.DeidentifyJobQueue) -> bool:
    """Run recorded work in order; return ``True`` if the process crashed."""

    executor = queue._executor
    assert isinstance(executor, ManualExecutor)
    while executor.work:
        operation, item, future = executor.work.pop(0)
        if not future.set_running_or_notify_cancel():
            continue
        try:
            operation(item)
        except SimulatedCrash:
            return True
        future.set_result(None)
    return False


def recover(
    harness: Harness,
    job_id: str,
    *,
    observations: Optional[list[dict[str, Any]]] = None,
    rounds: int = 12,
    **start: Any,
) -> jobs.DeidentifyJobQueue:
    """Restart, then advance time and reconcile until the job settles."""

    queue = harness.start(**start)
    for _ in range(rounds):
        crashed = drain(queue)
        record = queue.get(job_id)
        assert record is not None
        if observations is not None:
            observations.append(jobs.job_response_payload(record))
        if crashed:
            queue = harness.start(**start)
            continue
        intent = (record.get("recovery") or {}).get("webhook") or {}
        if record["status"] in {"done", "failed", "cancelled", "quarantined"} and (
            intent.get("state") != "pending"
        ):
            return queue
        harness.clock.advance(STEP_SECONDS)
        queue.reconcile()
    return queue


def _request(**overrides: Any) -> DeidentifyJobRequest:
    values: dict[str, Any] = {
        "documents": [
            DeidentifyJobDocument(id=f"d{index}", text=text)
            for index, text in enumerate(TEXT_SENTINELS)
        ],
        "webhook": {
            "url": URL_SENTINEL,
            "secret": SECRET_SENTINEL,
            "max_attempts": 2,
            "backoff_seconds": 0,
        },
    }
    values.update(overrides)
    return DeidentifyJobRequest(**values)


def _only_job_id(harness: Harness) -> str:
    records = json.loads(harness.store_path.read_text(encoding="utf-8"))["jobs"]
    assert len(records) == 1
    return next(iter(records))


def _all_bytes(root: Path) -> bytes:
    return b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())


def _assert_no_sentinels(blob: bytes | str) -> None:
    data = blob.encode("utf-8") if isinstance(blob, str) else blob
    for sentinel in SENTINELS:
        encoded = sentinel.encode("utf-8")
        assert encoded not in data
        assert base64.b64encode(encoded) not in data


def _assert_value_free(harness: Harness, queue: jobs.DeidentifyJobQueue) -> None:
    _assert_no_sentinels(_all_bytes(harness.root))
    _assert_no_sentinels(json.dumps(queue.decision_receipts()))
    _assert_no_sentinels(json.dumps(queue.recovery_evidence()))
    for record in queue.store.records():
        public = json.dumps(jobs.job_response_payload(record))
        _assert_no_sentinels(public)
        assert queue.worker_id not in public
        assert "lease_id" not in public and "worker_" not in public
        assert str(harness.root) not in public


def _assert_single_logical_completion(record: dict[str, Any]) -> None:
    assert record["status"] == "done"
    assert record["processed_count"] == 3
    assert record["failed_count"] == 0
    assert [span["document_id"] for span in record["spans"]] == ["d0", "d1", "d2"]
    assert record["label_histogram"] == {"NAME": 3}
    assert record["recovery"]["completed_documents"] == 3
    assert record["recovery"]["checkpoint"]["sequence"] == 3


@pytest.mark.parametrize("boundary", FAULT_BOUNDARIES)
def test_crash_at_every_boundary_resumes_without_duplicate_effects(
    boundary: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    occurrence = 2 if boundary in DOCUMENT_BOUNDARIES else 1
    first = harness.start(fault=crash_at(boundary, occurrence=occurrence))
    caplog.set_level(logging.DEBUG)

    if boundary == "submit.persisted":
        with pytest.raises(SimulatedCrash):
            first.submit(_request())
        job_id = _only_job_id(harness)
        assert first._executor.work == []
    else:
        job_id = first.submit(_request())["id"]
        assert drain(first) is True

    observations: list[dict[str, Any]] = []
    final_queue = recover(harness, job_id, observations=observations)
    record = final_queue.get(job_id)
    assert record is not None

    _assert_single_logical_completion(record)
    assert record["webhook_delivery"]["success"] is True
    assert record["recovery"]["webhook"]["state"] == "delivered"
    bodies = [call["payload"] for call in harness.sender.calls]
    # Delivery is at-least-once only across a crash after the HTTP send and
    # before its result was recorded; the redelivered body is identical.
    assert len(bodies) == (2 if boundary == "webhook.sent" else 1)
    assert all(body == bodies[0] for body in bodies)
    assert bodies[0]["status"] == "done" and len(bodies[0]["spans"]) == 3
    # Only a document whose result was never committed is processed again.
    expected_calls = (
        ["d0", "d1", "d1", "d2"]
        if boundary == "document.processed"
        else ["d0", "d1", "d2"]
    )
    assert harness.model.calls == expected_calls
    # Nothing protected is left behind once the webhook was delivered.
    assert list((tmp_path / "payloads").iterdir()) == []
    assert record["payloads"]["input"]["state"] == "deleted"
    assert record["payloads"]["terminal_event"]["state"] == "deleted"
    # Public polling only ever reported typed states, with a monotone version.
    versions = [item["state_version"] for item in observations]
    assert versions == sorted(versions)
    assert {item["status"] for item in observations} <= JOB_STATUSES
    for item in observations:
        assert item["recovery"]["schema_version"] == 1
    _assert_value_free(harness, final_queue)
    _assert_no_sentinels(caplog.text)


def test_lease_reclaim_fences_a_stale_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    worker_a = harness.start()
    job_id = worker_a.submit(_request())["id"]
    (_operation, item, _future) = worker_a._executor.work.pop(0)

    entered = threading.Event()
    release = threading.Event()
    stale_errors: list[BaseException] = []

    def stalled(
        _request: DeidentifyJobRequest, document: DeidentifyJobDocument
    ) -> SimpleNamespace:
        entered.set()
        assert release.wait(10)
        return SimpleNamespace(pii_entities=[])

    worker_a._deidentify_document = stalled  # type: ignore[method-assign]

    def run_a() -> None:
        try:
            worker_a._run_job(item)
        except BaseException as exc:  # pragma: no cover - surfaced below
            stale_errors.append(exc)

    thread = threading.Thread(target=run_a)
    thread.start()
    assert entered.wait(10)
    held = worker_a.get(job_id)
    assert held["status"] == "running"
    assert held["recovery"]["lease_epoch"] == 1

    # Worker A stalls past its lease; worker B reclaims and completes the job.
    harness.clock.advance(LEASE_SECONDS + 1)
    worker_b = harness.start()
    assert worker_b.reconcile() == {"waiting": 1}
    reclaimed = worker_b.get(job_id)
    assert reclaimed["status"] == "queued"
    assert reclaimed["recovery"]["retry"]["reason_code"] == "lease_expired"
    harness.clock.advance(5)
    assert worker_b.reconcile() == {"dispatched": 1}
    assert drain(worker_b) is False
    done = worker_b.get(job_id)
    _assert_single_logical_completion(done)
    assert done["recovery"]["lease_epoch"] == 2

    # The stale worker wakes up and cannot commit anything.
    release.set()
    thread.join(10)
    assert not thread.is_alive()
    assert stale_errors == []
    after = worker_b.get(job_id)
    assert after["spans"] == done["spans"]
    assert after["state_version"] == done["state_version"]
    assert len(harness.sender.calls) == 1
    actions = [receipt["action"] for receipt in worker_b.decision_receipts()]
    assert actions == ["lease_reclaimed"]
    _assert_value_free(harness, worker_b)


def test_stale_lease_cannot_commit_a_checkpoint_or_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    queue = harness.start()
    job_id = queue.submit(_request())["id"]
    claimed = queue._claim(job_id)
    assert claimed is not None
    stale_lease, record = claimed

    harness.clock.advance(LEASE_SECONDS + 1)
    queue.reconcile()
    harness.clock.advance(5)
    fresh = queue._claim(job_id)
    assert fresh is not None and fresh[0].epoch == stale_lease.epoch + 1

    summary = jobs._JobSummary.restore(record)
    with pytest.raises(StaleLeaseError):
        queue._commit_checkpoint(
            stale_lease,
            summary,
            sequence=1,
            error=None,
            output_digest="sha256:" + "c" * 64,
        )
    request = _request()
    retained_before = queue.payloads.retained_count()
    with pytest.raises(StaleLeaseError):
        queue._finish_job(stale_lease, request, record, summary, None)
    # The terminal event sealed by the stale worker was never referenced.
    assert queue.payloads.retained_count() == retained_before
    current = queue.get(job_id)
    assert current["status"] == "running"
    assert current["recovery"]["lease"]["epoch"] == fresh[0].epoch
    assert current["recovery"]["completed_documents"] == 0
    assert harness.sender.calls == []


def test_concurrent_claims_issue_exactly_one_lease_per_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    queue_a = harness.start()
    job_id = queue_a.submit(_request())["id"]
    queue_b = harness.start()
    barrier = threading.Barrier(8)
    results: list[Any] = []
    lock = threading.Lock()

    def claim(queue: jobs.DeidentifyJobQueue) -> None:
        barrier.wait(10)
        outcome = queue._claim(job_id)
        with lock:
            results.append(outcome)

    threads = [
        threading.Thread(target=claim, args=(queue,))
        for queue in (queue_a, queue_b) * 4
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    leases = [outcome[0] for outcome in results if outcome is not None]
    assert len(results) == 8
    assert len(leases) == 1 and leases[0].epoch == 1
    record = jobs.LocalJobStore(harness.store_path, clock=harness.clock).get(job_id)
    assert record["recovery"]["lease_epoch"] == 1
    assert record["recovery"]["attempt"] == 1
    claims = [
        entry
        for entry in record["recovery"]["transitions"]
        if entry["reason"] == "claimed"
    ]
    assert len(claims) == 1


def test_poison_work_is_quarantined_until_an_authorized_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)

    def poison(document_id: str) -> None:
        if document_id == "d1":
            raise SimulatedCrash("worker killed by document")

    harness.model.behaviour = poison
    first = harness.start()
    job_id = first.submit(_request())["id"]
    assert drain(first) is True

    queue = recover(harness, job_id)
    record = queue.get(job_id)
    assert record["status"] == "quarantined"
    assert record["error"]["code"] == "attempts_exhausted"
    quarantine = record["recovery"]["quarantine"]
    assert quarantine["classification"] == "unsafe"
    assert quarantine["reason_code"] == "attempts_exhausted"
    assert quarantine["failure_count"] == 3
    assert quarantine["candidate_count"] == 1
    assert record["recovery"]["completed_documents"] == 1
    assert [span["document_id"] for span in record["spans"]] == ["d0"]
    assert harness.model.calls == ["d0", "d1", "d1", "d1"]
    assert record["payloads"]["input"]["state"] == "retained"
    assert harness.sender.calls == []

    # Quarantine survives restarts and is never dispatched again on its own.
    restarted = harness.start()
    harness.clock.advance(STEP_SECONDS * 4)
    assert restarted.reconcile() == {}
    assert drain(restarted) is False
    assert restarted._claim(job_id) is None
    assert restarted.get(job_id)["status"] == "quarantined"

    # Replay requires an explicit, authorized decision.
    with pytest.raises(ReplayRefusedError) as missing:
        restarted.replay(job_id, None)  # type: ignore[arg-type]
    assert missing.value.code == "replay_unauthorized"
    with pytest.raises(ReplayRefusedError):
        JobReplayAuthorization(
            reviewer_digest="reviewer-name", evidence_digest=EVIDENCE
        )
    denied = harness.start(replay_authorizer=lambda _job_id, _auth: False)
    with pytest.raises(ReplayRefusedError) as refused:
        denied.replay(job_id, JobReplayAuthorization(REVIEWER, EVIDENCE))
    assert refused.value.code == "replay_unauthorized"
    assert denied.get(job_id)["status"] == "quarantined"

    harness.model.behaviour = lambda _document_id: None
    approvals: list[str] = []
    approver = harness.start(
        replay_authorizer=lambda job, auth: approvals.append(job) or True
    )
    replayed = approver.replay(job_id, JobReplayAuthorization(REVIEWER, EVIDENCE))
    assert approvals == [job_id]
    assert replayed["status"] == "queued"
    assert replayed["recovery"]["execution"] == 2
    assert replayed["recovery"]["attempt"] == 0
    with pytest.raises(ReplayRefusedError) as again:
        approver.replay(job_id, JobReplayAuthorization(REVIEWER, EVIDENCE))
    assert again.value.code == "job_not_quarantined"
    assert drain(approver) is False

    done = approver.get(job_id)
    _assert_single_logical_completion(done)
    # The committed first document was not processed again by the replay.
    assert harness.model.calls.count("d0") == 1
    history = done["recovery"]
    assert len(history["quarantines"]) == 1
    assert history["quarantine"] is None
    assert history["promotions"][0]["reviewer_digest"] == REVIEWER
    assert history["promotions"][0]["evidence_digest"] == EVIDENCE
    reasons = [(entry["from"], entry["to"]) for entry in history["transitions"]]
    assert ("running", "quarantined") in reasons
    assert ("quarantined", "queued") in reasons
    receipts = approver.decision_receipts()
    actions = [receipt["action"] for receipt in receipts]
    assert actions.count("lease_reclaimed") == 3
    assert actions.count("quarantined") == 1
    assert actions[-1] == "replay_authorized"
    assert receipts[-1]["actor_digest"] == REVIEWER
    assert receipts[-1]["evidence_digest"] == EVIDENCE
    assert len(harness.sender.calls) == 1
    _assert_value_free(harness, approver)


def test_retry_backoff_and_exhaustion_survive_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    failures = {"d0": 1}

    def flaky(document_id: str) -> None:
        if failures.get(document_id, 0) > 0:
            failures[document_id] -= 1
            raise TimeoutError("synthetic model timeout")

    harness.model.behaviour = flaky
    first = harness.start()
    job_id = first.submit(_request())["id"]
    assert drain(first) is False
    waiting = first.get(job_id)
    assert waiting["status"] == "queued"
    retry = waiting["recovery"]["retry"]
    assert retry["classification"] == "transient"
    assert retry["reason_code"] == "operation_timeout"
    assert retry["retry_after"] == "2026-03-01T00:00:05Z"

    # A restart before the backoff elapses keeps the job waiting.
    restarted = harness.start()
    assert restarted._executor.work == []
    harness.clock.advance(4)
    assert restarted.reconcile() == {"waiting": 1}
    harness.clock.advance(1)
    assert restarted.reconcile() == {"dispatched": 1}
    assert drain(restarted) is False
    _assert_single_logical_completion(restarted.get(job_id))

    # Retries that never succeed end in quarantine, not in a failed result.
    harness.model.behaviour = lambda _document_id: (_ for _ in ()).throw(
        ConnectionError("synthetic transport failure")
    )
    other = restarted.submit(_request())["id"]
    queue = restarted
    for _ in range(6):
        drain(queue)
        queue = harness.start()
        harness.clock.advance(STEP_SECONDS)
        queue.reconcile()
    record = queue.get(other)
    assert record["status"] == "quarantined"
    assert record["recovery"]["attempt"] == 3
    assert record["recovery"]["quarantine"]["reason_code"] == "attempts_exhausted"
    assert record["recovery"]["retry"]["reason_code"] == "transport_unavailable"
    assert record["processed_count"] == 0
    _assert_value_free(harness, queue)


def test_cancellation_and_expiry_survive_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    queue = harness.start()
    cancelled_id = queue.submit(_request())["id"]
    expiring_id = queue.submit(_request(payload_retention_seconds=60))["id"]
    queue._executor.work.clear()

    cancelled = queue.cancel(cancelled_id)
    assert cancelled["status"] == "cancelled"
    assert cancelled["payloads"]["input"]["state"] == "deleted"
    assert cancelled["recovery"]["cancellation"]["reason_code"] == "cancel_requested"
    assert queue.cancel(cancelled_id)["state_version"] == cancelled["state_version"]

    harness.clock.advance(61)
    restarted = harness.start()
    assert restarted.get(cancelled_id)["status"] == "cancelled"
    assert restarted._claim(cancelled_id) is None
    expired = restarted.get(expiring_id)
    assert expired["status"] == "failed"
    assert expired["error"]["code"] == "payload_expired"
    assert expired["payloads"]["input"]["state"] == "expired"
    assert restarted._executor.work == []
    assert list((tmp_path / "payloads").iterdir()) == []
    actions = {receipt["action"] for receipt in restarted.decision_receipts()}
    assert actions == {"cancelled", "recovery_failed"}
    with pytest.raises(jobs.JobRecoveryError) as finished:
        restarted.cancel(expiring_id)
    assert finished.value.code == "job_state_conflict"
    assert harness.sender.calls == []
    _assert_value_free(harness, restarted)


def test_cancelling_a_running_job_fences_its_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    queue = harness.start()
    job_id = queue.submit(_request())["id"]
    claimed = queue._claim(job_id)
    assert claimed is not None
    lease, record = claimed

    queue.cancel(job_id)

    with pytest.raises(StaleLeaseError):
        queue._commit_checkpoint(
            lease,
            jobs._JobSummary.restore(record),
            sequence=1,
            error=None,
            output_digest="sha256:" + "d" * 64,
        )
    final = queue.get(job_id)
    assert final["status"] == "cancelled"
    assert final["recovery"]["lease"] is None
    assert final["processed_count"] == 0


def test_durable_shutdown_keeps_queued_work_and_refunds_drained_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    queue = harness.start()
    queued_id = queue.submit(_request())["id"]
    running_id = queue.submit(_request())["id"]
    queue._executor.work.clear()
    claimed = queue._claim(running_id)
    assert claimed is not None
    queue._active[running_id] = claimed[0].epoch

    queue.shutdown()

    assert queue.get(queued_id)["status"] == "queued"
    draining = queue.get(running_id)
    assert draining["status"] == "running"
    assert draining["recovery"]["lease_draining"] is True
    with pytest.raises(RuntimeError, match="shutting down"):
        queue.submit(_request())

    harness.clock.advance(LEASE_SECONDS + 1)
    restarted = harness.start()
    reclaimed = restarted.get(running_id)
    assert reclaimed["status"] == "queued"
    assert reclaimed["recovery"]["attempt"] == 0
    assert reclaimed["recovery"]["retry"]["reason_code"] == "interrupted_by_shutdown"
    restarted.reconcile()
    assert drain(restarted) is False
    for job_id in (queued_id, running_id):
        _assert_single_logical_completion(restarted.get(job_id))
    _assert_value_free(harness, restarted)


@pytest.mark.parametrize(
    ("corrupt", "code"),
    [
        (
            lambda record: record["recovery"].update(completed_documents=99),
            "recovery_state_invalid",
        ),
        (
            lambda record: record["recovery"].update(lease_epoch="1"),
            "recovery_state_invalid",
        ),
        (
            lambda record: (
                record["recovery"]["transitions"].clear()
                or record["recovery"].update(execution=2)
            ),
            "recovery_state_invalid",
        ),
        (lambda record: record.update(status="paused"), "recovery_state_invalid"),
        (lambda record: record.pop("recovery"), "legacy_job_not_resumable"),
    ],
)
def test_corrupt_recovery_state_fails_closed(
    corrupt: Callable[[dict[str, Any]], Any],
    code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    queue = harness.start()
    job_id = queue.submit(_request())["id"]
    document = json.loads(harness.store_path.read_text(encoding="utf-8"))
    corrupt(document["jobs"][job_id])
    harness.store_path.write_text(json.dumps(document), encoding="utf-8")

    restarted = harness.start()
    record = restarted.get(job_id)
    assert record["status"] == "failed"
    assert record["error"] == {
        "code": code,
        "type": "JobRecoveryError",
        "message": "Job recovery state could not be trusted",
    }
    assert record["webhook_delivery"] is None
    assert record["payloads"]["input"]["state"] == "deleted"
    assert list((tmp_path / "payloads").iterdir()) == []
    assert restarted._executor.work == []
    receipt = restarted.decision_receipts()[-1]
    assert receipt["action"] == "recovery_failed"
    assert receipt["reason_code"] == code
    assert harness.sender.calls == []
    _assert_value_free(harness, restarted)


def test_idempotent_submission_survives_restart_and_detects_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    key = "0f3c1a52-5d4e-4c4f-9a65-2d4b8b8f6e10"
    queue = harness.start()
    first = queue.submit_job(_request(), idempotency_key=key)
    again = queue.submit_job(_request(), idempotency_key=key)
    assert first.created is True and again.created is False
    assert again.record["id"] == first.record["id"]
    assert queue.payloads.retained_count() == 1
    assert len(queue._executor.work) == 1

    with pytest.raises(IdempotencyConflictError):
        queue.submit_job(_request(method="remove"), idempotency_key=key)
    # The date shift never enters metadata; the retained snapshot still
    # detects that its value changed.
    shift_key = "7d2b6f0e-93a1-4e58-b4d3-0c6a9e1f2a77"
    queue.submit_job(
        _request(method="shift_dates", date_shift_days=7),
        idempotency_key=shift_key,
    )
    with pytest.raises(IdempotencyConflictError):
        queue.submit_job(
            _request(method="shift_dates", date_shift_days=11),
            idempotency_key=shift_key,
        )
    scoped = queue.submit_job(_request(), idempotency_key=key, idempotency_scope="b")
    assert scoped.created is True and scoped.record["id"] != first.record["id"]
    with pytest.raises(IdempotencyKeyError):
        queue.submit_job(_request(), idempotency_key="short")
    assert isinstance(IdempotencyKeyError(), ValueError)

    restarted = harness.start()
    replayed = restarted.submit_job(_request(), idempotency_key=key)
    assert replayed.created is False and replayed.record["id"] == first.record["id"]
    stored = json.loads(harness.store_path.read_text(encoding="utf-8"))
    assert key not in json.dumps(stored)
    digests = {
        record["idempotency"]["key_digest"]
        for record in stored["jobs"].values()
        if record.get("idempotency")
    }
    assert len(digests) == 3
    restarted.reconcile()
    assert drain(restarted) is False
    done = restarted.submit_job(_request(), idempotency_key=key)
    assert done.created is False and done.record["status"] == "done"
    # After the input is consumed, the metadata digest still detects changes.
    with pytest.raises(IdempotencyConflictError):
        restarted.submit_job(_request(keep_year=True), idempotency_key=key)
    _assert_value_free(harness, restarted)


def test_killed_child_process_resumes_after_the_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real process kill (``os._exit``) after the first checkpoint commit."""

    state = tmp_path / "state"
    state.mkdir()
    script = tmp_path / "scripts" / "child.py"
    script.parent.mkdir()
    script.write_text(
        textwrap.dedent(
            f"""
            import os
            import sys
            from pathlib import Path
            from types import SimpleNamespace

            from openmed.core.key_provider import InMemoryKeyProvider, KeyState
            from openmed.service import jobs
            from openmed.service.job_payloads import (
                JOB_PAYLOAD_KEY_PURPOSE,
                EncryptedJobPayloadStore,
            )
            from openmed.service.schemas import DeidentifyJobRequest

            root = Path(sys.argv[1])
            provider = InMemoryKeyProvider()
            provider.add_key(
                "recovery-key-1",
                bytes(range(32)),
                purpose=JOB_PAYLOAD_KEY_PURPOSE,
                state=KeyState.ACTIVE,
            )

            def kill(boundary, _job_id):
                if boundary == "checkpoint.committed":
                    os._exit(17)

            queue = jobs.DeidentifyJobQueue(
                SimpleNamespace(validate_served_model=lambda _name: None),
                store=jobs.LocalJobStore(root / "jobs.json"),
                payload_store=EncryptedJobPayloadStore(
                    root / "payloads", key_provider=provider
                ),
                max_workers=1,
                lease_seconds={LEASE_SECONDS},
                fault_hook=kill,
            )
            queue._deidentify_document = lambda _request, document: (
                SimpleNamespace(pii_entities=[])
            )
            queue.submit(
                DeidentifyJobRequest(
                    documents=[
                        {{"id": "d0", "text": {TEXT_SENTINELS[0]!r}}},
                        {{"id": "d1", "text": {TEXT_SENTINELS[1]!r}}},
                    ]
                )
            )
            queue._executor.shutdown(wait=True)
            os._exit(3)
            """
        ),
        encoding="utf-8",
    )
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    completed = subprocess.run(
        [sys.executable, str(script), str(state)],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 17, completed.stderr.decode("utf-8", "replace")
    _assert_no_sentinels(completed.stdout + completed.stderr)

    harness = Harness(state, monkeypatch)
    harness.clock.now = datetime.now(timezone.utc) + timedelta(
        seconds=LEASE_SECONDS + 1
    )
    job_id = _only_job_id(harness)
    killed = jobs.LocalJobStore(harness.store_path, clock=harness.clock).get(job_id)
    assert killed["status"] == "running"
    assert killed["recovery"]["completed_documents"] == 1

    queue = recover(harness, job_id)
    record = queue.get(job_id)
    assert record["status"] == "done"
    assert record["processed_count"] == 2
    assert harness.model.calls == ["d1"]
    assert record["recovery"]["lease_epoch"] == 2
    _assert_value_free(harness, queue)


def crash_plan(plan: list[str], fired: list[str]) -> Callable[[str, str], None]:
    """Return a fault hook that crashes at each planned boundary in order."""

    remaining = list(plan)

    def hook(hit: str, _job_id: str) -> None:
        if remaining and hit == remaining[0]:
            fired.append(remaining.pop(0))
            raise SimulatedCrash(hit)

    return hook


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(plan=st.lists(st.sampled_from(FAULT_BOUNDARIES), max_size=3))
def test_any_crash_sequence_ends_in_one_logical_completion(
    plan: list[str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Property: crashes at any boundaries, across restarts, never duplicate."""

    fired: list[str] = []
    hook = crash_plan(plan, fired)
    policy = RetryPolicy(max_attempts=5, base_delay_seconds=5, max_delay_seconds=60)
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = Harness(tmp_path_factory.mktemp("plan"), monkeypatch)
        first = harness.start(fault=hook, retry_policy=policy)
        try:
            job_id = first.submit(_request())["id"]
        except SimulatedCrash:
            job_id = _only_job_id(harness)
        else:
            drain(first)
        queue = recover(harness, job_id, rounds=30, fault=hook, retry_policy=policy)
        record = queue.get(job_id)

        _assert_single_logical_completion(record)
        assert record["webhook_delivery"]["success"] is True
        bodies = [call["payload"] for call in harness.sender.calls]
        assert len(bodies) == 1 + fired.count("webhook.sent")
        assert all(body == bodies[0] for body in bodies)
        assert list((harness.root / "payloads").iterdir()) == []
        _assert_value_free(harness, queue)
