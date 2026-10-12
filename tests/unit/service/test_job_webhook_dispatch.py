"""Tests for off-worker terminal webhook delivery in the async job queue."""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import httpx
import pytest

from openmed.service import jobs
from openmed.service.signing import NonceCache, verify_request_signature
from openmed.service.webhooks import (
    EVENT_HEADER,
    TIMESTAMP_HEADER,
    canonical_json_bytes,
    deliver_webhook,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SENTINEL = "SYNTHETIC-SENTINEL-3899"
RECEIVER_URL = f"https://receiver.invalid/{SENTINEL}/hook?token={SENTINEL}"
WEBHOOK_SECRET = "synthetic-webhook-secret"
SCHEMA_MAX_WEBHOOK = {
    "url": RECEIVER_URL,
    "secret": WEBHOOK_SECRET,
    "max_attempts": 10,
    "backoff_seconds": 60.0,
}
WAIT_SECONDS = 5.0


def _wait_until(predicate: Callable[[], bool], timeout: float = WAIT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _make_queue(
    tmp_path: Path,
    sender: Callable[..., Any],
    **kwargs: Any,
) -> tuple[jobs.DeidentifyJobQueue, jobs.LocalJobStore]:
    store = jobs.LocalJobStore(tmp_path / "jobs.json")
    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(validate_served_model=lambda _name: None),
        store=store,
        webhook_sender=sender,
        **kwargs,
    )
    queue._deidentify_document = lambda _payload, _document: SimpleNamespace(
        pii_entities=[]
    )
    return queue, store


def _request(webhook: Optional[dict[str, Any]] = None) -> Any:
    from openmed.service.schemas import DeidentifyJobRequest

    return DeidentifyJobRequest(
        documents=[{"id": "synthetic-0", "text": "synthetic note"}],
        webhook=webhook,
    )


def _delivery(store: jobs.LocalJobStore, job_id: str) -> Optional[dict[str, Any]]:
    record = store.get(job_id)
    return None if record is None else record.get("webhook_delivery")


def _settled(store: jobs.LocalJobStore, job_id: str) -> bool:
    delivery = _delivery(store, job_id)
    return delivery is not None and delivery.get("status") != "pending"


PENDING = {
    "status": "pending",
    "success": None,
    "attempts": 0,
    "status_code": None,
    "error": None,
}


def test_failing_receiver_with_schema_max_retries_does_not_block_queued_jobs(
    tmp_path: Path,
) -> None:
    attempted = threading.Event()
    release = threading.Event()
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempted.set()
        return httpx.Response(500)

    def sleeper(delay: float) -> None:
        delays.append(delay)
        release.wait(10)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = functools.partial(deliver_webhook, client=client, sleeper=sleeper)
    queue, store = _make_queue(tmp_path, sender, max_workers=1)
    try:
        failing = queue.submit(_request(SCHEMA_MAX_WEBHOOK))
        other = queue.submit(_request())

        assert attempted.wait(WAIT_SECONDS)
        assert _wait_until(lambda: store.get(other["id"])["status"] == "done")
        record = store.get(failing["id"])
        assert record is not None
        assert record["status"] == "done"
        assert record["webhook_delivery"] == PENDING
    finally:
        release.set()
        queue.shutdown()


def test_shutdown_returns_within_drain_bound_while_delivery_backs_off(
    tmp_path: Path,
) -> None:
    backing_off = threading.Event()
    release = threading.Event()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    def sleeper(_delay: float) -> None:
        backing_off.set()
        release.wait(10)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = functools.partial(deliver_webhook, client=client, sleeper=sleeper)
    queue, store = _make_queue(tmp_path, sender, webhook_drain_seconds=0.2)
    try:
        created = queue.submit(_request(SCHEMA_MAX_WEBHOOK))
        assert backing_off.wait(WAIT_SECONDS)

        started = time.monotonic()
        queue.shutdown()
        elapsed = time.monotonic() - started

        assert elapsed < 1.5
        assert _delivery(store, created["id"]) == {
            "status": "abandoned",
            "success": False,
            "attempts": 0,
            "status_code": None,
            "error": "cancelled",
        }
    finally:
        release.set()
        queue.shutdown()


_EXIT_CHILD = textwrap.dedent(
    """
    import functools
    import sys
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    import httpx

    from openmed.service import jobs
    from openmed.service.schemas import DeidentifyJobRequest
    from openmed.service.webhooks import deliver_webhook

    attempted = threading.Event()


    def handler(_request):
        attempted.set()
        return httpx.Response(500)


    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(validate_served_model=lambda _name: None),
        store=jobs.LocalJobStore(Path(sys.argv[1])),
        max_workers=1,
        webhook_sender=functools.partial(
            deliver_webhook,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    )
    queue._deidentify_document = lambda _payload, _document: SimpleNamespace(
        pii_entities=[]
    )
    queue.submit(
        DeidentifyJobRequest(
            documents=[{"id": "synthetic-0", "text": "synthetic note"}],
            webhook={
                "url": "https://receiver.invalid/hook",
                "secret": "synthetic-webhook-secret",
                "max_attempts": 10,
                "backoff_seconds": 60.0,
            },
        )
    )
    if not attempted.wait(30):
        sys.exit(3)
    print("shutdown", flush=True)
    queue.shutdown()
    """
)


def test_process_exits_within_drain_bound_while_delivery_backs_off(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", _EXIT_CHILD, str(tmp_path / "jobs.json")],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "shutdown"
        started = time.monotonic()
        try:
            returncode = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail("process stayed alive while a webhook was backing off")
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()

    assert returncode == 0
    assert elapsed < 5.0


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            httpx.ConnectError(f"refused {SENTINEL} {RECEIVER_URL}"),
            {"status_code": None, "error": "connect"},
        ),
        (
            httpx.ReadTimeout(f"timed out {SENTINEL} {RECEIVER_URL}"),
            {"status_code": None, "error": "timeout"},
        ),
        (None, {"status_code": 500, "error": "http_status"}),
    ],
    ids=["connect", "timeout", "http_status"],
)
def test_delivery_metadata_contains_only_controlled_codes(
    tmp_path: Path,
    failure: Optional[Exception],
    expected: dict[str, Any],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure is not None:
            raise failure
        return httpx.Response(500, text=f"{SENTINEL} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = functools.partial(deliver_webhook, client=client, sleeper=lambda _d: None)
    queue, store = _make_queue(tmp_path, sender)
    try:
        created = queue.submit(
            _request({**SCHEMA_MAX_WEBHOOK, "max_attempts": 2, "backoff_seconds": 0})
        )
        assert _wait_until(lambda: _settled(store, created["id"]))
    finally:
        queue.shutdown()

    assert _delivery(store, created["id"]) == {
        "status": "failed",
        "success": False,
        "attempts": 2,
        **expected,
    }
    stored = (tmp_path / "jobs.json").read_text(encoding="utf-8")
    assert SENTINEL not in stored
    assert "receiver.invalid" not in stored
    assert WEBHOOK_SECRET not in stored


def test_sender_exception_text_is_not_stored(tmp_path: Path) -> None:
    def sender(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{SENTINEL} {RECEIVER_URL}")

    queue, store = _make_queue(tmp_path, sender)
    try:
        created = queue.submit(_request(SCHEMA_MAX_WEBHOOK))
        assert _wait_until(lambda: _settled(store, created["id"]))
    finally:
        queue.shutdown()

    assert _delivery(store, created["id"]) == {
        "status": "failed",
        "success": False,
        "attempts": 0,
        "status_code": None,
        "error": "sender_error",
    }
    assert SENTINEL not in (tmp_path / "jobs.json").read_text(encoding="utf-8")


def test_successful_delivery_is_pending_until_settled_and_keeps_signed_bytes(
    tmp_path: Path,
) -> None:
    received = threading.Event()
    release = threading.Event()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        received.set()
        release.wait(10)
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = functools.partial(deliver_webhook, client=client)
    queue, store = _make_queue(tmp_path, sender)
    url = "https://receiver.invalid/openmed/jobs?source=synthetic"
    try:
        created = queue.submit(
            _request({"url": url, "secret": WEBHOOK_SECRET, "max_attempts": 3})
        )
        assert received.wait(WAIT_SECONDS)
        pending = store.get(created["id"])
        assert pending is not None
        assert pending["status"] == "done"
        assert pending["webhook_delivery"] == PENDING

        release.set()
        assert _wait_until(lambda: _settled(store, created["id"]))
    finally:
        release.set()
        queue.shutdown()

    final = store.get(created["id"])
    assert final is not None
    assert final["webhook_delivery"] == {
        "status": "delivered",
        "success": True,
        "attempts": 1,
        "status_code": 204,
        "error": None,
    }
    assert len(requests) == 1
    request = requests[0]
    assert request.content == canonical_json_bytes(jobs.webhook_payload(final))
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers[EVENT_HEADER] == "job.done"
    assert verify_request_signature(
        "POST",
        "/openmed/jobs?source=synthetic",
        request.content,
        request.headers,
        secret=WEBHOOK_SECRET,
        nonce_cache=NonceCache(),
        now=int(request.headers[TIMESTAMP_HEADER]),
    )


def test_total_delivery_wall_time_ceiling_settles_abandoned(tmp_path: Path) -> None:
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = functools.partial(deliver_webhook, client=client, sleeper=delays.append)
    queue, store = _make_queue(tmp_path, sender, webhook_max_delivery_seconds=5.0)
    try:
        created = queue.submit(_request(SCHEMA_MAX_WEBHOOK))
        assert _wait_until(lambda: _settled(store, created["id"]))
    finally:
        queue.shutdown()

    assert delays == []
    assert _delivery(store, created["id"]) == {
        "status": "abandoned",
        "success": False,
        "attempts": 1,
        "status_code": 500,
        "error": "deadline_exceeded",
    }


def test_delivery_scheduled_after_shutdown_settles_abandoned(tmp_path: Path) -> None:
    calls: list[str] = []

    def sender(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("called")
        raise AssertionError("sender must not run after shutdown")

    queue, store = _make_queue(tmp_path, sender)
    payload = _request(SCHEMA_MAX_WEBHOOK)
    record = queue._new_record(payload)
    store.create(record)
    queue.shutdown()

    queue._run_job(jobs._JobWorkItem(record["id"], payload))

    assert calls == []
    assert _delivery(store, record["id"]) == {
        "status": "abandoned",
        "success": False,
        "attempts": 0,
        "status_code": None,
        "error": "cancelled",
    }
