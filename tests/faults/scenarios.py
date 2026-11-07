"""Component adapters that drive existing service code through fault seams.

Each component declares its commit point, the ``(fault, injection_point)``
pairs it supports and the baseline outcome of an unfaulted run. A scenario
runs one case in one session and returns an :class:`Observation` made of a
stable outcome category, a controlled reason code and a residue count (files
or reservations left behind that no committed state references). Synthetic
canaries are used for every payload so tests can prove none reaches evidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from starlette.responses import Response

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
)
from openmed.clinical.journey_contracts import sha256_digest as journey_digest
from openmed.core.offline import HF_OFFLINE_ENV_VARS, network_blocked_if_offline
from openmed.service import jobs
from openmed.service.resilience import (
    CircuitBreakerOpenError,
    ResilienceManager,
    ServiceResilienceConfig,
)
from openmed.service.runtime import ServiceThrottleConfig
from openmed.service.schemas import (
    DeidentifyJobDocument,
    DeidentifyJobRequest,
    JobWebhookRequest,
)
from openmed.service.signing import (
    NonceCache,
    ReplayDetectedError,
    StaleTimestampError,
    sign_request,
    verify_request_signature,
)
from openmed.service.throttle import ConcurrencyGate, ServiceThrottle
from openmed.service.webhooks import canonical_json_bytes, deliver_webhook
from openmed.structured.store import (
    LocalArtifactStore,
    LocalJourneyStore,
    SQLiteJourneyStore,
    StoreState,
)

from .plan import AMBIGUOUS, FaultCase, FaultPlanError
from .seams import (
    CLOCK_MONOTONIC,
    CLOCK_WALL,
    DISK_FSYNC,
    DISK_LINK,
    DISK_READ,
    DISK_REPLACE,
    DISK_WRITE,
    NETWORK_EGRESS,
    NETWORK_REQUEST,
    SQLITE_COMMIT,
    STATE_RESTART,
    FaultClock,
    FaultSession,
    FaultTransport,
    LoopbackHTTPReceiver,
    SimulatedProcessKill,
    disk_faults,
    sqlite_commit_faults,
)

# Synthetic canaries: every one must stay out of manifests and durable metadata.
TEXT_CANARY = "SYNTHETIC-PHI-CANARY Maria Example 555-0100"
SIGNING_CANARY = "synthetic-signing-canary-aaaaaaaa"
URL_CANARY = "http://127.0.0.1:9/hooks/synthetic-url-canary"
CONTENT_CANARY = b"SYNTHETIC-JOURNEY-CANARY clinical note bytes"
CANARIES = (TEXT_CANARY, SIGNING_CANARY, "synthetic-url-canary", "SYNTHETIC-JOURNEY")

WALL_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
COMMITTED_AT = "2026-01-02T03:04:05Z"


def _fixed_wall() -> datetime:
    """Return the fixed scenario wall time, so no outcome depends on today."""
    return WALL_START


@dataclass(frozen=True)
class Observation:
    """Value-free result of one scenario run."""

    outcome: str
    reason_code: str
    residue: int = 0


@dataclass(frozen=True)
class Component:
    """One documented adapter boundary that plans may target."""

    name: str
    commit_point: str
    supported: frozenset[tuple[str, str]]
    run: Callable[[FaultSession], Observation]
    baseline: Callable[[FaultCase], str]


COMPONENTS: dict[str, Component] = {}


def _register(
    name: str,
    *,
    commit_point: str,
    supported: set[tuple[str, str]],
    baseline: Callable[[FaultCase], str] | str = "committed_exactly_once",
) -> Callable[[Callable[[FaultSession], Observation]], Callable[..., Observation]]:
    def decorator(
        run: Callable[[FaultSession], Observation],
    ) -> Callable[[FaultSession], Observation]:
        resolve = baseline if callable(baseline) else (lambda _case: baseline)
        COMPONENTS[name] = Component(
            name=name,
            commit_point=commit_point,
            supported=frozenset(supported),
            run=run,
            baseline=resolve,
        )
        return run

    return decorator


def component_for(case: FaultCase) -> Component:
    """Return the registered component a case targets.

    Raises:
        FaultPlanError: When the component or fault/point pair is unsupported.
    """
    component = COMPONENTS.get(case.component)
    if component is None:
        raise FaultPlanError("fault_component_unknown")
    if (case.fault, case.injection_point) not in component.supported:
        raise FaultPlanError("fault_case_unsupported")
    return component


def _files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


@contextmanager
def _preserved_environ(names: tuple[str, ...]) -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------------------
# service.jobs.local_store: atomic temp-file + os.replace publication
# ---------------------------------------------------------------------------

_JOB_ID = "synthetic_job_aaaaaaaaaaaaaaaa"
_RUNNING_JOB_ID = "synthetic_job_bbbbbbbbbbbbbbbb"


def _jobs_store_baseline(case: FaultCase) -> str:
    return (
        "no_effect" if case.injection_point == CLOCK_WALL else "committed_exactly_once"
    )


@_register(
    "service.jobs.local_store",
    commit_point=DISK_REPLACE,
    supported={
        ("enospc", DISK_WRITE),
        ("eio", DISK_WRITE),
        ("partial_write", DISK_WRITE),
        ("process_kill", DISK_WRITE),
        ("fsync_failure", DISK_FSYNC),
        ("enospc", DISK_REPLACE),
        ("eio", DISK_REPLACE),
        ("process_kill", DISK_REPLACE),
        ("cancellation", DISK_REPLACE),
        ("eio", DISK_READ),
        ("truncated_state", STATE_RESTART),
        ("garbled_state", STATE_RESTART),
        ("clock_skew", CLOCK_WALL),
        ("clock_jump_forward", CLOCK_WALL),
        ("clock_jump_backward", CLOCK_WALL),
    },
    baseline=_jobs_store_baseline,
)
def run_jobs_local_store(session: FaultSession) -> Observation:
    """Publish one job-status change and restart the store."""
    if session.case.injection_point == CLOCK_WALL:
        return _jobs_store_clock(session)
    path = session.state_dir / "jobs.json"
    with session.suspended():
        store = jobs.LocalJobStore(path, clock=_fixed_wall)
        store.create({"id": _JOB_ID, "status": "queued", "expires_at": None})
        committed_bytes = path.read_bytes()
        committed_records = json.loads(committed_bytes)["jobs"]

    interrupted: str | None = None
    with disk_faults(session, session.state_dir):
        try:
            store.update(_JOB_ID, status="running")
        except OSError:
            interrupted = "os_error"
        except SimulatedProcessKill:
            interrupted = "process_kill"
        except asyncio.CancelledError:
            interrupted = "cancelled"

        if interrupted in {"os_error", "cancelled"}:
            with session.suspended():
                unchanged = (
                    store._records == committed_records  # noqa: SLF001
                    and path.read_bytes() == committed_bytes
                )
                residue = len(_files(session.state_dir)) - 1
            if not unchanged:
                return Observation(AMBIGUOUS, "publish_partially_applied", residue)
            return Observation("rolled_back", "publish_rolled_back", residue)

        root = session.crash_dir if interrupted == "process_kill" else session.state_dir
        restart_path = root / "jobs.json"
        corrupted = session.checkpoint(STATE_RESTART, target=restart_path)
        with session.suspended():
            restart_bytes = restart_path.read_bytes()
        read_failures = 0
        reopened: jobs.LocalJobStore | None = None
        for _attempt in range(2):
            try:
                reopened = jobs.LocalJobStore(restart_path, clock=_fixed_wall)
                break
            except OSError:
                read_failures += 1
            except ValueError:
                with session.suspended():
                    preserved = restart_path.read_bytes() == restart_bytes
                    residue = len(_files(root)) - 1
                if corrupted and preserved:
                    return Observation("quarantined", "store_rejected_corrupt", residue)
                return Observation(AMBIGUOUS, "store_rejected_unexpectedly", residue)

    with session.suspended():
        residue = len(_files(root)) - 1
    if reopened is None:
        return Observation(AMBIGUOUS, "store_unreadable", residue)
    record = reopened.get(_JOB_ID)
    status = None if record is None else record.get("status")
    if read_failures and status == "running":
        return Observation("retryable", "store_read_retried", residue)
    if status == "running":
        return Observation("committed_exactly_once", "publish_committed", residue)
    if status == "queued" and interrupted == "process_kill":
        return Observation("no_effect", "restart_previous_state", residue)
    return Observation(AMBIGUOUS, "restart_state_unexpected", residue)


def _jobs_store_clock(session: FaultSession) -> Observation:
    clock = FaultClock(session, wall_start=WALL_START, tick_seconds=1.0)
    path = session.state_dir / "jobs.json"
    expires_at = (WALL_START + timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
    with session.suspended():
        store = jobs.LocalJobStore(path, ttl_seconds=60, clock=clock.wall)
        store.create({"id": _JOB_ID, "status": "done", "expires_at": expires_at})
        store.create(
            {"id": _RUNNING_JOB_ID, "status": "running", "expires_at": expires_at}
        )
    store.cleanup_expired()
    with session.suspended():
        reopened = jobs.LocalJobStore(path, ttl_seconds=60, clock=_fixed_wall)
        terminal_kept = reopened.get(_JOB_ID) is not None
        running_kept = reopened.get(_RUNNING_JOB_ID) is not None
        in_memory = sorted(store._records)  # noqa: SLF001
        on_disk = sorted(reopened._records)  # noqa: SLF001
    if not running_kept or in_memory != on_disk:
        return Observation(AMBIGUOUS, "non_terminal_record_expired")
    if terminal_kept:
        return Observation("no_effect", "nothing_expired")
    return Observation("committed_exactly_once", "terminal_record_expired_once")


# ---------------------------------------------------------------------------
# service.jobs.queue: worker lifecycle and terminal webhook
# ---------------------------------------------------------------------------


class _ControlledExecutor:
    """Deterministic stand-in for the job queue's thread pool."""

    def __init__(self, **_kwargs: Any) -> None:
        self.work: list[tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]] = []

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        self.work.append((future, fn, args))
        return future

    def run_pending(self) -> None:
        for future, fn, args in self.work:
            if future.set_running_or_notify_cancel():
                future.set_result(fn(*args))

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        if cancel_futures:
            for future, _fn, _args in self.work:
                future.cancel()


def _job_payload(*, url: str = URL_CANARY) -> DeidentifyJobRequest:
    return DeidentifyJobRequest(
        documents=[DeidentifyJobDocument(id="synthetic-doc-0", text=TEXT_CANARY)],
        webhook=JobWebhookRequest(
            url=url,
            secret=SIGNING_CANARY,
            max_attempts=3,
            backoff_seconds=0.5,
        ),
    )


def _fake_deidentify(_payload: Any, document: DeidentifyJobDocument) -> Any:
    entity = SimpleNamespace(
        label="NAME",
        text=document.text[:12],
        start=0,
        end=12,
        confidence=0.9,
    )
    return SimpleNamespace(pii_entities=[entity])


def _canary_persisted(directory: Path) -> bool:
    for path in _files(directory):
        data = path.read_bytes()
        if any(canary.encode("utf-8") in data for canary in CANARIES):
            return True
    return False


def _restart_record(path: Path, job_id: str) -> dict[str, Any]:
    """Open the store and queue as a fresh process would and read one job."""
    store = jobs.LocalJobStore(path, clock=_fixed_wall)
    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(),
        store=store,
        clock=_fixed_wall,
    )
    try:
        return queue.get(job_id) or {}
    finally:
        queue.shutdown()


@_register(
    "service.jobs.queue",
    commit_point="jobs.terminal_update",
    supported={
        ("connection_refused", NETWORK_REQUEST),
        ("timeout", NETWORK_REQUEST),
        ("connection_reset", NETWORK_REQUEST),
        ("process_kill", "model.call"),
        ("cancellation", "model.call"),
        ("cancellation", "queue.shutdown"),
        ("enospc", DISK_WRITE),
    },
)
def run_jobs_queue(session: FaultSession) -> Observation:
    """Run one job through the worker and observe the restarted record."""
    if session.case.injection_point == "queue.shutdown":
        return _jobs_queue_shutdown(session)
    path = session.state_dir / "jobs.json"
    transport = FaultTransport(session)
    client = httpx.Client(transport=transport)
    sleeps: list[float] = []

    def sender(url: str, payload: dict[str, Any], **kwargs: Any) -> Any:
        return deliver_webhook(
            url, payload, client=client, sleeper=sleeps.append, **kwargs
        )

    with session.suspended():
        store = jobs.LocalJobStore(path, clock=_fixed_wall)
        queue = jobs.DeidentifyJobQueue(
            SimpleNamespace(),
            store=store,
            webhook_sender=sender,
            clock=_fixed_wall,
        )
        payload = _job_payload()
        record = queue._new_record(payload)  # noqa: SLF001
        store.create(record)

    def model_call(job_payload: Any, document: DeidentifyJobDocument) -> Any:
        return session.call("model.call", _fake_deidentify, job_payload, document)

    queue._deidentify_document = model_call  # noqa: SLF001
    interrupted = False
    try:
        with disk_faults(session, session.state_dir):
            queue._run_job(jobs._JobWorkItem(record["id"], payload))  # noqa: SLF001
    except (OSError, SimulatedProcessKill, asyncio.CancelledError):
        interrupted = True
    finally:
        queue.shutdown()
        client.close()

    with session.suspended():
        root = session.crash_dir if session.crashed else session.state_dir
        if _canary_persisted(root):
            return Observation(AMBIGUOUS, "canary_persisted")
        persisted = _restart_record(root / "jobs.json", record["id"])
    status = persisted.get("status")
    if status not in jobs.TERMINAL_STATUSES:
        reason = "job_left_non_terminal" if interrupted else "job_not_terminal"
        return Observation(AMBIGUOUS, reason)
    delivery = persisted.get("webhook_delivery") or {}
    attempts = int(delivery.get("attempts") or 0)
    expected_sleeps = [0.5 * (2**index) for index in range(attempts - 1)]
    if status != "done" or sleeps != expected_sleeps or attempts > 3:
        return Observation(AMBIGUOUS, "terminal_webhook_unexpected")
    if delivery.get("success") and transport.received == 1:
        return Observation("committed_exactly_once", "terminal_webhook_delivered")
    if delivery.get("success") and transport.received == attempts:
        return Observation("committed_exactly_once", "terminal_webhook_redelivered")
    if delivery.get("success") is False and attempts == 3:
        return Observation("committed_exactly_once", "terminal_webhook_failed_bounded")
    return Observation(AMBIGUOUS, "terminal_webhook_unexpected")


def _jobs_queue_shutdown(session: FaultSession) -> Observation:
    path = session.state_dir / "jobs.json"
    runtime = SimpleNamespace(validate_served_model=lambda _name: None)
    original_executor = jobs.ThreadPoolExecutor
    jobs.ThreadPoolExecutor = _ControlledExecutor
    try:
        with session.suspended():
            store = jobs.LocalJobStore(path, clock=_fixed_wall)
            queue = jobs.DeidentifyJobQueue(
                runtime,
                store=store,
                webhook_sender=lambda *_args, **_kwargs: jobs.WebhookDeliveryResult(
                    success=True, attempts=1, status_code=204
                ),
                clock=_fixed_wall,
            )
    finally:
        jobs.ThreadPoolExecutor = original_executor
    queue._deidentify_document = _fake_deidentify  # noqa: SLF001
    executor: _ControlledExecutor = queue._executor  # noqa: SLF001
    submitted = queue.submit(_job_payload())
    if not session.trip("queue.shutdown"):
        executor.run_pending()
    queue.shutdown()
    with session.suspended():
        if _canary_persisted(session.state_dir):
            return Observation(AMBIGUOUS, "canary_persisted")
        record = _restart_record(path, submitted["id"])
    if record.get("status") in jobs.TERMINAL_STATUSES:
        return Observation("committed_exactly_once", "job_terminal_once")
    return Observation(AMBIGUOUS, "cancelled_job_left_non_terminal")


# ---------------------------------------------------------------------------
# service.webhooks.delivery: signed POST with bounded retries
# ---------------------------------------------------------------------------


@_register(
    "service.webhooks.delivery",
    commit_point="webhook.receiver_accept",
    supported={
        ("connection_refused", NETWORK_REQUEST),
        ("timeout", NETWORK_REQUEST),
        ("connection_reset", NETWORK_REQUEST),
    },
)
def run_webhook_delivery(session: FaultSession) -> Observation:
    """Deliver one signed webhook through the fault transport."""
    transport = FaultTransport(session)
    sleeps: list[float] = []
    payload = {"event": "job.done", "job_id": _JOB_ID, "status": "done"}
    with httpx.Client(transport=transport) as client:
        result = deliver_webhook(
            URL_CANARY,
            payload,
            secret=SIGNING_CANARY,
            max_attempts=3,
            backoff_seconds=0.5,
            client=client,
            sleeper=sleeps.append,
        )
    expected_sleeps = [0.5 * (2**index) for index in range(result.attempts - 1)]
    if sleeps != expected_sleeps or result.attempts > 3:
        return Observation(AMBIGUOUS, "retry_schedule_unbounded")
    if set(transport.body_digests) - {
        transport.body_digests[0] if transport.body_digests else ""
    }:
        return Observation(AMBIGUOUS, "retry_body_changed")
    if len(set(transport.nonce_digests)) != len(transport.nonce_digests):
        return Observation(AMBIGUOUS, "retry_nonce_reused")
    if result.success and transport.received == 1:
        return Observation("committed_exactly_once", "delivered_once")
    if transport.received == 0 and not result.success:
        return Observation("retryable", "delivery_failed_unsent")
    if transport.received >= 1:
        return Observation("retryable", "delivery_repeated_fresh_nonce")
    return Observation(AMBIGUOUS, "delivery_unexpected")


# ---------------------------------------------------------------------------
# core.offline.network_guard: process-level socket denial
# ---------------------------------------------------------------------------


def _guard_baseline(case: FaultCase) -> str:
    return (
        "rolled_back"
        if case.injection_point == "offline.scope"
        else ("committed_exactly_once")
    )


@_register(
    "core.offline.network_guard",
    commit_point=NETWORK_EGRESS,
    supported={("egress_denied", NETWORK_EGRESS), ("cancellation", "offline.scope")},
    baseline=_guard_baseline,
)
def run_offline_guard(session: FaultSession) -> Observation:
    """Deny or cancel egress through the product offline guard."""
    if session.case.injection_point == "offline.scope":
        return _offline_guard_cancellation(session)
    path = session.state_dir / "jobs.json"
    original_create_connection = socket.create_connection
    with _preserved_environ(HF_OFFLINE_ENV_VARS), LoopbackHTTPReceiver() as receiver:

        def sender(url: str, payload: dict[str, Any], **kwargs: Any) -> Any:
            if session.trip(NETWORK_EGRESS):
                with network_blocked_if_offline(local_only=True):
                    return deliver_webhook(
                        url, payload, sleeper=lambda _s: None, **kwargs
                    )
            return deliver_webhook(url, payload, sleeper=lambda _s: None, **kwargs)

        with session.suspended():
            store = jobs.LocalJobStore(path, clock=_fixed_wall)
            queue = jobs.DeidentifyJobQueue(
                SimpleNamespace(),
                store=store,
                webhook_sender=sender,
                clock=_fixed_wall,
            )
            queue._deidentify_document = _fake_deidentify  # noqa: SLF001
            payload = _job_payload(url=receiver.url)
            record = queue._new_record(payload)  # noqa: SLF001
            store.create(record)
        try:
            queue._run_job(jobs._JobWorkItem(record["id"], payload))  # noqa: SLF001
        finally:
            queue.shutdown()
        received = receiver.received
    with session.suspended():
        persisted = _restart_record(path, record["id"])
        canary = _canary_persisted(session.state_dir)
    if canary:
        return Observation(AMBIGUOUS, "canary_persisted")
    if socket.create_connection is not original_create_connection:
        return Observation(AMBIGUOUS, "socket_guard_leaked")
    delivery = persisted.get("webhook_delivery") or {}
    if persisted.get("status") != "done":
        return Observation(AMBIGUOUS, "job_not_terminal")
    if delivery.get("success") and received == 1:
        return Observation("committed_exactly_once", "egress_delivered")
    if delivery.get("success") is False and delivery.get("attempts") == 0:
        if received == 0:
            return Observation("committed_exactly_once", "egress_blocked")
    return Observation(AMBIGUOUS, "egress_unexpected")


def _offline_guard_cancellation(session: FaultSession) -> Observation:
    original = (socket.socket.connect, socket.create_connection)
    cancelled = False
    still_blocked = False
    with _preserved_environ(HF_OFFLINE_ENV_VARS):
        with network_blocked_if_offline(local_only=True):
            try:
                with network_blocked_if_offline(local_only=True):
                    session.checkpoint("offline.scope")
            except asyncio.CancelledError:
                cancelled = True
            still_blocked = socket.create_connection is not original[1]
    restored = (socket.socket.connect, socket.create_connection) == original
    if not still_blocked or not restored:
        return Observation(AMBIGUOUS, "guard_scope_leaked")
    reason = "inner_scope_cancelled_outer_held" if cancelled else "scopes_unwound"
    return Observation("rolled_back", reason)


# ---------------------------------------------------------------------------
# service.signing.verify: timestamp window and nonce claim
# ---------------------------------------------------------------------------


@_register(
    "service.signing.verify",
    commit_point="signing.nonce_claim",
    supported={
        ("clock_skew", CLOCK_WALL),
        ("clock_jump_forward", CLOCK_WALL),
        ("clock_jump_backward", CLOCK_WALL),
    },
)
def run_signature_verify(session: FaultSession) -> Observation:
    """Verify one signed delivery and one replay under a fault clock."""
    issued_at = int(WALL_START.timestamp())
    body = canonical_json_bytes({"event": "job.done", "job_id": _JOB_ID})
    headers = sign_request(
        "POST",
        "/hooks/job",
        body,
        secret=SIGNING_CANARY,
        timestamp=issued_at,
        nonce="synthetic-nonce-aaaaaaaaaaaaaaaa",
    )
    cache = NonceCache(window_seconds=300)
    clock = FaultClock(session, wall_start=WALL_START, tick_seconds=1.0)
    accepted = 0
    for _delivery in range(2):
        try:
            verify_request_signature(
                "POST",
                "/hooks/job",
                body,
                headers,
                secret=SIGNING_CANARY,
                nonce_cache=cache,
                max_skew_seconds=300,
                now=clock.wall_seconds(),
            )
            accepted += 1
        except (StaleTimestampError, ReplayDetectedError):
            continue
    if accepted == 1 and len(cache) == 1:
        return Observation("committed_exactly_once", "accepted_once_replay_rejected")
    if accepted == 0 and len(cache) == 0:
        return Observation("no_effect", "stale_rejected_nonce_unclaimed")
    return Observation(AMBIGUOUS, "replay_window_violated")


# ---------------------------------------------------------------------------
# service.resilience.circuit_breaker: open -> half-open probe admission
# ---------------------------------------------------------------------------


def _open_breaker(manager: ResilienceManager) -> None:
    def failing() -> None:
        raise RuntimeError("synthetic backend failure")

    try:
        manager.execute("synthetic-model", failing)
    except RuntimeError:
        pass


@_register(
    "service.resilience.circuit_breaker",
    commit_point="breaker.half_open_probe",
    supported={
        ("monotonic_stall", CLOCK_MONOTONIC),
        ("clock_jump_forward", CLOCK_MONOTONIC),
        ("clock_jump_backward", CLOCK_MONOTONIC),
        ("cancellation", "breaker.probe"),
    },
)
def run_circuit_breaker(session: FaultSession) -> Observation:
    """Admit at most one recovery probe after an open circuit."""
    config = ServiceResilienceConfig(
        max_attempts=1,
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        backoff_jitter_seconds=0.0,
    )
    if session.case.injection_point == "breaker.probe":
        return _circuit_breaker_cancellation(session, config)
    clock = FaultClock(
        session, wall_start=WALL_START, monotonic_start=1000.0, tick_seconds=10.0
    )
    manager = ResilienceManager(
        config=config, clock=clock.monotonic, sleep=lambda _s: None
    )
    with session.suspended():
        _open_breaker(manager)
    admitted = 0
    for _check in range(6):
        try:
            manager.check_available("synthetic-model")
            admitted += 1
        except CircuitBreakerOpenError:
            continue
    if admitted == 1:
        return Observation("committed_exactly_once", "single_probe_admitted")
    if admitted == 0:
        return Observation("no_effect", "circuit_held_open")
    return Observation(AMBIGUOUS, "multiple_probes_admitted")


def _circuit_breaker_cancellation(
    session: FaultSession, config: ServiceResilienceConfig
) -> Observation:
    now = [0.0]
    manager = ResilienceManager(
        config=config, clock=lambda: now[0], sleep=lambda _s: None
    )
    _open_breaker(manager)
    now[0] = 100.0
    cancelled = False
    try:
        manager.execute(
            "synthetic-model",
            lambda: session.call("breaker.probe", lambda: "probe-ok"),
        )
    except asyncio.CancelledError:
        cancelled = True
    try:
        manager.execute("synthetic-model", lambda: "follow-up-ok")
    except CircuitBreakerOpenError:
        return Observation(AMBIGUOUS, "probe_reservation_leaked")
    if manager.state_counts().get("closed") != 1:
        return Observation(AMBIGUOUS, "circuit_state_unexpected")
    if cancelled:
        return Observation("no_effect", "cancelled_probe_released")
    return Observation("committed_exactly_once", "probe_closed_circuit")


# ---------------------------------------------------------------------------
# service.throttle.concurrency_gate: permit acquire and release
# ---------------------------------------------------------------------------


@_register(
    "service.throttle.concurrency_gate",
    commit_point="gate.permit",
    supported={("cancellation", "gate.acquire"), ("cancellation", "request.call_next")},
)
def run_concurrency_gate(session: FaultSession) -> Observation:
    """Cancel a waiting or admitted request and probe for leaked permits."""
    if session.case.injection_point == "gate.acquire":
        return asyncio.run(_gate_waiting_acquire(session))
    return asyncio.run(_gate_admitted_request(session))


async def _gate_waiting_acquire(session: FaultSession) -> Observation:
    gate = ConcurrencyGate(max_concurrency=1, wait_seconds=30.0)
    holder = await gate.acquire()
    assert holder is not None
    waiter = asyncio.ensure_future(gate.acquire())
    for _tick in range(3):
        await asyncio.sleep(0)
    cancelled = False
    if session.trip("gate.acquire"):
        waiter.cancel()
        try:
            await waiter
        except asyncio.CancelledError:
            cancelled = True
    holder.release()
    if not cancelled:
        second = await waiter
        if second is None:
            return Observation(AMBIGUOUS, "waiter_not_admitted")
        second.release()
    semaphore = gate._semaphores["global"]  # noqa: SLF001
    if semaphore.locked():
        return Observation(AMBIGUOUS, "permit_leaked")
    if cancelled:
        return Observation("no_effect", "cancelled_waiter_released")
    return Observation("committed_exactly_once", "waiter_admitted_once")


async def _gate_admitted_request(session: FaultSession) -> Observation:
    throttle = ServiceThrottle(
        ServiceThrottleConfig(max_concurrency=1, concurrency_wait_seconds=0.0),
        error_response=lambda status, *_args, **_kwargs: Response(status_code=status),
    )
    request = SimpleNamespace(
        url=SimpleNamespace(path="/deidentify"),
        headers={},
        client=None,
        app=SimpleNamespace(state=SimpleNamespace(metrics=None)),
    )

    async def call_next(_request: Any) -> Response:
        session.checkpoint("request.call_next")
        return Response(status_code=200)

    async def plain(_request: Any) -> Response:
        return Response(status_code=200)

    cancelled = False
    try:
        await throttle.dispatch(request, call_next)
    except asyncio.CancelledError:
        cancelled = True
    follow_up = await throttle.dispatch(request, plain)
    semaphore = throttle.concurrency_gate._semaphores["global"]  # noqa: SLF001
    if follow_up.status_code != 200 or semaphore.locked():
        return Observation(AMBIGUOUS, "permit_leaked")
    if cancelled:
        return Observation("rolled_back", "cancelled_request_released_permit")
    return Observation("committed_exactly_once", "request_released_permit")


# ---------------------------------------------------------------------------
# structured.store.artifacts: content-addressed blob publication via os.link
# ---------------------------------------------------------------------------

_ARTIFACT_HASH = journey_digest(CONTENT_CANARY)


def _journey_artifact() -> ClinicalArtifact:
    return ClinicalArtifact(
        artifact_id="artifact_aaaaaaaaaaaaaaaa",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=_ARTIFACT_HASH,
        byte_size=len(CONTENT_CANARY),
        source_id="source_aaaaaaaaaaaaaaaa",
        recorded_at=COMMITTED_AT,
        subject_id="subject_aaaaaaaaaaaaaaaa",
        encounter_id="encounter_aaaaaaaaaaaaaaaa",
    )


def _journey_evidence() -> EvidenceLocator:
    return EvidenceLocator(
        locator_id="evidence_aaaaaaaaaaaaaaaa",
        artifact_id="artifact_aaaaaaaaaaaaaaaa",
        location_type="text_span",
        location={"start": 0, "end": 9},
    )


def _journey_fact() -> ClinicalFact:
    return ClinicalFact(
        fact_id="fact_aaaaaaaaaaaaaaaa",
        subject_id="subject_aaaaaaaaaaaaaaaa",
        fact_type="condition",
        value={"code": "synthetic-condition", "state": "active"},
        status="active",
        evidence_ids=("evidence_aaaaaaaaaaaaaaaa",),
        derivation_hash=canonical_digest({"fact": "synthetic"}),
        encounter_id="encounter_aaaaaaaaaaaaaaaa",
    )


def _blob_path(root: Path) -> Path:
    digest = _ARTIFACT_HASH.removeprefix("sha256:")
    return root / "blobs" / "sha256" / digest[:2] / digest


def _artifact_residue(root: Path, *, blob_expected: bool) -> int:
    """Count staged temp files, plus the blob when no commit references it."""
    blob = _blob_path(root)
    stray = [path for path in _files(root) if path != blob]
    return len(stray) + int(blob.exists() and not blob_expected)


@_register(
    "structured.store.artifacts",
    commit_point=DISK_LINK,
    supported={
        ("enospc", DISK_WRITE),
        ("eio", DISK_WRITE),
        ("partial_write", DISK_WRITE),
        ("fsync_failure", DISK_FSYNC),
        ("eio", DISK_LINK),
        ("process_kill", DISK_LINK),
        ("eio", DISK_READ),
        ("truncated_state", STATE_RESTART),
        ("garbled_state", STATE_RESTART),
    },
)
def run_artifact_store(session: FaultSession) -> Observation:
    """Publish one verified blob and read it back after a restart."""
    root = session.state_dir / "artifacts"
    reading = session.case.injection_point in {DISK_READ, STATE_RESTART}
    with disk_faults(session, session.state_dir):
        with session.suspended():
            store = LocalArtifactStore(root)
        interrupted = False
        if reading:
            with session.suspended():
                stored = store.put_bytes(_journey_artifact(), CONTENT_CANARY)
        else:
            try:
                stored = store.put_bytes(_journey_artifact(), CONTENT_CANARY)
            except SimulatedProcessKill:
                interrupted = True
        if not interrupted and not stored.ok:
            with session.suspended():
                residue = _artifact_residue(root, blob_expected=False)
                blob_absent = not _blob_path(root).exists()
            if stored.code == "artifact_write_failed" and blob_absent:
                return Observation("rolled_back", "artifact_write_failed", residue)
            return Observation(AMBIGUOUS, "artifact_write_unexpected", residue)

        restart_root = (session.crash_dir if interrupted else session.state_dir) / (
            "artifacts"
        )
        corrupted = session.checkpoint(STATE_RESTART, target=_blob_path(restart_root))
        with session.suspended():
            restart_bytes = _blob_path(restart_root).read_bytes()
        reopened = LocalArtifactStore(restart_root)
        first = reopened.get_bytes(_ARTIFACT_HASH)
        second = first if first.ok else reopened.get_bytes(_ARTIFACT_HASH)
    with session.suspended():
        residue = _artifact_residue(restart_root, blob_expected=True)
        preserved = _blob_path(restart_root).read_bytes() == restart_bytes
    if corrupted and first.code == "artifact_integrity_failed" and preserved:
        return Observation("quarantined", "artifact_integrity_failed", residue)
    if not first.ok and second.ok and second.value == CONTENT_CANARY:
        return Observation("retryable", "artifact_read_retried", residue)
    if first.ok and first.value == CONTENT_CANARY:
        return Observation("committed_exactly_once", "artifact_verified", residue)
    return Observation(AMBIGUOUS, "artifact_restart_unexpected", residue)


# ---------------------------------------------------------------------------
# structured.store.journey: SQLite metadata commit joined to blobs
# ---------------------------------------------------------------------------


def _orphan_blobs(store: LocalJourneyStore, root: Path) -> int:
    blob = _blob_path(root / "artifacts")
    stray = [
        path
        for path in _files(root / "artifacts")
        if path != blob and path.name.startswith(".journey-blob-")
    ]
    unreferenced = (
        blob.exists()
        and store.metadata.get_artifact_by_hash(_ARTIFACT_HASH).state
        is StoreState.UNKNOWN
    )
    return len(stray) + int(unreferenced)


def _ingest(store: LocalJourneyStore) -> Any:
    return store.ingest_graph(
        _journey_artifact(),
        CONTENT_CANARY,
        evidence=(_journey_evidence(),),
        facts=(_journey_fact(),),
        committed_at=COMMITTED_AT,
    )


@_register(
    "structured.store.journey",
    commit_point=SQLITE_COMMIT,
    supported={
        ("process_kill", SQLITE_COMMIT),
        ("eio", SQLITE_COMMIT),
        ("cancellation", SQLITE_COMMIT),
        ("truncated_state", STATE_RESTART),
        ("garbled_state", STATE_RESTART),
    },
)
def run_journey_store(session: FaultSession) -> Observation:
    """Commit one source-to-fact graph and reopen the store."""
    if session.case.injection_point == STATE_RESTART:
        return _journey_store_corruption(session)
    operation = session.case.token("operation", "ingest_graph")
    store = LocalJourneyStore(session.state_dir)
    outcome: Any = None
    escaped: str | None = None
    try:
        with sqlite_commit_faults(session, store.metadata):
            if operation == "put_artifact":
                outcome = store.metadata.put_artifact(
                    _journey_artifact(), committed_at=COMMITTED_AT
                )
            else:
                outcome = _ingest(store)
    except SimulatedProcessKill:
        escaped = "process_kill"
    except asyncio.CancelledError:
        escaped = "cancelled"
    finally:
        store.close()

    root = session.crash_dir if escaped == "process_kill" else session.state_dir
    reopened = LocalJourneyStore(root)
    try:
        residue = _orphan_blobs(reopened, root)
        revision = reopened.metadata.latest_revision
        integrity = reopened.metadata.integrity_check()
        if not integrity.ok:
            return Observation(AMBIGUOUS, "restart_integrity_failed", residue)
        if escaped == "process_kill":
            if revision is None:
                retried = _ingest(reopened)
                if retried.ok and reopened.metadata.latest_revision == 1:
                    return Observation(
                        "retryable", "uncommitted_graph_retried", residue
                    )
                return Observation(AMBIGUOUS, "retry_failed", residue)
            retried = _ingest(reopened)
            if (
                retried.ok
                and not retried.created
                and reopened.metadata.latest_revision == 1
            ):
                return Observation(
                    "committed_exactly_once", "committed_graph_deduplicated", residue
                )
            return Observation(AMBIGUOUS, "restart_duplicate_commit", residue)
        if escaped == "cancelled" or (outcome is not None and not outcome.ok):
            if revision is None:
                return Observation("rolled_back", "transaction_rolled_back", residue)
            return Observation(AMBIGUOUS, "partial_commit_visible", residue)
        if revision == 1 and outcome is not None and outcome.ok:
            return Observation("committed_exactly_once", "graph_committed", residue)
        return Observation(AMBIGUOUS, "commit_state_unexpected", residue)
    finally:
        reopened.close()


def _journey_store_corruption(session: FaultSession) -> Observation:
    with session.suspended():
        store = LocalJourneyStore(session.state_dir)
        committed = _ingest(store)
        store.close()
    if not committed.ok:
        return Observation(AMBIGUOUS, "setup_commit_failed")
    database = session.state_dir / "journey.sqlite3"
    corrupted = session.checkpoint(STATE_RESTART, target=database)
    restart_bytes = database.read_bytes()
    opened = SQLiteJourneyStore.open(database)
    preserved = database.read_bytes() == restart_bytes
    if opened.ok and opened.value is not None:
        revision = opened.value.latest_revision
        opened.value.close()
        if not corrupted and revision == 1:
            return Observation("committed_exactly_once", "store_reopened")
        return Observation(AMBIGUOUS, "corrupt_store_opened")
    if corrupted and preserved and opened.code == "store_open_failed":
        return Observation("quarantined", "store_open_failed")
    return Observation(AMBIGUOUS, "store_open_unexpected")


__all__ = [
    "CANARIES",
    "COMPONENTS",
    "Component",
    "Observation",
    "component_for",
]
