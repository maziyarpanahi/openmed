"""Unit tests for fair, bounded principal admission."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from openmed.service.admission import (
    ADMISSION_MAX_CONCURRENCY_ENV_VAR,
    ADMISSION_MAX_QUEUE_ENV_VAR,
    ADMISSION_NAMESPACE_CLAIM_ENV_VAR,
    ADMISSION_NAMESPACE_MAX_CONCURRENCY_ENV_VAR,
    ADMISSION_NAMESPACE_MAX_QUEUE_ENV_VAR,
    ADMISSION_PRINCIPAL_MAX_CONCURRENCY_ENV_VAR,
    ADMISSION_PRINCIPAL_MAX_QUEUE_ENV_VAR,
    ADMISSION_QUEUE_TIMEOUT_ENV_VAR,
    ADMISSION_RETRY_AFTER_ENV_VAR,
    ANONYMOUS_PRINCIPAL,
    AdmissionCancelled,
    AdmissionConfig,
    AdmissionController,
    AdmissionPermit,
    AdmissionSaturated,
    AdmissionStats,
    AdmissionTicket,
    PrincipalAdmissionLimited,
)
from openmed.service.metrics import PrometheusMetricsRegistry
from openmed.service.runtime import ServiceRuntime, parse_service_admission_config

_ADMISSION_ENV_VARS = (
    ADMISSION_MAX_CONCURRENCY_ENV_VAR,
    ADMISSION_MAX_QUEUE_ENV_VAR,
    ADMISSION_PRINCIPAL_MAX_CONCURRENCY_ENV_VAR,
    ADMISSION_PRINCIPAL_MAX_QUEUE_ENV_VAR,
    ADMISSION_NAMESPACE_MAX_CONCURRENCY_ENV_VAR,
    ADMISSION_NAMESPACE_MAX_QUEUE_ENV_VAR,
    ADMISSION_NAMESPACE_CLAIM_ENV_VAR,
    ADMISSION_QUEUE_TIMEOUT_ENV_VAR,
    ADMISSION_RETRY_AFTER_ENV_VAR,
)
# Waits in these tests end by explicit release; the timeout only bounds a hang.
_LONG_WAIT = 30.0


@dataclass(frozen=True)
class _Principal:
    subject: str
    claims: dict[str, Any] = field(default_factory=dict)


def _ticket(
    subject: str | None,
    *,
    namespace: str | None = None,
    transport: str = "rest",
) -> AdmissionTicket:
    principal = None
    if subject is not None:
        principal = _Principal(subject, {"tenant": namespace} if namespace else {})
    return AdmissionTicket.for_principal(
        principal, transport=transport, namespace_claim="tenant"
    )


async def _spin(turns: int = 10) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def clean_admission_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ADMISSION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_admission_is_disabled_by_default() -> None:
    config = parse_service_admission_config()
    controller = AdmissionController(config)

    permit = asyncio.run(controller.acquire(_ticket("svc-a")))

    assert config == AdmissionConfig()
    assert controller.enabled is False
    assert permit.release() is True
    assert permit.release() is False
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_admission_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in (
        (ADMISSION_MAX_CONCURRENCY_ENV_VAR, "8"),
        (ADMISSION_MAX_QUEUE_ENV_VAR, "16"),
        (ADMISSION_PRINCIPAL_MAX_CONCURRENCY_ENV_VAR, "2"),
        (ADMISSION_PRINCIPAL_MAX_QUEUE_ENV_VAR, "4"),
        (ADMISSION_NAMESPACE_MAX_CONCURRENCY_ENV_VAR, "4"),
        (ADMISSION_NAMESPACE_MAX_QUEUE_ENV_VAR, "6"),
        (ADMISSION_NAMESPACE_CLAIM_ENV_VAR, " tenant "),
        (ADMISSION_QUEUE_TIMEOUT_ENV_VAR, "2.5"),
        (ADMISSION_RETRY_AFTER_ENV_VAR, "3"),
    ):
        monkeypatch.setenv(name, value)

    config = parse_service_admission_config()
    runtime = ServiceRuntime.from_env()

    assert config == AdmissionConfig(
        max_concurrency=8,
        max_queue=16,
        principal_max_concurrency=2,
        principal_max_queue=4,
        namespace_max_concurrency=4,
        namespace_max_queue=6,
        namespace_claim="tenant",
        queue_timeout_seconds=2.5,
        retry_after_seconds=3.0,
    )
    assert config.enabled and config.namespaces_enabled
    assert runtime.admission == config
    assert runtime.admission_controller is runtime.admission_controller
    assert runtime.admission_controller.config == config


@pytest.mark.parametrize(
    ("env_var", "raw_value"),
    [
        (ADMISSION_PRINCIPAL_MAX_CONCURRENCY_ENV_VAR, "-1"),
        (ADMISSION_MAX_QUEUE_ENV_VAR, "SYNTHETIC_SECRET_VALUE"),
        (ADMISSION_QUEUE_TIMEOUT_ENV_VAR, "0"),
        (ADMISSION_RETRY_AFTER_ENV_VAR, "inf"),
        (ADMISSION_NAMESPACE_CLAIM_ENV_VAR, "SYNTHETIC SECRET claim!"),
    ],
)
def test_invalid_admission_env_is_rejected_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch, env_var: str, raw_value: str
) -> None:
    monkeypatch.setenv(env_var, raw_value)

    with pytest.raises(ValueError) as excinfo:
        parse_service_admission_config()

    assert env_var in str(excinfo.value)
    assert raw_value.strip() not in str(excinfo.value)


def test_tickets_use_opaque_keys_and_pool_anonymous_callers() -> None:
    first = _ticket("SYNTHETIC_PRINCIPAL_SUBJECT", namespace="SYNTHETIC_TENANT")
    second = _ticket("SYNTHETIC_PRINCIPAL_SUBJECT", namespace="SYNTHETIC_TENANT")
    other = _ticket("another-subject")
    literal = _ticket(ANONYMOUS_PRINCIPAL)

    assert first == second
    assert first.principal != other.principal
    assert "SYNTHETIC" not in repr(first)
    assert first.namespace is not None and first.namespace.startswith("n:")
    assert _ticket(None) == _ticket("") == AdmissionTicket(transport="rest")
    assert _ticket(None).principal == ANONYMOUS_PRINCIPAL
    assert literal.principal != ANONYMOUS_PRINCIPAL
    assert _ticket("svc", transport="soap").transport == "other"
    assert (
        AdmissionTicket.for_principal(
            _Principal("svc", {"tenant": "x" * 129}),
            transport="grpc",
            namespace_claim="tenant",
        ).namespace
        is None
    )


def test_anonymous_callers_share_one_bucket() -> None:
    controller = AdmissionController(AdmissionConfig(principal_max_concurrency=1))

    async def scenario() -> None:
        held = await controller.acquire(_ticket(None))
        with pytest.raises(PrincipalAdmissionLimited):
            await controller.acquire(_ticket(""))
        named = await controller.acquire(_ticket("svc-a"))
        assert controller.snapshot() == AdmissionStats(2, 0, 2)
        held.release()
        named.release()

    asyncio.run(scenario())
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_noisy_neighbour_cannot_take_a_quiet_principals_headroom() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            max_concurrency=4,
            max_queue=8,
            principal_max_concurrency=2,
            principal_max_queue=2,
            queue_timeout_seconds=_LONG_WAIT,
        )
    )
    noisy, quiet = _ticket("noisy"), _ticket("quiet")

    async def scenario() -> None:
        held = [await controller.acquire(noisy) for _ in range(2)]
        queued = [asyncio.create_task(controller.acquire(noisy)) for _ in range(2)]
        await _spin()
        with pytest.raises(PrincipalAdmissionLimited) as excinfo:
            await controller.acquire(noisy)
        assert excinfo.value.status_code == 429
        assert excinfo.value.reason == "principal_limit"

        quiet_permit = await controller.acquire(quiet)
        assert controller.snapshot() == AdmissionStats(3, 2, 2)

        quiet_permit.release()
        for permit in held:
            permit.release()
        for permit in await asyncio.gather(*queued):
            permit.release()

    asyncio.run(scenario())
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_round_robin_bounds_how_long_a_queued_principal_waits() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            max_concurrency=1,
            max_queue=16,
            principal_max_concurrency=1,
            principal_max_queue=8,
            queue_timeout_seconds=_LONG_WAIT,
        )
    )
    order: list[str] = []

    async def wait_for(name: str) -> AdmissionPermit:
        permit = await controller.acquire(_ticket(name.split("-")[0]))
        order.append(name)
        return permit

    async def scenario() -> None:
        current = await controller.acquire(_ticket("noisy"))
        tasks = [asyncio.create_task(wait_for(f"noisy-{index}")) for index in range(6)]
        await _spin()
        tasks += [asyncio.create_task(wait_for(name)) for name in ("quiet", "calm")]
        await _spin()
        assert controller.snapshot().queued == 8

        for _ in range(len(tasks)):
            granted = len(order)
            current.release()
            await _spin()
            assert len(order) == granted + 1
            current = next(
                task.result()
                for task in tasks
                if task.done() and not task.result().released
            )
        current.release()

    asyncio.run(scenario())

    # FIFO would serve all six noisy requests first. Rotation serves each
    # waiting principal within (principals waiting - 1) grants.
    assert order[:3] == ["noisy-0", "quiet", "calm"]
    assert order[3:] == [f"noisy-{index}" for index in range(1, 6)]
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_queue_bounds_reject_over_share_with_429_and_saturation_with_503() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            max_concurrency=2,
            max_queue=2,
            principal_max_concurrency=2,
            principal_max_queue=2,
            queue_timeout_seconds=_LONG_WAIT,
            retry_after_seconds=1.5,
        )
    )
    noisy, quiet = _ticket("noisy"), _ticket("quiet")

    async def scenario() -> None:
        held = [await controller.acquire(noisy) for _ in range(2)]
        queued = [asyncio.create_task(controller.acquire(noisy)) for _ in range(2)]
        await _spin()

        with pytest.raises(PrincipalAdmissionLimited) as over_share:
            await controller.acquire(noisy)
        with pytest.raises(AdmissionSaturated) as saturated:
            await controller.acquire(quiet)

        assert over_share.value.to_details() == {
            "reason": "principal_limit",
            "retry_after_seconds": 1.5,
        }
        assert saturated.value.status_code == 503
        assert saturated.value.error_code == "admission_saturated"
        assert saturated.value.to_details() == {
            "reason": "global_limit",
            "retry_after_seconds": 1.5,
        }
        assert saturated.value.retry_after_header == "2"
        assert controller.snapshot() == AdmissionStats(2, 2, 1)

        for permit in held:
            permit.release()
        for permit in await asyncio.gather(*queued):
            permit.release()

    asyncio.run(scenario())
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_namespace_ceiling_is_shared_by_its_principals_only() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            principal_max_concurrency=2,
            namespace_max_concurrency=2,
            namespace_claim="tenant",
        )
    )
    first = _ticket("a1", namespace="tenant-a")
    second = _ticket("a2", namespace="tenant-a")
    other = _ticket("b1", namespace="tenant-b")

    async def scenario() -> None:
        held = [await controller.acquire(first), await controller.acquire(second)]
        with pytest.raises(PrincipalAdmissionLimited) as excinfo:
            await controller.acquire(second)
        assert excinfo.value.reason == "namespace_limit"
        held += [await controller.acquire(other), await controller.acquire(other)]
        for permit in held:
            permit.release()

    asyncio.run(scenario())
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_namespace_is_ignored_without_a_configured_claim() -> None:
    controller = AdmissionController(
        AdmissionConfig(principal_max_concurrency=2, namespace_max_concurrency=1)
    )

    async def scenario() -> None:
        held = [
            await controller.acquire(_ticket("a1", namespace="tenant-a")),
            await controller.acquire(_ticket("a2", namespace="tenant-a")),
        ]
        for permit in held:
            permit.release()

    asyncio.run(scenario())


def test_cancelled_waiters_release_queue_slots_and_granted_units_once() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            max_concurrency=1,
            max_queue=4,
            queue_timeout_seconds=_LONG_WAIT,
        )
    )

    async def scenario() -> None:
        held = await controller.acquire(_ticket("svc-a"))
        cancelled_while_queued = asyncio.create_task(
            controller.acquire(_ticket("svc-b"))
        )
        await _spin()
        cancelled_while_queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_while_queued
        assert controller.snapshot() == AdmissionStats(1, 0, 1)

        granted_then_cancelled = asyncio.create_task(
            controller.acquire(_ticket("svc-b"))
        )
        next_in_line = asyncio.create_task(controller.acquire(_ticket("svc-c")))
        await _spin()
        assert controller.snapshot().queued == 2
        # The release grants svc-b's unit before its task can resume; the
        # cancellation must hand that unit on instead of leaking it.
        assert held.release() is True
        granted_then_cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await granted_then_cancelled
        survivor = await next_in_line
        assert controller.snapshot() == AdmissionStats(1, 0, 1)

        assert held.release() is False
        assert survivor.release() is True
        assert survivor.release() is False

    asyncio.run(scenario())
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_expired_waiters_are_rejected_by_the_level_that_blocked_them() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            max_concurrency=2,
            max_queue=4,
            principal_max_concurrency=1,
            principal_max_queue=2,
        )
    )

    async def scenario() -> None:
        own = await controller.acquire(_ticket("svc-a"))
        with pytest.raises(PrincipalAdmissionLimited) as own_share:
            await controller.acquire(_ticket("svc-a"), timeout=0)
        other = await controller.acquire(_ticket("svc-b"))
        with pytest.raises(AdmissionSaturated) as shared:
            await controller.acquire(_ticket("svc-c"), timeout=0)
        assert own_share.value.reason == "principal_limit"
        assert shared.value.reason == "global_limit"
        assert controller.snapshot() == AdmissionStats(2, 0, 2)
        own.release()
        other.release()

    asyncio.run(scenario())
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_worker_failure_inside_admit_releases_the_permit() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            max_concurrency=1, max_queue=1, queue_timeout_seconds=_LONG_WAIT
        )
    )

    async def failing_work() -> None:
        async with controller.admit(_ticket("svc-a")):
            await asyncio.sleep(0)
            raise RuntimeError("synthetic worker failure")

    async def scenario() -> None:
        waiter = asyncio.create_task(failing_work())
        await _spin(1)
        follower = asyncio.create_task(controller.acquire(_ticket("svc-b")))
        with pytest.raises(RuntimeError):
            await waiter
        (await follower).release()

    asyncio.run(scenario())
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_noisy_flood_never_rejects_quiet_principals_within_their_share() -> None:
    config = AdmissionConfig(
        max_concurrency=3,
        max_queue=6,
        principal_max_concurrency=2,
        principal_max_queue=2,
        queue_timeout_seconds=_LONG_WAIT,
    )
    peaks = {"in_flight": 0, "queued": 0}

    class Recorder:
        def record_principal_admission_state(self, **state: int) -> None:
            peaks["in_flight"] = max(peaks["in_flight"], state["in_flight"])
            peaks["queued"] = max(peaks["queued"], state["queued"])

        def record_principal_admission_rejection(self, **_: str) -> None:
            pass

    controller = AdmissionController(config, metrics=Recorder())
    active: dict[str, int] = {}
    completed: dict[str, int] = {}
    noisy_outcomes: dict[str, int] = {}

    async def unit_of_work(name: str, *, fail: bool = False) -> None:
        async with controller.admit(_ticket(name)):
            active[name] = active.get(name, 0) + 1
            assert active[name] <= config.principal_max_concurrency
            try:
                await _spin(2)
                if fail:
                    raise RuntimeError("synthetic worker failure")
            finally:
                active[name] -= 1
        completed[name] = completed.get(name, 0) + 1

    async def noisy_request(index: int) -> None:
        try:
            await unit_of_work("noisy", fail=index % 7 == 0)
            outcome = "completed"
        except PrincipalAdmissionLimited as exc:
            outcome = exc.reason
        except RuntimeError:
            outcome = "failed"
        noisy_outcomes[outcome] = noisy_outcomes.get(outcome, 0) + 1

    async def quiet_client(name: str) -> None:
        # A sequential client stays within its share; it must never be shed.
        for _ in range(10):
            await unit_of_work(name)

    async def noisy_flood() -> None:
        requests = []
        for index in range(80):
            requests.append(asyncio.create_task(noisy_request(index)))
            await asyncio.sleep(0)
        await asyncio.gather(*requests)

    async def scenario() -> None:
        await asyncio.gather(
            noisy_flood(), *(quiet_client(f"quiet-{index}") for index in range(3))
        )

    asyncio.run(scenario())

    assert {name: completed[name] for name in completed if name != "noisy"} == {
        "quiet-0": 10,
        "quiet-1": 10,
        "quiet-2": 10,
    }
    assert noisy_outcomes["completed"] >= 1
    assert noisy_outcomes["failed"] >= 1
    assert noisy_outcomes["principal_limit"] >= 1
    assert sum(noisy_outcomes.values()) == 80
    assert set(noisy_outcomes) <= {"completed", "failed", "principal_limit"}
    assert peaks["in_flight"] <= config.max_concurrency
    assert 0 < peaks["queued"] <= config.max_queue
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_blocking_adapter_shares_the_budget_with_async_callers() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            max_concurrency=4,
            max_queue=4,
            principal_max_concurrency=1,
            principal_max_queue=1,
            queue_timeout_seconds=_LONG_WAIT,
        )
    )
    rest_ticket = _ticket("svc-a", transport="rest")
    grpc_ticket = _ticket("svc-a", transport="grpc")
    results: list[AdmissionPermit] = []

    def blocking_call() -> None:
        results.append(controller.acquire_blocking(grpc_ticket))

    async def scenario() -> None:
        held = await controller.acquire(rest_ticket)
        worker = threading.Thread(target=blocking_call)
        worker.start()
        for _ in range(10_000):
            if controller.snapshot().queued == 1:
                break
            await asyncio.sleep(0.001)
        assert controller.snapshot() == AdmissionStats(1, 1, 1)

        with pytest.raises(PrincipalAdmissionLimited) as excinfo:
            await controller.acquire(_ticket("svc-a", transport="graphql"))
        assert excinfo.value.grpc_status == "RESOURCE_EXHAUSTED"
        assert excinfo.value.grpc_trailing_metadata() == (
            ("retry-after", "1"),
            ("openmed-admission-reason", "principal_limit"),
        )

        held.release()
        await asyncio.to_thread(worker.join, _LONG_WAIT)
        assert not worker.is_alive()

    asyncio.run(scenario())
    assert len(results) == 1
    assert controller.snapshot() == AdmissionStats(1, 0, 1)
    with controller.admit_blocking(_ticket("svc-b", transport="grpc")):
        assert controller.snapshot().in_flight == 2
    assert results[0].release() is True
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_blocking_waiter_cancellation_and_saturation() -> None:
    controller = AdmissionController(
        AdmissionConfig(
            max_concurrency=1, max_queue=1, queue_timeout_seconds=_LONG_WAIT
        )
    )
    held = controller.acquire_blocking(_ticket("svc-a", transport="grpc"))

    with pytest.raises(AdmissionCancelled):
        controller.acquire_blocking(
            _ticket("svc-b", transport="grpc"), cancelled=lambda: True
        )
    assert controller.snapshot() == AdmissionStats(1, 0, 1)
    with pytest.raises(AdmissionSaturated) as excinfo:
        controller.acquire_blocking(_ticket("svc-b", transport="grpc"), timeout=0)
    assert excinfo.value.grpc_status == "UNAVAILABLE"

    held.release()
    assert controller.snapshot() == AdmissionStats(0, 0, 0)


def test_metrics_and_errors_carry_no_principal_identifiers() -> None:
    registry = PrometheusMetricsRegistry()
    controller = AdmissionController(
        AdmissionConfig(
            principal_max_concurrency=1,
            namespace_max_concurrency=4,
            namespace_claim="tenant",
        ),
        metrics=registry,
    )
    ticket = _ticket(
        "SYNTHETIC_PRINCIPAL_SECRET",
        namespace="SYNTHETIC_TENANT_SECRET",
        transport="graphql",
    )

    async def scenario() -> PrincipalAdmissionLimited:
        held = await controller.acquire(ticket)
        try:
            await controller.acquire(ticket)
        except PrincipalAdmissionLimited as exc:
            rendered_while_held.append(registry.render())
            held.release()
            return exc
        raise AssertionError("second acquisition should be rejected")

    rendered_while_held: list[str] = []
    rejection = asyncio.run(scenario())
    rendered = rendered_while_held[0]

    assert "openmed_service_principal_admission_inflight 1" in rendered
    assert "openmed_service_principal_admission_queued 0" in rendered
    assert "openmed_service_principal_admission_active_principals 1" in rendered
    assert (
        "openmed_service_principal_admission_rejected_total"
        '{reason="principal_limit",transport="graphql"} 1'
    ) in rendered
    assert "openmed_service_principal_admission_inflight 0" in registry.render()
    surfaces = [
        rendered,
        registry.render(),
        str(rejection),
        repr(rejection.to_details()),
        repr(rejection.grpc_trailing_metadata()),
        repr(controller.snapshot()),
        repr(ticket),
    ]
    for surface in surfaces:
        assert "SYNTHETIC" not in surface


def test_rejection_metrics_only_use_closed_label_sets() -> None:
    registry = PrometheusMetricsRegistry()

    registry.record_principal_admission_rejection(
        reason="SYNTHETIC_REASON", transport="SYNTHETIC_TRANSPORT"
    )

    rendered = registry.render()
    assert 'reason="other",transport="other"' in rendered
    assert "SYNTHETIC" not in rendered


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_concurrency": -1},
        {"principal_max_queue": True},
        {"queue_timeout_seconds": 0},
        {"retry_after_seconds": float("nan")},
    ],
)
def test_admission_config_validates_programmatic_values(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        AdmissionConfig(**overrides)
