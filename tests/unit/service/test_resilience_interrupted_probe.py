"""Recovery probes remain usable after cancellation or process interruption."""

import asyncio
from unittest.mock import Mock

import pytest

from openmed.service.resilience import (
    CIRCUIT_CLOSED,
    CIRCUIT_HALF_OPEN,
    CIRCUIT_OPEN,
    CircuitBreakerOpenError,
    ResilienceManager,
    ServiceResilienceConfig,
)

_KEY = "synthetic-backend"


@pytest.fixture
def recovering_manager():
    now = [0.0]
    manager = ResilienceManager(
        config=ServiceResilienceConfig(
            max_attempts=1, failure_threshold=1, recovery_timeout_seconds=1.0
        ),
        clock=lambda: now[0],
        sleep=lambda delay: None,
    )
    manager.record_failure(_KEY)
    assert manager.snapshots()[_KEY].state == CIRCUIT_OPEN
    now[0] = 1.0
    return manager


@pytest.mark.parametrize(
    "exception_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_interrupted_probe_releases_slot_without_fabricating_success(
    recovering_manager, exception_type
):
    manager = recovering_manager
    error = exception_type("synthetic interruption")
    operation = Mock(side_effect=error)
    with pytest.raises(exception_type) as caught:
        manager.execute(_KEY, operation)
    assert caught.value is error
    assert operation.call_count == 1
    snapshot = manager.snapshots()[_KEY]
    assert snapshot.state == CIRCUIT_HALF_OPEN
    assert snapshot.failures == 1
    assert manager.execute(_KEY, lambda: "recovered") == "recovered"
    assert manager.snapshots()[_KEY].state == CIRCUIT_CLOSED


def test_failure_after_interrupted_probe_reopens_circuit(recovering_manager):
    manager = recovering_manager
    with pytest.raises(asyncio.CancelledError):
        manager.execute(_KEY, Mock(side_effect=asyncio.CancelledError()))
    error = RuntimeError("synthetic backend failure")
    with pytest.raises(RuntimeError) as caught:
        manager.execute(_KEY, Mock(side_effect=error))
    assert caught.value is error
    assert manager.snapshots()[_KEY].state == CIRCUIT_OPEN


def test_repeated_interruptions_do_not_permanently_occupy_probe(recovering_manager):
    for _ in range(3):
        with pytest.raises(asyncio.CancelledError):
            recovering_manager.execute(_KEY, Mock(side_effect=asyncio.CancelledError()))
    assert recovering_manager.execute(_KEY, lambda: 42) == 42


def test_an_active_probe_still_excludes_other_calls(recovering_manager):
    def probe():
        with pytest.raises(CircuitBreakerOpenError):
            recovering_manager.execute(_KEY, lambda: "must not run")
        return "outer probe"

    assert recovering_manager.execute(_KEY, probe) == "outer probe"
    assert recovering_manager.snapshots()[_KEY].state == CIRCUIT_CLOSED


def test_interruption_during_retry_sleep_releases_probe():
    now = [0.0]
    error = asyncio.CancelledError("synthetic sleep cancellation")
    manager = ResilienceManager(
        config=ServiceResilienceConfig(
            max_attempts=2, failure_threshold=1, recovery_timeout_seconds=1.0
        ),
        clock=lambda: now[0],
        sleep=Mock(side_effect=error),
        jitter=lambda upper: 0.0,
    )
    manager.record_failure(_KEY)
    now[0] = 1.0
    operation = Mock(side_effect=RuntimeError("synthetic retryable failure"))
    with pytest.raises(asyncio.CancelledError) as caught:
        manager.execute(_KEY, operation)
    assert caught.value is error
    assert operation.call_count == 1
    assert manager.execute(_KEY, lambda: "recovered") == "recovered"


def test_interruption_does_not_clear_closed_circuit_failure_evidence():
    manager = ResilienceManager(
        config=ServiceResilienceConfig(max_attempts=1, failure_threshold=3)
    )
    manager.record_failure(_KEY)
    with pytest.raises(asyncio.CancelledError):
        manager.execute(_KEY, Mock(side_effect=asyncio.CancelledError()))
    snapshot = manager.snapshots()[_KEY]
    assert snapshot.state == CIRCUIT_CLOSED
    assert snapshot.failures == 1


def test_disabled_manager_does_not_allocate_breaker():
    manager = ResilienceManager(config=ServiceResilienceConfig(enabled=False))
    error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError) as caught:
        manager.execute(_KEY, Mock(side_effect=error))
    assert caught.value is error
    assert manager.snapshots() == {}


def test_value_error_policy_remains_unchanged(recovering_manager):
    error = ValueError("synthetic invalid request")
    with pytest.raises(ValueError) as caught:
        recovering_manager.execute(_KEY, Mock(side_effect=error))
    assert caught.value is error
    assert recovering_manager.snapshots()[_KEY].state == CIRCUIT_CLOSED


def test_interrupted_closed_call_cannot_release_another_calls_probe():
    now = [0.0]
    manager = ResilienceManager(
        config=ServiceResilienceConfig(
            max_attempts=1, failure_threshold=1, recovery_timeout_seconds=1.0
        ),
        clock=lambda: now[0],
    )

    def operation():
        # Deterministically model a different request failing and a recovery
        # probe starting while this earlier closed-state call is still active.
        manager.record_failure(_KEY)
        now[0] = 1.0
        manager.check_available(_KEY)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        manager.execute(_KEY, operation)
    with pytest.raises(CircuitBreakerOpenError):
        manager.execute(_KEY, lambda: "must not run")
    manager.record_success(_KEY)
    assert manager.execute(_KEY, lambda: "done") == "done"


def test_old_probe_cancellation_cannot_release_a_new_probe(recovering_manager):
    manager = recovering_manager

    def operation():
        # An external health update retires the old probe and permits a new
        # generation. The cancelled old call must not release the new token.
        manager.record_success(_KEY)
        manager.record_failure(_KEY)
        manager.clock = lambda: 2.0
        manager._breakers[_KEY].clock = manager.clock
        manager.check_available(_KEY)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        manager.execute(_KEY, operation)
    with pytest.raises(CircuitBreakerOpenError):
        manager.execute(_KEY, lambda: "must not run")
    manager.record_success(_KEY)
