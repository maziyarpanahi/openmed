"""Retry and circuit-breaker primitives for REST model-backed work."""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

CIRCUIT_CLOSED = "closed"
CIRCUIT_OPEN = "open"
CIRCUIT_HALF_OPEN = "half_open"
CIRCUIT_STATES = (CIRCUIT_CLOSED, CIRCUIT_OPEN, CIRCUIT_HALF_OPEN)


@dataclass(frozen=True)
class ServiceResilienceConfig:
    """Retry and circuit-breaker settings for model-backed service work."""

    enabled: bool = True
    max_attempts: int = 3
    backoff_initial_seconds: float = 0.05
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 1.0
    backoff_jitter_seconds: float = 0.01
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    """Sanitized circuit-breaker state suitable for service metadata."""

    state: str
    failures: int
    retry_after_seconds: Optional[int] = None


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a model/backend circuit is open and should fail fast."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = max(int(retry_after_seconds), 0)
        super().__init__("Model backend is temporarily unavailable")


@dataclass
class _CircuitBreaker:
    config: ServiceResilienceConfig
    clock: Callable[[], float]
    state: str = CIRCUIT_CLOSED
    failures: int = 0
    opened_at: Optional[float] = None
    half_open_probe_active: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def before_call(self) -> None:
        with self._lock:
            now = self.clock()
            if self.state == CIRCUIT_OPEN:
                if self._open_remaining(now) <= 0:
                    self.state = CIRCUIT_HALF_OPEN
                    self.half_open_probe_active = False
                else:
                    raise CircuitBreakerOpenError(self._retry_after_seconds(now))

            if self.state == CIRCUIT_HALF_OPEN:
                if self.half_open_probe_active:
                    raise CircuitBreakerOpenError(1)
                self.half_open_probe_active = True

    def record_success(self) -> None:
        with self._lock:
            self.state = CIRCUIT_CLOSED
            self.failures = 0
            self.opened_at = None
            self.half_open_probe_active = False

    def record_failure(self) -> None:
        with self._lock:
            self.half_open_probe_active = False
            if self.state == CIRCUIT_HALF_OPEN:
                self._open()
                return

            self.failures += 1
            if self.failures >= self.config.failure_threshold:
                self._open()

    def snapshot(self) -> CircuitBreakerSnapshot:
        with self._lock:
            now = self.clock()
            state = self.state
            retry_after = None
            if state == CIRCUIT_OPEN:
                remaining = self._open_remaining(now)
                if remaining <= 0:
                    state = CIRCUIT_HALF_OPEN
                else:
                    retry_after = self._retry_after_seconds(now)
            return CircuitBreakerSnapshot(
                state=state,
                failures=self.failures,
                retry_after_seconds=retry_after,
            )

    def _open(self) -> None:
        self.state = CIRCUIT_OPEN
        self.opened_at = self.clock()
        self.half_open_probe_active = False

    def _open_remaining(self, now: float) -> float:
        if self.opened_at is None:
            return 0.0
        elapsed = max(now - self.opened_at, 0.0)
        return self.config.recovery_timeout_seconds - elapsed

    def _retry_after_seconds(self, now: float) -> int:
        remaining = max(self._open_remaining(now), 0.0)
        return max(int(math.ceil(remaining)), 1)


@dataclass
class ResilienceManager:
    """Coordinate retries and circuit breakers for service model/backend keys."""

    config: ServiceResilienceConfig = field(default_factory=ServiceResilienceConfig)
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[float], float] = field(
        default_factory=lambda: lambda upper: random.uniform(0.0, upper),
        repr=False,
    )
    _breakers: dict[str, _CircuitBreaker] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def execute(self, key: str, operation: Callable[[], Any]) -> Any:
        """Run one operation with retry and breaker accounting."""
        if not self.config.enabled:
            return operation()

        breaker = self._breaker_for(key)
        breaker.before_call()
        try:
            result = self._run_with_retry(operation)
        except Exception as exc:
            if _is_retryable_exception(exc):
                breaker.record_failure()
            else:
                breaker.record_success()
            raise
        breaker.record_success()
        return result

    def check_available(self, key: str) -> None:
        """Fail fast if the circuit for *key* is open."""
        if not self.config.enabled:
            return
        self._breaker_for(key).before_call()

    def record_success(self, key: str) -> None:
        """Record a successful operation for a previously checked key."""
        if self.config.enabled:
            self._breaker_for(key).record_success()

    def record_failure(self, key: str) -> None:
        """Record a failed operation for a previously checked key."""
        if self.config.enabled:
            self._breaker_for(key).record_failure()

    def record_error(self, key: str, exc: BaseException) -> None:
        """Record an operation error using the retryability policy."""
        if not self.config.enabled:
            return
        if _is_retryable_exception(exc):
            self._breaker_for(key).record_failure()
        else:
            self._breaker_for(key).record_success()

    def snapshots(self) -> Mapping[str, CircuitBreakerSnapshot]:
        """Return snapshots keyed by internal model/backend id."""
        with self._lock:
            breakers = dict(self._breakers)
        return {key: breaker.snapshot() for key, breaker in breakers.items()}

    def state_counts(self) -> dict[str, int]:
        """Return aggregate breaker counts by state without model labels."""
        counts = {state: 0 for state in CIRCUIT_STATES}
        for snapshot in self.snapshots().values():
            counts[snapshot.state] = counts.get(snapshot.state, 0) + 1
        return counts

    def _breaker_for(self, key: str) -> _CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(key)
            if breaker is None:
                breaker = _CircuitBreaker(config=self.config, clock=self.clock)
                self._breakers[key] = breaker
            return breaker

    def _run_with_retry(self, operation: Callable[[], Any]) -> Any:
        attempts = max(int(self.config.max_attempts), 1)
        for attempt_index in range(attempts):
            try:
                return operation()
            except Exception as exc:
                if not _is_retryable_exception(exc) or attempt_index >= attempts - 1:
                    raise
                self.sleep(self._delay_for_attempt(attempt_index))
        raise RuntimeError("Retry loop exited without a result")

    def _delay_for_attempt(self, attempt_index: int) -> float:
        base_delay = self.config.backoff_initial_seconds * (
            self.config.backoff_multiplier**attempt_index
        )
        capped_delay = min(base_delay, self.config.backoff_max_seconds)
        jitter_seconds = 0.0
        if self.config.backoff_jitter_seconds > 0:
            jitter_seconds = max(
                self.jitter(self.config.backoff_jitter_seconds),
                0.0,
            )
        return capped_delay + jitter_seconds


def circuit_breaker_details(exc: CircuitBreakerOpenError) -> dict[str, Any]:
    """Return a sanitized API details payload for an open circuit."""
    return {
        "state": CIRCUIT_OPEN,
        "retry_after_seconds": exc.retry_after_seconds,
    }


def _is_retryable_exception(exc: BaseException) -> bool:
    """Return whether an exception looks like backend/load infrastructure failure."""
    return not isinstance(exc, ValueError)
