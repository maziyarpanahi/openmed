"""Bounded admission and hysteretic load shedding for service batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AdmissionSnapshot:
    """Immutable aggregate state for one admission queue."""

    queue_name: str
    depth: int
    high_watermark: int
    low_watermark: int
    shedding: bool


class BackpressureError(RuntimeError):
    """Raised when the inference admission queue cannot accept more work."""

    def __init__(
        self,
        *,
        priority: str,
        queue_depth: int,
        queue_capacity: int,
        retry_after_seconds: float,
        queue_name: str = "batch",
        low_watermark: Optional[int] = None,
        max_wait_ms: Optional[float] = None,
        reason: str = "saturated",
    ) -> None:
        self.priority = str(priority)
        self.queue_name = str(queue_name)
        self.queue_depth = int(queue_depth)
        self.queue_capacity = int(queue_capacity)
        self.low_watermark = None if low_watermark is None else int(low_watermark)
        self.max_wait_ms = None if max_wait_ms is None else float(max_wait_ms)
        self.retry_after_seconds = max(float(retry_after_seconds), 0.0)
        self.reason = str(reason)
        super().__init__(f"{self.priority} inference queue is saturated; retry later")

    def to_details(self) -> dict[str, Any]:
        """Return the stable, PHI-free API details for this rejection."""
        details: dict[str, Any] = {
            "priority": self.priority,
            "queue": self.queue_name,
            "queue_depth": self.queue_depth,
            "queue_capacity": self.queue_capacity,
            "retry_after_seconds": self.retry_after_seconds,
            "reason": self.reason,
        }
        if self.low_watermark is not None:
            details["low_watermark"] = self.low_watermark
        if self.max_wait_ms is not None:
            details["max_wait_ms"] = self.max_wait_ms
        return details


class AdmissionQueue:
    """Track bounded outstanding work with high/low-watermark hysteresis.

    The owner must serialize calls to this object. ``DynamicBatcher`` does so
    with its event-loop lock, which keeps admission, dispatch, cancellation,
    and timeout transitions atomic without adding another lock.
    """

    def __init__(
        self,
        *,
        queue_name: str,
        high_watermark: int,
        low_watermark: int,
        max_wait_ms: float,
        metrics: Optional[Any] = None,
    ) -> None:
        high = int(high_watermark)
        low = int(low_watermark)
        max_wait = float(max_wait_ms)
        if high <= 0:
            raise ValueError("high_watermark must be positive")
        if low < 0:
            raise ValueError("low_watermark must be greater than or equal to 0")
        if low >= high:
            raise ValueError("low_watermark must be less than high_watermark")
        if max_wait < 0:
            raise ValueError("max_wait_ms must be greater than or equal to 0")

        self.queue_name = str(queue_name)
        self.high_watermark = high
        self.low_watermark = low
        self.max_wait_ms = max_wait
        self._metrics = metrics
        self._depth = 0
        self._shedding = False
        self._record_state()
        self.record_wait(0.0)

    @property
    def max_wait_seconds(self) -> float:
        """Return the maximum pre-dispatch wait in seconds."""
        return self.max_wait_ms / 1000.0

    def snapshot(self) -> AdmissionSnapshot:
        """Return the current queue state."""
        return AdmissionSnapshot(
            queue_name=self.queue_name,
            depth=self._depth,
            high_watermark=self.high_watermark,
            low_watermark=self.low_watermark,
            shedding=self._shedding,
        )

    def admit(self, *, priority: str) -> None:
        """Reserve one bounded slot or raise a backpressure error."""
        error = self.admission_error(priority=priority, record_shed=False)
        if error is not None:
            self._record_shed()
            raise error

        self._depth += 1
        if self._depth >= self.high_watermark:
            self._shedding = True
        self._record_state()

    def admission_error(
        self,
        *,
        priority: str,
        record_shed: bool = True,
    ) -> Optional[BackpressureError]:
        """Return an error while shedding, without reserving a slot."""
        if not self._shedding and self._depth < self.high_watermark:
            return None

        if not self._shedding:
            self._shedding = True
            self._record_state()
        if record_shed:
            self._record_shed()
        return self._error(priority=priority, reason="high_watermark")

    def release(self) -> None:
        """Release one completed or cancelled request from admission."""
        if self._depth <= 0:
            return
        self._depth -= 1
        if self._shedding and self._depth <= self.low_watermark:
            self._shedding = False
        self._record_state()

    def wait_expired(self, *, priority: str) -> BackpressureError:
        """Record and describe a request shed after excessive queue wait."""
        self._record_shed()
        return self._error(priority=priority, reason="max_wait")

    def record_wait(self, wait_seconds: float) -> None:
        """Record the latest pre-dispatch wait using aggregate metrics only."""
        record = getattr(self._metrics, "record_admission_queue_wait", None)
        if callable(record):
            record(
                queue=self.queue_name,
                wait_seconds=max(float(wait_seconds), 0.0),
            )

    def _error(self, *, priority: str, reason: str) -> BackpressureError:
        return BackpressureError(
            priority=priority,
            queue_name=self.queue_name,
            queue_depth=self._depth,
            queue_capacity=self.high_watermark,
            low_watermark=self.low_watermark,
            max_wait_ms=self.max_wait_ms,
            retry_after_seconds=max(self.max_wait_seconds, 0.001),
            reason=reason,
        )

    def _record_state(self) -> None:
        record = getattr(self._metrics, "record_admission_queue_state", None)
        if callable(record):
            record(
                queue=self.queue_name,
                depth=self._depth,
                shedding=self._shedding,
            )

    def _record_shed(self) -> None:
        record = getattr(self._metrics, "record_admission_shed", None)
        if callable(record):
            record(queue=self.queue_name)
