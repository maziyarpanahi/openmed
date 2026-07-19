"""Dynamic request batching primitives for the REST service."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Optional, Sequence, TypeVar, Union

from .backpressure import AdmissionQueue, AdmissionSnapshot, BackpressureError

T = TypeVar("T")
R = TypeVar("R")
BatchResult = Union[R, BaseException]
BatchDispatch = Callable[
    [Sequence[T]], Union[Sequence[BatchResult[R]], Awaitable[Sequence[BatchResult[R]]]]
]

PRIORITY_INTERACTIVE = "interactive"
PRIORITY_BULK = "bulk"
PRIORITY_CLASSES = (PRIORITY_INTERACTIVE, PRIORITY_BULK)
DEFAULT_PRIORITY_WEIGHTS = {
    PRIORITY_INTERACTIVE: 4,
    PRIORITY_BULK: 1,
}
DEFAULT_MAX_QUEUE_SIZE_PER_PRIORITY = 256

_PRIORITY_ALIASES = {
    "interactive": PRIORITY_INTERACTIVE,
    "high": PRIORITY_INTERACTIVE,
    "latency": PRIORITY_INTERACTIVE,
    "realtime": PRIORITY_INTERACTIVE,
    "real-time": PRIORITY_INTERACTIVE,
    "foreground": PRIORITY_INTERACTIVE,
    "normal": PRIORITY_INTERACTIVE,
    "bulk": PRIORITY_BULK,
    "batch": PRIORITY_BULK,
    "background": PRIORITY_BULK,
    "low": PRIORITY_BULK,
}
_PRIORITY_RANK = {priority: index for index, priority in enumerate(PRIORITY_CLASSES)}


def normalize_priority(priority: object = PRIORITY_INTERACTIVE) -> str:
    """Normalize a route-layer priority token to a supported priority class."""
    if priority is None:
        return PRIORITY_INTERACTIVE

    normalized = str(priority).strip().lower().replace("_", "-")
    if not normalized:
        return PRIORITY_INTERACTIVE
    try:
        return _PRIORITY_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(PRIORITY_CLASSES)
        raise ValueError(
            f"Unsupported request priority {priority!r}; expected one of: {supported}"
        ) from exc


def priority_is_higher(candidate: str, current: str) -> bool:
    """Return whether *candidate* outranks *current*."""
    return (
        _PRIORITY_RANK[normalize_priority(candidate)]
        < _PRIORITY_RANK[normalize_priority(current)]
    )


class BatchPriorityHandle:
    """Mutable priority handle shared by coalesced callers for one queued job."""

    def __init__(self, priority: object = PRIORITY_INTERACTIVE) -> None:
        self._priority = normalize_priority(priority)
        self._batcher: Optional[DynamicBatcher[Any, Any]] = None
        self._job_id: Optional[int] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def priority(self) -> str:
        """Return the highest priority requested by joined callers."""
        return self._priority

    def promote(self, priority: object) -> None:
        """Promote the queued job if *priority* outranks the current class."""
        normalized = normalize_priority(priority)
        if not priority_is_higher(normalized, self._priority):
            return

        self._priority = normalized
        if self._batcher is None or self._job_id is None or self._loop is None:
            return
        self._loop.create_task(self._batcher.promote(self._job_id, normalized))

    def _bind(
        self,
        batcher: DynamicBatcher[Any, Any],
        job_id: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._batcher = batcher
        self._job_id = job_id
        self._loop = loop

    def _clear(self, job_id: int) -> None:
        if self._job_id != job_id:
            return
        self._batcher = None
        self._job_id = None
        self._loop = None


@dataclass
class _QueuedJob(Generic[T, R]):
    job_id: int
    item: T
    future: asyncio.Future[R]
    priority: str
    queued_at: float
    priority_handle: Optional[BatchPriorityHandle]
    wait_timeout: Optional[asyncio.TimerHandle]


class DynamicBatcher(Generic[T, R]):
    """Collect concurrent jobs and dispatch them as priority-aware micro-batches."""

    def __init__(
        self,
        dispatch: BatchDispatch[T, R],
        *,
        max_batch_size: int,
        max_wait_ms: float,
        max_queue_size_per_priority: int | Mapping[str, int] = (
            DEFAULT_MAX_QUEUE_SIZE_PER_PRIORITY
        ),
        high_watermark: Optional[int] = None,
        low_watermark: Optional[int] = None,
        max_queue_wait_ms: float = 1000.0,
        queue_name: str = "batch",
        priority_weights: Optional[Mapping[str, int]] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be greater than or equal to 0")

        self.max_batch_size = int(max_batch_size)
        self.max_wait_ms = float(max_wait_ms)
        self._dispatch = dispatch
        self._queue_limits = _normalize_queue_limits(max_queue_size_per_priority)
        admission_high = (
            sum(self._queue_limits.values())
            if high_watermark is None
            else int(high_watermark)
        )
        admission_low = (
            max(0, admission_high // 2) if low_watermark is None else int(low_watermark)
        )
        self._admission = AdmissionQueue(
            queue_name=queue_name,
            high_watermark=admission_high,
            low_watermark=admission_low,
            max_wait_ms=max_queue_wait_ms,
            metrics=metrics,
        )
        self._queues: dict[str, deque[_QueuedJob[T, R]]] = {
            priority: deque() for priority in PRIORITY_CLASSES
        }
        self._jobs: dict[int, _QueuedJob[T, R]] = {}
        self._job_sequence = 0
        self._scheduler_cycle = _normalize_priority_weights(priority_weights)
        self._scheduler_index = 0
        self._metrics = metrics
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.TimerHandle] = None
        for priority in PRIORITY_CLASSES:
            self._record_queue_depth(priority)

    async def submit(
        self,
        item: T,
        *,
        priority: object = PRIORITY_INTERACTIVE,
        priority_handle: Optional[BatchPriorityHandle] = None,
    ) -> R:
        """Submit one job and wait for its per-request result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[R] = loop.create_future()
        requested_priority = normalize_priority(priority)
        if priority_handle is not None and priority_is_higher(
            priority_handle.priority,
            requested_priority,
        ):
            requested_priority = priority_handle.priority

        async with self._lock:
            queue = self._queues[requested_priority]
            queue_limit = self._queue_limits[requested_priority]
            if len(queue) >= queue_limit:
                error = self._backpressure_error_locked(requested_priority)
                self._record_shed(requested_priority)
                raise error

            try:
                self._admission.admit(priority=requested_priority)
            except BackpressureError:
                self._record_shed(requested_priority)
                raise

            self._job_sequence += 1
            job = _QueuedJob(
                job_id=self._job_sequence,
                item=item,
                future=future,
                priority=requested_priority,
                queued_at=time.monotonic(),
                priority_handle=priority_handle,
                wait_timeout=None,
            )
            if priority_handle is not None:
                priority_handle._bind(self, job.job_id, loop)
            queue.append(job)
            self._jobs[job.job_id] = job
            self._schedule_wait_timeout_locked(job, loop)
            self._record_queue_depth(job.priority)

            if self._pending_count_locked() == 1:
                self._schedule_flush_locked(loop)
            if self._pending_count_locked() >= self.max_batch_size:
                self._flush_locked(loop)

        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            await self._remove_pending_job(job.job_id)
            raise
        finally:
            async with self._lock:
                self._admission.release()

    async def admission_error(self, priority: object) -> Optional[BackpressureError]:
        """Return a backpressure error if *priority* cannot admit more work."""
        normalized = normalize_priority(priority)
        async with self._lock:
            queue = self._queues[normalized]
            if len(queue) >= self._queue_limits[normalized]:
                error = self._backpressure_error_locked(normalized)
                self._record_shed(normalized)
                return error

            error = self._admission.admission_error(priority=normalized)
            if error is not None:
                self._record_shed(normalized)
            return error

    async def promote(self, job_id: int, priority: object) -> None:
        """Move a queued job to a higher-priority class and flush promptly."""
        normalized = normalize_priority(priority)
        loop = asyncio.get_running_loop()

        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not priority_is_higher(normalized, job.priority):
                return

            old_priority = job.priority
            try:
                self._queues[old_priority].remove(job)
            except ValueError:
                return

            job.priority = normalized
            self._queues[normalized].appendleft(job)
            self._record_queue_depth(old_priority)
            self._record_queue_depth(normalized)
            self._flush_locked(loop)

    async def queue_depths(self) -> dict[str, int]:
        """Return current queued job counts by priority class."""
        async with self._lock:
            return {
                priority: len(self._queues[priority]) for priority in PRIORITY_CLASSES
            }

    async def admission_snapshot(self) -> AdmissionSnapshot:
        """Return aggregate admitted work and hysteresis state."""
        async with self._lock:
            return self._admission.snapshot()

    def _schedule_wait_timeout_locked(
        self,
        job: _QueuedJob[T, R],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        wait_seconds = self._admission.max_wait_seconds
        if wait_seconds <= 0:
            job.wait_timeout = loop.call_soon(self._on_wait_timeout, job.job_id)
        else:
            job.wait_timeout = loop.call_later(
                wait_seconds,
                self._on_wait_timeout,
                job.job_id,
            )

    def _on_wait_timeout(self, job_id: int) -> None:
        asyncio.get_running_loop().create_task(self._expire_waiting_job(job_id))

    async def _expire_waiting_job(self, job_id: int) -> None:
        async with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return
            try:
                self._queues[job.priority].remove(job)
            except ValueError:
                return

            job.wait_timeout = None
            if not self._jobs and self._timer is not None:
                self._timer.cancel()
                self._timer = None
            waited = max(time.monotonic() - job.queued_at, 0.0)
            self._record_queue_depth(job.priority)
            self._record_queue_wait(job.priority, waited)
            self._admission.record_wait(waited)
            self._record_shed(job.priority)
            if job.priority_handle is not None:
                job.priority_handle._clear(job.job_id)
            if not job.future.done():
                job.future.set_exception(
                    self._admission.wait_expired(priority=job.priority)
                )

    def _schedule_flush_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._timer is not None:
            return

        wait_seconds = self.max_wait_ms / 1000.0
        if wait_seconds <= 0:
            self._timer = loop.call_soon(self._on_timer)
        else:
            self._timer = loop.call_later(wait_seconds, self._on_timer)

    def _on_timer(self) -> None:
        loop = asyncio.get_running_loop()
        loop.create_task(self._flush_from_timer())

    async def _flush_from_timer(self) -> None:
        async with self._lock:
            self._flush_locked(asyncio.get_running_loop())

    def _flush_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self._jobs:
            return

        batch = self._next_batch_locked()
        if self._jobs:
            self._schedule_flush_locked(loop)
        if not batch:
            return

        loop.create_task(self._run_batch(batch))

    def _next_batch_locked(self) -> list[_QueuedJob[T, R]]:
        batch: list[_QueuedJob[T, R]] = []
        while len(batch) < self.max_batch_size and self._jobs:
            job = self._pop_next_job_locked()
            if job is None:
                break
            self._jobs.pop(job.job_id, None)
            if job.wait_timeout is not None:
                job.wait_timeout.cancel()
                job.wait_timeout = None
            self._record_queue_depth(job.priority)
            waited = max(time.monotonic() - job.queued_at, 0.0)
            self._record_queue_wait(job.priority, waited)
            self._admission.record_wait(waited)
            if job.priority_handle is not None:
                job.priority_handle._clear(job.job_id)
            batch.append(job)
        return batch

    def _pop_next_job_locked(self) -> Optional[_QueuedJob[T, R]]:
        for _ in range(len(self._scheduler_cycle)):
            priority = self._scheduler_cycle[self._scheduler_index]
            self._scheduler_index = (self._scheduler_index + 1) % len(
                self._scheduler_cycle
            )
            queue = self._queues[priority]
            if queue:
                return queue.popleft()

        for priority in PRIORITY_CLASSES:
            queue = self._queues[priority]
            if queue:
                return queue.popleft()
        return None

    async def _run_batch(self, batch: Sequence[_QueuedJob[T, R]]) -> None:
        active_jobs = [job for job in batch if not job.future.cancelled()]
        if not active_jobs:
            return

        try:
            results = self._dispatch([job.item for job in active_jobs])
            if inspect.isawaitable(results):
                results = await results
            results = list(results)
            if len(results) != len(active_jobs):
                raise ValueError(
                    "Batch dispatch returned "
                    f"{len(results)} results for {len(active_jobs)} jobs"
                )
        except Exception as exc:
            for job in active_jobs:
                if not job.future.done():
                    job.future.set_exception(exc)
            return

        for job, result in zip(active_jobs, results):
            if job.future.done():
                continue
            if isinstance(result, BaseException):
                job.future.set_exception(result)
            else:
                job.future.set_result(result)

    async def _remove_pending_job(self, job_id: int) -> None:
        async with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return
            try:
                self._queues[job.priority].remove(job)
            except ValueError:
                return
            if job.wait_timeout is not None:
                job.wait_timeout.cancel()
                job.wait_timeout = None
            if not self._jobs and self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._record_queue_depth(job.priority)
            if job.priority_handle is not None:
                job.priority_handle._clear(job.job_id)

    def _pending_count_locked(self) -> int:
        return len(self._jobs)

    def _backpressure_error_locked(self, priority: str) -> BackpressureError:
        return BackpressureError(
            priority=priority,
            queue_name=priority,
            queue_depth=len(self._queues[priority]),
            queue_capacity=self._queue_limits[priority],
            retry_after_seconds=max(self.max_wait_ms / 1000.0, 0.001),
        )

    def _record_queue_depth(self, priority: str) -> None:
        record = getattr(self._metrics, "record_batch_queue_depth", None)
        if callable(record):
            record(priority=priority, depth=len(self._queues[priority]))

    def _record_queue_wait(self, priority: str, wait_seconds: float) -> None:
        record = getattr(self._metrics, "record_batch_queue_wait", None)
        if callable(record):
            record(priority=priority, wait_seconds=max(wait_seconds, 0.0))

    def _record_shed(self, priority: str) -> None:
        record = getattr(self._metrics, "record_batch_shed", None)
        if callable(record):
            record(priority=priority)


def _normalize_queue_limits(
    max_queue_size_per_priority: int | Mapping[str, int],
) -> dict[str, int]:
    if isinstance(max_queue_size_per_priority, Mapping):
        limits = {
            priority: DEFAULT_MAX_QUEUE_SIZE_PER_PRIORITY
            for priority in PRIORITY_CLASSES
        }
        for priority, value in max_queue_size_per_priority.items():
            normalized = normalize_priority(priority)
            limits[normalized] = _validate_queue_limit(value)
        return limits

    limit = _validate_queue_limit(max_queue_size_per_priority)
    return {priority: limit for priority in PRIORITY_CLASSES}


def _validate_queue_limit(value: int) -> int:
    limit = int(value)
    if limit <= 0:
        raise ValueError("max_queue_size_per_priority must be positive")
    return limit


def _normalize_priority_weights(
    priority_weights: Optional[Mapping[str, int]],
) -> tuple[str, ...]:
    weights = dict(DEFAULT_PRIORITY_WEIGHTS)
    if priority_weights is not None:
        for priority, weight in priority_weights.items():
            normalized = normalize_priority(priority)
            if int(weight) <= 0:
                raise ValueError("priority weights must be positive")
            weights[normalized] = int(weight)

    cycle: list[str] = []
    for priority in PRIORITY_CLASSES:
        cycle.extend([priority] * int(weights[priority]))
    return tuple(cycle)
