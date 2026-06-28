"""Dynamic request batching primitives for the REST service."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Optional, Sequence, TypeVar, Union

T = TypeVar("T")
R = TypeVar("R")
BatchResult = Union[R, BaseException]
BatchDispatch = Callable[
    [Sequence[T]], Union[Sequence[BatchResult[R]], Awaitable[Sequence[BatchResult[R]]]]
]


@dataclass
class _QueuedJob(Generic[T, R]):
    item: T
    future: asyncio.Future[R]


class DynamicBatcher(Generic[T, R]):
    """Collect concurrent jobs and dispatch them as bounded micro-batches."""

    def __init__(
        self,
        dispatch: BatchDispatch[T, R],
        *,
        max_batch_size: int,
        max_wait_ms: float,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be greater than or equal to 0")

        self.max_batch_size = int(max_batch_size)
        self.max_wait_ms = float(max_wait_ms)
        self._dispatch = dispatch
        self._pending: list[_QueuedJob[T, R]] = []
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.TimerHandle] = None

    async def submit(self, item: T) -> R:
        """Submit one job and wait for its per-request result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[R] = loop.create_future()

        async with self._lock:
            self._pending.append(_QueuedJob(item=item, future=future))
            if len(self._pending) == 1:
                self._schedule_flush_locked(loop)
            if len(self._pending) >= self.max_batch_size:
                self._flush_locked(loop)

        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

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
        if not self._pending:
            return

        batch = self._pending[: self.max_batch_size]
        del self._pending[: self.max_batch_size]
        if self._pending:
            self._schedule_flush_locked(loop)

        loop.create_task(self._run_batch(batch))

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
