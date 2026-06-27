"""Single-flight request coalescing for REST service inference calls."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, TypeVar, Union

T = TypeVar("T")
AsyncOperation = Callable[[], Union[T, Awaitable[T]]]


@dataclass(frozen=True)
class _CoalescedRequest:
    task: asyncio.Task[Any]


class RequestCoalescer:
    """Share one in-flight computation across identical concurrent requests."""

    def __init__(self, *, eviction_delay_seconds: float = 0.05) -> None:
        if eviction_delay_seconds < 0:
            raise ValueError(
                "eviction_delay_seconds must be greater than or equal to 0"
            )

        self.eviction_delay_seconds = float(eviction_delay_seconds)
        self._entries: Dict[str, _CoalescedRequest] = {}
        self._lock = asyncio.Lock()

    async def run(self, key: str, operation: AsyncOperation[T]) -> T:
        """Run or join the operation associated with ``key``."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                task = asyncio.create_task(self._run_operation(operation))
                entry = _CoalescedRequest(task=task)
                self._entries[key] = entry
                task.add_done_callback(
                    lambda completed_task: self._schedule_eviction(
                        key,
                        completed_task,
                    )
                )

        return await asyncio.shield(entry.task)

    async def _run_operation(self, operation: AsyncOperation[T]) -> T:
        result = operation()
        if inspect.isawaitable(result):
            return await result
        return result

    def _schedule_eviction(self, key: str, task: asyncio.Task[Any]) -> None:
        loop = asyncio.get_running_loop()
        loop.create_task(self._evict_after_delay(key, task))

    async def _evict_after_delay(self, key: str, task: asyncio.Task[Any]) -> None:
        if self.eviction_delay_seconds > 0:
            await asyncio.sleep(self.eviction_delay_seconds)

        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.task is task:
                self._entries.pop(key, None)


def coalescing_key(endpoint: str, payload: Any) -> str:
    """Return a stable hash for an endpoint and normalized request payload."""
    canonical_payload = _canonical_payload(payload)
    encoded = json.dumps(
        {"endpoint": endpoint, "payload": canonical_payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_payload(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    elif hasattr(payload, "dict"):
        payload = payload.dict()

    if isinstance(payload, dict):
        return {str(key): _canonical_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_canonical_payload(value) for value in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return repr(payload)
