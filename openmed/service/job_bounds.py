"""Admission and retention bounds for in-memory FHIR bulk and SMART jobs.

Each job manager keeps its records in one dictionary. These helpers bound how
many of those records may be running and how many may be kept at all:

* A start is admitted only while fewer than ``max_running_jobs`` tasks are
  live. The check runs before any task, directory or transport is touched.
* Terminal records expire ``terminal_ttl_seconds`` after their last update.
  Expiry is lazy: managers prune on start and on lookup with their injected
  clock, so no background thread or timer is needed.
* At most ``max_retained_jobs`` records are kept. When a start needs room, the
  oldest terminal records are evicted first. Running records are never
  evicted. Because ``max_running_jobs <= max_retained_jobs`` is validated at
  configuration time, admission guarantees that room can always be made.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_MAX_RUNNING_JOBS = 4
DEFAULT_MAX_RETAINED_JOBS = 256
DEFAULT_TERMINAL_TTL_SECONDS = 3600.0
DEFAULT_RETRY_AFTER_SECONDS = 30.0


def _require_positive_int(name: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_number(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")


@dataclass(frozen=True)
class BulkJobBounds:
    """Server-owned ceilings for one in-memory bulk job manager.

    Attributes:
        max_running_jobs: Most jobs whose tasks may run at once.
        max_retained_jobs: Most job records, running or terminal, kept in
            memory. Must be at least ``max_running_jobs``.
        terminal_ttl_seconds: Age after the last status update at which a
            terminal record is dropped and its id reads as unknown.
        retry_after_seconds: Delay advertised to callers rejected because the
            running ceiling is reached.
    """

    max_running_jobs: int = DEFAULT_MAX_RUNNING_JOBS
    max_retained_jobs: int = DEFAULT_MAX_RETAINED_JOBS
    terminal_ttl_seconds: float = DEFAULT_TERMINAL_TTL_SECONDS
    retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS

    def __post_init__(self) -> None:
        _require_positive_int("max_running_jobs", self.max_running_jobs)
        _require_positive_int("max_retained_jobs", self.max_retained_jobs)
        _require_positive_number("terminal_ttl_seconds", self.terminal_ttl_seconds)
        _require_positive_number("retry_after_seconds", self.retry_after_seconds)
        if self.max_running_jobs > self.max_retained_jobs:
            raise ValueError("max_running_jobs must not exceed max_retained_jobs")


class BulkJobsSaturatedError(RuntimeError):
    """Raised when a job manager already runs its configured maximum.

    The error carries only the manager kind and configured numbers, never a
    request path, URL or credential.
    """

    def __init__(
        self,
        *,
        jobs: str,
        max_running_jobs: int,
        retry_after_seconds: float,
    ) -> None:
        self.jobs = str(jobs)
        self.max_running_jobs = int(max_running_jobs)
        self.retry_after_seconds = float(retry_after_seconds)
        super().__init__("Bulk job capacity is saturated; retry later")

    def to_details(self) -> dict[str, Any]:
        """Return the stable, value-free API details for this rejection."""

        return {
            "jobs": self.jobs,
            "max_running_jobs": self.max_running_jobs,
            "retry_after_seconds": self.retry_after_seconds,
        }


class _Status(Protocol):
    updated_at: float


class _Task(Protocol):
    def done(self) -> bool: ...


class _Record(Protocol):
    status: _Status
    task: _Task


def running_job_count(jobs: MutableMapping[str, _Record]) -> int:
    """Return how many records still own a live task."""

    return sum(1 for record in jobs.values() if not record.task.done())


def prune_job_records(
    jobs: MutableMapping[str, _Record],
    bounds: BulkJobBounds,
    now: float,
    *,
    reserve: int = 0,
) -> None:
    """Drop expired terminal records, then evict the oldest terminal ones.

    Args:
        jobs: The manager's job records, keyed by job id.
        bounds: Retention ceilings for this manager.
        now: Current time from the manager's clock.
        reserve: Slots to free for records about to be inserted.
    """

    terminal = [
        (record.status.updated_at, order, job_id)
        for order, (job_id, record) in enumerate(jobs.items())
        if record.task.done()
    ]
    terminal.sort()
    excess = len(jobs) + reserve - bounds.max_retained_jobs
    for updated_at, _, job_id in terminal:
        if excess > 0:
            excess -= 1
        elif now - updated_at < bounds.terminal_ttl_seconds:
            continue
        del jobs[job_id]


def ensure_job_capacity(
    jobs: MutableMapping[str, _Record],
    bounds: BulkJobBounds,
    *,
    kind: str,
) -> None:
    """Raise ``BulkJobsSaturatedError`` when no running slot is free."""

    if running_job_count(jobs) >= bounds.max_running_jobs:
        raise BulkJobsSaturatedError(
            jobs=kind,
            max_running_jobs=bounds.max_running_jobs,
            retry_after_seconds=bounds.retry_after_seconds,
        )


__all__ = [
    "BulkJobBounds",
    "BulkJobsSaturatedError",
    "DEFAULT_MAX_RETAINED_JOBS",
    "DEFAULT_MAX_RUNNING_JOBS",
    "DEFAULT_RETRY_AFTER_SECONDS",
    "DEFAULT_TERMINAL_TTL_SECONDS",
    "ensure_job_capacity",
    "prune_job_records",
    "running_job_count",
]
