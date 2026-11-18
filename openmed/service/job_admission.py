"""Submission-time capacity limits for async de-identification jobs.

The async job queue keeps each job's raw documents in memory until a worker
finishes it. :class:`JobAdmission` bounds that memory before any job metadata
is persisted: it caps the number of pending jobs, the UTF-8 bytes of document
text they hold, and the number of documents in a single job.

Rejections carry only controlled codes, reasons and configured limits. They
never echo document text, identifiers or other request values.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Protocol

DEFAULT_MAX_PENDING_JOBS = 64
DEFAULT_MAX_QUEUED_TEXT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_DOCUMENTS_PER_JOB = 1_000
DEFAULT_RETRY_AFTER_SECONDS = 5

REASON_PENDING_JOBS = "pending_jobs"
REASON_QUEUED_TEXT_BYTES = "queued_text_bytes"
REASON_DOCUMENTS_PER_JOB = "documents_per_job"
REASON_JOB_TEXT_BYTES = "job_text_bytes"


class JobQueueMetrics(Protocol):
    """Aggregate gauge recorder notified when job capacity changes."""

    def job_queue_admitted(self, *, text_bytes: int) -> None:
        """Record one admitted job holding ``text_bytes`` of document text."""

    def job_queue_released(self, *, text_bytes: int) -> None:
        """Record one released job that held ``text_bytes`` of document text."""


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class JobAdmissionLimits:
    """Configured ceilings for async job submission.

    Attributes:
        max_pending_jobs: Maximum jobs admitted and not yet finished, counting
            both queued and running jobs.
        max_queued_text_bytes: Maximum UTF-8 bytes of document text held by
            all pending jobs. A single job above this ceiling can never be
            admitted and is rejected as too large.
        max_documents_per_job: Maximum documents accepted in one job.
        retry_after_seconds: ``Retry-After`` hint returned when the queue is
            full.
    """

    max_pending_jobs: int = DEFAULT_MAX_PENDING_JOBS
    max_queued_text_bytes: int = DEFAULT_MAX_QUEUED_TEXT_BYTES
    max_documents_per_job: int = DEFAULT_MAX_DOCUMENTS_PER_JOB
    retry_after_seconds: int = DEFAULT_RETRY_AFTER_SECONDS

    def __post_init__(self) -> None:
        _require_positive_int(self.max_pending_jobs, "max_pending_jobs")
        _require_positive_int(self.max_queued_text_bytes, "max_queued_text_bytes")
        _require_positive_int(self.max_documents_per_job, "max_documents_per_job")
        _require_positive_int(self.retry_after_seconds, "retry_after_seconds")


@dataclass(frozen=True)
class JobCost:
    """Capacity one job consumes while it is pending.

    Attributes:
        documents: Number of documents in the job.
        text_bytes: Total UTF-8 bytes of the job's document text.
    """

    documents: int
    text_bytes: int

    @classmethod
    def for_texts(cls, texts: Iterable[str]) -> "JobCost":
        """Measure a job from its document texts.

        Args:
            texts: The raw document texts. They are measured, never retained.

        Returns:
            The job's document count and total UTF-8 text bytes.
        """
        documents = 0
        text_bytes = 0
        for text in texts:
            documents += 1
            # ``surrogatepass`` keeps measurement total for lone surrogates.
            text_bytes += len(text.encode("utf-8", "surrogatepass"))
        return cls(documents=documents, text_bytes=text_bytes)


@dataclass(frozen=True)
class JobAdmissionSnapshot:
    """Aggregate pending-job occupancy.

    Attributes:
        pending_jobs: Jobs admitted and not yet finished.
        queued_text_bytes: UTF-8 bytes of document text held by pending jobs.
    """

    pending_jobs: int
    queued_text_bytes: int


class JobAdmissionError(Exception):
    """Base class for value-free job submission rejections.

    Attributes:
        status_code: HTTP status used by the REST service.
        error_code: Stable public error code.
        message: Fixed, value-free error message.
        reason: Which configured ceiling rejected the job.
        limit: The configured value of that ceiling.
        retry_after_seconds: Retry hint, or ``None`` when retrying the same
            request cannot succeed.
    """

    status_code: int = 422
    error_code: str
    message: str

    def __init__(
        self,
        *,
        reason: str,
        limit: int,
        retry_after_seconds: Optional[int] = None,
    ) -> None:
        super().__init__(self.message)
        self.reason = reason
        self.limit = int(limit)
        self.retry_after_seconds = retry_after_seconds

    def to_details(self) -> dict[str, Any]:
        """Return the value-free error details for API envelopes."""
        details: dict[str, Any] = {"reason": self.reason, "limit": self.limit}
        if self.retry_after_seconds is not None:
            details["retry_after_seconds"] = self.retry_after_seconds
        return details


class JobLimitExceededError(JobAdmissionError):
    """A single job exceeds a per-job ceiling and can never be admitted."""

    status_code = 422


class JobTooManyDocumentsError(JobLimitExceededError):
    """A job has more documents than the per-job ceiling allows."""

    error_code = "job_too_many_documents"
    message = "Job exceeds the maximum number of documents per job"


class JobTooLargeError(JobLimitExceededError):
    """A job's text alone exceeds the queued text byte ceiling."""

    error_code = "job_too_large"
    message = "Job text exceeds the maximum queued text size"


class JobQueueFullError(JobAdmissionError):
    """The pending-job queue is at capacity; the job may be retried later."""

    status_code = 503
    error_code = "job_queue_full"
    message = "Job queue is full. Retry later."


class JobReservation:
    """Capacity held by one admitted job until it is released.

    Release is idempotent, so the submission failure path and the worker
    completion path can both release the same reservation safely.
    """

    __slots__ = ("cost", "_admission", "_released")

    def __init__(self, admission: "JobAdmission", cost: JobCost) -> None:
        self.cost = cost
        self._admission = admission
        self._released = False

    @property
    def released(self) -> bool:
        """Whether this reservation has been returned to the queue."""
        return self._released

    def release(self, *_args: Any) -> None:
        """Return the reserved capacity.

        Extra positional arguments are ignored so the method can be passed
        directly to ``Future.add_done_callback``.
        """
        self._admission._release(self)


class JobAdmission:
    """Thread-safe admission control for the async job queue.

    ``reserve`` checks every ceiling and claims capacity atomically, before the
    caller persists any job metadata. The caller must release the reservation
    when the job finishes, fails or is cancelled.

    Args:
        limits: Configured ceilings; safe defaults when omitted.
        metrics: Optional aggregate gauge recorder.
    """

    def __init__(
        self,
        limits: Optional[JobAdmissionLimits] = None,
        *,
        metrics: Optional[JobQueueMetrics] = None,
    ) -> None:
        self.limits = JobAdmissionLimits() if limits is None else limits
        self.metrics = metrics
        self._lock = threading.Lock()
        self._pending_jobs = 0
        self._queued_text_bytes = 0

    def check(self, cost: JobCost) -> None:
        """Reject a job that exceeds a per-job ceiling.

        Args:
            cost: The job's measured capacity.

        Raises:
            JobLimitExceededError: If the job has too many documents or more
                text than the whole queue may hold.
        """
        limits = self.limits
        if cost.documents > limits.max_documents_per_job:
            raise JobTooManyDocumentsError(
                reason=REASON_DOCUMENTS_PER_JOB,
                limit=limits.max_documents_per_job,
            )
        if cost.text_bytes > limits.max_queued_text_bytes:
            raise JobTooLargeError(
                reason=REASON_JOB_TEXT_BYTES,
                limit=limits.max_queued_text_bytes,
            )

    def reserve(self, cost: JobCost) -> JobReservation:
        """Claim queue capacity for one job.

        Args:
            cost: The job's measured capacity.

        Returns:
            A reservation to release once the job's work is finished.

        Raises:
            JobLimitExceededError: If the job exceeds a per-job ceiling.
            JobQueueFullError: If admitting the job would exceed the pending
                job or queued text byte ceiling.
        """
        self.check(cost)
        limits = self.limits
        with self._lock:
            if self._pending_jobs + 1 > limits.max_pending_jobs:
                raise self._queue_full(REASON_PENDING_JOBS, limits.max_pending_jobs)
            if self._queued_text_bytes + cost.text_bytes > limits.max_queued_text_bytes:
                raise self._queue_full(
                    REASON_QUEUED_TEXT_BYTES,
                    limits.max_queued_text_bytes,
                )
            if self.metrics is not None:
                self.metrics.job_queue_admitted(text_bytes=cost.text_bytes)
            self._pending_jobs += 1
            self._queued_text_bytes += cost.text_bytes
        return JobReservation(self, cost)

    def release(self, reservation: JobReservation) -> None:
        """Return a reservation's capacity; repeated releases are no-ops.

        Args:
            reservation: A reservation returned by :meth:`reserve`.

        Raises:
            ValueError: If the reservation belongs to another admission.
        """
        if reservation._admission is not self:
            raise ValueError("Reservation belongs to another job admission")
        self._release(reservation)

    def snapshot(self) -> JobAdmissionSnapshot:
        """Return current aggregate occupancy."""
        with self._lock:
            return JobAdmissionSnapshot(
                pending_jobs=self._pending_jobs,
                queued_text_bytes=self._queued_text_bytes,
            )

    def _release(self, reservation: JobReservation) -> None:
        with self._lock:
            if reservation._released:
                return
            reservation._released = True
            text_bytes = reservation.cost.text_bytes
            self._pending_jobs = max(self._pending_jobs - 1, 0)
            self._queued_text_bytes = max(self._queued_text_bytes - text_bytes, 0)
            if self.metrics is not None:
                self.metrics.job_queue_released(text_bytes=text_bytes)

    def _queue_full(self, reason: str, limit: int) -> JobQueueFullError:
        return JobQueueFullError(
            reason=reason,
            limit=limit,
            retry_after_seconds=self.limits.retry_after_seconds,
        )
