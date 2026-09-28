"""Idempotent step runner for durable ingestion ledgers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

from openmed.clinical.journey_contracts import canonical_digest
from openmed.structured.store import StoreResult, StoreState

from .contracts import (
    QUARANTINE_CLASSIFICATIONS,
    Cancellation,
    Checkpoint,
    IngestionJob,
    QuarantinePromotion,
    QuarantineResult,
    Retry,
    SourceManifest,
)
from .store import IngestionLedger, ManifestRegistration, _format_time, _parse_time

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")


@dataclass(frozen=True, slots=True)
class StepOutput:
    """Trusted, value-free output identity for one completed step."""

    output_digest: str
    committed_revision: int | None = None

    def __post_init__(self) -> None:
        _require_digest(self.output_digest, "output_digest")
        if self.committed_revision is not None and (
            type(self.committed_revision) is not int or self.committed_revision < 1
        ):
            raise ValueError("committed_revision must be positive")


@dataclass(frozen=True, slots=True)
class QuarantineRequest:
    """Value-free request to isolate malformed or partial step output."""

    classification: str
    reason_code: str
    candidate_count: int
    failure_count: int
    output_digest: str | None = None

    def __post_init__(self) -> None:
        if self.classification not in QUARANTINE_CLASSIFICATIONS:
            raise ValueError("quarantine classification is unsupported")
        _require_controlled(self.reason_code, "reason_code")
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if type(self.failure_count) is not int or self.failure_count < 1:
            raise ValueError("failure_count must be positive")
        if self.output_digest is not None:
            _require_digest(self.output_digest, "output_digest")


@dataclass(frozen=True, slots=True)
class StepExecution:
    """Auditable result for a completed, replayed, or quarantined step."""

    job_id: str
    step: str
    replayed: bool
    checkpoint: Checkpoint | None = None
    quarantine: QuarantineResult | None = None
    retry: Retry | None = None

    def __post_init__(self) -> None:
        _require_controlled(self.step, "step")
        populated = sum(
            value is not None
            for value in (self.checkpoint, self.quarantine, self.retry)
        )
        if populated != 1:
            raise ValueError("step execution requires exactly one outcome record")
        if self.replayed and self.checkpoint is None:
            raise ValueError("only a checkpoint can be replayed")


@dataclass(frozen=True, slots=True)
class FailureClassification:
    """Stable retry category and controlled reason code."""

    classification: str
    reason_code: str
    retry_delay_seconds: int | None


class IngestionCoordinator:
    """Run duplicate-safe steps against a durable ingestion ledger."""

    def __init__(self, ledger: IngestionLedger) -> None:
        if not isinstance(ledger, IngestionLedger):
            raise TypeError("ledger does not satisfy ingestion contracts")
        self.ledger = ledger

    def register(
        self,
        manifest: SourceManifest,
        *,
        recorded_at: str,
    ) -> StoreResult[ManifestRegistration]:
        """Register a new manifest or return its audited no-op replay."""

        return self.ledger.register_manifest(manifest, recorded_at=recorded_at)

    def execute_step(
        self,
        *,
        job_id: str,
        worker_id: str,
        step: str,
        input_digest: str,
        acquired_at: str,
        lease_seconds: int,
        attempt: int,
        operation: Callable[[], StepOutput | QuarantineRequest],
    ) -> StoreResult[StepExecution]:
        """Execute one logical step or return its prior checkpoint.

        ``BaseException`` is intentionally not caught: a process termination
        leaves the lease to expire and the absent checkpoint makes the step
        safely replayable. Ordinary exceptions become value-free retry records.
        """

        _require_controlled(step, "step")
        _require_digest(input_digest, "input_digest")
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be positive")

        prior = self.ledger.get_step_checkpoint(job_id, step, input_digest)
        if prior.ok and prior.value is not None:
            return StoreResult.success(
                StepExecution(
                    job_id=job_id,
                    step=step,
                    replayed=True,
                    checkpoint=prior.value,
                ),
                created=False,
            )
        if prior.state not in {StoreState.UNKNOWN}:
            return StoreResult.outcome(
                prior.state, prior.code or "checkpoint_read_failed"
            )

        lease_result = self.ledger.acquire_lease(
            job_id,
            worker_id,
            acquired_at=acquired_at,
            duration_seconds=lease_seconds,
        )
        if not lease_result.ok or lease_result.value is None:
            return StoreResult.outcome(
                lease_result.state,
                lease_result.code or "lease_acquisition_failed",
            )
        lease = lease_result.value

        prior = self.ledger.get_step_checkpoint(job_id, step, input_digest)
        if prior.ok and prior.value is not None:
            return StoreResult.success(
                StepExecution(
                    job_id=job_id,
                    step=step,
                    replayed=True,
                    checkpoint=prior.value,
                ),
                created=False,
            )
        try:
            output = operation()
            if not isinstance(output, (StepOutput, QuarantineRequest)):
                raise TypeError("operation returned an unsupported step result")
        except Exception as exc:
            classification = classify_failure(exc)
            acquired = _parse_time(acquired_at, "acquired_at")
            retry_after = (
                None
                if classification.retry_delay_seconds is None
                else _format_time(
                    acquired + timedelta(seconds=classification.retry_delay_seconds)
                )
            )
            retry = Retry(
                retry_id=_derived_id(
                    "retry",
                    job_id,
                    step,
                    input_digest,
                    str(attempt),
                ),
                job_id=job_id,
                classification=classification.classification,
                reason_code=classification.reason_code,
                attempt=attempt,
                recorded_at=acquired_at,
                retry_after=retry_after,
            )
            recorded = self.ledger.record_retry(retry, lease_id=lease.lease_id)
            if not recorded.ok:
                return StoreResult.outcome(
                    recorded.state,
                    recorded.code or "retry_write_failed",
                )
            return StoreResult.outcome(
                StoreState.FAILURE,
                classification.reason_code,
                value=StepExecution(
                    job_id=job_id,
                    step=step,
                    replayed=False,
                    retry=retry,
                ),
            )

        job_result = self.ledger.get_job(job_id)
        if not job_result.ok or job_result.value is None:
            return StoreResult.outcome(
                job_result.state,
                job_result.code or "job_read_failed",
            )
        job = job_result.value
        if isinstance(output, QuarantineRequest):
            quarantine = QuarantineResult(
                quarantine_id=_derived_id(
                    "quarantine",
                    job_id,
                    step,
                    input_digest,
                ),
                job_id=job_id,
                manifest_digest=job.manifest_digest,
                classification=output.classification,
                reason_code=output.reason_code,
                candidate_count=output.candidate_count,
                failure_count=output.failure_count,
                created_at=acquired_at,
                output_digest=output.output_digest,
            )
            quarantined = self.ledger.quarantine(
                quarantine,
                lease_id=lease.lease_id,
            )
            if not quarantined.ok:
                return StoreResult.outcome(
                    quarantined.state,
                    quarantined.code or "quarantine_write_failed",
                )
            return StoreResult.outcome(
                StoreState.PARTIAL,
                "source_quarantined",
                value=StepExecution(
                    job_id=job_id,
                    step=step,
                    replayed=False,
                    quarantine=quarantine,
                ),
            )

        checkpoint = Checkpoint(
            checkpoint_id=_derived_id(
                "checkpoint",
                job_id,
                step,
                input_digest,
            ),
            job_id=job_id,
            manifest_digest=job.manifest_digest,
            step=step,
            sequence=job.checkpoint_sequence + 1,
            input_digest=input_digest,
            output_digest=output.output_digest,
            committed_revision=output.committed_revision,
            completed_at=acquired_at,
        )
        committed = self.ledger.commit_checkpoint(
            checkpoint,
            lease_id=lease.lease_id,
            recorded_at=acquired_at,
        )
        if not committed.ok or committed.value is None:
            return StoreResult.outcome(
                committed.state,
                committed.code or "checkpoint_write_failed",
            )
        return StoreResult.success(
            StepExecution(
                job_id=job_id,
                step=step,
                replayed=not committed.created,
                checkpoint=committed.value,
            ),
            created=committed.created,
        )

    def complete(
        self,
        job_id: str,
        *,
        lease_id: str,
        completed_at: str,
    ) -> StoreResult[IngestionJob]:
        """Complete a job under its live lease."""

        return self.ledger.complete_job(
            job_id,
            lease_id=lease_id,
            completed_at=completed_at,
        )

    def cancel(self, cancellation: Cancellation) -> StoreResult[Cancellation]:
        """Cancel a job explicitly and durably."""

        return self.ledger.cancel_job(cancellation)

    def promote(
        self,
        promotion: QuarantinePromotion,
    ) -> StoreResult[QuarantinePromotion]:
        """Promote a quarantined result after explicit review evidence."""

        return self.ledger.promote_quarantine(promotion)


def classify_failure(error: Exception) -> FailureClassification:
    """Return a stable value-free classification without inspecting messages."""

    if isinstance(error, PermissionError):
        return FailureClassification("policy_denied", "policy_denied", None)
    if isinstance(error, TimeoutError):
        return FailureClassification("transient", "operation_timeout", 30)
    if isinstance(error, ConnectionError):
        return FailureClassification("transient", "transport_unavailable", 30)
    if isinstance(error, MemoryError):
        return FailureClassification("resource", "resource_exhausted", 120)
    if isinstance(error, FileNotFoundError):
        return FailureClassification("dependency", "dependency_unavailable", 60)
    if isinstance(error, OSError):
        return FailureClassification("dependency", "io_unavailable", 60)
    if isinstance(error, (TypeError, ValueError)):
        return FailureClassification("permanent", "invalid_step_input", None)
    return FailureClassification("unknown", "unclassified_failure", None)


def _derived_id(prefix: str, *values: str) -> str:
    digest = canonical_digest({"prefix": prefix, "values": list(values)})
    return f"{prefix}_{digest.removeprefix('sha256:')[:32]}"


def _require_digest(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return value


def _require_controlled(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a controlled identifier")
    return value


__all__ = [
    "FailureClassification",
    "IngestionCoordinator",
    "QuarantineRequest",
    "StepExecution",
    "StepOutput",
    "classify_failure",
]
