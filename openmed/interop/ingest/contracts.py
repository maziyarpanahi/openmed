"""Versioned, value-safe contracts for resumable Journey ingestion."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import Any, ClassVar, TypeVar

from openmed.clinical.journey_contracts import canonical_digest, canonical_json

INGESTION_SCHEMA_VERSION = "1.0.0"
INGESTION_SCHEMA_PACKAGE = "openmed.core.schemas.json"
INGESTION_SCHEMA_NAMES = (
    "ingestion_job",
    "source_manifest",
    "checkpoint",
    "lease",
    "retry",
    "cancellation",
    "quarantine_result",
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")

JOB_STATES = frozenset(
    {"queued", "running", "quarantined", "completed", "cancelled", "failed"}
)
RETRY_CLASSIFICATIONS = frozenset(
    {
        "cancelled",
        "contention",
        "dependency",
        "permanent",
        "policy_denied",
        "resource",
        "transient",
        "unknown",
    }
)
QUARANTINE_CLASSIFICATIONS = frozenset(
    {"ambiguous", "malformed", "partial", "policy_denied", "unsafe", "unsupported"}
)

R = TypeVar("R", bound="CanonicalIngestionRecord")


class IngestionContractError(ValueError):
    """Value-safe validation error for an ingestion record."""


class CanonicalIngestionRecord:
    """Shared deterministic JSON behavior for ingestion records."""

    _fields: ClassVar[frozenset[str]]

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a deterministic JSON-compatible mapping."""

        raise NotImplementedError

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls: type[R], payload: Mapping[str, Any]) -> R:
        """Parse one strict mapping."""

        raise NotImplementedError

    @classmethod
    def from_json(cls: type[R], payload: str) -> R:
        """Parse strict JSON with duplicate and non-finite values rejected."""

        return cls.from_dict(_parse_json(payload))


def load_ingestion_schema(name: str) -> dict[str, Any]:
    """Load one bundled ingestion-control JSON Schema by logical name."""

    normalized = name.removeprefix("ingestion_").removesuffix(".schema.json")
    if normalized == "job":
        normalized = "ingestion_job"
    if normalized not in INGESTION_SCHEMA_NAMES:
        raise KeyError("unknown ingestion schema")
    resource = resources.files(INGESTION_SCHEMA_PACKAGE).joinpath(
        f"ingestion_{normalized.removeprefix('ingestion_')}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_ingestion_schemas() -> dict[str, dict[str, Any]]:
    """Load every bundled ingestion-control JSON Schema."""

    return {name: load_ingestion_schema(name) for name in INGESTION_SCHEMA_NAMES}


@dataclass(frozen=True, slots=True)
class SourceManifest(CanonicalIngestionRecord):
    """Content-only identity for one bounded ingestion source set."""

    manifest_id: str
    source_id: str
    artifact_digests: tuple[str, ...]
    policy_digest: str
    pipeline_digest: str
    created_at: str
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "artifact_digests",
            "created_at",
            "manifest_id",
            "pipeline_digest",
            "policy_digest",
            "schema_version",
            "source_id",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.manifest_id, "manifest_id")
        _opaque(self.source_id, "source_id")
        if not isinstance(self.artifact_digests, tuple) or not self.artifact_digests:
            raise IngestionContractError("artifact_digests must be a non-empty tuple")
        for digest in self.artifact_digests:
            _digest(digest, "artifact_digests")
        if len(set(self.artifact_digests)) != len(self.artifact_digests):
            raise IngestionContractError("artifact_digests must be unique")
        object.__setattr__(
            self, "artifact_digests", tuple(sorted(self.artifact_digests))
        )
        _digest(self.policy_digest, "policy_digest")
        _digest(self.pipeline_digest, "pipeline_digest")
        _timestamp(self.created_at, "created_at")
        _schema(self.schema_version)

    @property
    def manifest_digest(self) -> str:
        """Return replay identity independent of record ID and creation time."""

        return canonical_digest(
            {
                "artifact_digests": list(self.artifact_digests),
                "pipeline_digest": self.pipeline_digest,
                "policy_digest": self.policy_digest,
                "schema_version": self.schema_version,
                "source_id": self.source_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical manifest mapping."""

        return {
            "artifact_digests": list(self.artifact_digests),
            "created_at": self.created_at,
            "manifest_id": self.manifest_id,
            "pipeline_digest": self.pipeline_digest,
            "policy_digest": self.policy_digest,
            "schema_version": self.schema_version,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceManifest":
        """Parse a strict source manifest."""

        data = _strict(payload, cls._fields, "source manifest")
        try:
            digests = data["artifact_digests"]
            if not isinstance(digests, list):
                raise IngestionContractError("artifact_digests must be an array")
            return cls(
                manifest_id=data["manifest_id"],
                source_id=data["source_id"],
                artifact_digests=tuple(digests),
                policy_digest=data["policy_digest"],
                pipeline_digest=data["pipeline_digest"],
                created_at=data["created_at"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError(
                "source manifest is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class IngestionJob(CanonicalIngestionRecord):
    """One append-only version of an ingestion job state."""

    job_id: str
    manifest_digest: str
    state: str
    checkpoint_sequence: int
    created_at: str
    updated_at: str
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "checkpoint_sequence",
            "created_at",
            "job_id",
            "manifest_digest",
            "schema_version",
            "state",
            "updated_at",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.job_id, "job_id")
        _digest(self.manifest_digest, "manifest_digest")
        if self.state not in JOB_STATES:
            raise IngestionContractError("job state is unsupported")
        _non_negative(self.checkpoint_sequence, "checkpoint_sequence")
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise IngestionContractError("updated_at cannot precede created_at")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical job mapping."""

        return {
            "checkpoint_sequence": self.checkpoint_sequence,
            "created_at": self.created_at,
            "job_id": self.job_id,
            "manifest_digest": self.manifest_digest,
            "schema_version": self.schema_version,
            "state": self.state,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IngestionJob":
        """Parse a strict job record."""

        data = _strict(payload, cls._fields, "ingestion job")
        try:
            return cls(
                job_id=data["job_id"],
                manifest_digest=data["manifest_digest"],
                state=data["state"],
                checkpoint_sequence=data["checkpoint_sequence"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError(
                "ingestion job is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class Checkpoint(CanonicalIngestionRecord):
    """One acknowledged idempotent job-step boundary."""

    checkpoint_id: str
    job_id: str
    manifest_digest: str
    step: str
    sequence: int
    input_digest: str
    output_digest: str
    completed_at: str
    committed_revision: int | None = None
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "checkpoint_id",
            "committed_revision",
            "completed_at",
            "input_digest",
            "job_id",
            "manifest_digest",
            "output_digest",
            "schema_version",
            "sequence",
            "step",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.checkpoint_id, "checkpoint_id")
        _opaque(self.job_id, "job_id")
        _digest(self.manifest_digest, "manifest_digest")
        _controlled(self.step, "step")
        _positive(self.sequence, "sequence")
        _digest(self.input_digest, "input_digest")
        _digest(self.output_digest, "output_digest")
        _timestamp(self.completed_at, "completed_at")
        if self.committed_revision is not None:
            _positive(self.committed_revision, "committed_revision")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical checkpoint mapping."""

        return {
            "checkpoint_id": self.checkpoint_id,
            "committed_revision": self.committed_revision,
            "completed_at": self.completed_at,
            "input_digest": self.input_digest,
            "job_id": self.job_id,
            "manifest_digest": self.manifest_digest,
            "output_digest": self.output_digest,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Checkpoint":
        """Parse a strict checkpoint."""

        data = _strict(payload, cls._fields, "checkpoint")
        try:
            return cls(
                checkpoint_id=data["checkpoint_id"],
                job_id=data["job_id"],
                manifest_digest=data["manifest_digest"],
                step=data["step"],
                sequence=data["sequence"],
                input_digest=data["input_digest"],
                output_digest=data["output_digest"],
                completed_at=data["completed_at"],
                committed_revision=data["committed_revision"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError(
                "checkpoint is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class Lease(CanonicalIngestionRecord):
    """Time-bounded exclusive ownership of one ingestion job."""

    lease_id: str
    job_id: str
    worker_id: str
    epoch: int
    acquired_at: str
    expires_at: str
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "acquired_at",
            "epoch",
            "expires_at",
            "job_id",
            "lease_id",
            "schema_version",
            "worker_id",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.lease_id, "lease_id")
        _opaque(self.job_id, "job_id")
        _opaque(self.worker_id, "worker_id")
        _positive(self.epoch, "epoch")
        acquired = _timestamp(self.acquired_at, "acquired_at")
        expires = _timestamp(self.expires_at, "expires_at")
        if expires <= acquired:
            raise IngestionContractError("expires_at must follow acquired_at")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical lease mapping."""

        return {
            "acquired_at": self.acquired_at,
            "epoch": self.epoch,
            "expires_at": self.expires_at,
            "job_id": self.job_id,
            "lease_id": self.lease_id,
            "schema_version": self.schema_version,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Lease":
        """Parse a strict lease."""

        data = _strict(payload, cls._fields, "lease")
        try:
            return cls(
                lease_id=data["lease_id"],
                job_id=data["job_id"],
                worker_id=data["worker_id"],
                epoch=data["epoch"],
                acquired_at=data["acquired_at"],
                expires_at=data["expires_at"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError("lease is missing a required field") from None


@dataclass(frozen=True, slots=True)
class Retry(CanonicalIngestionRecord):
    """Stable, value-free retry classification for one failed attempt."""

    retry_id: str
    job_id: str
    classification: str
    reason_code: str
    attempt: int
    recorded_at: str
    retry_after: str | None = None
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "attempt",
            "classification",
            "job_id",
            "reason_code",
            "recorded_at",
            "retry_after",
            "retry_id",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.retry_id, "retry_id")
        _opaque(self.job_id, "job_id")
        if self.classification not in RETRY_CLASSIFICATIONS:
            raise IngestionContractError("retry classification is unsupported")
        _controlled(self.reason_code, "reason_code")
        _positive(self.attempt, "attempt")
        recorded = _timestamp(self.recorded_at, "recorded_at")
        if self.retry_after is not None:
            retry_after = _timestamp(self.retry_after, "retry_after")
            if retry_after < recorded:
                raise IngestionContractError("retry_after cannot precede recorded_at")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical retry mapping."""

        return {
            "attempt": self.attempt,
            "classification": self.classification,
            "job_id": self.job_id,
            "reason_code": self.reason_code,
            "recorded_at": self.recorded_at,
            "retry_after": self.retry_after,
            "retry_id": self.retry_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Retry":
        """Parse a strict retry record."""

        data = _strict(payload, cls._fields, "retry")
        try:
            return cls(
                retry_id=data["retry_id"],
                job_id=data["job_id"],
                classification=data["classification"],
                reason_code=data["reason_code"],
                attempt=data["attempt"],
                recorded_at=data["recorded_at"],
                retry_after=data["retry_after"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError("retry is missing a required field") from None


@dataclass(frozen=True, slots=True)
class Cancellation(CanonicalIngestionRecord):
    """Explicit cancellation request without human-readable free text."""

    cancellation_id: str
    job_id: str
    actor_digest: str
    reason_code: str
    requested_at: str
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "actor_digest",
            "cancellation_id",
            "job_id",
            "reason_code",
            "requested_at",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.cancellation_id, "cancellation_id")
        _opaque(self.job_id, "job_id")
        _digest(self.actor_digest, "actor_digest")
        _controlled(self.reason_code, "reason_code")
        _timestamp(self.requested_at, "requested_at")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical cancellation mapping."""

        return {
            "actor_digest": self.actor_digest,
            "cancellation_id": self.cancellation_id,
            "job_id": self.job_id,
            "reason_code": self.reason_code,
            "requested_at": self.requested_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Cancellation":
        """Parse a strict cancellation record."""

        data = _strict(payload, cls._fields, "cancellation")
        try:
            return cls(
                cancellation_id=data["cancellation_id"],
                job_id=data["job_id"],
                actor_digest=data["actor_digest"],
                reason_code=data["reason_code"],
                requested_at=data["requested_at"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError(
                "cancellation is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class QuarantineResult(CanonicalIngestionRecord):
    """Untrusted parse result that cannot become a fact without promotion."""

    quarantine_id: str
    job_id: str
    manifest_digest: str
    classification: str
    reason_code: str
    candidate_count: int
    failure_count: int
    created_at: str
    output_digest: str | None = None
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "candidate_count",
            "classification",
            "created_at",
            "failure_count",
            "job_id",
            "manifest_digest",
            "output_digest",
            "quarantine_id",
            "reason_code",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.quarantine_id, "quarantine_id")
        _opaque(self.job_id, "job_id")
        _digest(self.manifest_digest, "manifest_digest")
        if self.classification not in QUARANTINE_CLASSIFICATIONS:
            raise IngestionContractError("quarantine classification is unsupported")
        _controlled(self.reason_code, "reason_code")
        _non_negative(self.candidate_count, "candidate_count")
        _positive(self.failure_count, "failure_count")
        _timestamp(self.created_at, "created_at")
        if self.output_digest is not None:
            _digest(self.output_digest, "output_digest")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical quarantine mapping."""

        return {
            "candidate_count": self.candidate_count,
            "classification": self.classification,
            "created_at": self.created_at,
            "failure_count": self.failure_count,
            "job_id": self.job_id,
            "manifest_digest": self.manifest_digest,
            "output_digest": self.output_digest,
            "quarantine_id": self.quarantine_id,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuarantineResult":
        """Parse a strict quarantine result."""

        data = _strict(payload, cls._fields, "quarantine result")
        try:
            return cls(
                quarantine_id=data["quarantine_id"],
                job_id=data["job_id"],
                manifest_digest=data["manifest_digest"],
                classification=data["classification"],
                reason_code=data["reason_code"],
                candidate_count=data["candidate_count"],
                failure_count=data["failure_count"],
                created_at=data["created_at"],
                output_digest=data["output_digest"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError(
                "quarantine result is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class QuarantinePromotion(CanonicalIngestionRecord):
    """Explicit review evidence promoting one quarantined result."""

    promotion_id: str
    quarantine_id: str
    reviewer_digest: str
    evidence_digest: str
    promoted_at: str
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "evidence_digest",
            "promoted_at",
            "promotion_id",
            "quarantine_id",
            "reviewer_digest",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.promotion_id, "promotion_id")
        _opaque(self.quarantine_id, "quarantine_id")
        _digest(self.reviewer_digest, "reviewer_digest")
        _digest(self.evidence_digest, "evidence_digest")
        _timestamp(self.promoted_at, "promoted_at")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical promotion mapping."""

        return {
            "evidence_digest": self.evidence_digest,
            "promoted_at": self.promoted_at,
            "promotion_id": self.promotion_id,
            "quarantine_id": self.quarantine_id,
            "reviewer_digest": self.reviewer_digest,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuarantinePromotion":
        """Parse a strict quarantine promotion."""

        data = _strict(payload, cls._fields, "quarantine promotion")
        try:
            return cls(
                promotion_id=data["promotion_id"],
                quarantine_id=data["quarantine_id"],
                reviewer_digest=data["reviewer_digest"],
                evidence_digest=data["evidence_digest"],
                promoted_at=data["promoted_at"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError(
                "quarantine promotion is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class ReplayAudit(CanonicalIngestionRecord):
    """Auditable created-or-no-op result for manifest registration."""

    replay_id: str
    manifest_digest: str
    job_id: str
    action: str
    recorded_at: str
    schema_version: str = INGESTION_SCHEMA_VERSION

    _fields = frozenset(
        {
            "action",
            "job_id",
            "manifest_digest",
            "recorded_at",
            "replay_id",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.replay_id, "replay_id")
        _digest(self.manifest_digest, "manifest_digest")
        _opaque(self.job_id, "job_id")
        if self.action not in {"created", "noop"}:
            raise IngestionContractError("replay action is unsupported")
        _timestamp(self.recorded_at, "recorded_at")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical replay mapping."""

        return {
            "action": self.action,
            "job_id": self.job_id,
            "manifest_digest": self.manifest_digest,
            "recorded_at": self.recorded_at,
            "replay_id": self.replay_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayAudit":
        """Parse a strict replay audit."""

        data = _strict(payload, cls._fields, "replay audit")
        try:
            return cls(
                replay_id=data["replay_id"],
                manifest_digest=data["manifest_digest"],
                job_id=data["job_id"],
                action=data["action"],
                recorded_at=data["recorded_at"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IngestionContractError(
                "replay audit is missing a required field"
            ) from None


def _strict(
    payload: Mapping[str, Any],
    fields: frozenset[str],
    record_name: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise IngestionContractError(f"{record_name} must be an object")
    data = dict(payload)
    if set(data) != fields:
        raise IngestionContractError(f"{record_name} has missing or unknown fields")
    return data


def _parse_json(payload: str) -> Mapping[str, Any]:
    if not isinstance(payload, str):
        raise IngestionContractError("record JSON must be text")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise IngestionContractError("record JSON is invalid") from None
    if not isinstance(value, Mapping):
        raise IngestionContractError("record JSON must contain an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite number")


def _opaque(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise IngestionContractError(f"{field_name} must be an opaque identifier")
    return value


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise IngestionContractError(f"{field_name} must be a SHA-256 digest")
    return value


def _controlled(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise IngestionContractError(f"{field_name} must be a controlled identifier")
    return value


def _timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise IngestionContractError(f"{field_name} must be timezone-aware")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise IngestionContractError(f"{field_name} must be timezone-aware") from None
    if parsed.tzinfo is None:
        raise IngestionContractError(f"{field_name} must be timezone-aware")
    return parsed


def _positive(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise IngestionContractError(f"{field_name} must be a positive integer")
    return value


def _non_negative(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise IngestionContractError(f"{field_name} must be non-negative")
    return value


def _schema(value: Any) -> str:
    if value != INGESTION_SCHEMA_VERSION:
        raise IngestionContractError("unsupported ingestion schema version")
    return value


__all__ = [
    "INGESTION_SCHEMA_VERSION",
    "JOB_STATES",
    "QUARANTINE_CLASSIFICATIONS",
    "RETRY_CLASSIFICATIONS",
    "Cancellation",
    "Checkpoint",
    "IngestionContractError",
    "IngestionJob",
    "Lease",
    "QuarantinePromotion",
    "QuarantineResult",
    "ReplayAudit",
    "Retry",
    "SourceManifest",
]
