"""Public protocols and typed outcomes for local Journey persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from openmed.clinical.journey_contracts import (
    JOURNEY_CONTRACT_SCHEMA_VERSION,
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    DatasetSnapshot,
    EvidenceLocator,
    ResolutionEvent,
    canonical_json,
)

LOCAL_STORE_SCHEMA_VERSION = "1.0.0"

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "credential",
        "note",
        "payload",
        "phi",
        "raw",
        "raw_text",
        "secret",
        "source_text",
        "text",
        "token",
        "vault",
    }
)

T = TypeVar("T")


class CommitStatusUnknown(RuntimeError):
    """Raised when a backend cannot prove whether a commit completed."""


class StoreState(str, Enum):
    """Explicit state for every Journey-store read or mutation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    DENIED = "denied"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class StoreResult(Generic[T]):
    """A value-safe result that never converts non-success into success."""

    state: StoreState
    value: T | None = field(default=None, repr=False)
    code: str | None = None
    created: bool = False
    revision: int | None = None

    def __post_init__(self) -> None:
        if self.code is not None and _CONTROLLED_RE.fullmatch(self.code) is None:
            raise ValueError("store result code must be a controlled identifier")
        if self.state is StoreState.SUCCESS and self.code is not None:
            raise ValueError("successful store results cannot carry an error code")
        if self.state is not StoreState.SUCCESS and self.code is None:
            raise ValueError("non-success store results require an error code")
        if self.created and self.state is not StoreState.SUCCESS:
            raise ValueError("only successful store results can create a record")
        if self.revision is not None and self.revision < 1:
            raise ValueError("store revision must be positive")

    @property
    def ok(self) -> bool:
        """Return whether the result is an explicit success."""

        return self.state is StoreState.SUCCESS

    @classmethod
    def success(
        cls,
        value: T,
        *,
        created: bool = False,
        revision: int | None = None,
    ) -> "StoreResult[T]":
        """Build a successful result."""

        return cls(
            state=StoreState.SUCCESS,
            value=value,
            created=created,
            revision=revision,
        )

    @classmethod
    def outcome(
        cls,
        state: StoreState,
        code: str,
        *,
        value: T | None = None,
        revision: int | None = None,
    ) -> "StoreResult[T]":
        """Build an explicit non-success result."""

        if state is StoreState.SUCCESS:
            raise ValueError("use StoreResult.success for success")
        return cls(state=state, value=value, code=code, revision=revision)


@dataclass(frozen=True, slots=True)
class StorePoint:
    """An exact committed revision used for point-in-time reads."""

    revision: int

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("store point revision must be a positive integer")


@dataclass(frozen=True, slots=True)
class CanonicalRecord:
    """An append-only pointer from one logical record to its selected fact."""

    canonical_id: str
    subject_id: str
    fact_id: str
    record_type: str
    state: str
    effective_at: str
    reason_code: str
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    schema_version: str = LOCAL_STORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("canonical_id", "subject_id", "fact_id"):
            if _OPAQUE_ID_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be an opaque identifier")
        for name in ("record_type", "state", "reason_code"):
            if _CONTROLLED_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a controlled identifier")
        _require_timestamp(self.effective_at, "effective_at")
        if self.schema_version != LOCAL_STORE_SCHEMA_VERSION:
            raise ValueError("unsupported canonical record schema version")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "canonical_id": self.canonical_id,
            "effective_at": self.effective_at,
            "fact_id": self.fact_id,
            "metadata": _plain(self.metadata),
            "reason_code": self.reason_code,
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "state": self.state,
            "subject_id": self.subject_id,
        }

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalRecord":
        """Parse a canonical record without echoing invalid values."""

        try:
            return cls(
                canonical_id=payload["canonical_id"],
                subject_id=payload["subject_id"],
                fact_id=payload["fact_id"],
                record_type=payload["record_type"],
                state=payload["state"],
                effective_at=payload["effective_at"],
                reason_code=payload["reason_code"],
                metadata=payload.get("metadata", {}),
                schema_version=payload["schema_version"],
            )
        except KeyError:
            raise ValueError("canonical record is missing a required field") from None


@dataclass(frozen=True, slots=True)
class CanonicalRecordVersion:
    """One persisted version of a canonical record."""

    record: CanonicalRecord = field(repr=False)
    version: int
    revision: int

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 1:
            raise ValueError("canonical record version must be positive")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("canonical record revision must be positive")


@dataclass(frozen=True, slots=True)
class JobMetadata:
    """PHI-free append-only metadata for one local job state."""

    job_id: str
    state: str
    recorded_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    schema_version: str = LOCAL_STORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if _OPAQUE_ID_RE.fullmatch(self.job_id) is None:
            raise ValueError("job_id must be an opaque identifier")
        if _CONTROLLED_RE.fullmatch(self.state) is None:
            raise ValueError("job state must be a controlled identifier")
        _require_timestamp(self.recorded_at, "recorded_at")
        if self.schema_version != LOCAL_STORE_SCHEMA_VERSION:
            raise ValueError("unsupported job metadata schema version")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "job_id": self.job_id,
            "metadata": _plain(self.metadata),
            "recorded_at": self.recorded_at,
            "schema_version": self.schema_version,
            "state": self.state,
        }

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JobMetadata":
        """Parse PHI-free job metadata."""

        try:
            return cls(
                job_id=payload["job_id"],
                state=payload["state"],
                recorded_at=payload["recorded_at"],
                metadata=payload.get("metadata", {}),
                schema_version=payload["schema_version"],
            )
        except KeyError:
            raise ValueError("job metadata is missing a required field") from None


@runtime_checkable
class StoragePolicy(Protocol):
    """Authorize a bounded store operation without inspecting clinical values."""

    def allows(self, operation: str, record_type: str) -> bool:
        """Return whether one controlled operation is allowed."""


@dataclass(frozen=True, slots=True)
class AllowAllStoragePolicy:
    """Default local policy that permits explicitly invoked store operations."""

    def allows(self, operation: str, record_type: str) -> bool:
        """Allow the requested local operation."""

        return True


@dataclass(frozen=True, slots=True)
class DenyStorageOperations:
    """Deny a fixed set of controlled operation names."""

    operations: frozenset[str]

    def __post_init__(self) -> None:
        if any(_CONTROLLED_RE.fullmatch(value) is None for value in self.operations):
            raise ValueError("denied operations must be controlled identifiers")

    def allows(self, operation: str, record_type: str) -> bool:
        """Return false for a denied operation."""

        return operation not in self.operations


@runtime_checkable
class ArtifactStore(Protocol):
    """Content-addressed artifact bytes under a bounded namespace."""

    def put_bytes(
        self, artifact: ClinicalArtifact, content: bytes
    ) -> StoreResult[ClinicalArtifact]:
        """Persist verified artifact bytes idempotently."""

    def get_bytes(self, content_hash: str) -> StoreResult[bytes]:
        """Read and verify one content-addressed blob."""


@runtime_checkable
class CompensatingArtifactStore(ArtifactStore, Protocol):
    """Artifact store that can compensate a newly created object."""

    def discard_if_created(self, content_hash: str) -> None:
        """Discard one digest-derived object after metadata rollback."""


@runtime_checkable
class EvidenceStore(Protocol):
    """Append-only evidence-locator persistence."""

    def put_evidence(
        self, locator: EvidenceLocator, *, committed_at: str
    ) -> StoreResult[EvidenceLocator]:
        """Persist an evidence locator idempotently."""

    def get_evidence(
        self, locator_id: str, *, as_of: StorePoint | None = None
    ) -> StoreResult[EvidenceLocator]:
        """Read an evidence locator at a committed revision."""


@runtime_checkable
class FactStore(Protocol):
    """Append-only clinical-fact persistence."""

    def put_fact(
        self, fact: ClinicalFact, *, committed_at: str
    ) -> StoreResult[ClinicalFact]:
        """Persist a fact version idempotently."""

    def get_fact(
        self, fact_id: str, *, as_of: StorePoint | None = None
    ) -> StoreResult[ClinicalFact]:
        """Read a fact visible at a committed revision."""


@runtime_checkable
class CanonicalStore(Protocol):
    """Versioned canonical-record pointers."""

    def put_canonical(
        self, record: CanonicalRecord, *, committed_at: str
    ) -> StoreResult[CanonicalRecordVersion]:
        """Append a canonical-record version."""

    def get_canonical(
        self, canonical_id: str, *, as_of: StorePoint | None = None
    ) -> StoreResult[CanonicalRecordVersion]:
        """Read the canonical state at one revision."""


@runtime_checkable
class ResolutionStore(Protocol):
    """Append-only conflict and resolution-event persistence."""

    def put_conflict(
        self, conflict: ConflictSet, *, committed_at: str
    ) -> StoreResult[ConflictSet]:
        """Persist a conflict set idempotently."""

    def put_resolution(
        self, resolution: ResolutionEvent, *, committed_at: str
    ) -> StoreResult[ResolutionEvent]:
        """Persist a resolution event idempotently."""


@runtime_checkable
class DatasetStore(Protocol):
    """Append-only governed dataset snapshots."""

    def put_dataset(
        self, snapshot: DatasetSnapshot, *, committed_at: str
    ) -> StoreResult[DatasetSnapshot]:
        """Persist a dataset snapshot idempotently."""

    def get_dataset(
        self, snapshot_id: str, *, as_of: StorePoint | None = None
    ) -> StoreResult[DatasetSnapshot]:
        """Read a snapshot visible at a committed revision."""


@runtime_checkable
class JobMetadataStore(Protocol):
    """Append-only PHI-free local job metadata."""

    def put_job(
        self, job: JobMetadata, *, committed_at: str
    ) -> StoreResult[JobMetadata]:
        """Append one job-state version."""

    def get_job(
        self, job_id: str, *, as_of: StorePoint | None = None
    ) -> StoreResult[JobMetadata]:
        """Read job metadata at one revision."""

    def list_job_versions(
        self, job_id: str, *, as_of: StorePoint | None = None
    ) -> StoreResult[tuple[JobMetadata, ...]]:
        """Read append-only job metadata versions through one revision."""


@runtime_checkable
class PointInTimeReader(Protocol):
    """Revision-bounded reads over immutable facts and canonical history."""

    @property
    def latest_revision(self) -> int | None:
        """Return the latest committed revision."""

    def get_fact(
        self, fact_id: str, *, as_of: StorePoint | None = None
    ) -> StoreResult[ClinicalFact]:
        """Read a fact visible at a committed revision."""

    def get_canonical(
        self, canonical_id: str, *, as_of: StorePoint | None = None
    ) -> StoreResult[CanonicalRecordVersion]:
        """Read canonical state visible at a committed revision."""


@runtime_checkable
class JourneyStoreTransaction(Protocol):
    """One atomic source-to-fact graph transaction."""

    @property
    def revision(self) -> int:
        """Return the transaction revision."""

    def put_artifact(self, artifact: ClinicalArtifact) -> StoreResult[ClinicalArtifact]:
        """Persist artifact metadata inside this transaction."""

    def put_evidence(self, locator: EvidenceLocator) -> StoreResult[EvidenceLocator]:
        """Persist evidence inside this transaction."""

    def put_fact(self, fact: ClinicalFact) -> StoreResult[ClinicalFact]:
        """Persist a fact inside this transaction."""

    def put_conflict(self, conflict: ConflictSet) -> StoreResult[ConflictSet]:
        """Persist a conflict inside this transaction."""

    def put_resolution(
        self, resolution: ResolutionEvent
    ) -> StoreResult[ResolutionEvent]:
        """Persist a resolution inside this transaction."""

    def put_dataset(self, snapshot: DatasetSnapshot) -> StoreResult[DatasetSnapshot]:
        """Persist a dataset snapshot inside this transaction."""

    def put_canonical(
        self, record: CanonicalRecord
    ) -> StoreResult[CanonicalRecordVersion]:
        """Persist a canonical-record version inside this transaction."""

    def put_job(self, job: JobMetadata) -> StoreResult[JobMetadata]:
        """Persist job metadata inside this transaction."""


@runtime_checkable
class TransactionalJourneyStore(
    EvidenceStore,
    FactStore,
    CanonicalStore,
    ResolutionStore,
    DatasetStore,
    JobMetadataStore,
    PointInTimeReader,
    Protocol,
):
    """Complete backend-neutral metadata-store surface."""

    def transaction(
        self, *, committed_at: str
    ) -> AbstractContextManager[JourneyStoreTransaction]:
        """Open an atomic append-only transaction."""


def _freeze_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be an object")
    _reject_sensitive_keys(value)
    normalized = canonical_json(value)
    parsed = json.loads(normalized)
    return _freeze(parsed)


def _reject_sensitive_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if key.casefold() in _SENSITIVE_METADATA_KEYS:
                raise ValueError("metadata contains a prohibited sensitive field")
            _reject_sensitive_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_keys(item)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _require_timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a timezone-aware timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise ValueError(f"{field_name} must be a timezone-aware timestamp") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware timestamp")
    return value


def assert_contract_compatibility() -> None:
    """Fail if the local store and Journey contracts drift by major version."""

    if JOURNEY_CONTRACT_SCHEMA_VERSION.split(".", 1)[0] != "1":
        raise RuntimeError("unsupported Journey contract schema major version")
