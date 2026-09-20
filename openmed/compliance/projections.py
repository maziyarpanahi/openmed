"""Separated identified/de-identified projections with fail-closed policy.

The public records in this module contain controlled metadata, digests, counts,
and timestamps only. Projection bytes and transform material are excluded from
representations, audit events, exceptions, and persisted public records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from openmed.clinical.consent_cache import (
    ConsentCache,
    ConsentInvalidationEvent,
    fingerprint_consent_revision,
    fingerprint_consent_scope,
)
from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.structured.store import StoreResult, StoreState

from .data_use import DataUseAction, DataUsePolicy, DataUseTag

PROJECTION_SCHEMA_VERSION = "1.0.0"
PROJECTION_STORAGE_SCHEMA_VERSION = 1
PROJECTION_AUDIT_SCHEMA_VERSION = 1
PROJECTION_SCHEMA_PACKAGE = "openmed.core.schemas.json"
PROJECTION_SCHEMA_NAMES = (
    "record",
    "policy_decision",
    "audit_event",
    "transform_record",
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class ProjectionContractError(ValueError):
    """Value-safe validation error for projection boundary contracts."""


def load_projection_schema(name: str) -> dict[str, Any]:
    """Load one bundled projection JSON Schema by logical name."""

    normalized = name.removeprefix("projection_").removesuffix(".schema.json")
    if normalized not in PROJECTION_SCHEMA_NAMES:
        raise KeyError("unknown projection schema")
    resource = resources.files(PROJECTION_SCHEMA_PACKAGE).joinpath(
        f"projection_{normalized}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_projection_schemas() -> dict[str, dict[str, Any]]:
    """Load all bundled projection JSON Schemas."""

    return {name: load_projection_schema(name) for name in PROJECTION_SCHEMA_NAMES}


class ProjectionNamespace(str, Enum):
    """Physically separated storage namespaces."""

    IDENTIFIED = "identified"
    DEIDENTIFIED = "deidentified"


class ProjectionOperation(str, Enum):
    """Audited operations supported by projection APIs."""

    READ = "read"
    WRITE = "write"
    EXPORT = "export"
    CORRECT = "correct"
    REVIEW = "review"


class ProjectionPolicyOutcome(str, Enum):
    """Explicit policy result that never collapses review into allow."""

    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ProjectionRecord:
    """Value-free metadata for one immutable projection version."""

    projection_id: str
    namespace: ProjectionNamespace
    version: int
    content_digest: str
    byte_size: int
    consent_scope_fingerprint: str
    consent_revision_fingerprint: str
    created_at: str
    schema_version: str = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _opaque_id(self.projection_id, "projection_id")
        object.__setattr__(self, "namespace", _namespace(self.namespace))
        _positive(self.version, "version")
        _digest(self.content_digest, "content_digest")
        _non_negative(self.byte_size, "byte_size")
        _digest(self.consent_scope_fingerprint, "consent_scope_fingerprint")
        _digest(self.consent_revision_fingerprint, "consent_revision_fingerprint")
        _timestamp(self.created_at, "created_at")
        _schema_version(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata without projection bytes."""

        return {
            "byte_size": self.byte_size,
            "consent_revision_fingerprint": self.consent_revision_fingerprint,
            "consent_scope_fingerprint": self.consent_scope_fingerprint,
            "content_digest": self.content_digest,
            "created_at": self.created_at,
            "namespace": self.namespace.value,
            "projection_id": self.projection_id,
            "schema_version": self.schema_version,
            "version": self.version,
        }

    def to_json(self) -> str:
        """Return canonical JSON metadata."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectionRecord":
        """Parse one strict projection record without echoing values."""

        data = _strict_mapping(
            payload,
            frozenset(
                {
                    "byte_size",
                    "consent_revision_fingerprint",
                    "consent_scope_fingerprint",
                    "content_digest",
                    "created_at",
                    "namespace",
                    "projection_id",
                    "schema_version",
                    "version",
                }
            ),
            "projection record",
        )
        try:
            return cls(**data)
        except KeyError:
            raise ProjectionContractError(
                "projection record is missing a required field"
            ) from None

    @classmethod
    def from_json(cls, payload: str) -> "ProjectionRecord":
        """Parse strict canonical JSON."""

        return cls.from_dict(_strict_json(payload, "projection record"))


@dataclass(frozen=True, slots=True)
class ProjectionPayload:
    """One projection record and its verified bytes."""

    record: ProjectionRecord
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise ProjectionContractError("projection content must be bytes")
        if len(self.content) != self.record.byte_size:
            raise ProjectionContractError("projection content size differs")
        if _sha256(self.content) != self.record.content_digest:
            raise ProjectionContractError("projection content digest differs")


@dataclass(frozen=True, slots=True)
class TransformRecord:
    """Public digest-only metadata for one protected vault transform."""

    transform_id: str
    transform_type: str
    source_digest: str
    output_digest: str
    created_at: str
    schema_version: str = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _opaque_id(self.transform_id, "transform_id")
        if self.transform_type not in {"pseudonymization", "date_shift"}:
            raise ProjectionContractError("transform type is unsupported")
        _digest(self.source_digest, "source_digest")
        _digest(self.output_digest, "output_digest")
        _timestamp(self.created_at, "created_at")
        _schema_version(self.schema_version)

    def to_dict(self) -> dict[str, str]:
        """Return digest-only transform metadata."""

        return {
            "created_at": self.created_at,
            "output_digest": self.output_digest,
            "schema_version": self.schema_version,
            "source_digest": self.source_digest,
            "transform_id": self.transform_id,
            "transform_type": self.transform_type,
        }

    def to_json(self) -> str:
        """Return canonical digest-only JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransformRecord":
        """Parse one strict transform record."""

        data = _strict_mapping(
            payload,
            frozenset(
                {
                    "created_at",
                    "output_digest",
                    "schema_version",
                    "source_digest",
                    "transform_id",
                    "transform_type",
                }
            ),
            "transform record",
        )
        try:
            return cls(**data)
        except KeyError:
            raise ProjectionContractError(
                "transform record is missing a required field"
            ) from None

    @classmethod
    def from_json(cls, payload: str) -> "TransformRecord":
        """Parse strict transform JSON."""

        return cls.from_dict(_strict_json(payload, "transform record"))


@dataclass(frozen=True, slots=True)
class ProjectionPolicyRequest:
    """Controlled, patient-free inputs for one access decision."""

    operation: ProjectionOperation
    namespace: ProjectionNamespace
    purpose: str
    role: str
    attributes: tuple[str, ...] = ()
    data_use_tags: tuple[DataUseTag | str, ...] = ()
    consent_state: str = "active"
    schema_version: str = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(self, "namespace", _namespace(self.namespace))
        _controlled(self.purpose, "purpose")
        _controlled(self.role, "role")
        object.__setattr__(
            self,
            "attributes",
            _controlled_tuple(self.attributes, "attributes"),
        )
        try:
            tags = tuple(
                sorted(
                    {
                        tag.value
                        if isinstance(tag, DataUseTag)
                        else DataUseTag(tag).value
                        for tag in self.data_use_tags
                    }
                )
            )
        except (TypeError, ValueError):
            raise ProjectionContractError("data-use tag is unsupported") from None
        object.__setattr__(self, "data_use_tags", tags)
        if self.consent_state not in {"active", "unknown", "withdrawn"}:
            raise ProjectionContractError("consent state is unsupported")
        _schema_version(self.schema_version)

    @property
    def request_digest(self) -> str:
        """Return a stable patient-free decision-input digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic policy metadata."""

        return {
            "attributes": list(self.attributes),
            "consent_state": self.consent_state,
            "data_use_tags": list(self.data_use_tags),
            "namespace": self.namespace.value,
            "operation": self.operation.value,
            "purpose": self.purpose,
            "role": self.role,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ProjectionPolicyDecision:
    """Typed allow, deny, or review result for one request."""

    decision_id: str
    request_digest: str
    outcome: ProjectionPolicyOutcome
    reason_code: str
    decided_at: str
    schema_version: str = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _opaque_id(self.decision_id, "decision_id")
        _digest(self.request_digest, "request_digest")
        try:
            outcome = (
                self.outcome
                if isinstance(self.outcome, ProjectionPolicyOutcome)
                else ProjectionPolicyOutcome(self.outcome)
            )
        except ValueError:
            raise ProjectionContractError("policy outcome is unsupported") from None
        object.__setattr__(self, "outcome", outcome)
        _controlled(self.reason_code, "reason_code")
        _timestamp(self.decided_at, "decided_at")
        _schema_version(self.schema_version)

    @property
    def allowed(self) -> bool:
        """Return whether the request may proceed."""

        return self.outcome is ProjectionPolicyOutcome.ALLOW

    def to_dict(self) -> dict[str, str]:
        """Return a value-free policy decision."""

        return {
            "decided_at": self.decided_at,
            "decision_id": self.decision_id,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
            "request_digest": self.request_digest,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return canonical policy-decision JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectionPolicyDecision":
        """Parse one strict policy decision."""

        data = _strict_mapping(
            payload,
            frozenset(
                {
                    "decided_at",
                    "decision_id",
                    "outcome",
                    "reason_code",
                    "request_digest",
                    "schema_version",
                }
            ),
            "projection policy decision",
        )
        try:
            return cls(**data)
        except KeyError:
            raise ProjectionContractError(
                "projection policy decision is missing a required field"
            ) from None

    @classmethod
    def from_json(cls, payload: str) -> "ProjectionPolicyDecision":
        """Parse strict policy-decision JSON."""

        return cls.from_dict(_strict_json(payload, "projection policy decision"))


@dataclass(frozen=True, slots=True)
class ProjectionAuditEvent:
    """PHI-safe event for projection access and policy outcomes."""

    event_id: str
    occurred_at: str
    operation: ProjectionOperation
    namespace: ProjectionNamespace
    outcome: str
    reason_code: str
    request_digest: str
    resource_digest: str | None = None
    schema_version: str = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _opaque_id(self.event_id, "event_id")
        _timestamp(self.occurred_at, "occurred_at")
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(self, "namespace", _namespace(self.namespace))
        if self.outcome not in {"allow", "deny", "review", "failure"}:
            raise ProjectionContractError("audit outcome is unsupported")
        _controlled(self.reason_code, "reason_code")
        _digest(self.request_digest, "request_digest")
        if self.resource_digest is not None:
            _digest(self.resource_digest, "resource_digest")
        _schema_version(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return audit metadata without subject or reviewer identity."""

        return {
            "event_id": self.event_id,
            "namespace": self.namespace.value,
            "occurred_at": self.occurred_at,
            "operation": self.operation.value,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "request_digest": self.request_digest,
            "resource_digest": self.resource_digest,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return canonical value-free audit JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectionAuditEvent":
        """Parse one strict audit event."""

        data = _strict_mapping(
            payload,
            frozenset(
                {
                    "event_id",
                    "namespace",
                    "occurred_at",
                    "operation",
                    "outcome",
                    "reason_code",
                    "request_digest",
                    "resource_digest",
                    "schema_version",
                }
            ),
            "projection audit event",
        )
        try:
            return cls(**data)
        except KeyError:
            raise ProjectionContractError(
                "projection audit event is missing a required field"
            ) from None

    @classmethod
    def from_json(cls, payload: str) -> "ProjectionAuditEvent":
        """Parse strict audit-event JSON."""

        return cls.from_dict(_strict_json(payload, "projection audit event"))


ConsentHook = Callable[[ProjectionPolicyRequest], str]


class ProjectionPolicy:
    """Purpose, role, attribute, consent, and data-use policy evaluator."""

    _identified_roles = frozenset({"clinician", "data_steward", "privacy_officer"})
    _deidentified_roles = frozenset(
        {"clinician", "data_steward", "privacy_officer", "researcher"}
    )
    _purposes = frozenset({"care", "operations", "privacy", "quality", "research"})
    _correction_roles = frozenset({"clinician", "data_steward", "privacy_officer"})

    def __init__(
        self,
        *,
        data_use_policy: DataUsePolicy | None = None,
        consent_hook: ConsentHook | None = None,
    ) -> None:
        self.data_use_policy = data_use_policy or DataUsePolicy()
        self.consent_hook = consent_hook

    def evaluate(
        self,
        request: ProjectionPolicyRequest,
        *,
        decided_at: str,
    ) -> ProjectionPolicyDecision:
        """Return an explicit policy decision without raising on hook failure."""

        _timestamp(decided_at, "decided_at")
        outcome, reason = self._evaluate(request)
        digest = request.request_digest
        decision_digest = canonical_digest(
            {
                "decided_at": decided_at,
                "outcome": outcome.value,
                "reason_code": reason,
                "request_digest": digest,
            }
        )
        return ProjectionPolicyDecision(
            decision_id=f"decision_{decision_digest.removeprefix('sha256:')[:32]}",
            request_digest=digest,
            outcome=outcome,
            reason_code=reason,
            decided_at=decided_at,
        )

    def _evaluate(
        self,
        request: ProjectionPolicyRequest,
    ) -> tuple[ProjectionPolicyOutcome, str]:
        if request.consent_state == "withdrawn":
            return ProjectionPolicyOutcome.DENY, "consent_withdrawn"
        if request.consent_state == "unknown":
            return ProjectionPolicyOutcome.REVIEW, "consent_unknown"
        if self.consent_hook is not None:
            try:
                hook_outcome = self.consent_hook(request)
            except Exception:
                return ProjectionPolicyOutcome.DENY, "consent_hook_failed"
            if hook_outcome == "deny":
                return ProjectionPolicyOutcome.DENY, "consent_denied"
            if hook_outcome in {"review", "unknown"}:
                return ProjectionPolicyOutcome.REVIEW, "consent_review_required"
            if hook_outcome != "allow":
                return ProjectionPolicyOutcome.DENY, "consent_hook_invalid"

        if request.purpose not in self._purposes:
            return ProjectionPolicyOutcome.DENY, "purpose_unsupported"
        allowed_roles = (
            self._identified_roles
            if request.namespace is ProjectionNamespace.IDENTIFIED
            else self._deidentified_roles
        )
        if request.role not in allowed_roles:
            return ProjectionPolicyOutcome.DENY, "role_denied"
        if request.namespace is ProjectionNamespace.IDENTIFIED and request.purpose in {
            "research",
            "quality",
        }:
            return ProjectionPolicyOutcome.DENY, "purpose_denied"
        if (
            request.namespace is ProjectionNamespace.IDENTIFIED
            and "identified_access" not in request.attributes
        ):
            return ProjectionPolicyOutcome.DENY, "identified_attribute_required"
        if (
            request.operation
            in {ProjectionOperation.CORRECT, ProjectionOperation.REVIEW}
            and request.role not in self._correction_roles
        ):
            return ProjectionPolicyOutcome.DENY, "role_denied"

        data_use_action = (
            DataUseAction.EXPORT
            if request.operation is ProjectionOperation.EXPORT
            else DataUseAction.PROCESS
        )
        evaluation = self.data_use_policy.evaluate(
            request.data_use_tags,
            data_use_action,
        )
        if not evaluation.allowed:
            return ProjectionPolicyOutcome.DENY, "data_use_denied"
        if request.operation is ProjectionOperation.EXPORT:
            if "export_approved" not in request.attributes:
                return ProjectionPolicyOutcome.REVIEW, "export_review_required"
            if (
                request.namespace is ProjectionNamespace.IDENTIFIED
                and "identified_export_approved" not in request.attributes
            ):
                return (
                    ProjectionPolicyOutcome.REVIEW,
                    "identified_export_review_required",
                )
        return ProjectionPolicyOutcome.ALLOW, "policy_allowed"


@runtime_checkable
class TransformVault(Protocol):
    """Protected storage interface available only to the identified API."""

    def record_transform(
        self,
        record: TransformRecord,
        material: bytes,
    ) -> StoreResult[TransformRecord]:
        """Store protected transform material under digest-only metadata."""

    def resolve_transform(self, transform_id: str) -> StoreResult[bytes]:
        """Resolve protected material inside an authorized boundary."""


class InMemoryTransformVault:
    """Process-local test vault; production callers should supply protected storage."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[TransformRecord, bytes]] = {}
        self._lock = threading.RLock()

    def record_transform(
        self,
        record: TransformRecord,
        material: bytes,
    ) -> StoreResult[TransformRecord]:
        """Store bytes without exposing them through metadata or representation."""

        if not isinstance(material, bytes) or not material:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_transform_material")
        with self._lock:
            existing = self._records.get(record.transform_id)
            if existing is not None:
                if existing[0] == record and existing[1] == material:
                    return StoreResult.success(record, created=False)
                return StoreResult.outcome(StoreState.CONFLICT, "transform_conflict")
            self._records[record.transform_id] = (record, bytes(material))
        return StoreResult.success(record, created=True)

    def resolve_transform(self, transform_id: str) -> StoreResult[bytes]:
        """Return protected material without including it in failure diagnostics."""

        if not _valid_opaque_id(transform_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_transform_id")
        with self._lock:
            entry = self._records.get(transform_id)
        if entry is None:
            return StoreResult.outcome(StoreState.UNKNOWN, "transform_not_found")
        return StoreResult.success(bytes(entry[1]))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(entries={len(self._records)})"


T = TypeVar("T")


class ProjectionCompatibilityError(RuntimeError):
    """Raised when persisted projection schema cannot be opened safely."""


class ProjectionStorageError(RuntimeError):
    """Value-safe projection storage lifecycle failure."""


_PROJECTION_MIGRATION_SQL = (
    """
    CREATE TABLE projection_revisions (
        revision INTEGER PRIMARY KEY AUTOINCREMENT,
        committed_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE projections (
        projection_id TEXT PRIMARY KEY
    )
    """,
    """
    CREATE TABLE projection_versions (
        projection_id TEXT NOT NULL REFERENCES projections(projection_id),
        version INTEGER NOT NULL,
        content_digest TEXT NOT NULL,
        byte_size INTEGER NOT NULL,
        consent_scope_fingerprint TEXT NOT NULL,
        consent_revision_fingerprint TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload_hash TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_revision INTEGER NOT NULL
            REFERENCES projection_revisions(revision),
        PRIMARY KEY (projection_id, version),
        UNIQUE (projection_id, payload_hash)
    )
    """,
    """
    CREATE INDEX projection_versions_digest_idx
    ON projection_versions(content_digest, projection_id, version)
    """,
)
_PROJECTION_MIGRATION_CHECKSUM = canonical_digest(_PROJECTION_MIGRATION_SQL)

_AUDIT_MIGRATION_SQL = (
    """
    CREATE TABLE projection_audit_events (
        event_id TEXT PRIMARY KEY,
        occurred_at TEXT NOT NULL,
        operation TEXT NOT NULL,
        namespace TEXT NOT NULL,
        outcome TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        payload_hash TEXT NOT NULL UNIQUE,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX projection_audit_time_idx
    ON projection_audit_events(occurred_at, event_id)
    """,
)
_AUDIT_MIGRATION_CHECKSUM = canonical_digest(_AUDIT_MIGRATION_SQL)


class ProjectionAuditLog:
    """Append-only SQLite log containing value-free projection events."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._closed = False
        _prepare_private_parent(self._path)
        try:
            self._connection = sqlite3.connect(
                self._path,
                isolation_level=None,
                check_same_thread=False,
            )
            os.chmod(self._path, 0o600)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            _apply_single_migration(
                self._connection,
                name="projection_audit_log",
                checksum=_AUDIT_MIGRATION_CHECKSUM,
                statements=_AUDIT_MIGRATION_SQL,
            )
        except ProjectionCompatibilityError:
            self._close_failed()
            raise
        except (OSError, sqlite3.Error) as exc:
            self._close_failed()
            raise ProjectionStorageError(
                "projection audit log cannot be initialized"
            ) from exc

    def append(self, event: ProjectionAuditEvent) -> StoreResult[ProjectionAuditEvent]:
        """Append one event idempotently and reject conflicting identifiers."""

        payload_hash = canonical_digest(event.to_dict())
        with self._lock:
            try:
                existing = self._connection.execute(
                    "SELECT payload_hash FROM projection_audit_events "
                    "WHERE event_id = ?",
                    (event.event_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["payload_hash"]) == payload_hash:
                        return StoreResult.success(event, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "audit_event_conflict",
                    )
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "INSERT INTO projection_audit_events("
                    "event_id, occurred_at, operation, namespace, outcome, "
                    "reason_code, payload_hash, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.occurred_at,
                        event.operation.value,
                        event.namespace.value,
                        event.outcome,
                        event.reason_code,
                        payload_hash,
                        event.to_json(),
                    ),
                )
                self._connection.execute("COMMIT")
                return StoreResult.success(event, created=True)
            except sqlite3.Error:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                return StoreResult.outcome(StoreState.FAILURE, "audit_write_failed")

    def events(self) -> StoreResult[tuple[ProjectionAuditEvent, ...]]:
        """Read all value-free events in deterministic order."""

        with self._lock:
            try:
                rows = self._connection.execute(
                    "SELECT payload_json FROM projection_audit_events "
                    "ORDER BY occurred_at, event_id"
                ).fetchall()
                events = tuple(
                    ProjectionAuditEvent.from_json(str(row["payload_json"]))
                    for row in rows
                )
            except (sqlite3.Error, ProjectionContractError):
                return StoreResult.outcome(StoreState.FAILURE, "audit_read_failed")
        return StoreResult.success(events)

    def integrity_check(self) -> StoreResult[int]:
        """Verify every event's canonical payload hash."""

        with self._lock:
            try:
                rows = self._connection.execute(
                    "SELECT payload_hash, payload_json FROM projection_audit_events"
                ).fetchall()
                for row in rows:
                    event = ProjectionAuditEvent.from_json(str(row["payload_json"]))
                    if canonical_digest(event.to_dict()) != str(row["payload_hash"]):
                        return StoreResult.outcome(
                            StoreState.FAILURE,
                            "audit_integrity_failed",
                        )
            except (sqlite3.Error, ProjectionContractError):
                return StoreResult.outcome(StoreState.FAILURE, "audit_integrity_failed")
        return StoreResult.success(len(rows))

    def close(self) -> None:
        """Checkpoint and close the audit database."""

        with self._lock:
            if self._closed:
                return
            try:
                self._connection.execute("PRAGMA wal_checkpoint(FULL)")
            finally:
                self._connection.close()
                self._closed = True

    def _close_failed(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()


class _ProjectionNamespaceStore:
    """Private content-addressed store for exactly one namespace."""

    def __init__(self, root: Path, namespace: ProjectionNamespace) -> None:
        self._root = root
        self.namespace = namespace
        self._objects = root / "objects" / "sha256"
        self._path = root / "metadata.sqlite3"
        self._lock = threading.RLock()
        self._closed = False
        _prepare_private_directory(root)
        _prepare_private_directory(self._objects)
        try:
            self._connection = sqlite3.connect(
                self._path,
                isolation_level=None,
                check_same_thread=False,
            )
            os.chmod(self._path, 0o600)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            _apply_single_migration(
                self._connection,
                name=f"{namespace.value}_projection_store",
                checksum=_PROJECTION_MIGRATION_CHECKSUM,
                statements=_PROJECTION_MIGRATION_SQL,
            )
        except ProjectionCompatibilityError:
            self._close_failed()
            raise
        except (OSError, sqlite3.Error) as exc:
            self._close_failed()
            raise ProjectionStorageError(
                "projection namespace cannot be initialized"
            ) from exc

    def write(
        self,
        record: ProjectionRecord,
        content: bytes,
        *,
        correcting: bool,
    ) -> StoreResult[ProjectionRecord]:
        """Commit one projection version with content-addressed bytes."""

        if record.namespace is not self.namespace:
            return StoreResult.outcome(StoreState.DENIED, "namespace_mismatch")
        try:
            ProjectionPayload(record, content)
        except ProjectionContractError:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_projection_content")
        with self._lock:
            latest = self._latest_record(record.projection_id)
            if latest is None:
                if correcting:
                    return StoreResult.outcome(
                        StoreState.UNKNOWN,
                        "projection_not_found",
                    )
                expected_version = 1
            else:
                if not correcting:
                    if latest == record:
                        verified = self._read_blob(record.content_digest)
                        if verified.ok:
                            return StoreResult.success(record, created=False)
                        return StoreResult.outcome(
                            verified.state,
                            verified.code or "projection_read_failed",
                        )
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "projection_exists",
                    )
                expected_version = latest.version + 1
                if latest.content_digest == record.content_digest:
                    return StoreResult.success(latest, created=False)
            if record.version != expected_version:
                return StoreResult.outcome(
                    StoreState.CONFLICT,
                    "projection_version_conflict",
                )

            blob_created = False
            try:
                blob_created = self._put_blob(record.content_digest, content)
                self._connection.execute("BEGIN IMMEDIATE")
                revision = self._connection.execute(
                    "INSERT INTO projection_revisions(committed_at) VALUES (?)",
                    (record.created_at,),
                ).lastrowid
                if revision is None:
                    raise sqlite3.IntegrityError
                if latest is None:
                    self._connection.execute(
                        "INSERT INTO projections(projection_id) VALUES (?)",
                        (record.projection_id,),
                    )
                self._connection.execute(
                    "INSERT INTO projection_versions("
                    "projection_id, version, content_digest, byte_size, "
                    "consent_scope_fingerprint, consent_revision_fingerprint, "
                    "created_at, payload_hash, payload_json, created_revision) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.projection_id,
                        record.version,
                        record.content_digest,
                        record.byte_size,
                        record.consent_scope_fingerprint,
                        record.consent_revision_fingerprint,
                        record.created_at,
                        canonical_digest(record.to_dict()),
                        record.to_json(),
                        int(revision),
                    ),
                )
                self._connection.execute("COMMIT")
                return StoreResult.success(
                    record,
                    created=True,
                    revision=int(revision),
                )
            except sqlite3.IntegrityError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                self._discard_unreferenced_blob(record.content_digest, blob_created)
                return StoreResult.outcome(StoreState.CONFLICT, "projection_conflict")
            except (OSError, ProjectionStorageError, sqlite3.Error):
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                self._discard_unreferenced_blob(record.content_digest, blob_created)
                return StoreResult.outcome(
                    StoreState.FAILURE, "projection_write_failed"
                )

    def read(self, projection_id: str) -> StoreResult[ProjectionPayload]:
        """Read the latest version and verify its namespace-local bytes."""

        if not _valid_opaque_id(projection_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_projection_id")
        with self._lock:
            try:
                record = self._latest_record(projection_id)
            except (ProjectionContractError, sqlite3.Error):
                return StoreResult.outcome(StoreState.FAILURE, "projection_read_failed")
            if record is None:
                return StoreResult.outcome(StoreState.UNKNOWN, "projection_not_found")
            content = self._read_blob(record.content_digest)
            if not content.ok or content.value is None:
                return StoreResult.outcome(
                    content.state,
                    content.code or "projection_read_failed",
                )
            try:
                payload = ProjectionPayload(record, content.value)
            except ProjectionContractError:
                return StoreResult.outcome(
                    StoreState.FAILURE,
                    "projection_integrity_failed",
                )
            return StoreResult.success(payload)

    def get_record(self, projection_id: str) -> StoreResult[ProjectionRecord]:
        """Read latest metadata without resolving content bytes."""

        if not _valid_opaque_id(projection_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_projection_id")
        with self._lock:
            try:
                record = self._latest_record(projection_id)
            except (ProjectionContractError, sqlite3.Error):
                return StoreResult.outcome(StoreState.FAILURE, "projection_read_failed")
        if record is None:
            return StoreResult.outcome(StoreState.UNKNOWN, "projection_not_found")
        return StoreResult.success(record)

    def integrity_check(self) -> StoreResult[Mapping[str, int]]:
        """Verify metadata hashes, namespace binding, and referenced blobs."""

        with self._lock:
            try:
                rows = self._connection.execute(
                    "SELECT payload_hash, payload_json FROM projection_versions"
                ).fetchall()
                digests: set[str] = set()
                for row in rows:
                    record = ProjectionRecord.from_json(str(row["payload_json"]))
                    if record.namespace is not self.namespace:
                        raise ProjectionContractError("stored namespace differs")
                    if canonical_digest(record.to_dict()) != str(row["payload_hash"]):
                        raise ProjectionContractError("stored payload hash differs")
                    digest_result = self._read_blob(record.content_digest)
                    if not digest_result.ok:
                        raise ProjectionContractError("stored content digest differs")
                    digests.add(record.content_digest)
            except (ProjectionContractError, sqlite3.Error):
                return StoreResult.outcome(
                    StoreState.FAILURE,
                    "projection_integrity_failed",
                )
        return StoreResult.success(
            {"projection_versions": len(rows), "content_objects": len(digests)}
        )

    def close(self) -> None:
        """Checkpoint and close the namespace metadata database."""

        with self._lock:
            if self._closed:
                return
            try:
                self._connection.execute("PRAGMA wal_checkpoint(FULL)")
            finally:
                self._connection.close()
                self._closed = True

    def _latest_record(self, projection_id: str) -> ProjectionRecord | None:
        row = self._connection.execute(
            "SELECT payload_json FROM projection_versions "
            "WHERE projection_id = ? ORDER BY version DESC LIMIT 1",
            (projection_id,),
        ).fetchone()
        if row is None:
            return None
        return ProjectionRecord.from_json(str(row["payload_json"]))

    def _blob_path(self, digest: str) -> Path:
        _digest(digest, "content_digest")
        value = digest.removeprefix("sha256:")
        path = self._objects / value[:2] / value
        if self._objects.resolve() not in path.parent.resolve().parents:
            raise ProjectionStorageError("projection object path is unsafe")
        return path

    def _put_blob(self, digest: str, content: bytes) -> bool:
        target = self._blob_path(digest)
        existing = self._read_blob(digest)
        if existing.ok:
            if existing.value != content:
                raise ProjectionStorageError("projection object digest collision")
            return False
        if existing.state is not StoreState.UNKNOWN:
            raise ProjectionStorageError("projection object cannot be verified")
        _prepare_private_directory(target.parent)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".projection-", dir=target.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp_name, target)
                created = True
            except FileExistsError:
                created = False
            Path(temp_name).unlink(missing_ok=True)
            _fsync_directory(target.parent)
            verified = self._read_blob(digest)
            if not verified.ok or verified.value != content:
                if created:
                    target.unlink(missing_ok=True)
                raise ProjectionStorageError("projection object verification failed")
            return created
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temp_name).unlink(missing_ok=True)
            raise

    def _read_blob(self, digest: str) -> StoreResult[bytes]:
        try:
            target = self._blob_path(digest)
            if target.is_symlink():
                return StoreResult.outcome(StoreState.FAILURE, "projection_path_unsafe")
            content = target.read_bytes()
        except FileNotFoundError:
            return StoreResult.outcome(
                StoreState.UNKNOWN, "projection_content_not_found"
            )
        except (OSError, ProjectionContractError, ProjectionStorageError):
            return StoreResult.outcome(StoreState.FAILURE, "projection_read_failed")
        if _sha256(content) != digest:
            return StoreResult.outcome(
                StoreState.FAILURE,
                "projection_integrity_failed",
            )
        return StoreResult.success(content)

    def _discard_unreferenced_blob(self, digest: str, created: bool) -> None:
        if not created:
            return
        referenced = self._connection.execute(
            "SELECT 1 FROM projection_versions WHERE content_digest = ? LIMIT 1",
            (digest,),
        ).fetchone()
        if referenced is not None:
            return
        try:
            target = self._blob_path(digest)
            if target.is_file() and not target.is_symlink():
                target.unlink()
                _fsync_directory(target.parent)
        except (OSError, ProjectionContractError, ProjectionStorageError):
            return

    def _close_failed(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()


class _ProjectionAPI:
    """Shared policy-and-audit facade for one fixed namespace."""

    def __init__(
        self,
        store: _ProjectionNamespaceStore,
        policy: ProjectionPolicy,
        audit_log: ProjectionAuditLog,
        cache: ConsentCache[bytes],
    ) -> None:
        self._store = store
        self._policy = policy
        self._audit_log = audit_log
        self._cache = cache
        self.namespace = store.namespace

    def write(
        self,
        projection_id: str,
        content: bytes,
        request: ProjectionPolicyRequest,
        *,
        consent_scope: Any,
        consent_revision: Any,
        occurred_at: str,
    ) -> StoreResult[ProjectionRecord]:
        """Write a first projection version after policy and consent checks."""

        decision = self._authorize(
            request,
            expected_operation=ProjectionOperation.WRITE,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
            occurred_at=occurred_at,
        )
        blocked = self._blocked_result(
            decision,
            projection_id,
            occurred_at,
            operation=ProjectionOperation.WRITE,
        )
        if blocked is not None:
            return blocked
        try:
            record = ProjectionRecord(
                projection_id=projection_id,
                namespace=self.namespace,
                version=1,
                content_digest=_sha256(content),
                byte_size=len(content),
                consent_scope_fingerprint=fingerprint_consent_scope(consent_scope),
                consent_revision_fingerprint=fingerprint_consent_revision(
                    consent_revision
                ),
                created_at=occurred_at,
            )
        except (ProjectionContractError, TypeError, ValueError):
            result: StoreResult[ProjectionRecord] = StoreResult.outcome(
                StoreState.FAILURE,
                "invalid_projection",
            )
        else:
            result = self._store.write(record, content, correcting=False)
            if result.ok:
                self._cache.put(
                    projection_id,
                    content,
                    scope=consent_scope,
                    revision=consent_revision,
                )
        return self._audit_result(
            result,
            decision,
            operation=ProjectionOperation.WRITE,
            projection_id=projection_id,
            occurred_at=occurred_at,
        )

    def read(
        self,
        projection_id: str,
        request: ProjectionPolicyRequest,
        *,
        consent_scope: Any,
        consent_revision: Any,
        occurred_at: str,
    ) -> StoreResult[ProjectionPayload]:
        """Read verified bytes from this namespace after policy checks."""

        decision = self._authorize(
            request,
            expected_operation=ProjectionOperation.READ,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
            occurred_at=occurred_at,
        )
        blocked = self._blocked_result(
            decision,
            projection_id,
            occurred_at,
            operation=ProjectionOperation.READ,
        )
        if blocked is not None:
            return blocked
        result = self._store.read(projection_id)
        if result.ok and result.value is not None:
            record = result.value.record
            if record.consent_scope_fingerprint != fingerprint_consent_scope(
                consent_scope
            ) or record.consent_revision_fingerprint != fingerprint_consent_revision(
                consent_revision
            ):
                result = StoreResult.outcome(
                    StoreState.DENIED,
                    "consent_binding_mismatch",
                )
            else:
                self._cache.put(
                    projection_id,
                    result.value.content,
                    scope=consent_scope,
                    revision=consent_revision,
                )
        return self._audit_result(
            result,
            decision,
            operation=ProjectionOperation.READ,
            projection_id=projection_id,
            occurred_at=occurred_at,
        )

    def export(
        self,
        projection_id: str,
        request: ProjectionPolicyRequest,
        *,
        consent_scope: Any,
        consent_revision: Any,
        occurred_at: str,
    ) -> StoreResult[ProjectionPayload]:
        """Return verified bytes only after explicit export policy approval."""

        decision = self._authorize(
            request,
            expected_operation=ProjectionOperation.EXPORT,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
            occurred_at=occurred_at,
        )
        blocked = self._blocked_result(
            decision,
            projection_id,
            occurred_at,
            operation=ProjectionOperation.EXPORT,
        )
        if blocked is not None:
            return blocked
        result = self._store.read(projection_id)
        if result.ok and result.value is not None:
            record = result.value.record
            if record.consent_scope_fingerprint != fingerprint_consent_scope(
                consent_scope
            ) or record.consent_revision_fingerprint != fingerprint_consent_revision(
                consent_revision
            ):
                result = StoreResult.outcome(
                    StoreState.DENIED,
                    "consent_binding_mismatch",
                )
        return self._audit_result(
            result,
            decision,
            operation=ProjectionOperation.EXPORT,
            projection_id=projection_id,
            occurred_at=occurred_at,
        )

    def correct(
        self,
        projection_id: str,
        content: bytes,
        request: ProjectionPolicyRequest,
        *,
        consent_scope: Any,
        consent_revision: Any,
        occurred_at: str,
    ) -> StoreResult[ProjectionRecord]:
        """Append a corrected projection version; previous bytes remain immutable."""

        decision = self._authorize(
            request,
            expected_operation=ProjectionOperation.CORRECT,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
            occurred_at=occurred_at,
        )
        blocked = self._blocked_result(
            decision,
            projection_id,
            occurred_at,
            operation=ProjectionOperation.CORRECT,
        )
        if blocked is not None:
            return blocked
        current = self._store.get_record(projection_id)
        if not current.ok or current.value is None:
            result: StoreResult[ProjectionRecord] = StoreResult.outcome(
                current.state,
                current.code or "projection_read_failed",
            )
        else:
            try:
                record = ProjectionRecord(
                    projection_id=projection_id,
                    namespace=self.namespace,
                    version=current.value.version + 1,
                    content_digest=_sha256(content),
                    byte_size=len(content),
                    consent_scope_fingerprint=fingerprint_consent_scope(consent_scope),
                    consent_revision_fingerprint=fingerprint_consent_revision(
                        consent_revision
                    ),
                    created_at=occurred_at,
                )
            except (ProjectionContractError, TypeError, ValueError):
                result = StoreResult.outcome(
                    StoreState.FAILURE,
                    "invalid_projection",
                )
            else:
                result = self._store.write(record, content, correcting=True)
                if result.ok:
                    self._cache.put(
                        projection_id,
                        content,
                        scope=consent_scope,
                        revision=consent_revision,
                    )
        return self._audit_result(
            result,
            decision,
            operation=ProjectionOperation.CORRECT,
            projection_id=projection_id,
            occurred_at=occurred_at,
        )

    def review(
        self,
        projection_id: str,
        request: ProjectionPolicyRequest,
        *,
        consent_scope: Any,
        consent_revision: Any,
        occurred_at: str,
    ) -> StoreResult[ProjectionRecord]:
        """Record an authorized review without accepting reviewer identity."""

        decision = self._authorize(
            request,
            expected_operation=ProjectionOperation.REVIEW,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
            occurred_at=occurred_at,
        )
        blocked = self._blocked_result(
            decision,
            projection_id,
            occurred_at,
            operation=ProjectionOperation.REVIEW,
        )
        if blocked is not None:
            return blocked
        result = self._store.get_record(projection_id)
        return self._audit_result(
            result,
            decision,
            operation=ProjectionOperation.REVIEW,
            projection_id=projection_id,
            occurred_at=occurred_at,
        )

    def withdraw_consent(
        self,
        consent_scope: Any,
        consent_revision: Any,
    ) -> ConsentInvalidationEvent:
        """Invalidate cached projections and retain a revocation tombstone."""

        return self._cache.revoke(consent_scope, consent_revision)

    def _authorize(
        self,
        request: ProjectionPolicyRequest,
        *,
        expected_operation: ProjectionOperation,
        consent_scope: Any,
        consent_revision: Any,
        occurred_at: str,
    ) -> ProjectionPolicyDecision:
        _timestamp(occurred_at, "occurred_at")
        try:
            revoked = self._cache.is_revoked(consent_scope, consent_revision)
        except (TypeError, ValueError):
            return _synthetic_decision(
                request,
                ProjectionPolicyOutcome.DENY,
                "consent_context_invalid",
                occurred_at,
            )
        if request.namespace is not self.namespace:
            return _synthetic_decision(
                request,
                ProjectionPolicyOutcome.DENY,
                "namespace_mismatch",
                occurred_at,
            )
        if request.operation is not expected_operation:
            return _synthetic_decision(
                request,
                ProjectionPolicyOutcome.DENY,
                "operation_mismatch",
                occurred_at,
            )
        if revoked:
            return _synthetic_decision(
                request,
                ProjectionPolicyOutcome.DENY,
                "consent_withdrawn",
                occurred_at,
            )
        return self._policy.evaluate(request, decided_at=occurred_at)

    def _blocked_result(
        self,
        decision: ProjectionPolicyDecision,
        projection_id: str,
        occurred_at: str,
        *,
        operation: ProjectionOperation,
    ) -> StoreResult[Any] | None:
        if decision.allowed:
            return None
        state = (
            StoreState.PARTIAL
            if decision.outcome is ProjectionPolicyOutcome.REVIEW
            else StoreState.DENIED
        )
        result: StoreResult[Any] = StoreResult.outcome(state, decision.reason_code)
        return self._audit_result(
            result,
            decision,
            operation=operation,
            projection_id=projection_id,
            occurred_at=occurred_at,
        )

    def _audit_result(
        self,
        result: StoreResult[T],
        decision: ProjectionPolicyDecision,
        *,
        operation: ProjectionOperation,
        projection_id: str,
        occurred_at: str,
    ) -> StoreResult[T]:
        if result.ok:
            outcome = "allow"
            reason = "operation_allowed"
        elif result.state is StoreState.PARTIAL:
            outcome = "review"
            reason = result.code or "review_required"
        elif result.state is StoreState.DENIED:
            outcome = "deny"
            reason = result.code or "policy_denied"
        else:
            outcome = "failure"
            reason = result.code or "operation_failed"
        try:
            resource_digest = canonical_digest({"projection_id": projection_id})
            event = ProjectionAuditEvent(
                event_id=f"event_{os.urandom(16).hex()}",
                occurred_at=occurred_at,
                operation=operation,
                namespace=self.namespace,
                outcome=outcome,
                reason_code=reason,
                request_digest=decision.request_digest,
                resource_digest=resource_digest,
            )
        except (OSError, ProjectionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "audit_event_invalid")
        audited = self._audit_log.append(event)
        if not audited.ok:
            return StoreResult.outcome(StoreState.FAILURE, "audit_write_failed")
        return result


class DeidentifiedProjectionAPI(_ProjectionAPI):
    """Projection API with no vault or identified-namespace capability."""


class IdentifiedProjectionAPI(_ProjectionAPI):
    """Identified projection API with an explicit protected-vault capability."""

    def __init__(
        self,
        store: _ProjectionNamespaceStore,
        policy: ProjectionPolicy,
        audit_log: ProjectionAuditLog,
        cache: ConsentCache[bytes],
        vault: TransformVault,
    ) -> None:
        super().__init__(store, policy, audit_log, cache)
        if not isinstance(vault, TransformVault):
            raise TypeError("vault does not satisfy the transform-vault protocol")
        self._vault = vault

    def record_transform(
        self,
        record: TransformRecord,
        material: bytes,
        request: ProjectionPolicyRequest,
        *,
        consent_scope: Any,
        consent_revision: Any,
        occurred_at: str,
    ) -> StoreResult[TransformRecord]:
        """Store protected transform material after identified write policy."""

        decision = self._authorize(
            request,
            expected_operation=ProjectionOperation.WRITE,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
            occurred_at=occurred_at,
        )
        blocked = self._blocked_result(
            decision,
            record.transform_id,
            occurred_at,
            operation=ProjectionOperation.WRITE,
        )
        if blocked is not None:
            return blocked
        result = self._vault.record_transform(record, material)
        return self._audit_result(
            result,
            decision,
            operation=ProjectionOperation.WRITE,
            projection_id=record.transform_id,
            occurred_at=occurred_at,
        )

    def resolve_transform(
        self,
        transform_id: str,
        request: ProjectionPolicyRequest,
        *,
        consent_scope: Any,
        consent_revision: Any,
        occurred_at: str,
    ) -> StoreResult[bytes]:
        """Resolve protected material after identified read policy."""

        decision = self._authorize(
            request,
            expected_operation=ProjectionOperation.READ,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
            occurred_at=occurred_at,
        )
        blocked = self._blocked_result(
            decision,
            transform_id,
            occurred_at,
            operation=ProjectionOperation.READ,
        )
        if blocked is not None:
            return blocked
        result = self._vault.resolve_transform(transform_id)
        return self._audit_result(
            result,
            decision,
            operation=ProjectionOperation.READ,
            projection_id=transform_id,
            occurred_at=occurred_at,
        )


class ProjectionBoundary:
    """Owner for physically separated namespaces and their shared safe audit log."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: ProjectionPolicy | None = None,
    ) -> None:
        self._root = Path(root)
        _prepare_private_directory(self._root)
        self._policy = policy or ProjectionPolicy()
        audit_log: ProjectionAuditLog | None = None
        identified_store: _ProjectionNamespaceStore | None = None
        try:
            audit_log = ProjectionAuditLog(self._root / "audit" / "events.sqlite3")
            identified_store = _ProjectionNamespaceStore(
                self._root / ProjectionNamespace.IDENTIFIED.value,
                ProjectionNamespace.IDENTIFIED,
            )
            deidentified_store = _ProjectionNamespaceStore(
                self._root / ProjectionNamespace.DEIDENTIFIED.value,
                ProjectionNamespace.DEIDENTIFIED,
            )
        except BaseException:
            if identified_store is not None:
                identified_store.close()
            if audit_log is not None:
                audit_log.close()
            raise
        self._audit_log = audit_log
        self._identified_store = identified_store
        self._deidentified_store = deidentified_store
        self._identified_cache: ConsentCache[bytes] = ConsentCache()
        self._deidentified_cache: ConsentCache[bytes] = ConsentCache()
        self._closed = False

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        policy: ProjectionPolicy | None = None,
    ) -> StoreResult["ProjectionBoundary"]:
        """Open with typed unsupported and failure outcomes."""

        try:
            return StoreResult.success(cls(root, policy=policy))
        except ProjectionCompatibilityError:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "schema_unsupported")
        except ProjectionStorageError:
            return StoreResult.outcome(StoreState.FAILURE, "projection_open_failed")

    def deidentified(self) -> DeidentifiedProjectionAPI:
        """Return the API that cannot resolve identified paths or vault material."""

        return DeidentifiedProjectionAPI(
            self._deidentified_store,
            self._policy,
            self._audit_log,
            self._deidentified_cache,
        )

    def identified(self, vault: TransformVault) -> IdentifiedProjectionAPI:
        """Return the identified API only when an explicit vault is supplied."""

        return IdentifiedProjectionAPI(
            self._identified_store,
            self._policy,
            self._audit_log,
            self._identified_cache,
            vault,
        )

    def audit_events(self) -> StoreResult[tuple[ProjectionAuditEvent, ...]]:
        """Return value-free audit events for authorized owners."""

        return self._audit_log.events()

    def integrity_check(self) -> StoreResult[Mapping[str, int]]:
        """Verify both namespaces and the independent audit log."""

        identified = self._identified_store.integrity_check()
        deidentified = self._deidentified_store.integrity_check()
        audit = self._audit_log.integrity_check()
        if not identified.ok:
            return StoreResult.outcome(
                identified.state,
                identified.code or "projection_integrity_failed",
            )
        if not deidentified.ok:
            return StoreResult.outcome(
                deidentified.state,
                deidentified.code or "projection_integrity_failed",
            )
        if not audit.ok:
            return StoreResult.outcome(
                audit.state,
                audit.code or "audit_integrity_failed",
            )
        return StoreResult.success(
            {
                "audit_events": int(audit.value or 0),
                "deidentified_versions": int(
                    (deidentified.value or {}).get("projection_versions", 0)
                ),
                "identified_versions": int(
                    (identified.value or {}).get("projection_versions", 0)
                ),
            }
        )

    def close(self) -> None:
        """Close both stores and the audit log."""

        if self._closed:
            return
        self._identified_store.close()
        self._deidentified_store.close()
        self._audit_log.close()
        self._closed = True

    def __enter__(self) -> "ProjectionBoundary":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _synthetic_decision(
    request: ProjectionPolicyRequest,
    outcome: ProjectionPolicyOutcome,
    reason_code: str,
    decided_at: str,
) -> ProjectionPolicyDecision:
    digest = request.request_digest
    identity = canonical_digest(
        {
            "decided_at": decided_at,
            "outcome": outcome.value,
            "reason_code": reason_code,
            "request_digest": digest,
        }
    )
    return ProjectionPolicyDecision(
        decision_id=f"decision_{identity.removeprefix('sha256:')[:32]}",
        request_digest=digest,
        outcome=outcome,
        reason_code=reason_code,
        decided_at=decided_at,
    )


def _apply_single_migration(
    connection: sqlite3.Connection,
    *,
    name: str,
    checksum: str,
    statements: tuple[str, ...],
) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL)"
    )
    rows = connection.execute(
        "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    if rows:
        if len(rows) != 1 or int(rows[0]["version"]) != 1:
            raise ProjectionCompatibilityError("projection schema is unsupported")
        if str(rows[0]["name"]) != name or str(rows[0]["checksum"]) != checksum:
            raise ProjectionCompatibilityError("projection migration checksum differs")
        return
    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in statements:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
            (1, name, checksum),
        )
        connection.execute("COMMIT")
    except sqlite3.Error as exc:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise ProjectionStorageError("projection migration failed") from exc


def _prepare_private_parent(path: Path) -> None:
    _prepare_private_directory(path.parent)
    if path.exists() and path.is_symlink():
        raise ProjectionStorageError("projection metadata path is unsafe")


def _prepare_private_directory(path: Path) -> None:
    try:
        if path.exists() and path.is_symlink():
            raise ProjectionStorageError("projection directory is unsafe")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise ProjectionStorageError(
            "projection directory cannot be initialized"
        ) from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(content: bytes) -> str:
    if not isinstance(content, bytes):
        raise ProjectionContractError("projection content must be bytes")
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _strict_mapping(
    payload: Mapping[str, Any],
    fields: frozenset[str],
    record_name: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProjectionContractError(f"{record_name} must be an object")
    data = dict(payload)
    if set(data) != fields:
        raise ProjectionContractError(f"{record_name} has missing or unknown fields")
    return data


def _strict_json(payload: str, record_name: str) -> Mapping[str, Any]:
    if not isinstance(payload, str):
        raise ProjectionContractError(f"{record_name} JSON must be text")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ProjectionContractError(f"{record_name} JSON is invalid") from None
    if not isinstance(value, Mapping):
        raise ProjectionContractError(f"{record_name} JSON must contain an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _schema_version(value: Any) -> str:
    if value != PROJECTION_SCHEMA_VERSION:
        raise ProjectionContractError("projection schema version is unsupported")
    return str(value)


def _valid_opaque_id(value: Any) -> bool:
    return isinstance(value, str) and _OPAQUE_ID_RE.fullmatch(value) is not None


def _opaque_id(value: Any, field_name: str) -> str:
    if not _valid_opaque_id(value):
        raise ProjectionContractError(f"{field_name} must be an opaque identifier")
    return str(value)


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ProjectionContractError(f"{field_name} must be a SHA-256 digest")
    return value


def _controlled(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise ProjectionContractError(f"{field_name} must be a controlled identifier")
    return value


def _controlled_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ProjectionContractError(f"{field_name} must be a tuple")
    result = tuple(sorted({_controlled(item, field_name) for item in values}))
    if len(result) != len(values):
        raise ProjectionContractError(f"{field_name} must be unique")
    return result


def _positive(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ProjectionContractError(f"{field_name} must be positive")
    return value


def _non_negative(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProjectionContractError(f"{field_name} must be non-negative")
    return value


def _timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise ProjectionContractError(
            f"{field_name} must be a timezone-aware ISO timestamp"
        )
    return value


def _namespace(value: ProjectionNamespace | str) -> ProjectionNamespace:
    try:
        return (
            value
            if isinstance(value, ProjectionNamespace)
            else ProjectionNamespace(value)
        )
    except (TypeError, ValueError):
        raise ProjectionContractError("projection namespace is unsupported") from None


def _operation(value: ProjectionOperation | str) -> ProjectionOperation:
    try:
        return (
            value
            if isinstance(value, ProjectionOperation)
            else ProjectionOperation(value)
        )
    except (TypeError, ValueError):
        raise ProjectionContractError("projection operation is unsupported") from None


__all__ = [
    "PROJECTION_AUDIT_SCHEMA_VERSION",
    "PROJECTION_SCHEMA_VERSION",
    "PROJECTION_SCHEMA_NAMES",
    "PROJECTION_SCHEMA_PACKAGE",
    "PROJECTION_STORAGE_SCHEMA_VERSION",
    "DeidentifiedProjectionAPI",
    "IdentifiedProjectionAPI",
    "InMemoryTransformVault",
    "ProjectionAuditEvent",
    "ProjectionAuditLog",
    "ProjectionBoundary",
    "ProjectionCompatibilityError",
    "ProjectionContractError",
    "ProjectionNamespace",
    "ProjectionOperation",
    "ProjectionPayload",
    "ProjectionPolicy",
    "ProjectionPolicyDecision",
    "ProjectionPolicyOutcome",
    "ProjectionPolicyRequest",
    "ProjectionRecord",
    "ProjectionStorageError",
    "TransformRecord",
    "TransformVault",
    "load_all_projection_schemas",
    "load_projection_schema",
]
