"""Encrypted, integrity-checked recovery points for durable service state.

A *recovery point* is an immutable directory holding one application-consistent
copy of the durable local state an OpenMed service needs to resume safely:

* the local Journey store: the SQLite metadata database (captured with the
  SQLite online backup API, never by copying live files), the
  content-addressed artifact blobs, and the artifact store's encryption
  metadata (key identifiers and verifiers, never key material);
* the async job metadata store;
* the model-registry slot, pointer, and lineage state;
* tamper-evident audit-chain files.

Everything else is classified instead of copied (see :data:`EXCLUDED_STATE`):
data keys and service credentials stay with caller-owned providers, model and
result caches are rebuildable, and SQLite journal files, staging files, and
in-memory job payloads are transient.

Layout of a recovery point::

    recovery-point.json     cleartext, value-free descriptor (format, id, kind)
    manifest.omae           AEAD envelope holding the canonical manifest
    objects/NNNNNN.omrc     chunked AEAD streams, one per stored object

Every chunk is sealed by :class:`~openmed.core.aead_envelope.EnvelopeCipher`
under the caller's ``service-backup`` key and bound to the recovery-point id,
the object name, the chunk index, and the chunk count, so corruption,
truncation, reordering, and objects moved between recovery points all fail
authentication.  The manifest is sealed the same way.  It lists, per object, a
canonical relative path, component category, size, and SHA-256 digest, plus
component schema versions, snapshot sequence bounds, consistency mode, and
retention metadata.  Paths come from a fixed layout; operator paths and file
names never enter a manifest, receipt, or error.  Keys never enter a backup:
they are resolved through the caller's
:class:`~openmed.core.key_provider.KeyProvider` at every operation.

Incremental recovery points store only objects whose digest changed since a
parent manifest and reference the rest by recovery-point id.  Restore
validates the whole chain (parent manifest digests, contiguous sequence
numbers, retention, and non-regressing sequence bounds) before writing.

Restore writes into a new, inactive target directory, verifies every digest,
the authenticated manifest, schema compatibility, cross-component references,
and the snapshot boundary, and only then seals a staging record.
:func:`activate_restored_state` re-verifies that record against the staged
files, runs caller-supplied checks (for example the production security
self-check) and recovery-epoch hooks, and then atomically repoints the live
state symlink.  Restore itself never writes to the live state.

Every outcome is a value-free :class:`RecoveryReceipt` with a controlled
reason code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import struct
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openmed.__about__ import __version__
from openmed.core.aead_envelope import (
    EnvelopeAuthenticationError,
    EnvelopeCipher,
    EnvelopeFormatError,
    EnvelopeVersionError,
)
from openmed.core.audit_chain import CHAIN_VERSION, AuditChain
from openmed.core.key_provider import DataKey, EncryptionError, KeyProvider
from openmed.core.registry_slots import REGISTRY_STATE_SCHEMA_VERSION
from openmed.structured.store.migrations import LATEST_MIGRATION_VERSION, MIGRATIONS

__all__ = [
    "AUDIT_CHAIN_DIRECTORY",
    "BACKUP_KEY_PURPOSE",
    "DEFAULT_CHUNK_BYTES",
    "EXCLUDED_STATE",
    "INVENTORY_SCHEMA",
    "JOB_STORE_PATH",
    "JOURNEY_DATABASE_PATH",
    "JOURNEY_DIRECTORY",
    "MANIFEST_SCHEMA",
    "MAX_CHAIN_LENGTH",
    "RECEIPT_SCHEMA",
    "RECOVERY_FORMAT",
    "RECOVERY_FORMAT_VERSION",
    "RECOVERY_METADATA_DIRECTORY",
    "RECOVERY_REASON_CODES",
    "REGISTRY_STATE_FILE",
    "ActivationCheck",
    "ActivationCheckResult",
    "ComponentCategory",
    "ComponentSummary",
    "InventoryEntry",
    "QuiesceHook",
    "RecoveryEpochContext",
    "RecoveryEpochHook",
    "RecoveryError",
    "RecoveryKind",
    "RecoveryOutcome",
    "RecoveryReceipt",
    "ServiceStateInventory",
    "ServiceStateSources",
    "StateDisposition",
    "activate_restored_state",
    "create_recovery_point",
    "discard_restored_state",
    "inventory_service_state",
    "journey_store_check",
    "restore_recovery_point",
    "verify_recovery_point",
]

RECOVERY_FORMAT = "openmed.service.recovery-point"
RECOVERY_FORMAT_VERSION = 1
MANIFEST_SCHEMA = "openmed.service.recovery-manifest.v1"
RECEIPT_SCHEMA = "openmed.service.recovery-receipt.v1"
INVENTORY_SCHEMA = "openmed.service.recovery-inventory.v1"
STAGING_FORMAT = "openmed.service.recovery-staging"
STAGING_SCHEMA = "openmed.service.recovery-staging.v1"
BACKUP_KEY_PURPOSE = "service-backup"
JOB_STORE_LAYOUT_VERSION = 1
JOURNEY_MARKER_SCHEMA_VERSION = 1
DEFAULT_CHUNK_BYTES = 1024 * 1024
MAX_CHUNK_BYTES = 8 * 1024 * 1024
MAX_CHAIN_LENGTH = 64
MAX_AUDIT_CHAINS = 10000

# Canonical layout of a service state root, relative and POSIX-separated.  A
# restored target always uses this layout; it mirrors the local Journey store
# (``artifacts/`` beside ``journey.sqlite3``) and the default job-store name.
JOURNEY_DIRECTORY = "journey"
JOURNEY_DATABASE_PATH = "journey/journey.sqlite3"
JOURNEY_MARKER_PATH = "journey/artifacts/encryption.json"
JOURNEY_BLOB_ROOT = "journey/artifacts/blobs/sha256"
JOB_STORE_PATH = "jobs/service-jobs.json"
REGISTRY_STATE_FILE = "registry/registry_state.json"
AUDIT_CHAIN_DIRECTORY = "audit"
RECOVERY_METADATA_DIRECTORY = ".openmed-recovery"

_JOURNEY_DATABASE_NAME = "journey.sqlite3"
_JOURNEY_ARTIFACTS_NAME = "artifacts"
_JOURNEY_MARKER_NAME = "encryption.json"
_DESCRIPTOR_NAME = "recovery-point.json"
_MANIFEST_NAME = "manifest.omae"
_OBJECTS_DIRECTORY = "objects"
_OBJECT_SUFFIX = ".omrc"
_STAGING_DESCRIPTOR = "staging.json"
_STAGING_SEAL = "staging.omae"
_ACTIVATION_RECORD = "activation.json"
_PARTIAL_PREFIX = ".recovery-partial-"

_STREAM_MAGIC = b"OMRC"
_STREAM_VERSION = 1
_STREAM_HEADER = struct.Struct(">4sBI")
_CHUNK_LENGTH = struct.Struct(">I")
_ENVELOPE_OVERHEAD = 512
_MAX_DESCRIPTOR_BYTES = 64 * 1024
_MAX_SEALED_DOCUMENT_BYTES = 64 * 1024 * 1024
_MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
_MAX_PRODUCER_VERSION = 64

_POINT_ID_RE = re.compile(r"rp_[0-9a-f]{32}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OBJECT_NAME_RE = re.compile(r"[0-9]{6}\Z")
_BLOB_PREFIX_RE = re.compile(r"[0-9a-f]{2}\Z")
_BLOB_NAME_RE = re.compile(r"[0-9a-f]{64}\Z")
_AUDIT_NAME_RE = re.compile(r"chain-[0-9]{4}\.json\Z")
_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_MARKER_FIELDS = frozenset(
    {"envelope_version", "key_purpose", "key_verifiers", "schema_version", "state"}
)
_KNOWN_MIGRATIONS = {
    migration.version: (migration.name, migration.checksum) for migration in MIGRATIONS
}

RECOVERY_REASON_CODES = frozenset(
    {
        "activation_check_failed",
        "activation_failed",
        "already_active",
        "atomic_swap_unsupported",
        "chain_incomplete",
        "chain_invalid",
        "chain_sequence_regressed",
        "chain_too_long",
        "cleanup_failed",
        "component_changed_during_capture",
        "component_invalid",
        "cross_reference_missing_artifact",
        "descriptor_invalid",
        "descriptor_mismatch",
        "destination_exists",
        "destination_unavailable",
        "backup_write_failed",
        "encryption_dependency_unavailable",
        "encryption_failed",
        "encryption_metadata_missing",
        "encryption_state_mismatch",
        "format_unsupported",
        "key_material_invalid",
        "key_provider_failed",
        "key_retired",
        "key_revoked",
        "key_unavailable",
        "live_target_unavailable",
        "live_target_unmanaged",
        "manifest_authentication_failed",
        "manifest_invalid",
        "not_a_restored_target",
        "object_authentication_failed",
        "object_corrupt",
        "object_digest_mismatch",
        "object_missing",
        "object_size_mismatch",
        "point_unavailable",
        "quiesce_failed",
        "quiesce_required",
        "recovery_epoch_refused",
        "recovery_failed",
        "recovery_point_expired",
        "recovery_point_stale",
        "restore_incomplete",
        "restore_write_failed",
        "schema_incompatible",
        "snapshot_boundary_mismatch",
        "snapshot_boundary_moved",
        "source_invalid",
        "source_unavailable",
        "staged_is_live",
        "staged_state_modified",
        "staged_state_unavailable",
        "staging_authentication_failed",
        "target_exists",
        "target_overlaps_live",
        "target_unavailable",
    }
)


class ComponentCategory(str, Enum):
    """Authoritative durable-state categories captured in a recovery point."""

    JOURNEY_METADATA = "journey_metadata"
    JOURNEY_ARTIFACTS = "journey_artifacts"
    JOURNEY_ENCRYPTION = "journey_encryption"
    JOB_METADATA = "job_metadata"
    MODEL_REGISTRY = "model_registry"
    AUDIT_CHAIN = "audit_chain"


_CATEGORY_ORDER = tuple(ComponentCategory)


class StateDisposition(str, Enum):
    """How a class of service state is treated by backup and restore.

    ``AUTHORITATIVE`` state is captured.  ``REBUILDABLE`` state is recreated
    after restore from verified sources.  ``EXCLUDED_SECRET`` state never
    enters a backup and is resolved through caller-owned providers.
    ``TRANSIENT`` state is never durable and is discarded.
    """

    AUTHORITATIVE = "authoritative"
    REBUILDABLE = "rebuildable"
    EXCLUDED_SECRET = "excluded_secret"
    TRANSIENT = "transient"


EXCLUDED_STATE: Mapping[str, StateDisposition] = {
    "encryption_keys": StateDisposition.EXCLUDED_SECRET,
    "service_credentials": StateDisposition.EXCLUDED_SECRET,
    "model_artifacts": StateDisposition.REBUILDABLE,
    "result_cache": StateDisposition.REBUILDABLE,
    "sqlite_journal_files": StateDisposition.TRANSIENT,
    "staging_files": StateDisposition.TRANSIENT,
    "job_payloads": StateDisposition.TRANSIENT,
}
"""State that is classified but never copied into a recovery point.

Data keys (Journey and backup) and service credentials stay with caller-owned
providers.  Model artifacts and result caches are re-acquired or rebuilt.
SQLite ``-wal``/``-shm`` files are folded in by the online backup API, blob
staging files are discarded, and raw job payloads only ever live in memory.
"""


class RecoveryOutcome(str, Enum):
    """Typed outcome of one recovery operation."""

    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"


class RecoveryKind(str, Enum):
    """Whether a recovery point is self-contained or extends a parent."""

    FULL = "full"
    INCREMENTAL = "incremental"


class RecoveryError(RuntimeError):
    """Value-free recovery failure carrying one controlled reason code.

    Attributes:
        code: Controlled reason code; never contains paths, identifiers,
            key material, or state content.
    """

    def __init__(self, code: str) -> None:
        selected = code if code in RECOVERY_REASON_CODES else "recovery_failed"
        self.code = selected
        super().__init__(f"recovery operation did not complete ({selected})")


class _Stop(Exception):
    """Internal fail-closed signal mapped to a receipt outcome."""

    def __init__(
        self,
        code: str,
        outcome: RecoveryOutcome = RecoveryOutcome.REFUSED,
        details: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code if code in RECOVERY_REASON_CODES else "recovery_failed"
        self.outcome = outcome
        self.details = dict(details or {})


QuiesceHook = Callable[[], AbstractContextManager[Any]]
"""Factory for a context manager that holds writers quiet during capture.

A drain coordinator can supply one: entering it must stop admission and wait
for in-flight writers; exiting it resumes them.  Without a hook, callers must
assert that writers are stopped.
"""


@dataclass(frozen=True)
class ActivationCheckResult:
    """Value-free result of one pre-activation check.

    Attributes:
        name: Controlled check name, for example ``"journey_store"``.
        passed: Whether the restored state may be activated.
        code: Controlled failure code when the check did not pass.
    """

    name: str
    passed: bool
    code: str | None = None


ActivationCheck = Callable[[Path], ActivationCheckResult | bool]
"""Callable receiving the staged state root and returning a check result."""


@dataclass(frozen=True)
class RecoveryEpochContext:
    """Value-free description of a restore about to be activated.

    Recovery-epoch hooks receive this before the live state is swapped so a
    component can open a new epoch, for example to burn approvals and fence
    authority issued after the recovery point.

    Attributes:
        recovery_point_id: Opaque identifier of the restored recovery point.
        manifest_digest: Digest of the authenticated manifest.
        content_digest: Deterministic digest of the restored logical state.
        kind: ``"full"`` or ``"incremental"``.
        chain_length: Number of recovery points in the restored chain.
        boundary: Snapshot sequence bounds recorded in the manifest.
        activated_at: UTC activation timestamp.
    """

    recovery_point_id: str
    manifest_digest: str
    content_digest: str
    kind: str
    chain_length: int
    boundary: Mapping[str, Any]
    activated_at: str


@runtime_checkable
class RecoveryEpochHook(Protocol):
    """Component that must observe an activation before traffic resumes."""

    def open_recovery_epoch(self, context: RecoveryEpochContext) -> None:
        """Open a new recovery epoch or raise to refuse activation.

        Args:
            context: Value-free description of the restore being activated.
        """


@dataclass(frozen=True, repr=False)
class ServiceStateSources:
    """Locations of the durable state one service instance owns.

    Attributes:
        journey_root: Local Journey store directory holding ``artifacts/`` and
            ``journey.sqlite3``.
        job_store: Async job metadata JSON file.
        registry_state: Model-registry slot and pointer state JSON file.
        audit_chains: Audit-chain JSON files, in a stable order.
    """

    journey_root: Path | None = None
    job_store: Path | None = None
    registry_state: Path | None = None
    audit_chains: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        for name in ("journey_root", "job_store", "registry_state"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        chains = tuple(Path(item) for item in self.audit_chains)
        if len(chains) > MAX_AUDIT_CHAINS:
            raise ValueError("too many audit-chain files")
        object.__setattr__(self, "audit_chains", chains)

    @classmethod
    def from_state_root(cls, root: str | Path) -> "ServiceStateSources":
        """Return the sources of a state root that uses the canonical layout.

        Args:
            root: Directory holding ``journey/``, ``jobs/``, ``registry/``, and
                ``audit/`` (any may be absent).

        Returns:
            Sources for every component present under ``root``.
        """

        base = Path(root)
        journey = base / JOURNEY_DIRECTORY
        jobs = base / JOB_STORE_PATH
        registry = base / REGISTRY_STATE_FILE
        audit = base / AUDIT_CHAIN_DIRECTORY
        chains: tuple[Path, ...] = ()
        if audit.is_dir():
            chains = tuple(
                sorted(
                    item
                    for item in audit.iterdir()
                    if _AUDIT_NAME_RE.fullmatch(item.name) is not None
                )
            )
        return cls(
            journey_root=journey if journey.is_dir() else None,
            job_store=jobs if os.path.lexists(jobs) else None,
            registry_state=registry if os.path.lexists(registry) else None,
            audit_chains=chains,
        )

    def __repr__(self) -> str:
        present = [
            name
            for name in ("journey_root", "job_store", "registry_state")
            if getattr(self, name) is not None
        ]
        return (
            f"{type(self).__name__}(components={present!r}, "
            f"audit_chains={len(self.audit_chains)})"
        )


@dataclass(frozen=True)
class InventoryEntry:
    """Category-only description of one class of service state.

    Attributes:
        category: Component or excluded-state category.
        disposition: How backup and restore treat the category.
        objects: Number of durable objects found (authoritative only).
        bytes: Total object size in bytes (authoritative only).
        schema_version: Component schema version, when one applies.
    """

    category: str
    disposition: StateDisposition
    objects: int = 0
    bytes: int = 0
    schema_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "bytes": self.bytes,
            "category": self.category,
            "disposition": self.disposition.value,
            "objects": self.objects,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class ServiceStateInventory:
    """Versioned, value-free inventory that doubles as the backup plan.

    Attributes:
        entries: Authoritative components followed by excluded categories.
        schema: Inventory document schema identifier.
    """

    entries: tuple[InventoryEntry, ...]
    schema: str = INVENTORY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "schema": self.schema,
        }


@dataclass(frozen=True)
class ComponentSummary:
    """Value-free summary of one captured component.

    Attributes:
        category: Component category.
        objects: Number of objects in the component.
        bytes: Total plaintext size of the component.
        digest: Aggregate digest over the component's paths, digests, and
            sizes.
        schema_version: Component schema version, when one applies.
    """

    category: str
    objects: int
    bytes: int
    digest: str
    schema_version: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "bytes": self.bytes,
            "category": self.category,
            "digest": self.digest,
            "objects": self.objects,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class RecoveryReceipt:
    """Versioned, value-free result of one backup, restore, or activation.

    Receipts carry only opaque recovery-point identifiers, component
    categories, digests, versions, counts, timestamps, and controlled reason
    codes.  They never include paths, identifiers from state, or key data.

    Attributes:
        operation: ``backup``, ``verify``, ``restore``, ``activate``, or
            ``cleanup``.
        outcome: Typed outcome.
        reason: Controlled reason code for a refused or failed operation.
        recovery_point_id: Opaque recovery-point identifier, when known.
        kind: ``full`` or ``incremental``, when known.
        sequence: Position of the recovery point in its chain.
        chain_length: Number of recovery points that were validated.
        content_digest: Deterministic digest of the captured logical state.
        manifest_digest: Digest of the authenticated manifest.
        components: Per-component summaries.
        boundary: Snapshot sequence bounds.
        counts: Aggregate counters.
        details: Controlled extra codes, for example a failing check name.
        started_at: UTC start timestamp.
        finished_at: UTC finish timestamp.
        schema: Receipt schema identifier.
    """

    operation: str
    outcome: RecoveryOutcome
    reason: str | None = None
    recovery_point_id: str | None = None
    kind: str | None = None
    sequence: int | None = None
    chain_length: int = 0
    content_digest: str | None = None
    manifest_digest: str | None = None
    components: tuple[ComponentSummary, ...] = ()
    boundary: Mapping[str, Any] | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, str] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    schema: str = RECEIPT_SCHEMA

    @property
    def completed(self) -> bool:
        """Return whether the operation completed."""

        return self.outcome is RecoveryOutcome.COMPLETED

    def require_completed(self) -> "RecoveryReceipt":
        """Return the receipt, or raise when the operation did not complete.

        Raises:
            RecoveryError: Carrying the receipt's reason code.
        """

        if not self.completed:
            raise RecoveryError(self.reason or "recovery_failed")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "boundary": None if self.boundary is None else dict(self.boundary),
            "chain_length": self.chain_length,
            "components": [item.to_dict() for item in self.components],
            "content_digest": self.content_digest,
            "counts": dict(sorted(self.counts.items())),
            "details": dict(sorted(self.details.items())),
            "finished_at": self.finished_at,
            "kind": self.kind,
            "manifest_digest": self.manifest_digest,
            "operation": self.operation,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "recovery_point_id": self.recovery_point_id,
            "schema": self.schema,
            "sequence": self.sequence,
            "started_at": self.started_at,
        }

    def to_json(self) -> str:
        """Return the receipt as deterministic JSON text."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------


def inventory_service_state(sources: ServiceStateSources) -> ServiceStateInventory:
    """Inventory authoritative state and classify excluded state.

    The inventory reads only metadata (sizes and schema versions).  Output is
    category-only: no paths, names, identifiers, or content.

    Args:
        sources: Locations of the service's durable state.

    Returns:
        A versioned inventory usable as the backup plan.

    Raises:
        RecoveryError: ``source_unavailable`` or ``source_invalid`` when a
            declared source cannot be read safely.
    """

    _require_sources(sources)
    try:
        database, files = _enumerate(sources)
        totals: dict[ComponentCategory, list[int]] = {}
        versions: dict[ComponentCategory, int | None] = {}
        if database is not None:
            facts = _live_journey_facts(database)
            totals[ComponentCategory.JOURNEY_METADATA] = [1, _size(database)]
            versions[ComponentCategory.JOURNEY_METADATA] = facts.schema_version
        for item in files:
            count = totals.setdefault(item.category, [0, 0])
            count[0] += 1
            count[1] += _size(item.file)
            if item.category in versions:
                continue
            versions[item.category] = _inventory_version(item)
    except _Stop as stop:
        raise RecoveryError(stop.code) from None
    except (OSError, sqlite3.Error):
        raise RecoveryError("source_unavailable") from None
    entries = [
        InventoryEntry(
            category=category.value,
            disposition=StateDisposition.AUTHORITATIVE,
            objects=totals[category][0],
            bytes=totals[category][1],
            schema_version=versions.get(category),
        )
        for category in _CATEGORY_ORDER
        if category in totals
    ]
    entries.extend(
        InventoryEntry(category=name, disposition=disposition)
        for name, disposition in EXCLUDED_STATE.items()
    )
    return ServiceStateInventory(entries=tuple(entries))


def create_recovery_point(
    sources: ServiceStateSources,
    destination: str | Path,
    *,
    key_provider: KeyProvider,
    parent: str | Path | None = None,
    quiesce: QuiesceHook | None = None,
    writers_stopped: bool = False,
    retention_days: int | None = None,
    clock: Callable[[], datetime] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> RecoveryReceipt:
    """Capture an encrypted, verified recovery point in a new directory.

    Capture happens inside ``quiesce`` when supplied; otherwise the caller
    must assert ``writers_stopped=True``.  The Journey database is copied with
    the SQLite online backup API into memory, so no plaintext temporary file
    is written.  A pre- and post-capture fingerprint of every source detects
    writers that were not actually stopped.  The point is written to a hidden
    sibling directory, verified by decrypting every stored object, and only
    then renamed into place, so a failed backup never leaves a valid-looking
    point behind.

    Args:
        sources: Locations of the service's durable state.
        destination: New directory for the recovery point; must not exist.
        key_provider: Caller-owned provider with an active
            ``service-backup`` key.
        parent: Existing recovery point to extend incrementally.
        quiesce: Optional context-manager factory that holds writers quiet.
        writers_stopped: Operator assertion that writers are stopped; required
            when no ``quiesce`` hook is supplied.
        retention_days: Optional retention after which restore refuses the
            point.
        clock: Optional UTC clock for deterministic timestamps.
        chunk_bytes: Plaintext bytes per sealed chunk.

    Returns:
        A value-free receipt; check :attr:`RecoveryReceipt.completed`.
    """

    _require_sources(sources)
    _require_chunk_bytes(chunk_bytes)
    if retention_days is not None and (
        type(retention_days) is not int or retention_days < 1
    ):
        raise ValueError("retention_days must be a positive integer")
    if not isinstance(writers_stopped, bool):
        raise TypeError("writers_stopped must be a boolean")
    now = _clock(clock)
    started = now()
    cipher = EnvelopeCipher(key_provider, purpose=BACKUP_KEY_PURPOSE)
    target = Path(destination)
    partial: Path | None = None
    try:
        if quiesce is None and not writers_stopped:
            raise _Stop("quiesce_required")
        key = _active_key(cipher)
        if os.path.lexists(target):
            raise _Stop("destination_exists")
        if not target.parent.is_dir():
            raise _Stop("destination_unavailable")
        parent_point = None if parent is None else _load_point(Path(parent), cipher)
        if parent_point is not None:
            if parent_point.sequence + 1 >= MAX_CHAIN_LENGTH:
                raise _Stop("chain_too_long")
            _check_freshness(parent_point, started)
        point_id = f"rp_{secrets.token_hex(16)}"
        mode = "quiesced" if quiesce is not None else "operator_attested"
        with ExitStack() as stack:
            if quiesce is not None:
                try:
                    stack.enter_context(quiesce())
                except Exception:
                    raise _Stop("quiesce_failed") from None
            capture = _capture(sources, chunk_bytes)
            partial = target.parent / f"{_PARTIAL_PREFIX}{secrets.token_hex(8)}"
            _make_directory(partial, failure="backup_write_failed")
            _make_directory(partial / _OBJECTS_DIRECTORY, failure="backup_write_failed")
            objects, counts = _store_objects(
                partial,
                capture,
                parent_point,
                cipher=cipher,
                key=key,
                point_id=point_id,
                chunk_bytes=chunk_bytes,
            )
            try:
                after = _fingerprint(sources)
            except _Stop:
                raise _Stop("snapshot_boundary_moved") from None
            if after != capture.fingerprint:
                raise _Stop("snapshot_boundary_moved")
        manifest = _build_manifest(
            point_id=point_id,
            parent=parent_point,
            created_at=started,
            retention_days=retention_days,
            mode=mode,
            boundary=capture.boundary,
            objects=objects,
        )
        manifest_bytes = _canonical_json(manifest)
        _write_new(
            partial / _MANIFEST_NAME,
            cipher.seal(manifest_bytes, context=_manifest_context(point_id), key=key),
            failure="backup_write_failed",
        )
        _write_new(
            partial / _DESCRIPTOR_NAME,
            _canonical_json(_descriptor(manifest)),
            failure="backup_write_failed",
        )
        written = _load_point(partial, cipher)
        _materialize([written], cipher, None, only=point_id)
        _fsync_directory(partial / _OBJECTS_DIRECTORY)
        _fsync_directory(partial)
        if os.path.lexists(target):
            raise _Stop("destination_exists")
        try:
            os.rename(partial, target)
        except OSError:
            raise _Stop("backup_write_failed", RecoveryOutcome.FAILED) from None
        partial = None
        _fsync_directory(target.parent)
        return _receipt(
            "backup",
            RecoveryOutcome.COMPLETED,
            started,
            now(),
            point=written,
            chain_length=written.sequence + 1,
            counts=counts,
        )
    except _Stop as stop:
        return _receipt("backup", stop.outcome, started, now(), reason=stop.code)
    except EncryptionError as exc:
        return _receipt(
            "backup", RecoveryOutcome.REFUSED, started, now(), reason=exc.code
        )
    except (OSError, sqlite3.Error):
        return _receipt(
            "backup",
            RecoveryOutcome.FAILED,
            started,
            now(),
            reason="backup_write_failed",
        )
    finally:
        if partial is not None:
            _remove_tree(partial)


def verify_recovery_point(
    point: str | Path,
    *,
    key_provider: KeyProvider,
    ancestors: Sequence[str | Path] = (),
    max_age: timedelta | None = None,
    clock: Callable[[], datetime] | None = None,
) -> RecoveryReceipt:
    """Verify a recovery point and its chain without writing anything.

    Every object in the chain is decrypted and hashed in memory, and the
    same schema, cross-reference, and boundary checks as restore are
    applied.  A completed verification receipt doubles as the restore plan.

    Args:
        point: Recovery-point directory to verify.
        key_provider: Caller-owned provider resolving the backup keys.
        ancestors: Directories of the parent recovery points an incremental
            chain needs.
        max_age: Optional maximum age; older points are refused as stale.
        clock: Optional UTC clock.

    Returns:
        A value-free receipt; check :attr:`RecoveryReceipt.completed`.
    """

    now = _clock(clock)
    started = now()
    cipher = EnvelopeCipher(key_provider, purpose=BACKUP_KEY_PURPOSE)
    try:
        chain = _open_chain(Path(point), [Path(item) for item in ancestors], cipher)
        _check_chain_freshness(chain, started, max_age)
        head = chain[0]
        _check_declared_compatibility(head.manifest)
        documents, blob_hashes = _materialize(chain, cipher, None)
        _verify_semantics(head, documents, blob_hashes)
        return _receipt(
            "verify",
            RecoveryOutcome.COMPLETED,
            started,
            now(),
            point=head,
            chain_length=len(chain),
            counts={"objects_verified": len(head.manifest["objects"])},
        )
    except _Stop as stop:
        return _receipt("verify", stop.outcome, started, now(), reason=stop.code)
    except EncryptionError as exc:
        return _receipt(
            "verify", RecoveryOutcome.REFUSED, started, now(), reason=exc.code
        )
    except OSError:
        return _receipt(
            "verify",
            RecoveryOutcome.FAILED,
            started,
            now(),
            reason="point_unavailable",
        )


def restore_recovery_point(
    point: str | Path,
    target: str | Path,
    *,
    key_provider: KeyProvider,
    ancestors: Sequence[str | Path] = (),
    live: str | Path | None = None,
    max_age: timedelta | None = None,
    clock: Callable[[], datetime] | None = None,
) -> RecoveryReceipt:
    """Restore a verified recovery point into a new, inactive directory.

    The target must not exist.  Restore validates the chain, retention, and
    declared schema compatibility, decrypts every object into the canonical
    layout under ``target``, then checks digests, schema compatibility,
    cross-component references, and the snapshot boundary.  Only after all of
    that succeeds is a sealed staging record written; activation refuses a
    target without one.  Any failure, including an interruption, removes the
    target; the live state is never written.

    Args:
        point: Recovery-point directory to restore.
        target: New, inactive directory to restore into.
        key_provider: Caller-owned provider resolving the backup keys; its
            active key seals the staging record.
        ancestors: Directories of the parent recovery points an incremental
            chain needs.
        live: Optional live state path; a target overlapping it is refused.
        max_age: Optional maximum age; older points are refused as stale.
        clock: Optional UTC clock.

    Returns:
        A value-free receipt; check :attr:`RecoveryReceipt.completed`.
    """

    now = _clock(clock)
    started = now()
    cipher = EnvelopeCipher(key_provider, purpose=BACKUP_KEY_PURPOSE)
    target_path = Path(target)
    created: Path | None = None
    try:
        chain = _open_chain(Path(point), [Path(item) for item in ancestors], cipher)
        _check_chain_freshness(chain, started, max_age)
        head = chain[0]
        _check_declared_compatibility(head.manifest)
        if os.path.lexists(target_path):
            raise _Stop("target_exists")
        if not target_path.parent.is_dir():
            raise _Stop("target_unavailable")
        if live is not None and _overlaps(target_path, Path(live)):
            raise _Stop("target_overlaps_live")
        key = _active_key(cipher)
        _make_directory(target_path, failure="restore_write_failed")
        created = target_path
        metadata = target_path / RECOVERY_METADATA_DIRECTORY
        _make_directory(metadata, failure="restore_write_failed")
        _write_new(
            metadata / _STAGING_DESCRIPTOR,
            _canonical_json(_staging_descriptor(head.recovery_point_id)),
            failure="restore_write_failed",
        )
        documents, blob_hashes = _materialize(chain, cipher, target_path)
        _verify_semantics(head, documents, blob_hashes)
        record = {
            "boundary": head.manifest["boundary"],
            "chain_length": len(chain),
            "content_digest": head.manifest["content_digest"],
            "kind": head.kind,
            "manifest_digest": head.manifest_digest,
            "objects": [
                {"digest": item["digest"], "path": item["path"], "size": item["size"]}
                for item in head.manifest["objects"]
            ],
            "recovery_point_id": head.recovery_point_id,
            "restored_at": _timestamp(started),
            "schema": STAGING_SCHEMA,
        }
        _write_new(
            metadata / _STAGING_SEAL,
            cipher.seal(
                _canonical_json(record),
                context=_staging_context(head.recovery_point_id),
                key=key,
            ),
            failure="restore_write_failed",
        )
        _fsync_tree(target_path)
        created = None
        return _receipt(
            "restore",
            RecoveryOutcome.COMPLETED,
            started,
            now(),
            point=head,
            chain_length=len(chain),
            counts={"objects_restored": len(head.manifest["objects"])},
        )
    except _Stop as stop:
        return _receipt("restore", stop.outcome, started, now(), reason=stop.code)
    except EncryptionError as exc:
        return _receipt(
            "restore", RecoveryOutcome.REFUSED, started, now(), reason=exc.code
        )
    except OSError:
        return _receipt(
            "restore",
            RecoveryOutcome.FAILED,
            started,
            now(),
            reason="restore_write_failed",
        )
    finally:
        if created is not None:
            _remove_tree(created)


def activate_restored_state(
    staged: str | Path,
    live: str | Path,
    *,
    key_provider: KeyProvider,
    checks: Sequence[ActivationCheck] = (),
    epoch_hooks: Sequence[RecoveryEpochHook] = (),
    clock: Callable[[], datetime] | None = None,
) -> RecoveryReceipt:
    """Atomically make a verified restored target the live service state.

    ``live`` must be absent or a symbolic link (the managed pointer).  The
    sealed staging record is authenticated and every staged file is re-hashed
    against it; then each check runs against the staged root, then each
    recovery-epoch hook observes the activation, and only then is ``live``
    atomically repointed.  Any refusal leaves ``live`` untouched.  The
    previous target is kept for inspection; roll back by restoring again.

    Args:
        staged: Directory produced by :func:`restore_recovery_point`.
        live: Path of the live state symlink the service is configured with.
        key_provider: Caller-owned provider resolving the backup keys.
        checks: Pre-activation checks, for example
            :func:`journey_store_check` and the production self-check.
        epoch_hooks: Recovery-epoch hooks that must accept the activation.
        clock: Optional UTC clock.

    Returns:
        A value-free receipt; check :attr:`RecoveryReceipt.completed`.
    """

    for hook in epoch_hooks:
        if not isinstance(hook, RecoveryEpochHook):
            raise TypeError("epoch hooks must implement open_recovery_epoch")
    for check in checks:
        if not callable(check):
            raise TypeError("activation checks must be callable")
    now = _clock(clock)
    started = now()
    cipher = EnvelopeCipher(key_provider, purpose=BACKUP_KEY_PURPOSE)
    try:
        if sys.platform == "win32":
            raise _Stop("atomic_swap_unsupported")
        staged_path = Path(staged)
        live_path = Path(live)
        if staged_path.is_symlink() or not staged_path.is_dir():
            raise _Stop("staged_state_unavailable")
        staged_root = staged_path.resolve(strict=True)
        if not live_path.parent.is_dir():
            raise _Stop("live_target_unavailable")
        previous = False
        if os.path.lexists(live_path):
            if not live_path.is_symlink():
                raise _Stop("live_target_unmanaged")
            current = live_path.resolve()
            if current == staged_root:
                raise _Stop("already_active")
            if _is_within(staged_root, current) or _is_within(current, staged_root):
                raise _Stop("target_overlaps_live")
            previous = True
        if _is_within(live_path.parent.resolve(), staged_root):
            raise _Stop("target_overlaps_live")
        record = _open_staging(staged_root, cipher)
        _verify_staged_tree(staged_root, record)
        for check in checks:
            _run_check(check, staged_root)
        context = RecoveryEpochContext(
            recovery_point_id=record["recovery_point_id"],
            manifest_digest=record["manifest_digest"],
            content_digest=record["content_digest"],
            kind=record["kind"],
            chain_length=record["chain_length"],
            boundary=record["boundary"],
            activated_at=_timestamp(started),
        )
        for hook in epoch_hooks:
            try:
                hook.open_recovery_epoch(context)
            except Exception:
                raise _Stop("recovery_epoch_refused") from None
        _swap_symlink(live_path, staged_root)
        _write_activation_record(staged_root, context)
        staged_objects = [
            {**item, "category": _path_category(item["path"])}
            for item in record["objects"]
        ]
        return RecoveryReceipt(
            operation="activate",
            outcome=RecoveryOutcome.COMPLETED,
            recovery_point_id=context.recovery_point_id,
            kind=context.kind,
            chain_length=context.chain_length,
            content_digest=context.content_digest,
            manifest_digest=context.manifest_digest,
            components=tuple(_components(staged_objects, context.boundary)),
            boundary=context.boundary,
            counts={
                "checks_passed": len(checks),
                "epoch_hooks": len(epoch_hooks),
                "previous_target": int(previous),
            },
            started_at=_timestamp(started),
            finished_at=_timestamp(now()),
        )
    except _Stop as stop:
        return _receipt(
            "activate",
            stop.outcome,
            started,
            now(),
            reason=stop.code,
            details=stop.details,
        )
    except EncryptionError as exc:
        return _receipt(
            "activate", RecoveryOutcome.REFUSED, started, now(), reason=exc.code
        )
    except OSError:
        return _receipt(
            "activate",
            RecoveryOutcome.FAILED,
            started,
            now(),
            reason="activation_failed",
        )


def discard_restored_state(
    staged: str | Path,
    *,
    live: str | Path | None = None,
    clock: Callable[[], datetime] | None = None,
) -> RecoveryReceipt:
    """Remove an inactive restored target and return a cleanup receipt.

    Only directories created by :func:`restore_recovery_point` (they carry a
    staging descriptor) are removed, and never the target ``live`` points to.

    Args:
        staged: Restored target directory to remove.
        live: Optional live state symlink; its current target is protected.
        clock: Optional UTC clock.

    Returns:
        A value-free receipt; check :attr:`RecoveryReceipt.completed`.
    """

    now = _clock(clock)
    started = now()
    staged_path = Path(staged)
    try:
        if staged_path.is_symlink() or not staged_path.is_dir():
            raise _Stop("not_a_restored_target")
        try:
            descriptor = _read_staging_descriptor(staged_path)
        except _Stop:
            raise _Stop("not_a_restored_target") from None
        if live is not None:
            live_path = Path(live)
            if os.path.lexists(live_path) and (
                live_path.resolve() == staged_path.resolve()
                or _is_within(live_path.resolve(), staged_path.resolve())
            ):
                raise _Stop("staged_is_live")
        try:
            shutil.rmtree(staged_path)
        except OSError:
            raise _Stop("cleanup_failed", RecoveryOutcome.FAILED) from None
        return RecoveryReceipt(
            operation="cleanup",
            outcome=RecoveryOutcome.COMPLETED,
            recovery_point_id=descriptor["recovery_point_id"],
            started_at=_timestamp(started),
            finished_at=_timestamp(now()),
        )
    except _Stop as stop:
        return _receipt("cleanup", stop.outcome, started, now(), reason=stop.code)


def journey_store_check(encryption: Any) -> ActivationCheck:
    """Return an activation check that opens the restored Journey store.

    The check opens the staged store with the caller's Journey encryption
    configuration, which verifies the persisted key verifiers, then runs the
    store's integrity check (SQLite integrity, foreign keys, and every payload
    hash) and reads and verifies every artifact.  Missing, revoked, or wrong
    Journey keys therefore fail closed before activation.

    A restored store older than this binary's schema (for example a
    pre-upgrade recovery point restored to roll back to the previous release)
    is never opened, because opening would forward-migrate it.  It is
    verified read-only in memory instead: SQLite integrity, foreign keys, and
    a migration history that is a known prefix of this binary's chain.  Such
    stores predate store encryption, so no key is involved.

    Args:
        encryption: A
            :class:`~openmed.structured.store.JourneyStoreEncryption` for the
            restored store.

    Returns:
        A check named ``journey_store``.
    """

    from openmed.structured.store import JourneyStoreEncryption

    if not isinstance(encryption, JourneyStoreEncryption):
        raise TypeError("encryption must be a JourneyStoreEncryption")

    def check(root: Path) -> ActivationCheckResult:
        from openmed.structured.store import LocalJourneyStore

        journey = Path(root) / JOURNEY_DIRECTORY
        if not journey.is_dir():
            return ActivationCheckResult("journey_store", True)
        try:
            staged_version = _staged_schema_version(journey / _JOURNEY_DATABASE_NAME)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return ActivationCheckResult("journey_store", False, "journey_open_failed")
        if staged_version < LATEST_MIGRATION_VERSION:
            return _check_previous_schema(journey / _JOURNEY_DATABASE_NAME)
        try:
            hashes = _staged_content_hashes(journey / _JOURNEY_DATABASE_NAME)
            store = LocalJourneyStore(journey, encryption=encryption)
        except EncryptionError as exc:
            return ActivationCheckResult("journey_store", False, exc.code)
        except (RuntimeError, OSError, sqlite3.Error, ValueError):
            return ActivationCheckResult("journey_store", False, "journey_open_failed")
        try:
            integrity = store.metadata.integrity_check()
            if not integrity.ok:
                return ActivationCheckResult(
                    "journey_store", False, integrity.code or "integrity_failed"
                )
            for content_hash in hashes:
                blob = store.artifacts.get_bytes(content_hash)
                if not blob.ok:
                    return ActivationCheckResult(
                        "journey_store", False, blob.code or "artifact_unreadable"
                    )
        finally:
            store.close()
        return ActivationCheckResult("journey_store", True)

    return check


# ---------------------------------------------------------------------------
# Source enumeration and capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SourceFile:
    logical: str
    category: ComponentCategory
    file: Path = field(repr=False)


@dataclass(frozen=True)
class _Captured:
    logical: str
    category: ComponentCategory
    size: int
    digest: str
    data: bytes | None = field(default=None, repr=False)
    file: Path | None = field(default=None, repr=False)


@dataclass(frozen=True)
class _Capture:
    fingerprint: tuple[Any, ...]
    boundary: dict[str, Any]
    items: tuple[_Captured, ...]


@dataclass(frozen=True)
class _JourneyFacts:
    migrations: tuple[tuple[int, str, str], ...]
    revision: int | None
    content_hashes: frozenset[str]
    encryption_state: str | None
    key_purpose: str | None
    key_ids: tuple[str, ...]

    @property
    def schema_version(self) -> int:
        return max((item[0] for item in self.migrations), default=0)

    def signature(self) -> tuple[Any, ...]:
        return (
            self.migrations,
            self.revision,
            len(self.content_hashes),
            self.encryption_state,
            self.key_purpose,
            self.key_ids,
        )


def _require_sources(sources: object) -> None:
    if not isinstance(sources, ServiceStateSources):
        raise TypeError("sources must be ServiceStateSources")


def _require_chunk_bytes(chunk_bytes: object) -> None:
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError("chunk_bytes must be a positive bounded integer")


def _enumerate(
    sources: ServiceStateSources,
) -> tuple[Path | None, list[_SourceFile]]:
    files: list[_SourceFile] = []
    database: Path | None = None
    try:
        if sources.journey_root is not None:
            root = sources.journey_root
            if root.is_symlink():
                raise _Stop("source_invalid")
            if not root.is_dir():
                raise _Stop("source_unavailable")
            database = root / _JOURNEY_DATABASE_NAME
            _require_regular(database)
            artifacts = root / _JOURNEY_ARTIFACTS_NAME
            if artifacts.is_symlink():
                raise _Stop("source_invalid")
            if artifacts.is_dir():
                marker = artifacts / _JOURNEY_MARKER_NAME
                if os.path.lexists(marker):
                    _require_regular(marker)
                    files.append(
                        _SourceFile(
                            JOURNEY_MARKER_PATH,
                            ComponentCategory.JOURNEY_ENCRYPTION,
                            marker,
                        )
                    )
                files.extend(_blob_files(artifacts / "blobs" / "sha256"))
        if sources.job_store is not None:
            _require_regular(sources.job_store)
            files.append(
                _SourceFile(
                    JOB_STORE_PATH, ComponentCategory.JOB_METADATA, sources.job_store
                )
            )
        if sources.registry_state is not None:
            _require_regular(sources.registry_state)
            files.append(
                _SourceFile(
                    REGISTRY_STATE_FILE,
                    ComponentCategory.MODEL_REGISTRY,
                    sources.registry_state,
                )
            )
        for index, chain in enumerate(sources.audit_chains):
            _require_regular(chain)
            files.append(
                _SourceFile(_audit_path(index), ComponentCategory.AUDIT_CHAIN, chain)
            )
    except OSError:
        raise _Stop("source_unavailable") from None
    return database, files


def _blob_files(blob_root: Path) -> list[_SourceFile]:
    if blob_root.is_symlink():
        raise _Stop("source_invalid")
    if not blob_root.is_dir():
        return []
    found: list[_SourceFile] = []
    for prefix in sorted(blob_root.iterdir()):
        if (
            prefix.is_symlink()
            or not prefix.is_dir()
            or _BLOB_PREFIX_RE.fullmatch(prefix.name) is None
        ):
            continue
        for candidate in sorted(prefix.iterdir()):
            if (
                _BLOB_NAME_RE.fullmatch(candidate.name) is not None
                and candidate.name.startswith(prefix.name)
                and not candidate.is_symlink()
                and candidate.is_file()
            ):
                found.append(
                    _SourceFile(
                        f"{JOURNEY_BLOB_ROOT}/{prefix.name}/{candidate.name}",
                        ComponentCategory.JOURNEY_ARTIFACTS,
                        candidate,
                    )
                )
    return found


def _audit_path(index: int) -> str:
    return f"{AUDIT_CHAIN_DIRECTORY}/chain-{index:04d}.json"


def _require_regular(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise _Stop("source_unavailable") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise _Stop("source_invalid")


def _size(path: Path) -> int:
    return os.lstat(path).st_size


def _inventory_version(item: _SourceFile) -> int | None:
    if item.category is ComponentCategory.JOURNEY_ARTIFACTS:
        return None
    if item.category is ComponentCategory.JOB_METADATA:
        return JOB_STORE_LAYOUT_VERSION
    if item.category is ComponentCategory.AUDIT_CHAIN:
        return CHAIN_VERSION
    data = _read_regular(item.file, limit=_MAX_DOCUMENT_BYTES)
    if item.category is ComponentCategory.JOURNEY_ENCRYPTION:
        return _parse_marker(data, failure="source_invalid")["schema_version"]
    return _parse_registry(data, failure="source_invalid")


def _fingerprint(sources: ServiceStateSources) -> tuple[Any, ...]:
    database, files = _enumerate(sources)
    stats = []
    try:
        for item in files:
            info = os.lstat(item.file)
            stats.append((item.logical, info.st_size, info.st_mtime_ns, info.st_ino))
    except OSError:
        raise _Stop("source_unavailable") from None
    journey = None if database is None else _live_journey_facts(database).signature()
    return (journey, tuple(stats))


def _capture(sources: ServiceStateSources, chunk_bytes: int) -> _Capture:
    fingerprint = _fingerprint(sources)
    database, files = _enumerate(sources)
    items: list[_Captured] = []
    documents: dict[str, bytes] = {}
    blob_hashes: set[str] = set()
    if database is not None:
        image = _snapshot_journey(database)
        documents[JOURNEY_DATABASE_PATH] = image
        items.append(
            _Captured(
                JOURNEY_DATABASE_PATH,
                ComponentCategory.JOURNEY_METADATA,
                len(image),
                _digest(image),
                data=image,
            )
        )
    for source in files:
        if source.category is ComponentCategory.JOURNEY_ARTIFACTS:
            size, digest = _hash_file(source.file, chunk_bytes)
            items.append(
                _Captured(
                    source.logical, source.category, size, digest, file=source.file
                )
            )
            blob_hashes.add("sha256:" + source.logical.rsplit("/", 1)[1])
            continue
        data = _read_regular(source.file, limit=_MAX_DOCUMENT_BYTES)
        documents[source.logical] = data
        items.append(
            _Captured(source.logical, source.category, len(data), _digest(data), data)
        )
    boundary, facts = _derive_boundary(
        documents, frozenset(blob_hashes), failure="source_invalid"
    )
    if facts is not None and facts.signature() != fingerprint[0]:
        raise _Stop("snapshot_boundary_moved")
    items.sort(key=lambda item: item.logical)
    return _Capture(fingerprint=fingerprint, boundary=boundary, items=tuple(items))


def _connect_existing(path: Path) -> sqlite3.Connection:
    try:
        # ``mode=rw`` never creates a missing database.
        return sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=rw",
            uri=True,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
    except (OSError, sqlite3.Error, ValueError):
        raise _Stop("source_unavailable") from None


def _snapshot_journey(database: Path) -> bytes:
    source = _connect_existing(database)
    try:
        memory = sqlite3.connect(":memory:")
        try:
            # One step copies every page under a single read transaction, so
            # the image is one committed snapshot even while readers run.
            source.backup(memory, pages=-1)
            image = bytearray(memory.serialize())
        finally:
            memory.close()
    except sqlite3.Error:
        raise _Stop("source_unavailable") from None
    finally:
        source.close()
    if len(image) >= 20:
        # The image is self-contained; mark it as a rollback-journal file so
        # verification and restore never create -wal/-shm side files.  The
        # store switches it back to WAL when it opens the database.
        image[18] = 1
        image[19] = 1
    return bytes(image)


def _journey_facts(connection: sqlite3.Connection) -> _JourneyFacts:
    migrations = tuple(
        (int(row[0]), str(row[1]), str(row[2]))
        for row in connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        )
    )
    revision = connection.execute(
        "SELECT MAX(revision) FROM store_revisions"
    ).fetchone()[0]
    hashes = frozenset(
        str(row[0]) for row in connection.execute("SELECT content_hash FROM artifacts")
    )
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    state: str | None = None
    purpose: str | None = None
    key_ids: tuple[str, ...] = ()
    if "store_encryption" in tables:
        row = connection.execute(
            "SELECT state, key_purpose FROM store_encryption WHERE singleton = 1"
        ).fetchone()
        if row is not None:
            state = None if row[0] is None else str(row[0])
            purpose = None if row[1] is None else str(row[1])
    if "store_encryption_keys" in tables:
        key_ids = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT key_id FROM store_encryption_keys ORDER BY key_id"
            )
        )
    if revision is not None and type(revision) is not int:
        raise ValueError("revision must be an integer")
    if any(_DIGEST_RE.fullmatch(item) is None for item in hashes):
        raise ValueError("content hashes must be normalized digests")
    return _JourneyFacts(
        migrations=migrations,
        revision=revision,
        content_hashes=hashes,
        encryption_state=state,
        key_purpose=purpose,
        key_ids=key_ids,
    )


def _live_journey_facts(database: Path) -> _JourneyFacts:
    connection = _connect_existing(database)
    try:
        return _journey_facts(connection)
    except (sqlite3.Error, ValueError):
        raise _Stop("source_invalid") from None
    finally:
        connection.close()


def _inspect_journey_image(image: bytes, *, failure: str) -> _JourneyFacts:
    memory = sqlite3.connect(":memory:")
    try:
        memory.deserialize(image)
        integrity = memory.execute("SELECT * FROM pragma_integrity_check()").fetchall()
        if [tuple(row) for row in integrity] != [("ok",)]:
            raise _Stop(failure)
        if memory.execute("SELECT * FROM pragma_foreign_key_check()").fetchone():
            raise _Stop(failure)
        return _journey_facts(memory)
    except (sqlite3.Error, ValueError, OverflowError):
        raise _Stop(failure) from None
    finally:
        memory.close()


def _parse_json(data: bytes, *, failure: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise _Stop(failure) from None


def _parse_marker(data: bytes, *, failure: str) -> dict[str, Any]:
    document = _parse_json(data, failure=failure)
    if (
        not isinstance(document, dict)
        or set(document) != _MARKER_FIELDS
        or type(document["schema_version"]) is not int
        or not isinstance(document["key_verifiers"], dict)
        or not isinstance(document["state"], str)
        or (
            document["key_purpose"] is not None
            and not isinstance(document["key_purpose"], str)
        )
    ):
        raise _Stop(failure)
    return document


def _parse_job_store(data: bytes, *, failure: str) -> int:
    document = _parse_json(data, failure=failure)
    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, dict) or not all(
        isinstance(record, dict) for record in jobs.values()
    ):
        raise _Stop(failure)
    return len(jobs)


def _parse_registry(data: bytes, *, failure: str) -> int:
    document = _parse_json(data, failure=failure)
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or not isinstance(document.get("slots"), dict)
    ):
        raise _Stop(failure)
    return int(document["schema_version"])


def _parse_audit(data: bytes, *, failure: str) -> dict[str, Any]:
    try:
        chain = AuditChain.from_json(data)
    except (ValueError, TypeError, AttributeError, RecursionError):
        raise _Stop(failure) from None
    if not chain.verify().valid:
        raise _Stop(failure)
    return {"entries": len(chain.entries), "head": chain.head_hash}


def _derive_boundary(
    documents: Mapping[str, bytes],
    blob_hashes: frozenset[str],
    *,
    failure: str,
) -> tuple[dict[str, Any], _JourneyFacts | None]:
    """Return snapshot bounds, verifying content and cross-references."""

    image = documents.get(JOURNEY_DATABASE_PATH)
    marker_data = documents.get(JOURNEY_MARKER_PATH)
    marker = (
        None if marker_data is None else _parse_marker(marker_data, failure=failure)
    )
    facts: _JourneyFacts | None = None
    journey: dict[str, Any] | None = None
    if image is not None:
        facts = _inspect_journey_image(image, failure=failure)
        if facts.content_hashes - blob_hashes:
            raise _Stop("cross_reference_missing_artifact")
        if blob_hashes and marker is None:
            raise _Stop("encryption_metadata_missing")
        if (
            marker is not None
            and facts.encryption_state not in {None, "plaintext"}
            and marker["state"] != "plaintext"
            and marker["key_purpose"] != facts.key_purpose
        ):
            raise _Stop("encryption_state_mismatch")
        journey = {
            "marker_schema_version": (
                None if marker is None else marker["schema_version"]
            ),
            "revision": facts.revision,
            "schema_version": facts.schema_version,
        }
    elif marker is not None or blob_hashes:
        raise _Stop(failure)
    job_data = documents.get(JOB_STORE_PATH)
    registry_data = documents.get(REGISTRY_STATE_FILE)
    audit = [
        _parse_audit(documents[path], failure=failure)
        for path in sorted(documents)
        if path.startswith(f"{AUDIT_CHAIN_DIRECTORY}/")
    ]
    boundary = {
        "audit": audit,
        "jobs": (
            None
            if job_data is None
            else {"records": _parse_job_store(job_data, failure=failure)}
        ),
        "journey": journey,
        "registry": (
            None
            if registry_data is None
            else {"schema_version": _parse_registry(registry_data, failure=failure)}
        ),
    }
    return boundary, facts


def _read_regular(path: Path, *, limit: int) -> bytes:
    if path.is_symlink():
        raise _Stop("source_invalid")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise _Stop("source_unavailable") from None
    except OSError:
        raise _Stop("source_invalid") from None
    try:
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise _Stop("source_invalid")
            data = handle.read(limit + 1)
    except OSError:
        raise _Stop("source_unavailable") from None
    if len(data) > limit:
        raise _Stop("source_invalid")
    return data


def _iter_file(path: Path, chunk_bytes: int) -> Iterator[bytes]:
    if path.is_symlink():
        raise _Stop("source_invalid")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise _Stop("component_changed_during_capture") from None
    except OSError:
        raise _Stop("source_invalid") from None
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise _Stop("source_invalid")
        while True:
            try:
                chunk = handle.read(chunk_bytes)
            except OSError:
                raise _Stop("source_unavailable") from None
            if not chunk:
                return
            yield chunk


def _hash_file(path: Path, chunk_bytes: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in _iter_file(path, chunk_bytes):
        digest.update(chunk)
        size += len(chunk)
    return size, f"sha256:{digest.hexdigest()}"


def _plaintext_chunks(item: _Captured, chunk_bytes: int) -> Iterator[bytes]:
    if item.data is not None:
        for offset in range(0, max(len(item.data), 1), chunk_bytes):
            yield item.data[offset : offset + chunk_bytes]
        return
    if item.file is None:
        raise _Stop("component_changed_during_capture")
    produced = False
    for chunk in _iter_file(item.file, chunk_bytes):
        produced = True
        yield chunk
    if not produced:
        yield b""


def _chunk_count(size: int, chunk_bytes: int) -> int:
    return max(1, -(-size // chunk_bytes))


# ---------------------------------------------------------------------------
# Sealed object streams
# ---------------------------------------------------------------------------


def _chunk_context(point_id: str, name: str, index: int, count: int) -> dict[str, Any]:
    return {
        "count": count,
        "index": index,
        "object": name,
        "point": point_id,
        "record": "recovery_chunk",
    }


def _manifest_context(point_id: str) -> dict[str, Any]:
    return {"point": point_id, "record": "recovery_manifest"}


def _staging_context(point_id: str) -> dict[str, Any]:
    return {"point": point_id, "record": "recovery_staging"}


def _store_objects(
    partial: Path,
    capture: _Capture,
    parent: "_Point | None",
    *,
    cipher: EnvelopeCipher,
    key: DataKey,
    point_id: str,
    chunk_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    inherited = (
        {}
        if parent is None
        else {item["path"]: item for item in parent.manifest["objects"]}
    )
    objects: list[dict[str, Any]] = []
    stored = reused = stored_bytes = 0
    for item in capture.items:
        previous = inherited.get(item.logical)
        if (
            previous is not None
            and previous["digest"] == item.digest
            and previous["size"] == item.size
        ):
            objects.append(
                _object_entry(
                    item,
                    stored_in=previous["stored_in"],
                    name=previous["object"],
                    chunks=previous["chunks"],
                )
            )
            reused += 1
            continue
        name = f"{stored:06d}"
        count = _chunk_count(item.size, chunk_bytes)
        _seal_object(
            partial / _OBJECTS_DIRECTORY / f"{name}{_OBJECT_SUFFIX}",
            item,
            cipher=cipher,
            key=key,
            point_id=point_id,
            name=name,
            count=count,
            chunk_bytes=chunk_bytes,
        )
        objects.append(_object_entry(item, stored_in=point_id, name=name, chunks=count))
        stored += 1
        stored_bytes += item.size
    counts = {
        "bytes_stored": stored_bytes,
        "objects_reused": reused,
        "objects_stored": stored,
        "objects_total": len(objects),
    }
    return objects, counts


def _object_entry(
    item: _Captured, *, stored_in: str, name: str, chunks: int
) -> dict[str, Any]:
    return {
        "category": item.category.value,
        "chunks": chunks,
        "digest": item.digest,
        "object": name,
        "path": item.logical,
        "size": item.size,
        "stored_in": stored_in,
    }


def _seal_object(
    path: Path,
    item: _Captured,
    *,
    cipher: EnvelopeCipher,
    key: DataKey,
    point_id: str,
    name: str,
    count: int,
    chunk_bytes: int,
) -> None:
    descriptor = _create_file(path, failure="backup_write_failed")
    try:
        digest = hashlib.sha256()
        size = 0
        index = 0
        _write_all(
            descriptor, _STREAM_HEADER.pack(_STREAM_MAGIC, _STREAM_VERSION, count)
        )
        for chunk in _plaintext_chunks(item, chunk_bytes):
            if index >= count:
                raise _Stop("component_changed_during_capture")
            digest.update(chunk)
            size += len(chunk)
            envelope = cipher.seal(
                chunk,
                context=_chunk_context(point_id, name, index, count),
                key=key,
            )
            _write_all(descriptor, _CHUNK_LENGTH.pack(len(envelope)) + envelope)
            index += 1
        if (
            index != count
            or size != item.size
            or f"sha256:{digest.hexdigest()}" != item.digest
        ):
            raise _Stop("component_changed_during_capture")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_stream(
    path: Path,
    *,
    point_id: str,
    name: str,
    chunks: int,
    cipher: EnvelopeCipher,
) -> Iterator[bytes]:
    if path.is_symlink():
        raise _Stop("object_corrupt")
    try:
        handle = open(path, "rb")  # noqa: SIM115 - closed in the finally block
    except FileNotFoundError:
        raise _Stop("object_missing") from None
    except OSError:
        raise _Stop("point_unavailable") from None
    try:
        header = _read_exact(handle, _STREAM_HEADER.size)
        magic, version, count = _STREAM_HEADER.unpack(header)
        if magic != _STREAM_MAGIC:
            raise _Stop("object_corrupt")
        if version != _STREAM_VERSION:
            raise _Stop("format_unsupported")
        if count != chunks:
            raise _Stop("object_corrupt")
        for index in range(count):
            (length,) = _CHUNK_LENGTH.unpack(_read_exact(handle, _CHUNK_LENGTH.size))
            if length > MAX_CHUNK_BYTES + _ENVELOPE_OVERHEAD:
                raise _Stop("object_corrupt")
            envelope = _read_exact(handle, length)
            try:
                yield cipher.open(
                    envelope, context=_chunk_context(point_id, name, index, count)
                )
            except EnvelopeAuthenticationError:
                raise _Stop("object_authentication_failed") from None
            except EnvelopeVersionError:
                raise _Stop("format_unsupported") from None
            except EnvelopeFormatError:
                raise _Stop("object_corrupt") from None
        if handle.read(1):
            raise _Stop("object_corrupt")
    except OSError:
        raise _Stop("point_unavailable") from None
    finally:
        handle.close()


def _read_exact(handle: Any, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise _Stop("object_corrupt")
    return bytes(data)


# ---------------------------------------------------------------------------
# Manifests and recovery-point loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Point:
    root: Path = field(repr=False)
    recovery_point_id: str
    kind: str
    sequence: int
    manifest: Mapping[str, Any] = field(repr=False)
    manifest_digest: str


def _descriptor(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": RECOVERY_FORMAT,
        "format_version": RECOVERY_FORMAT_VERSION,
        "kind": manifest["kind"],
        "parent_recovery_point_id": manifest["chain"]["parent_recovery_point_id"],
        "recovery_point_id": manifest["recovery_point_id"],
    }


def _build_manifest(
    *,
    point_id: str,
    parent: _Point | None,
    created_at: datetime,
    retention_days: int | None,
    mode: str,
    boundary: dict[str, Any],
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    if parent is None:
        chain = {
            "base_recovery_point_id": point_id,
            "parent_manifest_digest": None,
            "parent_recovery_point_id": None,
            "sequence": 0,
        }
        kind = RecoveryKind.FULL.value
    else:
        chain = {
            "base_recovery_point_id": parent.manifest["chain"][
                "base_recovery_point_id"
            ],
            "parent_manifest_digest": parent.manifest_digest,
            "parent_recovery_point_id": parent.recovery_point_id,
            "sequence": parent.sequence + 1,
        }
        kind = RecoveryKind.INCREMENTAL.value
    expires_at = (
        None
        if retention_days is None
        else _timestamp(created_at + timedelta(days=retention_days))
    )
    return {
        "boundary": boundary,
        "chain": chain,
        "components": [item.to_dict() for item in _components(objects, boundary)],
        "consistency": {"mode": mode},
        "content_digest": _content_digest(boundary, objects),
        "created_at": _timestamp(created_at),
        "format_version": RECOVERY_FORMAT_VERSION,
        "kind": kind,
        "objects": objects,
        "producer": {"openmed_version": str(__version__)[:_MAX_PRODUCER_VERSION]},
        "recovery_point_id": point_id,
        "retention": {"expires_at": expires_at},
        "schema": MANIFEST_SCHEMA,
    }


def _content_digest(
    boundary: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]
) -> str:
    logical = {
        "boundary": boundary,
        "objects": [
            {
                "category": item["category"],
                "digest": item["digest"],
                "path": item["path"],
                "size": item["size"],
            }
            for item in objects
        ],
    }
    return _digest(_canonical_json(logical))


def _components(
    objects: Sequence[Mapping[str, Any]], boundary: Mapping[str, Any]
) -> list[ComponentSummary]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in objects:
        grouped.setdefault(item["category"], []).append(item)
    journey = boundary.get("journey") or {}
    registry = boundary.get("registry") or {}
    versions: dict[str, int | None] = {
        ComponentCategory.JOURNEY_METADATA.value: journey.get("schema_version"),
        ComponentCategory.JOURNEY_ARTIFACTS.value: None,
        ComponentCategory.JOURNEY_ENCRYPTION.value: journey.get(
            "marker_schema_version"
        ),
        ComponentCategory.JOB_METADATA.value: JOB_STORE_LAYOUT_VERSION,
        ComponentCategory.MODEL_REGISTRY.value: registry.get("schema_version"),
        ComponentCategory.AUDIT_CHAIN.value: CHAIN_VERSION,
    }
    summaries = []
    for category in _CATEGORY_ORDER:
        members = grouped.get(category.value)
        if not members:
            continue
        summaries.append(
            ComponentSummary(
                category=category.value,
                objects=len(members),
                bytes=sum(int(item["size"]) for item in members),
                digest=_digest(
                    _canonical_json(
                        [
                            [item["path"], item["digest"], item["size"]]
                            for item in members
                        ]
                    )
                ),
                schema_version=versions[category.value],
            )
        )
    return summaries


def _read_descriptor(root: Path) -> dict[str, Any]:
    path = root / _DESCRIPTOR_NAME
    if root.is_symlink() or not root.is_dir():
        raise _Stop("point_unavailable")
    data = _read_bounded(
        path,
        _MAX_DESCRIPTOR_BYTES,
        missing="point_unavailable",
        invalid="descriptor_invalid",
        unavailable="point_unavailable",
    )
    document = _parse_json(data, failure="descriptor_invalid")
    if not isinstance(document, dict) or set(document) != {
        "format",
        "format_version",
        "kind",
        "parent_recovery_point_id",
        "recovery_point_id",
    }:
        raise _Stop("descriptor_invalid")
    if document["format"] != RECOVERY_FORMAT:
        raise _Stop("descriptor_invalid")
    if document["format_version"] != RECOVERY_FORMAT_VERSION:
        raise _Stop("format_unsupported")
    if not _valid_point_id(document["recovery_point_id"]) or document["kind"] not in {
        RecoveryKind.FULL.value,
        RecoveryKind.INCREMENTAL.value,
    }:
        raise _Stop("descriptor_invalid")
    parent = document["parent_recovery_point_id"]
    if parent is not None and not _valid_point_id(parent):
        raise _Stop("descriptor_invalid")
    return document


def _load_point(root: Path, cipher: EnvelopeCipher) -> _Point:
    descriptor = _read_descriptor(root)
    point_id = descriptor["recovery_point_id"]
    sealed = _read_bounded(
        root / _MANIFEST_NAME,
        _MAX_SEALED_DOCUMENT_BYTES,
        missing="manifest_invalid",
        invalid="manifest_invalid",
        unavailable="point_unavailable",
    )
    try:
        plaintext = cipher.open(sealed, context=_manifest_context(point_id))
    except EnvelopeAuthenticationError:
        raise _Stop("manifest_authentication_failed") from None
    except EnvelopeVersionError:
        raise _Stop("format_unsupported") from None
    except EnvelopeFormatError:
        raise _Stop("manifest_invalid") from None
    manifest = _parse_json(plaintext, failure="manifest_invalid")
    _validate_manifest(manifest)
    if (
        manifest["recovery_point_id"] != point_id
        or manifest["kind"] != descriptor["kind"]
        or manifest["chain"]["parent_recovery_point_id"]
        != descriptor["parent_recovery_point_id"]
    ):
        raise _Stop("descriptor_mismatch")
    return _Point(
        root=root,
        recovery_point_id=point_id,
        kind=manifest["kind"],
        sequence=manifest["chain"]["sequence"],
        manifest=manifest,
        manifest_digest=_digest(plaintext),
    )


def _validate_manifest(manifest: Any) -> None:
    expected = {
        "boundary",
        "chain",
        "components",
        "consistency",
        "content_digest",
        "created_at",
        "format_version",
        "kind",
        "objects",
        "producer",
        "recovery_point_id",
        "retention",
        "schema",
    }
    try:
        if not isinstance(manifest, dict) or set(manifest) != expected:
            raise ValueError
        if manifest["schema"] != MANIFEST_SCHEMA:
            raise _Stop("format_unsupported")
        if manifest["format_version"] != RECOVERY_FORMAT_VERSION:
            raise _Stop("format_unsupported")
        point_id = manifest["recovery_point_id"]
        if not _valid_point_id(point_id):
            raise ValueError
        _validate_chain(manifest["chain"], manifest["kind"], point_id)
        if not _valid_timestamp(manifest["created_at"]):
            raise ValueError
        retention = manifest["retention"]
        if not isinstance(retention, dict) or set(retention) != {"expires_at"}:
            raise ValueError
        if retention["expires_at"] is not None and not _valid_timestamp(
            retention["expires_at"]
        ):
            raise ValueError
        consistency = manifest["consistency"]
        if consistency != {"mode": "quiesced"} and consistency != {
            "mode": "operator_attested"
        }:
            raise ValueError
        producer = manifest["producer"]
        if (
            not isinstance(producer, dict)
            or set(producer) != {"openmed_version"}
            or not isinstance(producer["openmed_version"], str)
            or len(producer["openmed_version"]) > _MAX_PRODUCER_VERSION
        ):
            raise ValueError
        _validate_boundary(manifest["boundary"])
        objects = manifest["objects"]
        if not isinstance(objects, list):
            raise ValueError
        paths = [_validate_object(item, manifest["kind"], point_id) for item in objects]
        if paths != sorted(set(paths)):
            raise ValueError
        if manifest["components"] != [
            item.to_dict() for item in _components(objects, manifest["boundary"])
        ]:
            raise ValueError
        if manifest["content_digest"] != _content_digest(manifest["boundary"], objects):
            raise ValueError
    except _Stop:
        raise
    except (ValueError, TypeError, KeyError, AttributeError):
        raise _Stop("manifest_invalid") from None


def _validate_chain(chain: Any, kind: Any, point_id: str) -> None:
    if not isinstance(chain, dict) or set(chain) != {
        "base_recovery_point_id",
        "parent_manifest_digest",
        "parent_recovery_point_id",
        "sequence",
    }:
        raise ValueError
    sequence = chain["sequence"]
    if type(sequence) is not int or not 0 <= sequence < MAX_CHAIN_LENGTH:
        raise ValueError
    if kind == RecoveryKind.FULL.value:
        if (
            sequence != 0
            or chain["base_recovery_point_id"] != point_id
            or chain["parent_manifest_digest"] is not None
            or chain["parent_recovery_point_id"] is not None
        ):
            raise ValueError
    elif kind == RecoveryKind.INCREMENTAL.value:
        if (
            sequence < 1
            or not _valid_point_id(chain["base_recovery_point_id"])
            or not _valid_point_id(chain["parent_recovery_point_id"])
            or not _valid_digest(chain["parent_manifest_digest"])
            or chain["parent_recovery_point_id"] == point_id
        ):
            raise ValueError
    else:
        raise ValueError


def _validate_boundary(boundary: Any) -> None:
    if not isinstance(boundary, dict) or set(boundary) != {
        "audit",
        "jobs",
        "journey",
        "registry",
    }:
        raise ValueError
    journey = boundary["journey"]
    if journey is not None:
        if set(journey) != {"marker_schema_version", "revision", "schema_version"}:
            raise ValueError
        if journey["revision"] is not None and type(journey["revision"]) is not int:
            raise ValueError
        if type(journey["schema_version"]) is not int:
            raise ValueError
        marker = journey["marker_schema_version"]
        if marker is not None and type(marker) is not int:
            raise ValueError
    jobs = boundary["jobs"]
    if jobs is not None and (
        set(jobs) != {"records"} or type(jobs["records"]) is not int
    ):
        raise ValueError
    registry = boundary["registry"]
    if registry is not None and (
        set(registry) != {"schema_version"}
        or type(registry["schema_version"]) is not int
    ):
        raise ValueError
    audit = boundary["audit"]
    if not isinstance(audit, list) or len(audit) > MAX_AUDIT_CHAINS:
        raise ValueError
    for item in audit:
        if (
            not isinstance(item, dict)
            or set(item) != {"entries", "head"}
            or type(item["entries"]) is not int
            or not _valid_digest(item["head"])
        ):
            raise ValueError


def _validate_object(item: Any, kind: str, point_id: str) -> str:
    if not isinstance(item, dict) or set(item) != {
        "category",
        "chunks",
        "digest",
        "object",
        "path",
        "size",
        "stored_in",
    }:
        raise ValueError
    category = _category_for_path(item["path"])
    if category is None or category.value != item["category"]:
        raise ValueError
    if type(item["size"]) is not int or item["size"] < 0:
        raise ValueError
    if type(item["chunks"]) is not int or item["chunks"] < 1:
        raise ValueError
    if not _valid_digest(item["digest"]):
        raise ValueError
    if not isinstance(item["object"], str) or not _OBJECT_NAME_RE.fullmatch(
        item["object"]
    ):
        raise ValueError
    if not _valid_point_id(item["stored_in"]):
        raise ValueError
    if kind == RecoveryKind.FULL.value and item["stored_in"] != point_id:
        raise ValueError
    return str(item["path"])


def _path_category(path: str) -> str:
    category = _category_for_path(path)
    if category is None:
        raise _Stop("staging_authentication_failed")
    return category.value


def _category_for_path(path: Any) -> ComponentCategory | None:
    if not isinstance(path, str):
        return None
    fixed = {
        JOURNEY_DATABASE_PATH: ComponentCategory.JOURNEY_METADATA,
        JOURNEY_MARKER_PATH: ComponentCategory.JOURNEY_ENCRYPTION,
        JOB_STORE_PATH: ComponentCategory.JOB_METADATA,
        REGISTRY_STATE_FILE: ComponentCategory.MODEL_REGISTRY,
    }.get(path)
    if fixed is not None:
        return fixed
    parts = path.split("/")
    if (
        len(parts) == 6
        and "/".join(parts[:4]) == JOURNEY_BLOB_ROOT
        and _BLOB_PREFIX_RE.fullmatch(parts[4]) is not None
        and _BLOB_NAME_RE.fullmatch(parts[5]) is not None
        and parts[5].startswith(parts[4])
    ):
        return ComponentCategory.JOURNEY_ARTIFACTS
    if (
        len(parts) == 2
        and parts[0] == AUDIT_CHAIN_DIRECTORY
        and _AUDIT_NAME_RE.fullmatch(parts[1]) is not None
    ):
        return ComponentCategory.AUDIT_CHAIN
    return None


def _open_chain(
    point: Path, ancestors: Sequence[Path], cipher: EnvelopeCipher
) -> list[_Point]:
    head = _load_point(point, cipher)
    index: dict[str, Path] = {}
    for root in ancestors:
        descriptor = _read_descriptor(root)
        point_id = descriptor["recovery_point_id"]
        if point_id in index or point_id == head.recovery_point_id:
            raise _Stop("chain_invalid")
        index[point_id] = root
    chain = [head]
    current = head
    while current.kind == RecoveryKind.INCREMENTAL.value:
        if len(chain) >= MAX_CHAIN_LENGTH:
            raise _Stop("chain_too_long")
        link = current.manifest["chain"]
        root = index.get(link["parent_recovery_point_id"])
        if root is None:
            raise _Stop("chain_incomplete")
        parent = _load_point(root, cipher)
        parent_link = parent.manifest["chain"]
        if (
            parent.recovery_point_id != link["parent_recovery_point_id"]
            or parent.manifest_digest != link["parent_manifest_digest"]
            or parent.sequence != current.sequence - 1
            or parent_link["base_recovery_point_id"] != link["base_recovery_point_id"]
        ):
            raise _Stop("chain_invalid")
        _check_monotonic(parent.manifest, current.manifest)
        chain.append(parent)
        current = parent
    return chain


def _check_monotonic(parent: Mapping[str, Any], child: Mapping[str, Any]) -> None:
    if _parse_timestamp(parent["created_at"]) > _parse_timestamp(child["created_at"]):
        raise _Stop("chain_sequence_regressed")
    old = parent["boundary"]
    new = child["boundary"]
    old_journey = old["journey"] or {}
    new_journey = new["journey"] or {}
    if (
        old_journey.get("revision") is not None
        and new_journey.get("revision") is not None
        and new_journey["revision"] < old_journey["revision"]
    ):
        raise _Stop("chain_sequence_regressed")
    for before, after in zip(old["audit"], new["audit"]):
        if after["entries"] < before["entries"]:
            raise _Stop("chain_sequence_regressed")


def _check_freshness(
    point: _Point, now: datetime, max_age: timedelta | None = None
) -> None:
    manifest = point.manifest
    expires_at = manifest["retention"]["expires_at"]
    if expires_at is not None and _aware(now) >= _parse_timestamp(expires_at):
        raise _Stop("recovery_point_expired")
    if (
        max_age is not None
        and _aware(now) - _parse_timestamp(manifest["created_at"]) > max_age
    ):
        raise _Stop("recovery_point_stale")


def _check_chain_freshness(
    chain: Sequence[_Point], now: datetime, max_age: timedelta | None
) -> None:
    if max_age is not None and (
        not isinstance(max_age, timedelta) or max_age <= timedelta(0)
    ):
        raise ValueError("max_age must be a positive timedelta")
    _check_freshness(chain[0], now, max_age)
    for point in chain[1:]:
        _check_freshness(point, now)


def _check_declared_compatibility(manifest: Mapping[str, Any]) -> None:
    boundary = manifest["boundary"]
    journey = boundary["journey"]
    if journey is not None:
        if not 1 <= journey["schema_version"] <= LATEST_MIGRATION_VERSION:
            raise _Stop("schema_incompatible")
        if journey["marker_schema_version"] not in {
            None,
            JOURNEY_MARKER_SCHEMA_VERSION,
        }:
            raise _Stop("schema_incompatible")
    registry = boundary["registry"]
    if (
        registry is not None
        and registry["schema_version"] != REGISTRY_STATE_SCHEMA_VERSION
    ):
        raise _Stop("schema_incompatible")


def _materialize(
    chain: Sequence[_Point],
    cipher: EnvelopeCipher,
    target: Path | None,
    *,
    only: str | None = None,
) -> tuple[dict[str, bytes], frozenset[str]]:
    """Decrypt, hash, and optionally write every object of the chain head."""

    points = {point.recovery_point_id: point for point in chain}
    documents: dict[str, bytes] = {}
    blob_hashes: set[str] = set()
    for entry in chain[0].manifest["objects"]:
        if only is not None and entry["stored_in"] != only:
            continue
        holder = points.get(entry["stored_in"])
        if holder is None:
            raise _Stop("chain_incomplete")
        keep = entry["category"] != ComponentCategory.JOURNEY_ARTIFACTS.value
        buffer = bytearray() if keep and only is None else None
        digest = hashlib.sha256()
        size = 0
        descriptor = None
        if target is not None:
            destination = target.joinpath(*entry["path"].split("/"))
            _make_parents(target, destination)
            descriptor = _create_file(destination, failure="restore_write_failed")
        try:
            for chunk in _read_stream(
                holder.root / _OBJECTS_DIRECTORY / f"{entry['object']}{_OBJECT_SUFFIX}",
                point_id=holder.recovery_point_id,
                name=entry["object"],
                chunks=entry["chunks"],
                cipher=cipher,
            ):
                size += len(chunk)
                if size > entry["size"]:
                    raise _Stop("object_size_mismatch")
                digest.update(chunk)
                if buffer is not None:
                    buffer.extend(chunk)
                if descriptor is not None:
                    _write_all(descriptor, chunk)
            if descriptor is not None:
                os.fsync(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if size != entry["size"]:
            raise _Stop("object_size_mismatch")
        if f"sha256:{digest.hexdigest()}" != entry["digest"]:
            raise _Stop("object_digest_mismatch")
        if buffer is not None:
            documents[entry["path"]] = bytes(buffer)
        if not keep:
            blob_hashes.add("sha256:" + entry["path"].rsplit("/", 1)[1])
    return documents, frozenset(blob_hashes)


def _verify_semantics(
    head: _Point, documents: Mapping[str, bytes], blob_hashes: frozenset[str]
) -> None:
    boundary, facts = _derive_boundary(
        documents, blob_hashes, failure="component_invalid"
    )
    if facts is not None:
        for version, name, checksum in facts.migrations:
            if _KNOWN_MIGRATIONS.get(version) != (name, checksum):
                raise _Stop("schema_incompatible")
    if boundary != head.manifest["boundary"]:
        raise _Stop("snapshot_boundary_mismatch")


# ---------------------------------------------------------------------------
# Staging records and activation
# ---------------------------------------------------------------------------


def _staging_descriptor(point_id: str) -> dict[str, Any]:
    return {
        "format": STAGING_FORMAT,
        "format_version": RECOVERY_FORMAT_VERSION,
        "recovery_point_id": point_id,
    }


def _read_staging_descriptor(root: Path) -> dict[str, Any]:
    data = _read_bounded(
        root / RECOVERY_METADATA_DIRECTORY / _STAGING_DESCRIPTOR,
        _MAX_DESCRIPTOR_BYTES,
        missing="restore_incomplete",
        invalid="restore_incomplete",
        unavailable="staged_state_unavailable",
    )
    document = _parse_json(data, failure="restore_incomplete")
    if (
        not isinstance(document, dict)
        or document.keys() != _staging_descriptor("rp_").keys()
        or document["format"] != STAGING_FORMAT
        or document["format_version"] != RECOVERY_FORMAT_VERSION
        or not _valid_point_id(document["recovery_point_id"])
    ):
        raise _Stop("restore_incomplete")
    return document


def _open_staging(root: Path, cipher: EnvelopeCipher) -> dict[str, Any]:
    descriptor = _read_staging_descriptor(root)
    point_id = descriptor["recovery_point_id"]
    sealed = _read_bounded(
        root / RECOVERY_METADATA_DIRECTORY / _STAGING_SEAL,
        _MAX_SEALED_DOCUMENT_BYTES,
        missing="restore_incomplete",
        invalid="restore_incomplete",
        unavailable="staged_state_unavailable",
    )
    try:
        plaintext = cipher.open(sealed, context=_staging_context(point_id))
    except (EnvelopeAuthenticationError, EnvelopeFormatError, EnvelopeVersionError):
        raise _Stop("staging_authentication_failed") from None
    record = _parse_json(plaintext, failure="staging_authentication_failed")
    try:
        if (
            not isinstance(record, dict)
            or record.get("schema") != STAGING_SCHEMA
            or record.get("recovery_point_id") != point_id
            or not _valid_digest(record.get("manifest_digest"))
            or not _valid_digest(record.get("content_digest"))
            or record.get("kind")
            not in {RecoveryKind.FULL.value, RecoveryKind.INCREMENTAL.value}
            or type(record.get("chain_length")) is not int
        ):
            raise ValueError
        _validate_boundary(record.get("boundary"))
        objects = record.get("objects")
        if not isinstance(objects, list):
            raise ValueError
        for item in objects:
            if (
                not isinstance(item, dict)
                or set(item) != {"digest", "path", "size"}
                or _category_for_path(item["path"]) is None
                or not _valid_digest(item["digest"])
                or type(item["size"]) is not int
            ):
                raise ValueError
    except (ValueError, TypeError, KeyError):
        raise _Stop("staging_authentication_failed") from None
    return record


def _verify_staged_tree(root: Path, record: Mapping[str, Any]) -> None:
    expected = {item["path"]: item for item in record["objects"]}
    found: set[str] = set()
    try:
        for directory, subdirectories, names in os.walk(root, followlinks=False):
            relative = Path(directory).relative_to(root)
            if relative.parts[:1] == (RECOVERY_METADATA_DIRECTORY,):
                subdirectories[:] = []
                continue
            for name in list(subdirectories):
                if (Path(directory) / name).is_symlink():
                    raise _Stop("staged_state_modified")
            for name in names:
                candidate = Path(directory) / name
                logical = "/".join((*relative.parts, name))
                entry = expected.get(logical)
                if entry is None or candidate.is_symlink():
                    raise _Stop("staged_state_modified")
                size, digest = _hash_file(candidate, DEFAULT_CHUNK_BYTES)
                if size != entry["size"] or digest != entry["digest"]:
                    raise _Stop("staged_state_modified")
                found.add(logical)
    except _Stop as stop:
        if stop.code != "staged_state_modified":
            raise _Stop("staged_state_modified") from None
        raise
    except OSError:
        raise _Stop("staged_state_unavailable") from None
    if found != set(expected):
        raise _Stop("staged_state_modified")


def _run_check(check: ActivationCheck, root: Path) -> None:
    name = "custom"
    code = "check_failed"
    try:
        result = check(root)
    except EncryptionError as exc:
        result = ActivationCheckResult(name, False, exc.code)
    except Exception:
        result = ActivationCheckResult(name, False, "check_raised")
    if isinstance(result, ActivationCheckResult):
        if isinstance(result.name, str) and _CODE_RE.fullmatch(result.name):
            name = result.name
        if isinstance(result.code, str) and _CODE_RE.fullmatch(result.code):
            code = result.code
        passed = result.passed is True
    else:
        passed = result is True
    if not passed:
        raise _Stop(
            "activation_check_failed",
            details={"check": name, "check_reason": code},
        )


def _swap_symlink(live: Path, staged_root: Path) -> None:
    temporary = live.parent / f".{live.name}.activate-{secrets.token_hex(8)}"
    try:
        os.symlink(staged_root, temporary, target_is_directory=True)
    except OSError:
        raise _Stop("activation_failed", RecoveryOutcome.FAILED) from None
    try:
        os.replace(temporary, live)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise _Stop("activation_failed", RecoveryOutcome.FAILED) from None
    _fsync_directory(live.parent)


def _write_activation_record(root: Path, context: RecoveryEpochContext) -> None:
    record = {
        "activated_at": context.activated_at,
        "content_digest": context.content_digest,
        "manifest_digest": context.manifest_digest,
        "recovery_point_id": context.recovery_point_id,
        "schema": "openmed.service.recovery-activation.v1",
    }
    path = root / RECOVERY_METADATA_DIRECTORY / _ACTIVATION_RECORD
    try:
        _replace_file(path, _canonical_json(record))
    except (OSError, _Stop):
        # The swap already happened; the record is informational only.
        return


def _staged_schema_version(database: Path) -> int:
    if not database.is_file():
        return LATEST_MIGRATION_VERSION
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0
    )
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "schema_migrations" not in tables:
            return 0
        value = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()[0]
        return int(value)
    finally:
        connection.close()


def _check_previous_schema(database: Path) -> ActivationCheckResult:
    """Verify an older staged store in memory without migrating it."""

    try:
        facts = _inspect_journey_image(
            _read_regular(database, limit=_MAX_DOCUMENT_BYTES),
            failure="component_invalid",
        )
    except _Stop:
        return ActivationCheckResult("journey_store", False, "integrity_failed")
    except OSError:
        return ActivationCheckResult("journey_store", False, "journey_open_failed")
    for version, name, checksum in facts.migrations:
        if _KNOWN_MIGRATIONS.get(version) != (name, checksum):
            return ActivationCheckResult("journey_store", False, "schema_incompatible")
    if facts.encryption_state not in {None, "plaintext"}:
        return ActivationCheckResult("journey_store", False, "schema_incompatible")
    return ActivationCheckResult("journey_store", True)


def _staged_content_hashes(database: Path) -> tuple[str, ...]:
    if not database.is_file():
        return ()
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0
    )
    try:
        return tuple(
            sorted(
                str(row[0])
                for row in connection.execute("SELECT content_hash FROM artifacts")
            )
        )
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Receipts and small helpers
# ---------------------------------------------------------------------------


def _receipt(
    operation: str,
    outcome: RecoveryOutcome,
    started: datetime,
    finished: datetime,
    *,
    reason: str | None = None,
    point: _Point | None = None,
    chain_length: int = 0,
    counts: Mapping[str, int] | None = None,
    details: Mapping[str, str] | None = None,
) -> RecoveryReceipt:
    if reason is not None and reason not in RECOVERY_REASON_CODES:
        reason = "recovery_failed"
    safe_details = {
        key: value
        for key, value in (details or {}).items()
        if key in {"check", "check_reason"}
        and isinstance(value, str)
        and _CODE_RE.fullmatch(value)
    }
    if point is None:
        return RecoveryReceipt(
            operation=operation,
            outcome=outcome,
            reason=reason,
            counts=dict(counts or {}),
            details=safe_details,
            started_at=_timestamp(started),
            finished_at=_timestamp(finished),
        )
    manifest = point.manifest
    return RecoveryReceipt(
        operation=operation,
        outcome=outcome,
        reason=reason,
        recovery_point_id=point.recovery_point_id,
        kind=point.kind,
        sequence=point.sequence,
        chain_length=chain_length,
        content_digest=manifest["content_digest"],
        manifest_digest=point.manifest_digest,
        components=tuple(_components(manifest["objects"], manifest["boundary"])),
        boundary=manifest["boundary"],
        counts=dict(counts or {}),
        details=safe_details,
        started_at=_timestamp(started),
        finished_at=_timestamp(finished),
    )


def _active_key(cipher: EnvelopeCipher) -> DataKey:
    try:
        return cipher.active_key()
    except EncryptionError as exc:
        raise _Stop(exc.code) from None


def _clock(clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
    if clock is None:
        return lambda: datetime.now(timezone.utc)
    if not callable(clock):
        raise TypeError("clock must be callable")
    return clock


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _timestamp(moment: datetime) -> str:
    return _aware(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        return False
    try:
        _parse_timestamp(value)
    except ValueError:
        return False
    return True


def _valid_point_id(value: Any) -> bool:
    return isinstance(value, str) and _POINT_ID_RE.fullmatch(value) is not None


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _read_bounded(
    path: Path,
    limit: int,
    *,
    missing: str,
    invalid: str,
    unavailable: str,
) -> bytes:
    if path.is_symlink():
        raise _Stop(invalid)
    try:
        with open(path, "rb") as handle:
            data = handle.read(limit + 1)
    except FileNotFoundError:
        raise _Stop(missing) from None
    except OSError:
        raise _Stop(unavailable) from None
    if len(data) > limit:
        raise _Stop(invalid)
    return data


def _is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _overlaps(target: Path, live: Path) -> bool:
    target_resolved = target.parent.resolve() / target.name
    live_resolved = live.resolve()
    return _is_within(target_resolved, live_resolved) or _is_within(
        live_resolved, target_resolved
    )


def _make_directory(path: Path, *, failure: str) -> None:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        code = "target_exists" if failure == "restore_write_failed" else failure
        raise _Stop(code) from None
    except OSError:
        raise _Stop(failure, RecoveryOutcome.FAILED) from None


def _make_parents(root: Path, destination: Path) -> None:
    current = root
    for part in destination.relative_to(root).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise _Stop("restore_write_failed", RecoveryOutcome.FAILED)
        if not current.is_dir():
            _make_directory(current, failure="restore_write_failed")


def _create_file(path: Path, *, failure: str) -> int:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        return os.open(path, flags, 0o600)
    except OSError:
        raise _Stop(failure, RecoveryOutcome.FAILED) from None


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def _write_new(path: Path, data: bytes, *, failure: str) -> None:
    descriptor = _create_file(path, failure=failure)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    except OSError:
        raise _Stop(failure, RecoveryOutcome.FAILED) from None
    finally:
        os.close(descriptor)


def _replace_file(path: Path, data: bytes) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}"
    descriptor = _create_file(temporary, failure="activation_failed")
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for directory, _subdirectories, _names in os.walk(root, topdown=False):
        _fsync_directory(Path(directory))


def _remove_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        return
