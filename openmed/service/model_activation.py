"""Transactional model download, verification, activation, and rollback.

:class:`ModelActivationManager` composes the existing model primitives into one
recoverable transaction per service model slot::

    acquire -> stage -> verify -> load -> warm -> commit -> [health] -> retire
       \\______________________ rollback at any step ______________________/

* ``acquire`` is the only phase that may use the network. The default
  :class:`HubArtifactSource` delegates to :func:`openmed.core.hf_hub.prefetch_model`
  (per-file Hub digest checks, retries, bandwidth caps, offline mode) inside a
  private workspace; :class:`LocalBundleSource` copies an air-gapped bundle.
* ``stage`` materializes the artifact tree under ``staging/<transaction>``,
  outside every committed generation, rejecting links that escape the
  workspace and bounding file count and size.
* ``verify`` runs :func:`openmed.core.model_integrity.verify_artifact_manifest`
  on the staged tree, rejects unlisted files and links, applies caller digest
  pins and configured policy validators, and proves the retained last-green
  generation stays a compatible rollback target through
  :func:`openmed.core.registry_compatibility.build_rollback_compatibility_report`.
  Any failure quarantines the staged content, which is never loaded.
* ``load`` and ``warm`` prove the immutable candidate generation can serve.
* ``commit`` publishes the slot pointer and generation number with one atomic,
  fsynced state write, then swaps the in-process generation table under the
  same lock. Readers observe either the previous complete generation or the new
  complete generation.
* The previous generation stays ``last_green`` (resident for instant rollback)
  until the declared health window passes. A failed or unproven window rolls
  back automatically.
* ``retire`` promotes the new generation to last green and releases the old
  one once its in-flight :class:`GenerationLease` holders finish.

Every phase boundary is journaled with an atomic write, so
:meth:`ModelActivationManager.start` resolves interrupted transactions
deterministically by quarantining, rolling back, or resuming them, and
reconciles orphaned staging content. Journals, pointer state, lifecycle events,
hash-chained receipts, and results carry transaction ids, digests, generation
numbers, phases, timestamps, counts, and controlled reason codes only; never
model ids, revisions, local paths, URLs, credentials, or artifact content.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from openmed.core.audit import stable_hash
from openmed.core.hf_hub import DEFAULT_RETRIES, DownloadIntegrityError
from openmed.core.model_integrity import (
    ARTIFACT_MANIFEST_FILENAME,
    ModelIntegrityError,
    verify_artifact_manifest,
    write_local_artifact_manifest,
)
from openmed.core.offline import OfflineModeError
from openmed.core.registry_compatibility import (
    RegistryCheckpoint,
    build_rollback_compatibility_report,
)

from .warm_pool import (
    DEFAULT_WARM_AGGREGATION_STRATEGY,
    DEFAULT_WARM_PIPELINE_TASK,
    DEFAULT_WARM_USE_FAST_TOKENIZER,
)

__all__ = [
    "ACTIVATION_BOUNDARIES",
    "ActivationEvent",
    "ActivationOutcome",
    "ActivationPhase",
    "ActivationPolicy",
    "ActivationReason",
    "ActivationResult",
    "ArtifactSource",
    "GenerationContext",
    "GenerationLease",
    "GenerationLoader",
    "HubArtifactSource",
    "JOURNAL_SCHEMA_VERSION",
    "LocalBundleSource",
    "ModelActivationError",
    "ModelActivationManager",
    "ModelReference",
    "RECEIPT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "StagedArtifact",
    "TransactionStatus",
    "WarmPoolGenerationLoader",
    "verify_receipt_chain",
]

logger = logging.getLogger(__name__)

JOURNAL_SCHEMA_VERSION = "openmed.model_activation.journal.v1"
STATE_SCHEMA_VERSION = "openmed.model_activation.state.v1"
RECEIPT_SCHEMA_VERSION = "openmed.model_activation.receipt.v1"
RESULT_SCHEMA_VERSION = "openmed.model_activation.result.v1"
GENERATION_SCHEMA_VERSION = "openmed.model_activation.generation.v1"
QUARANTINE_SCHEMA_VERSION = "openmed.model_activation.quarantine.v1"

ACTIVATION_BOUNDARIES = (
    "acquire.before",
    "acquire.after",
    "stage.before",
    "stage.after",
    "verify.before",
    "verify.after",
    "generation.allocated",
    "load.before",
    "load.after",
    "warm.before",
    "warm.after",
    "commit.before",
    "commit.published",
    "commit.after",
    "retire.before",
    "retire.published",
    "retire.after",
    "rollback.before",
    "rollback.published",
    "rollback.after",
    "state.write",
    "journal.write",
)
"""Named transaction boundaries passed to the optional ``fault_hook``.

The journal guarantees deterministic recovery at each of these points; the
hook exists so tests can inject crashes or disk faults without patching.
"""

DEFAULT_MAX_ARTIFACT_FILES = 10_000
DEFAULT_MAX_QUARANTINE_ENTRIES = 8
DEFAULT_MAX_JOURNAL_HISTORY = 64

_MAX_JOURNAL_BYTES = 1 << 20
_MAX_STATE_BYTES = 4 << 20
_MAX_MANIFEST_BYTES = 8 << 20
_MAX_GENERATION_METADATA_BYTES = 1 << 20

_SLOT_PART = r"[a-z0-9][a-z0-9_.-]{0,63}"
_SLOT_RE = re.compile(rf"{_SLOT_PART}(?:::{_SLOT_PART}){{0,2}}")
_TRANSACTION_RE = re.compile(r"txn-[0-9a-f]{32}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_GENERATION_DIR_RE = re.compile(r"[0-9]{8}")
_TEMP_SUFFIX = ".tmp"

_STATE_FILENAME = "state.json"
_RECEIPTS_FILENAME = "receipts.jsonl"
_OWNER_LOCK_FILENAME = ".owner.lock"
_GENERATION_FILENAME = "generation.json"
_QUARANTINE_FILENAME = "quarantine.json"
_ARTIFACT_DIRNAME = "artifact"
_WORKSPACE_DIRNAME = "workspace"
_PAYLOAD_DIRNAME = "payload"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ActivationPhase(str, Enum):
    """Journaled phases of one activation transaction."""

    ACQUIRE = "acquire"
    STAGE = "stage"
    VERIFY = "verify"
    LOAD = "load"
    WARM = "warm"
    COMMIT = "commit"
    RETIRE = "retire"
    ROLLBACK = "rollback"


class TransactionStatus(str, Enum):
    """Durable status of a journaled transaction."""

    IN_PROGRESS = "in_progress"
    HEALTH_WINDOW = "health_window"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    QUARANTINED = "quarantined"


_TERMINAL_STATUSES = frozenset(
    {
        TransactionStatus.COMMITTED.value,
        TransactionStatus.ROLLED_BACK.value,
        TransactionStatus.QUARANTINED.value,
    }
)


class ActivationOutcome(str, Enum):
    """Typed outcome returned by every activation operation."""

    ACTIVATED = "activated"
    PENDING_HEALTH = "pending_health"
    ALREADY_ACTIVE = "already_active"
    BUSY = "busy"
    CONFLICT = "conflict"
    QUARANTINED = "quarantined"
    ROLLED_BACK = "rolled_back"
    REFUSED = "refused"
    RESUMED = "resumed"


_OK_OUTCOMES = frozenset(
    {
        ActivationOutcome.ACTIVATED,
        ActivationOutcome.PENDING_HEALTH,
        ActivationOutcome.ALREADY_ACTIVE,
        ActivationOutcome.RESUMED,
    }
)


class ActivationReason(str, Enum):
    """Controlled, value-free reason codes for activation outcomes."""

    SLOT_BUSY = "slot_busy"
    SLOT_CONFLICT = "slot_conflict"
    TRANSACTION_CONFLICT = "transaction_conflict"
    INVALID_REQUEST = "invalid_request"
    ACQUIRE_FAILED = "acquire_failed"
    OFFLINE_UNAVAILABLE = "offline_unavailable"
    DISK_EXHAUSTED = "disk_exhausted"
    SOURCE_OUTSIDE_WORKSPACE = "source_outside_workspace"
    STAGE_FAILED = "stage_failed"
    ARTIFACT_LINK = "artifact_link"
    ARTIFACT_UNSUPPORTED = "artifact_unsupported"
    ARTIFACT_UNLISTED = "artifact_unlisted"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_TOO_LARGE = "artifact_too_large"
    DIGEST_MISMATCH = "digest_mismatch"
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_INVALID = "manifest_invalid"
    MANIFEST_MISMATCH = "manifest_mismatch"
    PIN_MISMATCH = "pin_mismatch"
    VERIFY_FAILED = "verify_failed"
    POLICY_REJECTED = "policy_rejected"
    COMPATIBILITY_METADATA_INVALID = "compatibility_metadata_invalid"
    ROLLBACK_INCOMPATIBLE = "rollback_incompatible"
    LOAD_FAILED = "load_failed"
    WARM_FAILED = "warm_failed"
    COMMIT_FAILED = "commit_failed"
    HEALTH_WINDOW_FAILED = "health_window_failed"
    HEALTH_WINDOW_UNPROVEN = "health_window_unproven"
    HEALTH_WINDOW_INTERRUPTED = "health_window_interrupted"
    ROLLBACK_REQUESTED = "rollback_requested"
    INTERRUPTED = "interrupted"
    NO_HEALTH_WINDOW = "no_health_window"
    ROLLBACK_UNAVAILABLE = "rollback_unavailable"
    ROLLBACK_TARGET_UNVERIFIED = "rollback_target_unverified"
    ROLLBACK_TARGET_UNLOADABLE = "rollback_target_unloadable"
    ROLLBACK_FAILED = "rollback_failed"
    ORPHANED_STAGING = "orphaned_staging"
    JOURNAL_UNREADABLE = "journal_unreadable"
    STATE_UNREADABLE = "state_unreadable"
    ROOT_BUSY = "activation_root_busy"
    NOT_STARTED = "manager_not_started"
    NO_ACTIVE_GENERATION = "no_active_generation"
    ACTIVE_UNVERIFIED = "active_unverified"
    ACTIVE_UNLOADABLE = "active_unloadable"


_INTEGRITY_REASONS = {
    "artifact_missing": ActivationReason.ARTIFACT_MISSING,
    "artifact_digest_mismatch": ActivationReason.DIGEST_MISMATCH,
    "manifest_invalid": ActivationReason.MANIFEST_INVALID,
    "manifest_model_mismatch": ActivationReason.MANIFEST_MISMATCH,
    "manifest_revision_mismatch": ActivationReason.MANIFEST_MISMATCH,
}


class ModelActivationError(RuntimeError):
    """Raised when an activation operation cannot run at all.

    The message is the controlled reason code only; it never names models,
    revisions, paths, URLs, or credentials.

    Attributes:
        reason: The :class:`ActivationReason` that stopped the operation.
    """

    def __init__(self, reason: ActivationReason) -> None:
        """Store the reason and use its code as the message."""

        self.reason = reason
        super().__init__(reason.value)


class _PhaseFailure(Exception):
    """Internal controlled failure raised inside one transaction phase."""

    def __init__(
        self,
        reason: ActivationReason,
        *,
        compatibility: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.compatibility = dict(compatibility) if compatibility else None
        super().__init__(reason.value)


@dataclass(frozen=True)
class ModelReference:
    """The model a slot should serve.

    Only :attr:`digest` (and the optional pins) ever leave the process; the raw
    model id and revision are used solely by the artifact source and the
    integrity manifest check.

    Args:
        model_id: Registry key or Hub repository id; also the ``model_id`` the
            staged integrity manifest must declare.
        revision: Optional revision pinned at acquisition.
        expected_artifact_digest: Optional ``sha256:`` artifact-set digest
            (as computed by :func:`verify_artifact_manifest`) the staged tree
            must reproduce exactly.
        expected_reproducibility_hash: Optional ``sha256:`` revision digest
            the staged integrity manifest must declare.
    """

    model_id: str = field(repr=False)
    revision: str | None = field(default=None, repr=False)
    expected_artifact_digest: str | None = None
    expected_reproducibility_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate the reference shape without echoing any value."""

        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ModelActivationError(ActivationReason.INVALID_REQUEST)
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ModelActivationError(ActivationReason.INVALID_REQUEST)
        for pin in (self.expected_artifact_digest, self.expected_reproducibility_hash):
            if pin is not None and (
                not isinstance(pin, str) or not _DIGEST_RE.fullmatch(pin)
            ):
                raise ModelActivationError(ActivationReason.INVALID_REQUEST)

    @property
    def digest(self) -> str:
        """Return the stable, value-free digest of the model identity."""

        return stable_hash({"model_id": self.model_id, "revision": self.revision})


@dataclass(frozen=True)
class GenerationContext:
    """What a :class:`GenerationLoader` needs to load one generation.

    Attributes:
        slot: Service model slot.
        generation: Monotonic, service-visible generation number.
        transaction_id: Transaction that produced the generation.
        model_ref_digest: Digest of the model reference.
        artifact_digest: Verified artifact-set digest.
        artifact_dir: Local directory of the verified, immutable artifact
            tree. In-process only; never serialized.
    """

    slot: str
    generation: int
    transaction_id: str
    model_ref_digest: str
    artifact_digest: str
    artifact_dir: Path = field(repr=False)


@dataclass(frozen=True)
class StagedArtifact:
    """Value-free view of a staged artifact handed to policy validators.

    Attributes:
        slot: Service model slot being activated.
        model_ref_digest: Digest of the requested model reference.
        artifact_digest: Verified artifact-set digest.
        files: Sorted relative artifact paths, excluding the manifest.
        total_bytes: Total size of the staged tree in bytes.
        artifact_dir: Staged directory, for validators that inspect content.
            In-process only; never serialized.
    """

    slot: str
    model_ref_digest: str
    artifact_digest: str
    files: tuple[str, ...]
    total_bytes: int
    artifact_dir: Path = field(repr=False)


class ArtifactSource(Protocol):
    """Acquisition step: the only phase allowed to use the network."""

    def acquire(self, reference: ModelReference, workspace: Path) -> Path:
        """Place the artifact tree inside ``workspace`` and return its root.

        Args:
            reference: Model to acquire.
            workspace: Empty private directory under the transaction's staging
                area. The returned root must resolve inside it.

        Returns:
            Directory holding the artifact tree and its integrity manifest.
        """
        ...


class GenerationLoader(Protocol):
    """Loads, warms, and releases one generation's serving handle."""

    def load(self, context: GenerationContext) -> Any:
        """Load the generation and return an opaque serving handle."""
        ...

    def warm(self, handle: Any, context: GenerationContext) -> None:
        """Exercise the handle once (for example a synthetic inference)."""
        ...

    def unload(self, handle: Any, context: GenerationContext) -> None:
        """Release the handle after its last in-flight lease finished."""
        ...


@dataclass(frozen=True)
class HubArtifactSource:
    """Acquire a model through :func:`openmed.core.hf_hub.prefetch_model`.

    Downloads land in the transaction workspace, never in the shared Hugging
    Face cache. ``prefetch_model`` checks every file against the Hub's size and
    digest metadata before returning; when the snapshot carries no OpenMed
    integrity manifest one is measured from those verified bytes so the staged
    tree verifies offline. Offline mode is honoured by ``prefetch_model``.

    Args:
        allow_patterns: Optional glob patterns limiting downloaded files.
        retries: Retries for transient network failures.
        max_bandwidth: Optional aggregate cap in bytes per second.
    """

    allow_patterns: tuple[str, ...] | None = None
    retries: int = DEFAULT_RETRIES
    max_bandwidth: int | None = None

    def acquire(self, reference: ModelReference, workspace: Path) -> Path:
        """Download the reference into ``workspace`` and return the snapshot."""

        from openmed.core.hf_hub import prefetch_model

        snapshot = Path(
            prefetch_model(
                reference.model_id,
                revision=reference.revision,
                allow_patterns=self.allow_patterns,
                cache_dir=str(workspace / "hub"),
                retries=self.retries,
                max_bandwidth=self.max_bandwidth,
            )
        )
        if not (snapshot / ARTIFACT_MANIFEST_FILENAME).is_file():
            write_local_artifact_manifest(
                snapshot,
                model_id=reference.model_id,
                reproducibility_hash=(
                    reference.expected_reproducibility_hash or reference.digest
                ),
            )
        return snapshot


@dataclass(frozen=True)
class LocalBundleSource:
    """Acquire an already-downloaded bundle directory without network access.

    The bundle is copied into the transaction workspace, preserving links so
    staging can reject any that escape it. The source directory is never
    modified, moved, or loaded directly.

    Args:
        directory: Local bundle directory holding an integrity manifest.
    """

    directory: Path = field(repr=False)

    def acquire(self, reference: ModelReference, workspace: Path) -> Path:
        """Copy the bundle into ``workspace`` and return the copy."""

        del reference
        destination = workspace / "bundle"
        shutil.copytree(Path(self.directory), destination, symlinks=True)
        return destination


class WarmPoolGenerationLoader:
    """Load generations through a service :class:`~.warm_pool.WarmPool`.

    Each generation is a distinct, immutable local directory, so the pool keys
    its handles per generation: in-flight requests keep using the pipeline of
    the generation they leased while new leases get the new one, and the old
    pool entry is unloaded only after its last lease is released.

    Args:
        pool: The service warm pool (or any object with its
            ``create_pipeline`` and ``unload_model`` methods).
        task: Pipeline task used for the generation handle.
        aggregation_strategy: Pipeline aggregation strategy.
        use_fast_tokenizer: Whether the pipeline uses a fast tokenizer.
        warm_probe: Optional callable exercising the handle once with
            synthetic input during the ``warm`` phase.
    """

    def __init__(
        self,
        pool: Any,
        *,
        task: str = DEFAULT_WARM_PIPELINE_TASK,
        aggregation_strategy: str | None = DEFAULT_WARM_AGGREGATION_STRATEGY,
        use_fast_tokenizer: bool = DEFAULT_WARM_USE_FAST_TOKENIZER,
        warm_probe: Callable[[Any], Any] | None = None,
    ) -> None:
        """Bind the loader to a warm pool and pipeline shape."""

        self._pool = pool
        self._task = task
        self._aggregation_strategy = aggregation_strategy
        self._use_fast_tokenizer = use_fast_tokenizer
        self._warm_probe = warm_probe

    def load(self, context: GenerationContext) -> Any:
        """Create the generation's pipeline in the warm pool."""

        return self._pool.create_pipeline(
            str(context.artifact_dir),
            task=self._task,
            aggregation_strategy=self._aggregation_strategy,
            use_fast_tokenizer=self._use_fast_tokenizer,
        )

    def warm(self, handle: Any, context: GenerationContext) -> None:
        """Run the optional synthetic warm-up probe."""

        del context
        if self._warm_probe is not None:
            self._warm_probe(handle)

    def unload(self, handle: Any, context: GenerationContext) -> None:
        """Unload the generation's pool entry after it has drained."""

        del handle
        self._pool.unload_model(str(context.artifact_dir))


@dataclass(frozen=True)
class ActivationPolicy:
    """Configured activation policy.

    Args:
        health_window_seconds: Seconds a committed generation must stay healthy
            before it replaces the last-green target. ``0`` finalizes at commit.
        min_healthy_reports: Healthy reports required before the window can
            pass; a window that elapses with fewer reports rolls back.
        require_rollback_compatibility: Refuse a candidate unless the retained
            last-green generation is a proven compatible rollback target. When
            disabled an incompatible target is recorded as ``waived``.
        max_artifact_bytes: Optional bound on the staged tree size.
        max_artifact_files: Bound on the staged file count.
        max_quarantine_entries: Quarantined entries kept; older are cleaned.
        max_journal_history: Finished journals kept for idempotent replay.
        reverify_before_rollback: Re-verify the last-green tree before
            rolling back to it.
        reverify_on_start: Re-verify active generations before loading them at
            startup.
        keep_last_green_resident: Keep the previous generation loaded during
            the health window for instant rollback.
        validators: Additional policy checks (for example a deployment security
            profile). Each returns ``True`` to accept a :class:`StagedArtifact`;
            anything else, or an exception, quarantines it.
    """

    health_window_seconds: float = 0.0
    min_healthy_reports: int = 0
    require_rollback_compatibility: bool = True
    max_artifact_bytes: int | None = None
    max_artifact_files: int = DEFAULT_MAX_ARTIFACT_FILES
    max_quarantine_entries: int = DEFAULT_MAX_QUARANTINE_ENTRIES
    max_journal_history: int = DEFAULT_MAX_JOURNAL_HISTORY
    reverify_before_rollback: bool = True
    reverify_on_start: bool = True
    keep_last_green_resident: bool = True
    validators: tuple[Callable[[StagedArtifact], bool], ...] = ()

    def __post_init__(self) -> None:
        """Reject malformed bounds."""

        window = self.health_window_seconds
        if isinstance(window, bool) or not isinstance(window, (int, float)):
            raise ValueError("health_window_seconds must be a number")
        if not math.isfinite(window) or window < 0:
            raise ValueError("health_window_seconds must be a finite number >= 0")
        for name in (
            "min_healthy_reports",
            "max_artifact_files",
            "max_quarantine_entries",
            "max_journal_history",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_artifact_files < 1:
            raise ValueError("max_artifact_files must be at least 1")
        if self.max_artifact_bytes is not None and (
            isinstance(self.max_artifact_bytes, bool)
            or not isinstance(self.max_artifact_bytes, int)
            or self.max_artifact_bytes < 1
        ):
            raise ValueError("max_artifact_bytes must be a positive integer")
        object.__setattr__(self, "validators", tuple(self.validators))


@dataclass(frozen=True)
class ActivationResult:
    """Value-free result of an activation operation.

    Attributes:
        transaction_id: Transaction the result describes. ``busy`` and
            ``conflict`` results name a transaction that was not started.
        slot: Service model slot.
        outcome: Typed outcome.
        phase: Last journaled phase, when a transaction exists.
        model_ref_digest: Digest of the model reference.
        active_generation: Generation serving the slot after the operation.
        last_green_generation: Verified rollback target after the operation.
        candidate_generation: Generation the transaction produced, if any.
        artifact_digest: Verified artifact-set digest, if any.
        reason: Controlled reason code for non-success outcomes.
        health_deadline: ISO-8601 end of an open health window.
        replayed: Whether the result replays an existing transaction.
    """

    transaction_id: str | None
    slot: str
    outcome: ActivationOutcome
    phase: ActivationPhase | None = None
    model_ref_digest: str | None = None
    active_generation: int | None = None
    last_green_generation: int | None = None
    candidate_generation: int | None = None
    artifact_digest: str | None = None
    reason: ActivationReason | None = None
    health_deadline: str | None = None
    replayed: bool = False

    @property
    def ok(self) -> bool:
        """Return whether the slot serves the requested generation."""

        return self.outcome in _OK_OUTCOMES

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, value-free representation."""

        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "slot": self.slot,
            "outcome": self.outcome.value,
            "phase": None if self.phase is None else self.phase.value,
            "model_ref_digest": self.model_ref_digest,
            "active_generation": self.active_generation,
            "last_green_generation": self.last_green_generation,
            "candidate_generation": self.candidate_generation,
            "artifact_digest": self.artifact_digest,
            "reason": None if self.reason is None else self.reason.value,
            "health_deadline": self.health_deadline,
            "replayed": self.replayed,
        }


@dataclass(frozen=True)
class ActivationEvent:
    """Value-free lifecycle event delivered to the optional event sink.

    Attributes:
        transaction_id: Transaction the event belongs to, when any.
        slot: Service model slot.
        phase: Phase or lifecycle step name.
        state: ``started``, ``completed``, ``failed``, or a recovery outcome.
        generation: Generation involved, when known.
        reason: Controlled reason code, when any.
        at: ISO-8601 timestamp.
    """

    transaction_id: str | None
    slot: str
    phase: str
    state: str
    generation: int | None
    reason: str | None
    at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(eq=False)
class _LiveGeneration:
    context: GenerationContext
    handle: Any
    refs: int = 0
    retired: bool = False
    disposed: bool = False


class GenerationLease:
    """Pins one generation's serving handle for an in-flight request.

    A lease keeps its generation loaded even after a newer generation is
    committed or the slot rolls back; the handle is released only after the
    last lease on a retired generation ends. Use as a context manager.
    """

    __slots__ = ("_manager", "_live", "_released")

    def __init__(self, manager: "ModelActivationManager", live: _LiveGeneration):
        """Bind the lease to its manager and generation."""

        self._manager = manager
        self._live = live
        self._released = False

    @property
    def slot(self) -> str:
        """Return the leased slot."""

        return self._live.context.slot

    @property
    def generation(self) -> int:
        """Return the leased generation number."""

        return self._live.context.generation

    @property
    def transaction_id(self) -> str:
        """Return the transaction that produced the leased generation."""

        return self._live.context.transaction_id

    @property
    def artifact_digest(self) -> str:
        """Return the leased generation's verified artifact digest."""

        return self._live.context.artifact_digest

    @property
    def handle(self) -> Any:
        """Return the generation's serving handle."""

        return self._live.handle

    def release(self) -> None:
        """End the lease; idempotent."""

        if self._released:
            return
        self._released = True
        self._manager._release_lease(self._live)

    def __enter__(self) -> "GenerationLease":
        """Return the lease."""

        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Release the lease."""

        self.release()

    def __repr__(self) -> str:
        """Return a value-free representation."""

        return f"GenerationLease(slot={self.slot!r}, generation={self.generation})"


class _OwnerLock:
    """Non-blocking, process-exclusive lock on the activation root."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def acquire(self) -> bool:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self._path, flags, 0o600)
        try:
            if sys.platform == "win32":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(descriptor)
            return False
        self._descriptor = descriptor
        return True

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                with contextlib.suppress(OSError):
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class ModelActivationManager:
    """Recoverable, transactional model activation for service model slots.

    The activation root is owned by one process at a time. Layout::

        <root>/state.json                 active and last-green pointers
        <root>/journal/<txn>.json         one versioned journal per transaction
        <root>/receipts.jsonl             append-only, hash-chained receipts
        <root>/staging/<txn>/             in-flight staging (never served)
        <root>/generations/<slot>/<gen>/  immutable committed generations
        <root>/quarantine/<entry>/        rejected or orphaned content

    Call :meth:`start` (or use the manager as a context manager) before any
    other operation; it resolves interrupted journals and loads the active
    generations.

    Args:
        root: Activation root directory.
        loader: Loads, warms, and unloads generation handles.
        policy: Activation policy. Defaults to :class:`ActivationPolicy`.
        source: Default artifact source. Defaults to
            :class:`HubArtifactSource`.
        clock: Returns the current aware UTC time.
        event_sink: Optional callable receiving value-free
            :class:`ActivationEvent` objects.
        fault_hook: Optional test seam called with a name from
            :data:`ACTIVATION_BOUNDARIES` at each transaction boundary.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        loader: GenerationLoader,
        policy: ActivationPolicy | None = None,
        source: ArtifactSource | None = None,
        clock: Callable[[], datetime] | None = None,
        event_sink: Callable[[ActivationEvent], None] | None = None,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        """Configure the manager without touching the filesystem."""

        self._root = Path(root)
        self._loader = loader
        self._policy = policy or ActivationPolicy()
        self._default_source = source
        self._clock = clock or _utcnow
        self._event_sink = event_sink
        self._fault_hook = fault_hook
        self._state_lock = threading.RLock()
        self._slot_locks: dict[str, threading.Lock] = {}
        self._slot_locks_guard = threading.Lock()
        self._table_lock = threading.Lock()
        self._receipt_lock = threading.Lock()
        # Serializes in-process journal file I/O. Besides ordering, this keeps
        # an atomic replace from racing an open reader, which Windows rejects.
        self._journal_lock = threading.RLock()
        self._live: dict[str, dict[int, _LiveGeneration]] = {}
        self._active: dict[str, int | None] = {}
        self._owner: _OwnerLock | None = None
        self._started = False
        self._receipt_sequence = 0
        self._receipt_digest: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def policy(self) -> ActivationPolicy:
        """Return the configured activation policy."""

        return self._policy

    @property
    def started(self) -> bool:
        """Return whether :meth:`start` completed."""

        return self._started

    def start(self) -> tuple[ActivationResult, ...]:
        """Take ownership of the root, recover, and load active generations.

        Recovery resolves every unfinished journal without manual editing:
        content that was never verified is quarantined, unpublished candidates
        and unproven health windows roll back, and committed transactions
        interrupted after publication resume. Orphaned staging content is then
        quarantined, unreferenced generation directories are removed, and the
        quarantine and journal history are pruned to their bounds.

        Returns:
            One result per resolved journal, followed by a result for any
            active generation that could not be verified or loaded.

        Raises:
            ModelActivationError: ``activation_root_busy`` when another process
                owns the root, or ``state_unreadable`` when pointer state is
                corrupt (nothing is modified in that case).
        """

        if self._started:
            return ()
        for directory in (
            self._root,
            self._journal_dir,
            self._staging_dir,
            self._generations_dir,
            self._quarantine_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        owner = _OwnerLock(self._root / _OWNER_LOCK_FILENAME)
        if not owner.acquire():
            raise ModelActivationError(ActivationReason.ROOT_BUSY)
        self._owner = owner
        try:
            self._read_state()
            self._load_receipt_tail()
            results = list(self._recover_journals())
            open_transactions = {
                journal["transaction_id"]
                for journal in self._iter_journals()
                if journal["status"] not in _TERMINAL_STATUSES
            }
            self._sweep_orphans(open_transactions)
            self._prune_journals()
            self._prune_quarantine()
            results.extend(self._load_active_generations())
        except BaseException:
            self._owner = None
            owner.release()
            raise
        self._started = True
        return tuple(results)

    def close(self) -> None:
        """Release idle generation handles and ownership of the root.

        Durable state is never modified, so closing is safe at any point,
        including immediately after a simulated crash.
        """

        with self._table_lock:
            idle = [
                live
                for table in self._live.values()
                for live in table.values()
                if live.refs == 0 and not live.disposed
            ]
            for live in idle:
                live.disposed = True
            self._live.clear()
            self._active.clear()
        for live in idle:
            self._unload_quietly(live)
        self._started = False
        owner, self._owner = self._owner, None
        if owner is not None:
            owner.release()

    def __enter__(self) -> "ModelActivationManager":
        """Start the manager."""

        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the manager."""

        self.close()

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def activate(
        self,
        slot: str,
        reference: ModelReference,
        *,
        source: ArtifactSource | None = None,
        compatibility: RegistryCheckpoint | Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ActivationResult:
        """Run one activation transaction for ``slot``.

        Args:
            slot: Service model slot, lowercase ``name`` or
                ``family::tier::format``.
            reference: Model to activate.
            source: Artifact source for this transaction; defaults to the
                manager's source.
            compatibility: Registry checkpoint metadata for the candidate. It
                is compared with the retained last-green generation and stored
                with the generation for the next activation's check.
            idempotency_key: Optional caller key. The transaction id is derived
                from it, so repeating the call replays the recorded outcome
                instead of starting a second transaction.

        Returns:
            The transaction result. Concurrent activations of one slot return
            ``busy``; a different reference while a health window is open
            returns ``conflict``; repeating a committed reference returns
            ``already_active``.

        Raises:
            ModelActivationError: ``manager_not_started`` or
                ``invalid_request``.
        """

        self._require_started()
        slot = _normalize_slot(slot)
        if not isinstance(reference, ModelReference):
            raise ModelActivationError(ActivationReason.INVALID_REQUEST)
        transaction_id = _transaction_id(slot, idempotency_key)
        ref_digest = reference.digest
        lock = self._slot_lock(slot)
        if not lock.acquire(blocking=False):
            replay = self._replay(transaction_id, slot, ref_digest)
            if replay is not None:
                return replay
            return self._result(
                transaction_id,
                slot,
                ActivationOutcome.BUSY,
                model_ref_digest=ref_digest,
                reason=ActivationReason.SLOT_BUSY,
            )
        try:
            replay = self._replay(transaction_id, slot, ref_digest)
            if replay is not None:
                return replay
            entry = _slot_entry(self._read_state(), slot)
            pending = entry["pending_transaction"]
            if pending is not None:
                journal = self._read_journal(pending)
                if journal is not None and journal["model_ref_digest"] == ref_digest:
                    return self._journal_result(
                        journal, ActivationOutcome.PENDING_HEALTH, replayed=True
                    )
                return self._result(
                    transaction_id,
                    slot,
                    ActivationOutcome.CONFLICT,
                    model_ref_digest=ref_digest,
                    reason=ActivationReason.SLOT_CONFLICT,
                )
            active = entry["active"]
            if (
                active is not None
                and active["model_ref_digest"] == ref_digest
                and reference.expected_artifact_digest
                in (None, active["artifact_digest"])
            ):
                return self._result(
                    active["transaction_id"],
                    slot,
                    ActivationOutcome.ALREADY_ACTIVE,
                    model_ref_digest=ref_digest,
                    candidate_generation=active["generation"],
                    artifact_digest=active["artifact_digest"],
                    replayed=True,
                )
            try:
                checkpoint = _checkpoint_payload(compatibility)
            except _PhaseFailure as failure:
                return self._result(
                    transaction_id,
                    slot,
                    ActivationOutcome.REFUSED,
                    model_ref_digest=ref_digest,
                    reason=failure.reason,
                )
            chosen = source or self._default_source or HubArtifactSource()
            return self._run_transaction(
                transaction_id, slot, reference, chosen, checkpoint, entry
            )
        finally:
            lock.release()

    def record_health(self, slot: str, *, healthy: bool) -> ActivationResult:
        """Record one health signal for the slot's open health window.

        An unhealthy signal rolls back to the verified last-green generation
        immediately. A healthy signal is counted toward
        :attr:`ActivationPolicy.min_healthy_reports`.

        Args:
            slot: Service model slot.
            healthy: Whether the new generation is serving correctly.

        Returns:
            ``pending_health``, ``rolled_back``, ``busy``, or ``refused`` with
            ``no_health_window``.
        """

        self._require_started()
        slot = _normalize_slot(slot)
        with self._slot_operation(slot) as busy:
            if busy is not None:
                return busy
            journal = self._pending_journal(slot)
            if journal is None:
                return self._slot_result(
                    slot, ActivationOutcome.REFUSED, ActivationReason.NO_HEALTH_WINDOW
                )
            health = journal["health"]
            if not healthy:
                health["unhealthy_reports"] = (
                    int(health.get("unhealthy_reports", 0)) + 1
                )
                with contextlib.suppress(OSError):
                    self._write_journal(journal)
                return self._rollback(journal, ActivationReason.HEALTH_WINDOW_FAILED)
            health["healthy_reports"] = int(health.get("healthy_reports", 0)) + 1
            try:
                self._write_journal(journal)
            except OSError as exc:
                return self._journal_result(
                    journal,
                    ActivationOutcome.PENDING_HEALTH,
                    reason=_os_reason(exc, ActivationReason.COMMIT_FAILED),
                )
            return self._journal_result(journal, ActivationOutcome.PENDING_HEALTH)

    def poll_health(self, slot: str) -> ActivationResult:
        """Close the slot's health window once its deadline has passed.

        Returns:
            ``pending_health`` while the window is open, ``activated`` when it
            passed (the new generation becomes last green and the previous one
            is retired), ``rolled_back`` with ``health_window_unproven`` when
            too few healthy reports arrived, or ``refused`` with
            ``no_health_window``.
        """

        self._require_started()
        slot = _normalize_slot(slot)
        with self._slot_operation(slot) as busy:
            if busy is not None:
                return busy
            journal = self._pending_journal(slot)
            if journal is None:
                return self._slot_result(
                    slot, ActivationOutcome.REFUSED, ActivationReason.NO_HEALTH_WINDOW
                )
            health = journal["health"]
            if (
                journal["phase"] == ActivationPhase.ROLLBACK.value
                or int(health.get("unhealthy_reports", 0)) > 0
            ):
                # A failed window never passes; retry the refused rollback.
                return self._rollback(
                    journal,
                    _reason_or(journal.get("reason"), None)
                    or ActivationReason.HEALTH_WINDOW_FAILED,
                )
            if self._now() < _parse_time(health["deadline"]):
                return self._journal_result(journal, ActivationOutcome.PENDING_HEALTH)
            if int(health.get("healthy_reports", 0)) < int(
                health.get("min_healthy_reports", 0)
            ):
                return self._rollback(journal, ActivationReason.HEALTH_WINDOW_UNPROVEN)
            return self._finalize(journal)

    def rollback(self, slot: str) -> ActivationResult:
        """Roll the slot back to its verified last-green generation.

        Rollback is available while a health window retains a distinct
        last-green target; once the window passes there is nothing older to
        return to.

        Returns:
            ``rolled_back``, ``busy``, or ``refused`` with a controlled reason.
        """

        self._require_started()
        slot = _normalize_slot(slot)
        with self._slot_operation(slot) as busy:
            if busy is not None:
                return busy
            journal = self._pending_journal(slot)
            if journal is None:
                return self._slot_result(
                    slot,
                    ActivationOutcome.REFUSED,
                    ActivationReason.ROLLBACK_UNAVAILABLE,
                )
            return self._rollback(journal, ActivationReason.ROLLBACK_REQUESTED)

    def lease(self, slot: str) -> GenerationLease:
        """Pin the slot's currently served generation for one request.

        Raises:
            ModelActivationError: ``no_active_generation`` when the slot has no
                loaded generation.
        """

        self._require_started()
        slot = _normalize_slot(slot)
        with self._table_lock:
            generation = self._active.get(slot)
            if generation is None:
                raise ModelActivationError(ActivationReason.NO_ACTIVE_GENERATION)
            live = self._live[slot][generation]
            live.refs += 1
        return GenerationLease(self, live)

    def status(self, slot: str) -> dict[str, Any]:
        """Return the slot's value-free pointer and serving status."""

        self._require_started()
        slot = _normalize_slot(slot)
        with self._state_lock:
            entry = _slot_entry(self._read_state(), slot)
        with self._table_lock:
            serving = self._active.get(slot)
            resident = sorted(self._live.get(slot, {}))
        active = entry["active"]
        last_green = entry["last_green"]
        return {
            "slot": slot,
            "active_generation": None if active is None else active["generation"],
            "last_green_generation": (
                None if last_green is None else last_green["generation"]
            ),
            "pending_transaction": entry["pending_transaction"],
            "health_deadline": entry["health_deadline"],
            "serving_generation": serving,
            "resident_generations": resident,
        }

    def receipts(self) -> list[dict[str, Any]]:
        """Return every recorded receipt in order."""

        path = self._root / _RECEIPTS_FILENAME
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="ascii").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def verify_receipts(self) -> bool:
        """Return whether the receipt log is an unbroken hash chain."""

        return verify_receipt_chain(self._root / _RECEIPTS_FILENAME)

    # ------------------------------------------------------------------
    # Transaction
    # ------------------------------------------------------------------

    def _run_transaction(
        self,
        transaction_id: str,
        slot: str,
        reference: ModelReference,
        source: ArtifactSource,
        checkpoint: Mapping[str, Any] | None,
        entry: Mapping[str, Any],
    ) -> ActivationResult:
        now = self._now_text()
        active = entry["active"]
        journal: dict[str, Any] = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "transaction_id": transaction_id,
            "slot": slot,
            "model_ref_digest": reference.digest,
            "status": TransactionStatus.IN_PROGRESS.value,
            "outcome": None,
            "reason": None,
            "phase": ActivationPhase.ACQUIRE.value,
            "phase_completed": False,
            "generation": None,
            "previous_generation": None if active is None else active["generation"],
            "artifact_digest": None,
            "files": None,
            "bytes": None,
            "compatibility": None,
            "health": None,
            "created_at": now,
            "updated_at": now,
            "history": [],
        }
        staging = self._staging_dir / transaction_id
        try:
            self._begin(journal, ActivationPhase.ACQUIRE)
        except OSError as exc:
            with contextlib.suppress(OSError):
                self._journal_path(transaction_id).unlink()
            return self._result(
                transaction_id,
                slot,
                ActivationOutcome.REFUSED,
                model_ref_digest=reference.digest,
                reason=_os_reason(exc, ActivationReason.ACQUIRE_FAILED),
            )

        phase = ActivationPhase.ACQUIRE
        try:
            workspace = staging / _WORKSPACE_DIRNAME
            workspace.mkdir(parents=True)
            self._fault("acquire.before")
            acquired = source.acquire(reference, workspace)
            self._fault("acquire.after")
            acquired_root = _contained_directory(acquired, workspace)
            self._complete(journal, phase)

            phase = ActivationPhase.STAGE
            self._begin(journal, phase)
            self._fault("stage.before")
            artifact_dir = staging / _ARTIFACT_DIRNAME
            _materialize(
                acquired_root,
                workspace,
                artifact_dir,
                max_files=self._policy.max_artifact_files,
                max_bytes=self._policy.max_artifact_bytes,
            )
            shutil.rmtree(workspace)
            self._fault("stage.after")
            self._complete(journal, phase)

            phase = ActivationPhase.VERIFY
            self._begin(journal, phase)
            self._fault("verify.before")
            verified = self._verify_staged(
                slot, reference, artifact_dir, entry["last_green"], checkpoint
            )
            journal["artifact_digest"] = verified["artifact_digest"]
            journal["files"] = verified["files"]
            journal["bytes"] = verified["bytes"]
            journal["compatibility"] = verified["compatibility"]
            _write_json_atomic(
                staging / _GENERATION_FILENAME,
                {
                    "schema_version": GENERATION_SCHEMA_VERSION,
                    "transaction_id": transaction_id,
                    "model_ref_digest": reference.digest,
                    "artifact_digest": verified["artifact_digest"],
                    "checkpoint": checkpoint,
                },
            )
            self._fault("verify.after")
            self._complete(journal, phase)
        except _PhaseFailure as failure:
            if failure.compatibility is not None:
                journal["compatibility"] = failure.compatibility
            return self._quarantine_transaction(journal, failure.reason)
        except Exception as exc:
            return self._quarantine_transaction(journal, _failure_reason(exc, phase))

        return self._load_and_commit(journal, staging)

    def _load_and_commit(
        self, journal: dict[str, Any], staging: Path
    ) -> ActivationResult:
        slot = journal["slot"]
        handle: Any = None
        loaded = False
        context: GenerationContext | None = None
        phase = ActivationPhase.LOAD
        try:
            generation = self._allocate_generation(slot)
            journal["generation"] = generation
            self._begin(journal, phase)
            generation_dir = self._generation_dir(slot, generation)
            if generation_dir.exists():
                shutil.rmtree(generation_dir)
            generation_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, generation_dir)
            _fsync_directory(generation_dir.parent)
            context = GenerationContext(
                slot=slot,
                generation=generation,
                transaction_id=journal["transaction_id"],
                model_ref_digest=journal["model_ref_digest"],
                artifact_digest=journal["artifact_digest"],
                artifact_dir=generation_dir / _ARTIFACT_DIRNAME,
            )
            self._fault("load.before")
            handle = self._loader.load(context)
            loaded = True
            self._fault("load.after")
            self._complete(journal, phase)

            phase = ActivationPhase.WARM
            self._begin(journal, phase)
            self._fault("warm.before")
            self._loader.warm(handle, context)
            self._fault("warm.after")
            self._complete(journal, phase)
        except Exception as exc:
            if loaded and context is not None:
                self._unload_quietly(_LiveGeneration(context, handle))
            return self._discard_candidate(journal, _failure_reason(exc, phase))

        live = _LiveGeneration(context, handle)
        return self._publish(journal, live)

    def _publish(
        self, journal: dict[str, Any], live: _LiveGeneration
    ) -> ActivationResult:
        slot = journal["slot"]
        window = float(self._policy.health_window_seconds)
        committed_at = self._now()
        deadline = committed_at + timedelta(seconds=window)
        journal["health"] = {
            "window_seconds": window,
            "committed_at": _format_time(committed_at),
            "deadline": _format_time(deadline),
            "healthy_reports": 0,
            "min_healthy_reports": self._policy.min_healthy_reports,
        }
        record = {
            "generation": live.context.generation,
            "transaction_id": journal["transaction_id"],
            "model_ref_digest": journal["model_ref_digest"],
            "artifact_digest": journal["artifact_digest"],
            "committed_at": _format_time(committed_at),
        }
        previous: dict[str, Any] | None = None
        published = False
        try:
            self._begin(journal, ActivationPhase.COMMIT)
            self._fault("commit.before")
            with self._state_lock:
                state = self._read_state()
                entry = _slot_entry(state, slot)
                previous = copy.deepcopy(entry)
                entry["active"] = record
                entry["pending_transaction"] = journal["transaction_id"]
                entry["health_deadline"] = _format_time(deadline)
                _store_slot_entry(state, slot, entry)
                self._write_state(state)
                published = True
                self._fault("commit.published")
                self._set_active(slot, live)
            journal["status"] = TransactionStatus.HEALTH_WINDOW.value
            self._complete(journal, ActivationPhase.COMMIT)
            self._fault("commit.after")
        except Exception as exc:
            reason = _failure_reason(exc, ActivationPhase.COMMIT)
            if published and previous is not None:
                if not self._restore_entry(slot, previous):
                    # The restoring write failed too. Serve what is durable,
                    # the complete new generation in its health window, and
                    # let the next start resolve the open journal.
                    self._set_active(slot, live)
                    self._emit(journal, "commit", "failed", reason)
                    return self._journal_result(
                        journal,
                        ActivationOutcome.PENDING_HEALTH,
                        reason=ActivationReason.ROLLBACK_FAILED,
                    )
                self._restore_previous_live(slot, previous)
            self._drop_candidate_live(slot, live)
            return self._discard_candidate(journal, reason)

        if not self._policy.keep_last_green_resident and previous is not None:
            standby = previous["active"]
            if standby is not None:
                self._retire_live(slot, standby["generation"])
        if window <= 0:
            return self._finalize(journal)
        self._append_receipt("pending_health", journal)
        return self._journal_result(journal, ActivationOutcome.PENDING_HEALTH)

    def _finalize(self, journal: dict[str, Any]) -> ActivationResult:
        slot = journal["slot"]
        retired: dict[str, Any] | None = None
        promoted = False
        try:
            self._begin(journal, ActivationPhase.RETIRE)
            self._fault("retire.before")
            with self._state_lock:
                state = self._read_state()
                entry = _slot_entry(state, slot)
                retired = entry["last_green"]
                entry["last_green"] = entry["active"]
                entry["pending_transaction"] = None
                entry["health_deadline"] = None
                _store_slot_entry(state, slot, entry)
                self._write_state(state)
                promoted = True
                self._fault("retire.published")
        except Exception as exc:
            reason = _failure_reason(exc, ActivationPhase.RETIRE)
            if not promoted:
                return self._journal_result(
                    journal, ActivationOutcome.PENDING_HEALTH, reason=reason
                )
            # Promotion is durable; finish retiring and let recovery redo the
            # journal update if it cannot be written now.
            self._emit(journal, "retire", "failed", reason)
        if retired is not None and retired["generation"] != journal["generation"]:
            self._retire_generation(slot, retired["generation"])
        journal["status"] = TransactionStatus.COMMITTED.value
        journal["outcome"] = ActivationOutcome.ACTIVATED.value
        try:
            self._complete(journal, ActivationPhase.RETIRE)
            self._fault("retire.after")
        except Exception as exc:
            self._emit(
                journal,
                "retire",
                "failed",
                _failure_reason(exc, ActivationPhase.RETIRE),
            )
        self._append_receipt(
            "activated",
            journal,
            from_generation=None if retired is None else retired["generation"],
            to_generation=journal["generation"],
        )
        self._prune_journals()
        return self._journal_result(journal, ActivationOutcome.ACTIVATED)

    def _rollback(
        self,
        journal: dict[str, Any],
        reason: ActivationReason,
        *,
        load_target: bool = True,
    ) -> ActivationResult:
        slot = journal["slot"]
        transaction_id = journal["transaction_id"]
        with self._state_lock:
            entry = _slot_entry(self._read_state(), slot)
        failing = entry["active"]
        target = entry["last_green"]
        if failing is None or failing["transaction_id"] != transaction_id:
            # Nothing of this transaction is published any more.
            return self._close_rolled_back(journal, reason, None, target)
        compatibility = journal.get("compatibility") or {}
        if target is not None and compatibility.get("decision") not in (
            "compatible",
            "waived",
        ):
            return self._journal_result(
                journal,
                ActivationOutcome.REFUSED,
                reason=ActivationReason.ROLLBACK_INCOMPATIBLE,
            )
        target_live: _LiveGeneration | None = None
        if target is not None:
            if self._policy.reverify_before_rollback and not self._generation_verifies(
                slot, target
            ):
                return self._journal_result(
                    journal,
                    ActivationOutcome.REFUSED,
                    reason=ActivationReason.ROLLBACK_TARGET_UNVERIFIED,
                )
            if load_target:
                target_live = self._ensure_live(slot, target)
                if target_live is None:
                    return self._journal_result(
                        journal,
                        ActivationOutcome.REFUSED,
                        reason=ActivationReason.ROLLBACK_TARGET_UNLOADABLE,
                    )
        rolled_back = False
        try:
            journal["reason"] = reason.value
            self._begin(journal, ActivationPhase.ROLLBACK)
            self._fault("rollback.before")
            with self._state_lock:
                state = self._read_state()
                entry = _slot_entry(state, slot)
                entry["active"] = copy.deepcopy(target)
                entry["pending_transaction"] = None
                entry["health_deadline"] = None
                _store_slot_entry(state, slot, entry)
                self._write_state(state)
                rolled_back = True
                if load_target:
                    self._set_active(slot, target_live)
                self._fault("rollback.published")
        except Exception as exc:
            self._emit(
                journal,
                "rollback",
                "failed",
                _failure_reason(exc, ActivationPhase.ROLLBACK),
            )
            if not rolled_back:
                return self._journal_result(
                    journal,
                    ActivationOutcome.REFUSED,
                    reason=ActivationReason.ROLLBACK_FAILED,
                )
            # The pointer already moved durably: finish rolling forward.
        self._retire_generation(slot, failing["generation"])
        return self._close_rolled_back(journal, reason, failing, target)

    def _close_rolled_back(
        self,
        journal: dict[str, Any],
        reason: ActivationReason,
        failing: Mapping[str, Any] | None,
        target: Mapping[str, Any] | None,
    ) -> ActivationResult:
        journal["status"] = TransactionStatus.ROLLED_BACK.value
        journal["outcome"] = ActivationOutcome.ROLLED_BACK.value
        journal["reason"] = reason.value
        try:
            if journal["phase"] != ActivationPhase.ROLLBACK.value:
                self._begin(journal, ActivationPhase.ROLLBACK)
            self._complete(journal, ActivationPhase.ROLLBACK)
            self._fault("rollback.after")
        except Exception as exc:
            self._emit(
                journal,
                "rollback",
                "failed",
                _failure_reason(exc, ActivationPhase.ROLLBACK),
            )
        self._append_receipt(
            "rolled_back",
            journal,
            from_generation=None if failing is None else failing["generation"],
            to_generation=None if target is None else target["generation"],
        )
        self._prune_journals()
        return self._journal_result(
            journal, ActivationOutcome.ROLLED_BACK, reason=reason
        )

    def _quarantine_transaction(
        self, journal: dict[str, Any], reason: ActivationReason
    ) -> ActivationResult:
        transaction_id = journal["transaction_id"]
        self._emit(journal, journal["phase"], "failed", reason)
        try:
            self._quarantine(transaction_id, self._staging_dir / transaction_id, reason)
        except OSError:
            # Leave the staging entry for the next start's orphan sweep.
            pass
        journal["status"] = TransactionStatus.QUARANTINED.value
        journal["outcome"] = ActivationOutcome.QUARANTINED.value
        journal["reason"] = reason.value
        with contextlib.suppress(OSError):
            self._write_journal(journal)
        self._append_receipt("quarantined", journal)
        self._prune_journals()
        return self._journal_result(
            journal, ActivationOutcome.QUARANTINED, reason=reason
        )

    def _discard_candidate(
        self, journal: dict[str, Any], reason: ActivationReason
    ) -> ActivationResult:
        transaction_id = journal["transaction_id"]
        self._emit(journal, journal["phase"], "failed", reason)
        with contextlib.suppress(OSError):
            _remove_tree(self._staging_dir / transaction_id)
        generation = journal.get("generation")
        if isinstance(generation, int) and not self._generation_in_use(
            journal["slot"], generation
        ):
            with contextlib.suppress(OSError):
                self._remove_generation_dir_if_unreferenced(journal["slot"], generation)
        journal["status"] = TransactionStatus.ROLLED_BACK.value
        journal["outcome"] = ActivationOutcome.ROLLED_BACK.value
        journal["reason"] = reason.value
        with contextlib.suppress(OSError):
            self._write_journal(journal)
        self._append_receipt("rolled_back", journal)
        self._prune_journals()
        return self._journal_result(
            journal, ActivationOutcome.ROLLED_BACK, reason=reason
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_staged(
        self,
        slot: str,
        reference: ModelReference,
        artifact_dir: Path,
        last_green: Mapping[str, Any] | None,
        checkpoint: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        result, inventory = _verify_tree(
            artifact_dir,
            max_files=self._policy.max_artifact_files,
            expected_model_id=reference.model_id,
            expected_reproducibility_hash=reference.expected_reproducibility_hash,
        )
        total_bytes = sum(inventory.values())
        if (
            self._policy.max_artifact_bytes is not None
            and total_bytes > self._policy.max_artifact_bytes
        ):
            raise _PhaseFailure(ActivationReason.ARTIFACT_TOO_LARGE)
        artifact_digest = result.actual_sha256
        if (
            reference.expected_artifact_digest is not None
            and artifact_digest != reference.expected_artifact_digest
        ):
            raise _PhaseFailure(ActivationReason.PIN_MISMATCH)
        staged = StagedArtifact(
            slot=slot,
            model_ref_digest=reference.digest,
            artifact_digest=artifact_digest,
            files=tuple(
                sorted(name for name in inventory if name != ARTIFACT_MANIFEST_FILENAME)
            ),
            total_bytes=total_bytes,
            artifact_dir=artifact_dir,
        )
        for validator in self._policy.validators:
            try:
                accepted = validator(staged)
            except Exception:
                accepted = False
            if accepted is not True:
                raise _PhaseFailure(ActivationReason.POLICY_REJECTED)
        compatibility = self._compatibility(slot, last_green, checkpoint)
        return {
            "artifact_digest": artifact_digest,
            "files": result.files_checked,
            "bytes": total_bytes,
            "compatibility": compatibility,
        }

    def _compatibility(
        self,
        slot: str,
        last_green: Mapping[str, Any] | None,
        checkpoint: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if last_green is None:
            return {"decision": "not_applicable"}
        previous = self._generation_checkpoint(slot, last_green["generation"])
        report = build_rollback_compatibility_report(
            current=checkpoint, rollback=previous
        )
        payload = {
            "decision": report.decision,
            "fingerprint": report.fingerprint,
            "blocked_reasons": list(report.blocked_reasons),
        }
        if report.compatible:
            return payload
        if self._policy.require_rollback_compatibility:
            raise _PhaseFailure(
                ActivationReason.ROLLBACK_INCOMPATIBLE, compatibility=payload
            )
        payload["decision"] = "waived"
        return payload

    def _generation_checkpoint(
        self, slot: str, generation: int
    ) -> Mapping[str, Any] | None:
        path = self._generation_dir(slot, generation) / _GENERATION_FILENAME
        try:
            payload = _read_json(path, _MAX_GENERATION_METADATA_BYTES)
        except (OSError, ValueError):
            return None
        checkpoint = payload.get("checkpoint")
        return checkpoint if isinstance(checkpoint, Mapping) else None

    def _generation_verifies(self, slot: str, record: Mapping[str, Any]) -> bool:
        generation_dir = self._generation_dir(slot, record["generation"])
        try:
            metadata = _read_json(
                generation_dir / _GENERATION_FILENAME, _MAX_GENERATION_METADATA_BYTES
            )
            if metadata.get("transaction_id") != record["transaction_id"]:
                return False
            result, _ = _verify_tree(
                generation_dir / _ARTIFACT_DIRNAME,
                max_files=self._policy.max_artifact_files,
            )
        except (_PhaseFailure, OSError, ValueError):
            return False
        return result.actual_sha256 == record["artifact_digest"]

    # ------------------------------------------------------------------
    # Generation table
    # ------------------------------------------------------------------

    def _context_for(self, slot: str, record: Mapping[str, Any]) -> GenerationContext:
        return GenerationContext(
            slot=slot,
            generation=int(record["generation"]),
            transaction_id=str(record["transaction_id"]),
            model_ref_digest=str(record["model_ref_digest"]),
            artifact_digest=str(record["artifact_digest"]),
            artifact_dir=self._generation_dir(slot, int(record["generation"]))
            / _ARTIFACT_DIRNAME,
        )

    def _ensure_live(
        self, slot: str, record: Mapping[str, Any]
    ) -> _LiveGeneration | None:
        generation = int(record["generation"])
        with self._table_lock:
            live = self._live.get(slot, {}).get(generation)
            if live is not None and not live.disposed:
                live.retired = False
                return live
        context = self._context_for(slot, record)
        try:
            handle = self._loader.load(context)
        except Exception:
            return None
        live = _LiveGeneration(context, handle)
        try:
            self._loader.warm(handle, context)
        except Exception:
            self._unload_quietly(live)
            return None
        with self._table_lock:
            self._live.setdefault(slot, {})[generation] = live
        return live

    def _set_active(self, slot: str, live: _LiveGeneration | None) -> None:
        with self._table_lock:
            if live is None:
                self._active[slot] = None
                return
            live.retired = False
            self._live.setdefault(slot, {})[live.context.generation] = live
            self._active[slot] = live.context.generation

    def _restore_previous_live(self, slot: str, previous: Mapping[str, Any]) -> None:
        active = previous.get("active")
        with self._table_lock:
            if active is None:
                self._active[slot] = None
                return
            if active["generation"] in self._live.get(slot, {}):
                self._active[slot] = active["generation"]

    def _retire_live(self, slot: str, generation: int) -> None:
        dispose: _LiveGeneration | None = None
        with self._table_lock:
            live = self._live.get(slot, {}).get(generation)
            if live is None or self._active.get(slot) == generation:
                return
            live.retired = True
            if live.refs == 0 and not live.disposed:
                live.disposed = True
                self._live[slot].pop(generation, None)
                dispose = live
        if dispose is not None:
            self._unload_quietly(dispose)

    def _drop_candidate_live(self, slot: str, live: _LiveGeneration) -> None:
        dispose = False
        with self._table_lock:
            table = self._live.get(slot, {})
            if table.get(live.context.generation) is not live:
                dispose = not live.disposed
                live.disposed = True
            else:
                live.retired = True
                if live.refs == 0 and not live.disposed:
                    live.disposed = True
                    table.pop(live.context.generation, None)
                    dispose = True
        if dispose:
            self._unload_quietly(live)

    def _generation_in_use(self, slot: str, generation: int) -> bool:
        with self._table_lock:
            live = self._live.get(slot, {}).get(generation)
            return live is not None and live.refs > 0

    def _retire_generation(self, slot: str, generation: int) -> None:
        deferred = False
        dispose: _LiveGeneration | None = None
        with self._table_lock:
            live = self._live.get(slot, {}).get(generation)
            if live is not None:
                if self._active.get(slot) == generation:
                    self._active[slot] = None
                live.retired = True
                if live.refs == 0 and not live.disposed:
                    live.disposed = True
                    self._live[slot].pop(generation, None)
                    dispose = live
                elif live.refs > 0:
                    deferred = True
        if dispose is not None:
            self._unload_quietly(dispose)
        if not deferred:
            with contextlib.suppress(OSError):
                self._remove_generation_dir_if_unreferenced(slot, generation)

    def _release_lease(self, live: _LiveGeneration) -> None:
        dispose = False
        slot = live.context.slot
        generation = live.context.generation
        with self._table_lock:
            live.refs = max(live.refs - 1, 0)
            if live.retired and live.refs == 0 and not live.disposed:
                live.disposed = True
                table = self._live.get(slot, {})
                if table.get(generation) is live:
                    table.pop(generation, None)
                dispose = True
        if dispose:
            self._unload_quietly(live)
            if self._started:
                with contextlib.suppress(OSError, ModelActivationError):
                    self._remove_generation_dir_if_unreferenced(slot, generation)

    def _unload_quietly(self, live: _LiveGeneration) -> None:
        try:
            self._loader.unload(live.handle, live.context)
        except Exception:
            self._emit_raw(
                live.context.transaction_id,
                live.context.slot,
                "unload",
                "failed",
                live.context.generation,
                None,
            )

    def _remove_generation_dir_if_unreferenced(
        self, slot: str, generation: int
    ) -> None:
        with self._state_lock:
            entry = _slot_entry(self._read_state(), slot)
            referenced = {
                record["generation"]
                for record in (entry["active"], entry["last_green"])
                if record is not None
            }
            if generation in referenced:
                return
            _remove_tree(self._generation_dir(slot, generation))

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _recover_journals(self) -> Iterator[ActivationResult]:
        journals: list[dict[str, Any]] = []
        for path in sorted(self._journal_dir.glob("txn-*.json")):
            try:
                journal = self._load_journal_file(path)
            except ValueError:
                self._quarantine_journal(path)
                continue
            if journal["status"] not in _TERMINAL_STATUSES:
                journals.append(journal)
        journals.sort(key=lambda item: (item["created_at"], item["transaction_id"]))
        covered: set[str] = set()
        for journal in journals:
            covered.add(journal["transaction_id"])
            yield self._resolve_interrupted(journal)
        for slot, entry in sorted(self._read_state()["slots"].items()):
            pending = entry.get("pending_transaction")
            if pending is None or pending in covered:
                continue
            # The pending journal is gone: health is unproven, so roll back.
            yield self._resolve_interrupted(self._synthetic_journal(slot, entry))

    def _resolve_interrupted(self, journal: dict[str, Any]) -> ActivationResult:
        slot = journal["slot"]
        transaction_id = journal["transaction_id"]
        phase = journal["phase"]
        completed = bool(journal["phase_completed"])
        entry = _slot_entry(self._read_state(), slot)
        active = entry["active"]
        published = active is not None and active["transaction_id"] == transaction_id
        self._emit(journal, "recovery", "started", None)

        window = float((journal.get("health") or {}).get("window_seconds", 0))
        if phase == ActivationPhase.ROLLBACK.value:
            reason = _reason_or(journal.get("reason"), ActivationReason.INTERRUPTED)
            result = self._rollback(
                journal, reason or ActivationReason.INTERRUPTED, load_target=False
            )
        elif published and (phase == ActivationPhase.RETIRE.value or window <= 0):
            # Committed with no open health window: finish the retirement.
            result = self._finalize(journal)
            if result.outcome is ActivationOutcome.ACTIVATED:
                result = self._journal_result(
                    journal, ActivationOutcome.RESUMED, replayed=True
                )
        elif published:
            # A restart before the window passed leaves health unproven.
            result = self._rollback(
                journal,
                ActivationReason.HEALTH_WINDOW_INTERRUPTED,
                load_target=False,
            )
        elif phase in (ActivationPhase.ACQUIRE.value, ActivationPhase.STAGE.value) or (
            phase == ActivationPhase.VERIFY.value and not completed
        ):
            result = self._quarantine_transaction(journal, ActivationReason.INTERRUPTED)
        else:
            result = self._discard_candidate(journal, ActivationReason.INTERRUPTED)
        self._emit(journal, "recovery", result.outcome.value, result.reason)
        return result

    def _synthetic_journal(self, slot: str, entry: Mapping[str, Any]) -> dict[str, Any]:
        active = entry["active"] or {}
        now = self._now_text()
        return {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "transaction_id": entry["pending_transaction"],
            "slot": slot,
            "model_ref_digest": active.get("model_ref_digest"),
            "status": TransactionStatus.HEALTH_WINDOW.value,
            "outcome": None,
            "reason": None,
            "phase": ActivationPhase.COMMIT.value,
            "phase_completed": True,
            "generation": active.get("generation"),
            "previous_generation": None,
            "artifact_digest": active.get("artifact_digest"),
            "files": None,
            "bytes": None,
            "compatibility": {"decision": "waived"},
            "health": {
                "window_seconds": 1.0,
                "deadline": now,
                "healthy_reports": 0,
                "min_healthy_reports": 1,
            },
            "created_at": now,
            "updated_at": now,
            "history": [],
        }

    def _load_active_generations(self) -> Iterator[ActivationResult]:
        state = self._read_state()
        for slot, entry in sorted(state["slots"].items()):
            active = entry.get("active")
            if active is None:
                continue
            with self._table_lock:
                if active["generation"] in self._live.get(slot, {}):
                    self._active[slot] = active["generation"]
                    continue
            if self._policy.reverify_on_start and not self._generation_verifies(
                slot, active
            ):
                yield self._slot_result(
                    slot, ActivationOutcome.REFUSED, ActivationReason.ACTIVE_UNVERIFIED
                )
                continue
            live = self._ensure_live(slot, active)
            if live is None:
                yield self._slot_result(
                    slot, ActivationOutcome.REFUSED, ActivationReason.ACTIVE_UNLOADABLE
                )
                continue
            self._set_active(slot, live)

    def _sweep_orphans(self, open_transactions: set[str]) -> None:
        state = self._read_state()
        for child in sorted(self._staging_dir.iterdir()):
            if child.name in open_transactions:
                continue
            token = (
                child.name
                if _TRANSACTION_RE.fullmatch(child.name)
                else "orphan-" + uuid.uuid4().hex
            )
            with contextlib.suppress(OSError):
                self._quarantine(token, child, ActivationReason.ORPHANED_STAGING)
        referenced: dict[str, set[str]] = {}
        for slot, entry in state["slots"].items():
            names = referenced.setdefault(_slot_token(slot), set())
            for record in (entry.get("active"), entry.get("last_green")):
                if record is not None:
                    names.add(f"{int(record['generation']):08d}")
        for slot_dir in sorted(self._generations_dir.iterdir()):
            keep = referenced.get(slot_dir.name, set())
            if slot_dir.is_symlink() or not slot_dir.is_dir():
                with contextlib.suppress(OSError):
                    slot_dir.unlink()
                continue
            for generation_dir in sorted(slot_dir.iterdir()):
                if generation_dir.name not in keep:
                    with contextlib.suppress(OSError):
                        _remove_tree(generation_dir)
            if not keep:
                with contextlib.suppress(OSError):
                    slot_dir.rmdir()
        for directory in (self._root, self._journal_dir):
            for leftover in directory.glob(f".*{_TEMP_SUFFIX}"):
                with contextlib.suppress(OSError):
                    leftover.unlink()

    def _prune_journals(self) -> None:
        try:
            state = self._read_state()
        except ModelActivationError:
            return
        keep: set[str] = set()
        for entry in state["slots"].values():
            for record in (entry.get("active"), entry.get("last_green")):
                if record is not None:
                    keep.add(record["transaction_id"])
            if entry.get("pending_transaction"):
                keep.add(entry["pending_transaction"])
        finished = [
            journal
            for journal in self._iter_journals()
            if journal["status"] in _TERMINAL_STATUSES
            and journal["transaction_id"] not in keep
        ]
        finished.sort(key=lambda item: (item["updated_at"], item["transaction_id"]))
        excess = len(finished) - self._policy.max_journal_history
        for journal in finished[: max(excess, 0)]:
            with self._journal_lock, contextlib.suppress(OSError):
                self._journal_path(journal["transaction_id"]).unlink()

    def _prune_quarantine(self) -> None:
        entries: list[tuple[str, str, Path]] = []
        for child in self._quarantine_dir.iterdir():
            try:
                metadata = _read_json(child / _QUARANTINE_FILENAME, 64 * 1024)
                at = str(metadata.get("quarantined_at") or "")
            except (OSError, ValueError):
                at = ""
            entries.append((at, child.name, child))
        entries.sort()
        excess = len(entries) - self._policy.max_quarantine_entries
        for _, _, child in entries[: max(excess, 0)]:
            with contextlib.suppress(OSError):
                if child.is_dir() and not child.is_symlink():
                    _remove_tree(child)
                else:
                    child.unlink()

    def _quarantine(self, token: str, content: Path, reason: ActivationReason) -> None:
        if not (content.exists() or content.is_symlink()):
            return
        destination = self._quarantine_dir / token
        if destination.exists():
            destination = self._quarantine_dir / f"{token}-{uuid.uuid4().hex[:8]}"
        destination.mkdir(parents=True)
        os.replace(content, destination / _PAYLOAD_DIRNAME)
        _write_json_atomic(
            destination / _QUARANTINE_FILENAME,
            {
                "schema_version": QUARANTINE_SCHEMA_VERSION,
                "entry": token,
                "reason": reason.value,
                "quarantined_at": self._now_text(),
            },
        )
        self._prune_quarantine()

    def _quarantine_journal(self, path: Path) -> None:
        with contextlib.suppress(OSError):
            self._quarantine(
                "journal-" + uuid.uuid4().hex,
                path,
                ActivationReason.JOURNAL_UNREADABLE,
            )
        self._emit_raw(None, "-", "recovery", "quarantined", None, "journal_unreadable")

    # ------------------------------------------------------------------
    # Durable state, journals, receipts
    # ------------------------------------------------------------------

    @property
    def _journal_dir(self) -> Path:
        return self._root / "journal"

    @property
    def _staging_dir(self) -> Path:
        return self._root / "staging"

    @property
    def _generations_dir(self) -> Path:
        return self._root / "generations"

    @property
    def _quarantine_dir(self) -> Path:
        return self._root / "quarantine"

    def _generation_dir(self, slot: str, generation: int) -> Path:
        return self._generations_dir / _slot_token(slot) / f"{int(generation):08d}"

    def _journal_path(self, transaction_id: str) -> Path:
        return self._journal_dir / f"{transaction_id}.json"

    def _read_state(self) -> dict[str, Any]:
        path = self._root / _STATE_FILENAME
        with self._state_lock:
            if not path.exists():
                return {"schema_version": STATE_SCHEMA_VERSION, "slots": {}}
            try:
                payload = _read_json(path, _MAX_STATE_BYTES)
                _validate_state(payload)
            except (OSError, ValueError):
                raise ModelActivationError(ActivationReason.STATE_UNREADABLE) from None
        return payload

    def _write_state(self, state: Mapping[str, Any]) -> None:
        with self._state_lock:
            self._fault("state.write")
            _write_json_atomic(self._root / _STATE_FILENAME, state)

    def _allocate_generation(self, slot: str) -> int:
        with self._state_lock:
            state = self._read_state()
            entry = _slot_entry(state, slot)
            generation = int(entry["next_generation"])
            entry["next_generation"] = generation + 1
            _store_slot_entry(state, slot, entry)
            self._write_state(state)
        self._fault("generation.allocated")
        return generation

    def _restore_entry(self, slot: str, previous: Mapping[str, Any]) -> bool:
        try:
            with self._state_lock:
                state = self._read_state()
                entry = _slot_entry(state, slot)
                restored = copy.deepcopy(dict(previous))
                restored["next_generation"] = max(
                    int(entry["next_generation"]), int(restored["next_generation"])
                )
                _store_slot_entry(state, slot, restored)
                self._write_state(state)
        except (OSError, ModelActivationError):
            return False
        return True

    def _begin(self, journal: dict[str, Any], phase: ActivationPhase) -> None:
        journal["phase"] = phase.value
        journal["phase_completed"] = False
        journal["history"].append(
            {"phase": phase.value, "state": "started", "at": self._now_text()}
        )
        self._write_journal(journal)
        self._emit(journal, phase.value, "started", None)

    def _complete(self, journal: dict[str, Any], phase: ActivationPhase) -> None:
        journal["phase"] = phase.value
        journal["phase_completed"] = True
        journal["history"].append(
            {"phase": phase.value, "state": "completed", "at": self._now_text()}
        )
        self._write_journal(journal)
        self._emit(journal, phase.value, "completed", None)

    def _write_journal(self, journal: dict[str, Any]) -> None:
        journal["updated_at"] = self._now_text()
        with self._journal_lock:
            self._fault("journal.write")
            _write_json_atomic(self._journal_path(journal["transaction_id"]), journal)

    def _read_journal(self, transaction_id: str) -> dict[str, Any] | None:
        path = self._journal_path(transaction_id)
        try:
            with self._journal_lock:
                if not path.is_file():
                    return None
                return self._load_journal_file(path)
        except ValueError:
            return None

    def _load_journal_file(self, path: Path) -> dict[str, Any]:
        try:
            with self._journal_lock:
                payload = _read_json(path, _MAX_JOURNAL_BYTES)
        except OSError as exc:
            raise ValueError("journal unreadable") from exc
        _validate_journal(payload, path.stem)
        return payload

    def _iter_journals(self) -> Iterator[dict[str, Any]]:
        for path in sorted(self._journal_dir.glob("txn-*.json")):
            try:
                yield self._load_journal_file(path)
            except ValueError:
                continue

    def _pending_journal(self, slot: str) -> dict[str, Any] | None:
        entry = _slot_entry(self._read_state(), slot)
        pending = entry["pending_transaction"]
        if pending is None:
            return None
        journal = self._read_journal(pending)
        if journal is None:
            return self._synthetic_journal(slot, entry)
        return journal

    def _load_receipt_tail(self) -> None:
        path = self._root / _RECEIPTS_FILENAME
        self._receipt_sequence = 0
        self._receipt_digest = None
        if not path.is_file():
            return
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            # A torn final append from a crash: drop the partial record.
            keep = data.rfind(b"\n") + 1
            with path.open("r+b") as handle:
                handle.truncate(keep)
                handle.flush()
                os.fsync(handle.fileno())
            data = data[:keep]
        lines = [
            line
            for line in data.decode("ascii", errors="replace").splitlines()
            if line.strip()
        ]
        if not lines:
            return
        try:
            last = json.loads(lines[-1])
            self._receipt_sequence = int(last["sequence"])
            self._receipt_digest = str(last["digest"])
        except (ValueError, KeyError, TypeError):
            # A corrupt log stays detectable through verify_receipt_chain;
            # new receipts keep a monotonic sequence.
            self._receipt_sequence = len(lines)
            self._receipt_digest = None

    def _append_receipt(
        self,
        event: str,
        journal: Mapping[str, Any],
        *,
        from_generation: int | None = None,
        to_generation: int | None = None,
    ) -> None:
        compatibility = journal.get("compatibility") or {}
        with self._receipt_lock:
            body = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "sequence": self._receipt_sequence + 1,
                "event": event,
                "outcome": journal.get("outcome"),
                "reason": journal.get("reason"),
                "transaction_id": journal["transaction_id"],
                "slot": journal["slot"],
                "model_ref_digest": journal.get("model_ref_digest"),
                "artifact_digest": journal.get("artifact_digest"),
                "candidate_generation": journal.get("generation"),
                "from_generation": from_generation,
                "to_generation": to_generation,
                "compatibility_decision": compatibility.get("decision"),
                "compatibility_fingerprint": compatibility.get("fingerprint"),
                "at": self._now_text(),
                "previous_digest": self._receipt_digest,
            }
            body["digest"] = stable_hash(body)
            line = json.dumps(
                body,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            try:
                with (self._root / _RECEIPTS_FILENAME).open(
                    "a", encoding="ascii"
                ) as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError:
                self._emit(journal, "receipt", "failed", None)
                return
            self._receipt_sequence = body["sequence"]
            self._receipt_digest = body["digest"]

    # ------------------------------------------------------------------
    # Results and events
    # ------------------------------------------------------------------

    def _replay(
        self, transaction_id: str, slot: str, ref_digest: str
    ) -> ActivationResult | None:
        journal = self._read_journal(transaction_id)
        if journal is None:
            return None
        if journal["slot"] != slot or journal["model_ref_digest"] != ref_digest:
            return self._result(
                transaction_id,
                slot,
                ActivationOutcome.CONFLICT,
                model_ref_digest=ref_digest,
                reason=ActivationReason.TRANSACTION_CONFLICT,
            )
        status = journal["status"]
        if status == TransactionStatus.IN_PROGRESS.value:
            return self._journal_result(
                journal,
                ActivationOutcome.BUSY,
                reason=ActivationReason.SLOT_BUSY,
                replayed=True,
            )
        if status == TransactionStatus.HEALTH_WINDOW.value:
            return self._journal_result(
                journal, ActivationOutcome.PENDING_HEALTH, replayed=True
            )
        outcome = ActivationOutcome(journal["outcome"])
        return self._journal_result(
            journal,
            outcome,
            reason=_reason_or(journal.get("reason"), None),
            replayed=True,
        )

    def _journal_result(
        self,
        journal: Mapping[str, Any],
        outcome: ActivationOutcome,
        *,
        reason: ActivationReason | None = None,
        replayed: bool = False,
    ) -> ActivationResult:
        health = journal.get("health") or {}
        deadline = (
            health.get("deadline")
            if outcome is ActivationOutcome.PENDING_HEALTH
            else None
        )
        return self._result(
            journal["transaction_id"],
            journal["slot"],
            outcome,
            phase=ActivationPhase(journal["phase"]),
            model_ref_digest=journal.get("model_ref_digest"),
            candidate_generation=journal.get("generation"),
            artifact_digest=journal.get("artifact_digest"),
            reason=reason,
            health_deadline=deadline,
            replayed=replayed,
        )

    def _slot_result(
        self, slot: str, outcome: ActivationOutcome, reason: ActivationReason
    ) -> ActivationResult:
        return self._result(None, slot, outcome, reason=reason)

    def _result(
        self,
        transaction_id: str | None,
        slot: str,
        outcome: ActivationOutcome,
        **fields: Any,
    ) -> ActivationResult:
        try:
            entry = _slot_entry(self._read_state(), slot)
        except ModelActivationError:
            entry = _empty_slot_entry()
        active = entry["active"]
        last_green = entry["last_green"]
        return ActivationResult(
            transaction_id=transaction_id,
            slot=slot,
            outcome=outcome,
            active_generation=None if active is None else active["generation"],
            last_green_generation=(
                None if last_green is None else last_green["generation"]
            ),
            **fields,
        )

    def _emit(
        self,
        journal: Mapping[str, Any],
        phase: str,
        state: str,
        reason: ActivationReason | None,
    ) -> None:
        self._emit_raw(
            journal.get("transaction_id"),
            journal.get("slot", "-"),
            phase,
            state,
            journal.get("generation"),
            None if reason is None else reason.value,
        )

    def _emit_raw(
        self,
        transaction_id: str | None,
        slot: str,
        phase: str,
        state: str,
        generation: int | None,
        reason: str | None,
    ) -> None:
        event = ActivationEvent(
            transaction_id=transaction_id,
            slot=slot,
            phase=phase,
            state=state,
            generation=generation,
            reason=reason,
            at=self._now_text(),
        )
        logger.debug(
            "model activation transaction=%s slot=%s phase=%s state=%s reason=%s",
            transaction_id,
            slot,
            phase,
            state,
            reason,
        )
        if self._event_sink is not None:
            try:
                self._event_sink(event)
            except Exception:
                logger.debug("model activation event sink failed")

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _require_started(self) -> None:
        if not self._started:
            raise ModelActivationError(ActivationReason.NOT_STARTED)

    def _slot_lock(self, slot: str) -> threading.Lock:
        with self._slot_locks_guard:
            lock = self._slot_locks.get(slot)
            if lock is None:
                lock = threading.Lock()
                self._slot_locks[slot] = lock
            return lock

    @contextlib.contextmanager
    def _slot_operation(self, slot: str) -> Iterator[ActivationResult | None]:
        lock = self._slot_lock(slot)
        if not lock.acquire(blocking=False):
            yield self._slot_result(
                slot, ActivationOutcome.BUSY, ActivationReason.SLOT_BUSY
            )
            return
        try:
            yield None
        finally:
            lock.release()

    def _fault(self, point: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(point)

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    def _now_text(self) -> str:
        return _format_time(self._now())


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------


def verify_receipt_chain(path: str | Path) -> bool:
    """Return whether a receipt log is an unbroken, untampered hash chain.

    Args:
        path: Path to a ``receipts.jsonl`` log written by
            :class:`ModelActivationManager`.

    Returns:
        ``True`` when every receipt's digest matches its body, sequences are
        contiguous from 1, and each receipt names its predecessor's digest.
        An absent or empty log is a valid empty chain.
    """

    log = Path(path)
    if not log.is_file():
        return True
    previous: str | None = None
    expected_sequence = 1
    try:
        for line in log.read_text(encoding="ascii").splitlines():
            if not line.strip():
                continue
            body = json.loads(line)
            digest = body.pop("digest")
            if body.get("schema_version") != RECEIPT_SCHEMA_VERSION:
                return False
            if body.get("sequence") != expected_sequence:
                return False
            if body.get("previous_digest") != previous:
                return False
            if stable_hash(body) != digest:
                return False
            previous = digest
            expected_sequence += 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return True


def _normalize_slot(slot: Any) -> str:
    if not isinstance(slot, str):
        raise ModelActivationError(ActivationReason.INVALID_REQUEST)
    normalized = slot.strip().lower()
    if not _SLOT_RE.fullmatch(normalized):
        raise ModelActivationError(ActivationReason.INVALID_REQUEST)
    return normalized


def _slot_token(slot: str) -> str:
    return hashlib.sha256(slot.encode("utf-8")).hexdigest()[:16]


def _transaction_id(slot: str, idempotency_key: str | None) -> str:
    if idempotency_key is None:
        return f"txn-{uuid.uuid4().hex}"
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise ModelActivationError(ActivationReason.INVALID_REQUEST)
    material = f"{slot}\0{idempotency_key}".encode("utf-8")
    return f"txn-{hashlib.sha256(material).hexdigest()[:32]}"


def _checkpoint_payload(
    compatibility: RegistryCheckpoint | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if compatibility is None:
        return None
    if isinstance(compatibility, RegistryCheckpoint) or is_dataclass(compatibility):
        payload: Any = asdict(compatibility)  # type: ignore[arg-type]
    elif isinstance(compatibility, Mapping):
        payload = dict(compatibility)
    else:
        raise _PhaseFailure(ActivationReason.COMPATIBILITY_METADATA_INVALID)
    try:
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError):
        raise _PhaseFailure(ActivationReason.COMPATIBILITY_METADATA_INVALID) from None
    if len(encoded) > _MAX_GENERATION_METADATA_BYTES // 2:
        raise _PhaseFailure(ActivationReason.COMPATIBILITY_METADATA_INVALID)
    return json.loads(encoded)


def _empty_slot_entry() -> dict[str, Any]:
    return {
        "active": None,
        "last_green": None,
        "pending_transaction": None,
        "health_deadline": None,
        "next_generation": 1,
    }


def _slot_entry(state: Mapping[str, Any], slot: str) -> dict[str, Any]:
    entry = state["slots"].get(slot)
    if entry is None:
        return _empty_slot_entry()
    return copy.deepcopy(dict(entry))


def _store_slot_entry(
    state: dict[str, Any], slot: str, entry: Mapping[str, Any]
) -> None:
    state["slots"][slot] = copy.deepcopy(dict(entry))


def _validate_record(record: Any) -> None:
    if record is None:
        return
    if not isinstance(record, Mapping):
        raise ValueError("generation record must be an object")
    generation = record.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ValueError("generation must be an integer")
    if generation < 1:
        raise ValueError("generation must be positive")
    if not _TRANSACTION_RE.fullmatch(str(record.get("transaction_id"))):
        raise ValueError("generation transaction id is invalid")
    for name in ("model_ref_digest", "artifact_digest"):
        if not _DIGEST_RE.fullmatch(str(record.get(name))):
            raise ValueError("generation digest is invalid")


def _validate_state(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("state must be an object")
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("unsupported state schema")
    slots = payload.get("slots")
    if not isinstance(slots, dict):
        raise ValueError("state slots must be an object")
    for slot, entry in slots.items():
        if not isinstance(slot, str) or not _SLOT_RE.fullmatch(slot):
            raise ValueError("state slot is invalid")
        if not isinstance(entry, dict):
            raise ValueError("state slot entry must be an object")
        _validate_record(entry.get("active"))
        _validate_record(entry.get("last_green"))
        pending = entry.get("pending_transaction")
        if pending is not None and not _TRANSACTION_RE.fullmatch(str(pending)):
            raise ValueError("pending transaction is invalid")
        next_generation = entry.get("next_generation")
        if (
            isinstance(next_generation, bool)
            or not isinstance(next_generation, int)
            or next_generation < 1
        ):
            raise ValueError("next generation is invalid")
        for name in ("active", "last_green", "pending_transaction", "health_deadline"):
            entry.setdefault(name, None)


def _validate_journal(payload: Any, expected_id: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError("journal must be an object")
    if payload.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise ValueError("unsupported journal schema")
    if payload.get("transaction_id") != expected_id or not _TRANSACTION_RE.fullmatch(
        expected_id
    ):
        raise ValueError("journal transaction id is invalid")
    if not isinstance(payload.get("slot"), str) or not _SLOT_RE.fullmatch(
        payload["slot"]
    ):
        raise ValueError("journal slot is invalid")
    if not _DIGEST_RE.fullmatch(str(payload.get("model_ref_digest"))):
        raise ValueError("journal model digest is invalid")
    if payload.get("status") not in {status.value for status in TransactionStatus}:
        raise ValueError("journal status is invalid")
    if payload.get("phase") not in {phase.value for phase in ActivationPhase}:
        raise ValueError("journal phase is invalid")
    generation = payload.get("generation")
    if generation is not None and (
        isinstance(generation, bool) or not isinstance(generation, int)
    ):
        raise ValueError("journal generation is invalid")
    if not isinstance(payload.get("history"), list):
        raise ValueError("journal history is invalid")
    for name in ("created_at", "updated_at"):
        if not isinstance(payload.get(name), str):
            raise ValueError("journal timestamp is invalid")
    if payload.get("status") in _TERMINAL_STATUSES and payload.get("outcome") not in {
        outcome.value for outcome in ActivationOutcome
    }:
        raise ValueError("journal outcome is invalid")
    payload.setdefault("phase_completed", False)


def _reason_or(value: Any, default: ActivationReason | None) -> ActivationReason | None:
    try:
        return ActivationReason(value)
    except ValueError:
        return default


def _os_reason(exc: OSError, default: ActivationReason) -> ActivationReason:
    if exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}:
        return ActivationReason.DISK_EXHAUSTED
    return default


_PHASE_FAILURE_REASONS = {
    ActivationPhase.ACQUIRE: ActivationReason.ACQUIRE_FAILED,
    ActivationPhase.STAGE: ActivationReason.STAGE_FAILED,
    ActivationPhase.VERIFY: ActivationReason.VERIFY_FAILED,
    ActivationPhase.LOAD: ActivationReason.LOAD_FAILED,
    ActivationPhase.WARM: ActivationReason.WARM_FAILED,
    ActivationPhase.COMMIT: ActivationReason.COMMIT_FAILED,
    ActivationPhase.RETIRE: ActivationReason.COMMIT_FAILED,
    ActivationPhase.ROLLBACK: ActivationReason.ROLLBACK_FAILED,
}


def _failure_reason(exc: BaseException, phase: ActivationPhase) -> ActivationReason:
    default = _PHASE_FAILURE_REASONS[phase]
    if isinstance(exc, _PhaseFailure):
        return exc.reason
    if isinstance(exc, ModelIntegrityError):
        return _INTEGRITY_REASONS.get(exc.code, ActivationReason.VERIFY_FAILED)
    if isinstance(exc, OSError):
        return _os_reason(exc, default)
    if isinstance(exc, OfflineModeError):
        return ActivationReason.OFFLINE_UNAVAILABLE
    if isinstance(exc, DownloadIntegrityError):
        return ActivationReason.DIGEST_MISMATCH
    return default


def _contained_directory(candidate: Any, workspace: Path) -> Path:
    try:
        root = Path(candidate).resolve(strict=True)
        boundary = workspace.resolve(strict=True)
    except (OSError, TypeError, RuntimeError):
        raise _PhaseFailure(ActivationReason.SOURCE_OUTSIDE_WORKSPACE) from None
    if not root.is_dir() or not root.is_relative_to(boundary):
        raise _PhaseFailure(ActivationReason.SOURCE_OUTSIDE_WORKSPACE)
    return root


def _materialize(
    source_root: Path,
    workspace: Path,
    destination: Path,
    *,
    max_files: int,
    max_bytes: int | None,
) -> None:
    """Move the acquired tree into ``destination``, dereferencing safe links."""

    boundary = workspace.resolve(strict=True)
    destination.mkdir()
    files = 0
    total = 0
    for current, dirnames, filenames in os.walk(source_root, followlinks=False):
        base = Path(current)
        relative = base.relative_to(source_root)
        for name in dirnames:
            if (base / name).is_symlink():
                raise _PhaseFailure(ActivationReason.ARTIFACT_LINK)
            (destination / relative / name).mkdir()
        for name in filenames:
            source = base / name
            target = destination / relative / name
            metadata = source.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    resolved = source.resolve(strict=True)
                except (OSError, RuntimeError):
                    raise _PhaseFailure(ActivationReason.ARTIFACT_LINK) from None
                if not resolved.is_relative_to(boundary) or not resolved.is_file():
                    raise _PhaseFailure(ActivationReason.ARTIFACT_LINK)
                shutil.copyfile(resolved, target)
            elif stat.S_ISREG(metadata.st_mode):
                os.replace(source, target)
            else:
                raise _PhaseFailure(ActivationReason.ARTIFACT_UNSUPPORTED)
            files += 1
            total += target.stat().st_size
            if files > max_files or (max_bytes is not None and total > max_bytes):
                raise _PhaseFailure(ActivationReason.ARTIFACT_TOO_LARGE)
    if files == 0:
        raise _PhaseFailure(ActivationReason.MANIFEST_MISSING)


def _artifact_inventory(root: Path, *, max_files: int) -> dict[str, int]:
    inventory: dict[str, int] = {}
    if root.is_symlink() or not root.is_dir():
        raise _PhaseFailure(ActivationReason.ARTIFACT_MISSING)
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(current)
        for name in dirnames:
            if (base / name).is_symlink():
                raise _PhaseFailure(ActivationReason.ARTIFACT_LINK)
        for name in filenames:
            path = base / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise _PhaseFailure(ActivationReason.ARTIFACT_LINK)
            if not stat.S_ISREG(metadata.st_mode):
                raise _PhaseFailure(ActivationReason.ARTIFACT_UNSUPPORTED)
            inventory[path.relative_to(root).as_posix()] = metadata.st_size
            if len(inventory) > max_files + 1:
                raise _PhaseFailure(ActivationReason.ARTIFACT_TOO_LARGE)
    return inventory


def _manifest_listing(sidecar: Path) -> set[str]:
    try:
        payload = _read_json(sidecar, _MAX_MANIFEST_BYTES)
    except (OSError, ValueError):
        raise _PhaseFailure(ActivationReason.MANIFEST_INVALID) from None
    root_value = payload.get("artifact_root", ".")
    if not isinstance(root_value, str) or PurePosixPath(root_value) != PurePosixPath(
        "."
    ):
        # Staged trees must be self-contained; an external root could point
        # verification at the active cache instead of the staged bytes.
        raise _PhaseFailure(ActivationReason.MANIFEST_INVALID)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise _PhaseFailure(ActivationReason.MANIFEST_INVALID)
    listed: set[str] = set()
    for entry in artifacts:
        value = entry.get("path") if isinstance(entry, Mapping) else None
        if not isinstance(value, str):
            raise _PhaseFailure(ActivationReason.MANIFEST_INVALID)
        path = Path(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise _PhaseFailure(ActivationReason.MANIFEST_INVALID)
        listed.add(path.as_posix())
    return listed


def _verify_tree(
    artifact_dir: Path,
    *,
    max_files: int,
    expected_model_id: str | None = None,
    expected_reproducibility_hash: str | None = None,
) -> tuple[Any, dict[str, int]]:
    sidecar = artifact_dir / ARTIFACT_MANIFEST_FILENAME
    if sidecar.is_symlink() or not sidecar.is_file():
        raise _PhaseFailure(ActivationReason.MANIFEST_MISSING)
    listed = _manifest_listing(sidecar)
    inventory = _artifact_inventory(artifact_dir, max_files=max_files)
    unlisted = set(inventory) - listed - {ARTIFACT_MANIFEST_FILENAME}
    if unlisted:
        raise _PhaseFailure(ActivationReason.ARTIFACT_UNLISTED)
    try:
        result = verify_artifact_manifest(
            sidecar,
            expected_model_id=expected_model_id,
            expected_reproducibility_hash=expected_reproducibility_hash,
        )
    except ModelIntegrityError as exc:
        raise _PhaseFailure(
            _INTEGRITY_REASONS.get(exc.code, ActivationReason.VERIFY_FAILED)
        ) from None
    except (ValueError, KeyError, TypeError):
        raise _PhaseFailure(ActivationReason.MANIFEST_INVALID) from None
    if Path(result.artifact_root).resolve() != artifact_dir.resolve():
        raise _PhaseFailure(ActivationReason.MANIFEST_INVALID)
    return result, inventory


def _read_json(path: Path, max_bytes: int) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("not a regular file")
    if metadata.st_size > max_bytes:
        raise ValueError("file too large")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=_TEMP_SUFFIX
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _format_time(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment
