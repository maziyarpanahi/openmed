"""Durable run manifest for distributed batch de-identification runs.

The manifest records shard execution state: status, attempt counts, timings,
relative output paths and ``sha256:`` output digests. It has no field for
document text or document identifiers, and the failure field stores only an
exception *type* name rather than a message, because messages routinely echo
record content.

That last guard is a syntactic shape check, not a content guarantee: a dotted
identifier of bounded length is accepted, so a caller determined to smuggle a
value through ``error_type`` still can. It exists to turn the realistic leak --
passing ``str(exc)`` where ``type(exc).__name__`` was meant -- into a hard
construction error. ``worker_id`` and ``run_id`` are likewise operator-supplied
free text that is written verbatim; callers must not derive them from records.

Durability relies on the ``os.replace`` semantics of
:func:`openmed.processing.batch._atomic_write_bytes`: a same-directory
temporary file is written, flushed, ``fsync``-ed, renamed over the target, and
the containing directory is ``fsync``-ed. Those semantics hold on local POSIX
filesystems. They are *not* guaranteed on network filesystems such as NFS or
FUSE-backed object-storage mounts, where rename is not atomic. Point the store
at a local path on those deployments and publish the finished manifest
separately.

This module is a record and a validator only. It never executes shards and
never increments :attr:`ShardRecord.attempts`; executors own that.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol, runtime_checkable

from openmed.__about__ import __version__
from openmed.core.model_integrity import sha256_file
from openmed.processing.batch import AtomicWriteHook, _atomic_write_bytes
from openmed.processing.distributed import SHARDING_ALGORITHM, ShardPlan

RUN_MANIFEST_SCHEMA_VERSION = 1

MAX_LABEL_LENGTH = 128

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_ERROR_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class ShardStatus(str, Enum):
    """Lifecycle state of a single shard within a batch run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunManifestError(ValueError):
    """Base error raised when a run manifest cannot be used safely."""


class ManifestSchemaError(RunManifestError):
    """Raised when manifest JSON is malformed or of an unsupported version."""


class UnknownShardError(RunManifestError, KeyError):
    """Raised when a shard id is not present in a manifest.

    Derives from both :class:`RunManifestError` and :class:`KeyError` so that
    callers written against either hierarchy keep working.
    """

    def __str__(self) -> str:
        """Return the plain message, not :class:`KeyError`'s quoted form."""

        return ", ".join(str(arg) for arg in self.args)


class ShardOutputMissingError(RunManifestError):
    """Raised when a completed shard's recorded output cannot be read."""


class ShardOutputDigestMismatchError(RunManifestError):
    """Raised when a completed shard's output no longer matches its digest."""


@dataclass(frozen=True)
class ShardRecord:
    """Execution record for one deterministic shard, free of record content."""

    shard_id: int
    fingerprint: str
    document_count: int
    status: ShardStatus = ShardStatus.PENDING
    attempts: int = 0
    started_at: float | None = None
    completed_at: float | None = None
    output_path: str | None = None
    output_digest: str | None = None
    output_bytes: int | None = None
    worker_id: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ShardStatus(self.status))

        _require_integer(self.shard_id, "shard_id", minimum=0)
        _require_integer(self.document_count, "document_count", minimum=0)
        _require_integer(self.attempts, "attempts", minimum=0)

        _require_fingerprint(self.fingerprint, "shard fingerprint")

        for field_name in ("started_at", "completed_at"):
            value = getattr(self, field_name)
            if value is not None:
                _require_finite_float(value, field_name, minimum=0.0)
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise RunManifestError("completed_at must not precede started_at")

        if self.output_bytes is not None:
            _require_integer(self.output_bytes, "output_bytes", minimum=0)

        if self.output_path is not None:
            _validate_output_path(self.output_path)

        if self.output_digest is not None:
            if not isinstance(self.output_digest, str):
                raise RunManifestError("output_digest must be a string")
            if not _DIGEST_PATTERN.fullmatch(self.output_digest):
                raise RunManifestError(
                    f"shard {self.shard_id} output_digest must be a sha256: digest"
                )

        if self.worker_id is not None:
            _require_label(self.worker_id, "worker_id")

        if self.error_type is not None:
            _require_label(self.error_type, "error_type")
            if not _ERROR_TYPE_PATTERN.fullmatch(self.error_type):
                raise RunManifestError(
                    "error_type must be a bare exception type name such as "
                    "'TimeoutError'; pass type(exc).__name__, never str(exc)"
                )

        if self.status is ShardStatus.COMPLETED and (
            self.output_path is None or self.output_digest is None
        ):
            raise RunManifestError(
                f"completed shard {self.shard_id} requires an output path and digest"
            )

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration of the last attempt, when both timings exist."""

        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation free of record content."""

        return {
            "shard_id": self.shard_id,
            "fingerprint": self.fingerprint,
            "document_count": self.document_count,
            "status": self.status.value,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output_path": self.output_path,
            "output_digest": self.output_digest,
            "output_bytes": self.output_bytes,
            "worker_id": self.worker_id,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShardRecord":
        """Build a shard record from a JSON object, rejecting foreign types.

        Every field is type-checked rather than coerced: this is the boundary
        where bytes written by another process -- including the non-Python
        writers a Ray or Spark adapter may use -- enter the manifest, so a
        coercion here would launder arbitrary content back onto disk.
        """

        try:
            return cls(
                shard_id=_required_integer(payload, "shard_id"),
                fingerprint=_required_str(payload, "fingerprint"),
                document_count=_required_integer(payload, "document_count"),
                status=ShardStatus(_required_str(payload, "status")),
                attempts=_optional_integer(payload.get("attempts", 0), "attempts"),
                started_at=_optional_float(payload.get("started_at"), "started_at"),
                completed_at=_optional_float(
                    payload.get("completed_at"), "completed_at"
                ),
                output_path=_optional_str(payload.get("output_path"), "output_path"),
                output_digest=_optional_str(
                    payload.get("output_digest"), "output_digest"
                ),
                output_bytes=_optional_integer(
                    payload.get("output_bytes"), "output_bytes"
                ),
                worker_id=_optional_str(payload.get("worker_id"), "worker_id"),
                error_type=_optional_str(payload.get("error_type"), "error_type"),
            )
        except KeyError as error:
            raise ManifestSchemaError(f"shard record is missing {error}") from error
        except ValueError as error:
            if isinstance(error, RunManifestError):
                raise
            raise ManifestSchemaError(f"shard record is invalid: {error}") from error


@dataclass(frozen=True)
class BatchRunManifest:
    """Durable state of one distributed batch run, free of record content."""

    run_id: str
    created_at: float
    updated_at: float
    algorithm: str
    shard_count: int
    document_count: int
    plan_fingerprint: str
    openmed_version: str
    shards: tuple[ShardRecord, ...]
    schema_version: int = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.shards, (str, bytes)) or not isinstance(
            self.shards, (tuple, list)
        ):
            raise RunManifestError("shards must be a sequence of shard records")
        # Coerce so a caller-supplied list cannot be mutated behind the frozen
        # dataclass and break the shard_count invariant checked below.
        object.__setattr__(self, "shards", tuple(self.shards))
        if any(not isinstance(record, ShardRecord) for record in self.shards):
            raise RunManifestError("shards must contain ShardRecord entries")

        for field_name in (
            "run_id",
            "algorithm",
            "plan_fingerprint",
            "openmed_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise RunManifestError(f"{field_name} must be a string")
        if not self.run_id.strip():
            raise RunManifestError("run_id must be non-empty")
        # run_id is operator-supplied and flows verbatim into reports and CLI
        # output, so it carries the same bounds as the other free-text labels.
        _require_label(self.run_id, "run_id")
        _require_fingerprint(self.plan_fingerprint, "plan_fingerprint")
        if self.algorithm != SHARDING_ALGORITHM:
            raise RunManifestError(
                f"algorithm must be {SHARDING_ALGORITHM!r} for schema version "
                f"{RUN_MANIFEST_SCHEMA_VERSION}"
            )

        for field_name in ("created_at", "updated_at"):
            _require_finite_float(getattr(self, field_name), field_name, minimum=0.0)

        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise ManifestSchemaError(
                "unsupported run manifest schema version "
                f"{self.schema_version}; expected {RUN_MANIFEST_SCHEMA_VERSION}"
            )
        _require_integer(self.shard_count, "shard_count", minimum=1)
        _require_integer(self.document_count, "document_count", minimum=0)
        if len(self.shards) != self.shard_count:
            raise RunManifestError(
                f"manifest lists {len(self.shards)} shards for "
                f"shard_count {self.shard_count}"
            )

        shard_ids = [record.shard_id for record in self.shards]
        if shard_ids != sorted(shard_ids) or len(set(shard_ids)) != len(shard_ids):
            raise RunManifestError("shard records must be unique and ordered by id")
        if shard_ids and shard_ids[-1] >= self.shard_count:
            raise RunManifestError("shard ids must be below shard_count")

    def shard(self, shard_id: int) -> ShardRecord:
        """Return the record for ``shard_id``."""

        for record in self.shards:
            if record.shard_id == shard_id:
                return record
        raise UnknownShardError(f"unknown shard id: {shard_id}")

    def pending_shards(self) -> tuple[ShardRecord, ...]:
        """Return every shard record that has not completed."""

        return tuple(
            record
            for record in self.shards
            if record.status is not ShardStatus.COMPLETED
        )

    def completed_shards(self) -> tuple[ShardRecord, ...]:
        """Return every shard record marked completed."""

        return tuple(
            record for record in self.shards if record.status is ShardStatus.COMPLETED
        )

    def with_shard(
        self,
        record: ShardRecord,
        *,
        updated_at: float | None = None,
    ) -> "BatchRunManifest":
        """Return a copy of this manifest with ``record`` replacing its shard.

        The replacement must keep the shard's identity: ``shard_id``,
        ``fingerprint`` and ``document_count`` are the manifest's binding to
        the shard plan it was built from, so a resumed run can prove the corpus
        has not changed underneath it. Only execution state may move.
        """

        existing = self.shard(record.shard_id)
        if record.fingerprint != existing.fingerprint:
            raise RunManifestError(
                f"shard {record.shard_id} fingerprint must not change; "
                "the manifest is bound to its shard plan"
            )
        if record.document_count != existing.document_count:
            raise RunManifestError(
                f"shard {record.shard_id} document_count must not change; "
                "the manifest is bound to its shard plan"
            )

        shards = tuple(
            record if candidate.shard_id == record.shard_id else candidate
            for candidate in self.shards
        )
        return replace(
            self,
            shards=shards,
            updated_at=time.time() if updated_at is None else updated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation free of record content."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "algorithm": self.algorithm,
            "shard_count": self.shard_count,
            "document_count": self.document_count,
            "plan_fingerprint": self.plan_fingerprint,
            "openmed_version": self.openmed_version,
            "shards": [record.to_dict() for record in self.shards],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BatchRunManifest":
        """Build a manifest from a JSON object, validating its schema version.

        Like :meth:`ShardRecord.from_dict`, every field is type-checked rather
        than coerced.
        """

        if not isinstance(payload, Mapping):
            raise ManifestSchemaError("run manifest must contain a JSON object")

        version = payload.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ManifestSchemaError("run manifest schema_version must be an integer")
        if version != RUN_MANIFEST_SCHEMA_VERSION:
            raise ManifestSchemaError(
                "unsupported run manifest schema version "
                f"{version}; expected {RUN_MANIFEST_SCHEMA_VERSION}"
            )

        raw_shards = payload.get("shards")
        if not isinstance(raw_shards, list):
            raise ManifestSchemaError("run manifest shards must be a list")

        try:
            return cls(
                run_id=_required_str(payload, "run_id"),
                created_at=_required_float(payload, "created_at"),
                updated_at=_required_float(payload, "updated_at"),
                algorithm=_required_str(payload, "algorithm"),
                shard_count=_required_integer(payload, "shard_count"),
                document_count=_required_integer(payload, "document_count"),
                plan_fingerprint=_required_str(payload, "plan_fingerprint"),
                openmed_version=_required_str(payload, "openmed_version"),
                shards=tuple(
                    ShardRecord.from_dict(_mapping(entry, "shard"))
                    for entry in raw_shards
                ),
                schema_version=version,
            )
        except KeyError as error:
            if isinstance(error, RunManifestError):
                raise
            raise ManifestSchemaError(f"run manifest is missing {error}") from error
        except (TypeError, ValueError) as error:
            if isinstance(error, RunManifestError):
                raise
            raise ManifestSchemaError(f"run manifest is invalid: {error}") from error


@dataclass(frozen=True)
class ShardOutputValidation:
    """Outcome of re-checking the recorded outputs of completed shards."""

    valid: tuple[int, ...] = ()
    missing: tuple[int, ...] = ()
    mismatched: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("valid", "missing", "mismatched"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    @property
    def all_valid(self) -> bool:
        """Whether every completed shard still has a matching output."""

        return not self.missing and not self.mismatched


@runtime_checkable
class RunManifestStore(Protocol):
    """Minimal durable store for a batch run manifest."""

    def load(self) -> BatchRunManifest | None:
        """Return the persisted manifest, or ``None`` when absent."""

    def save(self, manifest: BatchRunManifest) -> None:
        """Persist ``manifest`` durably."""


class InMemoryRunManifestStore:
    """Manifest store useful for tests and embedded runtimes."""

    def __init__(self, manifest: BatchRunManifest | None = None) -> None:
        self._manifest = manifest

    def load(self) -> BatchRunManifest | None:
        """Return the stored manifest, or ``None`` when absent."""

        return self._manifest

    def save(self, manifest: BatchRunManifest) -> None:
        """Store ``manifest`` in memory."""

        self._manifest = manifest


class LocalFileRunManifestStore:
    """Crash-safe JSON manifest store for local filesystems.

    Writes go through the same temp-file, ``fsync``, ``os.replace`` and
    directory-``fsync`` sequence used by durable batch checkpoints, so a reader
    always observes either the previous manifest or the new one.

    Every read failure that is not "file absent" surfaces as a
    :class:`RunManifestError`, so a caller can recover from a corrupted or
    unreadable manifest with a single ``except`` clause.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        atomic_write_hook: AtomicWriteHook | None = None,
    ) -> None:
        self.path = Path(path)
        self.atomic_write_hook = atomic_write_hook

    def load(self) -> BatchRunManifest | None:
        """Return the persisted manifest, or ``None`` when the file is absent."""

        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeDecodeError as error:
            raise ManifestSchemaError(
                f"run manifest is not valid UTF-8: {self.path}"
            ) from error
        except OSError as error:
            raise RunManifestError(
                f"run manifest is not readable: {self.path}"
            ) from error
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ManifestSchemaError(
                f"run manifest is not valid JSON: {self.path}"
            ) from error
        if not isinstance(payload, Mapping):
            raise ManifestSchemaError("run manifest must contain a JSON object")
        return BatchRunManifest.from_dict(payload)

    def save(self, manifest: BatchRunManifest) -> None:
        """Persist ``manifest`` with temp-file plus ``os.replace`` semantics."""

        payload = (
            json.dumps(
                manifest.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _atomic_write_bytes(self.path, payload, hook=self.atomic_write_hook)


def build_run_manifest(
    plan: ShardPlan,
    *,
    run_id: str,
    created_at: float | None = None,
) -> BatchRunManifest:
    """Build a pending manifest for ``plan``.

    ``run_id`` is operator-supplied and is written verbatim, so it must not
    embed document identifiers or any other record-derived value.
    """

    timestamp = time.time() if created_at is None else created_at
    return BatchRunManifest(
        run_id=run_id,
        created_at=timestamp,
        updated_at=timestamp,
        algorithm=plan.algorithm,
        shard_count=plan.shard_count,
        document_count=plan.document_count,
        plan_fingerprint=plan.fingerprint,
        openmed_version=__version__,
        shards=tuple(
            ShardRecord(
                shard_id=shard.shard_id,
                fingerprint=shard.fingerprint,
                document_count=shard.document_count,
            )
            for shard in plan.shards
        ),
    )


def shard_output_digest(path: str | Path) -> str:
    """Return the ``sha256:`` digest recorded for a shard output file."""

    return sha256_file(path)


def validate_shard_outputs(
    manifest: BatchRunManifest,
    *,
    root: str | Path | None = None,
    strict: bool = False,
) -> ShardOutputValidation:
    """Re-check the recorded outputs of every completed shard.

    Output paths are stored relative to the run root; ``root`` supplies the
    directory they resolve against. A path that resolves outside that root --
    through a symlink, for example -- is reported as missing rather than read,
    so the containment the relative-path rule promises is enforced against the
    filesystem and not only against the recorded string.

    With ``strict`` the first missing or mismatched output raises instead of
    being reported.
    """

    base = Path() if root is None else Path(root)
    resolved_root = base.resolve()
    valid: list[int] = []
    missing: list[int] = []
    mismatched: list[int] = []

    for record in manifest.completed_shards():
        output_path = record.output_path
        if output_path is None:  # pragma: no cover - refused by ShardRecord
            raise RunManifestError(
                f"completed shard {record.shard_id} has no recorded output"
            )

        candidate = base / output_path
        resolved = candidate.resolve()
        if not resolved.is_relative_to(resolved_root):
            if strict:
                raise ShardOutputMissingError(
                    f"shard {record.shard_id} output resolves outside the run root"
                )
            missing.append(record.shard_id)
            continue
        if not candidate.is_file():
            if strict:
                raise ShardOutputMissingError(
                    f"shard {record.shard_id} output is missing"
                )
            missing.append(record.shard_id)
            continue
        if shard_output_digest(candidate) != record.output_digest:
            if strict:
                raise ShardOutputDigestMismatchError(
                    f"shard {record.shard_id} output digest does not match"
                )
            mismatched.append(record.shard_id)
            continue
        valid.append(record.shard_id)

    return ShardOutputValidation(
        valid=tuple(valid),
        missing=tuple(missing),
        mismatched=tuple(mismatched),
    )


def shards_to_execute(
    manifest: BatchRunManifest,
    *,
    root: str | Path | None = None,
) -> tuple[int, ...]:
    """Return the shard ids a resumed run still has to execute.

    Completed shards are skipped only while their recorded output digest still
    matches; missing or corrupted outputs are queued for re-execution.

    Every shard that is not completed is returned, which deliberately includes
    shards left in :attr:`ShardStatus.RUNNING` by a crashed worker and shards
    with no documents. Deciding whether a ``RUNNING`` shard is a straggler
    still making progress or an orphan to re-dispatch needs liveness
    information this module does not have; that policy belongs to the executor.
    Empty shards can be filtered with ``ShardPlan.non_empty_shards``.
    """

    validation = validate_shard_outputs(manifest, root=root)
    invalid = set(validation.missing) | set(validation.mismatched)
    return tuple(
        sorted(
            record.shard_id
            for record in manifest.shards
            if record.status is not ShardStatus.COMPLETED or record.shard_id in invalid
        )
    )


def _validate_output_path(output_path: str) -> None:
    if not isinstance(output_path, str):
        raise RunManifestError("output_path must be a string")
    if not output_path.strip():
        raise RunManifestError("output_path must be non-empty")
    if _CONTROL_CHARACTERS.search(output_path):
        raise RunManifestError("output_path must not contain control characters")
    pure = PurePosixPath(output_path)
    windows = PureWindowsPath(output_path)
    if (
        pure.is_absolute()
        or windows.anchor
        or windows.drive
        or Path(output_path).is_absolute()
    ):
        raise RunManifestError("output_path must be relative to the run root")
    if ".." in pure.parts or ".." in windows.parts:
        raise RunManifestError("output_path must not escape the run root")


def _require_fingerprint(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise RunManifestError(f"{field_name} must be a string")
    if not _FINGERPRINT_PATTERN.fullmatch(value):
        raise RunManifestError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_label(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise RunManifestError(f"{field_name} must be a string")
    if len(value) > MAX_LABEL_LENGTH:
        raise RunManifestError(
            f"{field_name} must be at most {MAX_LABEL_LENGTH} characters"
        )
    if _CONTROL_CHARACTERS.search(value):
        raise RunManifestError(f"{field_name} must not contain control characters")
    return value


def _require_integer(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunManifestError(f"{field_name} must be an integer")
    if value < minimum:
        raise RunManifestError(f"{field_name} must be at least {minimum}")
    return value


def _require_finite_float(value: Any, field_name: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunManifestError(f"{field_name} must be a number")
    if not math.isfinite(value):
        raise RunManifestError(f"{field_name} must be finite")
    if value < minimum:
        raise RunManifestError(f"{field_name} must be at least {minimum}")
    return float(value)


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestSchemaError(f"{field_name} must be an integer")
    return value


def _required_integer(payload: Mapping[str, Any], field_name: str) -> int:
    return _integer(payload[field_name], field_name)


def _optional_integer(value: Any, field_name: str) -> int | None:
    return None if value is None else _integer(value, field_name)


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestSchemaError(f"{field_name} must be a number")
    if not math.isfinite(value):
        raise ManifestSchemaError(f"{field_name} must be finite")
    return float(value)


def _required_float(payload: Mapping[str, Any], field_name: str) -> float:
    return _float(payload[field_name], field_name)


def _optional_float(value: Any, field_name: str) -> float | None:
    return None if value is None else _float(value, field_name)


def _str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ManifestSchemaError(f"{field_name} must be a string")
    return value


def _required_str(payload: Mapping[str, Any], field_name: str) -> str:
    return _str(payload[field_name], field_name)


def _optional_str(value: Any, field_name: str) -> str | None:
    return None if value is None else _str(value, field_name)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestSchemaError(f"{field_name} must be a mapping")
    return value


__all__ = [
    "MAX_LABEL_LENGTH",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "BatchRunManifest",
    "InMemoryRunManifestStore",
    "LocalFileRunManifestStore",
    "ManifestSchemaError",
    "RunManifestError",
    "RunManifestStore",
    "ShardOutputDigestMismatchError",
    "ShardOutputMissingError",
    "ShardOutputValidation",
    "ShardRecord",
    "ShardStatus",
    "UnknownShardError",
    "build_run_manifest",
    "shard_output_digest",
    "shards_to_execute",
    "validate_shard_outputs",
]
