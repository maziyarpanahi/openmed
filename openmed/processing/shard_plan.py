"""Deterministic, offline planning for batches of local file descriptors.

The planner consumes declared metadata only.  It never resolves, stats, or
opens a path, so callers can create a plan before granting a worker access to
the files themselves.  Paths are used only to derive stable SHA-256
fingerprints; the resulting plan retains fingerprints and byte counts, never
the original path values.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

SHARD_PLAN_SCHEMA_VERSION = 1
SHARD_PLAN_ALGORITHM = "sha256-path-balanced-v1"
_PATH_FINGERPRINT_NAMESPACE = "openmed.processing.shard-plan:path:v1"
_PLAN_FINGERPRINT_NAMESPACE = "openmed.processing.shard-plan:plan:v1"
_SHARD_FINGERPRINT_NAMESPACE = "openmed.processing.shard-plan:shard:v1"


class ShardPlanningError(ValueError):
    """Base error raised when file-shard planning cannot be completed."""


class InvalidFileDescriptorError(ShardPlanningError):
    """Raised when a descriptor is missing safe planning metadata."""


class DuplicateFileDescriptorError(ShardPlanningError):
    """Raised when two descriptors identify the same normalized path."""


class FileTooLargeError(ShardPlanningError):
    """Raised when one declared file cannot fit within the byte limit."""


@dataclass(frozen=True, slots=True, init=False)
class FileDescriptor:
    """Metadata needed to plan one local file without reading it.

    ``path`` is retained on the input descriptor so a caller can later open
    the selected file.  It is deliberately excluded from ``repr``.  The
    planner copies only its normalized fingerprint and declared byte count
    into a :class:`FileShardPlan`.

    ``byte_size`` is accepted as a constructor alias for ``size_bytes`` to
    make descriptors convenient to build from external metadata records.
    """

    path: str = field(repr=False)
    size_bytes: int

    def __init__(
        self,
        path: str | os.PathLike[str],
        size_bytes: int | None = None,
        *,
        byte_size: int | None = None,
    ) -> None:
        if size_bytes is None and byte_size is None:
            raise TypeError("a declared size_bytes value is required")
        if size_bytes is not None and byte_size is not None:
            raise TypeError("provide only one declared file size")

        normalized_path = _normalize_path(path)
        normalized_size = _normalize_size(
            size_bytes if size_bytes is not None else byte_size
        )
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "size_bytes", normalized_size)

    @property
    def byte_size(self) -> int:
        """Return the declared size using the alternate public spelling."""
        return self.size_bytes

    @property
    def path_fingerprint(self) -> str:
        """Return the stable fingerprint for this descriptor's path."""
        return stable_path_fingerprint(self.path)

    def __repr__(self) -> str:
        """Render metadata without exposing the original path."""
        return (
            "FileDescriptor("
            f"path_fingerprint={self.path_fingerprint!r}, "
            f"size_bytes={self.size_bytes})"
        )


@dataclass(frozen=True, slots=True)
class ShardLimits:
    """Hard per-shard limits used by the deterministic planner."""

    max_bytes: int
    max_files: int

    def __post_init__(self) -> None:
        _validate_positive_limit(self.max_bytes, "max_bytes")
        _validate_positive_limit(self.max_files, "max_files")

    @property
    def max_shard_bytes(self) -> int:
        """Return ``max_bytes`` using the planner-oriented spelling."""
        return self.max_bytes

    @property
    def max_files_per_shard(self) -> int:
        """Return ``max_files`` using the planner-oriented spelling."""
        return self.max_files


@dataclass(frozen=True, slots=True)
class FileShardEntry:
    """Safe in-memory membership metadata for one planned file."""

    path_fingerprint: str
    size_bytes: int

    @property
    def fingerprint(self) -> str:
        """Return the path fingerprint using the shorter spelling."""
        return self.path_fingerprint

    @property
    def byte_size(self) -> int:
        """Return the declared size using the alternate public spelling."""
        return self.size_bytes


@dataclass(frozen=True, slots=True)
class FileShard:
    """One bounded shard containing only safe file metadata."""

    shard_id: int
    entries: tuple[FileShardEntry, ...]
    total_bytes: int
    fingerprint: str

    @property
    def files(self) -> tuple[FileShardEntry, ...]:
        """Return shard entries using the file-oriented spelling."""
        return self.entries

    @property
    def file_count(self) -> int:
        """Return the number of files assigned to this shard."""
        return len(self.entries)

    @property
    def byte_count(self) -> int:
        """Return the sum of declared bytes assigned to this shard."""
        return self.total_bytes

    @property
    def size_bytes(self) -> int:
        """Return ``total_bytes`` using the descriptor-oriented spelling."""
        return self.total_bytes

    @property
    def path_fingerprints(self) -> tuple[str, ...]:
        """Return the ordered safe fingerprints assigned to this shard."""
        return tuple(entry.path_fingerprint for entry in self.entries)

    def to_counts_dict(self) -> dict[str, int | str]:
        """Return counts and a digest, without serializing file membership."""
        return {
            "shard_id": self.shard_id,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class FileShardPlan:
    """Deterministic bounded plan for a collection of local file metadata."""

    limits: ShardLimits
    shards: tuple[FileShard, ...]
    fingerprint: str
    algorithm: str = SHARD_PLAN_ALGORITHM
    schema_version: int = SHARD_PLAN_SCHEMA_VERSION

    @property
    def shard_count(self) -> int:
        """Return the number of non-empty shards in the plan."""
        return len(self.shards)

    @property
    def file_count(self) -> int:
        """Return the number of descriptors covered by the plan."""
        return sum(shard.file_count for shard in self.shards)

    @property
    def total_bytes(self) -> int:
        """Return the total declared byte count covered by the plan."""
        return sum(shard.total_bytes for shard in self.shards)

    @property
    def byte_count(self) -> int:
        """Return ``total_bytes`` using the shorter spelling."""
        return self.total_bytes

    def to_dict(self) -> dict[str, Any]:
        """Serialize only counts, limits, and safe digests.

        Individual paths and file membership are intentionally absent.  A
        caller that needs to process files can retain the in-memory entries or
        rebuild the plan from the original descriptors; a durable plan report
        should not become a path inventory.
        """
        return {
            "algorithm": self.algorithm,
            "file_count": self.file_count,
            "fingerprint": self.fingerprint,
            "limits": {
                "max_bytes": self.limits.max_bytes,
                "max_files": self.limits.max_files,
            },
            "schema_version": self.schema_version,
            "shard_count": self.shard_count,
            "shards": [shard.to_counts_dict() for shard in self.shards],
            "total_bytes": self.total_bytes,
        }

    def to_json(self) -> str:
        """Return canonical JSON for counts-only persistence or comparison."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def stable_path_fingerprint(path: str | os.PathLike[str]) -> str:
    """Return a stable SHA-256 fingerprint without touching ``path``.

    Normalization is lexical and platform-independent.  It does not call
    ``resolve``, ``stat``, or any other filesystem operation.
    """
    normalized_path = _normalize_path(path)
    material = f"{_PATH_FINGERPRINT_NAMESPACE}\0{normalized_path}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def plan_file_shards(
    descriptors: Iterable[FileDescriptor | Mapping[str, Any] | Any],
    *,
    limits: ShardLimits | None = None,
    max_bytes: int | None = None,
    max_files: int | None = None,
    max_shard_bytes: int | None = None,
    max_files_per_shard: int | None = None,
) -> FileShardPlan:
    """Build a deterministic balanced plan from declared file metadata.

    Files are ordered by descending declared size and then by path
    fingerprint.  Each file is assigned to the currently least-loaded shard
    that can accept it; a new shard is created only when no existing shard can
    satisfy both limits.  The resulting plan is therefore independent of
    input iteration order while respecting ``max_bytes`` and ``max_files``.

    The convenience keyword aliases ``max_shard_bytes`` and
    ``max_files_per_shard`` mirror the corresponding :class:`ShardLimits`
    properties.  Exactly one limits object or one complete pair of limits
    must be supplied.
    """
    resolved_limits = _resolve_limits(
        limits,
        max_bytes=max_bytes,
        max_files=max_files,
        max_shard_bytes=max_shard_bytes,
        max_files_per_shard=max_files_per_shard,
    )

    normalized: list[FileShardEntry] = []
    seen_fingerprints: dict[str, int] = {}
    for index, raw_descriptor in enumerate(descriptors):
        descriptor = _coerce_descriptor(raw_descriptor, index=index)
        path_fingerprint = descriptor.path_fingerprint
        if path_fingerprint in seen_fingerprints:
            raise DuplicateFileDescriptorError(
                "Duplicate file descriptor at index "
                f"{index}; first seen at index {seen_fingerprints[path_fingerprint]}"
            )
        seen_fingerprints[path_fingerprint] = index

        if descriptor.size_bytes > resolved_limits.max_bytes:
            raise FileTooLargeError(
                f"File descriptor at index {index} exceeds max_bytes"
            )
        normalized.append(
            FileShardEntry(
                path_fingerprint=path_fingerprint,
                size_bytes=descriptor.size_bytes,
            )
        )

    ordered_entries = sorted(
        normalized,
        key=lambda entry: (-entry.size_bytes, entry.path_fingerprint),
    )
    shard_entries: list[list[FileShardEntry]] = []
    shard_totals: list[int] = []
    for entry in ordered_entries:
        candidates = [
            shard_id
            for shard_id, entries in enumerate(shard_entries)
            if len(entries) < resolved_limits.max_files
            and shard_totals[shard_id] + entry.size_bytes <= resolved_limits.max_bytes
        ]
        if candidates:
            shard_id = min(
                candidates,
                key=lambda candidate: (
                    shard_totals[candidate],
                    len(shard_entries[candidate]),
                    candidate,
                ),
            )
        else:
            shard_id = len(shard_entries)
            shard_entries.append([])
            shard_totals.append(0)

        shard_entries[shard_id].append(entry)
        shard_totals[shard_id] += entry.size_bytes

    shards = tuple(
        _build_shard(
            shard_id,
            entries,
            total_bytes=shard_totals[shard_id],
        )
        for shard_id, entries in enumerate(shard_entries)
    )
    return FileShardPlan(
        limits=resolved_limits,
        shards=shards,
        fingerprint=_plan_fingerprint(resolved_limits, shards),
    )


def serialize_shard_plan(plan: FileShardPlan) -> str:
    """Return the canonical counts-only JSON representation of ``plan``."""
    if not isinstance(plan, FileShardPlan):
        raise TypeError("plan must be a FileShardPlan")
    return plan.to_json()


def _resolve_limits(
    limits: ShardLimits | None,
    *,
    max_bytes: int | None,
    max_files: int | None,
    max_shard_bytes: int | None,
    max_files_per_shard: int | None,
) -> ShardLimits:
    supplied_keywords = (
        max_bytes,
        max_files,
        max_shard_bytes,
        max_files_per_shard,
    )
    if limits is not None and any(value is not None for value in supplied_keywords):
        raise TypeError("provide either limits or limit keywords, not both")
    if limits is not None:
        if not isinstance(limits, ShardLimits):
            raise TypeError("limits must be a ShardLimits instance")
        return limits

    if max_bytes is not None and max_shard_bytes is not None:
        raise TypeError("provide only one byte limit keyword")
    if max_files is not None and max_files_per_shard is not None:
        raise TypeError("provide only one file-count limit keyword")

    resolved_max_bytes = max_bytes if max_bytes is not None else max_shard_bytes
    resolved_max_files = max_files if max_files is not None else max_files_per_shard
    if resolved_max_bytes is None or resolved_max_files is None:
        raise TypeError("both max_bytes and max_files limits are required")
    return ShardLimits(
        max_bytes=resolved_max_bytes,
        max_files=resolved_max_files,
    )


def _coerce_descriptor(raw: Any, *, index: int) -> FileDescriptor:
    if isinstance(raw, FileDescriptor):
        return raw

    if isinstance(raw, Mapping):
        path = raw.get("path")
        if "size_bytes" in raw:
            size = raw["size_bytes"]
        elif "byte_size" in raw:
            size = raw["byte_size"]
        elif "size" in raw:
            size = raw["size"]
        else:
            raise InvalidFileDescriptorError(
                f"File descriptor at index {index} has no declared size"
            )
    else:
        path = getattr(raw, "path", None)
        size = getattr(raw, "size_bytes", None)
        if size is None:
            size = getattr(raw, "byte_size", None)
        if size is None:
            size = getattr(raw, "size", None)

    if path is None:
        raise InvalidFileDescriptorError(
            f"File descriptor at index {index} has no path"
        )
    if size is None:
        raise InvalidFileDescriptorError(
            f"File descriptor at index {index} has no declared size"
        )
    try:
        return FileDescriptor(path, size)
    except (TypeError, ValueError) as exc:
        raise InvalidFileDescriptorError(
            f"File descriptor at index {index} has invalid metadata"
        ) from exc


def _build_shard(
    shard_id: int,
    entries: list[FileShardEntry],
    *,
    total_bytes: int,
) -> FileShard:
    ordered_entries = tuple(sorted(entries, key=lambda entry: entry.path_fingerprint))
    fingerprint = _shard_fingerprint(shard_id, ordered_entries)
    return FileShard(
        shard_id=shard_id,
        entries=ordered_entries,
        total_bytes=total_bytes,
        fingerprint=fingerprint,
    )


def _shard_fingerprint(
    shard_id: int,
    entries: tuple[FileShardEntry, ...],
) -> str:
    payload = {
        "namespace": _SHARD_FINGERPRINT_NAMESPACE,
        "shard_id": shard_id,
        "entries": [
            {
                "path_fingerprint": entry.path_fingerprint,
                "size_bytes": entry.size_bytes,
            }
            for entry in entries
        ],
    }
    return _digest(_canonical_json(payload))


def _plan_fingerprint(limits: ShardLimits, shards: tuple[FileShard, ...]) -> str:
    payload = {
        "namespace": _PLAN_FINGERPRINT_NAMESPACE,
        "limits": {
            "max_bytes": limits.max_bytes,
            "max_files": limits.max_files,
        },
        "shards": [
            {
                "fingerprint": shard.fingerprint,
                "shard_id": shard.shard_id,
            }
            for shard in shards
        ],
    }
    return _digest(_canonical_json(payload))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_path(path: str | os.PathLike[str]) -> str:
    try:
        raw_path = os.fspath(path)
    except TypeError as exc:
        raise ValueError("path must be a string or path-like value") from exc
    if isinstance(raw_path, bytes):
        raise ValueError("path must be text, not bytes")
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError("path must be non-empty text without NUL characters")

    # Lexical normalization avoids platform-dependent fingerprints and does
    # not access the filesystem.  NFC gives equivalent Unicode spellings one
    # stable identity while preserving case and relative/absolute semantics.
    portable_path = raw_path.replace("\\", "/")
    normalized = posixpath.normpath(unicodedata.normalize("NFC", portable_path))
    if normalized == "." and portable_path not in (".", "./"):
        raise ValueError("path must contain a usable path value")
    return normalized


def _normalize_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("declared size must be a non-negative integer")
    return value


def _validate_positive_limit(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


# These aliases keep the public surface discoverable for callers that phrase
# the operation as building a plan rather than planning file shards.
build_file_shard_plan = plan_file_shards
build_shard_plan = plan_file_shards
plan_shards = plan_file_shards


__all__ = [
    "DuplicateFileDescriptorError",
    "FileDescriptor",
    "FileShard",
    "FileShardEntry",
    "FileShardPlan",
    "FileTooLargeError",
    "InvalidFileDescriptorError",
    "SHARD_PLAN_ALGORITHM",
    "SHARD_PLAN_SCHEMA_VERSION",
    "ShardLimits",
    "ShardPlanningError",
    "build_file_shard_plan",
    "build_shard_plan",
    "plan_file_shards",
    "plan_shards",
    "serialize_shard_plan",
    "stable_path_fingerprint",
]
