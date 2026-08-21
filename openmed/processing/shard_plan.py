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
_MAX_FILE_DESCRIPTORS = 10_000
_MAX_PATH_CHARACTERS = 16_384
_MAX_DECLARED_BYTES = (1 << 63) - 1
_MISSING = object()


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

    def __post_init__(self) -> None:
        _validate_digest(self.path_fingerprint, "path_fingerprint")
        _normalize_size(self.size_bytes)

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

    def __post_init__(self) -> None:
        _validate_nonnegative_int(self.shard_id, "shard_id")
        entries = _bounded_tuple(
            self.entries,
            item_type=FileShardEntry,
            field="entries",
        )
        for entry in entries:
            _validate_digest(entry.path_fingerprint, "path_fingerprint")
            _normalize_size(entry.size_bytes)
        entries = tuple(sorted(entries, key=lambda entry: entry.path_fingerprint))
        if len({entry.path_fingerprint for entry in entries}) != len(entries):
            raise ValueError("entries must have unique path fingerprints")
        total_bytes = _normalize_size(self.total_bytes)
        if total_bytes != sum(entry.size_bytes for entry in entries):
            raise ValueError("total_bytes must equal the sum of entry sizes")
        _validate_digest(self.fingerprint, "fingerprint")
        if self.fingerprint != _shard_fingerprint(self.shard_id, entries):
            raise ValueError("fingerprint does not match shard contents")
        object.__setattr__(self, "entries", entries)

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
        validated = FileShard(
            shard_id=self.shard_id,
            entries=self.entries,
            total_bytes=self.total_bytes,
            fingerprint=self.fingerprint,
        )
        return {
            "shard_id": validated.shard_id,
            "file_count": validated.file_count,
            "total_bytes": validated.total_bytes,
            "fingerprint": validated.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class FileShardPlan:
    """Deterministic bounded plan for a collection of local file metadata."""

    limits: ShardLimits
    shards: tuple[FileShard, ...]
    fingerprint: str
    algorithm: str = SHARD_PLAN_ALGORITHM
    schema_version: int = SHARD_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.limits) is not ShardLimits:
            raise TypeError("limits must be a ShardLimits instance")
        limits = ShardLimits(
            max_bytes=self.limits.max_bytes,
            max_files=self.limits.max_files,
        )
        shards = _bounded_tuple(
            self.shards,
            item_type=FileShard,
            field="shards",
        )
        try:
            shards = tuple(
                FileShard(
                    shard_id=shard.shard_id,
                    entries=shard.entries,
                    total_bytes=shard.total_bytes,
                    fingerprint=shard.fingerprint,
                )
                for shard in shards
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("shards contain invalid metadata") from None
        if any(not shard.entries for shard in shards):
            raise ValueError("shards must not be empty")
        if tuple(shard.shard_id for shard in shards) != tuple(range(len(shards))):
            raise ValueError("shard identifiers must be contiguous and ordered")
        if any(
            shard.total_bytes > limits.max_bytes or shard.file_count > limits.max_files
            for shard in shards
        ):
            raise ValueError("shard contents exceed the declared limits")
        path_fingerprints = [
            entry.path_fingerprint for shard in shards for entry in shard.entries
        ]
        if len(path_fingerprints) > _MAX_FILE_DESCRIPTORS:
            raise ValueError("plan exceeds the file descriptor limit")
        if len(path_fingerprints) != len(set(path_fingerprints)):
            raise ValueError("plan contains duplicate path fingerprints")
        if type(self.algorithm) is not str or self.algorithm != SHARD_PLAN_ALGORITHM:
            raise ValueError("algorithm is not supported")
        if (
            type(self.schema_version) is not int
            or self.schema_version != SHARD_PLAN_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is not supported")
        _validate_digest(self.fingerprint, "fingerprint")
        if self.fingerprint != _plan_fingerprint(limits, shards):
            raise ValueError("fingerprint does not match plan contents")
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "shards", shards)

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
        validated = FileShardPlan(
            limits=self.limits,
            shards=self.shards,
            fingerprint=self.fingerprint,
            algorithm=self.algorithm,
            schema_version=self.schema_version,
        )
        return {
            "algorithm": validated.algorithm,
            "file_count": validated.file_count,
            "fingerprint": validated.fingerprint,
            "limits": {
                "max_bytes": validated.limits.max_bytes,
                "max_files": validated.limits.max_files,
            },
            "schema_version": validated.schema_version,
            "shard_count": validated.shard_count,
            "shards": [shard.to_counts_dict() for shard in validated.shards],
            "total_bytes": validated.total_bytes,
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

    try:
        descriptor_iterator = iter(descriptors)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise InvalidFileDescriptorError(
            "File descriptors must be a bounded iterable"
        ) from None

    normalized: list[FileShardEntry] = []
    seen_fingerprints: dict[str, int] = {}
    index = 0
    while True:
        try:
            raw_descriptor = next(descriptor_iterator)
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise InvalidFileDescriptorError(
                f"File descriptor iteration failed at index {index}"
            ) from None
        if index >= _MAX_FILE_DESCRIPTORS:
            raise InvalidFileDescriptorError(
                f"File descriptor limit of {_MAX_FILE_DESCRIPTORS} exceeded"
            )
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
        index += 1

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
    if type(plan) is not FileShardPlan:
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
        if type(limits) is not ShardLimits:
            raise TypeError("limits must be a ShardLimits instance")
        return ShardLimits(max_bytes=limits.max_bytes, max_files=limits.max_files)

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
    if type(raw) is FileDescriptor:
        try:
            return FileDescriptor(raw.path, raw.size_bytes)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise InvalidFileDescriptorError(
                f"File descriptor at index {index} has invalid metadata"
            ) from None

    try:
        if isinstance(raw, Mapping):
            path = _mapping_value(raw, "path")
            sizes = [
                value
                for key in ("size_bytes", "byte_size", "size")
                if (value := _mapping_value(raw, key)) is not _MISSING
            ]
        else:
            path = getattr(raw, "path", _MISSING)
            sizes = [
                value
                for name in ("size_bytes", "byte_size", "size")
                if (value := getattr(raw, name, _MISSING)) is not _MISSING
            ]
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise InvalidFileDescriptorError(
            f"File descriptor at index {index} could not be read safely"
        ) from None

    if path is _MISSING or path is None:
        raise InvalidFileDescriptorError(
            f"File descriptor at index {index} has no path"
        )
    if not sizes or sizes[0] is None:
        raise InvalidFileDescriptorError(
            f"File descriptor at index {index} has no declared size"
        )
    if len(sizes) != 1:
        raise InvalidFileDescriptorError(
            f"File descriptor at index {index} has conflicting size fields"
        )
    try:
        return FileDescriptor(path, sizes[0])
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise InvalidFileDescriptorError(
            f"File descriptor at index {index} has invalid metadata"
        ) from None


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError:
        return _MISSING


def _bounded_tuple(
    values: Any,
    *,
    item_type: type[Any],
    field: str,
) -> tuple[Any, ...]:
    """Snapshot a public iterable without allowing unbounded materialization."""
    try:
        iterator = iter(values)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise TypeError(f"{field} must contain {item_type.__name__} values") from None

    snapshot: list[Any] = []
    for _ in range(_MAX_FILE_DESCRIPTORS + 1):
        try:
            item = next(iterator)
        except StopIteration:
            return tuple(snapshot)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise TypeError(
                f"{field} must contain {item_type.__name__} values"
            ) from None
        if len(snapshot) >= _MAX_FILE_DESCRIPTORS:
            raise ValueError(f"{field} exceed the file descriptor limit")
        if type(item) is not item_type:
            raise TypeError(f"{field} must contain {item_type.__name__} values")
        snapshot.append(item)
    raise AssertionError("bounded iteration must return or raise")


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
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("path must be a string or path-like value") from None
    if type(raw_path) is bytes:
        raise ValueError("path must be text, not bytes")
    if type(raw_path) is not str:
        raise ValueError("path must be a string or path-like value")
    if not raw_path or "\x00" in raw_path or len(raw_path) > _MAX_PATH_CHARACTERS:
        raise ValueError("path must be usable bounded text without NUL characters")

    # Lexical normalization avoids platform-dependent fingerprints and does
    # not access the filesystem.  NFC gives equivalent Unicode spellings one
    # stable identity while preserving case and relative/absolute semantics.
    try:
        portable_path = raw_path.replace("\\", "/")
        normalized = posixpath.normpath(unicodedata.normalize("NFC", portable_path))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("path could not be normalized") from None
    if normalized == "." and portable_path not in (".", "./"):
        raise ValueError("path must contain a usable path value")
    return normalized


def _normalize_size(value: Any) -> int:
    if type(value) is not int or value < 0 or value > _MAX_DECLARED_BYTES:
        raise ValueError("declared size must be a non-negative integer")
    return value


def _validate_positive_limit(value: Any, name: str) -> None:
    maximum = _MAX_FILE_DESCRIPTORS if name == "max_files" else _MAX_DECLARED_BYTES
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_digest(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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
