"""Deterministic, offline inventory reports for local artifact files.

The inventory records file metadata and SHA-256 fingerprints without retaining
or rendering file contents.  Paths are normalized relative to a caller-
supplied root (or a safe local root inferred for convenience), sorted before
being emitted, and checked for traversal, duplicates, symlink escapes, and
read failures.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Final

SCHEMA_VERSION: Final = "openmed.interop.artifact_inventory.v1"
_CHUNK_SIZE: Final = 1024 * 1024
_FINGERPRINT_PREFIX: Final = "sha256:"
_MAX_ARTIFACTS: Final = 10_000
_MAX_JSON_INDENT: Final = 8

# A small explicit mapping keeps media type output independent of the host's
# optional ``mimetypes`` database.  Unknown extensions are deliberately
# treated as opaque binary artifacts.
_MEDIA_TYPES: Final = {
    ".csv": "text/csv",
    ".gif": "image/gif",
    ".gz": "application/gzip",
    ".html": "text/html",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".md": "text/markdown",
    ".ndjson": "application/x-ndjson",
    ".onnx": "application/octet-stream",
    ".parquet": "application/vnd.apache.parquet",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".safetensors": "application/octet-stream",
    ".tar": "application/x-tar",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".webp": "image/webp",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zip": "application/zip",
}
_SUPPORTED_MEDIA_TYPES: Final = frozenset(
    {*_MEDIA_TYPES.values(), "application/octet-stream"}
)


class ArtifactInventoryError(ValueError):
    """Base error for invalid or unreadable artifact inventory inputs."""


class ArtifactPathError(ArtifactInventoryError):
    """Raised when an artifact path is unsafe or cannot be normalized."""


class DuplicateArtifactError(ArtifactInventoryError):
    """Raised when two inputs normalize to the same artifact path."""


class UnreadableArtifactError(ArtifactInventoryError):
    """Raised when an artifact root or file cannot be read safely."""


@dataclass(frozen=True, slots=True)
class ArtifactInventoryEntry:
    """PHI-safe metadata for one local artifact.

    ``path`` is a normalized POSIX path relative to the inventory root.  The
    file contents are never stored on the entry; ``fingerprint`` is a
    content-addressed SHA-256 digest instead.
    """

    path: str
    byte_count: int
    media_type: str
    fingerprint: str

    def __post_init__(self) -> None:
        """Validate the serialized metadata without echoing caller values."""

        if type(self.path) is not str or not self.path or self.path.startswith("/"):
            raise ArtifactPathError("artifact paths must be non-empty and relative")
        if "\\" in self.path or any(ord(character) < 32 for character in self.path):
            raise ArtifactPathError("artifact paths must use safe relative separators")
        if any(part in {"", ".", ".."} for part in self.path.split("/")):
            raise ArtifactPathError("artifact paths must be normalized")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ArtifactInventoryError("artifact byte counts must be non-negative")
        if (
            type(self.media_type) is not str
            or self.media_type not in _SUPPORTED_MEDIA_TYPES
        ):
            raise ArtifactInventoryError("artifact media type is unsupported")
        if type(self.fingerprint) is not str or not self.fingerprint.startswith(
            _FINGERPRINT_PREFIX
        ):
            raise ArtifactInventoryError("artifact fingerprints must use SHA-256")
        digest = self.fingerprint.removeprefix(_FINGERPRINT_PREFIX)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ArtifactInventoryError(
                "artifact fingerprints must use lowercase SHA-256"
            )

    @property
    def bytes(self) -> int:
        """Return the byte count under the concise ``bytes`` name."""

        return self.byte_count

    @property
    def size_bytes(self) -> int:
        """Return the byte count under the descriptive ``size_bytes`` name."""

        return self.byte_count

    @property
    def content_fingerprint(self) -> str:
        """Return the content fingerprint under its descriptive name."""

        return self.fingerprint

    @property
    def sha256(self) -> str:
        """Return the SHA-256 fingerprint under its algorithm-specific name."""

        return self.fingerprint

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-ready metadata mapping with no file contents."""

        return {
            "byte_count": self.byte_count,
            "fingerprint": self.fingerprint,
            "media_type": self.media_type,
            "path": self.path,
        }


# ``ArtifactRecord`` is a useful compatibility spelling for callers that use
# record terminology for a manifest entry.
ArtifactRecord = ArtifactInventoryEntry


@dataclass(frozen=True, slots=True)
class ArtifactInventory:
    """Deterministically ordered metadata for a set of local artifacts."""

    entries: tuple[ArtifactInventoryEntry, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize direct construction to the same stable order as indexing."""

        if (
            type(self.schema_version) is not str
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ArtifactInventoryError("unsupported artifact inventory schema")
        try:
            entries = tuple(islice(iter(self.entries), _MAX_ARTIFACTS + 1))
        except Exception:
            raise TypeError(
                "inventory entries must be an iterable of artifact entries"
            ) from None
        if len(entries) > _MAX_ARTIFACTS:
            raise ArtifactInventoryError(
                "artifact inventory exceeds the supported entry limit"
            )
        if not all(type(entry) is ArtifactInventoryEntry for entry in entries):
            raise TypeError("inventory entries must be ArtifactInventoryEntry values")
        paths = [entry.path for entry in entries]
        if len(paths) != len(set(paths)):
            raise DuplicateArtifactError("inventory entries contain duplicate paths")
        object.__setattr__(
            self, "entries", tuple(sorted(entries, key=lambda item: item.path))
        )

    @property
    def artifacts(self) -> tuple[ArtifactInventoryEntry, ...]:
        """Return entries under the artifact-oriented alias."""

        return self.entries

    @property
    def artifact_count(self) -> int:
        """Return the number of indexed artifacts."""

        return len(self.entries)

    @property
    def total_bytes(self) -> int:
        """Return the aggregate byte count across indexed artifacts."""

        return sum(entry.byte_count for entry in self.entries)

    @property
    def media_type_counts(self) -> dict[str, int]:
        """Return media type counts in lexicographic key order."""

        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.media_type] = counts.get(entry.media_type, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def unique_fingerprint_count(self) -> int:
        """Return the number of distinct content fingerprints."""

        return len({entry.fingerprint for entry in self.entries})

    def __iter__(self):
        """Iterate over entries in stable path order."""

        return iter(self.entries)

    def __len__(self) -> int:
        """Return the number of indexed artifacts."""

        return len(self.entries)

    def to_counts_dict(self) -> dict[str, object]:
        """Return an aggregate-only payload with no paths or file contents."""

        return {
            "artifact_count": self.artifact_count,
            "media_type_counts": self.media_type_counts,
            "schema_version": self.schema_version,
            "total_bytes": self.total_bytes,
            "unique_fingerprint_count": self.unique_fingerprint_count,
        }

    def to_dict(self, *, counts_only: bool = False) -> dict[str, object]:
        """Return a deterministic JSON-ready inventory payload.

        Args:
            counts_only: When true, omit path-level metadata and return only
                aggregate counts.  The default retains the metadata index but
                still never includes file contents.
        """

        _require_bool(counts_only, "counts_only")
        payload = self.to_counts_dict()
        if not counts_only:
            payload["artifacts"] = [entry.to_dict() for entry in self.entries]
        return payload

    def to_json(self, *, indent: int | None = 2, counts_only: bool = True) -> str:
        """Render deterministic JSON, using an aggregate-only report by default."""

        validated_indent = _validate_json_indent(indent)
        return json.dumps(
            self.to_dict(counts_only=counts_only),
            ensure_ascii=False,
            indent=validated_indent,
            sort_keys=True,
        )

    def to_markdown(self, *, counts_only: bool = True) -> str:
        """Render deterministic Markdown, using aggregate counts by default."""

        _require_bool(counts_only, "counts_only")
        lines = _counts_markdown_lines(self)
        if not counts_only:
            lines.extend(
                [
                    "",
                    "## Artifacts",
                    "",
                    "| Path | Bytes | Media type | Fingerprint |",
                    "|---|---:|---|---|",
                ]
            )
            lines.extend(
                "| "
                f"{_markdown_cell(entry.path)} | "
                f"{entry.byte_count} | "
                f"{_markdown_cell(entry.media_type)} | "
                f"`{entry.fingerprint}` |"
                for entry in self.entries
            )
        return "\n".join(lines) + "\n"

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        counts_only: bool = True,
    ) -> Path:
        """Write a deterministic JSON report and return its output path."""

        report = self.to_json(indent=indent, counts_only=counts_only) + "\n"
        return _write_report(path, report)

    def write_markdown(self, path: str | Path, *, counts_only: bool = True) -> Path:
        """Write a deterministic Markdown report and return its output path."""

        report = self.to_markdown(counts_only=counts_only)
        return _write_report(path, report)


def _write_report(path: str | Path, report: str) -> Path:
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = None
            stream.write(report)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        return output_path
    except Exception:
        raise ArtifactInventoryError(
            "artifact inventory report could not be written"
        ) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


@dataclass(frozen=True, slots=True)
class _PathContext:
    lexical_root: Path
    resolved_root: Path


@dataclass(frozen=True, slots=True)
class _ValidatedPath:
    relative_path: str
    resolved_path: Path


def index_artifacts(
    paths: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
) -> ArtifactInventory:
    """Index local artifact files in stable path order.

    Args:
        paths: Relative or absolute file paths.  A single directory path is
            treated as a local artifact root and indexed recursively.
        root: Optional directory against which relative paths are resolved.
            Absolute inputs must be inside this root.  Supplying a root is
            recommended because it makes the report's relative path boundary
            explicit.

    Returns:
        An :class:`ArtifactInventory` containing byte counts, stable media
        types, and SHA-256 fingerprints.

    Raises:
        ArtifactPathError: If an input uses traversal, an unsafe path, or a
            symlink that escapes the root.
        DuplicateArtifactError: If normalized paths collide.
        UnreadableArtifactError: If a root or file cannot be read.
    """

    candidates: tuple[str | os.PathLike[str], ...]
    if isinstance(paths, (str, os.PathLike)):
        candidate = _coerce_path(paths, 0)
        try:
            is_directory = candidate.is_dir()
        except OSError:
            is_directory = False
        if is_directory and root is None:
            return inventory_directory(candidate)
        candidates = (candidate,)
    else:
        candidates = _materialize_paths(paths)

    raw_paths = tuple(
        _coerce_path(value, index) for index, value in enumerate(candidates)
    )
    for index, raw_path in enumerate(raw_paths):
        _validate_raw_path(raw_path, index)

    context = _prepare_path_context(raw_paths, root)
    entries: list[ArtifactInventoryEntry] = []
    seen_paths: set[str] = set()
    for index, raw_path in enumerate(raw_paths):
        validated = _validate_path(raw_path, context, index)
        if validated.relative_path in seen_paths:
            raise DuplicateArtifactError(
                f"duplicate artifact entry at index {index + 1}"
            )
        seen_paths.add(validated.relative_path)
        byte_count, fingerprint = _read_fingerprint(validated.resolved_path, index)
        entries.append(
            ArtifactInventoryEntry(
                path=validated.relative_path,
                byte_count=byte_count,
                media_type=_media_type_for_path(validated.relative_path),
                fingerprint=fingerprint,
            )
        )
    return ArtifactInventory(tuple(entries))


def build_artifact_inventory(
    paths: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
) -> ArtifactInventory:
    """Build an offline artifact inventory; alias of :func:`index_artifacts`."""

    return index_artifacts(paths, root=root)


def inventory_directory(root: str | os.PathLike[str]) -> ArtifactInventory:
    """Recursively index every regular file beneath a local directory."""

    root_path = _coerce_path(root, 0)
    try:
        if not root_path.is_dir():
            raise UnreadableArtifactError("artifact root is not a readable directory")
        relative_paths_list: list[str] = []
        for child in root_path.rglob("*"):
            if child.is_file() or child.is_symlink():
                relative_paths_list.append(child.relative_to(root_path).as_posix())
                if len(relative_paths_list) > _MAX_ARTIFACTS:
                    raise ArtifactInventoryError(
                        "artifact inventory exceeds the supported entry limit"
                    )
        relative_paths = tuple(sorted(relative_paths_list))
    except UnreadableArtifactError:
        raise
    except ArtifactInventoryError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise UnreadableArtifactError("artifact root cannot be read") from None
    return index_artifacts(relative_paths, root=root_path)


def render_artifact_inventory_json(
    inventory: ArtifactInventory,
    *,
    indent: int | None = 2,
    counts_only: bool = True,
) -> str:
    """Render an artifact inventory as deterministic JSON."""

    _require_inventory(inventory)
    return inventory.to_json(indent=indent, counts_only=counts_only)


def render_artifact_inventory_markdown(
    inventory: ArtifactInventory,
    *,
    counts_only: bool = True,
) -> str:
    """Render an artifact inventory as deterministic Markdown."""

    _require_inventory(inventory)
    return inventory.to_markdown(counts_only=counts_only)


def render_counts_json(
    inventory: ArtifactInventory,
    *,
    indent: int | None = 2,
) -> str:
    """Render only aggregate artifact counts as deterministic JSON."""

    return render_artifact_inventory_json(inventory, indent=indent, counts_only=True)


def render_counts_markdown(inventory: ArtifactInventory) -> str:
    """Render only aggregate artifact counts as deterministic Markdown."""

    return render_artifact_inventory_markdown(inventory, counts_only=True)


# Short aliases keep the API discoverable for callers that already use
# ``render_inventory_*`` terminology.
render_inventory_json = render_artifact_inventory_json
render_inventory_markdown = render_artifact_inventory_markdown
ArtifactInventoryRecord = ArtifactInventoryEntry


def _coerce_path(value: str | os.PathLike[str], index: int) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"artifact entry {index + 1} must be a path")
    try:
        return Path(value)
    except (TypeError, ValueError, OSError):
        raise ArtifactPathError(
            f"artifact entry {index + 1} has an invalid path"
        ) from None


def _materialize_paths(
    values: Iterable[str | os.PathLike[str]],
) -> tuple[str | os.PathLike[str], ...]:
    try:
        candidates = tuple(islice(iter(values), _MAX_ARTIFACTS + 1))
    except Exception:
        raise TypeError("paths must be a path or an iterable of paths") from None
    if len(candidates) > _MAX_ARTIFACTS:
        raise ArtifactInventoryError(
            "artifact inventory exceeds the supported entry limit"
        )
    return candidates


def _validate_raw_path(path: Path, index: int) -> None:
    try:
        text = os.fspath(path)
    except TypeError:
        raise ArtifactPathError(
            f"artifact entry {index + 1} has an invalid path"
        ) from None
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    if not text or "\x00" in text or any(ord(character) < 32 for character in text):
        raise ArtifactPathError(f"artifact entry {index + 1} has an unsafe path")
    slash_parts = text.replace("\\", "/").split("/")
    if ".." in path.parts or ".." in slash_parts:
        raise ArtifactPathError(
            f"artifact entry {index + 1} uses forbidden path traversal"
        )


def _prepare_path_context(
    raw_paths: tuple[Path, ...],
    root: str | os.PathLike[str] | None,
) -> _PathContext:
    if root is not None:
        root_path = _coerce_path(root, 0)
        lexical_root = _absolute_path(root_path)
    elif raw_paths and all(path.is_absolute() for path in raw_paths):
        try:
            common_parent = os.path.commonpath([str(path.parent) for path in raw_paths])
        except (OSError, ValueError):
            raise ArtifactPathError(
                "absolute artifact paths need a common local root"
            ) from None
        lexical_root = _absolute_path(Path(common_parent))
    else:
        lexical_root = _absolute_path(Path.cwd())

    try:
        resolved_root = lexical_root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise UnreadableArtifactError("artifact root cannot be read") from None
    if not resolved_root.is_dir():
        raise UnreadableArtifactError("artifact root is not a readable directory")
    return _PathContext(lexical_root=lexical_root, resolved_root=resolved_root)


def _validate_path(path: Path, context: _PathContext, index: int) -> _ValidatedPath:
    candidate = path if path.is_absolute() else context.lexical_root / path
    candidate_absolute = _absolute_path(candidate)
    try:
        relative = candidate_absolute.relative_to(context.lexical_root)
    except ValueError:
        raise ArtifactPathError(
            f"artifact entry {index + 1} escapes the inventory root"
        ) from None

    try:
        resolved = candidate_absolute.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise UnreadableArtifactError(
            f"artifact entry {index + 1} is unreadable"
        ) from None
    try:
        resolved.relative_to(context.resolved_root)
    except ValueError:
        raise ArtifactPathError(
            f"artifact entry {index + 1} escapes the inventory root"
        ) from None
    if not resolved.is_file():
        raise UnreadableArtifactError(
            f"artifact entry {index + 1} is not a readable file"
        )

    relative_path = relative.as_posix()
    if not relative_path or relative_path == ".":
        raise ArtifactPathError(f"artifact entry {index + 1} is not a file path")
    return _ValidatedPath(relative_path=relative_path, resolved_path=resolved)


def _read_fingerprint(path: Path, index: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    descriptor: int | None = None
    try:
        before_open = path.lstat()
        if not stat.S_ISREG(before_open.st_mode):
            raise OSError
        if not before_open.st_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH):
            raise PermissionError
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_snapshot(
            before_open, opened
        ):
            raise OSError
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            while chunk := stream.read(_CHUNK_SIZE):
                byte_count += len(chunk)
                digest.update(chunk)
            after_read = os.fstat(stream.fileno())
        after_path = path.lstat()
        if not _same_file_snapshot(opened, after_read) or not _same_file_snapshot(
            after_read, after_path
        ):
            raise OSError
    except (OSError, ValueError, TypeError):
        raise UnreadableArtifactError(
            f"artifact entry {index + 1} could not be read"
        ) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return byte_count, f"{_FINGERPRINT_PREFIX}{digest.hexdigest()}"


def _same_file_snapshot(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        stat.S_ISREG(first.st_mode)
        and stat.S_ISREG(second.st_mode)
        and first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
    )


def _media_type_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return _MEDIA_TYPES.get(suffix, "application/octet-stream")


def _absolute_path(path: Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError):
        raise ArtifactPathError("artifact path cannot be normalized") from None


def _counts_markdown_lines(inventory: ArtifactInventory) -> list[str]:
    lines = [
        "# Artifact Inventory",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Artifact count | {inventory.artifact_count} |",
        f"| Total bytes | {inventory.total_bytes} |",
        f"| Unique fingerprints | {inventory.unique_fingerprint_count} |",
        "",
        "## Media types",
        "",
        "| Media type | Artifact count |",
        "|---|---:|",
    ]
    if inventory.media_type_counts:
        lines.extend(
            f"| `{_markdown_cell(media_type)}` | {count} |"
            for media_type, count in inventory.media_type_counts.items()
        )
    else:
        lines.append("| _None_ | 0 |")
    return lines


def _markdown_cell(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _require_inventory(value: ArtifactInventory) -> None:
    if type(value) is not ArtifactInventory:
        raise TypeError("inventory must be an ArtifactInventory")


def _require_bool(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _validate_json_indent(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= _MAX_JSON_INDENT:
        raise ArtifactInventoryError(
            "JSON indentation must be an integer between 0 and 8"
        )
    return value


__all__ = [
    "ArtifactInventory",
    "ArtifactInventoryEntry",
    "ArtifactInventoryError",
    "ArtifactInventoryRecord",
    "ArtifactPathError",
    "ArtifactRecord",
    "DuplicateArtifactError",
    "SCHEMA_VERSION",
    "UnreadableArtifactError",
    "build_artifact_inventory",
    "index_artifacts",
    "inventory_directory",
    "render_artifact_inventory_json",
    "render_artifact_inventory_markdown",
    "render_counts_json",
    "render_counts_markdown",
    "render_inventory_json",
    "render_inventory_markdown",
]
