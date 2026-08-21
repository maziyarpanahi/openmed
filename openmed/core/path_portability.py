"""Deterministic, source-path-free audits for local resource paths.

The audit operates on path text only.  It never resolves, opens, or otherwise
consults the local filesystem, which keeps manifest validation offline and
portable across hosts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

PORTABILITY_AUDIT_SCHEMA_VERSION = "openmed.path_portability.v1"
MAX_AUDIT_PATHS = 10_000
MAX_PATH_CHARACTERS = 4_096
MAX_PATH_COMPONENTS = 256

ABSOLUTE_ROOT = "absolute_root"
CASE_FOLD_COLLISION = "case_fold_collision"
NORMALIZATION_DRIFT = "normalization_drift"
RESERVED_COMPONENT = "reserved_component"
TRAVERSAL = "traversal"

ISSUE_CATEGORIES = (
    ABSOLUTE_ROOT,
    CASE_FOLD_COLLISION,
    NORMALIZATION_DRIFT,
    RESERVED_COMPONENT,
    TRAVERSAL,
)

_ISSUE_CATEGORY_ORDER = {
    category: index for index, category in enumerate(ISSUE_CATEGORIES)
}
_MAX_ISSUE_CATEGORY_LENGTH = max(len(category) for category in ISSUE_CATEGORIES)
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_FINGERPRINT_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_WINDOWS_INVALID_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
        "nul",
        "prn",
    }
)


class PathPortabilityInputError(ValueError):
    """Raised when an audit input cannot be read as path text."""


@dataclass(frozen=True, slots=True)
class PathPortabilityRecord:
    """PHI-safe result for one normalized path identity.

    ``normalized_path_fingerprint`` is a SHA-256 digest of the normalized
    path.  The source path is deliberately not retained in this object or in
    any serialized representation.
    """

    normalized_path_fingerprint: str
    issue_categories: tuple[str, ...] = ()
    occurrences: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.normalized_path_fingerprint) is not str
            or len(self.normalized_path_fingerprint) != len("sha256:") + 64
            or _FINGERPRINT_RE.fullmatch(self.normalized_path_fingerprint) is None
        ):
            raise ValueError("normalized_path_fingerprint must be a SHA-256 digest")
        if (
            type(self.issue_categories) is not tuple
            or len(self.issue_categories) > len(ISSUE_CATEGORIES)
            or any(
                type(category) is not str
                or len(category) > _MAX_ISSUE_CATEGORY_LENGTH
                or category not in ISSUE_CATEGORIES
                for category in self.issue_categories
            )
            or len(set(self.issue_categories)) != len(self.issue_categories)
        ):
            raise ValueError("issue_categories contains an unknown category")
        if self.issue_categories != tuple(
            sorted(self.issue_categories, key=_ISSUE_CATEGORY_ORDER.__getitem__)
        ):
            raise ValueError("issue_categories must use the canonical order")
        if (
            type(self.occurrences) is not int
            or self.occurrences < 1
            or self.occurrences > MAX_AUDIT_PATHS
        ):
            raise ValueError("occurrences must be positive")

    @property
    def fingerprint(self) -> str:
        """Return the normalized path fingerprint."""

        return self.normalized_path_fingerprint

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the portability issue categories for this path."""

        return self.issue_categories

    @property
    def is_portable(self) -> bool:
        """Return whether this path has no portability findings."""

        return not self.issue_categories

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe, source-path-free record."""

        return {
            "normalized_path_fingerprint": self.normalized_path_fingerprint,
            "issue_categories": list(self.issue_categories),
            "occurrences": self.occurrences,
        }


@dataclass(frozen=True, slots=True)
class PathPortabilityReport:
    """Deterministic, source-path-free results for a path collection."""

    records: tuple[PathPortabilityRecord, ...]
    checked_count: int

    def __post_init__(self) -> None:
        if (
            type(self.records) is not tuple
            or len(self.records) > MAX_AUDIT_PATHS
            or any(type(record) is not PathPortabilityRecord for record in self.records)
        ):
            raise ValueError("records must be immutable portability records")
        fingerprints = tuple(
            record.normalized_path_fingerprint for record in self.records
        )
        if fingerprints != tuple(sorted(fingerprints)) or len(set(fingerprints)) != len(
            fingerprints
        ):
            raise ValueError("records must use unique canonical fingerprint order")
        if (
            type(self.checked_count) is not int
            or self.checked_count < 0
            or self.checked_count > MAX_AUDIT_PATHS
        ):
            raise ValueError("checked_count must be non-negative")
        if self.checked_count != sum(record.occurrences for record in self.records):
            raise ValueError("checked_count must equal recorded occurrences")

    @property
    def findings(self) -> tuple[PathPortabilityRecord, ...]:
        """Return only records with at least one portability issue."""

        return tuple(record for record in self.records if record.issue_categories)

    @property
    def affected_path_count(self) -> int:
        """Return the number of distinct normalized paths with findings."""

        return len(self.findings)

    @property
    def issue_count(self) -> int:
        """Return the total number of category findings."""

        return sum(len(record.issue_categories) for record in self.records)

    @property
    def is_clean(self) -> bool:
        """Return whether every audited path passed the portability checks."""

        return not self.findings

    @property
    def clean(self) -> bool:
        """Return the clean-audit verdict as a convenience alias."""

        return self.is_clean

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, source-path-free report mapping."""

        return {
            "schema_version": PORTABILITY_AUDIT_SCHEMA_VERSION,
            "checked_count": self.checked_count,
            "affected_path_count": self.affected_path_count,
            "issue_count": self.issue_count,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        """Return canonical JSON containing no source path text."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass
class _PathState:
    categories: set[str]
    occurrences: int = 0


def audit_resource_paths(
    paths: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
) -> PathPortabilityReport:
    """Audit local resource paths without reading the filesystem.

    Args:
        paths: An iterable of path strings or path-like objects.  A single
            path string or path-like object is also accepted.

    Returns:
        A deterministic report.  Every record contains only a normalized path
        fingerprint, issue categories, and an occurrence count.

    Raises:
        PathPortabilityInputError: If the collection or an entry cannot be
            converted to text.  The exception message never includes input
            values.
    """

    states: dict[str, _PathState] = {}
    case_fold_groups: dict[str, set[str]] = {}
    checked_count = 0

    for entry in _bounded_entries(paths):
        raw_path = _coerce_path_text(entry)
        normalized_path, categories = _normalize_path(raw_path)
        state = states.setdefault(normalized_path, _PathState(set()))
        state.categories.update(categories)
        state.occurrences += 1
        case_fold_key = _case_fold_key(normalized_path)
        case_fold_groups.setdefault(case_fold_key, set()).add(normalized_path)
        checked_count += 1

    for normalized_paths in case_fold_groups.values():
        if len(normalized_paths) < 2:
            continue
        for normalized_path in normalized_paths:
            states[normalized_path].categories.add(CASE_FOLD_COLLISION)

    records = tuple(
        PathPortabilityRecord(
            normalized_path_fingerprint=_fingerprint(normalized_path),
            issue_categories=tuple(
                sorted(
                    state.categories,
                    key=_ISSUE_CATEGORY_ORDER.__getitem__,
                )
            ),
            occurrences=state.occurrences,
        )
        for normalized_path, state in sorted(
            states.items(),
            key=lambda item: _fingerprint(item[0]),
        )
    )
    return PathPortabilityReport(records=records, checked_count=checked_count)


def audit_path_portability(
    paths: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
) -> PathPortabilityReport:
    """Alias for :func:`audit_resource_paths` with the audit name first."""

    return audit_resource_paths(paths)


def _as_entries(
    paths: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
) -> Iterator[str | os.PathLike[str]]:
    if isinstance(paths, (str, os.PathLike)):
        return iter((paths,))
    if isinstance(paths, (bytes, bytearray, memoryview)):
        raise PathPortabilityInputError("path collection must be iterable")
    try:
        return iter(paths)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise PathPortabilityInputError("path collection must be iterable") from None


def _bounded_entries(
    paths: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
) -> Iterator[str | os.PathLike[str]]:
    """Yield a bounded collection while containing iterator failures."""

    iterator = _as_entries(paths)
    for index in range(MAX_AUDIT_PATHS + 1):
        try:
            entry = next(iterator)
        except StopIteration:
            return
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise PathPortabilityInputError(
                "path collection could not be read"
            ) from None
        if index == MAX_AUDIT_PATHS:
            raise PathPortabilityInputError(
                "path collection exceeds the bounded maximum"
            )
        yield entry


def _coerce_path_text(value: object) -> str:
    if type(value) is str:
        path_text = value
    else:
        if not isinstance(value, os.PathLike):
            raise PathPortabilityInputError("path entries must be text or path-like")
        try:
            path_text = os.fspath(value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise PathPortabilityInputError(
                "path entries must be text or path-like"
            ) from None
        if type(path_text) is not str:
            raise PathPortabilityInputError("path entries must be text or path-like")
    if (
        not path_text
        or len(path_text) > MAX_PATH_CHARACTERS
        or any(unicodedata.category(character) == "Cs" for character in path_text)
    ):
        raise PathPortabilityInputError("path entries must be bounded valid text")
    return path_text


def _normalize_path(raw_path: str) -> tuple[str, set[str]]:
    try:
        normalized = unicodedata.normalize("NFKC", raw_path).replace("\\", "/")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise PathPortabilityInputError(
            "path entries could not be normalized"
        ) from None
    if len(normalized) > MAX_PATH_CHARACTERS:
        raise PathPortabilityInputError("path entries must be bounded valid text")
    root = _root_marker(normalized)
    remainder = _root_remainder(normalized, root)
    components = [
        component for component in remainder.split("/") if component not in ("", ".")
    ]
    if len(components) > MAX_PATH_COMPONENTS:
        raise PathPortabilityInputError("path entries have too many components")
    canonical = _join_normalized_path(root, components)

    categories: set[str] = set()
    if _has_absolute_root(normalized):
        categories.add(ABSOLUTE_ROOT)
    if ".." in components:
        categories.add(TRAVERSAL)
    if any(_is_reserved_component(component) for component in components):
        categories.add(RESERVED_COMPONENT)
    if raw_path != canonical:
        categories.add(NORMALIZATION_DRIFT)
    return canonical, categories


def _root_marker(normalized: str) -> str:
    if normalized.startswith("//"):
        return "//"
    if normalized.startswith("/"):
        return "/"
    drive = _DRIVE_PREFIX_RE.match(normalized)
    if drive:
        return drive.group(0) + ("/" if normalized[2:3] == "/" else "")
    return ""


def _root_remainder(normalized: str, root: str) -> str:
    if root in {"/", "//"}:
        return normalized.lstrip("/")
    if root:
        return normalized[len(root) :].lstrip("/")
    return normalized


def _join_normalized_path(root: str, components: list[str]) -> str:
    body = "/".join(components)
    if root == "//":
        return "//" + body if body else "//"
    if root == "/":
        return "/" + body if body else "/"
    if root.endswith("/"):
        return root + body
    if root:
        return root + body if body else root
    return body or "."


def _has_absolute_root(normalized: str) -> bool:
    if normalized.startswith("/") or _DRIVE_PREFIX_RE.match(normalized):
        return True
    return normalized.casefold().startswith("file:")


def _is_reserved_component(component: str) -> bool:
    if component == "..":
        return False
    if component != component.rstrip(" ."):
        return True
    if any(
        character in _WINDOWS_INVALID_CHARACTERS
        or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in component
    ):
        return True
    windows_name = component.split(".", 1)[0].rstrip(" ").casefold()
    return windows_name in _WINDOWS_RESERVED_NAMES


def _case_fold_key(normalized_path: str) -> str:
    return unicodedata.normalize("NFKC", normalized_path).casefold()


def _fingerprint(normalized_path: str) -> str:
    digest = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "ABSOLUTE_ROOT",
    "CASE_FOLD_COLLISION",
    "ISSUE_CATEGORIES",
    "MAX_AUDIT_PATHS",
    "MAX_PATH_CHARACTERS",
    "MAX_PATH_COMPONENTS",
    "NORMALIZATION_DRIFT",
    "PORTABILITY_AUDIT_SCHEMA_VERSION",
    "PathPortabilityInputError",
    "PathPortabilityRecord",
    "PathPortabilityReport",
    "RESERVED_COMPONENT",
    "TRAVERSAL",
    "audit_path_portability",
    "audit_resource_paths",
]
