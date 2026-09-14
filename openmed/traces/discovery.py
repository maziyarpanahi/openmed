"""Discover local trace stores using filesystem metadata only.

The discovery surface deliberately returns aggregate metadata instead of paths
or payloads.  It is therefore suitable for a first-pass inventory before a
caller decides whether a trace store should be opened or changed.
"""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

TRACE_DISCOVERY_ENV_VAR = "OPENMED_TRACE_DISCOVERY"
TRACE_ROOTS_ENV_VAR = "OPENMED_TRACE_ROOTS"

_ALL_PLATFORMS = frozenset({"Darwin", "Linux", "Windows"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_STORE_TYPE_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


@dataclass(frozen=True, slots=True)
class TraceRootRule:
    """Describe a supported trace-store path relative to a user home.

    Args:
        store_type: Stable, non-path identifier for the store family.
        relative_path: Path components below the selected home directory.
        platforms: Platform names returned by :func:`platform.system`.
    """

    store_type: str
    relative_path: tuple[str, ...]
    platforms: frozenset[str] = _ALL_PLATFORMS


# Hidden home-directory stores are used on every supported desktop platform.
# Platform-specific application-data variants below make the path policy
# explicit without probing arbitrary directories.
SUPPORTED_TRACE_ROOT_RULES = (
    TraceRootRule("claude", (".claude", "projects")),
    TraceRootRule("codex", (".codex", "sessions")),
    TraceRootRule("cursor", (".cursor", "projects")),
    TraceRootRule(
        "claude",
        ("Library", "Application Support", "Claude", "projects"),
        frozenset({"Darwin"}),
    ),
    TraceRootRule(
        "codex",
        ("Library", "Application Support", "Codex", "sessions"),
        frozenset({"Darwin"}),
    ),
    TraceRootRule(
        "cursor",
        ("Library", "Application Support", "Cursor", "projects"),
        frozenset({"Darwin"}),
    ),
    TraceRootRule(
        "claude",
        (".local", "share", "claude", "projects"),
        frozenset({"Linux"}),
    ),
    TraceRootRule(
        "codex",
        (".local", "share", "codex", "sessions"),
        frozenset({"Linux"}),
    ),
    TraceRootRule(
        "cursor",
        (".local", "share", "cursor", "projects"),
        frozenset({"Linux"}),
    ),
    TraceRootRule(
        "claude",
        ("AppData", "Roaming", "Claude", "projects"),
        frozenset({"Windows"}),
    ),
    TraceRootRule(
        "codex",
        ("AppData", "Local", "Codex", "sessions"),
        frozenset({"Windows"}),
    ),
    TraceRootRule(
        "cursor",
        ("AppData", "Roaming", "Cursor", "projects"),
        frozenset({"Windows"}),
    ),
)


@dataclass(frozen=True, slots=True, order=True)
class TraceStore:
    """Aggregate metadata for one discovered local trace-store family.

    ``byte_size`` is the sum of regular-file sizes below all readable roots
    for ``store_type``.  No path or trace payload is retained in this result.
    """

    store_type: str
    file_count: int
    byte_size: int

    def __post_init__(self) -> None:
        if _STORE_TYPE_PATTERN.fullmatch(self.store_type) is None:
            raise ValueError("store_type must be a PHI-free identifier")

    @property
    def size_bytes(self) -> int:
        """Return ``byte_size`` under the common ``size_bytes`` spelling."""

        return self.byte_size

    def to_dict(self) -> dict[str, int | str]:
        """Return a path-free, serializable summary."""

        return {
            "store_type": self.store_type,
            "file_count": self.file_count,
            "byte_size": self.byte_size,
        }


TraceStoreSummary = TraceStore

_PathLike: TypeAlias = str | os.PathLike[str]
_RootInput: TypeAlias = (
    Path
    | str
    | os.PathLike[str]
    | tuple[str, _PathLike]
    | Mapping[str, _PathLike | Iterable[_PathLike]]
    | Iterable[tuple[str, _PathLike] | _PathLike]
)


def discover_trace_stores(
    roots: _RootInput | None = None,
    *,
    platform_name: str | None = None,
    home: _PathLike | None = None,
    enabled: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[TraceStore, ...]:
    """Inventory supported local trace stores without opening payloads.

    Args:
        roots: Optional explicit root mapping or iterable.  A mapping uses
            ``store_type -> path`` (or an iterable of paths).  An iterable of
            ``(store_type, path)`` pairs is also accepted.  Supplying explicit
            roots replaces the built-in platform rules.
        platform_name: Override ``platform.system()`` for deterministic
            callers and tests.
        home: Override the home directory used by built-in rules.
        enabled: Set to ``False`` to opt out.  When omitted, the
            ``OPENMED_TRACE_DISCOVERY`` environment variable is honored.
        environ: Environment mapping used for opt-out and extra-root settings.

    Returns:
        A deterministic tuple sorted by ``store_type``.  Missing roots,
        unreadable roots, symlinks, and unreadable descendants are skipped.
        Each result contains only the store type, regular-file count, and
        aggregate byte size.

    The implementation uses directory enumeration and file metadata only. It
    makes no network calls and never reads a trace payload.
    """

    environment = os.environ if environ is None else environ
    if enabled is False or (enabled is None and _discovery_is_disabled(environment)):
        return ()

    if roots is None:
        root_specs = _default_root_specs(
            platform_name=platform_name,
            home=home,
            environ=environment,
        )
    else:
        root_specs = tuple(_coerce_root_specs(roots))

    root_specs += tuple(_environment_root_specs(environment))
    totals: dict[str, list[int]] = {}
    seen: set[tuple[str, str]] = set()

    for store_type, root in sorted(
        root_specs,
        key=lambda item: (
            _normalize_store_type(item[0]),
            os.fspath(item[1]).casefold(),
        ),
    ):
        normalized_type = _normalize_store_type(store_type)
        normalized_root = Path(root).expanduser()
        identity = (
            normalized_type,
            os.path.normcase(os.path.abspath(os.fspath(normalized_root))),
        )
        if identity in seen:
            continue
        seen.add(identity)

        measurement = _measure_root(normalized_root)
        if measurement is None:
            continue
        file_count, byte_size = measurement
        aggregate = totals.setdefault(normalized_type, [0, 0])
        aggregate[0] += file_count
        aggregate[1] += byte_size

    return tuple(
        TraceStore(store_type, values[0], values[1])
        for store_type, values in sorted(totals.items())
    )


def _discovery_is_disabled(environ: Mapping[str, str]) -> bool:
    value = environ.get(TRACE_DISCOVERY_ENV_VAR, "").strip().lower()
    return value in _DISABLED_VALUES


def _default_root_specs(
    *,
    platform_name: str | None,
    home: _PathLike | None,
    environ: Mapping[str, str],
) -> tuple[tuple[str, Path], ...]:
    selected_home = Path.home() if home is None else Path(home).expanduser()
    selected_platform = platform.system() if platform_name is None else platform_name
    specs = [
        (rule.store_type, selected_home.joinpath(*rule.relative_path))
        for rule in SUPPORTED_TRACE_ROOT_RULES
        if selected_platform in rule.platforms
    ]

    if selected_platform == "Linux":
        data_home = _environment_path(
            environ,
            "XDG_DATA_HOME",
            selected_home / ".local" / "share",
        )
        specs.extend(
            (
                ("claude", data_home / "claude" / "projects"),
                ("codex", data_home / "codex" / "sessions"),
                ("cursor", data_home / "cursor" / "projects"),
            )
        )
    elif selected_platform == "Windows":
        app_data = environ.get("APPDATA", "").strip()
        local_app_data = environ.get("LOCALAPPDATA", "").strip()
        if app_data:
            app_data_path = Path(app_data).expanduser()
            specs.extend(
                (
                    ("claude", app_data_path / "Claude" / "projects"),
                    ("cursor", app_data_path / "Cursor" / "projects"),
                )
            )
        if local_app_data:
            specs.append(
                ("codex", Path(local_app_data).expanduser() / "Codex" / "sessions")
            )

    return tuple(specs)


def _environment_path(environ: Mapping[str, str], name: str, fallback: Path) -> Path:
    value = environ.get(name, "").strip()
    return Path(value).expanduser() if value else fallback


def _environment_root_specs(environ: Mapping[str, str]) -> tuple[tuple[str, Path], ...]:
    """Parse ``store_type=path`` entries without reporting malformed values."""

    raw_roots = environ.get(TRACE_ROOTS_ENV_VAR, "")
    specs: list[tuple[str, Path]] = []
    for raw_root in raw_roots.split(os.pathsep):
        store_type, separator, raw_path = raw_root.partition("=")
        if not separator or not store_type.strip() or not raw_path.strip():
            continue
        specs.append((_normalize_store_type(store_type), Path(raw_path).expanduser()))
    return tuple(specs)


def _coerce_root_specs(roots: _RootInput) -> Iterable[tuple[str, Path]]:
    if isinstance(roots, Mapping):
        for store_type, raw_paths in roots.items():
            if isinstance(raw_paths, (str, os.PathLike)):
                yield _normalize_store_type(store_type), Path(raw_paths).expanduser()
                continue
            try:
                path_values = iter(raw_paths)
            except TypeError:
                continue
            for raw_path in path_values:
                if isinstance(raw_path, (str, os.PathLike)):
                    yield _normalize_store_type(store_type), Path(raw_path).expanduser()
        return

    if isinstance(roots, (str, os.PathLike)):
        yield "custom", Path(roots).expanduser()
        return

    if isinstance(roots, tuple) and len(roots) == 2 and isinstance(roots[0], str):
        if isinstance(roots[1], (str, os.PathLike)):
            yield _normalize_store_type(roots[0]), Path(roots[1]).expanduser()
            return

    try:
        root_values = iter(roots)
    except TypeError:
        return

    for item in root_values:
        if isinstance(item, (str, os.PathLike)):
            yield "custom", Path(item).expanduser()
        elif isinstance(item, tuple) and len(item) == 2:
            store_type, raw_path = item
            if isinstance(raw_path, (str, os.PathLike)):
                yield _normalize_store_type(store_type), Path(raw_path).expanduser()


def _normalize_store_type(store_type: object) -> str:
    normalized = str(store_type).strip().lower()
    if _STORE_TYPE_PATTERN.fullmatch(normalized) is None:
        return "custom"
    return normalized


def _measure_root(root: Path) -> tuple[int, int] | None:
    """Count regular files below ``root`` without following links or reading."""

    try:
        if root.is_symlink():
            return None
    except (OSError, ValueError):
        return None

    pending = [root]
    scanned_root = False
    file_count = 0
    byte_size = 0

    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                scanned_root = True
                for entry in sorted(entries, key=lambda item: item.name.casefold()):
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            size = entry.stat(follow_symlinks=False).st_size
                            if size >= 0:
                                file_count += 1
                                byte_size += size
                    except (OSError, ValueError):
                        continue
        except (OSError, ValueError):
            if not scanned_root:
                return None

    return file_count, byte_size


__all__ = [
    "TRACE_DISCOVERY_ENV_VAR",
    "TRACE_ROOTS_ENV_VAR",
    "SUPPORTED_TRACE_ROOT_RULES",
    "TraceRootRule",
    "TraceStore",
    "TraceStoreSummary",
    "discover_trace_stores",
]
