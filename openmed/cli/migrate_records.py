"""Directory dry run behind ``openmed migrate records``.

The walk reads ``.json`` and ``.jsonl`` files under one directory, runs each
record's migration chain in memory through the record migration registry and
aggregates the outcomes per schema id, version and outcome. Nothing is written
and nothing about a record leaves the process except counts, schema ids,
versions and controlled reason codes: no file names, paths, digests or values.

The walk is bounded. Files larger than the file limit, JSON Lines rows longer
than the line limit, symlinks that resolve outside the directory, broken
symlinks and non-regular files are counted as ``invalid`` without being read.
Exceeding the file, record or depth limit stops the run with a usage error, so
a partial count is never reported as complete.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Final

from ._output import EXIT_USAGE, CliError, emit

OUTCOMES: Final = ("current", "migratable", "lossy", "no_path", "invalid")
"""Outcome categories in report order."""

DEFAULT_MAX_FILES: Final = 10_000
DEFAULT_MAX_RECORDS: Final = 100_000
DEFAULT_MAX_FILE_BYTES: Final = 64 * 1024 * 1024
DEFAULT_MAX_LINE_BYTES: Final = 1024 * 1024
MAX_DEPTH: Final = 32

RECORD_SUFFIXES: Final = (".json", ".jsonl")

_NO_PATH_REASONS = frozenset(
    {"ambiguous_path", "ambiguous_schema", "no_path", "unknown_schema"}
)


@dataclass(frozen=True, slots=True)
class WalkLimits:
    """Bounds applied to one directory dry run.

    Attributes:
        max_files: Most candidate record files the walk may visit.
        max_records: Most records the run may migrate.
        max_file_bytes: Largest file that is read; larger files are invalid.
        max_line_bytes: Longest JSON Lines row; longer rows are invalid.
    """

    max_files: int = DEFAULT_MAX_FILES
    max_records: int = DEFAULT_MAX_RECORDS
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES

    def to_dict(self) -> dict[str, int]:
        """Return the limits as a JSON-ready mapping.

        Returns:
            Each limit by name, plus the directory depth limit.
        """

        return {
            "max_depth": MAX_DEPTH,
            "max_file_bytes": self.max_file_bytes,
            "max_files": self.max_files,
            "max_line_bytes": self.max_line_bytes,
            "max_records": self.max_records,
        }


@dataclass
class DirectoryDryRun:
    """Aggregated, value-free outcome of one directory dry run.

    Attributes:
        limits: Bounds that applied to the walk.
        files_scanned: Record files that were read.
        files_invalid: Record files counted invalid without being read.
        files_ignored: Entries skipped because they are not record files.
        groups: Record counts keyed by schema id, version and outcome.
        reasons: Counts of controlled reason codes for non-migratable records.
    """

    limits: WalkLimits
    files_scanned: int = 0
    files_invalid: int = 0
    files_ignored: int = 0
    groups: Counter[tuple[str | None, str | None, str]] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)

    @property
    def records(self) -> int:
        """Return how many records (or unreadable files) were counted."""

        return sum(self.groups.values())

    def add(
        self,
        outcome: str,
        *,
        schema_id: str | None = None,
        version: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Count one record outcome.

        Args:
            outcome: One of :data:`OUTCOMES`.
            schema_id: Identified schema id, if any.
            version: Record version, if known and safe.
            reason: Controlled reason code for ``no_path`` and ``invalid``.

        Raises:
            CliError: When the record limit is exceeded.
        """

        if self.records >= self.limits.max_records:
            raise CliError(
                "The directory holds more records than the "
                f"{self.limits.max_records} record limit.",
                code="records_limit_exceeded",
                exit_code=EXIT_USAGE,
            )
        self.groups[(schema_id, version, outcome)] += 1
        if reason is not None:
            self.reasons[reason] += 1

    def to_dict(self) -> dict[str, Any]:
        """Return the counts-only report.

        Returns:
            File counts, record counts per outcome, per-group counts, reason
            counts, the limits and the dry-run markers.
        """

        outcomes = dict.fromkeys(OUTCOMES, 0)
        for (_, _, outcome), count in self.groups.items():
            outcomes[outcome] += count
        groups = [
            {
                "schema_id": schema_id,
                "version": version,
                "outcome": outcome,
                "count": count,
            }
            for (schema_id, version, outcome), count in sorted(
                self.groups.items(),
                key=lambda item: (
                    item[0][0] or "",
                    item[0][1] or "",
                    OUTCOMES.index(item[0][2]),
                ),
            )
        ]
        return {
            "dry_run": True,
            "written": False,
            "files": {
                "ignored": self.files_ignored,
                "invalid": self.files_invalid,
                "scanned": self.files_scanned,
            },
            "records": self.records,
            "outcomes": outcomes,
            "groups": groups,
            "reasons": dict(sorted(self.reasons.items())),
            "limits": self.limits.to_dict(),
        }


def add_records_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``migrate records`` arguments to ``parser``.

    Args:
        parser: The ``records`` subcommand parser.
    """

    parser.add_argument(
        "directory", type=Path, help="Directory of .json and .jsonl records."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Required: migrate in memory only and write nothing.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        metavar="SCHEMA_ID",
        help="Treat every record as this schema instead of identifying it.",
    )
    for flag, default, help_text in (
        ("--max-files", DEFAULT_MAX_FILES, "Most record files to visit."),
        ("--max-records", DEFAULT_MAX_RECORDS, "Most records to migrate."),
        ("--max-file-bytes", DEFAULT_MAX_FILE_BYTES, "Largest file to read."),
        ("--max-line-bytes", DEFAULT_MAX_LINE_BYTES, "Longest JSON Lines row."),
    ):
        parser.add_argument(
            flag, type=_positive_int, default=default, metavar="N", help=help_text
        )


def handle_records(args: argparse.Namespace) -> int:
    """Run the directory dry run and print the counts-only report.

    Args:
        args: Parsed ``migrate records`` arguments.

    Returns:
        The process exit status.
    """

    from .migrate import _registry

    limits = WalkLimits(
        max_files=args.max_files,
        max_records=args.max_records,
        max_file_bytes=args.max_file_bytes,
        max_line_bytes=args.max_line_bytes,
    )
    run = dry_run_directory(
        args.directory, registry=_registry(), schema_id=args.schema, limits=limits
    )
    payload = run.to_dict()
    return emit(args, payload, human=_format(payload))


def dry_run_directory(
    directory: Path,
    *,
    registry: Any,
    schema_id: str | None = None,
    limits: WalkLimits | None = None,
) -> DirectoryDryRun:
    """Migrate every record under ``directory`` in memory and count outcomes.

    Args:
        directory: Directory to walk. It is never written.
        registry: Record migration registry used for every record.
        schema_id: Schema id applied to every record instead of identifying it.
        limits: Walk bounds; the defaults when omitted.

    Returns:
        The aggregated dry-run counts.

    Raises:
        CliError: With a usage exit code when the directory cannot be read or
            the file, record or depth limit is exceeded.
    """

    run = DirectoryDryRun(limits=limits or WalkLimits())
    try:
        root = Path(os.path.realpath(directory))
        if not root.is_dir():
            raise OSError
    except OSError:
        raise CliError(
            "The records directory could not be read.",
            code="records_directory_unreadable",
            exit_code=EXIT_USAGE,
        ) from None
    seen: set[tuple[int, int]] = set()
    candidates = 0
    for entry_path, target in _walk(root, run):
        candidates += 1
        if candidates > run.limits.max_files:
            raise CliError(
                "The directory holds more record files than the "
                f"{run.limits.max_files} file limit.",
                code="records_limit_exceeded",
                exit_code=EXIT_USAGE,
            )
        if target is None:
            continue
        _scan_file(entry_path, target, run, registry, schema_id, seen)
    return run


def _walk(root: Path, run: DirectoryDryRun) -> Iterator[tuple[Path, Path | None]]:
    """Yield ``(entry, target)`` for record files; ``target`` is None if refused."""

    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            run.files_invalid += 1
            run.add("invalid", reason="directory_unreadable")
            continue
        subdirectories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    yield from _symlink(root, path, run)
                elif entry.is_dir(follow_symlinks=False):
                    if depth + 1 > MAX_DEPTH:
                        raise CliError(
                            "The directory is nested deeper than the "
                            f"{MAX_DEPTH} level limit.",
                            code="records_limit_exceeded",
                            exit_code=EXIT_USAGE,
                        )
                    subdirectories.append(path)
                elif not _is_record_name(path):
                    run.files_ignored += 1
                elif entry.is_file(follow_symlinks=False):
                    yield path, path
                else:
                    run.files_invalid += 1
                    run.add("invalid", reason="not_regular_file")
                    yield path, None
            except OSError:
                run.files_invalid += 1
                run.add("invalid", reason="file_unreadable")
                yield path, None
        pending.extend((child, depth + 1) for child in reversed(subdirectories))


def _symlink(
    root: Path, path: Path, run: DirectoryDryRun
) -> Iterator[tuple[Path, Path | None]]:
    try:
        target = Path(os.path.realpath(path))
        info = target.stat()
    except OSError:
        if not _is_record_name(path):
            run.files_ignored += 1
            return
        run.files_invalid += 1
        run.add("invalid", reason="symlink_broken")
        yield path, None
        return
    if os.path.commonpath([str(root), str(target)]) != str(root):
        run.files_invalid += 1
        run.add("invalid", reason="symlink_outside_root")
        yield path, None
        return
    if stat.S_ISDIR(info.st_mode) or not _is_record_name(path):
        # Directory links inside the root are not followed, so a link loop
        # cannot repeat the walk; their targets are visited directly instead.
        run.files_ignored += 1
        return
    if not stat.S_ISREG(info.st_mode):
        run.files_invalid += 1
        run.add("invalid", reason="not_regular_file")
        yield path, None
        return
    yield path, target


def _scan_file(
    entry_path: Path,
    target: Path,
    run: DirectoryDryRun,
    registry: Any,
    schema_id: str | None,
    seen: set[tuple[int, int]],
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        # Non-blocking, no-follow open: a path swapped for a FIFO or a symlink
        # after the walk saw it is refused instead of blocking or escaping.
        handle = os.fdopen(os.open(target, flags), "rb")
    except OSError:
        run.files_invalid += 1
        run.add("invalid", reason="file_unreadable")
        return
    with handle:
        try:
            info = os.fstat(handle.fileno())
        except OSError:
            run.files_invalid += 1
            run.add("invalid", reason="file_unreadable")
            return
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            return
        seen.add(identity)
        if not stat.S_ISREG(info.st_mode):
            run.files_invalid += 1
            run.add("invalid", reason="not_regular_file")
            return
        if info.st_size > run.limits.max_file_bytes:
            run.files_invalid += 1
            run.add("invalid", reason="file_too_large")
            return
        run.files_scanned += 1
        if entry_path.suffix.lower() == ".jsonl":
            for line in _jsonl_rows(handle, run.limits.max_line_bytes):
                if line is None:
                    run.add("invalid", reason="line_too_long")
                    continue
                _count(_parse(line), run, registry, schema_id)
            return
        data = handle.read(run.limits.max_file_bytes + 1)
    if len(data) > run.limits.max_file_bytes:
        run.add("invalid", reason="file_too_large")
        return
    document = _parse(data)
    if isinstance(document, list):
        for item in document:
            _count(item, run, registry, schema_id)
        return
    _count(document, run, registry, schema_id)


_UNPARSABLE = object()


def _parse(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return _UNPARSABLE


def _jsonl_rows(handle: IO[bytes], max_line_bytes: int) -> Iterator[bytes | None]:
    while True:
        line = handle.readline(max_line_bytes + 1)
        if not line:
            return
        if len(line.rstrip(b"\r\n")) > max_line_bytes:
            while line and not line.endswith(b"\n"):
                line = handle.readline(max_line_bytes + 1)
            yield None
            continue
        stripped = line.strip()
        if stripped:
            yield stripped


def _count(
    record: Any, run: DirectoryDryRun, registry: Any, schema_id: str | None
) -> None:
    from openmed.core.record_migrations import RecordMigrationRefused

    if record is _UNPARSABLE or not isinstance(record, dict):
        run.add("invalid", reason="record_invalid")
        return
    try:
        report = registry.migrate_record(record, schema_id=schema_id).report
    except RecordMigrationRefused as exc:
        refusal = exc.refusal
        outcome = "no_path" if refusal.reason in _NO_PATH_REASONS else "invalid"
        run.add(
            outcome,
            schema_id=refusal.schema_id,
            version=refusal.from_version,
            reason=refusal.reason,
        )
        return
    if not report.steps:
        outcome = "current"
    elif report.lossy:
        outcome = "lossy"
    else:
        outcome = "migratable"
    run.add(outcome, schema_id=report.schema_id, version=report.source_version)


def _is_record_name(path: Path) -> bool:
    return path.suffix.lower() in RECORD_SUFFIXES


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _format(payload: dict[str, Any]) -> str:
    files = payload["files"]
    lines = [
        f"migrate records: dry run over {payload['records']} records in "
        f"{files['scanned']} files (no file was written)",
        "  "
        + ", ".join(f"{name}: {count}" for name, count in payload["outcomes"].items()),
    ]
    if files["invalid"] or files["ignored"]:
        lines.append(f"  files: {files['invalid']} invalid, {files['ignored']} ignored")
    for group in payload["groups"]:
        schema = group["schema_id"] or "unidentified"
        version = group["version"] or "-"
        lines.append(f"  {schema} {version}: {group['outcome']} {group['count']}")
    if payload["reasons"]:
        reasons = ", ".join(
            f"{reason} {count}" for reason, count in payload["reasons"].items()
        )
        lines.append(f"  reasons: {reasons}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_LINE_BYTES",
    "DEFAULT_MAX_RECORDS",
    "MAX_DEPTH",
    "OUTCOMES",
    "RECORD_SUFFIXES",
    "DirectoryDryRun",
    "WalkLimits",
    "add_records_arguments",
    "dry_run_directory",
    "handle_records",
]
