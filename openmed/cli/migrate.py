"""Offline ``openmed migrate`` commands for stored OpenMed records.

* ``check`` reads one JSON record, identifies its schema id and version,
  resolves a chain through the record migration registry, runs it in memory
  and prints the value-free migration report.
* ``records --dry-run`` does the same for every ``.json`` and ``.jsonl``
  record under a directory and prints aggregate counts per schema id, version
  and outcome (see :mod:`openmed.cli.migrate_records`).

Nothing is written: the input file is opened read-only and the migrated record
is discarded. Output contains schema ids, versions, counts, digests and JSON
Pointer field paths only; record values and file paths never appear.

Exit codes follow ``docs/output-formatting.md``: ``0`` when the record is
current or migratable, ``2`` (usage) when the record cannot be read or no
unique migration path exists, and ``1`` when a migrator fails.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, emit

if TYPE_CHECKING:
    from openmed.core.record_migrations import (
        MigrationRefusal,
        MigratorRegistry,
        RecordMigrationReport,
    )

MAX_RECORD_BYTES = 16 * 1024 * 1024
"""Largest record file ``migrate check`` reads."""

_RUNTIME_REASONS = frozenset(
    {"migrator_failed", "migrator_output_invalid", "version_mismatch"}
)


def add_migrate_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``openmed migrate`` command group.

    Args:
        subparsers: The top-level subparser collection.
    """

    parser = subparsers.add_parser(
        "migrate",
        help="Dry-run stored-record migrations without writing any file.",
    )
    commands = parser.add_subparsers(dest="migrate_command")

    check = commands.add_parser(
        "check",
        help="Dry-run the migration of one JSON record and report field paths.",
    )
    check.add_argument("record", type=Path, help="JSON file holding one record.")
    check.add_argument(
        "--schema",
        default=None,
        metavar="SCHEMA_ID",
        help="Schema id of the record; identified from its fields when omitted.",
    )
    check.add_argument(
        "--to-version",
        default=None,
        metavar="VERSION",
        help="Target version; the schema's current version when omitted.",
    )
    check.set_defaults(handler=_handle_check)

    from .migrate_records import add_records_arguments

    records = commands.add_parser(
        "records",
        help="Dry-run migrations over a directory and print counts only.",
    )
    add_records_arguments(records)
    records.set_defaults(handler=_handle_records)


def _registry() -> MigratorRegistry:
    from openmed.core.record_migrations import default_registry

    return default_registry()


def _handle_records(args: argparse.Namespace) -> int:
    from .migrate_records import handle_records

    return handle_records(args)


def _handle_check(args: argparse.Namespace) -> int:
    from openmed.core.record_migrations import RecordMigrationRefused

    record = read_record(args.record)
    try:
        result = _registry().migrate_record(
            record, schema_id=args.schema, to_version=args.to_version
        )
    except RecordMigrationRefused as exc:
        raise refusal_error(exc.refusal) from None
    payload = check_payload(result.report)
    return emit(args, payload, human=_format_check(payload))


def read_record(path: Path) -> dict[str, Any]:
    """Read one JSON object from ``path`` without modifying the file.

    Args:
        path: Record file supplied on the command line.

    Returns:
        The parsed record.

    Raises:
        CliError: With a usage exit code when the file is missing, not a
            regular file, larger than :data:`MAX_RECORD_BYTES`, not UTF-8
            JSON, or not a JSON object. Messages never include the path.
    """

    data = _read_regular_file(path)
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise CliError(
            "The record file is not valid UTF-8 JSON.",
            code="record_invalid",
            exit_code=EXIT_USAGE,
        ) from None
    if not isinstance(record, dict):
        raise CliError(
            "The record file must hold one JSON object.",
            code="record_invalid",
            exit_code=EXIT_USAGE,
        )
    return record


def refusal_error(refusal: MigrationRefusal) -> CliError:
    """Turn a typed migration refusal into a coded, value-free CLI error.

    Args:
        refusal: The refusal raised by the registry.

    Returns:
        A :class:`CliError` whose code is ``migration_<reason>`` and whose
        message names only the schema id and versions.
    """

    subject = refusal.schema_id or "unidentified schema"
    versions = " ".join(
        part
        for part in (
            f"from {refusal.from_version}" if refusal.from_version else "",
            f"to {refusal.to_version}" if refusal.to_version else "",
        )
        if part
    )
    message = f"Record migration refused ({refusal.reason}) for {subject}"
    if versions:
        message += f" {versions}"
    if refusal.error_type:
        message += f" after {refusal.error_type}"
    return CliError(
        message + ".",
        code=f"migration_{refusal.reason}",
        exit_code=EXIT_ERROR if refusal.reason in _RUNTIME_REASONS else EXIT_USAGE,
    )


def check_payload(report: RecordMigrationReport) -> dict[str, Any]:
    """Return the dry-run payload printed by ``migrate check``.

    Args:
        report: The value-free report of the in-memory migration.

    Returns:
        The report fields with ``status`` set to ``current`` or
        ``migratable`` and ``written`` set to ``False``.
    """

    payload = report.to_dict()
    payload["status"] = "migratable" if report.steps else "current"
    payload["dry_run"] = True
    payload["written"] = False
    return payload


def _read_regular_file(path: Path) -> bytes:
    def unreadable(message: str, code: str = "record_unreadable") -> CliError:
        return CliError(message, code=code, exit_code=EXIT_USAGE)

    try:
        # Check before opening so a FIFO or device never blocks the command,
        # then re-check the opened descriptor in case the path was swapped.
        if not stat.S_ISREG(path.stat().st_mode):
            raise unreadable("The record path is not a regular file.")
        with path.open("rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise unreadable("The record path is not a regular file.")
            if info.st_size > MAX_RECORD_BYTES:
                raise unreadable(
                    "The record file is larger than the supported limit.",
                    "record_too_large",
                )
            data = handle.read(MAX_RECORD_BYTES + 1)
    except CliError:
        raise
    except OSError:
        raise unreadable("The record file could not be read.") from None
    if len(data) > MAX_RECORD_BYTES:
        raise unreadable(
            "The record file is larger than the supported limit.", "record_too_large"
        )
    return data


def _format_check(payload: dict[str, Any]) -> str:
    schema = payload["schema_id"]
    if payload["status"] == "current":
        lines = [f"migrate check: current ({schema} {payload['source_version']})"]
    else:
        loss = "lossy" if payload["lossy"] else "lossless"
        lines = [
            f"migrate check: migratable ({schema} {payload['source_version']} -> "
            f"{payload['target_version']}, {loss})"
        ]
        for label in ("added", "moved", "changed", "dropped"):
            values = payload[label]
            if label == "moved":
                values = [f"{item['from']} -> {item['to']}" for item in values]
            lines.append(f"  {label}: {', '.join(values) if values else 'none'}")
        for step in payload["steps"]:
            kind = "lossless" if step["lossless"] else "lossy"
            lines.append(
                f"  step: {step['from_version']} -> {step['to_version']} ({kind})"
            )
    lines.append("dry run: no file was written")
    return "\n".join(lines)


__all__ = [
    "MAX_RECORD_BYTES",
    "add_migrate_command",
    "check_payload",
    "read_record",
    "refusal_error",
]
