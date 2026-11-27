#!/usr/bin/env python3
"""Check every bundled JSON Schema against the committed fingerprint snapshot.

The snapshot in ``scripts/release/schema_fingerprints.json`` has one entry per
``*.schema.json`` file in the ``openmed`` package, keyed by package path. Each
entry records the declared ``schema_version``, a SHA-256 fingerprint of the
canonical schema JSON, and the record's ``required`` and ``properties`` field
names (see :func:`openmed.core.schemas.build_packaged_schema_snapshot`).

``--check`` compares the current schemas with the snapshot. Removing a required
or declared property without a newer ``schema_version`` is breaking drift, and
so is deleting a schema that has an entry. A schema without an entry fails the
check too, so a new schema cannot ship unguarded. Adding properties, or
removing them together with a version bump, only changes the fingerprint.

``--write`` regenerates the snapshot from the current schemas. It refuses to
record breaking drift unless ``--allow-breaking`` is passed for an approved
break, so the removal stays visible in the snapshot diff under review.

The report lists only schema package paths, versions, fingerprints and field
names. It never prints schema descriptions, examples, defaults or local paths.

Usage::

    python scripts/release/check_schema_fingerprints.py --check
    python scripts/release/check_schema_fingerprints.py --check --json
    python scripts/release/check_schema_fingerprints.py --write

Exit codes: 0 when no schema has breaking drift and every schema has an entry,
1 on breaking drift, a removed schema or a schema without an entry (and when
``--write`` refuses breaking drift), 2 when the snapshot cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "scripts" / "release" / "schema_fingerprints.json"

# A direct ``python scripts/release/check_schema_fingerprints.py`` invocation
# otherwise puts only ``scripts/release`` at the front of ``sys.path``. Pin
# imports to the checkout being checked instead of an unrelated installation.
_root_string = str(ROOT)
if _root_string in sys.path:
    sys.path.remove(_root_string)
sys.path.insert(0, _root_string)

from openmed.core.schemas import (  # noqa: E402
    build_packaged_schema_snapshot,
    compare_all_packaged_schema_drift,
    load_packaged_schemas,
)

STATUS_UNCHANGED = "unchanged"
STATUS_CHANGED = "changed"
STATUS_BREAKING = "breaking"
STATUS_MISSING_ENTRY = "missing_entry"
STATUS_REMOVED = "removed"
STATUSES = (
    STATUS_UNCHANGED,
    STATUS_CHANGED,
    STATUS_BREAKING,
    STATUS_MISSING_ENTRY,
    STATUS_REMOVED,
)
FAILING_STATUSES = frozenset({STATUS_BREAKING, STATUS_MISSING_ENTRY, STATUS_REMOVED})
BREAKING_STATUSES = frozenset({STATUS_BREAKING, STATUS_REMOVED})

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_SNAPSHOT_ERROR = 2

_ENTRY_KEYS = frozenset({"fingerprint", "properties", "required", "schema_version"})


class SnapshotError(ValueError):
    """The fingerprint snapshot is missing or malformed.

    The message is a controlled reason code, never a path or file content.
    """


@dataclass(frozen=True)
class SchemaFinding:
    """Drift status of one bundled schema.

    Attributes:
        schema: Package path of the schema, such as
            ``openmed/core/schemas/json/span.schema.json``.
        status: One of :data:`STATUSES`.
        schema_version: Version the current schema declares, if any.
        snapshot_version: Version recorded in the snapshot, if any.
        fingerprint: Fingerprint of the current schema; ``None`` when removed.
        snapshot_fingerprint: Fingerprint in the snapshot; ``None`` when the
            schema has no entry.
        removed_required: Required field names missing from the current schema.
        removed_properties: Declared field names missing from the current
            schema.
    """

    schema: str
    status: str
    schema_version: int | str | None
    snapshot_version: int | str | None
    fingerprint: str | None
    snapshot_fingerprint: str | None
    removed_required: tuple[str, ...] = ()
    removed_properties: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible view of this finding."""

        return {
            "schema": self.schema,
            "status": self.status,
            "schema_version": self.schema_version,
            "snapshot_version": self.snapshot_version,
            "fingerprint": self.fingerprint,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "removed_required": list(self.removed_required),
            "removed_properties": list(self.removed_properties),
        }


@dataclass(frozen=True)
class FingerprintReport:
    """Drift findings for every current and snapshotted schema."""

    findings: tuple[SchemaFinding, ...]

    @property
    def failed(self) -> bool:
        """Whether any finding fails the check."""

        return any(finding.status in FAILING_STATUSES for finding in self.findings)

    @property
    def breaking(self) -> bool:
        """Whether any finding is a breaking change or a removed schema."""

        return any(finding.status in BREAKING_STATUSES for finding in self.findings)

    def summary(self) -> dict[str, int]:
        """Return the number of schemas per status plus the total."""

        counts = {status: 0 for status in STATUSES}
        for finding in self.findings:
            counts[finding.status] += 1
        return {"schemas": len(self.findings), **counts}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible view of this report."""

        return {
            "result": "fail" if self.failed else "pass",
            "summary": self.summary(),
            "schemas": [finding.to_dict() for finding in self.findings],
        }


def evaluate(
    snapshot: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
) -> FingerprintReport:
    """Compare schemas with a snapshot and classify every schema.

    Args:
        snapshot: Saved snapshot keyed by schema package path.
        schemas: Parsed schemas keyed by package path. Defaults to every
            bundled schema.

    Returns:
        One finding per schema in either ``schemas`` or ``snapshot``, sorted by
        package path.
    """

    current = load_packaged_schemas() if schemas is None else schemas
    results = compare_all_packaged_schema_drift(snapshot=snapshot, schemas=current)
    findings: list[SchemaFinding] = []
    for path, result in results.items():
        if path not in snapshot:
            status = STATUS_MISSING_ENTRY
        elif result.breaking_change:
            status = STATUS_BREAKING
        elif result.fingerprint != result.snapshot_fingerprint:
            status = STATUS_CHANGED
        else:
            status = STATUS_UNCHANGED
        findings.append(
            SchemaFinding(
                schema=path,
                status=status,
                schema_version=result.schema_version,
                snapshot_version=result.snapshot_version,
                fingerprint=result.fingerprint,
                snapshot_fingerprint=result.snapshot_fingerprint,
                removed_required=result.removed_required,
                removed_properties=result.removed_properties,
            )
        )
    for path in sorted(set(snapshot) - set(results)):
        entry = snapshot[path]
        findings.append(
            SchemaFinding(
                schema=path,
                status=STATUS_REMOVED,
                schema_version=None,
                snapshot_version=entry.get("schema_version"),
                fingerprint=None,
                snapshot_fingerprint=entry.get("fingerprint"),
                removed_required=tuple(entry.get("required") or ()),
                removed_properties=tuple(entry.get("properties") or ()),
            )
        )
    findings.sort(key=lambda finding: finding.schema)
    return FingerprintReport(findings=tuple(findings))


def load_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Read and validate a fingerprint snapshot file.

    Args:
        path: Snapshot file to read.

    Returns:
        The snapshot keyed by schema package path.

    Raises:
        SnapshotError: If the file is missing, is not JSON, or has malformed
            entries.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SnapshotError("snapshot_missing") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SnapshotError("snapshot_unreadable") from error
    if not isinstance(payload, dict):
        raise SnapshotError("snapshot_invalid")
    for key, entry in payload.items():
        if not _valid_entry(key, entry):
            raise SnapshotError("snapshot_invalid")
    return payload


def write_snapshot(path: Path, snapshot: Mapping[str, Mapping[str, Any]]) -> Path:
    """Write a snapshot as sorted, two-space indented JSON.

    Args:
        path: Destination file.
        snapshot: Snapshot built by
            :func:`openmed.core.schemas.build_packaged_schema_snapshot`.

    Returns:
        The written path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    path.write_bytes(payload.encode("utf-8"))
    return path


def render_text(report: FingerprintReport, *, verbose: bool = False) -> list[str]:
    """Render a report as one line per schema plus a summary line.

    Args:
        report: Report to render.
        verbose: Include unchanged schemas as well.

    Returns:
        Report lines without trailing newlines.
    """

    lines: list[str] = []
    for finding in report.findings:
        if finding.status == STATUS_UNCHANGED and not verbose:
            continue
        tokens = [
            finding.status,
            finding.schema,
            f"schema_version={_version_text(finding.schema_version)}",
            f"snapshot_version={_version_text(finding.snapshot_version)}",
            f"fingerprint={finding.fingerprint or 'none'}",
            f"snapshot_fingerprint={finding.snapshot_fingerprint or 'none'}",
        ]
        if finding.removed_required:
            tokens.append("removed_required=" + ",".join(finding.removed_required))
        if finding.removed_properties:
            tokens.append("removed_properties=" + ",".join(finding.removed_properties))
        lines.append(" ".join(tokens))
    summary = " ".join(f"{key}={value}" for key, value in report.summary().items())
    lines.append(f"{'fail' if report.failed else 'pass'} {summary}")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="compare the bundled schemas with the snapshot (default)",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="regenerate the snapshot from the bundled schemas",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="snapshot file (default: scripts/release/schema_fingerprints.json)",
    )
    parser.add_argument(
        "--allow-breaking",
        action="store_true",
        help="with --write, record an approved breaking change",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument(
        "--verbose", action="store_true", help="also list unchanged schemas"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check or regenerate the snapshot.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """

    args = parse_args(argv)
    if args.allow_breaking and not args.write:
        print("error: allow_breaking_requires_write", file=sys.stderr)
        return EXIT_SNAPSHOT_ERROR
    schemas = load_packaged_schemas()
    try:
        snapshot = load_snapshot(args.snapshot)
    except SnapshotError as error:
        if not (args.write and str(error) == "snapshot_missing"):
            print(f"error: {error}", file=sys.stderr)
            return EXIT_SNAPSHOT_ERROR
        snapshot = {}
    report = evaluate(snapshot, schemas)

    if args.write:
        if snapshot and report.breaking and not args.allow_breaking:
            _emit(report, as_json=args.json, verbose=args.verbose)
            print("error: breaking_drift_not_written", file=sys.stderr)
            return EXIT_DRIFT
        write_snapshot(args.snapshot, build_packaged_schema_snapshot(schemas))
        print(f"wrote schemas={len(schemas)}")
        return EXIT_OK

    _emit(report, as_json=args.json, verbose=args.verbose)
    return EXIT_DRIFT if report.failed else EXIT_OK


def _emit(report: FingerprintReport, *, as_json: bool, verbose: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    for line in render_text(report, verbose=verbose):
        print(line)


def _valid_entry(key: Any, entry: Any) -> bool:
    if not isinstance(key, str) or not isinstance(entry, dict):
        return False
    if set(entry) != _ENTRY_KEYS:
        return False
    version = entry["schema_version"]
    if version is not None and (
        isinstance(version, bool) or not isinstance(version, (int, str))
    ):
        return False
    return (
        isinstance(entry["fingerprint"], str)
        and _string_list(entry["properties"])
        and _string_list(entry["required"])
    )


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _version_text(version: int | str | None) -> str:
    return "none" if version is None else str(version)


if __name__ == "__main__":
    raise SystemExit(main())
