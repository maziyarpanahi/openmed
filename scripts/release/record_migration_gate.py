#!/usr/bin/env python3
"""Fail when a packaged schema version bump has no record migrator.

Bumping the ``schema_version`` of a packaged JSON Schema changes what stored
records look like, so every bump since the last release must either have a
registered migration chain in :mod:`openmed.core.record_migrations` or an
explicit ``no_record_migration`` exemption with a reason in
``scripts/release/record_migration_exemptions.json``. Every registered
migration edge must also name an existing ``before.json``/``after.json``
fixture pair under ``tests/fixtures/migrations/``.

The base side is read straight from Git (by default the newest ``v*`` tag
reachable from ``HEAD``) and is never imported. Only the candidate checkout's
migration registry is imported. Exit codes: ``0`` pass, ``1`` gate failure,
``2`` unreadable ref or exemption file.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXEMPTIONS = ROOT / "scripts" / "release" / "record_migration_exemptions.json"
DEFAULT_PACKAGE = "openmed"
WORKTREE_REF = "WORKTREE"
SCHEMA_SUFFIX = ".schema.json"
RELEASE_TAG_PATTERN = "v[0-9]*"
REPORT_SCHEMA_VERSION = 1

# A direct ``python scripts/release/record_migration_gate.py`` run otherwise
# puts only ``scripts/release`` first on ``sys.path``; pin imports to this
# checkout instead of an unrelated installed copy.
_root_string = str(ROOT)
if _root_string in sys.path:
    sys.path.remove(_root_string)
sys.path.insert(0, _root_string)

_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_ID_VERSION_RE = re.compile(r"(?:^|[/-])v?(\d+(?:\.\d+){0,2})(?=[/.]|$)")

FAILING_CODES = frozenset(
    {
        "bump_without_migrator",
        "exemption_invalid",
        "exemption_missing_reason",
        "fixture_missing",
    }
)


class GateError(RuntimeError):
    """Raised when a ref or the exemption file cannot be read."""


@dataclass(frozen=True)
class Bump:
    """A packaged schema whose record version changed since the base ref.

    Attributes:
        schema: Schema name, the file name without ``.schema.json``.
        from_version: Record version at the base ref.
        to_version: Record version in the candidate.
    """

    schema: str
    from_version: str
    to_version: str


@dataclass(frozen=True)
class Exemption:
    """A reviewed ``no_record_migration`` entry.

    Attributes:
        schema: Schema name.
        from_version: Record version the bump starts from.
        to_version: Record version the bump ends at.
        reason: Why stored records need no migration.
    """

    schema: str
    from_version: str
    to_version: str
    reason: str

    def covers(self, bump: Bump) -> bool:
        """Return whether this exemption names exactly ``bump``."""

        return (self.schema, self.from_version, self.to_version) == (
            bump.schema,
            bump.from_version,
            bump.to_version,
        )


@dataclass(frozen=True)
class Finding:
    """One gate result.

    Attributes:
        code: Controlled result code; codes in :data:`FAILING_CODES` fail.
        schema: Schema name or registry schema id, if any.
        from_version: Source version, if any.
        to_version: Target version, if any.
        detail: Fixture name or exemption reason, if any.
    """

    code: str
    schema: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    detail: str | None = None

    @property
    def failing(self) -> bool:
        """Return whether this finding fails the gate."""

        return self.code in FAILING_CODES

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-ready mapping."""

        return {
            "code": self.code,
            "schema": self.schema,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "detail": self.detail,
            "failing": self.failing,
        }


@dataclass(frozen=True)
class GateReport:
    """All findings of one gate run.

    Attributes:
        base_ref: Base ref, normally the last release tag.
        head_ref: Candidate ref or ``WORKTREE``.
        bumps: Schema version bumps found between the refs.
        findings: Results sorted by code, schema and versions.
    """

    base_ref: str
    head_ref: str
    bumps: tuple[Bump, ...]
    findings: tuple[Finding, ...]

    @property
    def failing(self) -> tuple[Finding, ...]:
        """Return the findings that fail the gate."""

        return tuple(finding for finding in self.findings if finding.failing)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable machine-readable report."""

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "passed": not self.failing,
            "bumps": [
                {
                    "schema": bump.schema,
                    "from_version": bump.from_version,
                    "to_version": bump.to_version,
                }
                for bump in self.bumps
            ],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _run_git(
    repo_root: Path, args: Sequence[str], *, stdin: bytes | None = None
) -> bytes:
    command = ["git", "-C", str(repo_root), *args]
    result = subprocess.run(command, capture_output=True, check=False, input=stdin)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise GateError(f"{' '.join(command)} failed: {stderr}")
    return result.stdout


def last_release_tag(repo_root: Path, ref: str = "HEAD") -> str:
    """Return the newest ``v*`` tag reachable from ``ref``.

    Args:
        repo_root: Git repository root.
        ref: Ref to describe.

    Returns:
        The tag name.

    Raises:
        GateError: When no release tag is reachable, as in a shallow clone.
    """

    try:
        described = _run_git(
            repo_root,
            ["describe", "--tags", "--abbrev=0", "--match", RELEASE_TAG_PATTERN, ref],
        )
    except GateError:
        raise GateError(
            f"no release tag is reachable from {ref}; fetch tags and history "
            "(actions/checkout fetch-depth: 0) or pass --base-ref"
        ) from None
    return described.decode("utf-8").strip()


def record_version(schema: Mapping[str, Any]) -> str | None:
    """Return the record version a packaged schema declares.

    Args:
        schema: Parsed schema document.

    Returns:
        The record ``schema_version`` property's ``const`` or highest
        ``enum`` value, else the top-level ``schema_version``, else the
        version in ``$id``, else ``None``.
    """

    record = (schema.get("properties") or {}).get("schema_version")
    if isinstance(record, Mapping):
        if "const" in record and not isinstance(record["const"], bool):
            return str(record["const"])
        if isinstance(record.get("enum"), list) and record["enum"]:
            return sorted(map(str, record["enum"]), key=_version_key)[-1]
    top = schema.get("schema_version")
    if top is not None and not isinstance(top, bool):
        return str(top)
    identifier = schema.get("$id")
    if isinstance(identifier, str):
        found = _ID_VERSION_RE.findall(identifier)
        if found:
            return found[-1]
    return None


def packaged_versions(
    repo_root: Path, ref: str, package: str = DEFAULT_PACKAGE
) -> dict[str, str | None]:
    """Return the record version of every packaged schema at ``ref``.

    Args:
        repo_root: Git repository root.
        ref: Git ref, or ``WORKTREE`` for the files on disk.
        package: Package directory to scan.

    Returns:
        Record versions keyed by schema name.

    Raises:
        GateError: When the ref cannot be read or a schema is not JSON.
    """

    raw: dict[str, bytes] = {}
    if ref == WORKTREE_REF:
        for path in sorted((repo_root / package).rglob(f"*{SCHEMA_SUFFIX}")):
            if path.is_file():
                raw[path.relative_to(repo_root).as_posix()] = path.read_bytes()
    else:
        listing = _run_git(repo_root, ["ls-tree", "-r", "-z", ref, "--", package])
        blobs: list[tuple[str, str]] = []
        for item in listing.decode("utf-8").split("\0"):
            meta, _, path = item.partition("\t")
            parts = meta.split()
            if len(parts) == 3 and parts[1] == "blob" and path.endswith(SCHEMA_SUFFIX):
                blobs.append((path, parts[2]))
        if blobs:
            output = _run_git(
                repo_root,
                ["cat-file", "--batch"],
                stdin="".join(f"{sha}\n" for _, sha in blobs).encode("ascii"),
            )
            offset = 0
            for path, _ in blobs:
                newline = output.index(b"\n", offset)
                size = int(output[offset:newline].split()[2])
                raw[path] = output[newline + 1 : newline + 1 + size]
                offset = newline + 1 + size + 1
    versions: dict[str, str | None] = {}
    for path, content in sorted(raw.items()):
        name = PurePosixPath(path).name[: -len(SCHEMA_SUFFIX)]
        try:
            document = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise GateError(f"{path} at {ref} is not valid JSON") from None
        versions[name] = (
            record_version(document) if isinstance(document, dict) else None
        )
    return versions


def find_bumps(
    base: Mapping[str, str | None], head: Mapping[str, str | None]
) -> tuple[Bump, ...]:
    """Return schemas present at both refs whose record version changed.

    Args:
        base: Record versions at the base ref.
        head: Record versions in the candidate.

    Returns:
        Bumps sorted by schema name.
    """

    return tuple(
        Bump(name, str(base[name]), str(head[name]))
        for name in sorted(set(base) & set(head))
        if base[name] is not None
        and head[name] is not None
        and base[name] != head[name]
    )


def load_exemptions(
    path: Path | None,
) -> tuple[tuple[Exemption, ...], tuple[Finding, ...]]:
    """Read the ``no_record_migration`` exemptions.

    Entries without a reason, or without a schema and both versions, are
    returned as failing findings instead of exemptions.

    Args:
        path: Exemption JSON file, or ``None`` for none.

    Returns:
        The valid exemptions and the findings for invalid entries.

    Raises:
        GateError: When the file cannot be read or has the wrong shape.
    """

    if path is None:
        return (), ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"could not read record migration exemptions: {exc}") from None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("no_record_migration"), list)
    ):
        raise GateError(
            "record migration exemptions must be an object with schema_version 1 "
            "and a no_record_migration list"
        )
    exemptions: list[Exemption] = []
    findings: list[Finding] = []
    for item in payload["no_record_migration"]:
        if not isinstance(item, dict):
            findings.append(Finding("exemption_invalid"))
            continue
        fields = {
            key: item.get(key) for key in ("schema", "from_version", "to_version")
        }
        if not all(isinstance(value, str) and value for value in fields.values()):
            findings.append(
                Finding(
                    "exemption_invalid",
                    schema=_text(fields["schema"]),
                    from_version=_text(fields["from_version"]),
                    to_version=_text(fields["to_version"]),
                )
            )
            continue
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            findings.append(Finding("exemption_missing_reason", **fields))
            continue
        exemptions.append(Exemption(reason=reason.strip(), **fields))
    return tuple(exemptions), tuple(findings)


def evaluate(
    bumps: Sequence[Bump],
    registry: Any,
    exemptions: Sequence[Exemption],
    *,
    repo_root: Path,
    base_ref: str = "base",
    head_ref: str = WORKTREE_REF,
    exemption_findings: Sequence[Finding] = (),
) -> GateReport:
    """Check every bump and every registered edge.

    Args:
        bumps: Schema version bumps since the base ref.
        registry: The candidate's record migration registry.
        exemptions: Valid ``no_record_migration`` entries.
        repo_root: Repository root holding ``tests/fixtures/migrations``.
        base_ref: Label of the base ref.
        head_ref: Label of the candidate ref.
        exemption_findings: Findings from :func:`load_exemptions`.

    Returns:
        The gate report.
    """

    from openmed.core.record_migrations import RecordMigrationRefused

    findings: list[Finding] = list(exemption_findings)
    used: set[Exemption] = set()
    for bump in bumps:
        exemption = next((item for item in exemptions if item.covers(bump)), None)
        try:
            chain = registry.migration_path(
                bump.schema, bump.from_version, bump.to_version
            )
        except RecordMigrationRefused:
            chain = ()
        if exemption is not None:
            used.add(exemption)
        versions = {
            "schema": bump.schema,
            "from_version": bump.from_version,
            "to_version": bump.to_version,
        }
        if chain:
            findings.append(
                Finding(
                    "migrated",
                    detail=" -> ".join(
                        [chain[0].from_version, *(edge.to_version for edge in chain)]
                    ),
                    **versions,
                )
            )
            if exemption is not None:
                findings.append(Finding("redundant_exemption", **versions))
        elif exemption is not None:
            findings.append(Finding("exempted", detail=exemption.reason, **versions))
        else:
            findings.append(Finding("bump_without_migrator", **versions))
    for exemption in exemptions:
        if exemption not in used:
            findings.append(
                Finding(
                    "stale_exemption",
                    schema=exemption.schema,
                    from_version=exemption.from_version,
                    to_version=exemption.to_version,
                )
            )
    for edge in registry.edges():
        before, after = edge.fixture_pair(repo_root)
        if not (before.is_file() and after.is_file()):
            findings.append(
                Finding(
                    "fixture_missing",
                    schema=edge.schema_id,
                    from_version=edge.from_version,
                    to_version=edge.to_version,
                    detail=edge.fixture,
                )
            )
    return GateReport(
        base_ref=base_ref,
        head_ref=head_ref,
        bumps=tuple(bumps),
        findings=tuple(
            sorted(
                findings,
                key=lambda item: (
                    item.code,
                    item.schema or "",
                    item.from_version or "",
                    item.to_version or "",
                ),
            )
        ),
    )


def run_gate(
    repo_root: Path,
    *,
    base_ref: str | None = None,
    head_ref: str = WORKTREE_REF,
    package: str = DEFAULT_PACKAGE,
    exemptions_path: Path | None = DEFAULT_EXEMPTIONS,
    registry: Any = None,
) -> GateReport:
    """Read both refs and evaluate the gate.

    Args:
        repo_root: Git repository root.
        base_ref: Base ref; the last release tag when omitted.
        head_ref: Candidate ref, or ``WORKTREE``.
        package: Package directory to scan.
        exemptions_path: Exemption file, or ``None`` for none.
        registry: Registry to check; the default registry when omitted.

    Returns:
        The gate report.
    """

    base = base_ref or last_release_tag(repo_root)
    exemptions, exemption_findings = load_exemptions(exemptions_path)
    if registry is None:
        from openmed.core.record_migrations import default_registry

        registry = default_registry()
    return evaluate(
        find_bumps(
            packaged_versions(repo_root, base, package),
            packaged_versions(repo_root, head_ref, package),
        ),
        registry,
        exemptions,
        repo_root=repo_root,
        base_ref=base,
        head_ref=head_ref,
        exemption_findings=exemption_findings,
    )


def render_summary(report: GateReport) -> str:
    """Render a concise, deterministic summary of the gate run.

    Args:
        report: The gate report.

    Returns:
        Human-readable lines.
    """

    lines = [
        f"Record migration gate: {report.base_ref} -> {report.head_ref}",
        f"bumps={len(report.bumps)}, failing={len(report.failing)}",
    ]
    for finding in report.findings:
        subject = finding.schema or "-"
        versions = (
            f" {finding.from_version} -> {finding.to_version}"
            if finding.from_version or finding.to_version
            else ""
        )
        detail = f" ({finding.detail})" if finding.detail else ""
        marker = "FAIL" if finding.failing else "ok"
        lines.append(f"  [{marker}] {finding.code}: {subject}{versions}{detail}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Baseline Git ref (default: newest v* tag reachable from HEAD)",
    )
    parser.add_argument(
        "--head-ref",
        default=WORKTREE_REF,
        help=f"Candidate Git ref or {WORKTREE_REF} for current files",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="Package directory")
    parser.add_argument(
        "--exemptions",
        type=Path,
        default=DEFAULT_EXEMPTIONS,
        help="no_record_migration exemption JSON",
    )
    parser.add_argument(
        "--json",
        dest="json_destination",
        metavar="PATH",
        help="Write the machine-readable report to PATH (use - for stdout)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, registry: Any = None) -> int:
    """Run the gate; exit 1 on failing findings and 2 on unreadable input."""

    args = build_parser().parse_args(argv)
    try:
        report = run_gate(
            args.repo_root,
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            package=args.package,
            exemptions_path=args.exemptions,
            registry=registry,
        )
    except GateError as exc:
        print(f"record-migration-gate: {exc}", file=sys.stderr)
        return 2
    if args.json_destination:
        rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        if args.json_destination == "-":
            sys.stdout.write(rendered)
        else:
            destination = Path(args.json_destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
    summary_stream = sys.stderr if args.json_destination == "-" else sys.stdout
    print(render_summary(report), file=summary_stream)
    if report.failing:
        print(
            "Each schema version bump needs a registered record migrator with a "
            "fixture pair or a no_record_migration exemption with a reason.",
            file=sys.stderr,
        )
        return 1
    return 0


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _version_key(value: str) -> tuple[Any, ...]:
    match = _SEMVER_RE.fullmatch(value)
    if match:
        return (0, *(int(part) for part in match.groups()))
    return (1, value)


if __name__ == "__main__":
    raise SystemExit(main())
