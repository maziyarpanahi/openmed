#!/usr/bin/env python3
"""Compare packaged JSON Schemas with the last release tag.

Every other schema gate compares the code with itself at one commit, so a
property can be deleted and ``schema-fingerprints.json`` regenerated in the
same pull request without any check failing. This script reads every packaged
``*.schema.json`` file at a base Git ref (by default the newest ``v*`` tag
reachable from ``HEAD``) and at the candidate (the working tree by default),
and classifies each schema as added, removed, unchanged, a compatible change,
a version bump, or a breaking change without a version bump.

Like ``api_surface_diff.py`` it reads files straight from Git and never imports
the package, so it is offline and free of import-time side effects. Breaking
changes and removals fail the check unless a reviewed entry in
``scripts/release/schema_break_allowlist.json`` names the schema with a reason
and the release that ships the break.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
WORKTREE_REF = "WORKTREE"
SCHEMA_SUFFIX = ".schema.json"
DEFAULT_PACKAGE = "openmed"
DEFAULT_ALLOWLIST = Path(__file__).with_name("schema_break_allowlist.json")
RELEASE_TAG_PATTERN = "v[0-9]*"

STATUSES = (
    "added",
    "removed",
    "unchanged",
    "compatible",
    "version_bump",
    "breaking",
)

_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_ID_VERSION_RE = re.compile(r"(?:^|[/-])v?(\d+(?:\.\d+){0,2})(?=[/.]|$)")
_LOWER_BOUNDS = ("minLength", "minItems", "minProperties", "minimum")
_UPPER_BOUNDS = ("maxLength", "maxItems", "maxProperties", "maximum")
_EXACT_CONSTRAINTS = (
    "pattern",
    "format",
    "multipleOf",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "uniqueItems",
)


class SchemaDiffError(RuntimeError):
    """Raised when a ref, schema file or allowlist cannot be read."""


@dataclass(frozen=True)
class SchemaFile:
    """One packaged JSON Schema read at a Git ref.

    Attributes:
        name: Schema name, the file name without ``.schema.json``.
        path: Repository-relative POSIX path of the file.
        document: Parsed schema document.
    """

    name: str
    path: str
    document: Mapping[str, Any] = field(repr=False)

    @property
    def fingerprint(self) -> str:
        """Return the canonical SHA-256 fingerprint of the document."""

        payload = json.dumps(
            self.document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Finding:
    """One narrowing change found inside a schema document.

    Attributes:
        code: Controlled change code, such as ``property_removed``.
        pointer: JSON Pointer of the changed keyword in the schema.
    """

    code: str
    pointer: str

    def to_dict(self) -> dict[str, str]:
        """Return the finding as a JSON-ready mapping."""

        return {"code": self.code, "pointer": self.pointer}


@dataclass(frozen=True)
class AllowlistEntry:
    """A reviewed break that may ship.

    Attributes:
        schema: Schema name the entry allows.
        reason: Why the break is acceptable.
        target_version: OpenMed release that ships the break.
    """

    schema: str
    reason: str
    target_version: str

    def to_dict(self) -> dict[str, str]:
        """Return the entry as a JSON-ready mapping."""

        return {
            "schema": self.schema,
            "reason": self.reason,
            "target_version": self.target_version,
        }


@dataclass(frozen=True)
class SchemaChange:
    """Classification of one schema between the base and candidate refs.

    Attributes:
        name: Schema name.
        status: One of :data:`STATUSES`.
        base_path: Path at the base ref, if present.
        head_path: Path at the candidate ref, if present.
        base_version: Declared version at the base ref.
        head_version: Declared version at the candidate ref.
        findings: Narrowing changes found between the two documents.
        allowed: The allowlist entry that accepts this break, if any.
    """

    name: str
    status: str
    base_path: str | None = None
    head_path: str | None = None
    base_version: str | None = None
    head_version: str | None = None
    findings: tuple[Finding, ...] = ()
    allowed: AllowlistEntry | None = None

    @property
    def failing(self) -> bool:
        """Return whether this change fails the gate."""

        return self.status in {"breaking", "removed"} and self.allowed is None

    def to_dict(self) -> dict[str, Any]:
        """Return the change as a JSON-ready mapping."""

        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "base_path": self.base_path,
            "head_path": self.head_path,
            "base_version": self.base_version,
            "head_version": self.head_version,
            "findings": [finding.to_dict() for finding in self.findings],
            "failing": self.failing,
        }
        if self.allowed is not None:
            payload["allowed"] = self.allowed.to_dict()
        return payload


@dataclass(frozen=True)
class SchemaReleaseDiff:
    """Comparison of every packaged schema between two refs.

    Attributes:
        base_ref: Base ref, normally the last release tag.
        head_ref: Candidate ref or ``WORKTREE``.
        package: Package directory that was scanned.
        changes: One classification per schema name, sorted by name.
        unused_allowlist: Allowlist entries that matched no break.
    """

    base_ref: str
    head_ref: str
    package: str
    changes: tuple[SchemaChange, ...]
    unused_allowlist: tuple[AllowlistEntry, ...] = ()

    @property
    def failing(self) -> tuple[SchemaChange, ...]:
        """Return the changes that fail the gate."""

        return tuple(change for change in self.changes if change.failing)

    @property
    def allowed(self) -> tuple[SchemaChange, ...]:
        """Return the breaks accepted by the allowlist."""

        return tuple(change for change in self.changes if change.allowed)

    def counts(self) -> dict[str, int]:
        """Return how many schemas have each status."""

        counts = dict.fromkeys(STATUSES, 0)
        for change in self.changes:
            counts[change.status] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return the stable machine-readable report."""

        return {
            "schema_version": SCHEMA_VERSION,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "package": self.package,
            "passed": not self.failing,
            "summary": {
                **self.counts(),
                "allowed": len(self.allowed),
                "failing": len(self.failing),
            },
            "changes": [
                change.to_dict()
                for change in self.changes
                if change.status != "unchanged"
            ],
            "unused_allowlist": [entry.to_dict() for entry in self.unused_allowlist],
        }


def _run_git(
    repo_root: Path, args: Sequence[str], *, stdin: bytes | None = None
) -> bytes:
    command = ["git", "-C", str(repo_root), *args]
    result = subprocess.run(command, capture_output=True, check=False, input=stdin)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise SchemaDiffError(f"{' '.join(command)} failed: {stderr}")
    return result.stdout


def last_release_tag(repo_root: Path, ref: str = "HEAD") -> str:
    """Return the newest ``v*`` tag reachable from ``ref``.

    Args:
        repo_root: Git repository root.
        ref: Ref to describe.

    Returns:
        The tag name.

    Raises:
        SchemaDiffError: When no release tag is reachable, for example in a
            shallow clone without tags.
    """

    try:
        described = _run_git(
            repo_root,
            ["describe", "--tags", "--abbrev=0", "--match", RELEASE_TAG_PATTERN, ref],
        )
    except SchemaDiffError:
        raise SchemaDiffError(
            "no release tag is reachable from "
            f"{ref}; fetch tags and history (actions/checkout fetch-depth: 0) "
            "or pass --base-ref"
        ) from None
    return described.decode("utf-8").strip()


def load_schemas(
    repo_root: Path, ref: str, package: str = DEFAULT_PACKAGE
) -> dict[str, SchemaFile]:
    """Read every packaged ``*.schema.json`` file at ``ref``.

    Args:
        repo_root: Git repository root.
        ref: Git ref, or ``WORKTREE`` for the files on disk.
        package: Package directory to scan.

    Returns:
        Schemas keyed by name.

    Raises:
        SchemaDiffError: When the ref cannot be read, a file is not a JSON
            object, or two files share a schema name.
    """

    raw: dict[str, bytes] = {}
    if ref == WORKTREE_REF:
        package_root = repo_root / package
        if not package_root.is_dir():
            raise SchemaDiffError(f"package directory not found: {package}")
        for path in sorted(package_root.rglob(f"*{SCHEMA_SUFFIX}")):
            if path.is_file() and "__pycache__" not in path.parts:
                raw[path.relative_to(repo_root).as_posix()] = path.read_bytes()
    else:
        listing = _run_git(repo_root, ["ls-tree", "-r", "-z", ref, "--", package])
        blobs: list[tuple[str, str]] = []
        for record in listing.decode("utf-8").split("\0"):
            if not record:
                continue
            meta, _, path = record.partition("\t")
            parts = meta.split()
            if len(parts) == 3 and parts[1] == "blob" and path.endswith(SCHEMA_SUFFIX):
                blobs.append((path, parts[2]))
        raw.update(_read_blobs(repo_root, blobs))
    schemas: dict[str, SchemaFile] = {}
    for path in sorted(raw):
        name = PurePosixPath(path).name[: -len(SCHEMA_SUFFIX)]
        if name in schemas:
            raise SchemaDiffError(
                f"two packaged schemas share the name {name}: "
                f"{schemas[name].path} and {path}"
            )
        try:
            document = json.loads(raw[path].decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise SchemaDiffError(f"{path} at {ref} is not valid JSON: {exc}") from None
        if not isinstance(document, dict):
            raise SchemaDiffError(f"{path} at {ref} is not a JSON object")
        schemas[name] = SchemaFile(name=name, path=path, document=document)
    return schemas


def _read_blobs(repo_root: Path, blobs: Sequence[tuple[str, str]]) -> dict[str, bytes]:
    if not blobs:
        return {}
    request = "".join(f"{sha}\n" for _, sha in blobs).encode("ascii")
    output = _run_git(repo_root, ["cat-file", "--batch"], stdin=request)
    contents: dict[str, bytes] = {}
    offset = 0
    for path, sha in blobs:
        newline = output.index(b"\n", offset)
        header = output[offset:newline].decode("ascii").split()
        if len(header) != 3 or header[0] != sha or header[1] != "blob":
            raise SchemaDiffError(f"could not read {path} from Git")
        size = int(header[2])
        start = newline + 1
        contents[path] = output[start : start + size]
        offset = start + size + 1
    return contents


def version_markers(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return every version marker a schema declares.

    Args:
        schema: Parsed schema document.

    Returns:
        The top-level ``schema_version``, the record ``schema_version``
        property's ``const`` or ``enum``, and the version embedded in ``$id``.
    """

    record = (schema.get("properties") or {}).get("schema_version")
    record_version: Any = None
    if isinstance(record, Mapping):
        if "const" in record:
            record_version = record["const"]
        elif isinstance(record.get("enum"), list):
            record_version = sorted(map(str, record["enum"]), key=_version_key)
    identifier = schema.get("$id")
    id_versions = (
        _ID_VERSION_RE.findall(identifier) if isinstance(identifier, str) else []
    )
    return {
        "schema_version": schema.get("schema_version"),
        "record_version": record_version,
        "id_version": id_versions[-1] if id_versions else None,
    }


def declared_version(schema: Mapping[str, Any]) -> str | None:
    """Return the most specific version a schema declares, as text.

    Args:
        schema: Parsed schema document.

    Returns:
        The record ``schema_version`` const or highest enum value, else the
        top-level ``schema_version``, else the ``$id`` version, else ``None``.
    """

    markers = version_markers(schema)
    record = markers["record_version"]
    if isinstance(record, list):
        return record[-1] if record else None
    for value in (record, markers["schema_version"], markers["id_version"]):
        if value is not None and not isinstance(value, bool):
            return str(value)
    return None


def breaking_findings(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> tuple[Finding, ...]:
    """Return the narrowing changes from ``before`` to ``after``.

    A change is breaking when a record that the older schema described may no
    longer be valid, or a field a reader relied on may be gone: removed
    properties or definitions, newly required fields, narrowed types, removed
    enum values, changed constants, closed ``additionalProperties``, new
    ``allOf`` constraints, removed ``anyOf``/``oneOf`` branches and tightened
    bounds or patterns.

    Args:
        before: Schema document at the base ref.
        after: Schema document at the candidate ref.

    Returns:
        Findings sorted by pointer and code.
    """

    findings: list[Finding] = []
    _compare(before, after, "", findings, 0)
    return tuple(sorted(set(findings), key=lambda item: (item.pointer, item.code)))


def _compare(
    before: Any, after: Any, pointer: str, findings: list[Finding], depth: int
) -> None:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        if before != after and isinstance(after, bool) and after is False:
            findings.append(Finding("schema_closed", pointer or "/"))
        return
    if depth > 64:
        return

    def add(code: str, *parts: str) -> None:
        findings.append(Finding(code, _join(pointer, *parts)))

    before_types = _types(before.get("type"))
    after_types = _types(after.get("type"))
    if after_types is not None and (
        before_types is None or not before_types <= after_types
    ):
        add("type_narrowed", "type")
    if isinstance(after.get("enum"), list):
        before_enum = before.get("enum")
        if not isinstance(before_enum, list):
            add("enum_added", "enum")
        elif any(value not in after["enum"] for value in before_enum):
            add("enum_value_removed", "enum")
    if "const" in after and before.get("const", object()) != after["const"]:
        add("const_changed", "const")
    for name in sorted(
        set(after.get("required") or ()) - set(before.get("required") or ())
    ):
        add("required_added", "required", str(name))
    before_props = before.get("properties") or {}
    after_props = after.get("properties") or {}
    if isinstance(before_props, Mapping) and isinstance(after_props, Mapping):
        for name in sorted(set(before_props) - set(after_props)):
            add("property_removed", "properties", str(name))
        for name in sorted(set(before_props) & set(after_props)):
            _compare(
                before_props[name],
                after_props[name],
                _join(pointer, "properties", str(name)),
                findings,
                depth + 1,
            )
    before_extra = before.get("additionalProperties", True)
    after_extra = after.get("additionalProperties", True)
    if after_extra is False and before_extra is not False:
        add("additional_properties_closed", "additionalProperties")
    elif isinstance(before_extra, Mapping) and isinstance(after_extra, Mapping):
        _compare(
            before_extra,
            after_extra,
            _join(pointer, "additionalProperties"),
            findings,
            depth + 1,
        )
    for keyword in ("items", "contains", "propertyNames", "not"):
        if isinstance(before.get(keyword), Mapping) and isinstance(
            after.get(keyword), Mapping
        ):
            _compare(
                before[keyword],
                after[keyword],
                _join(pointer, keyword),
                findings,
                depth + 1,
            )
    for keyword in ("$defs", "definitions"):
        before_defs = before.get(keyword) or {}
        after_defs = after.get(keyword) or {}
        if not isinstance(before_defs, Mapping) or not isinstance(after_defs, Mapping):
            continue
        for name in sorted(set(before_defs) - set(after_defs)):
            add("definition_removed", keyword, str(name))
        for name in sorted(set(before_defs) & set(after_defs)):
            _compare(
                before_defs[name],
                after_defs[name],
                _join(pointer, keyword, str(name)),
                findings,
                depth + 1,
            )
    for keyword in ("allOf", "anyOf", "oneOf"):
        before_items = before.get(keyword) or []
        after_items = after.get(keyword) or []
        if not isinstance(before_items, list) or not isinstance(after_items, list):
            continue
        if keyword == "allOf" and len(after_items) > len(before_items):
            add("all_of_constraint_added", keyword)
        if keyword != "allOf" and 0 < len(after_items) < len(before_items):
            add("branch_removed", keyword)
        if keyword != "allOf" and after_items and not before_items:
            add("branch_constraint_added", keyword)
        for index in range(min(len(before_items), len(after_items))):
            _compare(
                before_items[index],
                after_items[index],
                _join(pointer, keyword, str(index)),
                findings,
                depth + 1,
            )
    for keyword in _LOWER_BOUNDS:
        if _number(after.get(keyword)) is not None and (
            _number(before.get(keyword)) is None or after[keyword] > before[keyword]
        ):
            add("constraint_narrowed", keyword)
    for keyword in _UPPER_BOUNDS:
        if _number(after.get(keyword)) is not None and (
            _number(before.get(keyword)) is None or after[keyword] < before[keyword]
        ):
            add("constraint_narrowed", keyword)
    for keyword in _EXACT_CONSTRAINTS:
        if keyword in after and before.get(keyword, object()) != after[keyword]:
            add("constraint_narrowed", keyword)


def _types(value: Any) -> set[str] | None:
    if isinstance(value, str):
        types = {value}
    elif isinstance(value, list):
        types = {str(item) for item in value}
    else:
        return None
    if "number" in types:
        types.add("integer")
    return types


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _join(pointer: str, *parts: str) -> str:
    escaped = (part.replace("~", "~0").replace("/", "~1") for part in parts)
    return pointer + "".join(f"/{part}" for part in escaped)


def _version_key(value: str) -> tuple[Any, ...]:
    match = _SEMVER_RE.fullmatch(value)
    if match:
        return (0, *(int(part) for part in match.groups()))
    return (1, value)


def load_allowlist(path: Path | None) -> tuple[AllowlistEntry, ...]:
    """Read and validate the reviewed-break allowlist.

    Args:
        path: Allowlist JSON file, or ``None`` for no allowlist.

    Returns:
        The validated entries.

    Raises:
        SchemaDiffError: When the file is unreadable or an entry lacks a
            schema name, a reason, or a semantic ``target_version``, or names
            a schema twice.
    """

    if path is None:
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SchemaDiffError(f"could not read schema break allowlist: {exc}") from None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("entries"), list)
    ):
        raise SchemaDiffError(
            "schema break allowlist must be an object with schema_version 1 "
            "and an entries list"
        )
    entries: list[AllowlistEntry] = []
    for index, item in enumerate(payload["entries"]):
        if not isinstance(item, dict):
            raise SchemaDiffError(f"allowlist entry {index} must be an object")
        schema = item.get("schema")
        reason = item.get("reason")
        target = item.get("target_version")
        if not isinstance(schema, str) or not schema.strip():
            raise SchemaDiffError(f"allowlist entry {index} needs a schema name")
        if not isinstance(reason, str) or not reason.strip():
            raise SchemaDiffError(f"allowlist entry {index} ({schema}) needs a reason")
        if not isinstance(target, str) or not _SEMVER_RE.fullmatch(target):
            raise SchemaDiffError(
                f"allowlist entry {index} ({schema}) needs a semantic target_version"
            )
        entries.append(
            AllowlistEntry(schema=schema, reason=reason.strip(), target_version=target)
        )
    names = [entry.schema for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SchemaDiffError(
            f"schema break allowlist names a schema twice: {', '.join(duplicates)}"
        )
    return tuple(entries)


def diff_schemas(
    base: Mapping[str, SchemaFile],
    head: Mapping[str, SchemaFile],
    *,
    base_ref: str = "base",
    head_ref: str = "head",
    package: str = DEFAULT_PACKAGE,
    allowlist: Sequence[AllowlistEntry] = (),
) -> SchemaReleaseDiff:
    """Classify every schema between two loaded schema sets.

    Args:
        base: Schemas at the base ref.
        head: Schemas at the candidate ref.
        base_ref: Label of the base ref.
        head_ref: Label of the candidate ref.
        package: Package directory label.
        allowlist: Reviewed breaks that may ship.

    Returns:
        The comparison report.
    """

    allowed = {entry.schema: entry for entry in allowlist}
    used: set[str] = set()
    changes: list[SchemaChange] = []
    for name in sorted(set(base) | set(head)):
        before = base.get(name)
        after = head.get(name)
        if before is None and after is not None:
            changes.append(
                SchemaChange(
                    name,
                    "added",
                    head_path=after.path,
                    head_version=declared_version(after.document),
                )
            )
            continue
        if before is None:
            continue
        if after is None:
            entry = allowed.get(name)
            if entry is not None:
                used.add(name)
            changes.append(
                SchemaChange(
                    name,
                    "removed",
                    base_path=before.path,
                    base_version=declared_version(before.document),
                    allowed=entry,
                )
            )
            continue
        common = {
            "base_path": before.path,
            "head_path": after.path,
            "base_version": declared_version(before.document),
            "head_version": declared_version(after.document),
        }
        if before.fingerprint == after.fingerprint:
            changes.append(SchemaChange(name, "unchanged", **common))
            continue
        findings = breaking_findings(before.document, after.document)
        if version_markers(before.document) != version_markers(after.document):
            status = "version_bump"
        elif findings:
            status = "breaking"
        else:
            status = "compatible"
        entry = allowed.get(name) if status == "breaking" else None
        if entry is not None:
            used.add(name)
        changes.append(
            SchemaChange(name, status, findings=findings, allowed=entry, **common)
        )
    return SchemaReleaseDiff(
        base_ref=base_ref,
        head_ref=head_ref,
        package=package,
        changes=tuple(changes),
        unused_allowlist=tuple(
            entry for entry in allowlist if entry.schema not in used
        ),
    )


def compare_refs(
    repo_root: Path,
    base_ref: str | None = None,
    head_ref: str = WORKTREE_REF,
    *,
    package: str = DEFAULT_PACKAGE,
    allowlist: Sequence[AllowlistEntry] = (),
) -> SchemaReleaseDiff:
    """Load and compare packaged schemas at two refs.

    Args:
        repo_root: Git repository root.
        base_ref: Base ref; the last release tag reachable from ``HEAD`` when
            omitted.
        head_ref: Candidate ref, or ``WORKTREE`` for the files on disk.
        package: Package directory to scan.
        allowlist: Reviewed breaks that may ship.

    Returns:
        The comparison report.
    """

    base = base_ref or last_release_tag(repo_root)
    return diff_schemas(
        load_schemas(repo_root, base, package),
        load_schemas(repo_root, head_ref, package),
        base_ref=base,
        head_ref=head_ref,
        package=package,
        allowlist=allowlist,
    )


def render_summary(diff: SchemaReleaseDiff) -> str:
    """Render a concise, deterministic release-log summary.

    Args:
        diff: The comparison report.

    Returns:
        Human-readable lines.
    """

    counts = diff.counts()
    lines = [
        f"Packaged JSON Schemas: {diff.base_ref} -> {diff.head_ref} ({diff.package})",
        ", ".join(f"{status}={counts[status]}" for status in STATUSES)
        + f", allowed={len(diff.allowed)}, failing={len(diff.failing)}",
    ]
    for heading, selected in (
        ("Failing", diff.failing),
        ("Allowed", diff.allowed),
        (
            "Version bumps",
            tuple(c for c in diff.changes if c.status == "version_bump"),
        ),
    ):
        if not selected:
            continue
        lines.append(f"{heading}:")
        for change in selected:
            detail = f"{change.base_version} -> {change.head_version}"
            lines.append(f"  - {change.status}: {change.name} ({detail})")
            for finding in change.findings[:10]:
                lines.append(f"      {finding.code} at {finding.pointer}")
            if len(change.findings) > 10:
                lines.append(f"      ... {len(change.findings) - 10} more")
            if change.allowed is not None:
                lines.append(
                    f"      allowed for {change.allowed.target_version}: "
                    f"{change.allowed.reason}"
                )
    for entry in diff.unused_allowlist:
        lines.append(f"Unused allowlist entry: {entry.schema}")
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
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository root (default: current directory)",
    )
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="Package directory")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="Reviewed-break allowlist JSON",
    )
    parser.add_argument(
        "--json",
        dest="json_destination",
        metavar="PATH",
        help="Write the machine-readable report to PATH (use - for stdout)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison; exit 1 on unallowlisted breaks, 2 on errors."""

    args = build_parser().parse_args(argv)
    try:
        allowlist = load_allowlist(args.allowlist)
        diff = compare_refs(
            args.repo_root,
            args.base_ref,
            args.head_ref,
            package=args.package,
            allowlist=allowlist,
        )
    except SchemaDiffError as exc:
        print(f"schema-release-diff: {exc}", file=sys.stderr)
        return 2
    if args.json_destination:
        rendered = json.dumps(diff.to_dict(), indent=2, sort_keys=True) + "\n"
        if args.json_destination == "-":
            sys.stdout.write(rendered)
        else:
            destination = Path(args.json_destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
    summary_stream = sys.stderr if args.json_destination == "-" else sys.stdout
    print(render_summary(diff), file=summary_stream)
    if diff.failing:
        print(
            "Breaking schema changes need a version bump or a reviewed entry in "
            "the schema break allowlist.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
