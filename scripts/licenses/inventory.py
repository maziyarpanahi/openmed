#!/usr/bin/env python3
"""Check the checked-in dependency license inventory without network access.

The inventory is deliberately reviewed data, rather than package metadata
looked up at runtime.  A package is safe only when every token in its
SPDX-like expression is in the local permissive allowlist.  Missing,
malformed, and restricted expressions therefore fail closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "docs/security/license-inventory.md"
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"

NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
NORMALIZED_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LICENSE_TOKEN_RE = re.compile(r"[A-Za-z0-9.+-]+")
MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


class LicenseClass(str):
    """Stable classification values emitted by the gate."""

    PERMISSIVE = "permissive"
    UNKNOWN = "unknown"
    RESTRICTED = "restricted"


PERMISSIVE_LICENSES = frozenset(
    {
        "0bsd",
        "apache-2",
        "apache-2-0",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-3-0",
        "cc-by-4-0",
        "cc0-1-0",
        "hpnd",
        "isc",
        "mit",
        "mpl-2-0",
        "unlicense",
        "zlib",
    }
)

RESTRICTED_LICENSES = frozenset(
    {
        "agpl",
        "agpl-1-0",
        "agpl-3-0",
        "agpl-3-0-only",
        "agpl-3-0-or-later",
        "bsl-1-1",
        "bsd-4-clause",
        "commons-clause",
        "elastic-2-0",
        "gpl",
        "gpl-1-0",
        "gpl-2-0",
        "gpl-2-0-only",
        "gpl-2-0-or-later",
        "gpl-3-0",
        "gpl-3-0-only",
        "gpl-3-0-or-later",
        "lgpl",
        "lgpl-2-0",
        "lgpl-2-0-only",
        "lgpl-2-0-or-later",
        "lgpl-2-1",
        "lgpl-2-1-only",
        "lgpl-2-1-or-later",
        "lgpl-3-0",
        "lgpl-3-0-only",
        "lgpl-3-0-or-later",
        "polyform-strict-1-0",
        "sspl-1-0",
    }
)

UNKNOWN_LICENSES = frozenset(
    {
        "",
        "none",
        "noassertion",
        "unknown",
        "unlicensed",
    }
)

RESTRICTED_TEXT_MARKERS = (
    re.compile(r"\b(?:a?gpl|lgpl)(?:[-\s]|$)", re.IGNORECASE),
    re.compile(r"\b(?:business source|commons clause|elastic|polyform)", re.IGNORECASE),
    re.compile(
        r"\b(?:proprietary|commercial|source available|source-available)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsspl(?:[-\s]|$)", re.IGNORECASE),
)

LICENSE_OPERATORS = frozenset({"and", "or", "with"})
SENSITIVE_IDENTIFIER_RE = re.compile(
    r"(?:sk-[a-z0-9]|(?:api|secret|token|password|ssn|mrn|patient)[-_:=])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InventoryEntry:
    """One locally reviewed dependency license declaration."""

    name: str
    license_expression: str
    scope: str = ""
    version: str = ""
    line_number: int | None = None

    @property
    def license(self) -> str:
        """Return the SPDX-like expression for compatibility with callers."""

        return self.license_expression


@dataclass(frozen=True)
class InventoryRecord:
    """A dependency entry plus its fail-closed license classification."""

    entry: InventoryEntry
    classification: str
    reason: str

    @property
    def name(self) -> str:
        """Return the normalized dependency name."""

        return self.entry.name

    @property
    def license_expression(self) -> str:
        """Return the original local expression for programmatic inspection."""

        return self.entry.license_expression

    def to_safe_dict(self) -> dict[str, str]:
        """Return report data without license text, versions, or source values."""

        return {
            "classification": self.classification,
            "name": safe_identifier(self.name),
            "reason": self.reason,
        }


class InventoryError(ValueError):
    """Raised when the checked-in inventory cannot be safely interpreted."""


def safe_identifier(value: str) -> str:
    """Return a report-safe dependency identifier.

    Normal package names are retained for remediation. A malformed or
    synthetic identifier that resembles a credential or patient identifier is
    replaced by a stable digest so it cannot leak through logs or reports.
    """

    if not SENSITIVE_IDENTIFIER_RE.search(value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"dependency-{digest}"


def normalize_name(value: str) -> str:
    """Normalize a distribution name using PEP 503's separators."""

    if not isinstance(value, str) or not value.strip():
        raise InventoryError("dependency name is missing")
    stripped = value.strip()
    if not NORMALIZED_NAME_RE.fullmatch(stripped):
        raise InventoryError("dependency name contains invalid characters")
    return re.sub(r"[-_.]+", "-", stripped).lower()


def dependency_name(requirement: str) -> str:
    """Extract and normalize a dependency name from a PEP 508 declaration."""

    if not isinstance(requirement, str):
        raise InventoryError("dependency declaration is not text")
    match = NAME_RE.match(requirement)
    if not match:
        raise InventoryError("dependency declaration has no parseable name")
    return normalize_name(match.group(1))


def _normalize_license_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def classify_license(license_expression: object) -> str:
    """Classify a license expression using only the local allowlists.

    Compound expressions are conservative: a restricted token wins over an
    unknown token, and an unknown token wins over an otherwise permissive
    expression.  This makes ``MIT OR GPL-3.0-only`` restricted rather than
    silently treating the first option as approval.
    """

    if not isinstance(license_expression, str):
        return LicenseClass.UNKNOWN

    expression = license_expression.strip()
    if not expression:
        return LicenseClass.UNKNOWN

    normalized_expression = _normalize_license_token(expression)
    if normalized_expression in UNKNOWN_LICENSES:
        return LicenseClass.UNKNOWN

    if any(marker.search(expression) for marker in RESTRICTED_TEXT_MARKERS):
        return LicenseClass.RESTRICTED

    if re.search(r"[^A-Za-z0-9.+()_\-\s]", expression):
        return LicenseClass.UNKNOWN

    tokens = [
        _normalize_license_token(token)
        for token in LICENSE_TOKEN_RE.findall(expression)
    ]
    if not tokens:
        return LicenseClass.UNKNOWN

    license_tokens: list[str] = []
    expecting_license = True
    for token in tokens:
        if not token:
            continue
        is_operator = token in LICENSE_OPERATORS
        if expecting_license == is_operator:
            return LicenseClass.UNKNOWN
        if is_operator:
            expecting_license = True
        else:
            license_tokens.append(token)
            expecting_license = False
    if expecting_license:
        return LicenseClass.UNKNOWN
    if not license_tokens:
        return LicenseClass.UNKNOWN

    if any(token in RESTRICTED_LICENSES for token in license_tokens):
        return LicenseClass.RESTRICTED
    if any(token not in PERMISSIVE_LICENSES for token in license_tokens):
        return LicenseClass.UNKNOWN
    return LicenseClass.PERMISSIVE


def _classification_reason(classification: str) -> str:
    if classification == LicenseClass.PERMISSIVE:
        return "permissive SPDX expression"
    if classification == LicenseClass.RESTRICTED:
        return "restricted or incompatible license expression"
    return "missing or unrecognized license expression"


def _clean_cell(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    return cleaned.replace(r"\|", "|").strip()


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]
    return [_clean_cell(cell) for cell in re.split(r"(?<!\\)\|", stripped)]


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _entry_from_mapping(
    value: object,
    index: int,
    *,
    line_number: int | None = None,
) -> InventoryEntry:
    if not isinstance(value, Mapping):
        raise InventoryError(f"inventory entry #{index} must be an object")

    name_value = value.get("name", value.get("package", value.get("dependency")))
    if not isinstance(name_value, str) or not name_value.strip():
        raise InventoryError(f"inventory entry #{index} has no dependency name")
    name = normalize_name(name_value)

    license_value = value.get("license", value.get("license_expression", ""))
    license_expression = license_value.strip() if isinstance(license_value, str) else ""
    scope_value = value.get("scope", value.get("group", ""))
    scope = scope_value.strip() if isinstance(scope_value, str) else ""
    version_value = value.get("version", "")
    version = version_value.strip() if isinstance(version_value, str) else ""
    return InventoryEntry(name, license_expression, scope, version, line_number)


def _deduplicate_entries(
    entries: Iterable[InventoryEntry],
) -> tuple[InventoryEntry, ...]:
    by_name: dict[str, InventoryEntry] = {}
    for entry in entries:
        normalized_name = normalize_name(entry.name)
        if normalized_name != entry.name:
            entry = InventoryEntry(
                name=normalized_name,
                license_expression=entry.license_expression,
                scope=entry.scope,
                version=entry.version,
                line_number=entry.line_number,
            )
        previous = by_name.get(entry.name)
        if previous is None:
            by_name[entry.name] = entry
            continue
        if previous.license_expression != entry.license_expression:
            raise InventoryError("inventory contains conflicting license declarations")
        if (entry.scope, entry.version) < (previous.scope, previous.version):
            by_name[entry.name] = entry
    return tuple(sorted(by_name.values(), key=lambda item: item.name))


def _parse_markdown_inventory(text: str) -> tuple[InventoryEntry, ...]:
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        if "|" not in line:
            continue
        headers = _split_markdown_row(line)
        header_keys = [_header_key(header) for header in headers]
        name_index = next(
            (
                position
                for position, key in enumerate(header_keys)
                if key in {"dependency", "distribution", "name", "package"}
            ),
            None,
        )
        license_index = next(
            (
                position
                for position, key in enumerate(header_keys)
                if key in {"license", "license_expression", "spdx"}
            ),
            None,
        )
        if name_index is None or license_index is None:
            continue
        if not all(
            MARKDOWN_SEPARATOR_RE.fullmatch(cell)
            for cell in _split_markdown_row(lines[index + 1])
        ):
            continue

        scope_index = next(
            (
                position
                for position, key in enumerate(header_keys)
                if key in {"scope", "group", "extra"}
            ),
            None,
        )
        version_index = next(
            (position for position, key in enumerate(header_keys) if key == "version"),
            None,
        )
        entries: list[InventoryEntry] = []
        for line_number, row in enumerate(lines[index + 2 :], start=index + 3):
            if "|" not in row or not row.strip():
                if entries:
                    break
                continue
            cells = _split_markdown_row(row)
            if all(not cell for cell in cells):
                continue
            if all(MARKDOWN_SEPARATOR_RE.fullmatch(cell) for cell in cells):
                continue

            def cell_at(position: int | None) -> str:
                return (
                    cells[position]
                    if position is not None and position < len(cells)
                    else ""
                )

            entries.append(
                _entry_from_mapping(
                    {
                        "name": cell_at(name_index),
                        "license": cell_at(license_index),
                        "scope": cell_at(scope_index),
                        "version": cell_at(version_index),
                    },
                    len(entries) + 1,
                    line_number=line_number,
                )
            )
        if not entries:
            raise InventoryError("inventory table has no dependency rows")
        return _deduplicate_entries(entries)

    raise InventoryError("inventory document has no dependency license table")


def _rows_from_payload(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise InventoryError("inventory payload must be an object or array")

    for key in ("dependencies", "packages", "inventory", "entries"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
        if isinstance(rows, Mapping):
            return [
                {"name": name, **details}
                for name, details in rows.items()
                if isinstance(details, Mapping)
            ]
    raise InventoryError("inventory payload has no dependency entries")


def _parse_structured_inventory(text: str, suffix: str) -> tuple[InventoryEntry, ...]:
    try:
        if suffix == ".json":
            payload: object = json.loads(text)
        else:
            payload = tomllib.loads(text)
    except (ValueError, TypeError):
        raise InventoryError("inventory payload is malformed") from None

    entries = [
        _entry_from_mapping(row, index)
        for index, row in enumerate(_rows_from_payload(payload), start=1)
    ]
    parsed = _deduplicate_entries(entries)
    if not parsed:
        raise InventoryError("inventory has no dependency entries")
    return parsed


def _parse_csv_inventory(text: str) -> tuple[InventoryEntry, ...]:
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise InventoryError("inventory CSV has no dependency rows")
    entries = [
        _entry_from_mapping(row, index) for index, row in enumerate(rows, start=1)
    ]
    parsed = _deduplicate_entries(entries)
    if not parsed:
        raise InventoryError("inventory has no dependency entries")
    return parsed


def parse_inventory(path: Path = DEFAULT_INVENTORY) -> tuple[InventoryEntry, ...]:
    """Parse a local Markdown, JSON, TOML, or CSV inventory.

    The parser only opens the supplied local path.  It never resolves package
    metadata, imports dependencies, or contacts a registry.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise InventoryError("could not read the checked-in inventory") from None

    suffix = path.suffix.casefold()
    if suffix == ".md":
        return _parse_markdown_inventory(text)
    if suffix == ".json":
        return _parse_structured_inventory(text, suffix)
    if suffix == ".toml":
        return _parse_structured_inventory(text, suffix)
    if suffix == ".csv":
        return _parse_csv_inventory(text)
    raise InventoryError("inventory format is not supported")


def parse_project_dependencies(path: Path = DEFAULT_PYPROJECT) -> tuple[str, ...]:
    """Return non-development dependency names declared in a local pyproject."""

    try:
        project_data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise InventoryError("could not read the checked-in project metadata") from None

    project = project_data.get("project")
    if not isinstance(project, Mapping):
        raise InventoryError("project metadata has no project table")

    requirements: list[str] = []
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise InventoryError("project dependencies must be a list")
    if not all(isinstance(requirement, str) for requirement in dependencies):
        raise InventoryError("project dependency declaration must be text")
    requirements.extend(dependencies)

    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, Mapping):
        raise InventoryError("optional dependencies must be a table")
    for group, group_requirements in sorted(
        optional.items(), key=lambda item: str(item[0])
    ):
        if normalize_name(str(group)) == "dev":
            continue
        if not isinstance(group_requirements, list):
            raise InventoryError("optional dependency group must be a list")
        if not all(isinstance(requirement, str) for requirement in group_requirements):
            raise InventoryError("optional dependency declaration must be text")
        requirements.extend(group_requirements)

    return tuple(sorted({dependency_name(requirement) for requirement in requirements}))


def audit_inventory(
    entries: Iterable[InventoryEntry],
    required_dependencies: Iterable[str] = (),
) -> tuple[InventoryRecord, ...]:
    """Classify entries and add unknown records for missing dependencies."""

    normalized_entries = _deduplicate_entries(entries)
    by_name = {entry.name: entry for entry in normalized_entries}
    required_names = {normalize_name(name) for name in required_dependencies}

    for name in sorted(required_names - by_name.keys()):
        by_name[name] = InventoryEntry(
            name=name, license_expression="", scope="missing"
        )

    records: list[InventoryRecord] = []
    for entry in by_name.values():
        classification = classify_license(entry.license_expression)
        records.append(
            InventoryRecord(
                entry=entry,
                classification=classification,
                reason=_classification_reason(classification),
            )
        )
    return tuple(sorted(records, key=lambda record: record.name))


def audit_project(
    inventory_path: Path = DEFAULT_INVENTORY,
    pyproject_path: Path = DEFAULT_PYPROJECT,
) -> tuple[InventoryRecord, ...]:
    """Audit the local inventory against all non-development project extras."""

    return audit_inventory(
        parse_inventory(inventory_path),
        parse_project_dependencies(pyproject_path),
    )


def safe_report(records: Iterable[InventoryRecord]) -> dict[str, Any]:
    """Build deterministic report data that excludes raw license values."""

    ordered = tuple(sorted(records, key=lambda record: record.name))
    counts = Counter(record.classification for record in ordered)
    return {
        "entries": [record.to_safe_dict() for record in ordered],
        "schema_version": 1,
        "summary": {
            LicenseClass.PERMISSIVE: counts.get(LicenseClass.PERMISSIVE, 0),
            LicenseClass.RESTRICTED: counts.get(LicenseClass.RESTRICTED, 0),
            LicenseClass.UNKNOWN: counts.get(LicenseClass.UNKNOWN, 0),
        },
    }


def write_report(records: Iterable[InventoryRecord], path: Path) -> None:
    """Write a stable, privacy-safe JSON report to a local path."""

    payload = json.dumps(safe_report(records), indent=2, sort_keys=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    except (OSError, UnicodeError):
        raise InventoryError("could not write the inventory report") from None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the local license gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="checked-in Markdown, JSON, TOML, or CSV inventory",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="local pyproject whose non-dev dependencies must be inventoried",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional privacy-safe JSON report path",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="classify the inventory without checking pyproject coverage",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic, offline license inventory gate."""

    args = build_parser().parse_args(argv)
    try:
        if args.inventory_only:
            records = audit_inventory(parse_inventory(args.inventory))
        else:
            records = audit_project(args.inventory, args.pyproject)
        if args.report:
            write_report(records, args.report)
    except InventoryError as exc:
        # InventoryError messages are intentionally static and never include
        # source values, paths, dependency versions, or license text.
        print(f"license inventory gate failed: {exc}", file=sys.stderr)
        return 1

    failures = tuple(
        record for record in records if record.classification != LicenseClass.PERMISSIVE
    )
    if failures:
        print("license inventory gate failed:", file=sys.stderr)
        for record in failures:
            print(
                f"- {safe_identifier(record.name)}: "
                f"{record.classification} ({record.reason})",
                file=sys.stderr,
            )
        return 1

    print(f"license inventory gate passed: {len(records)} permissive entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
