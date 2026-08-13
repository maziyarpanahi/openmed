"""Build deterministic dependency risk reports from local metadata.

The report intentionally accepts only caller-supplied advisory data and a
local ``uv.lock``-style TOML document.  It does not import a package manager,
query an advisory service, or preserve advisory descriptions, URLs, paths, or
other source fields.  This keeps the serialized result suitable for a
privacy-sensitive offline build while still showing every locked package,
its locked version, and the highest known risk category.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

__all__ = [
    "AdvisoryFinding",
    "DependencyRisk",
    "LockedDependency",
    "RISK_CATEGORIES",
    "build_dependency_risk_report",
    "dependency_risk_report",
    "dependency_risk_report_json",
    "generate_dependency_risk_report",
    "parse_advisory_snapshot",
    "parse_lockfile",
    "write_dependency_risk_report",
]

SCHEMA_VERSION = 1
RISK_CATEGORIES = ("critical", "high", "medium", "low", "unknown", "none")

_RISK_RANK = {category: index for index, category in enumerate(RISK_CATEGORIES)}
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_~:-]{0,127}$")
_ADVISORY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

_ADVISORY_COLLECTION_KEYS = ("advisories", "vulnerabilities", "vulns")
_PACKAGE_NAME_KEYS = ("name", "package", "package_name", "project")
_VERSION_KEYS = ("version", "installed_version")
_ADVISORY_ID_KEYS = ("id", "advisory_id", "vulnerability_id", "cve")
_SEVERITY_KEYS = ("severity", "risk_category", "risk", "level")

JsonSource = str | bytes | Path | Mapping[str, Any] | Sequence[Any]
LockfileSource = str | bytes | Path | Mapping[str, Any]


@dataclass(frozen=True)
class LockedDependency:
    """A package/version pair read from the local lockfile."""

    name: str
    version: str


@dataclass(frozen=True)
class AdvisoryFinding:
    """A safe, normalized advisory record used for correlation.

    Descriptions, URLs, source paths, and the original advisory payload are
    deliberately not represented by this type.
    """

    package_name: str
    snapshot_version: str | None
    advisory_id: str | None
    risk_category: str


@dataclass(frozen=True)
class DependencyRisk:
    """The public package-level fields emitted by the report."""

    name: str
    version: str
    risk_category: str

    def to_dict(self) -> dict[str, str]:
        """Return the privacy-safe JSON representation."""
        return {
            "name": self.name,
            "risk_category": self.risk_category,
            "version": self.version,
        }


def parse_lockfile(lockfile: LockfileSource) -> tuple[LockedDependency, ...]:
    """Read package names and versions from a local TOML lockfile.

    ``uv.lock`` stores packages in repeated ``[[package]]`` tables.  A
    mapping with a ``packages`` list is also accepted for callers that have
    already parsed the same document.  Only the name and exact version are
    retained.

    Raises:
        ValueError: If the source cannot be read, parsed, or validated.
    """
    payload = _load_toml_source(lockfile)
    if not isinstance(payload, Mapping):
        raise ValueError("lockfile must contain a TOML mapping")

    entries = payload.get("package")
    if entries is None:
        entries = payload.get("packages")
    if not isinstance(entries, list):
        raise ValueError("lockfile must contain a package table list")

    dependencies: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("lockfile contains an invalid package entry")
        name = _package_name(entry.get("name"))
        if "version" not in entry and _is_editable_local_entry(entry):
            continue
        version = _version(entry.get("version"))
        dependencies.add((name, version))

    return tuple(
        LockedDependency(name=name, version=version)
        for name, version in sorted(dependencies, key=lambda item: (item[0], item[1]))
    )


def parse_advisory_snapshot(snapshot: JsonSource) -> tuple[AdvisoryFinding, ...]:
    """Normalize a caller-supplied offline advisory snapshot.

    The parser accepts pip-audit's ``dependencies`` shape, OSV's ``results``
    shape, and a compact ``packages``/``advisories`` shape.  Unknown advisory
    fields are ignored.  An advisory without a recognized severity is
    conservatively classified as ``unknown``.

    Raises:
        ValueError: If the source has an unsupported or malformed structure.
    """
    payload = _load_json_source(snapshot, "advisory snapshot")
    findings: list[AdvisoryFinding] = []

    if isinstance(payload, list):
        _parse_package_entries(payload, findings)
    elif isinstance(payload, Mapping):
        if "dependencies" in payload:
            _parse_dependency_entries(payload["dependencies"], findings)
        elif "results" in payload:
            _parse_osv_results(payload["results"], findings)
        elif "packages" in payload:
            _parse_package_entries(payload["packages"], findings)
        elif "advisories" in payload:
            _parse_advisory_entries(payload["advisories"], findings)
        else:
            _parse_package_mapping(payload, findings)
    else:
        raise ValueError("advisory snapshot must contain a JSON object or list")

    return tuple(
        sorted(
            set(findings),
            key=lambda finding: (
                finding.package_name,
                finding.snapshot_version or "",
                finding.advisory_id or "",
                _RISK_RANK[finding.risk_category],
            ),
        )
    )


def dependency_risk_report(
    advisory_snapshot: JsonSource,
    lockfile: LockfileSource,
) -> dict[str, Any]:
    """Return a deterministic, offline dependency risk report.

    Every package in the lockfile appears exactly once per unique locked
    version.  Advisories with no version apply to every matching package;
    versioned advisories apply only to the exact locked version.  A stale
    versioned snapshot is reported as ``unknown`` rather than silently
    treating the package as safe.

    The returned mapping contains only package names, versions, aggregate
    counts, and normalized risk categories.  It never includes raw advisory
    descriptions, URLs, paths, or source payloads.
    """
    locked_dependencies = parse_lockfile(lockfile)
    findings = parse_advisory_snapshot(advisory_snapshot)

    by_package: dict[str, list[AdvisoryFinding]] = {}
    for finding in findings:
        by_package.setdefault(finding.package_name, []).append(finding)

    package_rows: list[DependencyRisk] = []
    matched_finding_set: set[AdvisoryFinding] = set()
    for dependency in locked_dependencies:
        candidates = by_package.get(dependency.name, [])
        matching = [
            finding
            for finding in candidates
            if finding.snapshot_version is None
            or _same_version(finding.snapshot_version, dependency.version)
        ]
        matched_finding_set.update(matching)

        if matching:
            category = _highest_risk(finding.risk_category for finding in matching)
        elif candidates:
            category = "unknown"
        else:
            category = "none"

        package_rows.append(
            DependencyRisk(
                name=dependency.name,
                version=dependency.version,
                risk_category=category,
            )
        )

    matched_findings = len(matched_finding_set)
    category_counts = {category: 0 for category in RISK_CATEGORIES}
    for row in package_rows:
        category_counts[row.risk_category] += 1

    return {
        "artifact": "offline_dependency_risk",
        "offline": True,
        "packages": [row.to_dict() for row in package_rows],
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "affected_packages": len(package_rows) - category_counts["none"],
            "advisory_matches": matched_findings,
            "risk_categories": category_counts,
            "total_packages": len(package_rows),
            "unmatched_advisories": len(findings) - matched_findings,
        },
    }


def build_dependency_risk_report(
    lockfile: LockfileSource,
    advisory_snapshot: JsonSource,
) -> dict[str, Any]:
    """Build a report with lockfile-first argument ordering."""
    return dependency_risk_report(advisory_snapshot, lockfile)


def generate_dependency_risk_report(
    lockfile: LockfileSource,
    advisory_snapshot: JsonSource,
) -> dict[str, Any]:
    """Alias for :func:`build_dependency_risk_report`."""
    return build_dependency_risk_report(lockfile, advisory_snapshot)


def dependency_risk_report_json(
    advisory_snapshot: JsonSource,
    lockfile: LockfileSource,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize an offline dependency risk report as deterministic JSON."""
    return json.dumps(
        dependency_risk_report(advisory_snapshot, lockfile),
        allow_nan=False,
        ensure_ascii=True,
        indent=indent,
        sort_keys=True,
    )


def write_dependency_risk_report(
    advisory_snapshot: JsonSource,
    lockfile: LockfileSource,
    output_path: str | Path,
    *,
    indent: int | None = 2,
) -> Path:
    """Write a deterministic JSON report without contacting external services."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            dependency_risk_report_json(
                advisory_snapshot,
                lockfile,
                indent=indent,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        raise ValueError("dependency risk report could not be written") from None
    return path


def _load_json_source(source: JsonSource, label: str) -> Any:
    if isinstance(source, Mapping) or (
        isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray))
    ):
        return source

    text = _source_text(source, label, structured_prefixes=("{", "["))
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, UnicodeError):
        raise ValueError(f"{label} is not valid JSON") from None


def _load_toml_source(source: LockfileSource) -> Any:
    if isinstance(source, Mapping):
        return source

    text = _source_text(source, "lockfile", structured_prefixes=("[",))
    try:
        return tomllib.loads(text)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("lockfile is not valid TOML") from None


def _source_text(
    source: str | bytes | Path,
    label: str,
    *,
    structured_prefixes: tuple[str, ...],
) -> str:
    if isinstance(source, Path):
        return _read_text(source, label)
    if isinstance(source, bytes):
        try:
            return source.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{label} is not valid UTF-8") from None
    if not isinstance(source, str):
        raise ValueError(f"{label} must be a path, text, or parsed mapping")

    stripped = source.lstrip()
    if stripped.startswith(structured_prefixes) or "\n" in source:
        return source
    return _read_text(Path(source), label)


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ValueError(f"{label} could not be read") from None


def _parse_dependency_entries(value: Any, findings: list[AdvisoryFinding]) -> None:
    if not isinstance(value, list):
        raise ValueError("advisory snapshot dependencies must be a list")
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError("advisory snapshot contains an invalid dependency entry")
        name = _entry_package_name(entry)
        version = _entry_version(entry)
        advisories = _entry_advisories(entry)
        _append_findings(name, version, advisories, findings, entry)


def _parse_osv_results(value: Any, findings: list[AdvisoryFinding]) -> None:
    if not isinstance(value, list):
        raise ValueError("advisory snapshot results must be a list")
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError("advisory snapshot contains an invalid result entry")
        name = _entry_package_name(entry)
        version = _entry_version(entry)
        advisories = entry.get("vulnerabilities")
        if advisories is None:
            advisories = entry.get("advisories")
        _append_findings(name, version, advisories, findings, entry)


def _parse_package_entries(value: Any, findings: list[AdvisoryFinding]) -> None:
    if not isinstance(value, list):
        raise ValueError("advisory snapshot packages must be a list")
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError("advisory snapshot contains an invalid package entry")
        name = _entry_package_name(entry)
        version = _entry_version(entry)
        advisories = _entry_advisories(entry)
        _append_findings(name, version, advisories, findings, entry)


def _parse_advisory_entries(value: Any, findings: list[AdvisoryFinding]) -> None:
    if not isinstance(value, list):
        raise ValueError("advisory snapshot advisories must be a list")
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError("advisory snapshot contains an invalid advisory entry")
        name = _entry_package_name(entry)
        version = _entry_version(entry)
        _append_findings(name, version, (entry,), findings, entry)


def _parse_package_mapping(
    value: Mapping[str, Any], findings: list[AdvisoryFinding]
) -> None:
    if not value:
        return
    for raw_name, raw_advisories in value.items():
        if not isinstance(raw_name, str):
            raise ValueError("advisory snapshot contains an invalid package name")
        if raw_name in {"schema_version", "format", "metadata"}:
            continue
        name = _package_name(raw_name)
        version: str | None = None
        advisories = raw_advisories
        if isinstance(raw_advisories, Mapping):
            version = _optional_version(
                raw_advisories.get("version") or raw_advisories.get("installed_version")
            )
            collection = _entry_advisories(raw_advisories)
            if collection is not None:
                advisories = collection
            elif _has_advisory_identity(raw_advisories):
                advisories = (raw_advisories,)
            else:
                advisories = ()
        _append_findings(name, version, advisories, findings, {})


def _entry_package_name(entry: Mapping[str, Any]) -> str:
    for key in _PACKAGE_NAME_KEYS:
        if key not in entry:
            continue
        value = entry[key]
        if isinstance(value, Mapping):
            for nested_key in _PACKAGE_NAME_KEYS:
                if nested_key in value:
                    return _package_name(value[nested_key])
        else:
            return _package_name(value)
    raise ValueError("advisory snapshot entry is missing a package name")


def _is_editable_local_entry(entry: Mapping[str, Any]) -> bool:
    source = entry.get("source")
    return isinstance(source, Mapping) and "editable" in source


def _entry_version(entry: Mapping[str, Any]) -> str | None:
    for key in _VERSION_KEYS:
        if key in entry:
            return _optional_version(entry[key])
    package = entry.get("package")
    if isinstance(package, Mapping):
        for key in _VERSION_KEYS:
            if key in package:
                return _optional_version(package[key])
    return None


def _entry_advisories(entry: Mapping[str, Any]) -> Any:
    for key in _ADVISORY_COLLECTION_KEYS:
        if key in entry:
            return entry[key]
    if _has_advisory_identity(entry):
        return (entry,)
    return ()


def _has_advisory_identity(entry: Mapping[str, Any]) -> bool:
    return any(key in entry for key in _ADVISORY_ID_KEYS) or any(
        key in entry for key in _SEVERITY_KEYS
    )


def _append_findings(
    package_name: str,
    version: str | None,
    advisories: Any,
    findings: list[AdvisoryFinding],
    fallback: Mapping[str, Any],
) -> None:
    if advisories is None:
        advisories = ()
    if isinstance(advisories, Mapping) or isinstance(advisories, str):
        advisories = (advisories,)
    elif not isinstance(advisories, Sequence) or isinstance(
        advisories, (bytes, bytearray)
    ):
        raise ValueError("advisory snapshot contains an invalid advisory collection")

    for advisory in advisories:
        if isinstance(advisory, Mapping):
            record = advisory
        elif isinstance(advisory, str):
            record = {"id": advisory}
        else:
            raise ValueError("advisory snapshot contains an invalid advisory")
        findings.append(
            _make_finding(
                package_name,
                version,
                record,
                fallback_severity=fallback.get("severity"),
            )
        )


def _make_finding(
    package_name: str,
    version: str | None,
    advisory: Mapping[str, Any],
    *,
    fallback_severity: Any = None,
) -> AdvisoryFinding:
    nested = advisory.get("vulnerability")
    if isinstance(nested, Mapping):
        advisory = nested

    advisory_id = _optional_advisory_id(advisory)
    severity = _extract_severity(advisory)
    if severity is None:
        severity = _normalize_severity(fallback_severity) or "unknown"
    return AdvisoryFinding(
        package_name=package_name,
        snapshot_version=version,
        advisory_id=advisory_id,
        risk_category=severity,
    )


def _optional_advisory_id(advisory: Mapping[str, Any]) -> str | None:
    for key in _ADVISORY_ID_KEYS:
        if key in advisory:
            value = advisory[key]
            if isinstance(value, str):
                text = value.strip()
                if text and _ADVISORY_ID_RE.fullmatch(text):
                    return text
            return None
    return None


def _extract_severity(advisory: Mapping[str, Any]) -> str | None:
    values: list[Any] = []
    for key in _SEVERITY_KEYS:
        if key in advisory:
            values.append(advisory[key])
    database_specific = advisory.get("database_specific")
    if isinstance(database_specific, Mapping):
        for key in _SEVERITY_KEYS:
            if key in database_specific:
                values.append(database_specific[key])
    normalized = [
        category
        for value in values
        if (category := _normalize_severity(value)) is not None
    ]
    if not normalized:
        return None
    return _highest_risk(normalized)


def _normalize_severity(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return _cvss_category(float(value))
    if isinstance(value, Mapping):
        nested_values = [value.get(key) for key in ("score", "base_score", "severity")]
        normalized = [
            category
            for nested in nested_values
            if (category := _normalize_severity(nested)) is not None
        ]
        return _highest_risk(normalized) if normalized else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = [
            category
            for item in value
            if (category := _normalize_severity(item)) is not None
        ]
        return _highest_risk(normalized) if normalized else None
    if not isinstance(value, str):
        return None

    normalized = value.strip().casefold()
    aliases = {
        "critical": "critical",
        "crit": "critical",
        "urgent": "critical",
        "high": "high",
        "important": "high",
        "medium": "medium",
        "moderate": "medium",
        "med": "medium",
        "low": "low",
        "minor": "low",
        "none": "none",
        "informational": "none",
        "info": "none",
        "unknown": "unknown",
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return _cvss_category(float(normalized))
    except ValueError:
        return None


def _cvss_category(score: float) -> str | None:
    if score < 0 or score > 10:
        return None
    if score >= 9:
        return "critical"
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _highest_risk(categories: Sequence[str] | Any) -> str:
    values = tuple(categories)
    if not values:
        return "none"
    return min(values, key=lambda category: _RISK_RANK[category])


def _package_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("package names must be non-empty safe strings")
    text = value.strip()
    if not text or not _PACKAGE_NAME_RE.fullmatch(text):
        raise ValueError("package names must be non-empty safe strings")
    return re.sub(r"[-_.]+", "-", text.casefold())


def _version(value: Any) -> str:
    version = _optional_version(value)
    if version is None:
        raise ValueError("package versions must be non-empty safe strings")
    return version


def _optional_version(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("package versions must be non-empty safe strings")
    text = value.strip()
    if not text or not _VERSION_RE.fullmatch(text):
        raise ValueError("package versions must be non-empty safe strings")
    return text


def _same_version(left: str, right: str) -> bool:
    return left.casefold() == right.casefold()
