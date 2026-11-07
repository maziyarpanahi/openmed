"""Build deterministic dependency risk reports from local metadata.

The report intentionally accepts only caller-supplied advisory data and a
local ``uv.lock``-style TOML document.  It does not import a package manager,
query an advisory service, or preserve advisory descriptions, URLs, paths, or
other source fields.  This keeps the serialized result suitable for a
privacy-sensitive offline build while still showing every locked package,
its locked version, and the highest known risk category.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

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
MAX_ADVISORY_BYTES: Final = 16 * 1024 * 1024
MAX_LOCKFILE_BYTES: Final = 32 * 1024 * 1024
MAX_OUTPUT_BYTES: Final = 64 * 1024 * 1024
MAX_PACKAGE_RECORDS: Final = 100_000
MAX_ADVISORY_RECORDS: Final = 100_000
MAX_ADVISORIES_PER_PACKAGE: Final = 10_000
MAX_SEVERITY_DEPTH: Final = 16
MAX_INDENT: Final = 8

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

    def __post_init__(self) -> None:
        if (
            _package_name(self.name) != self.name
            or _version(self.version) != self.version
        ):
            raise ValueError("locked dependency fields are invalid")


@dataclass(frozen=True)
class AdvisoryFinding:
    """A safe, normalized advisory record used for correlation.

    Descriptions, URLs, source paths, and the original advisory payload are
    deliberately not represented by this type.
    """

    package_name: str
    snapshot_version: str | None
    advisory_fingerprint: str | None = field(repr=False)
    risk_category: str

    def __post_init__(self) -> None:
        if _package_name(self.package_name) != self.package_name:
            raise ValueError("advisory finding package name is invalid")
        if _optional_version(self.snapshot_version) != self.snapshot_version:
            raise ValueError("advisory finding version is invalid")
        if self.advisory_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.advisory_fingerprint
        ):
            raise ValueError("advisory finding identity is invalid")
        if self.risk_category not in RISK_CATEGORIES:
            raise ValueError("advisory finding risk category is invalid")


@dataclass(frozen=True)
class DependencyRisk:
    """The public package-level fields emitted by the report."""

    name: str
    version: str
    risk_category: str

    def __post_init__(self) -> None:
        if (
            _package_name(self.name) != self.name
            or _version(self.version) != self.version
        ):
            raise ValueError("dependency risk fields are invalid")
        if self.risk_category not in RISK_CATEGORIES:
            raise ValueError("dependency risk category is invalid")

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
    if type(payload) is not dict:
        raise ValueError("lockfile must contain a TOML mapping")

    entries = payload.get("package")
    if entries is None:
        entries = payload.get("packages")
    if type(entries) is not list:
        raise ValueError("lockfile must contain a package table list")
    if len(entries) > MAX_PACKAGE_RECORDS:
        raise ValueError("lockfile contains too many package entries")

    dependencies: set[tuple[str, str]] = set()
    for entry in entries:
        if type(entry) is not dict:
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

    if type(payload) is list:
        _parse_package_entries(payload, findings)
    elif type(payload) is dict:
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
                finding.advisory_fingerprint or "",
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
        package_findings = by_package.setdefault(finding.package_name, [])
        if len(package_findings) >= MAX_ADVISORIES_PER_PACKAGE:
            raise ValueError("advisory collection exceeds the supported item limit")
        package_findings.append(finding)

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
    indent = _validated_indent(indent)
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
    temporary_path: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = (
            dependency_risk_report_json(
                advisory_snapshot,
                lockfile,
                indent=indent,
            )
            + "\n"
        )
        if len(rendered.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError("dependency risk report exceeds the supported size limit")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".openmed-dependency-risk-",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except ValueError:
        raise
    except (OSError, TypeError, UnicodeError):
        raise ValueError("dependency risk report could not be written") from None
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass
    return path


def _validated_indent(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_INDENT:
        raise ValueError(
            "JSON indentation must be an integer within the supported range"
        )
    return value


def _load_json_source(source: JsonSource, label: str) -> Any:
    if type(source) in {dict, list}:
        return source

    text = _source_text(
        source,
        label,
        structured_prefixes=("{", "["),
        max_bytes=MAX_ADVISORY_BYTES,
    )
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, RecursionError, TypeError, UnicodeError, ValueError):
        raise ValueError(f"{label} is not valid JSON") from None


def _load_toml_source(source: LockfileSource) -> Any:
    if type(source) is dict:
        return source

    text = _source_text(
        source,
        "lockfile",
        structured_prefixes=("[",),
        max_bytes=MAX_LOCKFILE_BYTES,
    )
    try:
        return tomllib.loads(text)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("lockfile is not valid TOML") from None


def _source_text(
    source: str | bytes | Path,
    label: str,
    *,
    structured_prefixes: tuple[str, ...],
    max_bytes: int,
) -> str:
    if isinstance(source, Path):
        return _read_text(source, label, max_bytes=max_bytes)
    if type(source) is bytes:
        if len(source) > max_bytes:
            raise ValueError(f"{label} exceeds the supported size limit")
        try:
            return source.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{label} is not valid UTF-8") from None
    if type(source) is not str:
        raise ValueError(f"{label} must be a path, text, or parsed mapping")
    try:
        encoded_size = len(source.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError(f"{label} is not valid UTF-8") from None
    if encoded_size > max_bytes:
        raise ValueError(f"{label} exceeds the supported size limit")

    stripped = source.lstrip()
    if stripped.startswith(structured_prefixes) or "\n" in source:
        return source
    return _read_text(Path(source), label, max_bytes=max_bytes)


def _read_text(path: Path, label: str, *, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        raise ValueError(f"{label} could not be read") from None
    if len(raw) > max_bytes:
        raise ValueError(f"{label} exceeds the supported size limit")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{label} is not valid UTF-8") from None


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _parse_dependency_entries(value: Any, findings: list[AdvisoryFinding]) -> None:
    if type(value) is not list:
        raise ValueError("advisory snapshot dependencies must be a list")
    if len(value) > MAX_PACKAGE_RECORDS:
        raise ValueError("advisory snapshot contains too many package entries")
    for entry in value:
        if type(entry) is not dict:
            raise ValueError("advisory snapshot contains an invalid dependency entry")
        name = _entry_package_name(entry)
        version = _entry_version(entry)
        advisories = _entry_advisories(entry)
        _append_findings(name, version, advisories, findings, entry)


def _parse_osv_results(value: Any, findings: list[AdvisoryFinding]) -> None:
    if type(value) is not list:
        raise ValueError("advisory snapshot results must be a list")
    if len(value) > MAX_PACKAGE_RECORDS:
        raise ValueError("advisory snapshot contains too many package entries")
    for entry in value:
        if type(entry) is not dict:
            raise ValueError("advisory snapshot contains an invalid result entry")
        name = _entry_package_name(entry)
        version = _entry_version(entry)
        advisories = entry.get("vulnerabilities")
        if advisories is None:
            advisories = entry.get("advisories")
        _append_findings(name, version, advisories, findings, entry)


def _parse_package_entries(value: Any, findings: list[AdvisoryFinding]) -> None:
    if type(value) is not list:
        raise ValueError("advisory snapshot packages must be a list")
    if len(value) > MAX_PACKAGE_RECORDS:
        raise ValueError("advisory snapshot contains too many package entries")
    for entry in value:
        if type(entry) is not dict:
            raise ValueError("advisory snapshot contains an invalid package entry")
        name = _entry_package_name(entry)
        version = _entry_version(entry)
        advisories = _entry_advisories(entry)
        _append_findings(name, version, advisories, findings, entry)


def _parse_advisory_entries(value: Any, findings: list[AdvisoryFinding]) -> None:
    if type(value) is not list:
        raise ValueError("advisory snapshot advisories must be a list")
    if len(value) > MAX_ADVISORY_RECORDS:
        raise ValueError("advisory snapshot contains too many advisory records")
    for entry in value:
        if type(entry) is not dict:
            raise ValueError("advisory snapshot contains an invalid advisory entry")
        name = _entry_package_name(entry)
        version = _entry_version(entry)
        _append_findings(name, version, (entry,), findings, entry)


def _parse_package_mapping(
    value: Mapping[str, Any], findings: list[AdvisoryFinding]
) -> None:
    if not value:
        return
    if len(value) > MAX_PACKAGE_RECORDS:
        raise ValueError("advisory snapshot contains too many package entries")
    for raw_name, raw_advisories in value.items():
        if type(raw_name) is not str:
            raise ValueError("advisory snapshot contains an invalid package name")
        if raw_name in {"schema_version", "format", "metadata"}:
            continue
        name = _package_name(raw_name)
        version: str | None = None
        advisories = raw_advisories
        if type(raw_advisories) is dict:
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
        if type(value) is dict:
            for nested_key in _PACKAGE_NAME_KEYS:
                if nested_key in value:
                    return _package_name(value[nested_key])
        else:
            return _package_name(value)
    raise ValueError("advisory snapshot entry is missing a package name")


def _is_editable_local_entry(entry: Mapping[str, Any]) -> bool:
    source = entry.get("source")
    return type(source) is dict and "editable" in source


def _entry_version(entry: Mapping[str, Any]) -> str | None:
    for key in _VERSION_KEYS:
        if key in entry:
            return _optional_version(entry[key])
    package = entry.get("package")
    if type(package) is dict:
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
    if type(advisories) in {dict, str}:
        advisories = (advisories,)
    elif type(advisories) not in {list, tuple}:
        raise ValueError("advisory snapshot contains an invalid advisory collection")
    if len(advisories) > MAX_ADVISORIES_PER_PACKAGE:
        raise ValueError("advisory collection exceeds the supported item limit")

    for advisory in advisories:
        if type(advisory) is dict:
            record = advisory
        elif type(advisory) is str:
            record = {"id": advisory}
        else:
            raise ValueError("advisory snapshot contains an invalid advisory")
        if len(findings) >= MAX_ADVISORY_RECORDS:
            raise ValueError("advisory snapshot contains too many advisory records")
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
    if type(nested) is dict:
        advisory = nested

    advisory_fingerprint = _optional_advisory_fingerprint(advisory)
    severity = _extract_severity(advisory)
    if severity is None:
        severity = _normalize_severity(fallback_severity) or "unknown"
    return AdvisoryFinding(
        package_name=package_name,
        snapshot_version=version,
        advisory_fingerprint=advisory_fingerprint,
        risk_category=severity,
    )


def _optional_advisory_fingerprint(advisory: Mapping[str, Any]) -> str | None:
    for key in _ADVISORY_ID_KEYS:
        if key in advisory:
            value = advisory[key]
            if type(value) is str:
                text = value.strip()
                if text and _ADVISORY_ID_RE.fullmatch(text):
                    return hashlib.sha256(text.encode("ascii")).hexdigest()
            return None
    return None


def _extract_severity(advisory: Mapping[str, Any]) -> str | None:
    values: list[Any] = []
    for key in _SEVERITY_KEYS:
        if key in advisory:
            values.append(advisory[key])
    database_specific = advisory.get("database_specific")
    if type(database_specific) is dict:
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


def _normalize_severity(value: Any, *, depth: int = 0) -> str | None:
    if depth > MAX_SEVERITY_DEPTH:
        return None
    if type(value) is bool or value is None:
        return None
    if type(value) in {int, float}:
        try:
            return _cvss_category(float(value))
        except OverflowError:
            return None
    if type(value) is dict:
        nested_values = [value.get(key) for key in ("score", "base_score", "severity")]
        normalized = [
            category
            for nested in nested_values
            if (category := _normalize_severity(nested, depth=depth + 1)) is not None
        ]
        return _highest_risk(normalized) if normalized else None
    if type(value) is list:
        if len(value) > MAX_ADVISORIES_PER_PACKAGE:
            return None
        normalized = [
            category
            for item in value
            if (category := _normalize_severity(item, depth=depth + 1)) is not None
        ]
        return _highest_risk(normalized) if normalized else None
    if type(value) is not str:
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
    if not math.isfinite(score) or score < 0 or score > 10:
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
    if type(value) is not str:
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
    if type(value) is not str:
        raise ValueError("package versions must be non-empty safe strings")
    text = value.strip()
    if not text or not _VERSION_RE.fullmatch(text):
        raise ValueError("package versions must be non-empty safe strings")
    return text


def _same_version(left: str, right: str) -> bool:
    return left.casefold() == right.casefold()
