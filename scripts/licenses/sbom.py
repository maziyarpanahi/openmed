#!/usr/bin/env python3
"""Build a deterministic, offline CycloneDX SBOM from local project metadata.

The generator intentionally reads only ``pyproject.toml`` and ``uv.lock``.
It does not resolve packages, inspect an environment, or contact a package
index.  The lock file supplies versions, dependency edges, and artifact
hashes; the project metadata supplies the direct runtime dependency roots.

The output omits lock-file URLs and source paths.  This keeps credentials and
developer build locations out of the evidence while retaining PURLs, hashes,
the source revision, and hashes of the input manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_LOCKFILE = ROOT / "uv.lock"
DEFAULT_OUTPUT = ROOT / "sbom.cdx.json"
SPEC_VERSION = "1.6"
UNKNOWN_LICENSE = "NOASSERTION"

_DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]*$")
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_HASH_RE = re.compile(r"^sha(?P<bits>256|384|512):(?P<content>[0-9a-fA-F]+)$")
_SPDX_EXPRESSION_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9.+-]*(?:\s+(?:AND|OR|WITH)\s+"
    r"[A-Za-z0-9][A-Za-z0-9.+-]*)*$"
)

# These are license facts already reviewed in this repository's dependency
# policy.  A package record's explicit license value always takes precedence;
# packages without either source remain explicitly marked NOASSERTION.
KNOWN_LICENSES = {
    "faker": "MIT",
    "jieba": "MIT",
    "pysbd": "MIT",
    "pyyaml": "MIT",
}


class SbomError(ValueError):
    """Raised when local SBOM inputs cannot be represented safely."""


@dataclass(frozen=True)
class DependencyRef:
    """A dependency edge from a lock-file package record."""

    name: str
    version: str | None = None
    source_kind: str | None = None


@dataclass(frozen=True)
class PackageRecord:
    """The safe, non-path portion of a lock-file package record."""

    name: str
    normalized_name: str
    version: str | None
    source_kind: str
    data: Mapping[str, Any]

    @property
    def key(self) -> tuple[str, str, str]:
        """Return a stable identity for this lock-file record."""
        return (self.normalized_name, self.version or "", self.source_kind)


def normalize_name(name: str) -> str:
    """Normalize a Python package name using the PEP 503 spelling."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _safe_package_name(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not _PACKAGE_NAME_RE.fullmatch(value):
        raise SbomError("dependency manifest contains an invalid package name")
    return value, normalize_name(value)


def _safe_version(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise SbomError("dependency manifest contains an invalid package version")
    return value


def _source_kind(value: object) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    for key in ("registry", "git", "url", "directory", "editable", "virtual"):
        if key in value:
            return key
    return "unknown"


def _read_toml(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError:
        raise SbomError(f"unable to read {label}") from None
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise SbomError(f"unable to parse {label}") from None
    if not isinstance(data, dict):
        raise SbomError(f"{label} must contain a TOML table")
    return data, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_dependency_name(requirement: object) -> str:
    if not isinstance(requirement, str):
        raise SbomError("project dependency declaration must be a string")
    match = _DEPENDENCY_NAME_RE.match(requirement)
    if not match:
        raise SbomError("project dependency declaration has no package name")
    _, normalized = _safe_package_name(match.group(1))
    return normalized


def _project_table(pyproject: Mapping[str, Any]) -> Mapping[str, Any]:
    project = pyproject.get("project")
    if not isinstance(project, Mapping):
        raise SbomError("pyproject metadata has no project table")
    return project


def _project_dependencies(project: Mapping[str, Any]) -> list[str]:
    values = project.get("dependencies", [])
    if not isinstance(values, list):
        raise SbomError("project dependencies must be a TOML array")
    dependencies = {_parse_dependency_name(value) for value in values}
    return sorted(dependencies)


def _project_name(project: Mapping[str, Any]) -> tuple[str, str]:
    value = project.get("name", "openmed")
    return _safe_package_name(value)


def _project_version(project: Mapping[str, Any], pyproject_path: Path) -> str:
    value = project.get("version")
    if isinstance(value, str) and _VERSION_RE.fullmatch(value):
        return value

    # OpenMed uses a dynamic version.  Read its local source file without
    # importing the package or consulting a build backend.
    about_path = pyproject_path.parent / "openmed" / "__about__.py"
    try:
        about = about_path.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', about)
    version = match.group(1) if match else "unknown"
    return version if _VERSION_RE.fullmatch(version) else "unknown"


def _license_value(value: object) -> list[dict[str, Any]]:
    """Convert a small SPDX-like value into a CycloneDX license choice."""
    if isinstance(value, Mapping):
        for key in ("expression", "id", "name", "text"):
            if key in value:
                return _license_value(value[key])
        return _unknown_license()
    if not isinstance(value, str):
        return _unknown_license()

    candidate = " ".join(value.split())
    if not candidate or len(candidate) > 128:
        return _unknown_license()
    if not _SPDX_EXPRESSION_RE.fullmatch(candidate):
        return _unknown_license()
    if any(operator in candidate.split() for operator in ("AND", "OR", "WITH")):
        return [{"expression": candidate}]
    return [{"license": {"id": candidate}}]


def _unknown_license() -> list[dict[str, Any]]:
    return [{"license": {"name": UNKNOWN_LICENSE}}]


def _project_licenses(project: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _license_value(project.get("license"))


def _record_licenses(record: PackageRecord) -> list[dict[str, Any]]:
    for key in ("license", "license-expression", "license_expression"):
        if key in record.data:
            return _license_value(record.data[key])

    metadata = record.data.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("license", "license-expression", "license_expression"):
            if key in metadata:
                return _license_value(metadata[key])

    known = KNOWN_LICENSES.get(record.normalized_name)
    return _license_value(known) if known else _unknown_license()


def _dependency_ref(value: object) -> DependencyRef:
    if isinstance(value, str):
        return DependencyRef(_parse_dependency_name(value))
    if not isinstance(value, Mapping):
        raise SbomError("lockfile contains an invalid dependency edge")
    raw_name = value.get("name")
    if not isinstance(raw_name, str):
        raise SbomError("lockfile dependency edge has no package name")
    _, name = _safe_package_name(raw_name)
    version = _safe_version(value.get("version"))
    source = value.get("source")
    return DependencyRef(name, version, _source_kind(source) if source else None)


def _record_dependencies(record: PackageRecord) -> tuple[DependencyRef, ...]:
    values = record.data.get("dependencies", [])
    if not isinstance(values, list):
        raise SbomError("lockfile package dependencies must be an array")
    dependencies = {_dependency_ref(value) for value in values}
    return tuple(
        sorted(
            dependencies,
            key=lambda dependency: (
                dependency.name,
                dependency.version or "",
                dependency.source_kind or "",
            ),
        )
    )


def _package_index(lockfile: Mapping[str, Any]) -> dict[str, tuple[PackageRecord, ...]]:
    values = lockfile.get("package")
    if not isinstance(values, list):
        raise SbomError("uv.lock has no package records")

    records: dict[str, list[PackageRecord]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise SbomError("uv.lock contains an invalid package record")
        name, normalized_name = _safe_package_name(value.get("name"))
        record = PackageRecord(
            name=name,
            normalized_name=normalized_name,
            version=_safe_version(value.get("version")),
            source_kind=_source_kind(value.get("source")),
            data=value,
        )
        records.setdefault(normalized_name, []).append(record)

    return {
        name: tuple(
            sorted(
                entries,
                key=lambda record: (
                    record.version or "",
                    record.source_kind,
                    record.name,
                ),
            )
        )
        for name, entries in records.items()
    }


def _resolve(
    dependency: DependencyRef,
    package_index: Mapping[str, Sequence[PackageRecord]],
) -> tuple[PackageRecord, ...]:
    candidates = list(package_index.get(dependency.name, ()))
    if dependency.version is not None:
        candidates = [
            record for record in candidates if record.version == dependency.version
        ]
    if dependency.source_kind is not None:
        candidates = [
            record
            for record in candidates
            if record.source_kind == dependency.source_kind
        ]
    if not candidates:
        raise SbomError("uv.lock does not resolve a declared dependency")
    return tuple(candidates)


def _collect_runtime_records(
    dependencies: Iterable[str],
    package_index: Mapping[str, Sequence[PackageRecord]],
) -> tuple[PackageRecord, ...]:
    roots = [DependencyRef(name) for name in dependencies]
    queue: list[PackageRecord] = []
    for dependency in roots:
        queue.extend(_resolve(dependency, package_index))

    collected: dict[tuple[str, str, str], PackageRecord] = {}
    while queue:
        record = queue.pop(0)
        if record.key in collected:
            continue
        collected[record.key] = record
        queue.extend(
            child
            for dependency in _record_dependencies(record)
            for child in _resolve(dependency, package_index)
        )

    return tuple(
        sorted(
            collected.values(),
            key=lambda record: (
                record.normalized_name,
                record.version or "",
                record.source_kind,
            ),
        )
    )


def _purl(record: PackageRecord) -> str:
    name = quote(record.normalized_name, safe=".-_~")
    version = quote(record.version, safe=".-_~") if record.version else ""
    package_type = "pypi" if record.source_kind == "registry" else "generic"
    base = f"pkg:{package_type}/{name}"
    if version:
        base += f"@{version}"
    if package_type == "generic" and record.source_kind != "unknown":
        base += f"?source={quote(record.source_kind, safe='.-_~')}"
    return base


def _root_purl(name: str, version: str) -> str:
    return f"pkg:pypi/{quote(normalize_name(name), safe='.-_~')}@{quote(version, safe='.-_~')}"


def _hash_value(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = _HASH_RE.fullmatch(value.strip())
    if not match:
        return None
    bits = match.group("bits")
    content = match.group("content").lower()
    if len(content) != int(bits) // 4:
        return None
    algorithm = {"256": "SHA-256", "384": "SHA-384", "512": "SHA-512"}[bits]
    return algorithm, content


def _record_hashes(record: PackageRecord) -> list[dict[str, str]]:
    values: list[object] = []
    for key in ("hash", "hashes"):
        value = record.data.get(key)
        values.extend(value if isinstance(value, list) else [value])
    for key in ("sdist", "wheels"):
        artifacts = record.data.get(key)
        if isinstance(artifacts, Mapping):
            values.append(artifacts.get("hash"))
        elif isinstance(artifacts, list):
            values.extend(
                artifact.get("hash")
                for artifact in artifacts
                if isinstance(artifact, Mapping)
            )

    hashes = {_hash_value(value) for value in values}
    hashes.discard(None)
    return [
        {"alg": algorithm, "content": content} for algorithm, content in sorted(hashes)
    ]


def _component(
    record: PackageRecord, ref_by_key: Mapping[tuple[str, str, str], str]
) -> dict[str, Any]:
    component: dict[str, Any] = {
        "bom-ref": ref_by_key[record.key],
        "licenses": _record_licenses(record),
        "name": record.name,
        "purl": ref_by_key[record.key],
        "scope": "required",
        "type": "library",
    }
    if record.version is not None:
        component["version"] = record.version
    hashes = _record_hashes(record)
    if hashes:
        component["hashes"] = hashes
    return component


def _dependency_entries(
    records: Sequence[PackageRecord],
    package_index: Mapping[str, Sequence[PackageRecord]],
    ref_by_key: Mapping[tuple[str, str, str], str],
    root_ref: str,
    roots: Iterable[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    root_refs = sorted(
        {
            ref_by_key[record.key]
            for dependency in roots
            for record in _resolve(DependencyRef(dependency), package_index)
            if record.key in ref_by_key
        }
    )
    entries.append({"dependsOn": root_refs, "ref": root_ref})

    for record in records:
        child_refs = sorted(
            {
                ref_by_key[child.key]
                for dependency in _record_dependencies(record)
                for child in _resolve(dependency, package_index)
                if child.key in ref_by_key
            }
        )
        entries.append({"dependsOn": child_refs, "ref": ref_by_key[record.key]})
    return entries


def _revision_from_git(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        raise SbomError("unable to read the local source revision") from None
    if result.returncode != 0:
        raise SbomError("unable to read the local source revision")
    revision = result.stdout.strip()
    if not _REVISION_RE.fullmatch(revision):
        raise SbomError("local source revision is not a commit hash")
    return revision.lower()


def _validate_revision(value: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value.strip()):
        raise SbomError("source revision must be a hexadecimal commit hash")
    return value.strip().lower()


def build_sbom(
    pyproject_path: Path = DEFAULT_PYPROJECT,
    lockfile_path: Path = DEFAULT_LOCKFILE,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic CycloneDX document from local manifests.

    ``source_revision`` can be supplied by callers that already have a
    revision.  When omitted, the current local Git ``HEAD`` is read; no remote
    Git or package-index operation is performed.
    """
    pyproject_path = Path(pyproject_path)
    lockfile_path = Path(lockfile_path)
    pyproject, pyproject_raw = _read_toml(pyproject_path, "pyproject metadata")
    lockfile, lockfile_raw = _read_toml(lockfile_path, "dependency lockfile")
    project = _project_table(pyproject)
    project_name, _ = _project_name(project)
    project_version = _project_version(project, pyproject_path)
    revision = (
        _revision_from_git(pyproject_path.parent)
        if source_revision is None
        else _validate_revision(source_revision)
    )

    package_index = _package_index(lockfile)
    roots = _project_dependencies(project)
    records = _collect_runtime_records(roots, package_index)
    ref_by_key = {record.key: _purl(record) for record in records}
    root_ref = _root_purl(project_name, project_version)

    manifest_hash = _sha256(pyproject_raw + b"\x00openmed-sbom\x00" + lockfile_raw)
    properties = [
        {"name": "openmed:lockfile-sha256", "value": _sha256(lockfile_raw)},
        {"name": "openmed:manifest-sha256", "value": manifest_hash},
        {"name": "openmed:pyproject-sha256", "value": _sha256(pyproject_raw)},
        {"name": "openmed:source-revision", "value": revision},
    ]

    return {
        "bomFormat": "CycloneDX",
        "components": [_component(record, ref_by_key) for record in records],
        "dependencies": _dependency_entries(
            records, package_index, ref_by_key, root_ref, roots
        ),
        "metadata": {
            "component": {
                "bom-ref": root_ref,
                "licenses": _project_licenses(project),
                "name": project_name,
                "purl": root_ref,
                "type": "library",
                "version": project_version,
            },
            "properties": properties,
        },
        "specVersion": SPEC_VERSION,
        "version": 1,
    }


def render_sbom(document: Mapping[str, Any]) -> str:
    """Render a document with stable key and component ordering."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def write_sbom(output_path: Path, document: Mapping[str, Any]) -> None:
    """Write a rendered SBOM without exposing the output path in errors."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_sbom(document), encoding="utf-8")
    except OSError:
        raise SbomError("unable to write the SBOM output") from None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the offline generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="local project metadata file",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=DEFAULT_LOCKFILE,
        help="local uv lockfile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination for the CycloneDX JSON document",
    )
    parser.add_argument(
        "--source-revision",
        help="hexadecimal local source commit; defaults to Git HEAD",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write the local SBOM, returning a process status."""
    args = parse_args(argv)
    try:
        document = build_sbom(
            pyproject_path=args.pyproject,
            lockfile_path=args.lockfile,
            source_revision=args.source_revision,
        )
        write_sbom(args.output, document)
    except SbomError as exc:
        print(f"SBOM generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"SBOM generated with {len(document['components'])} components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
