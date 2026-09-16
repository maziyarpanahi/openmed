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
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
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
MAX_PYPROJECT_BYTES: Final = 2 * 1024 * 1024
MAX_LOCKFILE_BYTES: Final = 32 * 1024 * 1024
MAX_ABOUT_BYTES: Final = 64 * 1024
MAX_PACKAGE_RECORDS: Final = 100_000
MAX_RUNTIME_COMPONENTS: Final = 100_000
MAX_DEPENDENCIES_PER_RECORD: Final = 10_000
MAX_ARTIFACTS_PER_RECORD: Final = 10_000
MAX_PACKAGE_NAME_LENGTH: Final = 256
MAX_VERSION_LENGTH: Final = 256
MAX_LICENSE_LENGTH: Final = 512
MAX_SBOM_BYTES: Final = 128 * 1024 * 1024
GIT_TIMEOUT_SECONDS: Final = 10

_DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]*$")
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_HASH_RE = re.compile(r"^sha(?P<bits>256|384|512):(?P<content>[0-9a-fA-F]+)$")
_SPDX_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*|[()]")
_SPDX_OPERATORS: Final = frozenset({"AND", "OR", "WITH"})
_KNOWN_SPDX_LICENSE_IDS: Final = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC0-1.0",
        "HPND",
        "ISC",
        "MIT",
        "MPL-2.0",
        "Unlicense",
        "Zlib",
    }
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

    def __post_init__(self) -> None:
        _, normalized = _safe_package_name(self.name)
        if normalized != self.name:
            raise SbomError("dependency reference name must be normalized")
        _safe_version(self.version)
        if self.source_kind is not None and self.source_kind not in {
            "directory",
            "editable",
            "git",
            "registry",
            "unknown",
            "url",
            "virtual",
        }:
            raise SbomError("dependency reference source is unsupported")


@dataclass(frozen=True)
class PackageRecord:
    """The safe, non-path portion of a lock-file package record."""

    name: str
    normalized_name: str
    version: str | None
    source_kind: str
    data: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        name, normalized = _safe_package_name(self.name)
        if name != self.name or normalized != self.normalized_name:
            raise SbomError("package record name is inconsistent")
        _safe_version(self.version)
        if self.source_kind not in {
            "directory",
            "editable",
            "git",
            "registry",
            "unknown",
            "url",
            "virtual",
        }:
            raise SbomError("package record source is unsupported")
        if type(self.data) is not dict:
            raise SbomError("package record data must be a local manifest object")

    @property
    def key(self) -> tuple[str, str, str]:
        """Return a stable identity for this lock-file record."""
        return (self.normalized_name, self.version or "", self.source_kind)


def normalize_name(name: str) -> str:
    """Normalize a Python package name using the PEP 503 spelling."""

    if type(name) is not str:
        raise SbomError("dependency manifest contains an invalid package name")
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _safe_package_name(value: object) -> tuple[str, str]:
    if (
        type(value) is not str
        or len(value) > MAX_PACKAGE_NAME_LENGTH
        or not _PACKAGE_NAME_RE.fullmatch(value)
    ):
        raise SbomError("dependency manifest contains an invalid package name")
    return value, normalize_name(value)


def _safe_version(value: object) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) > MAX_VERSION_LENGTH
        or not _VERSION_RE.fullmatch(value)
    ):
        raise SbomError("dependency manifest contains an invalid package version")
    return value


def _source_kind(value: object) -> str:
    if value is None:
        return "unknown"
    if type(value) is not dict:
        raise SbomError("dependency manifest contains an invalid source record")
    kinds = [
        key
        for key in ("registry", "git", "url", "directory", "editable", "virtual")
        if key in value
    ]
    if len(kinds) > 1:
        raise SbomError("dependency source declares multiple source kinds")
    return kinds[0] if kinds else "unknown"


def _read_toml(
    path: Path,
    label: str,
    *,
    max_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        raise SbomError(f"unable to read {label}") from None
    if len(raw) > max_bytes:
        raise SbomError(f"{label} exceeds the supported size limit")
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise SbomError(f"unable to parse {label}") from None
    if type(data) is not dict:
        raise SbomError(f"{label} must contain a TOML table")
    return data, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_dependency_name(requirement: object) -> str:
    if type(requirement) is not str or len(requirement) > 4_096:
        raise SbomError("project dependency declaration must be a string")
    match = _DEPENDENCY_NAME_RE.match(requirement)
    if not match:
        raise SbomError("project dependency declaration has no package name")
    remainder = requirement[match.end() :].lstrip()
    if remainder and remainder[0] not in "[<>=!~@;":
        raise SbomError("project dependency declaration is malformed")
    _, normalized = _safe_package_name(match.group(1))
    return normalized


def _project_table(pyproject: Mapping[str, Any]) -> Mapping[str, Any]:
    project = pyproject.get("project")
    if type(project) is not dict:
        raise SbomError("pyproject metadata has no project table")
    return project


def _project_dependencies(project: Mapping[str, Any]) -> list[str]:
    values = project.get("dependencies", [])
    if type(values) is not list:
        raise SbomError("project dependencies must be a TOML array")
    if len(values) > MAX_DEPENDENCIES_PER_RECORD:
        raise SbomError("project dependencies exceed the supported item limit")
    dependencies = {_parse_dependency_name(value) for value in values}
    return sorted(dependencies)


def _project_name(project: Mapping[str, Any]) -> tuple[str, str]:
    return _safe_package_name(project.get("name"))


def _project_version(
    project: Mapping[str, Any], pyproject_path: Path
) -> tuple[str, bytes | None]:
    value = project.get("version")
    if value is not None:
        version = _safe_version(value)
        if version is None:
            raise SbomError("project version is missing")
        return version, None

    dynamic = project.get("dynamic", [])
    if type(dynamic) is not list or "version" not in dynamic:
        raise SbomError("project version is missing")

    # OpenMed uses a dynamic version.  Read its local source file without
    # importing the package or consulting a build backend.
    about_path = pyproject_path.parent / "openmed" / "__about__.py"
    try:
        with about_path.open("rb") as handle:
            raw = handle.read(MAX_ABOUT_BYTES + 1)
    except OSError:
        raise SbomError("unable to read the dynamic project version") from None
    if len(raw) > MAX_ABOUT_BYTES:
        raise SbomError("dynamic project version source exceeds the supported limit")
    try:
        about = raw.decode("utf-8")
        module = ast.parse(about)
    except (SyntaxError, UnicodeDecodeError):
        raise SbomError("unable to read the dynamic project version") from None

    assignments: list[object] = []
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            assignments.append(statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__version__"
        ):
            assignments.append(statement.value)

    if len(assignments) != 1:
        raise SbomError("dynamic project version is missing")
    assignment = assignments[0]
    if not isinstance(assignment, ast.Constant) or type(assignment.value) is not str:
        raise SbomError("dynamic project version must be a string literal")
    version = _safe_version(assignment.value)
    if version is None:
        raise SbomError("dynamic project version is missing")
    return version, raw


def _license_value(value: object) -> list[dict[str, Any]]:
    """Convert a small SPDX-like value into a CycloneDX license choice."""
    if type(value) is dict:
        keys = [key for key in ("expression", "id", "name", "text") if key in value]
        return _license_value(value[keys[0]]) if len(keys) == 1 else _unknown_license()
    if type(value) is not str:
        return _unknown_license()

    candidate = " ".join(value.split())
    if not candidate or len(candidate) > MAX_LICENSE_LENGTH:
        return _unknown_license()
    if not _valid_spdx_expression(candidate):
        return _unknown_license()
    tokens = _SPDX_TOKEN_RE.findall(candidate)
    license_ids = {
        token for token in tokens if token not in _SPDX_OPERATORS | {"(", ")"}
    }
    if not license_ids <= _KNOWN_SPDX_LICENSE_IDS:
        return _unknown_license()
    if any(operator in tokens for operator in _SPDX_OPERATORS):
        return [{"expression": candidate}]
    return [{"license": {"id": next(iter(license_ids))}}]


def _valid_spdx_expression(value: str) -> bool:
    tokens = _SPDX_TOKEN_RE.findall(value)
    if "".join(tokens) != re.sub(r"\s+", "", value):
        return False
    if not tokens:
        return False

    depth = 0
    expect_operand = True
    for token in tokens:
        if token == "(":
            if not expect_operand:
                return False
            depth += 1
        elif token == ")":
            if expect_operand or depth == 0:
                return False
            depth -= 1
            expect_operand = False
        elif token in _SPDX_OPERATORS:
            if expect_operand or token == "WITH":
                return False
            expect_operand = True
        else:
            if not expect_operand:
                return False
            expect_operand = False
    return not expect_operand and depth == 0


def _unknown_license() -> list[dict[str, Any]]:
    return [{"license": {"name": UNKNOWN_LICENSE}}]


def _project_licenses(project: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _license_value(project.get("license"))


def _record_licenses(record: PackageRecord) -> list[dict[str, Any]]:
    for key in ("license", "license-expression", "license_expression"):
        if key in record.data:
            return _license_value(record.data[key])

    metadata = record.data.get("metadata")
    if type(metadata) is dict:
        for key in ("license", "license-expression", "license_expression"):
            if key in metadata:
                return _license_value(metadata[key])

    known = KNOWN_LICENSES.get(record.normalized_name)
    return _license_value(known) if known else _unknown_license()


def _dependency_ref(value: object) -> DependencyRef:
    if type(value) is str:
        return DependencyRef(_parse_dependency_name(value))
    if type(value) is not dict:
        raise SbomError("lockfile contains an invalid dependency edge")
    raw_name = value.get("name")
    if type(raw_name) is not str:
        raise SbomError("lockfile dependency edge has no package name")
    _, name = _safe_package_name(raw_name)
    version = _safe_version(value.get("version"))
    source = value.get("source")
    return DependencyRef(
        name,
        version,
        _source_kind(source) if source is not None else None,
    )


def _record_dependencies(record: PackageRecord) -> tuple[DependencyRef, ...]:
    values = record.data.get("dependencies", [])
    if type(values) is not list:
        raise SbomError("lockfile package dependencies must be an array")
    if len(values) > MAX_DEPENDENCIES_PER_RECORD:
        raise SbomError("lockfile package dependencies exceed the supported limit")
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
    if type(values) is not list:
        raise SbomError("uv.lock has no package records")
    if len(values) > MAX_PACKAGE_RECORDS:
        raise SbomError("uv.lock contains too many package records")

    records: dict[str, list[PackageRecord]] = {}
    identities: set[tuple[str, str, str]] = set()
    for value in values:
        if type(value) is not dict:
            raise SbomError("uv.lock contains an invalid package record")
        name, normalized_name = _safe_package_name(value.get("name"))
        record = PackageRecord(
            name=name,
            normalized_name=normalized_name,
            version=_safe_version(value.get("version")),
            source_kind=_source_kind(value.get("source")),
            data=value,
        )
        if record.key in identities:
            raise SbomError("uv.lock contains a duplicate package identity")
        identities.add(record.key)
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
    if len(roots) > MAX_DEPENDENCIES_PER_RECORD:
        raise SbomError("project dependencies exceed the supported item limit")
    queue: deque[PackageRecord] = deque()
    for dependency in roots:
        queue.extend(_resolve(dependency, package_index))

    collected: dict[tuple[str, str, str], PackageRecord] = {}
    while queue:
        record = queue.popleft()
        if record.key in collected:
            continue
        if len(collected) >= MAX_RUNTIME_COMPONENTS:
            raise SbomError("runtime dependency closure exceeds the supported limit")
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
    if value is None:
        return None
    if type(value) is not str:
        raise SbomError("lockfile contains an invalid artifact hash")
    match = _HASH_RE.fullmatch(value.strip())
    if not match:
        raise SbomError("lockfile contains an invalid artifact hash")
    bits = match.group("bits")
    content = match.group("content").lower()
    if len(content) != int(bits) // 4:
        raise SbomError("lockfile contains an invalid artifact hash")
    algorithm = {"256": "SHA-256", "384": "SHA-384", "512": "SHA-512"}[bits]
    return algorithm, content


def _record_hashes(record: PackageRecord) -> list[dict[str, str]]:
    values: list[object] = []
    for key in ("hash", "hashes"):
        value = record.data.get(key)
        if type(value) is list:
            if len(value) > MAX_ARTIFACTS_PER_RECORD:
                raise SbomError("lockfile package hashes exceed the supported limit")
            values.extend(value)
        elif value is not None:
            values.append(value)
    for key in ("sdist", "wheels"):
        artifacts = record.data.get(key)
        if type(artifacts) is dict:
            values.append(artifacts.get("hash"))
        elif type(artifacts) is list:
            if len(artifacts) > MAX_ARTIFACTS_PER_RECORD:
                raise SbomError("lockfile package artifacts exceed the supported limit")
            for artifact in artifacts:
                if type(artifact) is not dict:
                    raise SbomError("lockfile contains an invalid package artifact")
                values.append(artifact.get("hash"))
        elif artifacts is not None:
            raise SbomError("lockfile contains an invalid package artifact")

    hashes = {_hash_value(value) for value in values if value is not None}
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
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        raise SbomError("unable to read the local source revision") from None
    if result.returncode != 0:
        raise SbomError("unable to read the local source revision")
    revision = result.stdout.strip()
    if not _REVISION_RE.fullmatch(revision):
        raise SbomError("local source revision is not a commit hash")
    return revision.lower()


def _validate_revision(value: str) -> str:
    if type(value) is not str or not _REVISION_RE.fullmatch(value.strip()):
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
    pyproject, pyproject_raw = _read_toml(
        pyproject_path,
        "pyproject metadata",
        max_bytes=MAX_PYPROJECT_BYTES,
    )
    lockfile, lockfile_raw = _read_toml(
        lockfile_path,
        "dependency lockfile",
        max_bytes=MAX_LOCKFILE_BYTES,
    )
    project = _project_table(pyproject)
    project_name, _ = _project_name(project)
    project_version, version_source = _project_version(project, pyproject_path)
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

    manifest_hash = _sha256(
        pyproject_raw
        + b"\x00openmed-sbom\x00"
        + lockfile_raw
        + b"\x00version-source\x00"
        + (version_source or b"")
    )
    properties = [
        {"name": "openmed:lockfile-sha256", "value": _sha256(lockfile_raw)},
        {"name": "openmed:manifest-sha256", "value": manifest_hash},
        {"name": "openmed:pyproject-sha256", "value": _sha256(pyproject_raw)},
        {"name": "openmed:source-revision", "value": revision},
    ]
    if version_source is not None:
        properties.append(
            {
                "name": "openmed:version-source-sha256",
                "value": _sha256(version_source),
            }
        )
    properties.sort(key=lambda item: item["name"])

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
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_sbom(output_path: Path, document: Mapping[str, Any]) -> None:
    """Write a rendered SBOM without exposing the output path in errors."""
    temporary_path: str | None = None
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_sbom(document)
        if len(rendered.encode("utf-8")) > MAX_SBOM_BYTES:
            raise SbomError("SBOM output exceeds the supported size limit")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=".openmed-sbom-",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except SbomError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError):
        raise SbomError("unable to write the SBOM output") from None
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass


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
