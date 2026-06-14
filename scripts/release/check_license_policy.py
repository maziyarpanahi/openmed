#!/usr/bin/env python3
"""Fail closed when installable direct dependencies are not permissive."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"

EXCLUDED_OPTIONAL_GROUPS = frozenset({"dev"})

ALLOWED_LICENSE_MARKERS = (
    "MIT",
    "APACHE-2.0",
    "APACHE SOFTWARE LICENSE",
    "BSD",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
)

DISALLOWED_LICENSE_MARKERS = (
    "AGPL",
    "AFFRO GENERAL PUBLIC LICENSE",
    "COMMONS CLAUSE",
    "ELASTIC",
    "GENERAL PUBLIC LICENSE",
    "GPL",
    "LGPL",
    "LESSER GENERAL PUBLIC LICENSE",
    "POLYFORM",
    "SERVER SIDE PUBLIC LICENSE",
    "SSPL",
)

GPL_BRIDGE_EXCEPTIONS = {
    "sdcmicro": "GPL-2.0-only; optional out-of-process disclosure-control bridge",
}

REVIEWED_LICENSES = {
    "accelerate": "Apache-2.0",
    "coremltools": "BSD-3-Clause",
    "faker": "MIT",
    "fastapi": "MIT",
    "gliner": "Apache-2.0",
    "huggingface-hub": "Apache-2.0",
    "httpx": "BSD-3-Clause",
    "mcp": "MIT",
    "mkdocs": "BSD-2-Clause",
    "mkdocs-git-revision-date-localized-plugin": "MIT",
    "mkdocs-material": "MIT",
    "mkdocs-minify-plugin": "MIT",
    "mlx": "MIT",
    "pymdown-extensions": "MIT",
    "pysbd": "MIT",
    "rich": "MIT",
    "safetensors": "Apache-2.0",
    "tiktoken": "MIT",
    "tokenizers": "Apache-2.0",
    "torch": "BSD-3-Clause",
    "transformers": "Apache-2.0",
    "typer": "MIT",
    "uvicorn": "BSD-3-Clause",
}

NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


@dataclass(frozen=True)
class RequirementEntry:
    """A direct requirement declared in project metadata."""

    group: str
    requirement: str
    name: str


@dataclass(frozen=True)
class LicenseAuditResult:
    """License policy decision for one direct requirement."""

    entry: RequirementEntry
    license_text: str
    allowed: bool
    reason: str


def normalize_name(name: str) -> str:
    """Normalize package names according to PEP 503."""

    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_name(requirement: str) -> str:
    """Extract the distribution name from a PEP 508 requirement string."""

    match = NAME_RE.match(requirement)
    if not match:
        raise ValueError(f"Cannot parse dependency name from requirement: {requirement!r}")
    return normalize_name(match.group(1))


def iter_installable_requirements(pyproject: Mapping[str, object]) -> list[RequirementEntry]:
    """Return direct non-dev dependencies from project metadata."""

    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml is missing a [project] table")

    entries: list[RequirementEntry] = []

    dependencies = project.get("dependencies", [])
    if dependencies:
        if not isinstance(dependencies, list):
            raise ValueError("[project].dependencies must be a list")
        entries.extend(
            RequirementEntry("default", str(requirement), dependency_name(str(requirement)))
            for requirement in dependencies
        )

    optional_dependencies = project.get("optional-dependencies", {})
    if optional_dependencies:
        if not isinstance(optional_dependencies, dict):
            raise ValueError("[project.optional-dependencies] must be a table")
        for group, requirements in sorted(optional_dependencies.items()):
            if normalize_name(str(group)) in EXCLUDED_OPTIONAL_GROUPS:
                continue
            if not isinstance(requirements, list):
                raise ValueError(f"[project.optional-dependencies].{group} must be a list")
            entries.extend(
                RequirementEntry(str(group), str(requirement), dependency_name(str(requirement)))
                for requirement in requirements
            )

    return entries


def read_pyproject(path: Path) -> dict[str, object]:
    """Load a pyproject file as TOML."""

    with path.open("rb") as handle:
        return tomllib.load(handle)


def installed_license_text(name: str) -> str | None:
    """Resolve license text from installed distribution metadata, when available."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None

    package_metadata = distribution.metadata
    candidates: list[str] = []

    for field in ("License-Expression", "License"):
        value = package_metadata.get(field)
        if value:
            candidates.append(value)

    for classifier in package_metadata.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            candidates.append(classifier)

    return " ; ".join(candidates) if candidates else None


def resolve_license(
    name: str,
    reviewed_licenses: Mapping[str, str] = REVIEWED_LICENSES,
) -> str:
    """Resolve a direct dependency license from review data or package metadata."""

    normalized = normalize_name(name)
    if normalized in reviewed_licenses:
        return reviewed_licenses[normalized]
    return installed_license_text(normalized) or ""


def contains_marker(license_text: str, markers: Sequence[str]) -> bool:
    """Return true when license text includes one of the normalized markers."""

    normalized = license_text.upper().replace("_", "-")
    return any(marker in normalized for marker in markers)


def is_allowed_license(name: str, license_text: str) -> tuple[bool, str]:
    """Classify license text against the permissive-only policy."""

    normalized_name = normalize_name(name)
    if not license_text:
        return False, "license could not be resolved; add a reviewed license entry"

    has_disallowed = contains_marker(license_text, DISALLOWED_LICENSE_MARKERS)
    has_allowed = contains_marker(license_text, ALLOWED_LICENSE_MARKERS)
    is_bridge_exception = normalized_name in GPL_BRIDGE_EXCEPTIONS

    if is_bridge_exception and has_disallowed and "ELASTIC" not in license_text.upper():
        return True, f"allowed only as bridge exception: {GPL_BRIDGE_EXCEPTIONS[normalized_name]}"

    if has_disallowed:
        return False, "license is not allowed for bundled or in-process dependencies"

    if has_allowed:
        return True, "permissive license"

    return False, "license is outside the MIT/Apache-2.0/BSD allowlist"


def audit_pyproject(
    pyproject_path: Path = DEFAULT_PYPROJECT,
    reviewed_licenses: Mapping[str, str] = REVIEWED_LICENSES,
) -> list[LicenseAuditResult]:
    """Audit direct installable dependencies declared by a pyproject file."""

    pyproject = read_pyproject(pyproject_path)
    results: list[LicenseAuditResult] = []

    for entry in iter_installable_requirements(pyproject):
        license_text = resolve_license(entry.name, reviewed_licenses)
        allowed, reason = is_allowed_license(entry.name, license_text)
        results.append(LicenseAuditResult(entry, license_text, allowed, reason))

    return results


def print_results(results: Sequence[LicenseAuditResult]) -> None:
    """Print a compact human-readable audit summary."""

    failures = [result for result in results if not result.allowed]

    if failures:
        print("License policy failed:", file=sys.stderr)
        for result in failures:
            print(
                f"- {result.entry.group}:{result.entry.name} "
                f"({result.license_text or 'unknown'}): {result.reason}",
                file=sys.stderr,
            )
        return

    print("License policy passed")
    for result in results:
        print(f"- {result.entry.group}:{result.entry.name}: {result.license_text}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Check direct non-dev dependency licenses against OpenMed policy."
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="Path to pyproject.toml to audit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dependency license audit."""

    args = parse_args(argv)
    results = audit_pyproject(args.pyproject)
    print_results(results)
    return 1 if any(not result.allowed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
