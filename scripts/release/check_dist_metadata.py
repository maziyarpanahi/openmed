#!/usr/bin/env python3
"""Check built wheel and sdist metadata against ``pyproject.toml``.

Reads ``METADATA`` from the wheel and ``PKG-INFO`` from the sdist in a dist
directory and compares them with the ``[project]`` table:

* ``Requires-Python`` equals ``requires-python``;
* the ``Programming Language :: Python`` classifiers equal the declared ones;
* ``License-Expression`` equals the SPDX ``license`` expression (PEP 639);
* the ``License-File`` entries equal ``license-files``, and each listed file
  ships in the artifact.

Failures are printed as controlled issue codes, one per line, and the script
exits 1. It reads only the two archives and ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from email.parser import HeaderParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PYTHON_CLASSIFIER_PREFIX = "Programming Language :: Python"


@dataclass(frozen=True)
class Contract:
    """Distribution metadata promised by ``pyproject.toml``."""

    requires_python: str
    python_classifiers: frozenset[str]
    license_expression: str
    license_files: tuple[str, ...]


@dataclass(frozen=True)
class Issue:
    """One metadata mismatch, reported by artifact kind and code."""

    artifact: str
    code: str

    def __str__(self) -> str:
        return f"{self.artifact}: {self.code}"


class ContractError(ValueError):
    """``pyproject.toml`` does not declare a checkable contract."""


def load_contract(pyproject: Path) -> Contract:
    """Read the metadata contract from ``pyproject.toml``.

    Args:
        pyproject: Path to the project's ``pyproject.toml``.

    Returns:
        The contract the built artifacts must match.

    Raises:
        ContractError: If ``license`` is not an SPDX expression string or
            ``license-files`` is not a list of literal paths.
    """
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    license_expression = project.get("license")
    if not isinstance(license_expression, str) or not license_expression.strip():
        raise ContractError("pyproject_license_not_spdx_expression")
    license_files = project.get("license-files")
    if (
        not isinstance(license_files, list)
        or not license_files
        or not all(isinstance(path, str) for path in license_files)
        or any(set(path) & set("*?[") for path in license_files)
    ):
        raise ContractError("pyproject_license_files_not_literal_paths")
    classifiers = project.get("classifiers", [])
    return Contract(
        requires_python=project["requires-python"],
        python_classifiers=frozenset(
            c for c in classifiers if c.startswith(PYTHON_CLASSIFIER_PREFIX)
        ),
        license_expression=license_expression,
        license_files=tuple(license_files),
    )


def check_metadata(
    artifact: str,
    metadata_text: str,
    shipped_license_files: set[str],
    contract: Contract,
) -> list[Issue]:
    """Compare one artifact's core metadata with the contract.

    Args:
        artifact: ``"wheel"`` or ``"sdist"``, used in issue reports.
        metadata_text: The artifact's ``METADATA`` or ``PKG-INFO`` text.
        shipped_license_files: License paths present in the artifact, relative
            to where core metadata expects them.
        contract: The contract from :func:`load_contract`.

    Returns:
        The mismatches found, empty when the artifact matches.
    """
    headers = HeaderParser().parsestr(metadata_text)
    issues: list[Issue] = []

    if headers.get("Requires-Python") != contract.requires_python:
        issues.append(Issue(artifact, "requires_python_mismatch"))

    classifiers = {
        c
        for c in headers.get_all("Classifier", [])
        if c.startswith(PYTHON_CLASSIFIER_PREFIX)
    }
    if classifiers != contract.python_classifiers:
        issues.append(Issue(artifact, "python_classifiers_mismatch"))

    if headers.get("License-Expression") != contract.license_expression:
        issues.append(Issue(artifact, "license_expression_mismatch"))

    declared = headers.get_all("License-File", [])
    if sorted(declared) != sorted(contract.license_files):
        issues.append(Issue(artifact, "license_file_entries_mismatch"))

    if not set(contract.license_files) <= shipped_license_files:
        issues.append(Issue(artifact, "license_file_not_shipped"))

    return issues


def read_wheel(path: Path) -> tuple[str, set[str]]:
    """Return a wheel's ``METADATA`` text and its PEP 639 license paths."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = [
            n for n in names if n.count("/") == 1 and n.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1:
            raise ContractError("wheel_metadata_not_found")
        prefix = f"{metadata[0].split('/')[0]}/licenses/"
        shipped = {n[len(prefix) :] for n in names if n.startswith(prefix)}
        return archive.read(metadata[0]).decode("utf-8"), shipped


def read_sdist(path: Path) -> tuple[str, set[str]]:
    """Return an sdist's ``PKG-INFO`` text and its files relative to the root."""
    with tarfile.open(path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        roots = {PurePosixPath(m.name).parts[0] for m in members}
        if len(roots) != 1:
            raise ContractError("sdist_root_not_unique")
        root = roots.pop()
        pkg_info = archive.extractfile(f"{root}/PKG-INFO")
        if pkg_info is None:
            raise ContractError("sdist_metadata_not_found")
        shipped = {
            str(PurePosixPath(*PurePosixPath(m.name).parts[1:]))
            for m in members
            if len(PurePosixPath(m.name).parts) > 1
        }
        return pkg_info.read().decode("utf-8"), shipped


def check_dist(dist_dir: Path, contract: Contract) -> list[Issue]:
    """Check the single wheel and single sdist in ``dist_dir``.

    Args:
        dist_dir: Directory holding exactly one ``.whl`` and one ``.tar.gz``.
        contract: The contract from :func:`load_contract`.

    Returns:
        Every mismatch found across both artifacts.
    """
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    issues: list[Issue] = []
    for kind, found, reader in (
        ("wheel", wheels, read_wheel),
        ("sdist", sdists, read_sdist),
    ):
        if len(found) != 1:
            issues.append(Issue(kind, "artifact_count_not_one"))
            continue
        try:
            text, shipped = reader(found[0])
        except (
            ContractError,
            KeyError,
            OSError,
            tarfile.TarError,
            zipfile.BadZipFile,
        ) as exc:
            code = str(exc) if isinstance(exc, ContractError) else "artifact_unreadable"
            issues.append(Issue(kind, code))
            continue
        issues.extend(check_metadata(kind, text, shipped, contract))
    return issues


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    args = parser.parse_args(argv)

    try:
        contract = load_contract(args.pyproject)
    except ContractError as exc:
        print(f"pyproject: {exc}", file=sys.stderr)
        return 1

    issues = check_dist(args.dist_dir, contract)
    for issue in issues:
        print(issue, file=sys.stderr)
    if issues:
        return 1
    print("dist metadata matches pyproject.toml: wheel, sdist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
