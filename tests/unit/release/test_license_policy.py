"""Dependency license policy tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_license_policy.py"

spec = importlib.util.spec_from_file_location("check_license_policy", SCRIPT)
assert spec is not None
policy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def write_pyproject(path: Path, body: str) -> Path:
    target = path / "pyproject.toml"
    target.write_text(body, encoding="utf-8")
    return target


def test_current_non_dev_dependencies_are_permissive():
    results = policy.audit_pyproject(ROOT / "pyproject.toml")

    assert results
    assert not [result for result in results if not result.allowed]
    assert {result.entry.name for result in results if result.entry.group == "dev"} == set()


def test_gpl_dependency_fails_policy(tmp_path):
    pyproject = write_pyproject(
        tmp_path,
        """
[project]
name = "sample"
version = "0.0.1"
dependencies = ["example-gpl>=1"]
""",
    )

    results = policy.audit_pyproject(
        pyproject,
        reviewed_licenses={"example-gpl": "GPL-3.0-only"},
    )

    assert len(results) == 1
    assert results[0].allowed is False
    assert "not allowed" in results[0].reason


def test_elastic_dependency_fails_policy(tmp_path):
    pyproject = write_pyproject(
        tmp_path,
        """
[project]
name = "sample"
version = "0.0.1"
dependencies = ["source-available-package>=1"]
""",
    )

    results = policy.audit_pyproject(
        pyproject,
        reviewed_licenses={"source-available-package": "Elastic-2.0"},
    )

    assert len(results) == 1
    assert results[0].allowed is False
    assert "not allowed" in results[0].reason


def test_dev_optional_dependencies_are_not_audited(tmp_path):
    pyproject = write_pyproject(
        tmp_path,
        """
[project]
name = "sample"
version = "0.0.1"
dependencies = ["runtime-package>=1"]

[project.optional-dependencies]
dev = ["dev-only-gpl>=1"]
cli = ["cli-package>=1"]
""",
    )

    results = policy.audit_pyproject(
        pyproject,
        reviewed_licenses={
            "cli-package": "BSD-3-Clause",
            "dev-only-gpl": "GPL-3.0-only",
            "runtime-package": "MIT",
        },
    )

    assert {result.entry.name for result in results} == {"runtime-package", "cli-package"}
    assert all(result.allowed for result in results)


def test_main_returns_nonzero_for_disallowed_dependency(tmp_path, monkeypatch):
    pyproject = write_pyproject(
        tmp_path,
        """
[project]
name = "sample"
version = "0.0.1"
dependencies = ["example-gpl>=1"]
""",
    )
    monkeypatch.setattr(policy, "REVIEWED_LICENSES", {"example-gpl": "GPL-3.0-only"})

    assert policy.main(["--pyproject", str(pyproject)]) == 1
