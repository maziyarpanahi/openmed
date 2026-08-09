"""Focused tests for the deterministic offline SBOM evidence bundle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[3] / "scripts" / "licenses" / "sbom.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "openmed_license_sbom", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
sbom = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = sbom
MODULE_SPEC.loader.exec_module(sbom)


def _write_manifests(tmp_path: Path) -> tuple[Path, Path]:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "openmed"
version = "1.2.3"
license = { text = "Apache-2.0" }
dependencies = ["Alpha>=1", "beta"]
""".lstrip(),
        encoding="utf-8",
    )

    alpha_hash = "sha256:" + "a" * 64
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        f"""
version = 1
revision = 1

[[package]]
name = "alpha"
version = "1.0.0"
source = {{ registry = "https://example.invalid/synthetic-package" }}
license = "MIT"
dependencies = [{{ name = "beta" }}]
sdist = {{ url = "file:///synthetic/source/alpha.tar.gz", hash = "{alpha_hash}" }}

[[package]]
name = "beta"
version = "2.0.0"
source = {{ directory = "/synthetic/source" }}
license = {{ id = "Apache-2.0" }}

[[package]]
name = "unused"
version = "9.9.9"
source = {{ registry = "https://pypi.org/simple" }}
""".lstrip(),
        encoding="utf-8",
    )
    return pyproject, lockfile


def test_build_sbom_is_deterministic_and_contains_safe_provenance(
    tmp_path: Path,
) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)

    first = sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")
    second = sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")

    assert sbom.render_sbom(first) == sbom.render_sbom(second)
    assert first["bomFormat"] == "CycloneDX"
    assert first["specVersion"] == "1.6"
    assert "serialNumber" not in first
    assert "timestamp" not in first["metadata"]

    components = {component["name"]: component for component in first["components"]}
    assert set(components) == {"alpha", "beta"}
    assert components["alpha"]["licenses"] == [{"license": {"id": "MIT"}}]
    assert components["alpha"]["hashes"] == [{"alg": "SHA-256", "content": "a" * 64}]

    root_ref = first["metadata"]["component"]["bom-ref"]
    root_dependencies = next(
        item for item in first["dependencies"] if item["ref"] == root_ref
    )
    assert root_dependencies["dependsOn"] == [
        "pkg:generic/beta@2.0.0?source=directory",
        "pkg:pypi/alpha@1.0.0",
    ]

    properties = {
        item["name"]: item["value"] for item in first["metadata"]["properties"]
    }
    assert properties["openmed:source-revision"] == "deadbeef"
    assert properties["openmed:manifest-sha256"]
    assert properties["openmed:pyproject-sha256"]
    assert properties["openmed:lockfile-sha256"]

    rendered = sbom.render_sbom(first)
    assert "/synthetic/source" not in rendered
    assert "example.invalid" not in rendered


def test_cli_writes_without_network_and_does_not_log_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    output = tmp_path / "nested" / "evidence.json"

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("SBOM generation must not invoke a subprocess")

    monkeypatch.setattr(sbom.subprocess, "run", fail_if_called)
    assert (
        sbom.main(
            [
                "--pyproject",
                str(pyproject),
                "--lockfile",
                str(lockfile),
                "--output",
                str(output),
                "--source-revision",
                "deadbeef",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert output.exists()
    assert str(tmp_path) not in captured.out
    assert str(tmp_path) not in captured.err
    assert sbom.render_sbom(sbom.build_sbom(pyproject, lockfile, "deadbeef")) == (
        output.read_text(encoding="utf-8")
    )
