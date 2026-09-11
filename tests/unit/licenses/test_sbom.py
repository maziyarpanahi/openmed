"""Focused tests for the deterministic offline SBOM evidence bundle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[3] / "scripts" / "licenses" / "sbom.py"
ROOT = MODULE_PATH.parents[2]
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


def test_current_manifests_generate_the_base_runtime_closure() -> None:
    document = sbom.build_sbom(
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        source_revision="deadbeef",
    )

    component_names = {component["name"] for component in document["components"]}
    assert component_names == {"faker", "jieba", "pysbd", "pyyaml", "tzdata"}
    root_ref = document["metadata"]["component"]["bom-ref"]
    root_dependencies = next(
        entry for entry in document["dependencies"] if entry["ref"] == root_ref
    )
    assert [
        ref.split("/")[-1].split("@")[0] for ref in root_dependencies["dependsOn"]
    ] == [
        "faker",
        "jieba",
        "pysbd",
        "pyyaml",
    ]
    properties = {
        item["name"]: item["value"] for item in document["metadata"]["properties"]
    }
    assert "openmed:version-source-sha256" in properties
    rendered = sbom.render_sbom(document)
    assert "https://" not in rendered
    assert str(ROOT) not in rendered


@pytest.mark.parametrize(
    "expression",
    ["((MIT)", "MIT OR", "MIT Apache-2.0", "Synthetic-Private-License"],
)
def test_invalid_or_unreviewed_license_syntax_becomes_noassertion(
    expression: str,
) -> None:
    assert sbom._license_value(expression) == [
        {"license": {"name": sbom.UNKNOWN_LICENSE}}
    ]


def test_valid_compound_spdx_expression_is_preserved() -> None:
    assert sbom._license_value("MIT OR Apache-2.0") == [
        {"expression": "MIT OR Apache-2.0"}
    ]


def test_manifest_reads_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    monkeypatch.setattr(sbom, "MAX_PYPROJECT_BYTES", 8)

    with pytest.raises(sbom.SbomError, match="supported size limit"):
        sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")


def test_invalid_artifact_hash_fails_without_echoing_source_values(
    tmp_path: Path,
) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    marker = "synthetic-private-artifact"
    contents = lockfile.read_text(encoding="utf-8").replace(
        "sha256:" + "a" * 64,
        marker,
    )
    lockfile.write_text(contents, encoding="utf-8")

    with pytest.raises(sbom.SbomError) as error:
        sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")
    assert marker not in str(error.value)


def test_duplicate_package_identity_fails_closed(tmp_path: Path) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    with lockfile.open("a", encoding="utf-8") as handle:
        handle.write(
            """
[[package]]
name = "alpha"
version = "1.0.0"
source = { registry = "https://example.invalid/duplicate" }
"""
        )

    with pytest.raises(sbom.SbomError, match="duplicate package identity"):
        sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")


def test_package_record_repr_hides_raw_manifest_data() -> None:
    marker = "https://example.invalid/synthetic-private-source"
    record = sbom.PackageRecord(
        name="synthetic-package",
        normalized_name="synthetic-package",
        version="1.0.0",
        source_kind="registry",
        data={"source": {"registry": marker}},
    )

    assert marker not in repr(record)


def test_dynamic_version_requires_a_bounded_local_source(tmp_path: Path) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'version = "1.2.3"',
            'dynamic = ["version"]',
        ),
        encoding="utf-8",
    )

    with pytest.raises(sbom.SbomError, match="dynamic project version"):
        sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")


def test_dynamic_version_is_read_from_a_literal_assignment(tmp_path: Path) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'version = "1.2.3"',
            'dynamic = ["version"]',
        ),
        encoding="utf-8",
    )
    about = tmp_path / "openmed" / "__about__.py"
    about.parent.mkdir()
    about.write_text(
        '# __version__ = "9.9.9"\n__version__: str = "1.2.3"\n',
        encoding="utf-8",
    )

    document = sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")

    assert document["metadata"]["component"]["version"] == "1.2.3"


def test_ambiguous_source_kinds_fail_closed(tmp_path: Path) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    contents = lockfile.read_text(encoding="utf-8").replace(
        'source = { registry = "https://example.invalid/synthetic-package" }',
        'source = { registry = "https://example.invalid", directory = "/private" }',
    )
    lockfile.write_text(contents, encoding="utf-8")

    with pytest.raises(sbom.SbomError, match="multiple source kinds"):
        sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")


def test_invalid_source_record_fails_closed(tmp_path: Path) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    contents = lockfile.read_text(encoding="utf-8").replace(
        'source = { registry = "https://example.invalid/synthetic-package" }',
        'source = "synthetic-private-source"',
    )
    lockfile.write_text(contents, encoding="utf-8")

    with pytest.raises(sbom.SbomError) as error:
        sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")
    assert "synthetic-private" not in str(error.value)


def test_dependency_and_package_counts_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    monkeypatch.setattr(sbom, "MAX_DEPENDENCIES_PER_RECORD", 1)
    with pytest.raises(sbom.SbomError, match="dependencies exceed"):
        sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")

    monkeypatch.setattr(sbom, "MAX_DEPENDENCIES_PER_RECORD", 10)
    monkeypatch.setattr(sbom, "MAX_PACKAGE_RECORDS", 2)
    with pytest.raises(sbom.SbomError, match="too many package records"):
        sbom.build_sbom(pyproject, lockfile, source_revision="deadbeef")


def test_invalid_revision_failure_does_not_echo_the_value(tmp_path: Path) -> None:
    pyproject, lockfile = _write_manifests(tmp_path)
    marker = "synthetic-private-revision"
    with pytest.raises(sbom.SbomError) as error:
        sbom.build_sbom(pyproject, lockfile, source_revision=marker)
    assert marker not in str(error.value)


def test_failed_atomic_replace_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "evidence.json"
    output.write_text("existing\n", encoding="utf-8")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("synthetic-private-path")

    monkeypatch.setattr(sbom.os, "replace", fail_replace)
    with pytest.raises(sbom.SbomError) as error:
        sbom.write_sbom(output, {"bomFormat": "CycloneDX"})

    assert "synthetic-private" not in str(error.value)
    assert output.read_text(encoding="utf-8") == "existing\n"
    assert list(tmp_path.glob(".openmed-sbom-*")) == []


def test_non_finite_output_is_rejected_without_creating_a_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"

    with pytest.raises(sbom.SbomError, match="unable to write"):
        sbom.write_sbom(output, {"value": float("nan")})

    assert not output.exists()
    assert list(tmp_path.glob(".openmed-sbom-*")) == []


def test_git_revision_timeout_is_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(*args: object, **kwargs: object) -> None:
        raise sbom.subprocess.TimeoutExpired("synthetic-private-command", 1)

    monkeypatch.setattr(sbom.subprocess, "run", fail_run)
    with pytest.raises(sbom.SbomError) as error:
        sbom._revision_from_git(tmp_path)
    assert "synthetic-private" not in str(error.value)


def test_malformed_requirement_fails_without_echoing_the_value() -> None:
    marker = "synthetic-package private-value"
    with pytest.raises(sbom.SbomError) as error:
        sbom._parse_dependency_name(marker)
    assert marker not in str(error.value)


@pytest.mark.parametrize("expression", ["(MIT)", "((Apache-2.0))"])
def test_parenthesized_single_license_emits_a_license_id(expression: str) -> None:
    assert sbom._license_value(expression) == [
        {"license": {"id": expression.strip("()")}}
    ]
