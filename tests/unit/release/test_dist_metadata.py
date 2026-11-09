"""Tests for the built-distribution metadata check (#5400)."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_dist_metadata.py"

spec = importlib.util.spec_from_file_location("check_dist_metadata", SCRIPT)
assert spec is not None
dist_metadata = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = dist_metadata
spec.loader.exec_module(dist_metadata)

PYPROJECT = """\
[project]
name = "synthetic-pkg"
version = "0.0.1"
requires-python = ">=3.11"
classifiers = [
  "Programming Language :: Python :: 3 :: Only",
  "Programming Language :: Python :: 3.11",
  "Operating System :: OS Independent",
]
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE"]
"""

GOOD_HEADERS = [
    ("Metadata-Version", "2.4"),
    ("Name", "synthetic-pkg"),
    ("Version", "0.0.1"),
    ("Classifier", "Operating System :: OS Independent"),
    ("Classifier", "Programming Language :: Python :: 3 :: Only"),
    ("Classifier", "Programming Language :: Python :: 3.11"),
    ("License-Expression", "Apache-2.0"),
    ("License-File", "LICENSE"),
    ("License-File", "NOTICE"),
    ("Requires-Python", ">=3.11"),
]


def metadata_text(headers: list[tuple[str, str]]) -> str:
    return (
        "".join(f"{key}: {value}\n" for key, value in headers) + "\nlong description\n"
    )


def without(key: str, value: str | None = None) -> list[tuple[str, str]]:
    return [(k, v) for k, v in GOOD_HEADERS if not (k == key and value in (None, v))]


def replaced(key: str, value: str) -> list[tuple[str, str]]:
    return [(k, value if k == key else v) for k, v in GOOD_HEADERS]


def write_wheel(
    dist: Path,
    headers: list[tuple[str, str]],
    licenses: tuple[str, ...] = ("LICENSE", "NOTICE"),
    name: str = "synthetic_pkg-0.0.1-py3-none-any.whl",
) -> None:
    info = "synthetic_pkg-0.0.1.dist-info"
    with zipfile.ZipFile(dist / name, "w") as archive:
        archive.writestr("synthetic_pkg/__init__.py", "")
        archive.writestr(f"{info}/METADATA", metadata_text(headers))
        for path in licenses:
            archive.writestr(f"{info}/licenses/{path}", "synthetic license text\n")


def write_sdist(
    dist: Path,
    headers: list[tuple[str, str]],
    licenses: tuple[str, ...] = ("LICENSE", "NOTICE"),
) -> None:
    root = "synthetic_pkg-0.0.1"
    files = {f"{root}/PKG-INFO": metadata_text(headers)}
    files.update({f"{root}/{path}": "synthetic license text\n" for path in licenses})
    with tarfile.open(dist / f"{root}.tar.gz", "w:gz") as archive:
        for member_name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


@pytest.fixture
def contract(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT, encoding="utf-8")
    return dist_metadata.load_contract(pyproject)


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    path = tmp_path / "dist"
    path.mkdir()
    return path


def codes(issues) -> set[str]:
    return {str(issue) for issue in issues}


def test_matching_artifacts_pass(contract, dist):
    write_wheel(dist, GOOD_HEADERS)
    write_sdist(dist, GOOD_HEADERS)

    assert dist_metadata.check_dist(dist, contract) == []


def test_missing_python_classifier_fails(contract, dist):
    headers = without("Classifier", "Programming Language :: Python :: 3.11")
    write_wheel(dist, headers)
    write_sdist(dist, GOOD_HEADERS)

    assert codes(dist_metadata.check_dist(dist, contract)) == {
        "wheel: python_classifiers_mismatch"
    }


def test_extra_python_classifier_fails(contract, dist):
    headers = [*GOOD_HEADERS, ("Classifier", "Programming Language :: Python :: 3.10")]
    write_wheel(dist, GOOD_HEADERS)
    write_sdist(dist, headers)

    assert codes(dist_metadata.check_dist(dist, contract)) == {
        "sdist: python_classifiers_mismatch"
    }


def test_non_python_classifiers_are_not_compared(contract, dist):
    headers = without("Classifier", "Operating System :: OS Independent")
    write_wheel(dist, headers)
    write_sdist(dist, headers)

    assert dist_metadata.check_dist(dist, contract) == []


def test_wrong_requires_python_fails(contract, dist):
    headers = replaced("Requires-Python", ">=3.10")
    write_wheel(dist, headers)
    write_sdist(dist, headers)

    assert codes(dist_metadata.check_dist(dist, contract)) == {
        "wheel: requires_python_mismatch",
        "sdist: requires_python_mismatch",
    }


def test_legacy_license_field_without_expression_fails(contract, dist):
    # The pre-PEP 639 table form emits `License:` and no `License-Expression`.
    headers = [*without("License-Expression"), ("License", "Apache-2.0")]
    write_wheel(dist, headers)
    write_sdist(dist, GOOD_HEADERS)

    assert codes(dist_metadata.check_dist(dist, contract)) == {
        "wheel: license_expression_mismatch"
    }


def test_missing_license_file_entry_fails(contract, dist):
    write_wheel(dist, GOOD_HEADERS)
    write_sdist(dist, without("License-File", "NOTICE"))

    assert codes(dist_metadata.check_dist(dist, contract)) == {
        "sdist: license_file_entries_mismatch"
    }


def test_license_file_missing_from_archive_fails(contract, dist):
    write_wheel(dist, GOOD_HEADERS, licenses=("LICENSE",))
    write_sdist(dist, GOOD_HEADERS, licenses=("LICENSE",))

    assert codes(dist_metadata.check_dist(dist, contract)) == {
        "wheel: license_file_not_shipped",
        "sdist: license_file_not_shipped",
    }


def test_wrong_artifact_count_fails(contract, dist):
    write_wheel(dist, GOOD_HEADERS)
    write_wheel(dist, GOOD_HEADERS, name="synthetic_pkg-0.0.2-py3-none-any.whl")

    assert codes(dist_metadata.check_dist(dist, contract)) == {
        "wheel: artifact_count_not_one",
        "sdist: artifact_count_not_one",
    }


def test_unreadable_wheel_is_reported(contract, dist):
    (dist / "synthetic_pkg-0.0.1-py3-none-any.whl").write_bytes(b"not a zip")
    write_sdist(dist, GOOD_HEADERS)

    assert codes(dist_metadata.check_dist(dist, contract)) == {
        "wheel: artifact_unreadable"
    }


def test_table_license_is_rejected(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        PYPROJECT.replace(
            'license = "Apache-2.0"', 'license = { text = "Apache-2.0" }'
        ),
        encoding="utf-8",
    )

    with pytest.raises(dist_metadata.ContractError, match="not_spdx_expression"):
        dist_metadata.load_contract(pyproject)


def test_glob_license_files_are_rejected(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        PYPROJECT.replace('["LICENSE", "NOTICE"]', '["LICEN[CS]E*"]'),
        encoding="utf-8",
    )

    with pytest.raises(dist_metadata.ContractError, match="not_literal_paths"):
        dist_metadata.load_contract(pyproject)


def test_cli_exit_codes(contract, dist, tmp_path: Path, capsys):
    pyproject = tmp_path / "pyproject.toml"
    write_wheel(dist, GOOD_HEADERS)
    write_sdist(dist, GOOD_HEADERS)
    argv = ["--dist-dir", str(dist), "--pyproject", str(pyproject)]

    assert dist_metadata.main(argv) == 0

    (dist / "synthetic_pkg-0.0.1.tar.gz").unlink()
    write_sdist(dist, replaced("Requires-Python", ">=3.12"))

    assert dist_metadata.main(argv) == 1
    assert "sdist: requires_python_mismatch" in capsys.readouterr().err


def test_repository_declares_pep_639_contract():
    contract = dist_metadata.load_contract(ROOT / "pyproject.toml")
    build_system = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))[
        "build-system"
    ]

    assert contract.license_expression == "Apache-2.0"
    assert contract.license_files == ("LICENSE", "NOTICE")
    assert all((ROOT / path).is_file() for path in contract.license_files)
    assert "hatchling>=1.27" in build_system["requires"]
