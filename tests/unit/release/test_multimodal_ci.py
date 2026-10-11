"""Offline controls for the optional-dependency CI gate."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import yaml

from scripts.release.check_license_policy import dependency_name, read_pyproject
from scripts.testing import multimodal_ci as lane


def test_requirements_use_extra_bounds_and_exclude_heavy_backends():
    selected = lane.requirements()
    extra = read_pyproject(lane.ROOT / "pyproject.toml")["project"][
        "optional-dependencies"
    ]["multimodal"]
    assert set(selected) <= set(extra)
    assert {dependency_name(item) for item in selected} == set(lane.DEPENDENCIES)
    assert not set(lane.DEPENDENCIES) & {
        "torch",
        "onnx",
        "python-doctr",
        "easyocr",
        "paddleocr",
    }


def test_requirement_missing_from_extra_fails_closed(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text("[project.optional-dependencies]\nmultimodal = []\n")
    with pytest.raises(ValueError, match="extra missing required dependency: fsspec"):
        lane.requirements(project)


def test_unreviewed_license_fails_closed(monkeypatch):
    monkeypatch.setitem(lane.REVIEWED_LICENSES, "pydicom", "GPL-3.0-only")
    with pytest.raises(ValueError, match="dependency license not allowed: pydicom"):
        lane.requirements()


@pytest.mark.parametrize("removed", list(lane.DEPENDENCIES))
def test_each_missing_dependency_is_named_without_exception_values(
    monkeypatch, removed
):
    def import_module(module):
        if module == lane.DEPENDENCIES[removed]:
            raise ImportError("private input and path must never be reported")
        return object()

    monkeypatch.setattr(lane.importlib, "import_module", import_module)
    monkeypatch.setattr(lane.shutil, "which", lambda name: "synthetic-binary")
    assert lane.missing_resources() == [removed]


def test_missing_binary_is_required(monkeypatch):
    monkeypatch.setattr(lane.importlib, "import_module", lambda module: object())
    monkeypatch.setattr(lane.shutil, "which", lambda name: None)
    assert lane.missing_resources() == ["tesseract-binary"]


def test_missing_dependency_prevents_pytest(monkeypatch, capsys):
    monkeypatch.setattr(lane, "missing_resources", lambda: ["pydicom"])

    def unexpected(*args, **kwargs):
        pytest.fail("pytest must not run with a missing required dependency")

    monkeypatch.setattr(lane.pytest, "main", unexpected)
    assert lane.main([]) == 2
    assert capsys.readouterr().err == (
        "multimodal-ci: missing or unusable required resources: pydicom\n"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("source", "code", "message"),
    [
        ("def test_ok(): assert True", 0, "1 passed"),
        (
            "import pytest\npytest.importorskip('absent_synthetic_dependency_3811')",
            1,
            "zero-skip gate failed; skipped reports=1",
        ),
        (
            "import pytest\ndef test_skip(): pytest.skip('synthetic')",
            1,
            "zero-skip gate failed; skipped reports=1",
        ),
        (
            "import pytest\n@pytest.fixture\ndef resource(): pytest.skip('synthetic')\n"
            "def test_setup(resource): pass",
            1,
            "zero-skip gate failed; skipped reports=1",
        ),
        ("def test_fail(): assert False", 1, "1 failed"),
        (
            "import socket\ndef test_network():\n"
            " socket.create_connection(('localhost', 1))",
            1,
            "multimodal-ci: outbound network forbidden",
        ),
    ],
)
def test_guard_real_pytest_reports(tmp_path, source, code, message):
    test_file = tmp_path / "test_synthetic.py"
    test_file.write_text(source + "\n")
    script = (
        "import pytest; from scripts.testing.multimodal_ci import ZeroSkipGuard; "
        f"raise SystemExit(pytest.main([{str(test_file)!r}, '-q', "
        f"'--confcutdir={tmp_path}', '--tb=short'], plugins=[ZeroSkipGuard()]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=lane.ROOT,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == code, result.stdout + result.stderr
    assert message in result.stdout


def test_workflow_runs_both_platforms_and_build_requires_lane():
    workflow = yaml.safe_load((lane.ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["multimodal-tests"]
    assert job["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-latest"]
    assert job["strategy"]["fail-fast"] is False
    assert job["timeout-minutes"] == 20
    steps = job["steps"]
    assert any("apt-get install -y tesseract-ocr" in s.get("run", "") for s in steps)
    assert any("brew install tesseract" in s.get("run", "") for s in steps)
    install = next(s["run"] for s in steps if "lightweight" in s.get("name", ""))
    assert "--requirements" in install
    assert "--extra multimodal" not in install
    test = next(s for s in steps if s.get("run") == "make test-multimodal")
    assert test["timeout-minutes"] == 10
    assert "multimodal-tests" in workflow["jobs"]["build"]["needs"]
    assert (
        "needs.multimodal-tests.result == 'success'" in workflow["jobs"]["build"]["if"]
    )


def test_lane_includes_existing_multimodal_eval_and_parser_targets():
    expected = set((lane.ROOT / "tests/unit/eval").glob("test_*multimodal*.py"))
    expected |= set((lane.ROOT / "tests/unit/eval").glob("test_dicom*.py"))
    expected.add(lane.ROOT / "tests/unit/eval/test_ocr_routing.py")
    actual = {lane.ROOT / p for p in lane.TEST_PATHS}
    assert expected <= actual
    assert lane.ROOT / "tests/fuzz/test_format_parsers_fuzz.py" in actual


def test_requirements_cli_prints_installable_lines(capsys):
    assert lane.main(["--requirements"]) == 0
    assert capsys.readouterr().out.splitlines() == lane.requirements()
