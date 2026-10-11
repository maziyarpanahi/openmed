"""Run the synthetic multimodal CI slice with required resources and zero skips."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import socket
import sys
from pathlib import Path

import pytest

from scripts.release.check_license_policy import (
    REVIEWED_LICENSES,
    dependency_name,
    is_allowed_license,
    read_pyproject,
)

ROOT = Path(__file__).resolve().parents[2]
# These distributions are already declared by the multimodal extra. Do not install
# torch, onnx, python-doctr, easyocr, PaddleOCR, or any model weights in this lane.
DEPENDENCIES = {
    "fsspec": "fsspec",
    "jinja2": "jinja2",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pdfplumber": "pdfplumber",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "pillow": "PIL",
    "piexif": "piexif",
    "pikepdf": "pikepdf",
    "pydicom": "pydicom",
    "pytesseract": "pytesseract",
    "markdown-it-py": "markdown_it",
}
TEST_PATHS = (
    "tests/unit/multimodal",
    "tests/unit/eval/test_multimodal_dicom_suite.py",
    "tests/unit/eval/test_dicom_sr_content_eval.py",
    "tests/unit/eval/test_ocr_routing.py",
    "tests/fuzz/test_format_parsers_fuzz.py",
)
OFFLINE_ENV = {
    "OPENMED_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HYPOTHESIS_PROFILE": "fuzz-default",
}


def requirements(pyproject: Path = ROOT / "pyproject.toml") -> list[str]:
    """Select reviewed lightweight requirements using the extra's exact bounds."""
    extra = read_pyproject(pyproject)["project"]["optional-dependencies"]["multimodal"]
    selected = {dependency_name(item): item for item in extra}
    result = []
    for name in DEPENDENCIES:
        if name not in selected:
            raise ValueError(
                f"multimodal-ci: extra missing required dependency: {name}"
            )
        license_text = REVIEWED_LICENSES.get(name, "")
        if not is_allowed_license(name, license_text)[0]:
            raise ValueError(f"multimodal-ci: dependency license not allowed: {name}")
        result.append(selected[name])
    return result


def missing_resources() -> list[str]:
    """Return controlled dependency names, without import errors or private paths."""
    missing = []
    for name, module in DEPENDENCIES.items():
        try:
            importlib.import_module(module)
        except Exception:
            # Broken native imports must fail just like absent packages. Never
            # include third-party exception text, which can contain input values.
            missing.append(name)
    if shutil.which("tesseract") is None:
        missing.append("tesseract-binary")
    return missing


class ZeroSkipGuard:
    """Fail on collection/runtime skips and prevent outbound test connections."""

    def __init__(self) -> None:
        self.skips = 0

    def pytest_collectreport(self, report) -> None:
        """Count module-level importorskip reports."""
        if report.skipped:
            self.skips += 1

    def pytest_runtest_logreport(self, report) -> None:
        """Count skips at setup, call or teardown."""
        if report.skipped:
            self.skips += 1

    @pytest.fixture(autouse=True)
    def offline_connections(self, monkeypatch):
        """Block outbound sockets without echoing addresses or payloads."""

        def deny(*args, **kwargs):
            raise RuntimeError("multimodal-ci: outbound network forbidden")

        monkeypatch.setattr(socket.socket, "connect", deny)
        monkeypatch.setattr(socket.socket, "connect_ex", deny)
        monkeypatch.setattr(socket, "create_connection", deny)

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        """Make a skipped module or test a failed lane even if others passed."""
        if self.skips:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter) -> None:
        """Emit counts only; do not repeat potentially sensitive skip reasons."""
        if self.skips:
            terminalreporter.write_line(
                f"multimodal-ci: zero-skip gate failed; skipped reports={self.skips}",
                red=True,
            )


def main(argv: list[str] | None = None) -> int:
    """Print install requirements or run the bounded offline CI slice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", action="store_true")
    args = parser.parse_args(argv)
    try:
        selected = requirements()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.requirements:
        print("\n".join(selected))
        return 0
    missing = missing_resources()
    if missing:
        print(
            "multimodal-ci: missing or unusable required resources: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    os.environ.update(OFFLINE_ENV)
    os.chdir(ROOT)
    return int(
        pytest.main(
            [*TEST_PATHS, "-q", "--tb=short", "--show-capture=no", "--timeout=120"],
            plugins=[ZeroSkipGuard()],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
