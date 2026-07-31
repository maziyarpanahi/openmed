"""Pytest bridge for the OpenMedKit web npm package checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "js" / "openmedkit-web"


def test_openmedkit_web_package_checks() -> None:
    node = shutil.which("node")
    npm = shutil.which("npm")
    if node is None or npm is None:
        pytest.skip("node and npm are required for OpenMedKit web package checks")

    _run([npm, "ci"])
    _run([npm, "test"])


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PACKAGE_DIR,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stdout
