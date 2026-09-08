"""Pytest bridge for the OpenMedKit Electron integration checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ELECTRON_PACKAGE_DIR = ROOT / "js" / "openmedkit-electron"


def test_openmedkit_electron_package_checks() -> None:
    node = shutil.which("node")
    npm = shutil.which("npm")
    if node is None or npm is None:
        pytest.skip("node and npm are required for OpenMedKit Electron checks")

    _run([npm, "ci"], cwd=ELECTRON_PACKAGE_DIR)
    _run([npm, "test"], cwd=ELECTRON_PACKAGE_DIR)


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stdout
