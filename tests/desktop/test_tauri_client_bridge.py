"""Pytest bridge for the OpenMedKit Tauri client checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TAURI_PACKAGE_DIR = ROOT / "js" / "openmedkit-tauri"


def test_openmedkit_tauri_package_checks() -> None:
    node = shutil.which("node")
    npm = shutil.which("npm")
    if node is None or npm is None:
        pytest.skip("node and npm are required for OpenMedKit Tauri checks")

    _run([npm, "ci"], cwd=TAURI_PACKAGE_DIR)
    _run([npm, "test"], cwd=TAURI_PACKAGE_DIR)


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
