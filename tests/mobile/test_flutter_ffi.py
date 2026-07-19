"""Pytest bridge for the OpenMedKit Flutter FFI package checks."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "js" / "openmedkit-flutter"


def test_openmedkit_flutter_ffi_package(tmp_path: Path) -> None:
    flutter = shutil.which("flutter")
    cmake = shutil.which("cmake")
    if flutter is None or cmake is None:
        pytest.skip("Flutter and CMake are required for OpenMedKit Flutter checks")

    build_dir = tmp_path / "native"
    configure = [
        cmake,
        "-S",
        str(PACKAGE_DIR / "native"),
        "-B",
        str(build_dir),
        "-DOPENMED_FFI_ENABLE_TEST_SESSION=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if platform.system() == "Darwin":
        configure.append("-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")
    _run(configure, cwd=ROOT)
    _run([cmake, "--build", str(build_dir), "--config", "Release"], cwd=ROOT)

    library = _find_native_library(build_dir)
    environment = os.environ.copy()
    environment["OPENMED_FFI_LIBRARY"] = str(library)
    environment["OPENMED_FFI_TEST_SESSION"] = "1"
    _run([flutter, "pub", "get"], cwd=PACKAGE_DIR, env=environment)
    _run(
        [flutter, "test", "test/openmedkit_ffi_test.dart"],
        cwd=PACKAGE_DIR,
        env=environment,
    )


def _find_native_library(build_dir: Path) -> Path:
    candidates = [
        *build_dir.rglob("libopenmed_ffi.dylib"),
        *build_dir.rglob("libopenmed_ffi.so"),
        *build_dir.rglob("openmed_ffi.dll"),
    ]
    assert len(candidates) == 1, f"expected one native FFI library, got {candidates}"
    return candidates[0]


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stdout
