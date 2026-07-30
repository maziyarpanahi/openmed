"""Opt-in network-disabled installation test for a real ARM64 air-gap kit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from openmed.cli.airgap import DEFAULT_BUNDLE_LIMIT_BYTES, MANIFEST_NAME

_BUNDLE_ENV = "OPENMED_AIRGAP_TEST_BUNDLE"
_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"


@pytest.mark.integration
def test_arm64_cp311_bundle_installs_and_deidentifies_without_network() -> None:
    """Install a prebuilt real bundle in an ARM64 container with no network."""
    configured_bundle = os.environ.get(_BUNDLE_ENV)
    if not configured_bundle:
        pytest.skip(f"set {_BUNDLE_ENV} to a real ARM64/cp311 bundle")
    bundle = Path(configured_bundle).expanduser().resolve(strict=True)
    docker = shutil.which("docker")
    if docker is None:
        pytest.fail("docker is required when OPENMED_AIRGAP_TEST_BUNDLE is set")

    manifest = json.loads((bundle / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["target"] == {
        "abi": "cp311",
        "implementation": "cp",
        "platform": "manylinux2014_aarch64",
        "python_version": "3.11",
    }
    assert manifest["total_size_bytes"] <= DEFAULT_BUNDLE_LIMIT_BYTES

    script = f"""\
/bundle/install.sh
export OPENMED_OFFLINE=1
openmed doctor
printf '%s\\n' 'Patient Jane Example called 555-0100 on 2025-01-20.' | \\
  openmed deid --model {_MODEL}
"""
    completed = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--platform",
            "linux/arm64",
            "--mount",
            f"type=bind,src={bundle},dst=/bundle,readonly",
            "python:3.11-slim",
            "sh",
            "-ceu",
            script,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert "PASS python_arch: aarch64" in completed.stdout
    assert "PASS openmed_offline: 1" in completed.stdout
    assert "Patient [first_name] Example called 555-0100 on [date]." in (
        completed.stdout
    )
