"""Process-boundary checks for openmed-synth surrogate-provider drift."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.training.synthetic.provider_versions import (
    PROVIDER_VERSION_DRIFT,
    faker_version,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]
_PATCHED_SUFFIX = "Synthetique"
_RUNNER = f"""
import socket
import sys

def forbidden(*args, **kwargs):
    raise AssertionError("network use forbidden")

socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden

if sys.argv[1] == "patch-fr-city":
    from faker.providers.address import fr_FR

    fr_FR.Provider.city_formats = ("{{{{last_name}}}}-{_PATCHED_SUFFIX}",)

from openmed.cli.main import main

sys.exit(main(sys.argv[2:]))
"""


def _locked_faker_version() -> str:
    lock = (_ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(r'name = "faker"\nversion = "([^"]+)"', lock)
    assert match is not None
    return match.group(1)


def _run(mode: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OPENMED_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(_ROOT), os.environ.get("PYTHONPATH")))
        ),
    }
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            mode,
            "benchmark",
            "pii",
            "--suite",
            "openmed-synth",
            *args,
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_locked_environment_passes_strict_mode_without_warning() -> None:
    if faker_version() != _locked_faker_version():
        pytest.skip("requires the Faker version pinned in uv.lock")

    result = _run("locked", "--strict")

    assert result.returncode == 0
    assert PROVIDER_VERSION_DRIFT not in result.stderr
    drift = json.loads(result.stdout)["metadata"]["provider_drift"]
    assert drift["status"] == "match"
    assert drift["reasons"] == []


def test_changed_provider_table_warns_by_default() -> None:
    result = _run("patch-fr-city")

    assert result.returncode == 0
    assert f"Warning: {PROVIDER_VERSION_DRIFT}" in result.stderr
    assert "faker[fr_FR]:probe_digest" in result.stderr
    drift = json.loads(result.stdout)["metadata"]["provider_drift"]
    assert drift["reasons"] == [PROVIDER_VERSION_DRIFT]
    assert drift["content_hash_match"] is False
    findings = json.dumps(drift["findings"])
    assert _PATCHED_SUFFIX not in findings
    assert _PATCHED_SUFFIX not in result.stderr


def test_changed_provider_table_fails_strict_mode() -> None:
    result = _run("patch-fr-city", "--strict", "--json")

    assert result.returncode == 1
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == PROVIDER_VERSION_DRIFT
    assert "corpus[fr_FR]:output_digest" in envelope["error"]["message"]
    assert _PATCHED_SUFFIX not in result.stdout
    assert _PATCHED_SUFFIX not in result.stderr
