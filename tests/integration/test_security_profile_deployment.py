"""Synthetic deployment test: the launched service enforces its security profile.

The reference launcher (the Docker, Compose and Helm entrypoint) starts a real
Uvicorn listener on loopback. The test checks startup, ``/readyz`` and the CLI
preflight agree on one report for a valid and an invalid production profile.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

from openmed.service.auth import hash_api_key  # noqa: E402

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_API_KEY = "synthetic-deployment-key-0001"
PHI_SENTINEL = "Synthetic Patient Sentinel-Roe MRN 9931-0007"
_STARTUP_TIMEOUT_SECONDS = 90.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _base_env(**overrides: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OPENMED_")
    }
    env.update(
        {
            "OPENMED_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OPENMED_PROFILE": "prod",
            "OPENMED_SERVICE_SERVED_MODELS": "pii_detection_superclinical",
            "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "2",
        }
    )
    env.update(overrides)
    return env


def _production_env() -> dict[str, str]:
    keys = [
        {
            "id": "deployment-key",
            "key_hash": hash_api_key(SYNTHETIC_API_KEY),
            "principal": "synthetic-deployment",
            "scopes": ["models:read", "pii:read"],
        }
    ]
    return _base_env(
        OPENMED_SERVICE_AUTH_ENABLED="true",
        OPENMED_SERVICE_AUTH_API_KEYS=json.dumps(keys),
    )


def _request(
    url: str, *, method: str = "GET", body: Any = None
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - loopback test listener
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Host": "127.0.0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


@contextmanager
def _service(env: dict[str, str]) -> Iterator[tuple[str, subprocess.Popen[str]]]:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openmed.service.logging",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while True:
            if process.poll() is not None:
                pytest.fail("service exited before it began listening")
            try:
                if _request(f"{base_url}/livez")[0] == 200:
                    break
            except (ConnectionError, OSError):
                pass
            if time.monotonic() > deadline:
                pytest.fail("service did not start listening")
            time.sleep(0.2)
        yield base_url, process
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _preflight(env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from openmed.cli.main import main; sys.exit(main(sys.argv[1:]))",
            "service",
            "preflight",
            "--json",
            "--host",
            "127.0.0.1",
            "--no-tls",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return completed.returncode, json.loads(completed.stdout)["data"]


def test_synthetic_production_deployment_becomes_ready() -> None:
    env = _production_env()

    with _service(env) as (base_url, _):
        readyz = _request(f"{base_url}/readyz")
        unauthenticated = _request(f"{base_url}/models/loaded")

    code, report = _preflight(env)
    assert readyz == (200, {"status": "ready", "service": "openmed-rest"})
    assert unauthenticated[0] == 401
    assert code == 0
    assert report["ready"] is True
    assert report["profile"]["id"] == "openmed.service.security_profile.prod.v1"
    listener = next(
        item for item in report["checks"] if item["id"] == "listener.transport"
    )
    assert listener["reason"] == "loopback_listener"


def test_insecure_production_deployment_stays_closed_with_one_report() -> None:
    env = _base_env()

    with _service(env) as (base_url, process):
        status, readyz = _request(f"{base_url}/readyz")
        refused = _request(
            f"{base_url}/pii/extract", method="POST", body={"text": PHI_SENTINEL}
        )
        process.terminate()
        stdout, stderr = process.communicate(timeout=30)

    code, report = _preflight(env)
    assert status == 503
    assert readyz["error"]["code"] == "security_profile_invalid"
    assert readyz["error"]["details"]["blocking_checks"] == ["auth.enabled"]
    assert refused[0] == 503
    assert refused[1]["error"]["code"] == "security_profile_invalid"
    assert code == 1
    assert report["blocking_checks"] == ["auth.enabled"]
    assert report["digest"] == readyz["error"]["details"]["report_digest"]
    logs = stdout + stderr
    assert "security_profile_invalid" in logs
    assert PHI_SENTINEL not in logs
    assert "Sentinel" not in json.dumps(readyz) + json.dumps(refused[1])
