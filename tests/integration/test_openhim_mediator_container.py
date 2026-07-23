"""Container smoke coverage for the OpenHIM mediator example."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = BASE_DIR / "examples" / "openhim-mediator"
COMPOSE_FILE = EXAMPLE_DIR / "docker-compose.yml"
DOCKERFILE = EXAMPLE_DIR / "Dockerfile"
RUN_CONTAINER_TEST_ENV = "OPENMED_RUN_OPENHIM_CONTAINER_TEST"


def test_openhim_container_example_mounts_registration_config() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    mediator = compose["services"]["openmed-mediator"]

    assert mediator["build"]["dockerfile"] == "examples/openhim-mediator/Dockerfile"
    assert (
        "./mediator-config.json:/etc/openmed/mediator-config.json:ro"
        in mediator["volumes"]
    )
    assert "openhim-fixture" in compose["services"]
    assert compose["services"]["openhim-fixture"]["profiles"] == ["fixture"]


def test_openhim_dockerfile_probes_local_heartbeat() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "OPENMED_OPENHIM_MEDIATOR_ENABLED=true" in dockerfile
    assert "OPENMED_OPENHIM_CONFIG_PATH=/etc/openmed/mediator-config.json" in dockerfile
    assert "127.0.0.1:8080/openhim/heartbeat" in dockerfile


def _run(
    args: list[str],
    *,
    env: dict[str, str],
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=EXAMPLE_DIR,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_json(url: str, *, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(1)
    raise AssertionError(f"{url} did not become ready: {last_error!r}")


@pytest.mark.integration
@pytest.mark.slow
def test_compose_example_registers_and_serves_heartbeat() -> None:
    if os.environ.get(RUN_CONTAINER_TEST_ENV) != "1":
        pytest.skip(f"set {RUN_CONTAINER_TEST_ENV}=1 to build and run the example")
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed")

    project = f"openmed-openhim-{uuid4().hex[:10]}"
    mediator_port = _unused_port()
    fixture_port = _unused_port()
    env = {
        **os.environ,
        "OPENMED_OPENHIM_HOST_PORT": str(mediator_port),
        "OPENMED_OPENHIM_FIXTURE_HOST_PORT": str(fixture_port),
        "OPENMED_OPENHIM_CORE_URL": "http://openhim-fixture:8081",
        "OPENMED_OPENHIM_USERNAME": "openhim@example.org",
        "OPENMED_OPENHIM_PASSWORD": "synthetic-password",
        "OPENMED_OPENHIM_ALLOW_INSECURE_HTTP": "true",
    }
    compose = ["docker", "compose", "-p", project, "-f", str(COMPOSE_FILE)]

    try:
        _run(
            [*compose, "--profile", "fixture", "up", "-d", "openhim-fixture"],
            env=env,
            timeout=180,
        )
        _wait_for_json(f"http://127.0.0.1:{fixture_port}/health", timeout_seconds=60)
        _run(
            [*compose, "up", "-d", "--build", "openmed-mediator"],
            env=env,
            timeout=1200,
        )
        mediator = _wait_for_json(
            f"http://127.0.0.1:{mediator_port}/openhim/heartbeat",
            timeout_seconds=120,
        )
        fixture = _wait_for_json(
            f"http://127.0.0.1:{fixture_port}/health", timeout_seconds=30
        )

        assert mediator["registered"] is True
        assert mediator["last_heartbeat_at"] is not None
        assert fixture["registration_count"] == 1
        assert fixture["heartbeat_count"] >= 1
    finally:
        _run(
            [*compose, "--profile", "fixture", "down", "--volumes", "--remove-orphans"],
            env=env,
            timeout=180,
            check=False,
        )
