"""Integration coverage for the hardened distroless service image."""

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

BASE_DIR = Path(__file__).resolve().parents[2]
DOCKERFILE = BASE_DIR / "deploy" / "docker" / "Dockerfile.distroless"
RUN_DISTROLESS_IMAGE_TEST_ENV = "OPENMED_RUN_DISTROLESS_IMAGE_TEST"


def _dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _final_stage(text: str) -> str:
    marker = "\nFROM "
    _, _, final = text.rpartition(marker)
    assert final
    return f"FROM {final}"


def test_distroless_dockerfile_uses_nonroot_runtime() -> None:
    text = _dockerfile_text()
    final_stage = _final_stage(text)

    assert "FROM gcr.io/distroless/python3-debian12:nonroot" in final_stage
    assert "USER 65532:65532" in final_stage
    assert 'VOLUME ["/cache"]' in final_stage
    assert "OPENMED_CACHE_DIR=/cache/openmed" in final_stage
    assert "HF_HOME=/cache/huggingface" in final_stage


def test_distroless_dockerfile_keeps_runtime_surface_small() -> None:
    final_stage = _final_stage(_dockerfile_text())

    assert "/usr/bin/tini" in final_stage
    assert 'ENTRYPOINT ["/usr/bin/tini", "--", "/usr/bin/python3"' in final_stage
    assert "apt-get" not in final_stage
    assert "pip install" not in final_stage
    assert "/bin/sh" not in final_stage


def test_distroless_healthcheck_uses_readiness_probe() -> None:
    final_stage = _final_stage(_dockerfile_text())

    assert "HEALTHCHECK" in final_stage
    assert "127.0.0.1:8080/readyz" in final_stage
    assert "127.0.0.1:8080/health" not in final_stage


def _run(
    args: list[str],
    *,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        command = " ".join(args)
        raise AssertionError(
            f"Command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _require_docker() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed")
    try:
        _run(["docker", "info"], timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Docker daemon is not available: {exc}")


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_json(url: str, *, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_json(url: str, *, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _get_json(url)
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise AssertionError(f"{url} did not become ready: {last_error!r}")


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.mark.integration
@pytest.mark.slow
def test_distroless_image_runs_with_hardened_container_flags() -> None:
    if os.environ.get(RUN_DISTROLESS_IMAGE_TEST_ENV) != "1":
        pytest.skip(f"set {RUN_DISTROLESS_IMAGE_TEST_ENV}=1 to build and run the image")
    if os.environ.get("OPENMED_SKIP_HF_TESTS") == "1":
        pytest.skip("Skipping Hugging Face dependent smoke test")

    _require_docker()
    suffix = uuid4().hex[:12]
    image = f"openmed:distroless-smoke-{suffix}"
    volume = f"openmed-distroless-cache-{suffix}"
    container = f"openmed-distroless-smoke-{suffix}"
    host_port = _unused_port()

    try:
        _run(
            [
                "docker",
                "build",
                "-f",
                str(DOCKERFILE),
                "-t",
                image,
                ".",
            ],
            timeout=1800,
        )
        _run(["docker", "volume", "create", volume], timeout=30)
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=64m",
                "--mount",
                f"type=volume,source={volume},target=/cache",
                "--entrypoint",
                "/usr/bin/python3",
                image,
                "-c",
                (
                    "import os, pathlib; "
                    "assert os.getuid() == 65532; "
                    "pathlib.Path('/cache/probe').write_text('ok'); "
                    "\ntry:\n"
                    "    pathlib.Path('/app/rootfs-probe').write_text('bad')\n"
                    "except OSError:\n"
                    "    pass\n"
                    "else:\n"
                    "    raise SystemExit('root filesystem is writable')\n"
                ),
            ],
            timeout=120,
        )
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=128m",
                "--mount",
                f"type=volume,source={volume},target=/cache",
                "-p",
                f"127.0.0.1:{host_port}:8080",
                "-e",
                "OPENMED_PROFILE=test",
                "-e",
                "OPENMED_SERVICE_KEEP_ALIVE=0",
                "-e",
                "OPENMED_SERVICE_MAX_RESIDENT_MODELS=1",
                image,
            ],
            timeout=120,
        )

        livez = _wait_for_json(
            f"http://127.0.0.1:{host_port}/livez",
            timeout_seconds=120,
        )
        readyz = _wait_for_json(
            f"http://127.0.0.1:{host_port}/readyz",
            timeout_seconds=120,
        )
        redacted = _post_json(
            f"http://127.0.0.1:{host_port}/pii/deidentify",
            {
                "text": "Patient Jane Doe emailed jane.doe@example.com.",
                "method": "mask",
                "keep_alive": 0,
            },
            timeout=300,
        )

        assert livez == {"status": "ok", "service": "openmed-rest"}
        assert readyz == {"status": "ready", "service": "openmed-rest"}
        assert "Jane Doe" not in redacted["deidentified_text"]
        assert "jane.doe@example.com" not in redacted["deidentified_text"]
    finally:
        _run(["docker", "rm", "-f", container], timeout=30, check=False)
        _run(["docker", "volume", "rm", "-f", volume], timeout=30, check=False)
        _run(["docker", "image", "rm", "-f", image], timeout=60, check=False)
