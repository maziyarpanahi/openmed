"""Real CLI subprocesses using only the synthetic session example."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY = "from openmed.cli import main; raise SystemExit(main())"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.name != "posix", reason="POSIX protected output only"),
]


@pytest.mark.parametrize("task", ["process-document", "transcribe", "draft"])
def test_explicit_synthetic_factory(tmp_path, task):
    media = tmp_path / "media.txt"
    media.write_bytes(b"synthetic media\n")
    model = tmp_path / "artifact"
    model.write_bytes(b"synthetic placeholder, not model weights\n")
    consent = tmp_path / "consent.txt"
    consent.write_bytes(b"synthetic consent\n")
    destination = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ENTRY,
            "multimodal",
            task,
            str(media),
            "--model-artifact",
            str(model),
            "--consent-evidence",
            str(consent),
            "--session-factory",
            "examples.multimodal.offline_cli:create_service",
            "--confirm-review",
            "--output-directory",
            str(destination),
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "OPENMED_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)["data"]
    assert evidence["task"] == task
    assert "Synthetic draft only" not in result.stdout + result.stderr
    assert str(tmp_path) not in result.stdout + result.stderr
    assert (destination / "COMPLETE").exists()
    protected = json.loads((destination / "result.json").read_text())
    assert protected["result"]["text"] == "Synthetic draft only"
    assert protected["requires_clinical_review"] is True


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_process_interruption_never_finalizes(tmp_path, signum):
    # Private synthetic fixture provider signals readiness on stdout suppressed
    # by the adapter; a separate value-free local sentinel synchronizes the test.
    provider = tmp_path / "slow_service.py"
    sentinel = tmp_path / "opened"
    provider.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "from examples.multimodal.offline_cli import SyntheticService, SyntheticSession\n"
        "class SlowSession(SyntheticSession):\n"
        "    def run(self):\n"
        f"        Path({str(sentinel)!r}).touch()\n"
        "        time.sleep(30)\n"
        "        return super().run()\n"
        "class SlowService(SyntheticService):\n"
        "    def open(self, request): return SlowSession(request)\n"
        "def create_service(): return SlowService()\n"
    )
    media = tmp_path / "source"
    media.write_bytes(b"synthetic media\n")
    model = tmp_path / "model"
    model.touch()
    destination = tmp_path / "outputs"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            ENTRY,
            "multimodal",
            "process-document",
            str(media),
            "--model-artifact",
            str(model),
            "--session-factory",
            "slow_service:create_service",
            "--confirm-review",
            "--output-directory",
            str(destination),
            "--json",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(REPO_ROOT), str(tmp_path))),
        },
    )
    try:
        import time

        deadline = time.monotonic() + 10
        while not sentinel.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.01)
        assert sentinel.exists()
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130
        assert json.loads(stdout)["error"]["code"] == "cancelled"
        assert not destination.exists()
        assert str(tmp_path) not in stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
