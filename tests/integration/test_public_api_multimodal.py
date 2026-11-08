"""Offline command-line coverage of nested multimodal compatibility capture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_nested_multimodal_cli_with_standard_library_only(tmp_path):
    script = tmp_path / "scripts/release/check_public_api.py"
    script.parent.mkdir(parents=True)
    script.write_bytes((ROOT / "scripts/release/check_public_api.py").read_bytes())
    sources = {
        "__init__.py": "__all__ = []\n",
        "multimodal/__init__.py": "__all__ = []\n",
        "multimodal/streaming/__init__.py": "__all__ = []\n",
        "multimodal/streaming/speech/__init__.py": (
            "def transcribe(audio):\n    return None\n__all__ = ['transcribe']\n"
        ),
        "multimodal/_private/__init__.py": (
            "raise RuntimeError('private import forbidden')\n__all__ = ['hidden']\n"
        ),
        "multimodal/internal/__init__.py": (
            "raise RuntimeError('internal import forbidden')\n"
        ),
        "multimodal/optional/__init__.py": (
            "import unavailable_api_fixture_provider\n__all__ = ['Extra']\n"
        ),
    }
    for relative, source in sources.items():
        path = tmp_path / "openmed" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text('{"announced_breaks": {}}\n', encoding="utf-8")
    command = [
        sys.executable,
        "-B",
        "-S",  # No site-packages: the checker needs only the standard library.
        str(script),
        "--baseline",
        str(baseline),
        "--allowlist",
        str(allowlist),
    ]
    updated = subprocess.run(
        [*command, "--update"], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert updated.returncode == 0, updated.stderr
    original = baseline.read_bytes()
    modules = json.loads(original)["modules"]
    assert set(modules) == {
        "openmed",
        "openmed.multimodal",
        "openmed.multimodal.streaming",
        "openmed.multimodal.streaming.speech",
    }
    assert modules["openmed.multimodal.streaming.speech"]["transcribe"] == {
        "kind": "function",
        "signature": "(audio)",
    }
    checked = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert checked.returncode == 0, checked.stderr
    subprocess.run([*command, "--update"], cwd=tmp_path, check=True, timeout=30)
    assert baseline.read_bytes() == original

    (tmp_path / "openmed/multimodal/streaming/speech/__init__.py").write_text(
        "__all__ = []\n", encoding="utf-8"
    )
    removed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert removed.returncode == 1
    assert "BREAKING openmed.multimodal.streaming.speech.transcribe" in removed.stderr
    assert "unavailable_api_fixture_provider" not in removed.stdout + removed.stderr
