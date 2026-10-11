"""Exercise clinical CLI process boundaries with synthetic local inputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_RUNNER = """
import socket
import sys
from openmed.cli.main import main
from openmed.core.models import ModelLoader

def forbidden(*args, **kwargs):
    raise AssertionError('model or network use forbidden')

socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
ModelLoader.load_model = forbidden
sys.exit(main(sys.argv[1:]))
"""


@pytest.mark.parametrize("name", ["sdoh", "relations", "timeline"])
def test_offline_process_returns_one_value_free_envelope(tmp_path, name):
    text = (
        "Synthetic Example takes Metformin 500 mg. Smokes daily. Fever began yesterday."
    )
    entities = []
    for surface, label in (
        ("Metformin", "MEDICATION"),
        ("500 mg", "DOSAGE"),
        ("Fever", "SYMPTOM"),
        ("yesterday", "DATE"),
    ):
        start = text.index(surface)
        entities.append(
            {
                "text": surface,
                "label": label,
                "start": start,
                "end": start + len(surface),
                "confidence": 0.9,
            }
        )
    note = tmp_path / "PRIVATE_NOTE.txt"
    spans = tmp_path / "PRIVATE_SPANS.json"
    output = tmp_path / "PRIVATE_RESULT.json"
    note.write_text(text)
    spans.write_text(
        json.dumps(
            {
                "ok": True,
                "command": "analyze",
                "data": {"text": text, "entities": entities},
            }
        )
    )
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OPENMED_OFFLINE": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            "clinical",
            name,
            "--note",
            str(note),
            "--spans",
            str(spans),
            "--output",
            str(output),
            "--json",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0 and result.stderr == ""
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is True and envelope["command"] == f"clinical {name}"
    assert envelope["data"]["review_required"]
    saved = json.loads(output.read_text())
    assert saved["data"]["source_digest"] == envelope["data"]["source_digest"]
    for protected in (
        text,
        "Synthetic Example",
        "Metformin",
        "500 mg",
        "Fever",
        "yesterday",
        str(tmp_path),
    ):
        assert protected not in result.stdout + result.stderr + output.read_text()


def test_help_and_invalid_arguments_are_private_across_process(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    help_result = subprocess.run(
        [sys.executable, "-c", _RUNNER, "clinical", "timeline", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--reference-time" in help_result.stdout
    failure = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            "clinical",
            "PRIVATE_COMMAND",
            "--note",
            str(tmp_path / "PRIVATE_NOTE"),
            "--json",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert failure.returncode == 2 and failure.stderr == ""
    assert json.loads(failure.stdout)["error"]["code"] == "clinical_arguments_invalid"
    assert "PRIVATE" not in failure.stdout
