"""Synthetic subprocess controls for the guarded clinical review CLI."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SOURCE = "Patient Casey Example presented with a cough. The synthetic admission was uncomplicated."
CANARY = "SYNTHETIC_PRIVATE_CANARY"
BOOTSTRAP = r"""
import ctypes
import importlib
import logging
import os
import socket
import sys
from datetime import datetime
from types import SimpleNamespace

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.core.models import ModelLoader
from openmed.clinical.summarize_backends import ExtractiveSummarizerBackend
from openmed.clinical.nli_backends import LocalNLIError

summary = importlib.import_module("openmed.clinical.summarize")
kind = os.environ["OPENMED_TEST_CLINICAL_CASE"]
canary = "SYNTHETIC_PRIVATE_CANARY"

def forbidden(*args, **kwargs):
    raise AssertionError(canary)

socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
ModelLoader.load_model = forbidden
ModelLoader.load_local_sequence_classifier = forbidden

try:
    import huggingface_hub
    huggingface_hub.hf_hub_download = forbidden
    huggingface_hub.snapshot_download = forbidden
except ImportError:
    pass

def deidentify(text, **kwargs):
    assert kwargs["config"].local_only is True
    print(canary)
    os.write(2, canary.encode())
    return DeidentificationResult(
        original_text=text,
        deidentified_text=text.replace("Casey Example", "[NAME]"),
        pii_entities=[PIIEntity(
            text="Casey Example", label="NAME", start=8, end=21,
            confidence=0.99, redacted_text="[NAME]"
        )],
        method="mask", timestamp=datetime(2026, 1, 1)
    )
summary.deidentify = deidentify

if kind == "leak":
    ExtractiveSummarizerBackend.summarize = lambda *a, **k: "Casey Example has a cough."
if kind == "failure":
    def failed(*args, **kwargs):
        print(canary)
        os.write(1, canary.encode())
        raise RuntimeError(canary)
    summary.summarize = failed
if kind == "missing":
    import openmed.clinical.nli_backends as nli_backends
    nli_backends.get_default_nli_model = lambda: None

class Backend:
    backend_id = "private-caller-name"
    def predict(self, premise, hypothesis):
        print(canary)
        print(canary, file=sys.stderr)
        os.write(1, canary.encode())
        os.write(2, canary.encode())
        logging.error(canary)
        ctypes.CDLL(None).printf(canary.encode())
        return {"label": kind, "score": 0.8}

sys.modules["synthetic_cli_backend"] = SimpleNamespace(create=Backend)
from openmed.cli import main
raise SystemExit(main())
"""


@pytest.fixture
def files(tmp_path):
    source = tmp_path / "synthetic-note.txt"
    source.write_text(SOURCE, encoding="utf-8")
    claims = tmp_path / "synthetic-claims.json"
    claims.write_text(json.dumps(["Casey Example presented with a cough."]))
    return source, claims, tmp_path / "summary.txt", tmp_path / "metadata.json"


def run_cli(argv, *, kind="summary", cwd):
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                [
                    str(Path(__file__).resolve().parents[2]),
                    os.environ.get("PYTHONPATH"),
                ],
            )
        ),
        "OPENMED_TEST_CLINICAL_CASE": kind,
        "OPENMED_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP, *argv],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert CANARY not in result.stdout + result.stderr
    assert "Casey Example" not in result.stdout + result.stderr
    assert SOURCE not in result.stdout + result.stderr
    assert result.stderr == ""
    return result, json.loads(result.stdout)


def summary_argv(files):
    return [
        "summarize",
        str(files[0]),
        "--model",
        "extractive",
        "--summary-output",
        str(files[2]),
        "--metadata-output",
        str(files[3]),
        "--json",
    ]


def nli_argv(files, *extra):
    return [
        "nli",
        "verify",
        "--source",
        str(files[0]),
        "--claims",
        str(files[1]),
        *extra,
        "--json",
    ]


def test_summary_subprocess_writes_guarded_text_and_only_value_free_metadata(files):
    result, envelope = run_cli(summary_argv(files), cwd=files[0].parent)
    assert result.returncode == 0 and envelope["ok"] is True
    assert envelope["command"] == "summarize"
    assert files[2].read_text() == (
        "Patient [NAME] presented with a cough. The synthetic admission was uncomplicated."
    )
    assert envelope == json.loads(files[3].read_text())
    assert envelope["data"]["leakage_check"]["passed"] is True
    assert envelope["data"]["human_review_required"] is True
    for path in files[2:]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert files[2].read_text() not in result.stdout + files[3].read_text()


@pytest.mark.parametrize(
    "kind,code", [("leak", "summary_leakage_rejected"), ("failure", "summary_failed")]
)
def test_summary_subprocess_refuses_leakage_or_exception_without_partial_files(
    kind, code, files
):
    result, envelope = run_cli(summary_argv(files), kind=kind, cwd=files[0].parent)
    assert result.returncode == 1 and envelope["ok"] is False
    assert envelope["error"]["code"] == code
    assert not any(path.exists() for path in files[2:])


def test_summary_subprocess_preserves_existing_destination(files):
    files[3].write_text(CANARY)
    result, envelope = run_cli(summary_argv(files), cwd=files[0].parent)
    assert result.returncode == 1
    assert envelope["error"]["code"] == "summary_output_failed"
    assert files[3].read_text() == CANARY
    assert not files[2].exists()


@pytest.mark.parametrize(
    "label", ["entailment", "contradiction", "neutral", "abstention"]
)
def test_local_factory_subprocess_returns_only_native_claim_projection(label, files):
    result, envelope = run_cli(
        nli_argv(files, "--backend-factory", "synthetic_cli_backend:create"),
        kind=label,
        cwd=files[0].parent,
    )
    assert result.returncode == int(label != "entailment")
    assert envelope == {
        "ok": True,
        "command": "nli verify",
        "data": {
            "claims": [
                {
                    "claim_index": 0,
                    "label": label,
                    "score": 0.8,
                    "backend_id": "caller-supplied-local",
                    "contradicted": label == "contradiction",
                    "review_required": label == "abstention",
                }
            ]
        },
    }
    assert "private-caller-name" not in result.stdout
    assert set(files[0].parent.iterdir()) == set(files[:2])


def test_explicit_heuristic_subprocess_matches_existing_api(files):
    from openmed.clinical.nli import verify

    result, envelope = run_cli(
        nli_argv(files, "--backend", "heuristic"),
        cwd=files[0].parent,
    )
    expected = verify(json.loads(files[1].read_text()), SOURCE, backend="heuristic")
    assert envelope["data"]["claims"] == expected
    assert result.returncode == int(
        any(row["label"] != "entailment" for row in expected)
    )


def test_missing_local_checkpoint_subprocess_has_no_heuristic_fallback(files):
    result, envelope = run_cli(nli_argv(files), kind="missing", cwd=files[0].parent)
    assert result.returncode == 1 and envelope["ok"] is False
    assert envelope["error"]["code"] == "nli_backend_unavailable"


@pytest.mark.parametrize("command", ["summarize", "nli"])
def test_protected_usage_subprocess_never_echoes_private_arguments(command, files):
    result, envelope = run_cli([command, CANARY, "--json"], cwd=files[0].parent)
    assert result.returncode == 2 and envelope["ok"] is False
    assert envelope["error"]["code"] == (
        "summary_arguments_invalid"
        if command == "summarize"
        else "nli_arguments_invalid"
    )
