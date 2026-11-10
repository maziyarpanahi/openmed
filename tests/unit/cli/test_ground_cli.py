"""Focused tests for grounding snapshot and CLI output parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.clinical.grounding import VocabLoader

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"


def test_grounding_import_and_ground_cli_emit_the_shared_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_dir = tmp_path / "grounding"
    exit_code = main_module.main(
        [
            "grounding",
            "import",
            "--system",
            "icd10cm",
            "--input",
            str(FIXTURE),
            "--version",
            "synthetic-fixture-1",
            "--cache-dir",
            str(cache_dir),
            "--json",
        ]
    )
    imported = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert imported["ok"] is True
    assert imported["data"]["version"] == "synthetic-fixture-1"

    exit_code = main_module.main(
        [
            "ground",
            "--text",
            "type 2 diabetes",
            "--system",
            "icd10cm",
            "--cache-dir",
            str(cache_dir),
            "--json",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0, output
    payload = json.loads(output)
    assert payload["ok"] is True
    assert payload["data"]["schema_version"] == "openmed.grounding.v1"
    assert payload["data"]["results"][0]["code"] == "E11.9"
    assert payload["data"]["results"][0]["confidence"] == 1.0


def test_ground_cli_rejects_restricted_system_without_endpoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main_module.main(
        [
            "ground",
            "--text",
            "synthetic finding",
            "--system",
            "snomed",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["error"]["code"] == "restricted_terminology_unconfigured"
    assert "synthetic finding" not in json.dumps(payload)
