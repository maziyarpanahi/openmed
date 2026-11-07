"""CLI acceptance tests for ``openmed profile quality``."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "quality" / "profiler_batch.jsonl"
ATHENA = ROOT / "fixtures" / "quality" / "athena"


def test_quality_profile_cli_emits_json_report(capsys) -> None:
    exit_code = main(
        [
            "profile",
            "quality",
            "--input",
            str(FIXTURE),
            "--athena",
            str(ATHENA),
            "--json",
        ]
    )

    assert exit_code == 1
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is True
    assert body["data"]["grounding"]["grounded_spans"] == 5
    assert body["data"]["status"] == "fail"
    assert "diabetes" not in json.dumps(body)


def test_quality_profile_cli_returns_gate_failure_for_low_completeness(
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "incomplete.jsonl"
    input_path.write_text(
        '{"required_fields":["condition"],"condition":null}\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "profile",
            "--input",
            str(input_path),
            "--completeness-floor",
            "0.5",
            "--json",
        ]
    )

    assert exit_code == 1
    body = json.loads(capsys.readouterr().out)
    assert body["data"]["gate"]["passed"] is False
    assert body["data"]["status"] == "fail"
