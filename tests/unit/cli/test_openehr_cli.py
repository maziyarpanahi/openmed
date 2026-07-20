"""Tests for the ``openmed export openehr`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "openehr"
TEMPLATE = FIXTURES / "openmed_grounded_sample_webtemplate.json"
ENTITIES = FIXTURES / "grounded_entities.json"


def test_export_openehr_cli_writes_valid_flat_composition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "composition.json"

    exit_code = main_module.main(
        [
            "export",
            "openehr",
            "--input",
            str(ENTITIES),
            "--template",
            str(TEMPLATE),
            "--output",
            str(output),
            "--vocabulary-key",
            "local-user-vocab",
            "--time",
            "2026-01-01T00:00:00+00:00",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"openEHR COMPOSITION written to: {output}" in captured.out
    assert captured.err == ""

    composition = json.loads(output.read_text(encoding="utf-8"))
    assert composition["ctx/time"] == "2026-01-01T00:00:00+00:00"
    assert any(path.endswith("|code") for path in composition)


def test_export_openehr_cli_reports_invalid_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({"resources": []}), encoding="utf-8")
    output = tmp_path / "composition.json"

    exit_code = main_module.main(
        [
            "export",
            "openehr",
            "--input",
            str(result_file),
            "--template",
            str(TEMPLATE),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "clinical entities" in captured.err
    assert captured.out == ""
    assert not output.exists()
