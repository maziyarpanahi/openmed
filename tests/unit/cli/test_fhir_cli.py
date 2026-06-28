"""Tests for the ``openmed fhir`` CLI commands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.cli import main_module
from openmed.clinical.exporters.fhir import to_bundle

ROOT = Path(__file__).resolve().parents[3]


def _fixture_resources() -> list[dict[str, Any]]:
    return [
        {"resourceType": "Patient", "id": "patient-1"},
        {
            "resourceType": "Observation",
            "id": "obs-1",
            "status": "final",
            "subject": {"reference": "Patient/patient-1"},
            "code": {"text": "Synthetic glucose"},
        },
        {
            "resourceType": "DiagnosticReport",
            "id": "report-1",
            "status": "final",
            "result": [{"reference": "Observation/obs-1"}],
        },
    ]


def test_fhir_bundle_cli_writes_byte_identical_transaction_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    resources = _fixture_resources()
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps({"doc_id": "fixture-doc", "resources": resources}),
        encoding="utf-8",
    )
    first_output = tmp_path / "bundle-first.json"
    second_output = tmp_path / "bundle-second.json"

    for output in (first_output, second_output):
        exit_code = main_module.main(
            [
                "fhir",
                "bundle",
                "--input",
                str(result_file),
                "--type",
                "transaction",
                "--output",
                str(output),
            ]
        )
        captured = capsys.readouterr()

        assert exit_code == 0
        assert f"FHIR Bundle written to: {output}" in captured.out
        assert captured.err == ""

    assert first_output.read_bytes() == second_output.read_bytes()

    serialized = first_output.read_text(encoding="utf-8")
    emitted = json.loads(serialized)
    assert serialized == json.dumps(emitted, indent=2, sort_keys=True) + "\n"
    assert emitted == to_bundle(
        resources,
        doc_id="fixture-doc",
        bundle_type="transaction",
    )

    full_urls = {entry["fullUrl"] for entry in emitted["entry"]}
    report = emitted["entry"][2]["resource"]
    assert report["result"][0]["reference"] in full_urls


def test_fhir_bundle_cli_invalid_type_exits_nonzero(tmp_path: Path) -> None:
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps({"resources": _fixture_resources()}),
        encoding="utf-8",
    )
    output = tmp_path / "bundle.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openmed.cli.main",
            "fhir",
            "bundle",
            "--input",
            str(result_file),
            "--type",
            "collection",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice: 'collection'" in result.stderr
    assert "transaction" in result.stderr
    assert "batch" in result.stderr
    assert not output.exists()


def test_fhir_bundle_cli_reports_result_without_resources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({"entities": []}), encoding="utf-8")
    output = tmp_path / "bundle.json"

    exit_code = main_module.main(
        [
            "fhir",
            "bundle",
            "--input",
            str(result_file),
            "--type",
            "batch",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "standalone FHIR resources" in captured.err
    assert captured.out == ""
    assert not output.exists()
