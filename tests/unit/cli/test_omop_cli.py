"""Tests for the ``openmed omop load`` CLI command."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from openmed.cli import main_module

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "interop" / "omop" / "grounded_notes.jsonl"

# Synthetic note text that must never appear in the PHI-free CLI output.
_NOTE_TEXT_MARKERS = ("Patient reports", "diabetes", "aspirin", "Glucose")


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict:
    exit_code = main_module.main(args)
    output = capsys.readouterr().out
    assert exit_code == 0, output
    return json.loads(output)


def test_omop_load_writes_sqlite_and_reports_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "omop.sqlite"
    payload = _run(
        [
            "omop",
            "load",
            "--input",
            str(FIXTURE),
            "--target",
            str(target),
            "--writer",
            "sqlite",
            "--vocabulary-version",
            "SNOMED-2026",
            "--json",
        ],
        capsys,
    )

    assert payload["ok"] is True
    data = payload["data"]
    assert data["writer"] == "sqlite"
    assert data["vocabulary_version"] == "SNOMED-2026"
    assert data["row_counts"]["note"] == 2
    assert data["row_counts"]["note_nlp"] == 4
    assert data["row_counts"]["condition_occurrence"] == 1
    assert data["row_counts"]["drug_exposure"] == 1
    assert data["rejection_counts"] == {"invalid_offset": 1}
    assert len(data["source_note_hashes"]) == 2

    assert target.exists()
    connection = sqlite3.connect(target)
    try:
        note_rows = connection.execute("SELECT COUNT(*) FROM note").fetchone()[0]
        nlp_rows = connection.execute("SELECT COUNT(*) FROM note_nlp").fetchone()[0]
    finally:
        connection.close()
    assert note_rows == 2
    assert nlp_rows == 4


def test_omop_load_output_is_phi_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main_module.main(
        [
            "omop",
            "load",
            "--input",
            str(FIXTURE),
            "--json",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    for marker in _NOTE_TEXT_MARKERS:
        assert marker not in output


def test_omop_load_is_idempotent_in_append_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "omop.sqlite"
    args = [
        "omop",
        "load",
        "--input",
        str(FIXTURE),
        "--target",
        str(target),
        "--json",
    ]

    first = _run(args, capsys)["data"]["row_counts"]
    second = _run(args, capsys)["data"]["row_counts"]
    assert first == second

    connection = sqlite3.connect(target)
    try:
        note_rows = connection.execute("SELECT COUNT(*) FROM note").fetchone()[0]
    finally:
        connection.close()
    assert note_rows == 2


def test_omop_load_validate_reports_zero_violations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _run(
        [
            "omop",
            "load",
            "--input",
            str(FIXTURE),
            "--validate",
            "--json",
        ],
        capsys,
    )["data"]
    assert data["constraint_violations"] == {"count": 0, "by_reason": {}}


def test_omop_load_dry_run_without_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _run(
        [
            "omop",
            "load",
            "--input",
            str(FIXTURE),
            "--json",
        ],
        capsys,
    )["data"]
    assert data["target"] is None
    assert data["writer"] is None
    assert data["row_counts"]["note"] == 2


def test_omop_load_missing_input_reports_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.jsonl"
    exit_code = main_module.main(
        [
            "omop",
            "load",
            "--input",
            str(missing),
            "--json",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 1
    envelope = json.loads(output)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "input_not_found"
