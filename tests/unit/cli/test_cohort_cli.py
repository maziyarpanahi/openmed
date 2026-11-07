"""Focused tests for ``openmed cohort resolve``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.interop.omop import (
    deterministic_omop_id,
    load_grounded_jsonl,
    write_omop_duckdb,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cohort"


def test_cohort_resolve_cli_returns_only_safe_patient_and_evidence_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "cohort.duckdb"
    connection = write_omop_duckdb(
        load_grounded_jsonl(FIXTURES / "synthetic_grounded.jsonl"),
        database,
    )
    connection.close()

    exit_code = main_module.main(
        [
            "cohort",
            "resolve",
            "--definition",
            str(FIXTURES / "phenotypes" / "diabetes_on_metformin.json"),
            "--duckdb",
            str(database),
            "--athena",
            str(FIXTURES / "athena"),
            "--json",
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["command"] == "cohort resolve"
    assert payload["data"]["patient_ids"] == sorted(
        [
            deterministic_omop_id("person", "raw-person-alpha"),
            deterministic_omop_id("person", "raw-person-epsilon"),
        ]
    )
    for raw_marker in (
        "raw-person-",
        "synthetic-alpha",
        "Synthetic diabetes marker",
    ):
        assert raw_marker not in output


def test_cohort_resolve_cli_requires_hierarchy_for_descendants(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "cohort.duckdb"
    connection = write_omop_duckdb(
        load_grounded_jsonl(FIXTURES / "synthetic_grounded.jsonl"),
        database,
    )
    connection.close()

    exit_code = main_module.main(
        [
            "cohort",
            "resolve",
            "--definition",
            str(FIXTURES / "phenotypes" / "diabetes_on_metformin.json"),
            "--duckdb",
            str(database),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "resolution_failed"
    assert "requires an Athena hierarchy" in payload["error"]["message"]
