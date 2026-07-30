"""Tests for the standalone model manifest validator."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.manifest_schema import (
    format_manifest_validation,
    validate_manifest_file,
)
from scripts.manifest import validate_manifest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "models.jsonl"


VALID_ROW = {
    "repo_id": "OpenMed/example-model",
    "family": "NER",
    "task": "token-classification",
    "languages": ["en"],
    "tier": "Small",
    "param_count": 44_000_000,
    "architecture": "bert",
    "base_model": "OpenMed/base-model",
    "formats": ["pytorch"],
    "canonical_labels": ["DISEASE"],
    "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
    "arxiv": None,
    "license": "apache-2.0",
    "reproducibility_hash": f"sha256:{'a' * 64}",
    "released": "2026-06-24",
}


def test_committed_manifest_validates() -> None:
    result = validate_manifest_file(MANIFEST_PATH)

    assert result.ok, "\n".join(format_manifest_validation(result))
    assert result.row_count > 0


def test_broken_manifest_reports_all_line_numbered_violations(tmp_path: Path) -> None:
    manifest = _write_broken_manifest(tmp_path)

    result = validate_manifest_file(manifest)

    assert result.ok is False
    assert [str(violation) for violation in result.violations] == [
        "line 1: missing required key: repo_id",
        "line 2: license must be one of: apache-2.0, other, or null",
        "line 3: reproducibility_hash must match sha256:<64 lowercase hex characters>",
    ]


def test_manifest_validator_requires_reproducibility_hash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = _row()
    del row["reproducibility_hash"]
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    exit_code = validate_manifest.main(["--manifest", str(manifest)])

    assert exit_code == 1
    assert "missing required key: reproducibility_hash" in capsys.readouterr().err


def test_module_validator_exits_zero_for_committed_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = validate_manifest.main(["--manifest", str(MANIFEST_PATH)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OK" in captured.out
    assert captured.err == ""


def test_pii_entry_without_script_coverage_fails_validation(tmp_path: Path) -> None:
    row = _row(family="PII", canonical_labels=["PERSON"])
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = validate_manifest_file(manifest)

    assert [str(violation) for violation in result.violations] == [
        "line 1: PII entry missing required key: script_coverage"
    ]


def test_openmed_models_validate_shares_validator_verdict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_broken_manifest(tmp_path)

    script_exit = validate_manifest.main(["--manifest", str(manifest)])
    script_output = capsys.readouterr()
    cli_exit = main_module.main(["models", "validate", "--manifest", str(manifest)])
    cli_output = capsys.readouterr()

    assert script_exit == cli_exit == 1
    assert script_output.out == cli_output.out == ""
    assert script_output.err == cli_output.err


def _write_broken_manifest(tmp_path: Path) -> Path:
    rows = []

    missing_repo_id = _row()
    del missing_repo_id["repo_id"]
    rows.append(missing_repo_id)

    bad_license = _row(license="gpl-3.0")
    rows.append(bad_license)

    bad_hash = _row(reproducibility_hash="sha256:not-a-real-hash")
    rows.append(bad_hash)

    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _row(**overrides: object) -> dict[str, object]:
    row = deepcopy(VALID_ROW)
    row.update(overrides)
    return row
