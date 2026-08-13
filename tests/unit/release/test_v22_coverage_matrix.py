"""Release-contract tests for the generated V2.2 standards matrix."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from openmed.eval.v22_conformance import (
    V22_FOCUSED_TEST_COMMAND,
    V22CoverageMatrixError,
    load_coverage_matrix,
    render_coverage_matrix_markdown,
    validate_coverage_matrix,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "v22"
MATRIX_PATH = FIXTURE_DIR / "coverage_matrix.json"
DOC_PATH = ROOT / "docs" / "release" / "v2.2-standards-matrix.md"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "tests.yml"


def test_coverage_matrix_validates_declared_artifacts() -> None:
    matrix = validate_coverage_matrix(ROOT, MATRIX_PATH)
    declared_fixtures = {
        path for entry in matrix["entries"] for path in entry["fixture_sha256"]
    }
    committed_fixtures = {
        path.relative_to(ROOT).as_posix()
        for path in FIXTURE_DIR.iterdir()
        if path.is_file() and path.name != MATRIX_PATH.name
    }

    assert matrix["test_command"] == V22_FOCUSED_TEST_COMMAND
    assert declared_fixtures == committed_fixtures
    assert all(entry["supported_subset"] for entry in matrix["entries"])
    assert all(entry["known_gaps"] for entry in matrix["entries"])


def test_coverage_matrix_rejects_missing_contract_parts() -> None:
    original = load_coverage_matrix(MATRIX_PATH)

    missing_version = copy.deepcopy(original)
    missing_version["entries"][0]["version"] = ""
    with pytest.raises(V22CoverageMatrixError, match="version"):
        validate_coverage_matrix(ROOT, missing_version)

    missing_fixture = copy.deepcopy(original)
    fixtures = missing_fixture["entries"][0]["fixture_sha256"]
    digest = next(iter(fixtures.values()))
    missing_fixture["entries"][0]["fixture_sha256"] = {
        "tests/fixtures/v22/not-present.json": digest
    }
    with pytest.raises(V22CoverageMatrixError, match="fixture is missing"):
        validate_coverage_matrix(ROOT, missing_fixture)

    missing_test = copy.deepcopy(original)
    missing_test["entries"][0]["test_nodes"] = [
        "tests/integration/test_v22_exchange_conformance.py::not_a_test"
    ]
    with pytest.raises(V22CoverageMatrixError, match="test node is missing"):
        validate_coverage_matrix(ROOT, missing_test)

    missing_module = copy.deepcopy(original)
    missing_module["entries"][0]["source_modules"] = ["openmed.eval.not_a_real_module"]
    with pytest.raises(V22CoverageMatrixError, match="source module is missing"):
        validate_coverage_matrix(ROOT, missing_module)


def test_standards_document_is_generated_from_matrix() -> None:
    matrix = validate_coverage_matrix(ROOT, MATRIX_PATH)
    expected = render_coverage_matrix_markdown(matrix)
    actual = DOC_PATH.read_text(encoding="utf-8")

    assert actual == expected
    assert "not certification" in actual
    assert "tested subset" in actual.casefold()
    assert "known gaps" in actual.casefold()
    assert V22_FOCUSED_TEST_COMMAND in actual


def test_hosted_workflow_runs_focused_command() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "v22-conformance:" in workflow
    assert "uv sync --frozen --extra dev --python 3.11" in workflow
    assert V22_FOCUSED_TEST_COMMAND in workflow
    assert ".venv/bin/python -m pytest tests/ -q" not in workflow
