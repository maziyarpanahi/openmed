"""Tests for data-only declarative pipeline migration scanning."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from openmed.interop.bridges.pipeline_migration import (
    MAX_PIPELINE_STAGES,
    PipelineMigrationError,
    PipelineMigrationState,
    PipelineStageDisposition,
    load_pipeline_migration_schema,
    scan_pipeline_json,
    scan_pipeline_mapping,
)

FIXTURE = Path("tests/fixtures/annotation/pipeline.json")
MODULE = Path("openmed/interop/bridges/pipeline_migration.py")


def test_fixture_produces_deterministic_native_stub_and_valid_schema() -> None:
    payload = FIXTURE.read_bytes()

    first = scan_pipeline_json(payload)
    second = scan_pipeline_json(payload)

    assert first == second
    assert first.state is PipelineMigrationState.PARTIAL
    assert not first.can_auto_migrate
    assert [item["stage"] for item in first.native_config["stages"]] == [
        "source_adaptation",
        "extraction",
        "validation",
    ]
    schema = load_pipeline_migration_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    assert not tuple(validator(schema).iter_errors(first.to_dict()))


def test_unsupported_stage_remains_visible_and_blocks_success() -> None:
    report = scan_pipeline_mapping(
        {"stages": [{"type": "unmapped_stage", "config": {}}]}
    )

    assert report.state is PipelineMigrationState.UNSUPPORTED
    assert report.stages[0].disposition is PipelineStageDisposition.UNSUPPORTED
    assert report.native_config["stages"] == []


def test_empty_pipeline_is_unknown_and_cannot_auto_migrate() -> None:
    report = scan_pipeline_mapping({"stages": []})

    assert report.state is PipelineMigrationState.UNKNOWN
    assert not report.can_auto_migrate


def test_executable_fields_require_review_and_are_never_copied() -> None:
    report = scan_pipeline_mapping(
        {
            "stages": [
                {
                    "type": "extraction",
                    "config": {"module": "synthetic.module", "model_id": "model_a"},
                }
            ]
        }
    )

    assert report.state is PipelineMigrationState.UNSUPPORTED
    assert report.stages[0].disposition is PipelineStageDisposition.MANUAL_REVIEW
    assert report.stages[0].omitted_fields == ("module",)
    assert report.native_config["stages"] == []


def test_unknown_declarative_fields_are_declared_as_omitted() -> None:
    report = scan_pipeline_mapping(
        {
            "stages": [
                {
                    "type": "validation",
                    "display_hint": "review_only",
                    "config": {"profile": "clinical_default", "extra": "ignored"},
                }
            ]
        }
    )

    assert report.state is PipelineMigrationState.PARTIAL
    assert report.stages[0].omitted_fields == ("display_hint", "extra")
    assert report.native_config["stages"][0]["config"] == {
        "profile": "clinical_default"
    }


def test_invalid_field_names_require_manual_review() -> None:
    report = scan_pipeline_mapping(
        {"stages": [{"type": "validation", "config": {"bad field": "value"}}]}
    )

    assert report.state is PipelineMigrationState.UNSUPPORTED
    assert report.stages[0].disposition is PipelineStageDisposition.MANUAL_REVIEW
    assert report.native_config["stages"] == []


def test_scanner_source_has_no_dynamic_execution_calls() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    forbidden_calls = {"eval", "exec", "compile", "__import__", "import_module"}
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not calls & forbidden_calls
    assert "subprocess" not in imported_roots


def test_pipeline_abuse_limits_and_duplicate_keys_fail_closed() -> None:
    with pytest.raises(PipelineMigrationError, match="stage limit"):
        scan_pipeline_mapping(
            {"stages": [{"type": "validation"}] * (MAX_PIPELINE_STAGES + 1)}
        )
    with pytest.raises(PipelineMigrationError, match="invalid JSON"):
        scan_pipeline_json('{"stages": [], "stages": []}')
