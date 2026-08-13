"""Tests for optional Android metadata in canonical manifest rows."""

from __future__ import annotations

from openmed.core import model_registry
from openmed.core.manifest_schema import validate_manifest_row


def _manifest_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repo_id": "OpenMed/android-fixture-v1",
        "family": "NER",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": 65_000_000,
        "architecture": "distilbert",
        "base_model": "OpenMed/android-fixture",
        "formats": ["pytorch"],
        "canonical_labels": ["PERSON", "DATE"],
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:" + "1" * 64,
        "released": "2026-07-18",
    }
    row.update(overrides)
    return row


def test_manifest_row_accepts_optional_android_fields() -> None:
    row = _manifest_row(
        repo_id="OpenMed/android-fixture-v1-onnx-android",
        formats=["onnx-android", "onnx-int8", "ort-android"],
        nnapi_compatible=True,
        min_sdk=27,
        execution_providers=["NNAPI", "XNNPACK"],
        tokenizer_assets=["tokenizer.json", "tokenizer_config.json"],
    )

    assert validate_manifest_row(row, line_number=1) == []

    info = next(iter(model_registry._build_registry([row]).values()))
    assert info.formats == ["onnx-android", "onnx-int8", "ort-android"]
    assert info.nnapi_compatible is True
    assert info.min_sdk == 27
    assert info.execution_providers == ["NNAPI", "XNNPACK"]
    assert info.tokenizer_assets == ["tokenizer.json", "tokenizer_config.json"]


def test_manifest_row_without_android_fields_remains_valid() -> None:
    row = _manifest_row()

    assert validate_manifest_row(row, line_number=1) == []

    info = next(iter(model_registry._build_registry([row]).values()))
    assert info.nnapi_compatible is None
    assert info.min_sdk is None
    assert info.execution_providers == []
    assert info.tokenizer_assets == []


def test_manifest_row_rejects_invalid_android_field_types() -> None:
    row = _manifest_row(
        nnapi_compatible="yes",
        min_sdk=True,
        execution_providers=[],
        tokenizer_assets=[""],
    )

    assert [str(item) for item in validate_manifest_row(row, line_number=3)] == [
        "line 3: nnapi_compatible must be a boolean",
        "line 3: min_sdk must be a positive integer",
        "line 3: execution_providers must be a non-empty list",
        "line 3: tokenizer_assets[0] must be a non-empty string",
    ]
