from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from scripts.android.build_android_catalog import (
    ANDROID_CATALOG_FIELDS,
    build_catalog_rows,
    is_android_runnable_format,
    is_permissive_license,
    load_manifest_rows,
    run,
    write_catalog,
)


def test_build_catalog_rows_filters_to_permissive_android_formats() -> None:
    rows = [
        {
            "repo_id": "OpenMed/pytorch-only",
            "formats": ["pytorch"],
            "tier": "tiny",
            "param_count": 1,
            "languages": ["en"],
            "license": "apache-2.0",
            "reproducibility_hash": "sha256:" + "0" * 64,
        },
        {
            "repo_id": "OpenMed/onnx",
            "formats": ["pytorch", "onnx", "onnx-int8"],
            "tier": "tiny",
            "param_count": 33_000_000,
            "languages": ["en", "es"],
            "license": "apache-2.0",
            "reproducibility_hash": "sha256:" + "1" * 64,
            "family": "PII",
            "task": "token-classification",
            "canonical_labels": ["NAME", "DATE"],
        },
        {
            "repo_id": "OpenMed/tflite",
            "formats": ["tflite-int8"],
            "tier": "base",
            "param_count": 125_000_000,
            "languages": ["fr"],
            "license": "MIT",
            "reproducibility_hash": "sha256:" + "2" * 64,
        },
        {
            "repo_id": "OpenMed/restricted",
            "formats": ["onnx"],
            "tier": "base",
            "param_count": 125_000_000,
            "languages": ["en"],
            "license": "gpl-3.0",
            "reproducibility_hash": "sha256:" + "3" * 64,
        },
        {
            "repo_id": "OpenMed/mlx-quantized",
            "formats": ["mlx-8bit"],
            "tier": "base",
            "param_count": 125_000_000,
            "languages": ["en"],
            "license": "apache-2.0",
            "reproducibility_hash": "sha256:" + "4" * 64,
        },
    ]

    catalog_rows = build_catalog_rows(rows)

    assert catalog_rows == [
        {
            "repo_id": "OpenMed/onnx",
            "formats": ["onnx", "onnx-int8"],
            "tier": "tiny",
            "param_count": 33_000_000,
            "languages": ["en", "es"],
            "license": "apache-2.0",
            "reproducibility_hash": "sha256:" + "1" * 64,
            "family": "PII",
            "task": "token-classification",
            "canonical_labels": ["NAME", "DATE"],
        },
        {
            "repo_id": "OpenMed/tflite",
            "formats": ["tflite-int8"],
            "tier": "base",
            "param_count": 125_000_000,
            "languages": ["fr"],
            "license": "MIT",
            "reproducibility_hash": "sha256:" + "2" * 64,
            "family": None,
            "task": None,
            "canonical_labels": [],
        },
    ]


def test_write_catalog_emits_compact_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "catalog.jsonl"
    row = {
        "repo_id": "OpenMed/onnx",
        "formats": ["onnx"],
        "tier": None,
        "param_count": None,
        "languages": ["en"],
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:" + "1" * 64,
        "ignored": "not written",
    }

    write_catalog([row], output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        field: row.get(field) for field in ANDROID_CATALOG_FIELDS
    }
    assert list(json.loads(lines[0]))[-3:] == ["family", "task", "canonical_labels"]


def test_generated_asset_includes_family_task_and_canonical_labels(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "models.jsonl"
    output = tmp_path / "catalog.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/pii-onnx",
                "formats": ["onnx"],
                "tier": "tiny",
                "param_count": 33_000_000,
                "languages": ["en"],
                "license": "apache-2.0",
                "reproducibility_hash": "sha256:" + "5" * 64,
                "family": "PII",
                "task": "token-classification",
                "canonical_labels": ["NAME", "PHONE"],
                "source_model_id": "not written",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert run(manifest, output) == 0

    (line,) = output.read_text(encoding="utf-8").splitlines()
    written = json.loads(line)
    assert set(written) == set(ANDROID_CATALOG_FIELDS)
    assert written["family"] == "PII"
    assert written["task"] == "token-classification"
    assert written["canonical_labels"] == ["NAME", "PHONE"]


def test_committed_manifest_derives_only_safe_android_rows() -> None:
    manifest_rows = load_manifest_rows(Path("models.jsonl"))

    catalog_rows = build_catalog_rows(manifest_rows)

    assert catalog_rows
    catalog_repo_ids = {row["repo_id"] for row in catalog_rows}
    assert "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android" in (
        catalog_repo_ids
    )
    assert "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-onnx-android" in (
        catalog_repo_ids
    )
    for row in catalog_rows:
        assert is_permissive_license(row["license"])
        assert row["repo_id"]
        assert row["reproducibility_hash"].startswith("sha256:")
        assert set(row) == set(ANDROID_CATALOG_FIELDS)
        assert row["formats"]
        assert all(
            is_android_runnable_format(format_name) for format_name in row["formats"]
        )
        assert isinstance(row["family"], str) and row["family"]
        assert isinstance(row["task"], str) and row["task"]
        assert all(isinstance(label, str) for label in row["canonical_labels"])
    manifest_by_repo = {row["repo_id"]: row for row in manifest_rows}
    pii = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android"
    pii_row = next(row for row in catalog_rows if row["repo_id"] == pii)
    assert pii_row["family"] == "PII"
    assert pii_row["task"] == "token-classification"
    assert pii_row["canonical_labels"] == manifest_by_repo[pii]["canonical_labels"]


def test_run_refuses_to_write_an_empty_catalog(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    output = tmp_path / "catalog.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/pytorch-only",
                "formats": ["pytorch"],
                "license": "apache-2.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stderr = StringIO()

    exit_code = run(manifest, output, stderr=stderr)

    assert exit_code == 1
    assert not output.exists()
    assert "Refusing to write an empty Android model catalog" in stderr.getvalue()
