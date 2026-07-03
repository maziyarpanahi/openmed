from __future__ import annotations

import json
from pathlib import Path

from scripts.android.build_android_catalog import (
    ANDROID_CATALOG_FIELDS,
    build_catalog_rows,
    is_android_runnable_format,
    is_permissive_license,
    load_manifest_rows,
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


def test_committed_manifest_derives_only_safe_android_rows() -> None:
    manifest_rows = load_manifest_rows(Path("models.jsonl"))

    catalog_rows = build_catalog_rows(manifest_rows)

    for row in catalog_rows:
        assert is_permissive_license(row["license"])
        assert row["repo_id"]
        assert row["reproducibility_hash"].startswith("sha256:")
        assert set(row) == set(ANDROID_CATALOG_FIELDS)
        assert row["formats"]
        assert all(
            is_android_runnable_format(format_name) for format_name in row["formats"]
        )
