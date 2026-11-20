"""Tests for the generated web package ONNX model catalog."""

from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

import pytest

from scripts.android.build_android_catalog import (
    ANDROID_CATALOG_FIELDS,
    is_permissive_license,
    write_catalog,
)
from scripts.web.build_web_catalog import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    MAX_SAFE_INTEGER,
    REGENERATE_COMMAND,
    WEB_CATALOG_FIELDS,
    WebCatalogError,
    build_web_catalog_rows,
    catalog_digest,
    is_web_runnable_format,
    main,
    render_catalog_module,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _hash(digit: str) -> str:
    return "sha256:" + digit * 64


def _synthetic_manifest() -> list[dict[str, object]]:
    return [
        {
            "repo_id": "Synthetic/zulu-pytorch-only",
            "formats": ["pytorch"],
            "tier": "Small",
            "param_count": 1,
            "languages": ["en"],
            "license": "apache-2.0",
            "reproducibility_hash": _hash("0"),
        },
        {
            "repo_id": "Synthetic/bravo-onnx",
            "formats": ["pytorch", "onnx", "ONNX_INT8", "tflite-int8"],
            "tier": "Small",
            "param_count": 44_000_000,
            "languages": ["en", "es"],
            "license": "Apache-2.0",
            "reproducibility_hash": _hash("1"),
            "task": "token-classification",
        },
        {
            "repo_id": "Synthetic/alpha-onnx",
            "formats": ["onnx"],
            "tier": None,
            "param_count": None,
            "languages": [],
            "license": "MIT",
            "reproducibility_hash": _hash("2"),
        },
        {
            "repo_id": "Synthetic/restricted-onnx",
            "formats": ["onnx"],
            "tier": "Small",
            "param_count": 1,
            "languages": ["en"],
            "license": "other",
            "reproducibility_hash": _hash("3"),
        },
        {
            "repo_id": "Synthetic/unlicensed-onnx",
            "formats": ["onnx"],
            "tier": "Small",
            "param_count": 1,
            "languages": ["en"],
            "license": None,
            "reproducibility_hash": _hash("4"),
        },
        {
            "repo_id": "Synthetic/tflite-only",
            "formats": ["tflite-int8"],
            "tier": "Small",
            "param_count": 1,
            "languages": ["fr"],
            "license": "apache-2.0",
            "reproducibility_hash": _hash("5"),
        },
    ]


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_rows_keep_only_permissive_onnx_formats_with_android_fields() -> None:
    rows = build_web_catalog_rows(_synthetic_manifest())

    assert rows == [
        {
            "repo_id": "Synthetic/alpha-onnx",
            "formats": ["onnx"],
            "tier": None,
            "param_count": None,
            "languages": [],
            "license": "MIT",
            "reproducibility_hash": _hash("2"),
        },
        {
            "repo_id": "Synthetic/bravo-onnx",
            "formats": ["onnx", "ONNX_INT8"],
            "tier": "Small",
            "param_count": 44_000_000,
            "languages": ["en", "es"],
            "license": "Apache-2.0",
            "reproducibility_hash": _hash("1"),
        },
    ]
    assert WEB_CATALOG_FIELDS == ANDROID_CATALOG_FIELDS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("onnx", True),
        ("onnx-int8", True),
        ("ONNX_FP16", True),
        ("tflite-int8", False),
        ("pytorch", False),
        ("mlx-8bit", False),
    ],
)
def test_web_runnable_formats_are_onnx_only(value: str, expected: bool) -> None:
    assert is_web_runnable_format(value) is expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("param_count", 1.5),
        ("param_count", True),
        ("param_count", -1),
        ("param_count", MAX_SAFE_INTEGER + 1),
        ("tier", 3),
        ("reproducibility_hash", None),
    ],
)
def test_rows_refuse_values_the_typed_catalog_cannot_hold(
    field: str, value: object
) -> None:
    row = dict(_synthetic_manifest()[1])
    row[field] = value

    with pytest.raises(WebCatalogError):
        build_web_catalog_rows([row])


def test_rows_refuse_duplicate_repo_ids() -> None:
    row = _synthetic_manifest()[1]

    with pytest.raises(WebCatalogError, match="duplicate repo_id"):
        build_web_catalog_rows([row, dict(row)])


def test_digest_hashes_the_android_writer_bytes(tmp_path: Path) -> None:
    rows = build_web_catalog_rows(_synthetic_manifest())
    android_jsonl = tmp_path / "catalog.jsonl"

    write_catalog(rows, android_jsonl)

    expected = hashlib.sha256(android_jsonl.read_bytes()).hexdigest()
    assert catalog_digest(rows) == f"sha256:{expected}"


def test_rendered_module_is_one_row_per_line_and_marked_generated() -> None:
    rows = build_web_catalog_rows(_synthetic_manifest())

    rendered = render_catalog_module(rows)

    assert rendered.startswith("// GENERATED FILE. Do not edit by hand.\n")
    assert f"// Regenerate: {REGENERATE_COMMAND}\n" in rendered
    assert f'export const MODEL_CATALOG_DIGEST = "{catalog_digest(rows)}";' in (
        rendered
    )
    row_lines = [line for line in rendered.splitlines() if line.startswith("  {")]
    assert [json.loads(line.strip().rstrip(",")) for line in row_lines] == rows


def test_check_passes_after_write_and_fails_after_manifest_change(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "models.jsonl", _synthetic_manifest())
    output = tmp_path / "catalog.generated.ts"

    assert run(manifest, output) == 0
    assert run(manifest, output, check=True) == 0

    changed = _synthetic_manifest()
    changed[1]["languages"] = ["en", "es", "pt"]
    _write_manifest(manifest, changed)
    stderr = StringIO()

    assert run(manifest, output, check=True, stderr=stderr) == 1
    assert "web model catalog is stale" in stderr.getvalue()
    assert REGENERATE_COMMAND in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()

    assert run(manifest, output) == 0
    assert run(manifest, output, check=True) == 0


def test_check_fails_when_the_generated_module_is_missing(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "models.jsonl", _synthetic_manifest())

    assert run(manifest, tmp_path / "missing.ts", check=True) == 1
    assert not (tmp_path / "missing.ts").exists()


def test_empty_or_invalid_manifests_refuse_without_writing(tmp_path: Path) -> None:
    output = tmp_path / "catalog.generated.ts"
    empty = _write_manifest(tmp_path / "empty.jsonl", [_synthetic_manifest()[0]])
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json\n", encoding="utf-8")
    stderr = StringIO()

    assert run(empty, output, stderr=stderr) == 2
    assert run(broken, output, stderr=stderr) == 2
    assert run(tmp_path / "absent.jsonl", output, stderr=stderr) == 2

    assert not output.exists()
    assert "Refusing to write an empty web model catalog" in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_committed_catalog_is_current_with_the_manifest() -> None:
    assert main(["--check"]) == 0


def test_committed_catalog_holds_only_permissive_onnx_rows() -> None:
    text = DEFAULT_OUTPUT.read_text(encoding="utf-8")
    rows = [
        json.loads(line.strip().rstrip(","))
        for line in text.splitlines()
        if line.startswith("  {")
    ]
    manifest_repo_ids = {
        json.loads(line)["repo_id"]
        for line in DEFAULT_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    assert rows
    assert len({row["repo_id"] for row in rows}) == len(rows)
    for row in rows:
        assert tuple(row) == WEB_CATALOG_FIELDS
        assert row["repo_id"] in manifest_repo_ids
        assert is_permissive_license(row["license"])
        assert row["formats"]
        assert all(is_web_runnable_format(name) for name in row["formats"])


def test_catalog_drift_is_gated_and_refreshed_with_the_manifest() -> None:
    coherence = (WORKFLOWS / "registry-coherence.yml").read_text(encoding="utf-8")
    refresh = (WORKFLOWS / "manifest-refresh.yml").read_text(encoding="utf-8")

    assert f"{REGENERATE_COMMAND} --check" in coherence
    assert "js/openmedkit-web/src/catalog.generated.ts" in coherence
    assert "scripts/web/build_web_catalog.py" in coherence
    assert REGENERATE_COMMAND in refresh
