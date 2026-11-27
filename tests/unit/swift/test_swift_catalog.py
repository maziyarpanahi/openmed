from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

import scripts.android.build_android_catalog as android_catalog
from scripts.android.build_android_catalog import (
    ANDROID_CATALOG_FIELDS,
    build_catalog_rows,
    is_permissive_license,
    load_manifest_rows,
)
from scripts.swift.build_swift_catalog import (
    DEFAULT_OUTPUT,
    SWIFT_CATALOG_FIELDS,
    build_swift_catalog_rows,
    is_apple_runnable_format,
    render_swift_catalog,
    run,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "model_catalog"
VECTORS = FIXTURES / "catalog_filter_vectors.json"


def _vectors() -> dict[str, Any]:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def _matches(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return True
    return actual is not None and actual.lower() == expected.lower()


def _matches_any(values: list[str], expected: str | None) -> bool:
    if expected is None:
        return True
    return any(value.lower() == expected.lower() for value in values)


def _reference_filter(rows: list[dict[str, Any]], query: dict[str, Any]) -> list[str]:
    """Android ``ModelCatalog.filter`` semantics, used to pin the vectors."""
    max_params = query.get("max_param_count")
    return [
        row["repo_id"]
        for row in rows
        if _matches_any(row["formats"], query.get("format"))
        and _matches(row["tier"], query.get("tier"))
        and (
            max_params is None
            or (row["param_count"] is not None and row["param_count"] <= max_params)
        )
        and _matches_any(row["languages"], query.get("language"))
        and _matches(row["license"], query.get("license"))
    ]


def test_apple_format_filter_keeps_mlx_and_coreml_only() -> None:
    for name in ("mlx", "mlx-fp", "MLX-8bit", "mlx_4bit", "coreml", "coreml-int8"):
        assert is_apple_runnable_format(name), name
    for name in ("onnx", "onnx-int8", "tflite", "pytorch", "gguf", ""):
        assert not is_apple_runnable_format(name), name


def test_swift_catalog_pins_the_leading_android_fields() -> None:
    assert SWIFT_CATALOG_FIELDS == (
        "repo_id",
        "formats",
        "tier",
        "param_count",
        "languages",
        "license",
        "reproducibility_hash",
    )
    assert ANDROID_CATALOG_FIELDS[: len(SWIFT_CATALOG_FIELDS)] == SWIFT_CATALOG_FIELDS


def test_android_only_fields_do_not_reach_the_swift_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _vectors()
    monkeypatch.setattr(
        android_catalog,
        "ANDROID_CATALOG_FIELDS",
        (*SWIFT_CATALOG_FIELDS, "family"),
    )

    text = render_swift_catalog(ROOT / vectors["manifest"])

    assert text.encode("utf-8") == (ROOT / vectors["catalog"]).read_bytes()


def test_generator_reproduces_the_committed_fixture_catalog() -> None:
    vectors = _vectors()

    text = render_swift_catalog(ROOT / vectors["manifest"])

    assert text.encode("utf-8") == (ROOT / vectors["catalog"]).read_bytes()


def test_filter_vectors_match_the_reference_semantics() -> None:
    vectors = _vectors()
    rows = build_swift_catalog_rows(load_manifest_rows(ROOT / vectors["manifest"]))

    assert vectors["cases"]
    for case in vectors["cases"]:
        assert _reference_filter(rows, case["query"]) == case["expected_repo_ids"], (
            case["id"]
        )
    repo_ids = {row["repo_id"] for row in rows}
    for lookup in vectors["lookups"]:
        assert (lookup["repo_id"] in repo_ids) is lookup["found"], lookup["repo_id"]


def test_rows_are_narrowed_to_apple_formats_and_sorted() -> None:
    vectors = _vectors()
    rows = build_swift_catalog_rows(load_manifest_rows(ROOT / vectors["manifest"]))

    assert [row["repo_id"] for row in rows] == sorted(
        (row["repo_id"] for row in rows), key=str.lower
    )
    for row in rows:
        assert row["formats"]
        assert all(is_apple_runnable_format(name) for name in row["formats"])
        assert is_permissive_license(row["license"])


def test_android_builder_default_is_unchanged_by_the_format_seam() -> None:
    vectors = _vectors()
    manifest_rows = load_manifest_rows(ROOT / vectors["manifest"])

    android_ids = [row["repo_id"] for row in build_catalog_rows(manifest_rows)]

    assert android_ids == [
        "OpenMed/Synthetic-NER-Large-434M-coreml",
        "OpenMed/Synthetic-PII-Small-onnx",
    ]


def test_rows_without_a_reproducibility_hash_are_refused(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    output = tmp_path / "catalog.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/Synthetic-NoHash-mlx",
                "formats": ["mlx-fp"],
                "license": "apache-2.0",
                "reproducibility_hash": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stderr = StringIO()

    assert run(manifest, output, stderr=stderr) == 1
    assert not output.exists()
    assert "no reproducibility_hash" in stderr.getvalue()
    assert "Synthetic-NoHash" not in stderr.getvalue()


def test_run_refuses_to_write_an_empty_catalog(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    output = tmp_path / "catalog.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/Synthetic-onnx",
                "formats": ["onnx"],
                "license": "apache-2.0",
                "reproducibility_hash": "sha256:" + "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stderr = StringIO()

    assert run(manifest, output, stderr=stderr) == 1
    assert not output.exists()
    assert "Refusing to write an empty Swift model catalog" in stderr.getvalue()


@pytest.mark.parametrize("initial", [None, b"", b"stale\n"])
def test_check_mode_fails_on_missing_or_stale_resource(
    tmp_path: Path, initial: bytes | None
) -> None:
    output = tmp_path / "catalog.jsonl"
    if initial is not None:
        output.write_bytes(initial)
    stderr = StringIO()

    assert (
        run(FIXTURES / "manifest_subset.jsonl", output, check=True, stderr=stderr) == 1
    )
    assert "Swift model catalog is stale" in stderr.getvalue()
    if initial is None:
        assert not output.exists()
    else:
        assert output.read_bytes() == initial


def test_check_mode_passes_after_a_write(tmp_path: Path) -> None:
    output = tmp_path / "catalog.jsonl"
    manifest = FIXTURES / "manifest_subset.jsonl"

    assert run(manifest, output) == 0
    stdout = StringIO()
    assert run(manifest, output, check=True, stdout=stdout) == 0
    assert "is current (5 rows)" in stdout.getvalue()


def test_committed_swift_catalog_is_current() -> None:
    stderr = StringIO()

    assert run(ROOT / "models.jsonl", DEFAULT_OUTPUT, check=True, stderr=stderr) == 0, (
        stderr.getvalue()
    )


def test_committed_swift_catalog_rows_are_safe_and_hashed() -> None:
    lines = DEFAULT_OUTPUT.read_text(encoding="utf-8").splitlines()

    assert lines
    for line in lines:
        row = json.loads(line)
        assert list(row) == list(SWIFT_CATALOG_FIELDS)
        assert row["repo_id"]
        assert row["formats"]
        assert all(is_apple_runnable_format(name) for name in row["formats"])
        assert is_permissive_license(row["license"])
        assert row["reproducibility_hash"].startswith("sha256:")


def test_ci_checks_and_refresh_regenerates_the_swift_catalog() -> None:
    workflows = ROOT / ".github" / "workflows"
    coherence = (workflows / "registry-coherence.yml").read_text(encoding="utf-8")
    refresh = (workflows / "manifest-refresh.yml").read_text(encoding="utf-8")

    assert "scripts/swift/build_swift_catalog.py --check" in coherence
    assert '"models.jsonl"' in coherence
    assert "Resources/openmed_model_catalog.jsonl" in coherence
    assert "python scripts/swift/build_swift_catalog.py\n" in refresh
