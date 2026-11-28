"""Regression tests for packaged JSON Schema identifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _assert_unique_schema_ids(schema_paths: Iterable[Path]) -> None:
    """Require every schema to declare a unique, non-empty $id."""
    paths_by_id: dict[str, list[Path]] = {}
    missing_ids: list[Path] = []

    for path in sorted(schema_paths):
        schema = json.loads(path.read_text(encoding="utf-8"))
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            missing_ids.append(path)
            continue
        paths_by_id.setdefault(schema_id, []).append(path)

    duplicates = {
        schema_id: paths for schema_id, paths in paths_by_id.items() if len(paths) > 1
    }
    problems = [
        f"duplicate $id {schema_id!r}: {', '.join(str(path) for path in paths)}"
        for schema_id, paths in sorted(duplicates.items())
    ]
    problems.extend(f"missing $id: {path}" for path in missing_ids)

    if problems:
        details = "\n".join(problems)
        raise AssertionError(f"Invalid packaged JSON Schema identifiers:\n{details}")


def _packaged_schema_paths(repository_root: Path) -> list[Path]:
    """Return every packaged JSON Schema covered by the repository contract."""
    schema_directories = (
        repository_root / "openmed/core/schemas/json",
        repository_root / "openmed/compliance/schemas",
    )
    paths = [
        path
        for directory in schema_directories
        for path in directory.rglob("*.schema.json")
    ]
    paths.append(repository_root / "openmed/core/config.schema.json")
    return paths


def test_packaged_schema_ids_are_present_and_unique() -> None:
    repository_root = Path(__file__).resolve().parents[3]

    _assert_unique_schema_ids(_packaged_schema_paths(repository_root))


def test_duplicate_schema_id_reports_both_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.schema.json"
    second = tmp_path / "second.schema.json"
    schema_id = "https://example.invalid/schemas/shared"

    first.write_text(json.dumps({"$id": schema_id}), encoding="utf-8")
    second.write_text(json.dumps({"$id": schema_id}), encoding="utf-8")

    try:
        _assert_unique_schema_ids([first, second])
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError("duplicate schema $id was not detected")

    assert schema_id in message
    assert str(first) in message
    assert str(second) in message


def test_missing_schema_id_reports_file_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing-id.schema.json"
    missing.write_text(json.dumps({"type": "object"}), encoding="utf-8")

    try:
        _assert_unique_schema_ids([missing])
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError("schema without $id was not detected")

    assert str(missing) in message
    assert "$id" in message
