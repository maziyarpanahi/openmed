"""Tests for packaged JSON Schema $id uniqueness and presence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def collect_schema_ids(schema_paths: Iterable[Path]) -> dict[str, Path]:
    """Inspect schema files for presence and uniqueness of $id.

    Args:
        schema_paths: Paths to *.schema.json files.

    Returns:
        Mapping of $id -> Path for all inspected schemas.

    Raises:
        ValueError: If a schema is missing $id or if duplicate $ids are found,
            listing file paths and duplicated identifiers.
    """
    seen_ids: dict[str, Path] = {}
    for path in schema_paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        schema_id = data.get("$id")
        if not schema_id or not isinstance(schema_id, str) or not schema_id.strip():
            raise ValueError(f"Schema {path.name} ({path}) is missing $id")

        if schema_id in seen_ids:
            first_path = seen_ids[schema_id]
            raise ValueError(
                f"Duplicate $id '{schema_id}' found in: {first_path.name} ({first_path}) and {path.name} ({path})"
            )
        seen_ids[schema_id] = path

    return seen_ids


def get_packaged_schema_paths() -> list[Path]:
    """Return all packaged schema paths defined across openmed schema packages."""
    openmed_dir = REPO_ROOT / "openmed"
    return sorted(openmed_dir.rglob("*.schema.json"))


def test_packaged_schemas_have_unique_ids() -> None:
    schemas = get_packaged_schema_paths()
    assert len(schemas) > 0
    ids = collect_schema_ids(schemas)
    assert len(ids) == len(schemas)


def test_duplicate_id_reports_both_files(tmp_path: Path) -> None:
    f1 = tmp_path / "schema_a.schema.json"
    f2 = tmp_path / "schema_b.schema.json"
    shared_id = "https://schemas.openmed.org/v1/shared.json"

    f1.write_text(json.dumps({"$id": shared_id, "type": "object"}), encoding="utf-8")
    f2.write_text(json.dumps({"$id": shared_id, "type": "object"}), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        collect_schema_ids([f1, f2])

    err_msg = str(exc_info.value)
    assert shared_id in err_msg
    assert f1.name in err_msg
    assert f2.name in err_msg


def test_missing_id_reports_file_name(tmp_path: Path) -> None:
    f = tmp_path / "missing_id.schema.json"
    f.write_text(json.dumps({"type": "object", "properties": {}}), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        collect_schema_ids([f])

    err_msg = str(exc_info.value)
    assert f.name in err_msg
    assert "missing $id" in err_msg.lower()
