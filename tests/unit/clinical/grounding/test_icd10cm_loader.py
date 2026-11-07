"""Focused tests for the local ICD-10-CM vocabulary loader."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical.grounding import (
    ICD10CM_LICENSE_NOTE,
    ICD10CM_SYSTEM_URI,
    Icd10cmLoader,
    VocabularyLoaderRegistry,
    validate_vocabulary_loader,
)


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "synthetic-icd10cm-release"
    release.mkdir()
    (release / "icd10cm_tabular.tsv").write_text(
        "code\tdescription\tsynonyms\n"
        "A00\tSynthetic parent category\tparent synthetic category\n"
        "A00.1\tSynthetic intermediate condition\tintermediate synthetic condition\n"
        "A00.11\tSynthetic billable condition\tcondition alias\n",
        encoding="utf-8",
    )
    (release / "icd10cm_index.txt").write_text(
        "synthetic condition alias A00.11\nsynthetic intermediate condition A00.1\n",
        encoding="utf-8",
    )
    return release


def test_loader_resolves_index_alias_to_billable_code_with_category_path(tmp_path):
    loader = Icd10cmLoader(_release(tmp_path))

    match = loader.resolve_one("synthetic condition alias")

    assert match is not None
    assert match.system_uri == ICD10CM_SYSTEM_URI
    assert match.code == "A00.11"
    assert match.metadata["billable"] is True
    assert match.metadata["category_path"] == ("A00", "A00.1", "A00.11")
    assert match.metadata["license_note"] == ICD10CM_LICENSE_NOTE


def test_header_codes_are_valid_but_not_billable_and_hierarchy_is_traversable(
    tmp_path,
):
    loader = Icd10cmLoader(_release(tmp_path))

    header = loader.resolve_one("synthetic parent category")

    assert header is not None
    assert header.code == "A00"
    assert header.metadata["billable"] is False
    assert loader.is_valid_code("a0011")
    assert loader.is_valid_code("A00")
    assert not loader.is_valid_code("A00", billable_only=True)
    assert loader.is_billable_code("A00.11")
    assert not loader.is_valid_code("A00.99")
    assert loader.ancestors("A00.11") == ("A00", "A00.1")
    assert loader.category_path("A00.11") == ("A00", "A00.1", "A00.11")
    assert loader.children("A00") == ("A00.1",)
    assert loader.descendants("A00") == ("A00.1", "A00.11")
    assert loader.resolve("synthetic parent category", billable_only=True) == ()


def test_loader_accepts_json_projection_and_registers_with_offline_matcher(tmp_path):
    projection = tmp_path / "codes.jsonl"
    projection.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "code": "B20",
                        "description": "Synthetic leaf finding",
                        "billable": True,
                        "aliases": ["synthetic leaf"],
                    }
                ),
                json.dumps(
                    {
                        "code": "B2",
                        "description": "Not an ICD-10-CM code",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loader = Icd10cmLoader(projection)

    assert validate_vocabulary_loader(loader) is loader
    registry = VocabularyLoaderRegistry()
    registry.register(loader)
    matcher = registry.matcher(ICD10CM_SYSTEM_URI)
    matches = matcher.lookup("synthetic leaf")

    assert matches
    assert matches[0].code == "B20"
    assert matches[0].metadata["billable"] is True
    assert loader.codes == ("B20",)


def test_loader_parses_cms_shaped_tabular_text(tmp_path):
    tabular = tmp_path / "tabular.txt"
    tabular.write_text(
        "A01 Synthetic category\n"
        "  A01.0 Synthetic billable entry\n"
        "  A01.1 Synthetic second entry\n",
        encoding="utf-8",
    )
    loader = Icd10cmLoader(tabular)

    assert loader.is_billable_code("A01.0")
    assert loader.resolve_one("Synthetic billable entry").code == "A01.0"
