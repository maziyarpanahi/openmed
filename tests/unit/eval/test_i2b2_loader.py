"""Unit tests for the eval-only i2b2 de-identification loader."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    DATE,
    ID_NUM,
    OCCUPATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    USERNAME,
)
from openmed.eval.datasets import i2b2
from openmed.eval.datasets.i2b2 import (
    I2B2,
    I2B2_PATH_ENV,
    I2B2_PHI_TAG_ALIASES,
    I2B2_PHI_TAG_TO_CANONICAL,
    I2B2_PHI_TAGS,
    I2B2_YEAR_ENV,
    I2B2CredentialRequired,
    load_i2b2_deid,
    map_i2b2_phi_tag,
)
from openmed.eval.suites import load_suite_fixtures, suite_metadata


def test_load_i2b2_deid_parses_synthetic_xml_fixture(tmp_path: Path) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    xml_path, expected_spans = _write_i2b2_xml(
        source,
        "record-001.xml",
        [
            ("NAME", "PATIENT", "Jordan Smith"),
            ("DATE", None, "2024-05-01"),
            ("AGE", None, "47"),
            ("LOCATION", "HOSPITAL", "Mercy General"),
            ("ID", "MEDICALRECORD", "MRN-001"),
            ("CONTACT", "PHONE", "555-0101"),
            ("PHI", "USERNAME", "jsmith"),
            ("PROFESSION", None, "nurse"),
        ],
    )

    fixtures = load_i2b2_deid(source, year=2014)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.fixture_id.startswith("i2b2-2014-")
    assert fixture.metadata["dua"] == "i2b2/DBMI DUA"
    assert fixture.metadata["source_path_hash"] != xml_path.name
    assert fixture.metadata["year"] == 2014
    assert [span.label for span in fixture.gold_spans] == [
        PERSON,
        DATE,
        AGE,
        ORGANIZATION,
        ID_NUM,
        PHONE,
        USERNAME,
        OCCUPATION,
    ]
    for span, expected in zip(fixture.gold_spans, expected_spans, strict=True):
        assert (span.start, span.end, span.text) == expected
        assert fixture.text[span.start : span.end] == span.text
        assert span.metadata["i2b2_tag"] in I2B2_PHI_TAG_TO_CANONICAL


def test_loader_refuses_missing_empty_and_repo_internal_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(I2B2_PATH_ENV, raising=False)

    with pytest.raises(I2B2CredentialRequired, match="i2b2/DBMI DUA"):
        load_i2b2_deid()

    empty_source = tmp_path / "empty"
    empty_source.mkdir()
    with pytest.raises(I2B2CredentialRequired, match="i2b2/DBMI DUA"):
        load_i2b2_deid(empty_source)

    def fail_if_scanned(root: Path):  # pragma: no cover - should not be called
        raise AssertionError(f"repo path was scanned: {root}")

    monkeypatch.setattr(i2b2, "_iter_xml_files", fail_if_scanned)
    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(I2B2CredentialRequired, match="repository tree"):
        load_i2b2_deid(repo_root)


def test_i2b2_category_map_is_total_canonical_and_strict() -> None:
    assert set(I2B2_PHI_TAG_TO_CANONICAL) == set(I2B2_PHI_TAGS)
    assert set(I2B2_PHI_TAG_TO_CANONICAL.values()) <= CANONICAL_LABELS
    assert map_i2b2_phi_tag("NAME/PATIENT") == PERSON
    assert map_i2b2_phi_tag("patient") == PERSON
    assert map_i2b2_phi_tag("CONTACT/PHONE") == PHONE
    assert map_i2b2_phi_tag("medical record number") == ID_NUM
    assert map_i2b2_phi_tag("ID/MEDICAL_RECORD_NUMBER") == ID_NUM

    for tag in I2B2_PHI_TAGS:
        assert map_i2b2_phi_tag(tag) == I2B2_PHI_TAG_TO_CANONICAL[tag]

    for alias in I2B2_PHI_TAG_ALIASES:
        assert map_i2b2_phi_tag(alias) in CANONICAL_LABELS

    with pytest.raises(ValueError, match="unknown i2b2 PHI tag"):
        map_i2b2_phi_tag("NAME/MASCOT")


def test_unknown_i2b2_xml_tag_is_surfaced(tmp_path: Path) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    _write_i2b2_xml(
        source,
        "unknown.xml",
        [("NAME", "MASCOT", "Example")],
    )

    with pytest.raises(ValueError, match="unknown i2b2 PHI tag"):
        load_i2b2_deid(source)


def test_suite_registry_loads_i2b2_from_configured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    _write_i2b2_xml(source, "record-001.xml", [("NAME", "PATIENT", "Jordan")])
    monkeypatch.setenv(I2B2_PATH_ENV, str(source))
    monkeypatch.setenv(I2B2_YEAR_ENV, "2006")

    fixtures = load_suite_fixtures(I2B2)
    metadata = suite_metadata(I2B2)

    assert len(fixtures) == 1
    assert fixtures[0].metadata["year"] == 2006
    assert fixtures[0].gold_spans[0].label == PERSON
    assert metadata["suite"] == I2B2
    assert metadata["label_mapping"]["NAME/PATIENT"] == PERSON


def _write_i2b2_xml(
    source: Path,
    filename: str,
    pieces: list[tuple[str, str | None, str]],
) -> tuple[Path, list[tuple[int, int, str]]]:
    document = ET.Element("deIdi2b2")
    text_node = ET.SubElement(document, "TEXT")
    tags_node = ET.SubElement(document, "TAGS")
    text = ""
    expected: list[tuple[int, int, str]] = []
    for index, (category, source_type, value) in enumerate(pieces, start=1):
        if text:
            text += " "
        start = len(text)
        text += value
        end = len(text)
        attrs = {
            "end": str(end),
            "id": f"P{index}",
            "start": str(start),
            "text": value,
        }
        if source_type is not None:
            attrs["TYPE"] = source_type
        ET.SubElement(tags_node, category, attrs)
        expected.append((start, end, value))

    text_node.text = text
    path = source / filename
    path.write_text(ET.tostring(document, encoding="unicode"), encoding="utf-8")
    return path, expected
