"""Structural checks for the synthetic ICD-10-CM tabular XML fixture."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "clinical"
    / "grounding"
    / "icd10cm_tabular_sample.xml"
)

NOTE_TAGS = (
    "excludes1",
    "excludes2",
    "codeFirst",
    "useAdditionalCode",
    "inclusionTerm",
    "sevenChrDef",
)


def _root() -> ET.Element:
    return ET.fromstring(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_is_well_formed_without_doctype_and_marked_as_subset() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    assert "<!DOCTYPE" not in text
    assert text.lstrip().split("?>", 1)[1].lstrip().startswith("<!--")
    assert "Hand-built subset" in text
    assert len(text.splitlines()) < 300
    assert _root().tag == "ICD10CM.tabular"


def test_fixture_has_two_chapters_and_about_thirty_codes() -> None:
    root = _root()
    assert len(root.findall("chapter")) == 2
    names = [node.findtext("name") for node in root.iter("diag")]
    assert all(names)
    assert len(names) == len(set(names))
    assert 25 <= len(names) <= 35


def test_fixture_contains_each_instructional_note_type() -> None:
    root = _root()
    for tag in NOTE_TAGS:
        assert len(list(root.iter(tag))) >= 1, tag


def test_seventh_character_definition_lists_extensions() -> None:
    definition = next(_root().iter("sevenChrDef"))
    chars = [ext.get("char") for ext in definition.findall("extension")]
    assert chars and all(len(char) == 1 for char in chars)


def test_code_first_and_excludes1_notes_have_text() -> None:
    root = _root()
    for tag in ("codeFirst", "excludes1"):
        for block in root.iter(tag):
            notes = [note.text for note in block.findall("note")]
            assert notes and all(notes)
