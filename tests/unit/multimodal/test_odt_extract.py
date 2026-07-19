"""Tests for ODT/ODF text extraction with character-offset maps."""

from __future__ import annotations

import inspect
import zipfile
from pathlib import Path

import pytest

from openmed.multimodal import (
    ExtractedDocument,
    extract_docx,
    extract_odt,
    redact_document,
)
from openmed.multimodal.exceptions import UnsupportedDocumentError

CONTENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    office:version="1.3">
  <office:body>
    <office:text>
      <text:h text:outline-level="1">Visit summary</text:h>
      <text:p>Patient <text:span>Jane</text:span><text:s text:c="2"/>Roe<text:tab/>ID<text:line-break/>A123</text:p>
      <text:list>
        <text:list-item><text:p>First medicine</text:p></text:list-item>
        <text:list-item>
          <text:p>Second <text:span>medicine</text:span></text:p>
          <text:list>
            <text:list-item><text:p>Nested detail</text:p></text:list-item>
          </text:list>
        </text:list-item>
      </text:list>
      <table:table table:name="Identifiers">
        <table:table-row>
          <table:table-cell><text:p>Field</text:p></table:table-cell>
          <table:table-cell><text:p>Value</text:p></table:table-cell>
        </table:table-row>
        <table:table-row>
          <table:table-cell><text:p>Name</text:p></table:table-cell>
          <table:table-cell><text:p>Jane Roe</text:p></table:table-cell>
        </table:table-row>
        <table:table-row>
          <table:table-cell><text:p>MRN</text:p></table:table-cell>
          <table:table-cell><text:p>A123</text:p></table:table-cell>
        </table:table-row>
      </table:table>
    </office:text>
  </office:body>
</office:document-content>
"""

EXPECTED_TEXT = (
    "Visit summary\n"
    "Patient Jane  Roe\tID\nA123\n"
    "First medicine\n"
    "Second medicine\n"
    "Nested detail\n"
    "Field\tValue\n"
    "Name\tJane Roe\n"
    "MRN\tA123"
)


def _write_synthetic_odt(path: Path, content: str = CONTENT_XML) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        archive.writestr("content.xml", content)
    return path


def test_extract_odt_preserves_paragraph_and_list_reading_order(tmp_path: Path):
    path = _write_synthetic_odt(tmp_path / "synthetic_phi.odt")

    document = extract_odt(path)

    assert isinstance(document, ExtractedDocument)
    assert document.text == EXPECTED_TEXT
    assert document.metadata["format"] == "odt"
    assert document.metadata["content_path"] == "content.xml"
    assert document.metadata["paragraph_count"] == 11
    assert document.metadata["list_item_count"] == 3
    assert document.metadata["table_count"] == 1
    assert document.metadata["table_row_count"] == 3
    assert document.metadata["table_cell_count"] == 6
    assert "source_text" not in document.metadata

    list_span = document.location_at(document.text.index("Nested detail"))
    assert list_span is not None
    assert list_span.metadata["block_type"] == "list_item"
    assert list_span.metadata["list_level"] == 2


def test_odt_offsets_round_trip_inline_and_explicit_whitespace(tmp_path: Path):
    path = _write_synthetic_odt(tmp_path / "synthetic_phi.odt")
    document = extract_odt(path)

    for span in document.spans:
        assert document.text_for(span) == document.text[span.start : span.end]
        assert document.location_at(span.start) == span

    patient_name = document.text.index("Jane  Roe")
    name_span = document.location_at(patient_name)
    spaces_span = document.location_at(patient_name + len("Jane"))
    line_break_span = document.location_at(document.text.index("A123"))

    assert name_span is not None
    assert document.text_for(name_span) == "Jane"
    assert name_span.metadata["format"] == "odt"
    assert name_span.metadata["block_type"] == "paragraph"
    assert spaces_span is not None
    assert document.text_for(spaces_span) == "  "
    assert spaces_span.metadata["node_type"] == "space"
    assert line_break_span is not None
    assert line_break_span.metadata["paragraph_index"] == 1


def test_odt_tables_linearize_predictably_and_keep_cell_offsets(tmp_path: Path):
    path = _write_synthetic_odt(tmp_path / "synthetic_phi.odt")
    document = extract_odt(path)

    assert document.text.endswith("Field\tValue\nName\tJane Roe\nMRN\tA123")

    table_name = document.text.rindex("Jane Roe")
    table_name_span = document.location_at(table_name)
    table_mrn = document.location_at(document.text.rindex("A123"))

    assert table_name_span is not None
    assert document.text_for(table_name_span) == "Jane Roe"
    assert table_name_span.metadata["block_type"] == "table_cell"
    assert table_name_span.metadata["table_index"] == 0
    assert table_name_span.metadata["row_index"] == 1
    assert table_name_span.metadata["cell_index"] == 1
    assert table_mrn is not None
    assert document.text_for(table_mrn) == "A123"
    assert table_mrn.metadata["row_index"] == 2
    assert table_mrn.metadata["cell_index"] == 1


def test_odt_ingester_is_discoverable_through_registry(tmp_path: Path):
    path = _write_synthetic_odt(tmp_path / "synthetic_phi.odt")

    document = redact_document(path)

    assert isinstance(document, ExtractedDocument)
    assert document.text == EXPECTED_TEXT


def test_odt_extractor_matches_docx_extractor_contract(tmp_path: Path):
    path = _write_synthetic_odt(tmp_path / "synthetic_phi.odt")

    assert inspect.signature(extract_odt) == inspect.signature(extract_docx)
    assert type(extract_odt(path)) is ExtractedDocument


def test_invalid_odt_archive_raises_clear_error(tmp_path: Path):
    path = tmp_path / "invalid.odt"
    path.write_bytes(b"not an odt archive")

    with pytest.raises(UnsupportedDocumentError, match="valid ZIP archive"):
        extract_odt(path)


def test_odt_rejects_unsafe_xml_declarations(tmp_path: Path):
    unsafe = CONTENT_XML.replace(
        "<office:document-content",
        '<!DOCTYPE office:document-content [<!ENTITY phi "Jane Roe">]>\n'
        "<office:document-content",
        1,
    )
    path = _write_synthetic_odt(tmp_path / "unsafe.odt", unsafe)

    with pytest.raises(UnsupportedDocumentError, match="DOCTYPE or ENTITY"):
        extract_odt(path)
