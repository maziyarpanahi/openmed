"""Tests for DOCX text extraction and run-level redaction."""

from __future__ import annotations

from pathlib import Path

import pytest

import openmed.multimodal.base as base
from openmed.multimodal import (
    ExtractedDocument,
    extract_docx,
    map_text_spans_to_docx_runs,
    redact_document,
    write_redacted_docx,
)


@pytest.fixture
def docx_deps_present(monkeypatch):
    """Pretend the multimodal extra is installed for dispatcher tests."""
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])


@pytest.fixture
def synthetic_docx(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx")
    document = docx.Document()

    header = document.sections[0].header.paragraphs[0]
    header.add_run("Clinic contact Dr. ")
    header.add_run("Alice Smith")

    paragraph = document.add_paragraph()
    paragraph.add_run("Patient ")
    paragraph.add_run("Ja")
    paragraph.add_run("ne")
    paragraph.add_run(" Roe")

    table = document.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    cell.paragraphs[0].add_run("MRN ")
    cell.paragraphs[0].add_run("A123")

    footer = document.sections[0].footer.paragraphs[0]
    footer.add_run("Confidential discharge summary")

    path = tmp_path / "synthetic_phi.docx"
    document.save(path)
    return path


def _docx_text_parts(path: Path) -> tuple[str, str, str, str]:
    docx = pytest.importorskip("docx")
    document = docx.Document(path)
    header = "\n".join(
        paragraph.text for paragraph in document.sections[0].header.paragraphs
    )
    body = "\n".join(paragraph.text for paragraph in document.paragraphs)
    tables = "\n".join(
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
    )
    footer = "\n".join(
        paragraph.text for paragraph in document.sections[0].footer.paragraphs
    )
    return header, body, tables, footer


def test_extract_docx_maps_offsets_to_source_runs(synthetic_docx: Path):
    doc = extract_docx(synthetic_docx)

    assert isinstance(doc, ExtractedDocument)
    assert doc.metadata["format"] == "docx"
    assert doc.metadata["paragraph_count"] == 4
    assert "Clinic contact Dr. Alice Smith" in doc.text
    assert "Patient Jane Roe" in doc.text
    assert "MRN A123" in doc.text
    assert "Confidential discharge summary" in doc.text

    for span in doc.spans:
        assert doc.text_for(span) == doc.text[span.start : span.end]
        assert doc.location_at(span.start) == span

    jane = doc.location_at(doc.text.index("Jane"))
    assert jane is not None
    assert doc.text_for(jane) == "Ja"
    assert jane.metadata["part"] == "body"
    assert jane.metadata["run_index"] == 1

    continued = doc.location_at(doc.text.index("Jane") + len("Ja"))
    assert continued is not None
    assert doc.text_for(continued) == "ne"
    assert continued.metadata["run_index"] == 2

    header = doc.location_at(doc.text.index("Alice"))
    assert header is not None
    assert header.metadata["part"] == "header"
    assert header.metadata["block_type"] == "header"

    table = doc.location_at(doc.text.index("A123"))
    assert table is not None
    assert table.metadata["part"] == "body"
    assert table.metadata["block_type"] == "table_cell"
    assert table.metadata["table_index"] == 0
    assert table.metadata["row_index"] == 0
    assert table.metadata["cell_index"] == 0


def test_write_redacted_docx_handles_phi_split_across_runs(
    synthetic_docx: Path, tmp_path: Path
):
    doc = extract_docx(synthetic_docx)
    start = doc.text.index("Jane")
    end = start + len("Jane Roe")
    entities = [{"start": start, "end": end, "label": "PERSON"}]

    redactions = map_text_spans_to_docx_runs(doc, entities)

    assert len(redactions) == 1
    assert redactions[0].replacement == "[PERSON]"
    assert [
        run_range.metadata["run_index"] for run_range in redactions[0].run_ranges
    ] == [1, 2, 3]

    output = tmp_path / "redacted.docx"
    write_redacted_docx(synthetic_docx, output, entities)

    header, body, tables, footer = _docx_text_parts(output)
    assert "Patient [PERSON]" in body
    assert "Jane" not in body
    assert "Roe" not in body
    assert "Alice Smith" in header
    assert "MRN A123" in tables
    assert "Confidential discharge summary" in footer


def test_redact_document_docx_maps_and_writes_header_and_table_entities(
    synthetic_docx: Path,
    tmp_path: Path,
    docx_deps_present,
):
    output = tmp_path / "redacted_header_table.docx"

    def detector(text: str, *, lang: str | None = None):
        assert lang == "en"
        return {
            "entities": [
                {
                    "start": text.index("Alice Smith"),
                    "end": text.index("Alice Smith") + len("Alice Smith"),
                    "label": "PERSON",
                    "confidence": 0.97,
                },
                {
                    "start": text.index("A123"),
                    "end": text.index("A123") + len("A123"),
                    "label": "ID_NUM",
                    "confidence": 0.99,
                },
            ]
        }

    doc = redact_document(
        synthetic_docx,
        models={"detector": detector},
        lang="en",
        policy={"output_path": output},
    )

    assert doc.metadata["detected_span_count"] == 2
    assert doc.metadata["redacted_docx_path"] == str(output)

    redactions = doc.metadata["docx_redactions"]
    header_redaction = next(item for item in redactions if item["label"] == "PERSON")
    table_redaction = next(item for item in redactions if item["label"] == "ID_NUM")
    assert header_redaction["run_ranges"][0]["metadata"]["part"] == "header"
    assert table_redaction["run_ranges"][0]["metadata"]["block_type"] == "table_cell"

    header, body, tables, footer = _docx_text_parts(output)
    assert "[PERSON]" in header
    assert "Alice Smith" not in header
    assert "[ID_NUM]" in tables
    assert "A123" not in tables
    assert "Patient Jane Roe" in body
    assert "Confidential discharge summary" in footer
