"""Tests for structured PDF table and caption extraction."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed.multimodal.base as base
from openmed.multimodal import (
    CaptionRegion,
    TableRegion,
    extract_pdf,
    extract_pdf_regions,
    extract_pdf_tables,
    project_structured_spans,
    redact_document,
)

_TABLE_BBOX = (10.0, 60.0, 170.0, 90.0)
_CELL_BBOXES = (
    ((10.0, 60.0, 80.0, 75.0), (80.0, 60.0, 170.0, 75.0)),
    ((10.0, 75.0, 80.0, 90.0), (80.0, 75.0, 170.0, 90.0)),
)


class _FakePage:
    def __init__(self, words, tables=()):
        self._words = words
        self._tables = tables

    def extract_words(self, **kwargs):
        return list(self._words)

    def find_tables(self):
        return list(self._tables)


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _word(text: str, x0: float, top: float, x1: float) -> dict[str, object]:
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": top + 10}


def _structured_page() -> _FakePage:
    words = [
        _word("Figure", 10, 20, 45),
        _word("1:", 48, 20, 60),
        _word("MRI", 63, 20, 85),
        _word("of", 88, 20, 98),
        _word("Synthetic", 101, 20, 150),
        _word("Table", 10, 40, 42),
        _word("1:", 45, 40, 57),
        _word("Synthetic", 60, 40, 110),
        _word("results", 113, 40, 150),
        _word("Patient", 15, 63, 55),
        _word("ID", 85, 63, 98),
        _word("Name", 15, 78, 48),
        _word("Synthetic", 85, 78, 135),
    ]
    table = SimpleNamespace(
        bbox=_TABLE_BBOX,
        rows=[
            SimpleNamespace(cells=_CELL_BBOXES[0]),
            SimpleNamespace(cells=_CELL_BBOXES[1]),
        ],
        extract=lambda: [["Patient", "ID"], ["Name", "Synthetic"]],
    )
    return _FakePage(words, tables=[table])


@pytest.fixture
def fake_structured_pdf(monkeypatch):
    page = _structured_page()
    module = SimpleNamespace(open=lambda path: _FakePdf([page]))
    monkeypatch.setitem(sys.modules, "pdfplumber", module)
    return page


@pytest.fixture
def fake_plain_pdf(monkeypatch):
    page = _FakePage([_word("Plain", 10, 20, 35), _word("page", 40, 20, 65)])
    module = SimpleNamespace(open=lambda path: _FakePdf([page]))
    monkeypatch.setitem(sys.modules, "pdfplumber", module)
    return page


def test_extract_pdf_tables_maps_cells_and_caption_offsets(fake_structured_pdf):
    document = extract_pdf("synthetic_table.pdf")

    regions = extract_pdf_regions("synthetic_table.pdf", document=document)

    assert len(regions.tables) == 1
    table = regions.tables[0]
    assert isinstance(table, TableRegion)
    assert table.bbox == _TABLE_BBOX
    assert [cell.text for cell in table.cells] == [
        "Patient",
        "ID",
        "Name",
        "Synthetic",
    ]
    synthetic = table.cells[-1]
    assert synthetic.offset_range is not None
    assert document.text[synthetic.start : synthetic.end] == "Synthetic"
    assert synthetic.bbox == _CELL_BBOXES[1][1]

    assert [caption.kind for caption in regions.captions] == ["figure", "table"]
    caption = regions.captions[0]
    assert isinstance(caption, CaptionRegion)
    assert caption.kind == "figure"
    assert document.text[caption.start : caption.end] == "Figure 1: MRI of Synthetic"
    assert caption.bbox == (10.0, 20.0, 150.0, 30.0)

    assert (
        extract_pdf_tables("synthetic_table.pdf", document=document) == regions.tables
    )


def test_project_structured_span_uses_cell_and_caption_bboxes(fake_structured_pdf):
    document = extract_pdf("synthetic_table.pdf")
    regions = extract_pdf_regions("synthetic_table.pdf", document=document)
    cell_start = document.text.index("Synthetic", document.text.index("Name"))
    caption_start = document.text.index("Synthetic")

    rectangles = project_structured_spans(
        document,
        regions,
        [
            {
                "start": cell_start,
                "end": cell_start + len("Synthetic"),
                "label": "PERSON",
            },
            {
                "start": caption_start,
                "end": caption_start + len("Synthetic"),
                "label": "PERSON",
            },
        ],
    )

    assert len(rectangles) == 2
    assert rectangles[0].bbox == _CELL_BBOXES[1][1]
    assert rectangles[0].metadata["region_type"] == "table_cell"
    assert rectangles[1].bbox == (10.0, 20.0, 150.0, 30.0)
    assert rectangles[1].metadata["region_type"] == "caption"


def test_redact_document_exposes_structured_regions_and_boxes(
    fake_structured_pdf, monkeypatch
):
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])

    def detector(text, *, lang=None):
        start = text.index("Synthetic", text.index("Name"))
        return {"entities": [{"start": start, "end": start + 9, "label": "PERSON"}]}

    document = redact_document(
        "synthetic_table.pdf",
        models={"detector": detector},
    )

    assert len(document.metadata["table_regions"]) == 1
    assert len(document.metadata["caption_regions"]) == 2
    rectangle = document.metadata["redaction_rectangles"][0]
    assert rectangle["bbox"] == _CELL_BBOXES[1][1]
    assert rectangle["metadata"]["region_type"] == "table_cell"


def test_pages_without_tables_or_captions_return_empty_regions(fake_plain_pdf):
    document = extract_pdf("plain.pdf")

    regions = extract_pdf_regions("plain.pdf", document=document)

    assert regions.tables == ()
    assert regions.captions == ()


def test_synthetic_pdf_fixture_extracts_tables_and_caption():
    pdfplumber = pytest.importorskip("pdfplumber")
    del pdfplumber
    fixture = Path(__file__).with_name("fixtures") / "synthetic_phi_table.pdf"

    document = extract_pdf(fixture)
    regions = extract_pdf_regions(fixture, document=document)

    assert regions.tables
    assert regions.tables[0].cells
    assert regions.captions
    assert regions.captions[0].kind == "figure"
