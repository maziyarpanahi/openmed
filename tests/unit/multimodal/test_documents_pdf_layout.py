"""Synthetic regression tests for multi-column PDF reading order."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed.multimodal.base as base
from openmed.multimodal import (
    detect_pdf_columns,
    extract_pdf,
    project_text_spans,
    reconstruct_pdf_reading_order,
    redact_document,
)


def _word(text: str, x0: float, top: float, x1: float) -> dict[str, str | float]:
    return {
        "text": text,
        "x0": x0,
        "top": top,
        "x1": x1,
        "bottom": top + 10.0,
    }


def _two_column_words() -> tuple[dict[str, str | float], ...]:
    """Return row-interleaved source words from two physical columns."""

    return (
        _word("Patient", 40, 20, 80),
        _word("Avery", 84, 20, 115),
        _word("Sample", 119, 20, 160),
        _word("Study", 330, 20, 360),
        _word("Cardiology", 364, 20, 420),
        _word("Address", 40, 42, 80),
        _word("12", 84, 42, 96),
        _word("Synthetic", 100, 42, 150),
        _word("Lane", 154, 42, 180),
        _word("Finding", 330, 42, 370),
        _word("Stable", 374, 42, 405),
    )


class _FakePage:
    width = 612.0

    def __init__(self, words):
        self._words = tuple(words)

    def extract_words(self, **_kwargs):
        return list(self._words)


class _FakePdf:
    def __init__(self, pages):
        self.pages = tuple(pages)

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False


@pytest.fixture
def fake_two_column_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(_two_column_words())
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: _FakePdf((page,))),
    )


def test_detects_two_columns_and_returns_column_major_indexes() -> None:
    words = _two_column_words()

    layout = detect_pdf_columns(words, page_width=612)

    assert layout.is_multicolumn
    assert layout.column_count == 2
    assert layout.reading_order == (0, 1, 2, 5, 6, 7, 8, 3, 4, 9, 10)
    assert [column.word_count for column in layout.columns] == [7, 4]
    assert len(layout.column_boundaries) == 1
    assert 180 < layout.column_boundaries[0] < 330
    assert reconstruct_pdf_reading_order(words) == detect_pdf_columns(words)


def test_detects_three_columns_without_interleaving_rows() -> None:
    words = (
        _word("left-one", 40, 20, 90),
        _word("middle-one", 240, 20, 300),
        _word("right-one", 440, 20, 495),
        _word("left-two", 40, 42, 90),
        _word("middle-two", 240, 42, 300),
        _word("right-two", 440, 42, 495),
    )

    layout = detect_pdf_columns(words, page_width=612)

    assert layout.column_count == 3
    assert layout.reading_order == (0, 3, 1, 4, 2, 5)
    assert layout.word_columns == (0, 1, 2, 0, 1, 2)


def test_full_width_heading_stays_before_column_major_body() -> None:
    words = (
        _word("Synthetic", 80, 5, 170),
        _word("Clinical", 174, 5, 265),
        _word("Report", 269, 5, 360),
        _word("left-one", 40, 25, 100),
        _word("right-one", 330, 25, 400),
        _word("left-two", 40, 45, 100),
        _word("right-two", 330, 45, 400),
    )

    layout = detect_pdf_columns(words, page_width=612)

    assert layout.reading_order == (0, 1, 2, 3, 5, 4, 6)
    assert layout.word_columns[:3] == (None, None, None)


def test_extract_pdf_reconstructs_contiguous_phi_and_preserves_bboxes(
    fake_two_column_pdf: None,
) -> None:
    document = extract_pdf("synthetic_phi_twocol.pdf")

    assert document.text == (
        "Patient Avery Sample Address 12 Synthetic Lane Study Cardiology Finding Stable"
    )
    assert "Avery Sample Address 12 Synthetic Lane" in document.text
    assert document.metadata["reading_order"] == "column-major"
    assert document.metadata["reconstructed_page_count"] == 1
    assert document.metadata["page_layouts"][0]["column_count"] == 2
    assert "Avery" not in repr(document.metadata)
    assert document == extract_pdf("synthetic_phi_twocol.pdf")

    source_words = _two_column_words()
    assert len(document.spans) == len(source_words)
    for span in document.spans:
        source_index = span.metadata["source_page_word_index"]
        source = source_words[source_index]
        assert document.text_for(span) == source["text"]
        assert span.bbox == (
            source["x0"],
            source["top"],
            source["x1"],
            source["bottom"],
        )

    avery = document.location_at(document.text.index("Avery"))
    assert avery is not None
    assert avery.bbox == (84.0, 20.0, 115.0, 30.0)
    assert avery.metadata["source_page_word_index"] == 1
    assert avery.metadata["column_index"] == 0

    start = document.text.index("Avery")
    end = document.text.index("Sample") + len("Sample")
    rectangles = project_text_spans(document, ((start, end),))
    assert len(rectangles) == 1
    assert rectangles[0].bbox == (84.0, 20.0, 160.0, 30.0)


def test_source_mode_retains_interleaved_om060_order(
    fake_two_column_pdf: None,
) -> None:
    document = extract_pdf("synthetic_phi_twocol.pdf", reading_order="source")

    assert document.text == (
        "Patient Avery Sample Study Cardiology Address 12 Synthetic Lane Finding Stable"
    )
    assert "reading_order_reconstructed" not in document.metadata
    assert "source_page_word_index" not in document.spans[0].metadata


def test_single_column_auto_output_is_identical_to_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = (
        _word("Second", 40, 42, 80),
        _word("line", 84, 42, 105),
        _word("First", 40, 20, 68),
        _word("line", 72, 20, 93),
    )
    page = _FakePage(words)
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: _FakePdf((page,))),
    )

    automatic = extract_pdf("single.pdf")
    source = extract_pdf("single.pdf", reading_order="source")

    assert automatic == source
    assert automatic.text == "Second line First line"


def test_redact_document_detects_on_reconstructed_text_and_projects_address(
    fake_two_column_pdf: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])
    observed: list[str] = []

    def detector(text: str, *, lang: str | None = None):
        observed.append(text)
        start = text.index("12 Synthetic Lane")
        return {
            "entities": [
                {
                    "start": start,
                    "end": start + len("12 Synthetic Lane"),
                    "label": "ADDRESS",
                }
            ]
        }

    document = redact_document(
        "synthetic_phi_twocol.pdf",
        models={"detector": detector},
    )

    assert observed == [document.text]
    assert "Avery Sample Address 12 Synthetic Lane" in observed[0]
    rectangles = document.metadata["redaction_rectangles"]
    assert len(rectangles) == 1
    assert rectangles[0]["bbox"] == (84.0, 42.0, 180.0, 52.0)
    assert rectangles[0]["label"] == "ADDRESS"


def test_auto_mode_does_not_treat_a_small_two_column_table_as_page_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = (
        _word("Figure", 10, 20, 45),
        _word("1:", 48, 20, 60),
        _word("Synthetic", 63, 20, 112),
        _word("Table", 10, 40, 42),
        _word("1:", 45, 40, 57),
        _word("Results", 60, 40, 104),
        _word("Patient", 15, 63, 55),
        _word("ID", 85, 63, 98),
        _word("Name", 15, 78, 48),
        _word("Synthetic", 85, 78, 135),
    )
    page = _FakePage(words)
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: _FakePdf((page,))),
    )

    assert extract_pdf("table.pdf") == extract_pdf("table.pdf", reading_order="source")


def test_auto_mode_does_not_reorder_a_balanced_narrow_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = tuple(
        word
        for row, top in enumerate((20.0, 40.0, 60.0, 80.0))
        for word in (
            _word(f"key-{row}", 15, top, 55),
            _word(f"value-{row}", 100, top, 145),
        )
    )
    page = _FakePage(words)
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: _FakePdf((page,))),
    )

    automatic = extract_pdf("balanced-table.pdf")
    source = extract_pdf("balanced-table.pdf", reading_order="source")

    assert automatic == source
    assert automatic.text == ("key-0 value-0 key-1 value-1 key-2 value-2 key-3 value-3")


def test_real_synthetic_fixture_preserves_column_geometry() -> None:
    pytest.importorskip("pdfplumber")
    fixture = Path(__file__).with_name("fixtures") / "synthetic_phi_twocol.pdf"

    automatic = extract_pdf(fixture)
    source = extract_pdf(fixture, reading_order="source")

    assert automatic.metadata["reading_order_reconstructed"] is True
    assert automatic.text.index("Address 12 Synthetic Lane") < automatic.text.index(
        "Study Cardiology"
    )
    assert source.text.index("Study Cardiology") < source.text.index(
        "Address 12 Synthetic Lane"
    )
    address_start = automatic.text.index("12 Synthetic Lane")
    address_end = address_start + len("12 Synthetic Lane")
    boxes = project_text_spans(automatic, ((address_start, address_end),))
    assert len(boxes) == 1
    assert boxes[0].bbox[0] < 200


def test_extract_pdf_rejects_unknown_reading_order(fake_two_column_pdf: None) -> None:
    with pytest.raises(ValueError, match="reading_order"):
        extract_pdf("synthetic_phi_twocol.pdf", reading_order="rows")  # type: ignore[arg-type]
