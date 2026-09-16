"""Synthetic offline tests for OCR multi-column layout reconstruction."""

from __future__ import annotations

from openmed.multimodal import (
    FakeLayoutEngine,
    LayoutDocument,
    OcrResult,
    OcrWord,
    parse_layout,
)


def _synthetic_words() -> tuple[OcrWord, ...]:
    """Return deliberately shuffled two-column geometry."""
    words = (
        OcrWord("Left", (20.0, 20.0, 50.0, 30.0), 0.99, page=0),
        OcrWord("column", (56.0, 20.0, 104.0, 30.0), 0.98, page=0),
        OcrWord("second", (20.0, 46.0, 62.0, 56.0), 0.97, page=0),
        OcrWord("line", (68.0, 46.0, 94.0, 56.0), 0.96, page=0),
        OcrWord("Right", (320.0, 20.0, 352.0, 30.0), 0.95, page=0),
        OcrWord("column", (358.0, 20.0, 406.0, 30.0), 0.94, page=0),
        OcrWord("second", (320.0, 46.0, 362.0, 56.0), 0.93, page=0),
        OcrWord("line", (368.0, 46.0, 394.0, 56.0), 0.92, page=0),
        OcrWord("Page", (20.0, 20.0, 48.0, 30.0), 0.91, page=1),
        OcrWord("two", (54.0, 20.0, 82.0, 30.0), 0.90, page=1),
    )
    return (
        words[6],
        words[4],
        words[9],
        words[2],
        words[7],
        words[0],
        words[5],
        words[8],
        words[3],
        words[1],
    )


def test_parse_layout_reconstructs_column_major_reading_order() -> None:
    result = OcrResult(words=_synthetic_words(), metadata={"engine": "fake"})

    document = parse_layout(result)

    assert isinstance(document, LayoutDocument)
    assert len(document.columns) == 3
    assert [column.page for column in document.columns] == [0, 0, 1]
    assert [column.index for column in document.columns] == [0, 1, 0]
    assert document.text.split() == [
        "Left",
        "column",
        "second",
        "line",
        "Right",
        "column",
        "second",
        "line",
        "Page",
        "two",
    ]
    assert [block.page for block in document.blocks] == [0, 0, 0, 0, 1]
    assert [block.column_index for block in document.blocks] == [0, 0, 1, 1, 0]
    assert document.metadata["column_count"] == 3


def test_fake_layout_engine_and_ocr_result_helper_are_offline() -> None:
    engine = FakeLayoutEngine(_synthetic_words(), source="synthetic")

    from_engine = parse_layout(engine.recognize("ignored", languages=["en"]))
    from_helper = engine.recognize("ignored").to_layout()

    assert from_engine.text == from_helper.text
    assert from_engine.metadata["engine"] == "fake-layout"
    assert from_engine.metadata["languages"] == ["en"]


def test_layout_char_bbox_map_round_trips_page_and_offsets() -> None:
    document = parse_layout(OcrResult(words=_synthetic_words()))
    target = next(
        span
        for span in document.spans
        if span.text == "second" and span.column_index == 1
    )

    assert document.location_at(target.start) == target
    assert document.text[target.start : target.end] == "second"
    assert document.bbox_for_span(target.start, target.end) == (target,)
    assert document.offsets_for_bbox(target.page, target.bbox) == (target.offsets,)
    assert document.offset_for_bbox(target.page, target.bbox) == target.offsets
    assert document.bbox_map[(target.page, target.bbox)] == target.offsets

    extracted = document.to_document()
    source_span = extracted.location_at(target.start)
    assert source_span is not None
    assert source_span.page == target.page
    assert source_span.bbox == target.bbox
