"""Synthetic offline tests for OCR multi-column layout reconstruction."""

from __future__ import annotations

import pytest

from openmed.multimodal import (
    FakeLayoutEngine,
    LayoutDocument,
    OcrResult,
    OcrWord,
    evaluate_layout,
    parse_layout,
    transform_ocr_result,
)
from openmed.multimodal.box_normalization import PageSize
from openmed.multimodal.page_rotation import PageSize as RotationPageSize


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


def _clinical_page() -> OcrResult:
    rows = (
        ("Synthetic", 20, 20, 85, 30),
        ("report", 91, 20, 133, 30),
        ("Medications:", 30, 120, 112, 132),
        ("Aspirin", 30, 150, 74, 162),
        ("daily", 80, 150, 110, 162),
        ("Allergies:", 280, 120, 345, 132),
        ("None", 280, 150, 312, 162),
        ("known", 318, 150, 355, 162),
        ("Test", 30, 350, 60, 362),
        ("Result", 200, 350, 243, 362),
        ("Reference", 350, 350, 416, 362),
        ("Hemoglobin", 30, 380, 102, 392),
        ("13", 200, 380, 215, 392),
        ("12-16", 350, 380, 386, 392),
        ("Sodium", 30, 410, 76, 422),
        ("140", 200, 410, 222, 422),
        ("135-145", 350, 410, 400, 422),
        ("Page", 30, 650, 59, 662),
        ("1", 65, 650, 72, 662),
    )
    words = tuple(
        OcrWord(text, (x0, y0, x1, y1), 0.99 - index * 0.01)
        for index, (text, x0, y0, x1, y1) in enumerate(rows)
    )
    return OcrResult(
        words=tuple(reversed(words)),
        metadata={"page_dimensions": {0: (500, 700)}, "source": "synthetic"},
    )


def test_layout_reconstructs_headers_columns_and_table_cells() -> None:
    result = _clinical_page()
    document = parse_layout(result)

    assert len(document.headers) == len(document.footers) == 1
    assert len(document.columns) == 2
    assert len(document.tables) == 1
    assert document.metadata["table_count"] == 1
    assert [len(row) for row in document.tables[0].rows] == [3, 3, 3]
    structured = document.tables[0].as_structured_table()
    assert structured["n_rows"] == structured["n_columns"] == 3
    assert len(structured["cells"]) == 9
    assert all(
        document.text[cell["start"] : cell["end"]] == cell["text"]
        for cell in structured["cells"]
    )
    assert document.text.split() == [
        "Synthetic",
        "report",
        "Medications:",
        "Aspirin",
        "daily",
        "Allergies:",
        "None",
        "known",
        "Test",
        "Result",
        "Reference",
        "Hemoglobin",
        "13",
        "12-16",
        "Sodium",
        "140",
        "135-145",
        "Page",
        "1",
    ]
    for table_row in document.tables[0].rows:
        for cell in table_row:
            assert document.text[cell.start : cell.end] == cell.text
            for span in cell.spans:
                assert document.offsets_for_bbox(span.page, span.bbox) == (
                    span.offsets,
                )


def test_layout_sections_and_rotated_source_boxes_round_trip() -> None:
    result = _clinical_page()
    document = parse_layout(result)
    sections = document.detect_sections()

    assert sections[0].start == 0
    assert sections[-1].end == len(document.text)
    assert all(left.end == right.start for left, right in zip(sections, sections[1:]))
    assert {section.label for section in sections} >= {"medications", "allergies"}

    rotated = transform_ocr_result(result, (500, 700), 90)
    restored = transform_ocr_result(rotated, (700, 500), 270)
    assert [word.bbox for word in restored.words] == [
        word.bbox for word in result.words
    ]
    assert parse_layout(restored).text == document.text


def test_isolated_bands_are_detected_without_page_dimensions() -> None:
    result = _clinical_page()
    with_dimensions = parse_layout(result)
    inferred = parse_layout(OcrResult(words=result.words))

    assert len(inferred.headers) == len(inferred.footers) == 1
    assert len(inferred.columns) == 2
    assert len(inferred.tables) == 1
    assert inferred.text == with_dimensions.text


@pytest.mark.parametrize("size", [PageSize(500, 700), RotationPageSize(500, 700)])
def test_page_size_contracts_keep_layout_geometry(size: object) -> None:
    result = _clinical_page()
    sized = parse_layout(
        OcrResult(words=result.words, metadata={"page_dimensions": {0: size}})
    )

    assert sized.text == parse_layout(result).text
    assert len(sized.headers) == len(sized.footers) == 1


def test_synthetic_layout_quality_exceeds_acceptance_thresholds() -> None:
    result = _clinical_page()
    document = parse_layout(result)
    expected_cells = {
        len(result.words) - 1 - original_index: (0, row, column)
        for row in range(3)
        for column in range(3)
        for original_index in (8 + row * 3 + column,)
    }

    quality = evaluate_layout(
        document,
        expected_word_order=tuple(reversed(range(len(result.words)))),
        expected_cells=expected_cells,
    )

    assert quality.expected_words == 19
    assert quality.expected_cells == 9
    assert quality.reading_order_accuracy >= 0.90
    assert quality.table_cell_accuracy >= 0.85
    assert quality == evaluate_layout(
        parse_layout(result),
        expected_word_order=tuple(reversed(range(len(result.words)))),
        expected_cells=expected_cells,
    )


def test_layout_preserves_low_confidence_and_rejects_unbounded_pixels() -> None:
    result = _clinical_page()
    words = list(result.words)
    words[0] = OcrWord(words[0].text, words[0].bbox, 0.05)
    document = parse_layout(OcrResult(words=tuple(words), metadata=result.metadata))
    assert (
        next(span for span in document.spans if span.word_index == 0).confidence == 0.05
    )

    bad = OcrResult(
        words=(OcrWord("SYNTHETIC-SENSITIVE-SENTINEL", (490, 20, 510, 30), 0.9),),
        metadata={"page_dimensions": {0: (500, 700)}},
    )
    with pytest.raises(ValueError) as exc:
        parse_layout(bad)
    assert "SYNTHETIC-SENSITIVE-SENTINEL" not in str(exc.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_layout_rejects_ambiguous_grouping_tolerance(value: float) -> None:
    with pytest.raises(ValueError, match="column_gap"):
        parse_layout(_clinical_page(), column_gap=value)


def test_layout_rejects_fractional_page_and_negative_box_without_text() -> None:
    for word in (
        {"text": "SYNTHETIC-SENSITIVE-SENTINEL", "bbox": (1, 1, 5, 5), "page": 0.5},
        {"text": "SYNTHETIC-SENSITIVE-SENTINEL", "bbox": (-1, 1, 5, 5), "page": 0},
    ):
        with pytest.raises(ValueError) as exc:
            parse_layout(OcrResult(words=(word,)))
        assert "SYNTHETIC-SENSITIVE-SENTINEL" not in str(exc.value)
