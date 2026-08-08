"""Deterministic OCR layout reconstruction for multi-column pages.

The layout parser deliberately operates on the lightweight :class:`OcrResult`
contract and uses only the Python standard library. It clusters words by their
horizontal gaps on each page, groups words with aligned vertical bounds into
line-level blocks, and emits a column-major reading order. Every emitted word
has a character-offset mapping back to its original page and pixel bbox.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from statistics import median
from typing import Any

from .base import ExtractedDocument, SourceSpan
from .ocr import OcrResult, OcrWord

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class LayoutSpan:
    """Map a word's linearized character range to its source geometry."""

    start: int
    end: int
    page: int
    bbox: BBox
    text: str = ""
    column_index: int = 0
    block_index: int = 0
    word_index: int = 0
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def offsets(self) -> tuple[int, int]:
        """Return the half-open character-offset pair."""
        return self.start, self.end

    @property
    def char_offsets(self) -> tuple[int, int]:
        """Alias for :attr:`offsets` used by projection callers."""
        return self.offsets

    def to_source_span(self) -> SourceSpan:
        """Convert this entry to the shared multimodal source-span contract."""
        metadata = dict(self.metadata)
        metadata.setdefault("column_index", self.column_index)
        metadata.setdefault("block_index", self.block_index)
        metadata.setdefault("word_index", self.word_index)
        if self.confidence is not None:
            metadata.setdefault("confidence", self.confidence)
        return SourceSpan(
            start=self.start,
            end=self.end,
            page=self.page,
            bbox=self.bbox,
            metadata=metadata,
        )


# Descriptive aliases make the two directions of the map discoverable without
# duplicating the immutable mapping-entry implementation.
LayoutMapEntry = LayoutSpan
LayoutWordSpan = LayoutSpan


@dataclass(frozen=True)
class LayoutBlock:
    """A line-level OCR block in page reading order."""

    text: str
    words: tuple[OcrWord, ...]
    page: int
    column_index: int
    start: int
    end: int
    bbox: BBox
    index: int = 0
    spans: tuple[LayoutSpan, ...] = ()

    @property
    def column(self) -> int:
        """Return the zero-based column index on :attr:`page`."""
        return self.column_index

    @property
    def word_spans(self) -> tuple[LayoutSpan, ...]:
        """Return the word mappings contained by this block."""
        return self.spans

    @property
    def word_count(self) -> int:
        """Return the number of OCR words in this block."""
        return len(self.words)


@dataclass(frozen=True)
class LayoutColumn:
    """A page column containing ordered line-level blocks."""

    page: int
    index: int
    blocks: tuple[LayoutBlock, ...]
    bbox: BBox

    @property
    def column_index(self) -> int:
        """Return the zero-based column index on :attr:`page`."""
        return self.index

    @property
    def words(self) -> tuple[OcrWord, ...]:
        """Return all column words in reading order."""
        return tuple(word for block in self.blocks for word in block.words)

    @property
    def text(self) -> str:
        """Return the column's linearized block text."""
        return " ".join(block.text for block in self.blocks)

    @property
    def start(self) -> int:
        """Return the first mapped character offset in the column."""
        return self.blocks[0].start

    @property
    def end(self) -> int:
        """Return the exclusive end offset of the column."""
        return self.blocks[-1].end


@dataclass(frozen=True)
class LayoutDocument:
    """Linearized OCR text with columns, blocks, and bidirectional maps."""

    text: str
    columns: tuple[LayoutColumn, ...] = ()
    blocks: tuple[LayoutBlock, ...] = ()
    spans: tuple[LayoutSpan, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reading_order_blocks(self) -> tuple[LayoutBlock, ...]:
        """Return blocks in the exact order used to build :attr:`text`."""
        return self.blocks

    @property
    def word_spans(self) -> tuple[LayoutSpan, ...]:
        """Return one character-to-bbox entry for each OCR word."""
        return self.spans

    @property
    def char_map(self) -> tuple[LayoutSpan, ...]:
        """Return the character-offset-to-bbox side of the mapping."""
        return self.spans

    @property
    def offset_map(self) -> tuple[LayoutSpan, ...]:
        """Alias for :attr:`char_map`."""
        return self.spans

    @property
    def bbox_map(self) -> dict[tuple[int, BBox], tuple[int, int]]:
        """Return exact ``(page, bbox) -> (start, end)`` mappings.

        If identical geometry is supplied for multiple OCR words, the last
        word in reading order wins in this convenience dictionary. Call
        :meth:`offsets_for_bbox` to retrieve every matching word instead.
        """
        return {(span.page, span.bbox): span.offsets for span in self.spans}

    @property
    def char_to_bbox_map(self) -> tuple[LayoutSpan, ...]:
        """Return the character-to-bbox mapping entries."""
        return self.spans

    @property
    def bbox_to_char_map(self) -> dict[tuple[int, BBox], tuple[int, int]]:
        """Return the exact bbox-to-character convenience mapping."""
        return self.bbox_map

    @property
    def page_count(self) -> int:
        """Return the number of pages represented by mapped words."""
        return len({span.page for span in self.spans})

    @property
    def word_count(self) -> int:
        """Return the number of mapped OCR words."""
        return len(self.spans)

    def location_at(self, offset: int) -> LayoutSpan | None:
        """Return the word mapping covering ``offset``, if it is mapped."""
        if offset < 0 or offset >= len(self.text):
            return None
        for span in self.spans:
            if span.start <= offset < span.end:
                return span
        return None

    def spans_for_range(self, start: int, end: int) -> tuple[LayoutSpan, ...]:
        """Return word mappings touched by a half-open character range."""
        _validate_offsets(self.text, start, end)
        if start == end:
            return ()
        return tuple(
            span for span in self.spans if span.start < end and span.end > start
        )

    def bbox_for_span(self, start: int, end: int) -> tuple[LayoutSpan, ...]:
        """Project a character span back to its source word bboxes and pages."""
        return self.spans_for_range(start, end)

    def offsets_for_bbox(
        self,
        page: int,
        bbox: Sequence[float],
    ) -> tuple[tuple[int, int], ...]:
        """Return all character ranges whose source bbox exactly matches.

        ``page`` is the zero-based source page and ``bbox`` is ordered as
        ``(x0, y0, x1, y1)``. Exact matching keeps reverse projection
        deterministic and avoids silently selecting neighboring words.
        """
        target = _coerce_bbox(bbox, index=0)
        return tuple(
            span.offsets
            for span in self.spans
            if span.page == int(page) and span.bbox == target
        )

    def offset_for_bbox(
        self,
        page: int,
        bbox: Sequence[float],
    ) -> tuple[int, int] | None:
        """Return the first exact bbox-to-offset mapping, if present."""
        matches = self.offsets_for_bbox(page, bbox)
        return matches[0] if matches else None

    def bbox_to_offsets(
        self,
        page: int,
        bbox: Sequence[float],
    ) -> tuple[tuple[int, int], ...]:
        """Alias for :meth:`offsets_for_bbox`."""
        return self.offsets_for_bbox(page, bbox)

    def to_document(self) -> ExtractedDocument:
        """Convert the layout result to the shared extraction contract."""
        return ExtractedDocument(
            text=self.text,
            spans=tuple(span.to_source_span() for span in self.spans),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class _WordRecord:
    word: OcrWord
    bbox: BBox
    page: int
    index: int

    @property
    def text(self) -> str:
        return self.word.text

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class _ColumnDraft:
    page: int
    index: int
    blocks: tuple[tuple[_WordRecord, ...], ...]
    bbox: BBox


class FakeLayoutInput:
    """Deterministic in-memory layout input for offline tests.

    This mirrors :class:`~openmed.multimodal.ocr.FakeOcrEngine` while exposing
    the OCR result shape directly to :func:`parse_layout`.
    """

    def __init__(self, words: Iterable[OcrWord], **metadata: Any) -> None:
        self.words = tuple(words)
        self.metadata = {"engine": "fake-layout", **metadata}

    def to_ocr_result(self) -> OcrResult:
        """Return the deterministic OCR result represented by this input."""
        return OcrResult(words=self.words, metadata=dict(self.metadata))

    @property
    def result(self) -> OcrResult:
        """Return the input as an OCR result for concise test setup."""
        return self.to_ocr_result()


class FakeLayoutEngine(FakeLayoutInput):
    """Deterministic fake engine returning a fixed OCR layout input."""

    name = "fake-layout"

    def recognize(
        self,
        image: Any = None,
        *,
        languages: Sequence[str] | None = None,
    ) -> OcrResult:
        """Return fixed words and record the requested languages."""
        del image
        metadata = dict(self.metadata)
        metadata["languages"] = list(languages) if languages is not None else None
        return OcrResult(words=self.words, metadata=metadata)


def parse_layout(
    ocr_result: OcrResult | FakeLayoutInput,
    *,
    separator: str = " ",
    column_gap: float | None = None,
    line_tolerance: float | None = None,
) -> LayoutDocument:
    """Reconstruct deterministic reading order from positioned OCR words.

    Args:
        ocr_result: An :class:`OcrResult` or compatible object exposing
            ``words`` and optional ``metadata`` attributes.
        separator: Text inserted between ordered OCR words and blocks.
        column_gap: Optional absolute x-gap threshold. When omitted, the
            threshold is inferred independently for each page.
        line_tolerance: Optional vertical-center tolerance used to group words
            into line-level blocks.

    Returns:
        A :class:`LayoutDocument` whose columns are ordered left-to-right per
        page, whose blocks are ordered top-to-bottom within each column, and
        whose spans map every emitted word in both directions.

    Raises:
        TypeError: If the input or separator does not have the expected shape.
        ValueError: If an OCR word has invalid page or bbox geometry.
    """
    if not isinstance(separator, str):
        raise TypeError("separator must be a string")
    if column_gap is not None and column_gap < 0:
        raise ValueError("column_gap must be non-negative")
    if line_tolerance is not None and line_tolerance < 0:
        raise ValueError("line_tolerance must be non-negative")

    records = _coerce_records(ocr_result)
    source_metadata = getattr(ocr_result, "metadata", {})
    metadata = dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
    if not records:
        metadata.update(
            {
                "format": "ocr_layout",
                "page_count": 0,
                "column_count": 0,
                "block_count": 0,
                "word_count": 0,
            }
        )
        return LayoutDocument(text="", metadata=metadata)

    page_records: dict[int, list[_WordRecord]] = {}
    for record in records:
        page_records.setdefault(record.page, []).append(record)

    drafts: list[_ColumnDraft] = []
    for page in sorted(page_records):
        records_for_page = tuple(page_records[page])
        tolerance = line_tolerance
        if tolerance is None:
            tolerance = _default_line_tolerance(records_for_page)
        column_groups = _cluster_columns(
            records_for_page,
            column_gap=column_gap,
            line_tolerance=tolerance,
        )
        for column_index, group in enumerate(column_groups):
            lines = _group_lines(group, tolerance)
            drafts.append(
                _ColumnDraft(
                    page=page,
                    index=column_index,
                    blocks=lines,
                    bbox=_union_bbox(record.bbox for record in group),
                )
            )

    text_parts: list[str] = []
    spans: list[LayoutSpan] = []
    blocks: list[LayoutBlock] = []
    columns: list[LayoutColumn] = []
    cursor = 0

    for draft in drafts:
        column_blocks: list[LayoutBlock] = []
        for line in draft.blocks:
            block_index = len(blocks)
            block_spans: list[LayoutSpan] = []
            block_start: int | None = None
            for record in line:
                if cursor:
                    text_parts.append(separator)
                    cursor += len(separator)
                start = cursor
                text_parts.append(record.text)
                cursor += len(record.text)
                if block_start is None:
                    block_start = start
                span = LayoutSpan(
                    start=start,
                    end=cursor,
                    page=record.page,
                    bbox=record.bbox,
                    text=record.text,
                    column_index=draft.index,
                    block_index=block_index,
                    word_index=record.index,
                    confidence=record.word.confidence,
                    metadata={"block_type": "line"},
                )
                spans.append(span)
                block_spans.append(span)

            if block_start is None:
                continue
            block = LayoutBlock(
                text=separator.join(record.text for record in line),
                words=tuple(record.word for record in line),
                page=draft.page,
                column_index=draft.index,
                start=block_start,
                end=cursor,
                bbox=_union_bbox(record.bbox for record in line),
                index=block_index,
                spans=tuple(block_spans),
            )
            blocks.append(block)
            column_blocks.append(block)

        if column_blocks:
            columns.append(
                LayoutColumn(
                    page=draft.page,
                    index=draft.index,
                    blocks=tuple(column_blocks),
                    bbox=draft.bbox,
                )
            )

    metadata.update(
        {
            "format": "ocr_layout",
            "page_count": len(page_records),
            "column_count": len(columns),
            "block_count": len(blocks),
            "word_count": len(spans),
            "separator": separator,
        }
    )
    return LayoutDocument(
        text="".join(text_parts),
        columns=tuple(columns),
        blocks=tuple(blocks),
        spans=tuple(spans),
        metadata=metadata,
    )


def _coerce_records(ocr_result: Any) -> tuple[_WordRecord, ...]:
    raw_words = getattr(ocr_result, "words", None)
    if raw_words is None:
        raise TypeError("ocr_result must expose a words iterable")

    records: list[_WordRecord] = []
    for index, raw_word in enumerate(raw_words):
        text = str(_word_value(raw_word, "text", "")).strip()
        if not text:
            continue
        bbox = _coerce_bbox(_word_value(raw_word, "bbox", None), index=index)
        try:
            page = int(_word_value(raw_word, "page", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"OCR word {index} has an invalid page") from exc
        if page < 0:
            raise ValueError(f"OCR word {index} has an invalid page")
        try:
            confidence = float(_word_value(raw_word, "confidence", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"OCR word {index} has an invalid confidence") from exc
        if not isfinite(confidence):
            raise ValueError(f"OCR word {index} has an invalid confidence")
        word = (
            raw_word
            if isinstance(raw_word, OcrWord)
            else OcrWord(text=text, bbox=bbox, confidence=confidence, page=page)
        )
        records.append(_WordRecord(word=word, bbox=bbox, page=page, index=index))
    return tuple(records)


def _word_value(word: Any, name: str, default: Any) -> Any:
    if isinstance(word, Mapping):
        return word.get(name, default)
    return getattr(word, name, default)


def _coerce_bbox(value: Any, *, index: int) -> BBox:
    if isinstance(value, Mapping):
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            value = tuple(value[key] for key in ("x0", "y0", "x1", "y1"))
        elif all(key in value for key in ("left", "top", "right", "bottom")):
            value = tuple(value[key] for key in ("left", "top", "right", "bottom"))
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        raise ValueError(f"OCR word {index} has an invalid bbox")
    try:
        values = tuple(float(number) for number in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OCR word {index} has an invalid bbox") from exc
    if len(values) != 4 or not all(isfinite(number) for number in values):
        raise ValueError(f"OCR word {index} has an invalid bbox")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"OCR word {index} has an invalid bbox")
    return values  # type: ignore[return-value]


def _default_line_tolerance(records: Sequence[_WordRecord]) -> float:
    heights = [record.y1 - record.y0 for record in records]
    return max(median(heights) * 0.6, 0.5)


def _cluster_columns(
    records: Sequence[_WordRecord],
    *,
    column_gap: float | None,
    line_tolerance: float,
) -> tuple[tuple[_WordRecord, ...], ...]:
    if len(records) <= 1:
        return (tuple(records),)

    sorted_by_x = sorted(
        records, key=lambda record: (record.x0, record.x1, record.index)
    )
    widths = [record.x1 - record.x0 for record in records]
    page_left = min(record.x0 for record in records)
    page_right = max(record.x1 for record in records)
    page_width = max(page_right - page_left, 1.0)
    typical_gap = _typical_line_gap(records, line_tolerance, page_width)
    threshold = (
        float(column_gap)
        if column_gap is not None
        else max(median(widths) * 1.5, page_width * 0.05)
    )
    if typical_gap > 0 and typical_gap < page_width * 0.05:
        threshold = max(threshold, typical_gap * 3.0)

    groups: list[list[_WordRecord]] = [[]]
    previous = sorted_by_x[0]
    groups[0].append(previous)
    for record in sorted_by_x[1:]:
        gap = max(0.0, record.x0 - previous.x1)
        if gap > threshold:
            groups.append([])
        groups[-1].append(record)
        previous = record

    ordered_groups = sorted(
        (tuple(group) for group in groups if group),
        key=lambda group: (
            min(record.x0 for record in group),
            min(record.index for record in group),
        ),
    )
    return tuple(ordered_groups)


def _typical_line_gap(
    records: Sequence[_WordRecord],
    line_tolerance: float,
    page_width: float,
) -> float:
    gaps: list[float] = []
    for line in _group_lines(records, line_tolerance):
        ordered = sorted(line, key=lambda record: (record.x0, record.index))
        for previous, current in zip(ordered, ordered[1:]):
            gap = max(0.0, current.x0 - previous.x1)
            if 0 < gap < page_width * 0.05:
                gaps.append(gap)
    return median(gaps) if gaps else 0.0


def _group_lines(
    records: Sequence[_WordRecord],
    tolerance: float,
) -> tuple[tuple[_WordRecord, ...], ...]:
    lines: list[list[_WordRecord]] = []
    for record in sorted(records, key=lambda item: (item.y0, item.x0, item.index)):
        candidates = [
            (line_index, line)
            for line_index, line in enumerate(lines)
            if _same_line(record, line, tolerance)
        ]
        if not candidates:
            lines.append([record])
            continue
        line_index, _ = min(
            candidates,
            key=lambda item: abs(record.center_y - _line_center(item[1])),
        )
        lines[line_index].append(record)

    ordered_lines = sorted(
        lines,
        key=lambda line: (
            min(record.y0 for record in line),
            min(record.x0 for record in line),
            min(record.index for record in line),
        ),
    )
    return tuple(
        tuple(sorted(line, key=lambda record: (record.x0, record.y0, record.index)))
        for line in ordered_lines
    )


def _same_line(
    record: _WordRecord, line: Sequence[_WordRecord], tolerance: float
) -> bool:
    line_top = min(item.y0 for item in line)
    line_bottom = max(item.y1 for item in line)
    vertical_overlap = min(record.y1, line_bottom) - max(record.y0, line_top)
    return (
        vertical_overlap > 0 or abs(record.center_y - _line_center(line)) <= tolerance
    )


def _line_center(line: Sequence[_WordRecord]) -> float:
    return median([record.center_y for record in line])


def _union_bbox(boxes: Iterable[BBox]) -> BBox:
    values = tuple(boxes)
    if not values:
        raise ValueError("cannot compute a bbox for an empty layout group")
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def _validate_offsets(text: str, start: int, end: int) -> None:
    if start < 0 or end < start or end > len(text):
        raise ValueError("character offsets are outside the layout document")


__all__ = [
    "BBox",
    "FakeLayoutEngine",
    "FakeLayoutInput",
    "LayoutBlock",
    "LayoutColumn",
    "LayoutDocument",
    "LayoutMapEntry",
    "LayoutSpan",
    "LayoutWordSpan",
    "parse_layout",
]
