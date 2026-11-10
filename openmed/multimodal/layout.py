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
from functools import wraps
from itertools import islice
from math import isfinite
from statistics import median
from typing import Any

from .base import ExtractedDocument, SourceSpan
from .ocr import OcrResult, OcrWord

BBox = tuple[float, float, float, float]


def _safe_boundary(function):
    """Expose stable layout failures without external exception context."""

    @wraps(function)
    def checked(*args, **kwargs):
        error_type, message = ValueError, "invalid OCR layout input"
        try:
            return function(*args, **kwargs)
        except TypeError:
            error_type = TypeError
        except ValueError as error:
            if str(error) in {
                "column_gap must be finite and non-negative",
                "line_tolerance must be finite and non-negative",
                "character offsets are outside the layout document",
            }:
                message = str(error)
        except Exception:
            pass
        raise error_type(message)

    return checked


def _bounded(values, limit=4096):
    result = tuple(islice(iter(values), limit + 1))
    if len(result) > limit:
        raise ValueError("layout input limit exceeded")
    return result


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
class LayoutBand:
    """A page header or footer with source-linked reading-order blocks."""

    page: int
    kind: str
    blocks: tuple[LayoutBlock, ...]
    bbox: BBox


@dataclass(frozen=True)
class LayoutTableCell:
    """One reconstructed table cell with exact text and source offsets."""

    row: int
    column: int
    text: str
    start: int
    end: int
    bbox: BBox
    spans: tuple[LayoutSpan, ...]


@dataclass(frozen=True)
class LayoutTable:
    """A page-space table retaining its rows and original word geometry."""

    page: int
    rows: tuple[tuple[LayoutTableCell, ...], ...]
    bbox: BBox
    start: int
    end: int

    def as_structured_table(self) -> Mapping[str, Any]:
        """Adapt cells to the existing ``openmed.structured.Table`` contract."""
        from openmed.structured.tables import Table, TableCell

        return Table(
            n_rows=len(self.rows),
            n_columns=max((len(row) for row in self.rows), default=0),
            cells=[
                TableCell(
                    row=cell.row,
                    column=cell.column,
                    colspan=1,
                    text=cell.text,
                    start=cell.start,
                    end=cell.end,
                    is_header=False,
                )
                for row in self.rows
                for cell in row
            ],
            header_rows=[],
            header_columns=[],
        )


@dataclass(frozen=True)
class LayoutQuality:
    """Synthetic gold comparison for reading order and table assignments."""

    reading_order_accuracy: float
    table_cell_accuracy: float
    expected_words: int
    expected_cells: int


@dataclass(frozen=True)
class LayoutDocument:
    """Linearized OCR text with regions and bidirectional source maps."""

    text: str
    columns: tuple[LayoutColumn, ...] = ()
    tables: tuple[LayoutTable, ...] = ()
    headers: tuple[LayoutBand, ...] = ()
    footers: tuple[LayoutBand, ...] = ()
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

    @_safe_boundary
    def location_at(self, offset: int) -> LayoutSpan | None:
        """Return the word mapping covering ``offset``, if it is mapped."""
        if type(offset) is not int:
            raise ValueError("invalid character offset")
        if offset < 0 or offset >= len(self.text):
            return None
        for span in self.spans:
            if span.start <= offset < span.end:
                return span
        return None

    @_safe_boundary
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

    @_safe_boundary
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
        if type(page) is not int or page < 0:
            raise ValueError("invalid page index")
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

    def detect_sections(self, *, language: str | None = None) -> tuple[Any, ...]:
        """Detect clinical sections on layout-correct, line-delimited text."""
        from openmed.clinical.sections import detect_sections

        return detect_sections(self.text, language=language)


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
    kind: str = "body"
    table_rows: tuple[tuple[tuple[_WordRecord, ...], ...], ...] = ()


class FakeLayoutInput:
    """Deterministic in-memory layout input for offline tests.

    This mirrors :class:`~openmed.multimodal.ocr.FakeOcrEngine` while exposing
    the OCR result shape directly to :func:`parse_layout`.
    """

    @_safe_boundary
    def __init__(self, words: Iterable[OcrWord], **metadata: Any) -> None:
        self.words = _bounded(words)
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


@_safe_boundary
def parse_layout(
    ocr_result: OcrResult | FakeLayoutInput,
    *,
    separator: str = " ",
    line_separator: str = "\n",
    column_gap: float | None = None,
    line_tolerance: float | None = None,
) -> LayoutDocument:
    """Reconstruct deterministic reading order from positioned OCR words.

    Args:
        ocr_result: An :class:`OcrResult` or compatible object exposing
            ``words`` and optional ``metadata`` attributes.
        separator: Text inserted between words in one reconstructed line.
        line_separator: Text inserted between reconstructed lines and regions.
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
    if not isinstance(line_separator, str):
        raise TypeError("line_separator must be a string")
    if len(separator) > 4096 or len(line_separator) > 4096:
        raise ValueError("layout input limit exceeded")
    for name, value in (
        ("column_gap", column_gap),
        ("line_tolerance", line_tolerance),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative")

    source_metadata = getattr(ocr_result, "metadata", {})
    metadata = dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
    records = _coerce_records(ocr_result)
    if not records:
        metadata.update(
            {
                "format": "ocr_layout",
                "page_count": 0,
                "column_count": 0,
                "table_count": 0,
                "header_count": 0,
                "footer_count": 0,
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
        drafts.extend(
            _page_drafts(
                page,
                records_for_page,
                metadata=metadata,
                column_gap=column_gap,
                line_tolerance=tolerance,
            )
        )

    text_parts: list[str] = []
    spans: list[LayoutSpan] = []
    blocks: list[LayoutBlock] = []
    columns: list[LayoutColumn] = []
    tables: list[LayoutTable] = []
    headers: list[LayoutBand] = []
    footers: list[LayoutBand] = []
    cursor = 0

    for draft in drafts:
        region_blocks: list[LayoutBlock] = []
        for line in draft.blocks:
            block_index = len(blocks)
            block_spans: list[LayoutSpan] = []
            block_start: int | None = None
            for record in line:
                if cursor:
                    boundary = separator if block_start is not None else line_separator
                    text_parts.append(boundary)
                    cursor += len(boundary)
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
                    metadata={"block_type": draft.kind},
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
            region_blocks.append(block)

        if not region_blocks:
            continue
        if draft.kind == "body":
            columns.append(
                LayoutColumn(
                    page=draft.page,
                    index=draft.index,
                    blocks=tuple(region_blocks),
                    bbox=draft.bbox,
                )
            )
        elif draft.kind in {"header", "footer"}:
            band = LayoutBand(
                page=draft.page,
                kind=draft.kind,
                blocks=tuple(region_blocks),
                bbox=draft.bbox,
            )
            (headers if draft.kind == "header" else footers).append(band)
        elif draft.kind == "table":
            mapped = {
                span.word_index: span for block in region_blocks for span in block.spans
            }
            rows: list[tuple[LayoutTableCell, ...]] = []
            for row_index, row in enumerate(draft.table_rows):
                cells: list[LayoutTableCell] = []
                for column_index, cell_words in enumerate(row):
                    cell_spans = tuple(mapped[word.index] for word in cell_words)
                    cells.append(
                        LayoutTableCell(
                            row=row_index,
                            column=column_index,
                            text=separator.join(word.text for word in cell_words),
                            start=cell_spans[0].start,
                            end=cell_spans[-1].end,
                            bbox=_union_bbox(word.bbox for word in cell_words),
                            spans=cell_spans,
                        )
                    )
                rows.append(tuple(cells))
            tables.append(
                LayoutTable(
                    page=draft.page,
                    rows=tuple(rows),
                    bbox=draft.bbox,
                    start=region_blocks[0].start,
                    end=region_blocks[-1].end,
                )
            )

    metadata.update(
        {
            "format": "ocr_layout",
            "page_count": len(page_records),
            "column_count": len(columns),
            "table_count": len(tables),
            "header_count": len(headers),
            "footer_count": len(footers),
            "block_count": len(blocks),
            "word_count": len(spans),
            "separator": separator,
            "line_separator": line_separator,
        }
    )
    return LayoutDocument(
        text="".join(text_parts),
        columns=tuple(columns),
        tables=tuple(tables),
        headers=tuple(headers),
        footers=tuple(footers),
        blocks=tuple(blocks),
        spans=tuple(spans),
        metadata=metadata,
    )


def evaluate_layout(
    document: LayoutDocument,
    *,
    expected_word_order: Sequence[int],
    expected_cells: Mapping[int, tuple[int, int, int]],
) -> LayoutQuality:
    """Score a layout against synthetic original-word indices and table cells.

    ``expected_cells`` maps each original word index to its expected
    ``(table, row, column)`` assignment. Extra, missing, and misplaced words
    count against the corresponding accuracy denominator.
    """
    actual_order = tuple(span.word_index for span in document.spans)
    ordered = tuple(expected_word_order)
    correct_order = sum(
        actual == expected for actual, expected in zip(actual_order, ordered)
    )
    order_denominator = max(len(actual_order), len(ordered), 1)
    actual_cells = {
        span.word_index: (table_index, row_index, column_index)
        for table_index, table in enumerate(document.tables)
        for row_index, row in enumerate(table.rows)
        for column_index, cell in enumerate(row)
        for span in cell.spans
    }
    correct_cells = sum(
        actual_cells.get(index) == assignment
        for index, assignment in expected_cells.items()
    )
    cell_denominator = max(len(actual_cells), len(expected_cells), 1)
    return LayoutQuality(
        reading_order_accuracy=correct_order / order_denominator,
        table_cell_accuracy=correct_cells / cell_denominator,
        expected_words=len(ordered),
        expected_cells=len(expected_cells),
    )


def _body_drafts(
    page: int,
    records: Sequence[_WordRecord],
    *,
    column_gap: float | None,
    line_tolerance: float,
) -> tuple[_ColumnDraft, ...]:
    if not records:
        return ()
    column_groups = _cluster_columns(
        records,
        column_gap=column_gap,
        line_tolerance=line_tolerance,
    )
    return tuple(
        _ColumnDraft(
            page=page,
            index=index,
            blocks=_group_lines(group, line_tolerance),
            bbox=_union_bbox(record.bbox for record in group),
        )
        for index, group in enumerate(column_groups)
    )


def _page_drafts(
    page: int,
    records: Sequence[_WordRecord],
    *,
    metadata: Mapping[str, Any],
    column_gap: float | None,
    line_tolerance: float,
) -> tuple[_ColumnDraft, ...]:
    size = _page_dimensions(metadata, page)
    if size is None:
        header_words, footer_words, body_words = _infer_bands(records, line_tolerance)
    else:
        from .box_normalization import normalize_box

        for record in records:
            normalize_box(record.bbox, unit="pixel", page_size=size, page=page)
        height = size[1]
        header_words = tuple(record for record in records if record.y1 <= height * 0.12)
        footer_words = tuple(record for record in records if record.y0 >= height * 0.88)
        typical_height = median(record.y1 - record.y0 for record in records)
        minimum_gap = max(typical_height * 2.0, height * 0.025)
        middle_words = tuple(
            record
            for record in records
            if record not in header_words and record not in footer_words
        )
        if (
            not middle_words
            or not header_words
            or min(word.y0 for word in middle_words)
            - max(word.y1 for word in header_words)
            < minimum_gap
        ):
            header_words = ()
        if (
            not middle_words
            or not footer_words
            or min(word.y0 for word in footer_words)
            - max(word.y1 for word in middle_words)
            < minimum_gap
        ):
            footer_words = ()
        band_indices = {record.index for record in (*header_words, *footer_words)}
        body_words = tuple(
            record for record in records if record.index not in band_indices
        )

    drafts: list[_ColumnDraft] = []
    if header_words:
        drafts.append(
            _ColumnDraft(
                page=page,
                index=-1,
                blocks=_group_lines(header_words, line_tolerance),
                bbox=_union_bbox(word.bbox for word in header_words),
                kind="header",
            )
        )

    tables = _detect_tables(body_words, line_tolerance)
    table_indices = {
        word.index for table in tables for row in table for cell in row for word in cell
    }
    remaining = tuple(word for word in body_words if word.index not in table_indices)
    lower_bound = float("-inf")
    emitted_indices: set[int] = set()
    for table in tables:
        table_words = tuple(word for row in table for cell in row for word in cell)
        table_top = min(word.y0 for word in table_words)
        before = tuple(
            word
            for word in remaining
            if word.index not in emitted_indices
            and lower_bound <= word.center_y < table_top
        )
        emitted_indices.update(word.index for word in before)
        drafts.extend(
            _body_drafts(
                page,
                before,
                column_gap=column_gap,
                line_tolerance=line_tolerance,
            )
        )
        drafts.append(
            _ColumnDraft(
                page=page,
                index=-1,
                blocks=tuple(
                    tuple(word for cell in row for word in cell) for row in table
                ),
                bbox=_union_bbox(word.bbox for word in table_words),
                kind="table",
                table_rows=table,
            )
        )
        lower_bound = max(word.y1 for word in table_words)
    after = tuple(word for word in remaining if word.index not in emitted_indices)
    drafts.extend(
        _body_drafts(
            page,
            after,
            column_gap=column_gap,
            line_tolerance=line_tolerance,
        )
    )

    if footer_words:
        drafts.append(
            _ColumnDraft(
                page=page,
                index=-1,
                blocks=_group_lines(footer_words, line_tolerance),
                bbox=_union_bbox(word.bbox for word in footer_words),
                kind="footer",
            )
        )
    return tuple(drafts)


def _infer_bands(
    records: Sequence[_WordRecord], tolerance: float
) -> tuple[
    tuple[_WordRecord, ...],
    tuple[_WordRecord, ...],
    tuple[_WordRecord, ...],
]:
    lines = _group_lines(records, tolerance)
    if len(lines) < 4:
        return (), (), tuple(records)
    gaps = [
        min(word.y0 for word in right) - max(word.y1 for word in left)
        for left, right in zip(lines, lines[1:])
    ]
    typical_height = median(word.y1 - word.y0 for word in records)
    ordinary_gaps = sorted(max(gap, 0.0) for gap in gaps)[: max(1, len(gaps) - 2)]
    threshold = max(typical_height * 2.0, median(ordinary_gaps) * 2.0)
    top_count = 1 if gaps[0] > threshold else 0
    bottom_count = 1 if gaps[-1] > threshold else 0
    if top_count + bottom_count >= len(lines):
        return (), (), tuple(records)
    header = tuple(word for line in lines[:top_count] for word in line)
    footer = tuple(word for line in lines[len(lines) - bottom_count :] for word in line)
    excluded = {word.index for word in (*header, *footer)}
    body = tuple(word for word in records if word.index not in excluded)
    return header, footer, body


def _page_dimensions(
    metadata: Mapping[str, Any], page: int
) -> tuple[float, float] | None:
    sizes = metadata.get("page_dimensions")
    if sizes is None:
        return None
    if isinstance(sizes, Mapping):
        if page in sizes and str(page) in sizes and sizes[page] != sizes[str(page)]:
            raise ValueError("conflicting OCR page dimensions")
        value = sizes.get(page, sizes.get(str(page)))
    elif isinstance(sizes, Sequence) and not isinstance(sizes, (str, bytes)):
        value = sizes[page] if page < len(sizes) else None
    else:
        raise ValueError("invalid OCR page dimensions")
    if value is None:
        raise ValueError("missing OCR page dimensions")
    from .box_normalization import PageSize
    from .page_rotation import PageSize as RotationPageSize

    try:
        if isinstance(value, PageSize):
            size = PageSize(value.width, value.height)
        elif isinstance(value, RotationPageSize):
            size = PageSize(value.width, value.height)
        elif isinstance(value, Mapping):
            size = PageSize(value["width"], value["height"])
        else:
            size = PageSize(*_bounded(value, 2))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid OCR page dimensions") from exc
    return size.as_tuple()


def _detect_tables(
    records: Sequence[_WordRecord], tolerance: float
) -> tuple[tuple[tuple[tuple[_WordRecord, ...], ...], ...], ...]:
    if len(records) < 6:
        return ()
    page_width = max(word.x1 for word in records) - min(word.x0 for word in records)
    heights = [word.y1 - word.y0 for word in records]
    gap_threshold = max(median(heights) * 1.5, page_width * 0.035)
    aligned_tolerance = max(median(heights) * 2.0, page_width * 0.04)
    candidates: list[tuple[tuple[_WordRecord, ...], ...] | None] = []
    for line in _group_lines(records, tolerance):
        cells: list[list[_WordRecord]] = [[]]
        for word in line:
            if cells[-1] and word.x0 - cells[-1][-1].x1 > gap_threshold:
                cells.append([])
            cells[-1].append(word)
        candidates.append(
            tuple(tuple(cell) for cell in cells) if len(cells) >= 3 else None
        )

    tables: list[tuple[tuple[tuple[_WordRecord, ...], ...], ...]] = []
    run: list[tuple[tuple[_WordRecord, ...], ...]] = []
    for row in (*candidates, None):
        if row is not None and (
            not run
            or (
                len(row) == len(run[-1])
                and all(
                    abs(cell[0].x0 - previous[0].x0) <= aligned_tolerance
                    for cell, previous in zip(row, run[-1])
                )
                and min(word.y0 for cell in row for word in cell)
                - max(word.y1 for cell in run[-1] for word in cell)
                <= median(heights) * 4
            )
        ):
            run.append(row)
            continue
        if len(run) >= 2:
            tables.append(tuple(run))
        run = [row] if row is not None else []
    return tuple(tables)


def _coerce_records(ocr_result: Any) -> tuple[_WordRecord, ...]:
    raw_words = getattr(ocr_result, "words", None)
    if raw_words is None:
        raise TypeError("ocr_result must expose a words iterable")

    records: list[_WordRecord] = []
    total_characters = 0
    for index, raw_word in enumerate(_bounded(raw_words)):
        text = _word_value(raw_word, "text", "")
        if not isinstance(text, str) or len(text) > 4096:
            raise ValueError("invalid OCR word text")
        total_characters += len(text)
        if total_characters > 1048576:
            raise ValueError("layout input limit exceeded")
        if not text.strip():
            continue
        bbox = _coerce_bbox(_word_value(raw_word, "bbox", None), index=index)
        page = _word_value(raw_word, "page", 0)
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            raise ValueError(f"OCR word {index} has an invalid page")
        raw_confidence = _word_value(raw_word, "confidence", 1.0)
        if isinstance(raw_confidence, bool):
            raise ValueError("invalid OCR confidence")
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"OCR word {index} has an invalid confidence") from exc
        if not isfinite(confidence) or not 0 <= confidence <= 1:
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
        styles = (("x0", "y0", "x1", "y1"), ("left", "top", "right", "bottom"))
        present = [keys for keys in styles if any(key in value for key in keys)]
        if len(present) != 1 or not all(key in value for key in present[0]):
            raise ValueError("ambiguous OCR bbox")
        value = tuple(value[key] for key in present[0])
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        raise ValueError("invalid OCR bbox")
    raw_values = _bounded(value, 4)
    if len(raw_values) != 4 or any(isinstance(number, bool) for number in raw_values):
        raise ValueError("invalid OCR bbox")
    values = tuple(float(number) for number in raw_values)
    if not all(isfinite(number) for number in values):
        raise ValueError("invalid OCR bbox")
    x0, y0, x1, y1 = values
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
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
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end < start
        or end > len(text)
    ):
        raise ValueError("character offsets are outside the layout document")


__all__ = [
    "BBox",
    "FakeLayoutEngine",
    "FakeLayoutInput",
    "LayoutBand",
    "LayoutBlock",
    "LayoutColumn",
    "LayoutDocument",
    "LayoutMapEntry",
    "LayoutQuality",
    "LayoutSpan",
    "LayoutTable",
    "LayoutTableCell",
    "LayoutWordSpan",
    "parse_layout",
    "evaluate_layout",
]
