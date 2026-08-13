"""Structured PDF tables and caption regions with source-offset projection.

The flat PDF extractor remains the source of truth for normalized text and
character offsets.  This module uses pdfplumber's table geometry and the flat
extractor's word spans to add structure without changing that contract.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan
from .documents_pdf import (
    ProjectedRectangle,
    _coerce_entity,
    extract_pdf,
    project_text_spans,
)

_PDFPLUMBER_HINT = 'Install with: pip install "openmed[multimodal]".'
_LINE_TOLERANCE = 2.0
_BBox = tuple[float, float, float, float]
_CaptionMatch = re.compile(
    r"^\s*(?P<kind>figure|fig\.?|table)\b"
    r"(?:\s+(?:[A-Za-z]?\d+(?:\.\d+)*|N))?"
    r"\s*[:.)-]\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TableCell:
    """One table cell mapped to a PDF page rectangle and text offsets."""

    text: str
    start: int | None
    end: int | None
    page: int
    bbox: _BBox
    row_index: int
    column_index: int

    @property
    def char_start(self) -> int | None:
        """Alias for the inclusive normalized-text start offset."""

        return self.start

    @property
    def char_end(self) -> int | None:
        """Alias for the exclusive normalized-text end offset."""

        return self.end

    @property
    def row(self) -> int:
        """Alias for the zero-based row index."""

        return self.row_index

    @property
    def column(self) -> int:
        """Alias for the zero-based column index."""

        return self.column_index

    @property
    def offset_range(self) -> tuple[int, int] | None:
        """Return the mapped offset range, or ``None`` for an empty cell."""

        if self.start is None or self.end is None:
            return None
        return self.start, self.end

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable cell representation.

        ``include_text=False`` is intended for audit metadata so that the
        region geometry and offsets can be retained without copying source
        text into a report.
        """
        payload: dict[str, Any] = {
            "page": self.page,
            "bbox": self.bbox,
            "row_index": self.row_index,
            "column_index": self.column_index,
        }
        if self.start is not None:
            payload["start"] = self.start
        if self.end is not None:
            payload["end"] = self.end
        if include_text:
            payload["text"] = self.text
        else:
            payload["text_sha256"] = _text_sha256(self.text)
        return payload


@dataclass(frozen=True)
class TableRegion:
    """A detected table and its individually projected cells."""

    page: int
    bbox: _BBox
    cells: tuple[TableCell, ...]
    start: int | None = None
    end: int | None = None
    table_index: int = 0

    @property
    def char_start(self) -> int | None:
        """Return the first mapped cell offset."""

        return self.start

    @property
    def char_end(self) -> int | None:
        """Return the exclusive end of the mapped cell range."""

        return self.end

    @property
    def offset_range(self) -> tuple[int, int] | None:
        """Return the table's combined mapped offset range."""

        if self.start is None or self.end is None:
            return None
        return self.start, self.end

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable table representation."""
        payload: dict[str, Any] = {
            "page": self.page,
            "bbox": self.bbox,
            "table_index": self.table_index,
            "cells": [cell.to_dict(include_text=include_text) for cell in self.cells],
        }
        if self.start is not None:
            payload["start"] = self.start
        if self.end is not None:
            payload["end"] = self.end
        return payload


@dataclass(frozen=True)
class CaptionRegion:
    """A figure or table caption line mapped to text offsets and a bbox."""

    text: str
    start: int
    end: int
    page: int
    bbox: _BBox
    kind: str

    @property
    def caption_type(self) -> str:
        """Return ``"figure"`` or ``"table"`` for the caption."""

        return self.kind

    @property
    def char_start(self) -> int:
        """Alias for the inclusive normalized-text start offset."""

        return self.start

    @property
    def char_end(self) -> int:
        """Alias for the exclusive normalized-text end offset."""

        return self.end

    @property
    def offset_range(self) -> tuple[int, int]:
        """Return the caption's normalized-text offset range."""

        return self.start, self.end

    @property
    def caption(self) -> str:
        """Return the caption text."""

        return self.text

    @property
    def caption_text(self) -> str:
        """Alias for the caption text."""

        return self.text

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable caption representation."""
        payload: dict[str, Any] = {
            "page": self.page,
            "bbox": self.bbox,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
        }
        if include_text:
            payload["text"] = self.text
        else:
            payload["text_sha256"] = _text_sha256(self.text)
        return payload


@dataclass(frozen=True)
class PdfRegions:
    """All structured table and caption regions found in a PDF."""

    tables: tuple[TableRegion, ...] = ()
    captions: tuple[CaptionRegion, ...] = ()

    @property
    def table_regions(self) -> tuple[TableRegion, ...]:
        """Alias for the detected table regions."""

        return self.tables

    @property
    def caption_regions(self) -> tuple[CaptionRegion, ...]:
        """Alias for the detected caption regions."""

        return self.captions

    def __iter__(self):
        """Iterate as ``(tables, captions)`` for tuple-style consumers."""
        yield self.tables
        yield self.captions

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable structured-region representation."""
        return {
            "tables": [
                table.to_dict(include_text=include_text) for table in self.tables
            ],
            "captions": [
                caption.to_dict(include_text=include_text) for caption in self.captions
            ],
        }


def extract_pdf_regions(
    path: str | Path,
    document: ExtractedDocument | None = None,
) -> PdfRegions:
    """Extract table cells and caption lines from ``path``.

    ``document`` may be supplied when the caller already has the OM-060 flat
    extraction.  Reusing it guarantees that every structured offset refers to
    the exact same normalized character stream.
    """
    flat_document = document if document is not None else extract_pdf(path)
    pdfplumber = _import_pdfplumber()
    tables: list[TableRegion] = []

    with pdfplumber.open(path) as pdf:
        pages = tuple(getattr(pdf, "pages", ()))
        for page_index, page in enumerate(pages):
            page_spans = tuple(
                span for span in flat_document.spans if span.page == page_index
            )
            for table_index, table in enumerate(_find_tables(page)):
                table_cells = tuple(
                    _build_table_cell(
                        flat_document,
                        page_spans,
                        page_index,
                        bbox,
                        text,
                        row_index,
                        column_index,
                    )
                    for bbox, text, row_index, column_index in _iter_table_cells(table)
                )
                if not table_cells:
                    continue
                table_bbox = _coerce_bbox(getattr(table, "bbox", None))
                if table_bbox is None:
                    table_bbox = _union_bbox(cell.bbox for cell in table_cells)
                offsets = [
                    (cell.start, cell.end)
                    for cell in table_cells
                    if cell.start is not None and cell.end is not None
                ]
                tables.append(
                    TableRegion(
                        page=page_index,
                        bbox=table_bbox,
                        cells=table_cells,
                        start=min(start for start, _ in offsets) if offsets else None,
                        end=max(end for _, end in offsets) if offsets else None,
                        table_index=table_index,
                    )
                )

    return PdfRegions(
        tables=tuple(tables),
        captions=_extract_caption_regions(flat_document),
    )


def extract_pdf_tables(
    path: str | Path,
    document: ExtractedDocument | None = None,
) -> tuple[TableRegion, ...]:
    """Return detected table regions with per-cell text and bboxes."""
    return extract_pdf_regions(path, document=document).tables


def extract_pdf_captions(
    path_or_document: str | Path | ExtractedDocument,
    document: ExtractedDocument | None = None,
) -> tuple[CaptionRegion, ...]:
    """Return caption regions mapped to the flat PDF character stream."""
    if isinstance(path_or_document, ExtractedDocument):
        return _extract_caption_regions(path_or_document)
    return extract_pdf_regions(path_or_document, document=document).captions


def project_structured_spans(
    document: ExtractedDocument,
    regions: PdfRegions,
    spans: Iterable[Any],
    *,
    fallback_to_words: bool = True,
) -> tuple[ProjectedRectangle, ...]:
    """Project detected spans to whole table-cell or caption rectangles.

    A span overlapping a structured region uses that region's bbox.  Spans
    outside tables and captions retain the OM-060 word-level projection when
    ``fallback_to_words`` is true.
    """
    rectangles: list[ProjectedRectangle] = []
    for raw_span in spans:
        entity = _coerce_entity(raw_span)
        if entity is None:
            continue
        start, end, label, confidence = entity
        if end <= start:
            continue

        matched = False
        for table in regions.tables:
            for cell in table.cells:
                if cell.start is None or cell.end is None:
                    continue
                if not _ranges_overlap(start, end, cell.start, cell.end):
                    continue
                matched = True
                rectangles.append(
                    _structured_rectangle(
                        document,
                        start=start,
                        end=end,
                        page=cell.page,
                        bbox=cell.bbox,
                        label=label,
                        confidence=confidence,
                        metadata={
                            "region_type": "table_cell",
                            "table_index": table.table_index,
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                        },
                    )
                )

        for caption in regions.captions:
            if not _ranges_overlap(start, end, caption.start, caption.end):
                continue
            matched = True
            rectangles.append(
                _structured_rectangle(
                    document,
                    start=start,
                    end=end,
                    page=caption.page,
                    bbox=caption.bbox,
                    label=label,
                    confidence=confidence,
                    metadata={
                        "region_type": "caption",
                        "caption_type": caption.kind,
                    },
                )
            )

        if not matched and fallback_to_words:
            rectangles.extend(project_text_spans(document, (raw_span,)))
    return tuple(rectangles)


def project_region_spans(
    document: ExtractedDocument,
    regions: PdfRegions,
    spans: Iterable[Any],
    *,
    fallback_to_words: bool = True,
) -> tuple[ProjectedRectangle, ...]:
    """Compatibility alias for :func:`project_structured_spans`."""
    return project_structured_spans(
        document,
        regions,
        spans,
        fallback_to_words=fallback_to_words,
    )


def _import_pdfplumber() -> Any:
    try:
        return importlib.import_module("pdfplumber")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        from .exceptions import MissingDependencyError

        raise MissingDependencyError(
            dependency="pdfplumber", instruction=_PDFPLUMBER_HINT
        ) from exc


def _find_tables(page: Any) -> tuple[Any, ...]:
    finder = getattr(page, "find_tables", None)
    if not callable(finder):
        return ()
    tables = finder()
    return tuple(tables or ())


def _iter_table_cells(
    table: Any,
) -> Iterable[tuple[_BBox, str, int, int]]:
    extracted = _extract_table_text(table)
    rows = tuple(getattr(table, "rows", ()) or ())
    if rows:
        yielded = False
        for row_index, row in enumerate(rows):
            cells = tuple(getattr(row, "cells", ()) or ())
            for column_index, raw_cell in enumerate(cells):
                bbox = _coerce_bbox(raw_cell)
                if bbox is None:
                    continue
                yielded = True
                yield (
                    bbox,
                    _cell_text(extracted, row_index, column_index, raw_cell=raw_cell),
                    row_index,
                    column_index,
                )
        if yielded:
            return

    raw_cells = tuple(getattr(table, "cells", ()) or ())
    grouped = _group_cells_by_row(raw_cells)
    for row_index, row in enumerate(grouped):
        for column_index, raw_cell in enumerate(row):
            bbox = _coerce_bbox(raw_cell)
            if bbox is None:
                continue
            yield (
                bbox,
                _cell_text(extracted, row_index, column_index, raw_cell=raw_cell),
                row_index,
                column_index,
            )


def _extract_table_text(table: Any) -> tuple[tuple[str | None, ...], ...]:
    extractor = getattr(table, "extract", None)
    if not callable(extractor):
        return ()
    extracted = extractor()
    if not extracted:
        return ()
    if isinstance(extracted, (str, bytes, bytearray)):
        return ((str(extracted).strip(),),)
    rows: list[tuple[str | None, ...]] = []
    for row in extracted:
        if isinstance(row, (str, bytes, bytearray)):
            row = (row,)
        rows.append(
            tuple(None if value is None else str(value).strip() for value in row)
        )
    return tuple(rows)


def _cell_text(
    rows: Sequence[Sequence[str | None]],
    row_index: int,
    column_index: int,
    *,
    raw_cell: Any = None,
) -> str:
    if row_index < len(rows) and column_index < len(rows[row_index]):
        text = rows[row_index][column_index]
        if text:
            return text
    return _raw_cell_text(raw_cell)


def _raw_cell_text(cell: Any) -> str:
    if isinstance(cell, Mapping):
        value = cell.get("text")
    else:
        value = getattr(cell, "text", None)
    return "" if value is None else str(value).strip()


def _group_cells_by_row(raw_cells: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    positioned = [
        (bbox, index)
        for index, raw_cell in enumerate(raw_cells)
        if (bbox := _coerce_bbox(raw_cell)) is not None
    ]
    positioned.sort(key=lambda item: (item[0][1], item[0][0], item[1]))
    rows: list[list[tuple[_BBox, int]]] = []
    for bbox, index in positioned:
        for row in rows:
            if _same_line_bbox(row[0][0], bbox):
                row.append((bbox, index))
                break
        else:
            rows.append([(bbox, index)])
    return tuple(
        tuple(raw_cells[index] for _, index in sorted(row, key=lambda item: item[0][0]))
        for row in rows
    )


def _same_line_bbox(first: _BBox, second: _BBox) -> bool:
    overlap = min(first[3], second[3]) - max(first[1], second[1])
    if overlap >= 0:
        return True
    first_center = (first[1] + first[3]) / 2.0
    second_center = (second[1] + second[3]) / 2.0
    return abs(first_center - second_center) <= _LINE_TOLERANCE


def _build_table_cell(
    document: ExtractedDocument,
    page_spans: Sequence[SourceSpan],
    page: int,
    bbox: _BBox,
    extracted_text: str,
    row_index: int,
    column_index: int,
) -> TableCell:
    covered = tuple(
        source
        for source in page_spans
        if source.bbox and _bbox_intersects(source.bbox, bbox)
    )
    start = min((source.start for source in covered), default=None)
    end = max((source.end for source in covered), default=None)
    text = extracted_text
    if not text and start is not None and end is not None:
        text = document.text[start:end]
    return TableCell(
        text=text,
        start=start,
        end=end,
        page=page,
        bbox=bbox,
        row_index=row_index,
        column_index=column_index,
    )


def _extract_caption_regions(document: ExtractedDocument) -> tuple[CaptionRegion, ...]:
    captions: list[CaptionRegion] = []
    page_spans: dict[int, list[SourceSpan]] = {}
    for span in document.spans:
        if span.bbox is not None:
            page_spans.setdefault(span.page, []).append(span)

    for page, spans in page_spans.items():
        ordered = sorted(spans, key=_span_sort_key)
        for line in _group_source_lines(ordered):
            start = min(span.start for span in line)
            end = max(span.end for span in line)
            text = document.text[start:end]
            match = _CaptionMatch.match(text)
            if match is None:
                continue
            kind = match.group("kind").lower().rstrip(".")
            if kind.startswith("fig"):
                kind = "figure"
            else:
                kind = "table"
            captions.append(
                CaptionRegion(
                    text=text,
                    start=start,
                    end=end,
                    page=page,
                    bbox=_union_bbox(span.bbox for span in line if span.bbox),
                    kind=kind,
                )
            )
    return tuple(captions)


def _group_source_lines(
    spans: Sequence[SourceSpan],
) -> tuple[tuple[SourceSpan, ...], ...]:
    lines: list[list[SourceSpan]] = []
    for span in spans:
        for line in lines:
            if _same_line_bbox(line[0].bbox, span.bbox):  # type: ignore[arg-type]
                line.append(span)
                break
        else:
            lines.append([span])
    return tuple(tuple(line) for line in lines)


def _span_sort_key(span: SourceSpan) -> tuple[int, float, float, int]:
    if span.bbox is None:
        return span.page, 0.0, 0.0, span.start
    return span.page, span.bbox[1], span.bbox[0], span.start


def _structured_rectangle(
    document: ExtractedDocument,
    *,
    start: int,
    end: int,
    page: int,
    bbox: _BBox,
    label: str | None,
    confidence: float | None,
    metadata: Mapping[str, Any],
) -> ProjectedRectangle:
    enriched = dict(metadata)
    enriched["text_sha256"] = _text_sha256(document.text[start:end])
    return ProjectedRectangle(
        start=start,
        end=end,
        page=page,
        bbox=bbox,
        label=label,
        confidence=confidence,
        metadata=enriched,
    )


def _ranges_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> bool:
    return first_end > second_start and first_start < second_end


def _bbox_intersects(first: _BBox, second: _BBox) -> bool:
    horizontal = min(first[2], second[2]) - max(first[0], second[0])
    vertical = min(first[3], second[3]) - max(first[1], second[1])
    if horizontal > 0 and vertical > 0:
        return True
    center_x = (first[0] + first[2]) / 2.0
    center_y = (first[1] + first[3]) / 2.0
    return second[0] <= center_x <= second[2] and second[1] <= center_y <= second[3]


def _coerce_bbox(value: Any) -> _BBox | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        nested = value.get("bbox")
        if nested is not None:
            return _coerce_bbox(nested)
        values = (
            value.get("x0"),
            value.get("top", value.get("y0")),
            value.get("x1"),
            value.get("bottom", value.get("y1")),
        )
    else:
        nested = getattr(value, "bbox", None)
        if nested is not None and nested is not value:
            return _coerce_bbox(nested)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            values = tuple(value[:4])
        else:
            return None
    if len(values) < 4 or any(item is None for item in values):
        return None
    x0, y0, x1, y1 = (float(item) for item in values[:4])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _union_bbox(bboxes: Iterable[_BBox]) -> _BBox:
    boxes = tuple(bboxes)
    if not boxes:
        raise ValueError("at least one bbox is required")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "CaptionRegion",
    "PdfRegions",
    "TableCell",
    "TableRegion",
    "extract_pdf_captions",
    "extract_pdf_regions",
    "extract_pdf_tables",
    "project_region_spans",
    "project_structured_spans",
]
