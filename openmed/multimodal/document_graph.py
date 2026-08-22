"""Deterministic, provenance-aware graph construction for clinical documents.

The graph is the layout contract between document intake and structured
extraction.  It accepts coordinate-bearing PDF/OCR blocks, reconstructs page
reading order, preserves table and form relationships, and maps every emitted
character range back to a page region.  The core implementation is standard
library only; PDF parsing imports ``pdfplumber`` lazily.

This module deliberately stores offsets into the graph's normalized text.  A
caller that supplies source offsets (for example, an OCR engine that already
tracks offsets into a raw transcript) gets those offsets retained in node
metadata without confusing them with the graph offsets used by downstream
extractors.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import ExtractedDocument
from .exceptions import (
    EncryptedDocumentError,
    MalformedDocumentError,
    MissingDependencyError,
)

BBox = tuple[float, float, float, float]
BoundingBox = BBox
_FORM_LINE_RE = re.compile(r"^\s*(.{1,100}?)\s*[:=：﹕꞉]\s*(.*?)\s*$")


def _value(source: Any, *names: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _text_value(source: Any, *names: str, default: str | None = None) -> str | None:
    value = _value(source, *names, default=default)
    if value is None:
        return default
    return str(value)


def _int_value(source: Any, *names: str, default: int | None = None) -> int | None:
    value = _value(source, *names, default=default)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("document integer fields cannot be boolean")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("document integer field is invalid") from exc


def _float_value(
    source: Any, *names: str, default: float | None = None
) -> float | None:
    value = _value(source, *names, default=default)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("document coordinate fields cannot be boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("document coordinate field is invalid") from exc
    if not math.isfinite(result):
        raise ValueError("document coordinate field must be finite")
    return result


def _coerce_bbox(value: Any) -> BBox | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        values = (
            _value(value, "x0", "left"),
            _value(value, "y0", "top"),
            _value(value, "x1", "right"),
            _value(value, "y1", "bottom"),
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            raise ValueError("document bounding boxes must contain four values")
        values = tuple(value)
    else:
        values = (
            _value(value, "x0", "left"),
            _value(value, "y0", "top"),
            _value(value, "x1", "right"),
            _value(value, "y1", "bottom"),
        )
    if any(item is None for item in values):
        raise ValueError("document bounding boxes require x0, y0, x1, and y1")
    result = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("document bounding boxes must be finite")
    if result[2] < result[0] or result[3] < result[1]:
        raise ValueError("document bounding boxes must have positive orientation")
    return result  # type: ignore[return-value]


def _read_bbox(source: Any) -> BBox | None:
    value = _value(source, "bbox", "bounding_box", "box", default=None)
    if value is not None:
        return _coerce_bbox(value)
    x0 = _value(source, "x0", "left", default=None)
    y0 = _value(source, "y0", "top", default=None)
    x1 = _value(source, "x1", "right", default=None)
    y1 = _value(source, "y1", "bottom", default=None)
    if any(item is not None for item in (x0, y0, x1, y1)):
        return _coerce_bbox((x0, y0, x1, y1))
    return None


def _union_bboxes(bboxes: Iterable[BBox | None]) -> BBox | None:
    values = tuple(box for box in bboxes if box is not None)
    if not values:
        return None
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def _bbox_sort_key(bbox: BBox | None) -> tuple[float, float]:
    if bbox is None:
        return (float("inf"), float("inf"))
    return (bbox[1], bbox[0])


def _normalise_kind(value: Any) -> str:
    kind = str(value or "text").strip().lower().replace("-", "_")
    aliases = {
        "word": "text",
        "line": "text",
        "text_block": "text",
        "paragraph": "text",
        "tablecell": "table_cell",
        "cell": "table_cell",
        "form": "form_field",
        "field": "form_field",
        "label": "form_key",
        "key": "form_key",
        "answer": "form_value",
        "value": "form_value",
    }
    return aliases.get(kind, kind)


@dataclass(frozen=True)
class SourceRegion:
    """A graph character range projected to one source-page region."""

    start: int
    end: int
    page: int
    bbox: BBox | None
    node_id: str | None = None
    kind: str = "text"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the half-open graph offsets represented by this region."""
        return (self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible provenance record without source text."""
        result: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": self.bbox,
            "kind": self.kind,
        }
        if self.node_id is not None:
            result["node_id"] = self.node_id
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class DocumentNode:
    """One ordered text/layout node in a :class:`DocumentGraph`."""

    id: str
    kind: str
    text: str
    page: int
    bbox: BBox | None
    start: int
    end: int
    reading_order: int
    column: int | None = None
    table_id: str | None = None
    row: int | None = None
    column_index: int | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_start(self) -> int:
        """Alias for the beginning of this node in normalized graph text."""
        return self.start

    @property
    def source_end(self) -> int:
        """Alias for the exclusive end of this node in graph text."""
        return self.end

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return this node's half-open graph offsets."""
        return (self.start, self.end)

    def region(
        self, *, start: int | None = None, end: int | None = None
    ) -> SourceRegion:
        """Return the node or a clipped part of it as a source region."""
        return SourceRegion(
            start=self.start if start is None else start,
            end=self.end if end is None else end,
            page=self.page,
            bbox=self.bbox,
            node_id=self.id,
            kind=self.kind,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a graph node with explicit source offsets."""
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "page": self.page,
            "bbox": self.bbox,
            "start": self.start,
            "end": self.end,
            "source_offsets": {"start": self.start, "end": self.end},
            "reading_order": self.reading_order,
            "column": self.column,
            "table_id": self.table_id,
            "row": self.row,
            "column_index": self.column_index,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


DocumentBlock = DocumentNode
TextBlock = DocumentNode
Caption = DocumentNode


@dataclass(frozen=True)
class DocumentTableCell:
    """A table cell retaining its row/column and text provenance."""

    id: str
    table_id: str
    row: int
    column: int
    text: str
    page: int
    bbox: BBox | None
    start: int
    end: int
    colspan: int = 1
    is_header: bool = False
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> str:
        """Alias for the cell text used by form extraction callers."""
        return self.text

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the half-open graph offsets for the cell value."""
        return (self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        """Return the cell and its privacy-safe location metadata."""
        return {
            "id": self.id,
            "table_id": self.table_id,
            "row": self.row,
            "column": self.column,
            "colspan": self.colspan,
            "text": self.text,
            "page": self.page,
            "bbox": self.bbox,
            "start": self.start,
            "end": self.end,
            "source_offsets": {"start": self.start, "end": self.end},
            "is_header": self.is_header,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


TableCell = DocumentTableCell


@dataclass(frozen=True)
class DocumentTable:
    """A table reconstructed from coordinate-bearing cells."""

    id: str
    page: int
    bbox: BBox | None
    cells: tuple[DocumentTableCell, ...]
    start: int
    end: int
    caption: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Return the number of occupied row bands."""
        return max((cell.row for cell in self.cells), default=-1) + 1

    @property
    def n_columns(self) -> int:
        """Return the number of occupied columns, including colspans."""
        return max((cell.column + cell.colspan for cell in self.cells), default=0)

    @property
    def rows(self) -> tuple[tuple[DocumentTableCell, ...], ...]:
        """Return cells grouped in deterministic row/column order."""
        return tuple(
            tuple(cell for cell in self.cells if cell.row == row)
            for row in range(self.n_rows)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the table without dropping cell provenance."""
        return {
            "id": self.id,
            "page": self.page,
            "bbox": self.bbox,
            "start": self.start,
            "end": self.end,
            "source_offsets": {"start": self.start, "end": self.end},
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "caption": self.caption,
            "cells": [cell.to_dict() for cell in self.cells],
            "metadata": dict(self.metadata),
        }


Table = DocumentTable


@dataclass(frozen=True)
class DocumentColumn:
    """One inferred or explicit page column."""

    page: int
    index: int
    bbox: BBox | None
    nodes: tuple[DocumentNode, ...]

    @property
    def blocks(self) -> tuple[DocumentNode, ...]:
        """Alias for nodes for callers using block terminology."""
        return self.nodes

    def to_dict(self) -> dict[str, Any]:
        """Return the column and its ordered node IDs."""
        return {
            "page": self.page,
            "index": self.index,
            "bbox": self.bbox,
            "node_ids": [node.id for node in self.nodes],
        }


Column = DocumentColumn


@dataclass(frozen=True)
class DocumentFormField:
    """An explicit key/value relationship supplied by an OCR/PDF adapter."""

    id: str
    key: str
    value: str
    page: int
    key_start: int
    key_end: int
    value_start: int
    value_end: int
    bbox: BBox | None = None
    key_bbox: BBox | None = None
    value_bbox: BBox | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> int:
        """Return the start offset of the complete key/value relationship."""
        return min(self.key_start, self.value_start)

    @property
    def end(self) -> int:
        """Return the end offset of the complete key/value relationship."""
        return max(self.key_end, self.value_end)

    @property
    def source_offsets(self) -> dict[str, dict[str, int]]:
        """Return separately addressable key and value offsets."""
        return {
            "key": {"start": self.key_start, "end": self.key_end},
            "value": {"start": self.value_start, "end": self.value_end},
        }

    def value_region(self) -> SourceRegion:
        """Return the value's most precise known page region."""
        return SourceRegion(
            start=self.value_start,
            end=self.value_end,
            page=self.page,
            bbox=self.value_bbox or self.bbox,
            node_id=self.id,
            kind="form_value",
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return key/value data with explicit provenance ranges."""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "page": self.page,
            "bbox": self.bbox,
            "key_bbox": self.key_bbox,
            "value_bbox": self.value_bbox,
            "key_start": self.key_start,
            "key_end": self.key_end,
            "value_start": self.value_start,
            "value_end": self.value_end,
            "source_offsets": self.source_offsets,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


FormFieldNode = DocumentFormField


@dataclass(frozen=True)
class DocumentPage:
    """A page index and the graph nodes that belong to it."""

    number: int
    node_ids: tuple[str, ...]
    width: float | None = None
    height: float | None = None

    @property
    def page(self) -> int:
        """Alias for the page number used by provenance consumers."""
        return self.number

    def to_dict(self) -> dict[str, Any]:
        """Return page metadata and node membership."""
        return {
            "number": self.number,
            "page": self.number,
            "width": self.width,
            "height": self.height,
            "node_ids": list(self.node_ids),
        }


@dataclass(frozen=True)
class DocumentGraph:
    """Normalized document text plus ordered layout/provenance relationships."""

    text: str
    nodes: tuple[DocumentNode, ...] = ()
    pages: tuple[DocumentPage, ...] = ()
    columns: tuple[DocumentColumn, ...] = ()
    tables: tuple[DocumentTable, ...] = ()
    form_fields: tuple[DocumentFormField, ...] = ()
    captions: tuple[DocumentNode, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> tuple[DocumentNode, ...]:
        """Return all ordered graph nodes."""
        return self.nodes

    @property
    def text_blocks(self) -> tuple[DocumentNode, ...]:
        """Return text-like nodes, excluding table cells and form values."""
        return tuple(
            node
            for node in self.nodes
            if node.kind not in {"table_cell", "form_key", "form_value"}
        )

    @property
    def source_spans(self) -> tuple[SourceRegion, ...]:
        """Return one source region per ordered graph node."""
        return tuple(node.region() for node in self.nodes)

    @property
    def reading_order(self) -> tuple[DocumentNode, ...]:
        """Return nodes in their deterministic reconstruction order."""
        return self.nodes

    def location_at(self, offset: int) -> SourceRegion | None:
        """Return the region covering one normalized text offset."""
        if offset < 0 or offset >= len(self.text):
            return None
        for node in self.nodes:
            if node.start <= offset < node.end:
                return node.region(start=offset, end=offset + 1)
        return None

    def project_span(self, start: int, end: int) -> tuple[SourceRegion, ...]:
        """Project a normalized character span to page regions.

        A span crossing blocks returns one region for each intersecting block.
        Regions are ordered by graph reading order and contain no source text.
        """
        if start < 0 or end < start or end > len(self.text):
            raise ValueError("span must be within normalized document text")
        if start == end:
            return ()
        regions: list[SourceRegion] = []
        for node in self.nodes:
            if node.end <= start or node.start >= end:
                continue
            regions.append(
                node.region(start=max(start, node.start), end=min(end, node.end))
            )
        return tuple(regions)

    project_text_span = project_span
    project_offsets = project_span

    def project_field(self, field: DocumentFormField) -> SourceRegion:
        """Project an explicit form field to its value region."""
        return field.value_region()

    def node_at(self, offset: int) -> DocumentNode | None:
        """Return the node covering ``offset`` or ``None``."""
        if offset < 0 or offset >= len(self.text):
            return None
        return next(
            (node for node in self.nodes if node.start <= offset < node.end), None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete graph, including normalized text and provenance."""
        return {
            "text": self.text,
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "nodes": [node.to_dict() for node in self.nodes],
            "pages": [page.to_dict() for page in self.pages],
            "columns": [column.to_dict() for column in self.columns],
            "tables": [table.to_dict() for table in self.tables],
            "form_fields": [field.to_dict() for field in self.form_fields],
            "captions": [caption.id for caption in self.captions],
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize the graph deterministically for local inspection."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_blocks(
        cls,
        blocks: Any,
        *,
        source_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        row_tolerance: float = 4.0,
        column_tolerance: float = 18.0,
    ) -> "DocumentGraph":
        """Build a graph from coordinate-bearing OCR/PDF blocks."""
        return build_document_graph(
            blocks,
            source_text=source_text,
            metadata=metadata,
            row_tolerance=row_tolerance,
            column_tolerance=column_tolerance,
        )

    @classmethod
    def from_ocr(cls, result: Any, **kwargs: Any) -> "DocumentGraph":
        """Build a graph from an OCR result or synthetic OCR mapping."""
        return build_document_graph(result, **kwargs)

    @classmethod
    def from_pdf(cls, path: str | Path, **kwargs: Any) -> "DocumentGraph":
        """Build a graph from a digital PDF using a lazy optional parser."""
        return extract_pdf_graph(path, **kwargs)


class DocumentGraphBuilder:
    """Reusable configured builder for local PDF/OCR graph intake."""

    def __init__(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        row_tolerance: float = 4.0,
        column_tolerance: float = 18.0,
    ) -> None:
        self.metadata = dict(metadata or {})
        self.row_tolerance = row_tolerance
        self.column_tolerance = column_tolerance

    def build(self, source: Any) -> DocumentGraph:
        """Build a graph using this builder's deterministic layout settings."""
        return build_document_graph(
            source,
            metadata=self.metadata,
            row_tolerance=self.row_tolerance,
            column_tolerance=self.column_tolerance,
        )

    __call__ = build


@dataclass(frozen=True)
class _RawBlock:
    index: int
    text: str
    page: int
    bbox: BBox | None
    kind: str
    column: int | None
    row: int | None
    table_id: str | None
    order: int | None
    confidence: float | None
    metadata: Mapping[str, Any]
    key: str | None = None
    value: str | None = None
    key_bbox: BBox | None = None
    value_bbox: BBox | None = None
    form_id: str | None = None


def _with_page(entry: Any, page: int) -> Any:
    if isinstance(entry, Mapping):
        result = dict(entry)
        result.setdefault("page", page)
        return result
    return {
        "text": _text_value(entry, "text", "value", default="") or "",
        "page": page,
        "bbox": _read_bbox(entry),
        "kind": _value(entry, "kind", "type", "block_type", default="text"),
        "metadata": dict(_value(entry, "metadata", default={}) or {}),
    }


def _flatten_source(source: Any) -> list[Any]:
    if isinstance(source, DocumentGraph):
        return list(source.nodes)
    if isinstance(source, ExtractedDocument):
        return [
            {
                "text": source.text[span.start : span.end],
                "page": span.page,
                "bbox": span.bbox,
                "kind": span.metadata.get("block_type", "text"),
                "start": span.start,
                "end": span.end,
                "metadata": dict(span.metadata),
            }
            for span in source.spans
        ] or [{"text": source.text, "page": 0}]
    object_entries = _value(source, "blocks", "words", "tokens", default=None)
    if object_entries is not None and not isinstance(object_entries, (str, bytes)):
        return list(object_entries)
    if isinstance(source, Mapping):
        pages = _value(source, "pages", default=None)
        if pages is not None:
            flattened: list[Any] = []
            for page_index, page in enumerate(pages):
                page_number = _int_value(
                    page,
                    "page",
                    "page_number",
                    "number",
                    default=page_index,
                )
                entries = _value(
                    page,
                    "blocks",
                    "text_blocks",
                    "words",
                    "tokens",
                    default=None,
                )
                if entries is None:
                    page_text = _text_value(page, "text", default=None)
                    entries = [{"text": page_text}] if page_text is not None else []
                flattened.extend(
                    _with_page(entry, page_number or 0) for entry in entries
                )
            return flattened
        entries = _value(
            source,
            "blocks",
            "text_blocks",
            "words",
            "tokens",
            default=None,
        )
        if entries is not None:
            return list(entries)
        tables = _value(source, "tables", default=None)
        if tables is not None:
            return [{"kind": "table", "tables": tables}]
        text = _text_value(source, "text", "content", default=None)
        return [{"text": text, "page": 0}] if text is not None else []
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return list(source)
    if isinstance(source, (str, Path)):
        return [{"text": str(source), "page": 0}]
    return [source]


def _flatten_nested_tables(entries: Iterable[Any]) -> list[Any]:
    flattened: list[Any] = []
    for entry in entries:
        cells = _value(entry, "cells", default=None)
        if cells is None:
            tables = _value(entry, "tables", default=None)
            if tables is not None:
                for table in tables:
                    flattened.extend(_flatten_nested_tables([table]))
                continue
            flattened.append(entry)
            continue
        table_id = _text_value(entry, "table_id", "id", default=None)
        page = _int_value(entry, "page", "page_number", default=0) or 0
        caption = _text_value(entry, "caption", "title", default=None)
        for cell in cells:
            cell_entry = dict(cell) if isinstance(cell, Mapping) else {"text": cell}
            cell_entry.setdefault("page", page)
            cell_entry.setdefault("kind", "table_cell")
            if table_id is not None:
                cell_entry.setdefault("table_id", table_id)
            if caption is not None:
                cell_entry.setdefault("table_caption", caption)
            flattened.append(cell_entry)
    return flattened


def _normalise_raw_blocks(source: Any) -> list[_RawBlock]:
    entries = _flatten_nested_tables(_flatten_source(source))
    result: list[_RawBlock] = []
    next_table = 0
    for index, entry in enumerate(entries):
        if isinstance(entry, DocumentNode):
            result.append(
                _RawBlock(
                    index=index,
                    text=entry.text,
                    page=entry.page,
                    bbox=entry.bbox,
                    kind=entry.kind,
                    column=entry.column,
                    row=entry.row,
                    table_id=entry.table_id,
                    order=entry.reading_order,
                    confidence=entry.confidence,
                    metadata=entry.metadata,
                )
            )
            continue
        kind = _normalise_kind(
            _value(entry, "kind", "type", "block_type", "role", default="text")
        )
        key = _text_value(entry, "key", "label", "field", default=None)
        value = _text_value(entry, "value", "answer", default=None)
        text = _text_value(entry, "text", "content", default=None)
        if text is None and key is not None and value is not None:
            text = f"{key}: {value}"
            kind = "form_field"
        elif text is None and value is not None:
            text = value
        if text is None:
            text = ""
        if not text and kind not in {"table", "caption"}:
            continue
        page = _int_value(entry, "page", "page_number", default=0) or 0
        table_id = _text_value(entry, "table_id", "table", default=None)
        if kind == "table" and table_id is None:
            table_id = f"table-{page}-{next_table}"
            next_table += 1
        if kind == "table_cell" and table_id is None:
            table_id = f"table-{page}-0"
        raw_metadata = _value(entry, "metadata", default={}) or {}
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        for name in (
            "source_start",
            "source_end",
            "start",
            "end",
            "line_id",
            "ambiguous",
            "review_required",
            "warnings",
        ):
            value_for_metadata = _value(entry, name, default=None)
            if value_for_metadata is not None and name not in metadata:
                metadata[name] = value_for_metadata
        table_caption = _text_value(entry, "table_caption", "caption", default=None)
        if table_caption is not None:
            metadata.setdefault("table_caption", table_caption)
        result.append(
            _RawBlock(
                index=index,
                text=text,
                page=page,
                bbox=_read_bbox(entry),
                kind=kind,
                column=_int_value(entry, "column", "column_index", default=None),
                row=_int_value(entry, "row", "row_index", default=None),
                table_id=table_id,
                order=_int_value(
                    entry,
                    "reading_order",
                    "order",
                    "sequence",
                    default=None,
                ),
                confidence=_float_value(entry, "confidence", "score", default=None),
                metadata=metadata,
                key=key,
                value=value,
                key_bbox=_coerce_bbox(_value(entry, "key_bbox", default=None))
                if _value(entry, "key_bbox", default=None) is not None
                else None,
                value_bbox=_coerce_bbox(_value(entry, "value_bbox", default=None))
                if _value(entry, "value_bbox", default=None) is not None
                else None,
                form_id=_text_value(entry, "form_id", "field_id", default=None),
            )
        )
    return result


def _cluster(values: Sequence[float], tolerance: float) -> list[int]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    clusters = [0] * len(values)
    current = 0
    previous: float | None = None
    for index in order:
        if previous is not None and values[index] - previous > tolerance:
            current += 1
        clusters[index] = current
        previous = values[index]
    return clusters


def _infer_table_coordinates(
    records: list[_RawBlock],
    *,
    row_tolerance: float,
    column_tolerance: float,
) -> list[_RawBlock]:
    grouped: dict[str, list[_RawBlock]] = defaultdict(list)
    for record in records:
        if record.kind == "table_cell" and record.table_id is not None:
            grouped[record.table_id].append(record)
    replacements: dict[int, _RawBlock] = {}
    for table_records in grouped.values():
        missing_coordinates = any(
            record.row is None or record.column is None for record in table_records
        )
        if not missing_coordinates:
            continue
        y_values = [
            ((record.bbox[1] + record.bbox[3]) / 2) if record.bbox else float(index)
            for index, record in enumerate(table_records)
        ]
        x_values = [
            record.bbox[0] if record.bbox else float(index)
            for index, record in enumerate(table_records)
        ]
        rows = _cluster(y_values, row_tolerance)
        columns = _cluster(x_values, column_tolerance)
        for index, record in enumerate(table_records):
            replacements[record.index] = _RawBlock(
                **{
                    **record.__dict__,
                    "row": record.row if record.row is not None else rows[index],
                    "column": (
                        record.column if record.column is not None else columns[index]
                    ),
                }
            )
    return [replacements.get(record.index, record) for record in records]


def _column_indices(
    records: Sequence[_RawBlock],
    *,
    tolerance: float,
) -> dict[int, int]:
    explicit = [record.column for record in records if record.column is not None]
    if explicit and len(explicit) == len(records):
        ordered = {value: index for index, value in enumerate(sorted(set(explicit)))}
        return {record.index: ordered[record.column] for record in records}  # type: ignore[index]
    representative_by_index: dict[int, float] = {}
    table_representatives: dict[str, float] = {}
    for record in records:
        if record.bbox is None:
            continue
        if record.kind == "table_cell" and record.table_id is not None:
            table_representatives[record.table_id] = min(
                record.bbox[0],
                table_representatives.get(record.table_id, record.bbox[0]),
            )
        else:
            representative_by_index[record.index] = record.bbox[0]
    for record in records:
        if record.kind == "table_cell" and record.table_id is not None:
            representative_by_index[record.index] = table_representatives.get(
                record.table_id,
                record.bbox[0] if record.bbox is not None else 0.0,
            )
        elif record.index not in representative_by_index:
            representative_by_index[record.index] = 0.0
    clusters = _cluster(
        [representative_by_index[record.index] for record in records],
        tolerance,
    )
    return {record.index: cluster for record, cluster in zip(records, clusters)}


def _flow_units(
    records: Sequence[_RawBlock],
    *,
    column_tolerance: float,
) -> list[tuple[tuple[Any, ...], tuple[_RawBlock, ...], int]]:
    """Group cells into row-major table units and retain page flow geometry."""
    by_page: dict[int, list[_RawBlock]] = defaultdict(list)
    for record in records:
        by_page[record.page].append(record)
    page_columns = {
        page: _column_indices(page_records, tolerance=column_tolerance)
        for page, page_records in by_page.items()
    }
    by_table: dict[tuple[int, str], list[_RawBlock]] = defaultdict(list)
    regular: list[_RawBlock] = []
    for record in records:
        if record.kind == "table_cell" and record.table_id is not None:
            by_table[(record.page, record.table_id)].append(record)
        else:
            regular.append(record)
    units: list[tuple[tuple[Any, ...], tuple[_RawBlock, ...], int]] = []
    for record in regular:
        column = page_columns[record.page].get(record.index, 0)
        y, x = _bbox_sort_key(record.bbox)
        explicit = record.order is not None
        key = (
            record.page,
            0 if explicit else 1,
            record.order if explicit else column,
            y,
            x,
            record.index,
        )
        units.append((key, (record,), column))
    for (page, _), table_records in by_table.items():
        table_columns = page_columns[page]
        ordered_cells = tuple(
            sorted(
                table_records,
                key=lambda record: (
                    record.row if record.row is not None else 0,
                    record.column
                    if record.column is not None
                    else table_columns.get(record.index, 0),
                    _bbox_sort_key(record.bbox),
                    record.index,
                ),
            )
        )
        first = min(
            table_records,
            key=lambda record: (_bbox_sort_key(record.bbox), record.index),
        )
        column = table_columns.get(first.index, 0)
        explicit = any(record.order is not None for record in table_records)
        explicit_order = min(
            (record.order for record in table_records if record.order is not None),
            default=0,
        )
        y, x = _bbox_sort_key(first.bbox)
        key = (
            page,
            0 if explicit else 1,
            explicit_order if explicit else column,
            y,
            x,
            first.index,
        )
        units.append((key, ordered_cells, column))
    return sorted(units, key=lambda unit: unit[0])


def _node_form_fields(
    raw_records: Sequence[_RawBlock],
    nodes_by_index: Mapping[int, DocumentNode],
) -> tuple[DocumentFormField, ...]:
    fields: list[DocumentFormField] = []
    grouped: dict[str, list[tuple[_RawBlock, DocumentNode]]] = defaultdict(list)
    for record in raw_records:
        node = nodes_by_index.get(record.index)
        if node is None:
            continue
        if record.key is not None and record.value is not None:
            key_start = node.start
            key_end = key_start + len(record.key)
            value_start = node.start + node.text.find(record.value, len(record.key))
            if value_start < node.start:
                value_start = node.end - len(record.value)
            fields.append(
                DocumentFormField(
                    id=f"{node.id}-field",
                    key=record.key,
                    value=record.value,
                    page=node.page,
                    key_start=key_start,
                    key_end=key_end,
                    value_start=value_start,
                    value_end=value_start + len(record.value),
                    bbox=record.bbox,
                    key_bbox=record.key_bbox or record.bbox,
                    value_bbox=record.value_bbox or record.bbox,
                    confidence=record.confidence,
                    metadata=record.metadata,
                )
            )
        form_id = record.form_id or _text_value(
            record.metadata, "form_id", default=None
        )
        if form_id is not None and record.kind in {"form_key", "form_value"}:
            grouped[form_id].append((record, node))
    for form_id, pairs in grouped.items():
        key_pair = next((item for item in pairs if item[0].kind == "form_key"), None)
        value_pair = next(
            (item for item in pairs if item[0].kind == "form_value"), None
        )
        if key_pair is None or value_pair is None:
            continue
        key_record, key_node = key_pair
        value_record, value_node = value_pair
        fields.append(
            DocumentFormField(
                id=f"form-{form_id}",
                key=key_node.text.strip(),
                value=value_node.text.strip(),
                page=value_node.page,
                key_start=key_node.start,
                key_end=key_node.end,
                value_start=value_node.start,
                value_end=value_node.end,
                bbox=_union_bboxes((key_node.bbox, value_node.bbox)),
                key_bbox=key_record.bbox,
                value_bbox=value_record.bbox,
                confidence=value_record.confidence or key_record.confidence,
                metadata={**dict(key_record.metadata), **dict(value_record.metadata)},
            )
        )
    return tuple(sorted(fields, key=lambda item: (item.value_start, item.id)))


def _infer_line_form_fields(
    nodes: Sequence[DocumentNode],
) -> tuple[DocumentFormField, ...]:
    """Infer simple labelled lines while retaining exact node offsets."""
    fields: list[DocumentFormField] = []
    for node in nodes:
        if node.kind in {"table_cell", "form_key", "form_value"}:
            continue
        match = _FORM_LINE_RE.match(node.text)
        if match is None:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if not key or not value:
            continue
        key_start = node.start + match.start(1)
        key_start += len(match.group(1)) - len(match.group(1).lstrip())
        value_start = node.start + match.start(2)
        value_start += len(match.group(2)) - len(match.group(2).lstrip())
        fields.append(
            DocumentFormField(
                id=f"{node.id}-field",
                key=key,
                value=value,
                page=node.page,
                key_start=key_start,
                key_end=key_start + len(key),
                value_start=value_start,
                value_end=value_start + len(value),
                bbox=node.bbox,
                key_bbox=node.bbox,
                value_bbox=node.bbox,
                confidence=node.confidence,
                metadata=node.metadata,
            )
        )
    return tuple(fields)


def build_document_graph(
    source: Any,
    *,
    source_text: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    row_tolerance: float = 4.0,
    column_tolerance: float = 18.0,
) -> DocumentGraph:
    """Build a deterministic graph from PDF/OCR blocks or an OCR result.

    Accepted inputs include a sequence of mappings, an object exposing
    ``blocks``/``pages``/``words``, an :class:`ExtractedDocument`, or a mapping
    with nested table cells.  Each block may provide ``page``, ``bbox`` or
    ``x0``/``top``/``x1``/``bottom``, ``column``, ``row``, ``table_id``,
    ``reading_order``, ``confidence``, and ``metadata``.
    """
    if isinstance(source, DocumentGraph):
        return source
    if isinstance(source, Mapping):
        if bool(_value(source, "encrypted", "is_encrypted", default=False)):
            raise EncryptedDocumentError(
                "encrypted document inputs require an explicit local decryption step"
            )
        if bool(_value(source, "malformed", "invalid", default=False)):
            raise MalformedDocumentError("document input is marked malformed")
    if isinstance(source, (str, Path)):
        candidate = Path(source)
        if candidate.suffix.lower() == ".pdf" and candidate.is_file():
            return extract_pdf_graph(
                candidate,
                metadata=metadata,
                row_tolerance=row_tolerance,
                column_tolerance=column_tolerance,
            )
    if row_tolerance < 0 or column_tolerance < 0:
        raise ValueError("layout tolerances must be non-negative")
    records = _normalise_raw_blocks(source)
    records = _infer_table_coordinates(
        records,
        row_tolerance=row_tolerance,
        column_tolerance=column_tolerance,
    )
    if not records:
        return DocumentGraph(text="", metadata=dict(metadata or {}))

    units = _flow_units(records, column_tolerance=column_tolerance)
    ordered_records = [record for _, unit, _ in units for record in unit]
    nodes: list[DocumentNode] = []
    nodes_by_index: dict[int, DocumentNode] = {}
    used_ids: set[str] = set()
    parts: list[str] = []
    cursor = 0
    for reading_order, record in enumerate(ordered_records):
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(record.text)
        cursor += len(record.text)
        requested_id = _text_value(record.metadata, "id", "node_id", default=None)
        node_id = requested_id or f"page-{record.page}-node-{reading_order}"
        if node_id in used_ids:
            node_id = f"{node_id}-{reading_order}"
        used_ids.add(node_id)
        node = DocumentNode(
            id=node_id,
            kind=record.kind,
            text=record.text,
            page=record.page,
            bbox=record.bbox,
            start=start,
            end=cursor,
            reading_order=reading_order,
            column=next(
                (
                    column
                    for key, unit, column in units
                    if any(item.index == record.index for item in unit)
                ),
                record.column,
            ),
            table_id=record.table_id,
            row=record.row,
            column_index=record.column,
            confidence=record.confidence,
            metadata=record.metadata,
        )
        nodes.append(node)
        nodes_by_index[record.index] = node

    graph_text = "".join(parts)
    if source_text is not None and not isinstance(source_text, str):
        raise TypeError("source_text must be a string")
    graph_metadata = dict(metadata or {})
    graph_metadata.setdefault("node_count", len(nodes))
    graph_metadata.setdefault("page_count", len({node.page for node in nodes}))
    graph_metadata["text_sha256"] = hashlib.sha256(
        graph_text.encode("utf-8")
    ).hexdigest()

    table_groups: dict[str, list[DocumentTableCell]] = defaultdict(list)
    table_captions: dict[str, str | None] = {}
    for record in ordered_records:
        if record.kind != "table_cell" or record.table_id is None:
            continue
        node = nodes_by_index[record.index]
        table_groups[record.table_id].append(
            DocumentTableCell(
                id=node.id,
                table_id=record.table_id,
                row=record.row or 0,
                column=record.column or 0,
                text=node.text,
                page=node.page,
                bbox=node.bbox,
                start=node.start,
                end=node.end,
                colspan=max(1, int(_value(record.metadata, "colspan", default=1))),
                is_header=bool(
                    _value(record.metadata, "is_header", "header", default=False)
                ),
                confidence=node.confidence,
                metadata=node.metadata,
            )
        )
        table_captions.setdefault(
            record.table_id,
            _text_value(record.metadata, "table_caption", "caption", default=None),
        )
    tables = tuple(
        DocumentTable(
            id=table_id,
            page=min(cell.page for cell in cells),
            bbox=_union_bboxes(cell.bbox for cell in cells),
            cells=tuple(
                sorted(cells, key=lambda cell: (cell.row, cell.column, cell.id))
            ),
            start=min(cell.start for cell in cells),
            end=max(cell.end for cell in cells),
            caption=table_captions.get(table_id),
        )
        for table_id, cells in sorted(
            table_groups.items(),
            key=lambda item: (
                min(cell.page for cell in item[1]),
                min(cell.start for cell in item[1]),
                item[0],
            ),
        )
    )
    explicit_form_fields = _node_form_fields(records, nodes_by_index)
    inferred_form_fields = _infer_line_form_fields(nodes)
    known_value_ranges = {
        (field.value_start, field.value_end) for field in explicit_form_fields
    }
    form_fields = tuple(
        sorted(
            (
                *explicit_form_fields,
                *tuple(
                    field
                    for field in inferred_form_fields
                    if (field.value_start, field.value_end) not in known_value_ranges
                ),
            ),
            key=lambda item: (item.value_start, item.id),
        )
    )
    captions = tuple(node for node in nodes if node.kind == "caption")

    page_nodes: dict[int, list[DocumentNode]] = defaultdict(list)
    for node in nodes:
        page_nodes[node.page].append(node)
    pages = tuple(
        DocumentPage(number=page, node_ids=tuple(node.id for node in page_nodes[page]))
        for page in sorted(page_nodes)
    )

    column_nodes: dict[tuple[int, int], list[DocumentNode]] = defaultdict(list)
    for node in nodes:
        column_nodes[(node.page, node.column or 0)].append(node)
    columns = tuple(
        DocumentColumn(
            page=page,
            index=index,
            bbox=_union_bboxes(node.bbox for node in column_nodes[(page, index)]),
            nodes=tuple(column_nodes[(page, index)]),
        )
        for page, index in sorted(column_nodes)
    )
    return DocumentGraph(
        text=graph_text,
        nodes=tuple(nodes),
        pages=pages,
        columns=columns,
        tables=tables,
        form_fields=form_fields,
        captions=captions,
        metadata=graph_metadata,
    )


def _bbox_contains(container: BBox, candidate: BBox) -> bool:
    return (
        candidate[0] >= container[0]
        and candidate[1] >= container[1]
        and candidate[2] <= container[2]
        and candidate[3] <= container[3]
    )


def _pdf_line_blocks(
    page: Any,
    page_number: int,
    *,
    excluded_bboxes: Sequence[BBox] = (),
) -> list[dict[str, Any]]:
    try:
        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
        )
    except TypeError:
        words = page.extract_words()
    cleaned = [
        word
        for word in words
        if str(_value(word, "text", default="")).strip()
        and not any(
            _bbox_contains(
                box,
                (
                    float(_value(word, "x0", default=0)),
                    float(_value(word, "top", default=0)),
                    float(_value(word, "x1", default=0)),
                    float(_value(word, "bottom", default=0)),
                ),
            )
            for box in excluded_bboxes
        )
    ]
    if not cleaned:
        return []
    heights = [
        float(_value(word, "bottom", default=0)) - float(_value(word, "top", default=0))
        for word in cleaned
    ]
    tolerance = max(2.0, (sorted(heights)[len(heights) // 2]) * 0.6)
    lines: list[list[Mapping[str, Any]]] = []
    for word in sorted(
        cleaned,
        key=lambda item: (
            float(_value(item, "top", default=0)),
            float(_value(item, "x0", default=0)),
        ),
    ):
        top = float(_value(word, "top", default=0))
        for line in lines:
            first_top = float(_value(line[0], "top", default=0))
            if abs(top - first_top) <= tolerance:
                line.append(word)
                break
        else:
            lines.append([word])
    blocks: list[dict[str, Any]] = []
    for line in lines:
        ordered = sorted(line, key=lambda item: float(_value(item, "x0", default=0)))
        segments: list[list[Mapping[str, Any]]] = []
        for word in ordered:
            if not segments:
                segments.append([word])
                continue
            previous = segments[-1][-1]
            gap = float(_value(word, "x0", default=0)) - float(
                _value(previous, "x1", default=0)
            )
            if gap > max(24.0, tolerance * 4.0):
                segments.append([word])
            else:
                segments[-1].append(word)
        for segment in segments:
            text = " ".join(str(_value(word, "text", default="")) for word in segment)
            bbox = (
                min(float(_value(word, "x0", default=0)) for word in segment),
                min(float(_value(word, "top", default=0)) for word in segment),
                max(float(_value(word, "x1", default=0)) for word in segment),
                max(float(_value(word, "bottom", default=0)) for word in segment),
            )
            blocks.append(
                {
                    "text": text,
                    "page": page_number,
                    "bbox": bbox,
                    "kind": "text",
                }
            )
    return blocks


def _pdf_table_blocks(page: Any, page_number: int) -> list[dict[str, Any]]:
    """Return ruled-table cells when the local PDF parser exposes them."""
    find_tables = getattr(page, "find_tables", None)
    if not callable(find_tables):
        return []
    try:
        tables = tuple(find_tables())
    except Exception:
        return []
    blocks: list[dict[str, Any]] = []
    for table_index, table in enumerate(tables):
        table_id = f"pdf-table-{page_number}-{table_index}"
        rows = tuple(getattr(table, "rows", ()) or ())
        extracted = tuple(getattr(table, "extract", lambda: ())() or ())
        for row_index, row in enumerate(rows):
            cells = tuple(getattr(row, "cells", ()) or ())
            values = tuple(extracted[row_index]) if row_index < len(extracted) else ()
            for column_index, cell in enumerate(cells):
                if cell is None:
                    continue
                value = values[column_index] if column_index < len(values) else ""
                if value is None or not str(value).strip():
                    continue
                blocks.append(
                    {
                        "text": str(value).strip(),
                        "page": page_number,
                        "bbox": tuple(float(item) for item in cell),
                        "kind": "table_cell",
                        "table_id": table_id,
                        "row": row_index,
                        "column": column_index,
                        "metadata": {
                            "is_header": row_index == 0,
                            "table_bbox": getattr(table, "bbox", None),
                        },
                    }
                )
    return blocks


def extract_pdf_graph(
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    row_tolerance: float = 4.0,
    column_tolerance: float = 18.0,
) -> DocumentGraph:
    """Extract a digital PDF into a graph, failing closed on unsafe inputs.

    The parser never submits document bytes elsewhere.  Encrypted PDFs and
    malformed files raise explicit exceptions before any graph is returned.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise MalformedDocumentError("PDF input is not a readable regular file")
    try:
        with pdf_path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise MalformedDocumentError("PDF input failed signature validation")
    except OSError as exc:
        raise MalformedDocumentError("PDF input could not be read") from exc
    try:
        pdfplumber = importlib.import_module("pdfplumber")
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise MissingDependencyError(
            dependency="pdfplumber",
            instruction='Install with: pip install "openmed[multimodal]".',
        ) from exc
    blocks: list[dict[str, Any]] = []
    page_metadata: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if bool(getattr(pdf, "is_encrypted", False)):
                raise EncryptedDocumentError(
                    "encrypted PDF inputs require an explicit local decryption step"
                )
            pages = tuple(getattr(pdf, "pages", ()))
            if not pages:
                raise MalformedDocumentError("PDF input contains no pages")
            for page_number, page in enumerate(pages):
                table_blocks = _pdf_table_blocks(page, page_number)
                blocks.extend(table_blocks)
                table_bboxes = tuple(
                    box
                    for block in table_blocks
                    if isinstance(
                        metadata_value := _value(block, "metadata", default={}),
                        Mapping,
                    )
                    for box in (_coerce_bbox(metadata_value.get("table_bbox")),)
                    if box is not None
                )
                blocks.extend(
                    _pdf_line_blocks(
                        page,
                        page_number,
                        excluded_bboxes=table_bboxes,
                    )
                )
                page_metadata.append(
                    {
                        "page": page_number,
                        "width": _value(page, "width", default=None),
                        "height": _value(page, "height", default=None),
                    }
                )
    except (EncryptedDocumentError, MalformedDocumentError, MissingDependencyError):
        raise
    except Exception as exc:  # parser-specific exceptions must fail closed.
        message = f"{type(exc).__name__} {exc}".lower()
        if "encrypt" in message or "password" in message:
            raise EncryptedDocumentError(
                "encrypted PDF inputs require an explicit local decryption step"
            ) from exc
        raise MalformedDocumentError("PDF input could not be parsed safely") from exc
    graph = build_document_graph(
        blocks,
        metadata={**dict(metadata or {}), "format": "pdf", "pages": page_metadata},
        row_tolerance=row_tolerance,
        column_tolerance=column_tolerance,
    )
    return graph


def graph_from_ocr(result: Any, **kwargs: Any) -> DocumentGraph:
    """Alias for :func:`build_document_graph` for OCR adapters."""
    return build_document_graph(result, **kwargs)


def extract_document_graph(source: Any, **kwargs: Any) -> DocumentGraph:
    """Dispatch a PDF path or synthetic/OCR source to the graph builder."""
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".pdf":
        return extract_pdf_graph(source, **kwargs)
    return build_document_graph(source, **kwargs)


document_graph = build_document_graph
parse_pdf_document = extract_pdf_graph
ingest_document_graph = extract_document_graph


__all__ = [
    "BBox",
    "BoundingBox",
    "Caption",
    "Column",
    "DocumentBlock",
    "DocumentColumn",
    "DocumentFormField",
    "DocumentGraph",
    "DocumentGraphBuilder",
    "DocumentNode",
    "DocumentPage",
    "DocumentTable",
    "DocumentTableCell",
    "FormFieldNode",
    "Table",
    "TableCell",
    "TextBlock",
    "SourceRegion",
    "build_document_graph",
    "document_graph",
    "extract_pdf_graph",
    "extract_document_graph",
    "graph_from_ocr",
    "ingest_document_graph",
    "parse_pdf_document",
]
