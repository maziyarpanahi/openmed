"""PDF text extraction and coordinate projection for multimodal redaction.

The PDF ingester uses pdfplumber lazily so importing :mod:`openmed.multimodal`
does not pull optional dependencies into the base install. It extracts words in
source or automatically reconstructed column-major reading order, records one
:class:`~openmed.multimodal.base.SourceSpan` per word, and can project detected
PHI character spans back to page rectangles.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan, register_handler
from .documents_pdf_layout import (
    PdfPageLayout,
    PdfReadingOrder,
    detect_pdf_columns,
    validate_pdf_reading_order,
)
from .exceptions import MissingDependencyError

_PDFPLUMBER_HINT = 'Install with: pip install "openmed[multimodal]".'
_PDF_WORD_FIELDS = ("x0", "top", "x1", "bottom")


@dataclass(frozen=True)
class ProjectedRectangle:
    """A source-page rectangle covering one detected text span."""

    start: int
    end: int
    page: int
    bbox: tuple[float, float, float, float]
    label: str | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-safe metadata representation."""
        payload: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": self.bbox,
        }
        if self.label is not None:
            payload["label"] = self.label
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def _import_pdfplumber() -> Any:
    try:
        return importlib.import_module("pdfplumber")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="pdfplumber", instruction=_PDFPLUMBER_HINT
        ) from exc


def extract_pdf(
    path: str | Path, *, reading_order: PdfReadingOrder = "auto"
) -> ExtractedDocument:
    """Extract normalized PDF text plus char-offset source spans.

    Each pdfplumber word is joined with a single space on its page; pages are
    joined with newlines. ``reading_order="auto"`` reconstructs only pages with
    confidently repeated column gutters. ``reading_order="source"`` preserves
    the original OM-060 word sequence. Single-column auto output is exactly the
    source-order output. Source spans always retain the original 0-based page
    indexes and pdfplumber bounding boxes in PDF coordinate units.

    Args:
        path: Local PDF path. Document bytes are never sent elsewhere.
        reading_order: ``"auto"`` for conservative multi-column reconstruction
            or ``"source"`` for the original pdfplumber text-flow order.

    Returns:
        Extracted text with one bbox-preserving source span per word.

    Raises:
        ValueError: If ``reading_order`` is unsupported.
    """
    reading_order = validate_pdf_reading_order(reading_order)
    pdfplumber = _import_pdfplumber()
    parts: list[str] = []
    spans: list[SourceSpan] = []
    reconstructed_layouts: list[tuple[int, PdfPageLayout]] = []
    cursor = 0
    page_count = 0
    word_count = 0

    with pdfplumber.open(path) as pdf:
        pages = tuple(getattr(pdf, "pages", ()))
        page_count = len(pages)
        for page_index, page in enumerate(pages):
            words = _extract_page_words(page)
            if not words:
                continue
            layout = _page_layout(page, words, reading_order=reading_order)
            if layout is not None and layout.is_multicolumn:
                reconstructed_layouts.append((page_index, layout))
                word_indexes = layout.reading_order
            else:
                word_indexes = tuple(range(len(words)))
            if parts:
                parts.append("\n")
                cursor += 1
            for word_index, source_word_index in enumerate(word_indexes):
                word = words[source_word_index]
                if word_index > 0:
                    parts.append(" ")
                    cursor += 1
                text = str(word.get("text", "")).strip()
                start = cursor
                parts.append(text)
                cursor += len(text)
                bbox = _word_bbox(word)
                span_metadata: dict[str, Any] = {
                    "format": "pdf",
                    "block_type": "word",
                    "page_word_index": word_index,
                    "document_word_index": word_count,
                }
                if layout is not None and layout.is_multicolumn:
                    span_metadata["source_page_word_index"] = source_word_index
                    column_index = layout.word_columns[source_word_index]
                    if column_index is None:
                        span_metadata["spans_columns"] = True
                    else:
                        span_metadata["column_index"] = column_index
                spans.append(
                    SourceSpan(
                        start=start,
                        end=cursor,
                        page=page_index,
                        bbox=bbox,
                        metadata=span_metadata,
                    )
                )
                word_count += 1

    metadata: dict[str, Any] = {
        "format": "pdf",
        "page_count": page_count,
        "word_count": word_count,
    }
    if reconstructed_layouts:
        metadata.update(
            {
                "reading_order": "column-major",
                "reading_order_reconstructed": True,
                "reconstructed_page_count": len(reconstructed_layouts),
                "page_layouts": tuple(
                    {
                        "page": page_index,
                        "column_count": layout.column_count,
                        "column_boundaries": layout.column_boundaries,
                    }
                    for page_index, layout in reconstructed_layouts
                ),
            }
        )
    return ExtractedDocument(
        text="".join(parts),
        spans=tuple(spans),
        metadata=metadata,
    )


def project_text_spans(
    document: ExtractedDocument,
    spans: Iterable[Any],
    *,
    line_tolerance: float = 2.0,
) -> tuple[ProjectedRectangle, ...]:
    """Project detected char spans to PDF page rectangles.

    ``spans`` accepts objects, mappings, or ``(start, end)`` tuples with
    ``start``/``end`` offsets. Words are grouped into line-level rectangles so a
    span crossing a line break emits one rectangle per line instead of one tall
    page box.
    """
    rectangles: list[ProjectedRectangle] = []
    for span in spans:
        entity = _coerce_entity(span)
        if entity is None:
            continue
        start, end, label, confidence = entity
        if end <= start:
            continue
        covered = [
            source
            for source in document.spans
            if source.bbox is not None and source.end > start and source.start < end
        ]
        if not covered:
            continue
        rectangles.extend(
            _line_rectangles(
                covered,
                start=start,
                end=end,
                label=label,
                confidence=confidence,
                text=document.text[start:end],
                line_tolerance=line_tolerance,
            )
        )
    return tuple(rectangles)


def _extract_page_words(page: Any) -> tuple[Mapping[str, Any], ...]:
    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=True,
    )
    return tuple(word for word in words if str(word.get("text", "")).strip())


def _page_layout(
    page: Any,
    words: Sequence[Mapping[str, Any]],
    *,
    reading_order: PdfReadingOrder,
) -> PdfPageLayout | None:
    if reading_order == "source":
        return None
    raw_width = getattr(page, "width", None)
    try:
        page_width = float(raw_width) if raw_width is not None else None
        if page_width is not None and page_width <= 0:
            page_width = None
        return detect_pdf_columns(words, page_width=page_width)
    except ValueError:
        # Auto layout is deliberately fail-soft. The legacy extractor still
        # validates and projects every source bbox below, so an uncertain page
        # never gains a new failure mode merely because reconstruction is on.
        return None


def _word_bbox(word: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(word[field]) for field in _PDF_WORD_FIELDS)  # type: ignore[return-value]


def _coerce_entity(
    span: Any,
) -> tuple[int, int, str | None, float | None] | None:
    if isinstance(span, Sequence) and not isinstance(span, (str, bytes, bytearray)):
        if len(span) >= 2:
            return int(span[0]), int(span[1]), None, None
        return None

    if isinstance(span, Mapping):
        start = span.get("start")
        end = span.get("end")
        label = span.get("label", span.get("entity_type"))
        confidence = span.get("confidence", span.get("score"))
    else:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        label = getattr(span, "label", getattr(span, "entity_type", None))
        confidence = getattr(span, "confidence", getattr(span, "score", None))

    if start is None or end is None:
        return None
    return (
        int(start),
        int(end),
        _coerce_optional_str(label),
        _coerce_confidence(confidence),
    )


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _line_rectangles(
    spans: Sequence[SourceSpan],
    *,
    start: int,
    end: int,
    label: str | None,
    confidence: float | None,
    text: str,
    line_tolerance: float,
) -> tuple[ProjectedRectangle, ...]:
    lines: list[list[SourceSpan]] = []
    for span in sorted(spans, key=lambda item: (item.page, item.bbox[1], item.bbox[0])):  # type: ignore[index]
        for line in lines:
            if _same_line(line[0], span, tolerance=line_tolerance):
                line.append(span)
                break
        else:
            lines.append([span])

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return tuple(
        ProjectedRectangle(
            start=start,
            end=end,
            page=line[0].page,
            bbox=_union_bbox(source.bbox for source in line if source.bbox is not None),
            label=label,
            confidence=confidence,
            metadata={
                "text_sha256": text_hash,
                "source_span_count": len(line),
            },
        )
        for line in lines
    )


def _same_line(first: SourceSpan, second: SourceSpan, *, tolerance: float) -> bool:
    if first.page != second.page or first.bbox is None or second.bbox is None:
        return False
    first_top, first_bottom = first.bbox[1], first.bbox[3]
    second_top, second_bottom = second.bbox[1], second.bbox[3]
    overlaps = min(first_bottom, second_bottom) - max(first_top, second_top)
    if overlaps >= 0:
        return True
    first_center = (first_top + first_bottom) / 2.0
    second_center = (second_top + second_bottom) / 2.0
    return abs(first_center - second_center) <= tolerance


def _union_bbox(
    bboxes: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    boxes = tuple(bboxes)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _detect_entities(document: ExtractedDocument, models: Any, lang: str | None) -> Any:
    detector = _resolve_detector(models)
    if detector is None:
        return ()
    try:
        return detector(document.text, lang=lang)
    except TypeError:
        return detector(document.text)


def _resolve_detector(models: Any) -> Any:
    if models is None:
        return None
    if callable(models):
        return models
    if isinstance(models, Mapping):
        for key in ("detector", "extract_pii", "analyze_text", "predict_entities"):
            candidate = models.get(key)
            if callable(candidate):
                return candidate
        return None
    for name in (
        "detect",
        "extract_pii",
        "analyze_text",
        "predict_entities",
        "predict",
    ):
        candidate = getattr(models, name, None)
        if callable(candidate):
            return candidate
    return None


def _iter_entities(result: Any) -> tuple[Any, ...]:
    if result is None:
        return ()
    entities = getattr(result, "entities", None)
    if entities is not None:
        return tuple(entities)
    pii_entities = getattr(result, "pii_entities", None)
    if pii_entities is not None:
        return tuple(pii_entities)
    if isinstance(result, Mapping):
        for key in ("entities", "pii_entities", "spans"):
            entities = result.get(key)
            if entities is not None:
                return tuple(entities)
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray)):
        return tuple(result)
    return ()


def _pdf_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    document = extract_pdf(path)
    # Keep this bbox-preserving extraction as the canonical text/offset map
    # while adding structured boxes for table cells and caption lines.
    from .documents_pdf_tables import extract_pdf_regions, project_structured_spans

    regions = extract_pdf_regions(path, document=document)
    entities = _iter_entities(_detect_entities(document, models, lang))
    rectangles = project_structured_spans(document, regions, entities)
    metadata = dict(document.metadata)
    metadata.update(
        {
            "table_regions": [
                table.to_dict(include_text=False) for table in regions.tables
            ],
            "caption_regions": [
                caption.to_dict(include_text=False) for caption in regions.captions
            ],
        }
    )
    if rectangles:
        metadata.update(
            {
                "detected_span_count": len(entities),
                "redaction_rectangles": [
                    rectangle.to_dict() for rectangle in rectangles
                ],
            }
        )
    if policy is not None:
        metadata["policy"] = policy
    if metadata == document.metadata:
        return document
    return ExtractedDocument(
        text=document.text,
        spans=document.spans,
        metadata=metadata,
    )


register_handler(".pdf", _pdf_handler)


__all__ = [
    "ProjectedRectangle",
    "extract_pdf",
    "project_text_spans",
]
