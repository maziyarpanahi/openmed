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
import inspect
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from itertools import islice
from numbers import Integral
from pathlib import Path
from typing import Any, BinaryIO

from .base import ExtractedDocument, SourceSpan, register_handler
from .documents_pdf_layout import (
    PdfPageLayout,
    PdfReadingOrder,
    detect_pdf_columns,
    validate_pdf_reading_order,
)
from .exceptions import MissingDependencyError, UnsupportedDocumentError

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


def _snapshot_pdf_source(source: str | Path | BinaryIO) -> bytes:
    """Read one immutable input without closing a caller-owned binary stream."""
    try:
        if isinstance(source, (str, Path)):
            return Path(source).read_bytes()
        source.seek(0)
        content = source.read()
        if not isinstance(content, bytes):
            raise TypeError
        return bytes(content)
    except Exception:
        raise ValueError("pdf_source_unreadable") from None


def extract_pdf(
    path: str | Path | BinaryIO,
    *,
    reading_order: PdfReadingOrder = "auto",
    preserve_lines: bool = False,
    include_annotations: bool = False,
) -> ExtractedDocument:
    """Extract normalized PDF text plus char-offset source spans.

    Each pdfplumber word is joined with a single space on its page; pages are
    joined with newlines. ``reading_order="auto"`` reconstructs only pages with
    confidently repeated column gutters. ``reading_order="source"`` preserves
    the original OM-060 word sequence. Single-column auto output is exactly the
    source-order output. Source spans always retain the original 0-based page
    indexes and pdfplumber bounding boxes in PDF coordinate units.

    Args:
        path: Local PDF path or seekable binary stream. Document bytes are
            never sent elsewhere; caller-owned streams remain open.
        reading_order: ``"auto"`` for conservative multi-column reconstruction
            or ``"source"`` for the original pdfplumber text-flow order.
        preserve_lines: Replace inter-word spaces with newlines at visual line
            or reconstructed column boundaries. This retains clinical section
            headings without changing word order, offsets, or page rectangles.
            The default preserves the legacy single-space-per-page contract.
        include_annotations: Opt into annotation Contents and mapped appearance
            text. Unsupported appearances fail closed. Annotation spans cover
            their entire appearance rectangle, not just individual glyphs.

    Returns:
        Extracted text with one bbox-preserving source span per word.

    Raises:
        ValueError: If a control is invalid, or line preservation encounters
            nonfinite or nonpositive word geometry.
    """
    reading_order = validate_pdf_reading_order(reading_order)
    if type(preserve_lines) is not bool:
        raise ValueError("preserve_lines must be a boolean")
    if type(include_annotations) is not bool:
        raise ValueError("include_annotations must be a boolean")
    if include_annotations:
        try:
            content = _snapshot_pdf_source(path)
        except ValueError:
            raise ValueError("annotation_appearance_unmappable") from None
        prepared = _prepare_annotation_source(
            BytesIO(content), include_annotations=True
        )
        document = extract_pdf(
            BytesIO(content), reading_order=reading_order, preserve_lines=preserve_lines
        )
        separator = "\n" if document.text and prepared.document.text else ""
        offset = len(document.text) + len(separator)
        return ExtractedDocument(
            text=document.text + separator + prepared.document.text,
            spans=document.spans
            + tuple(
                SourceSpan(
                    start=span.start + offset,
                    end=span.end + offset,
                    page=span.page,
                    bbox=span.bbox,
                    metadata=span.metadata,
                )
                for span in prepared.document.spans
            ),
            metadata={**document.metadata, "annotations": list(prepared.report)},
        )
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
            previous_bbox = None
            previous_column = None
            for word_index, source_word_index in enumerate(word_indexes):
                word = words[source_word_index]
                bbox = _word_bbox(word)
                column = (
                    layout.word_columns[source_word_index]
                    if layout is not None and layout.is_multicolumn
                    else None
                )
                if preserve_lines and (
                    not all(math.isfinite(value) for value in bbox)
                    or bbox[2] <= bbox[0]
                    or bbox[3] <= bbox[1]
                ):
                    raise ValueError("line preservation requires valid word geometry")
                if word_index > 0:
                    new_line = preserve_lines and (
                        column != previous_column
                        or not _same_text_line(previous_bbox, bbox)
                    )
                    parts.append("\n" if new_line else " ")
                    cursor += 1
                text = str(word.get("text", "")).strip()
                start = cursor
                parts.append(text)
                cursor += len(text)
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
                previous_bbox, previous_column = bbox, column

    metadata: dict[str, Any] = {
        "format": "pdf",
        "page_count": page_count,
        "word_count": word_count,
    }
    if preserve_lines:
        metadata["line_breaks_preserved"] = True
        metadata["line_break_method"] = "word-vertical-overlap-v1"
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


@dataclass(frozen=True)
class _AnnotationSource:
    content: bytes = field(repr=False)
    report: tuple[Mapping[str, Any], ...]
    document: ExtractedDocument = field(repr=False)


_ANNOTATION_SUBTYPES = frozenset(
    {"FreeText", "Stamp", "Text", "Widget", "Link", "Highlight", "Popup", "Ink"}
)
_ANNOTATION_TEXT_OPERATORS = frozenset(
    "q Q cm BT ET Tc Tw Tz TL Tf Tr Ts Td TD Tm T* Tj TJ ' \" g rg k G RG K".split()
)


def _prepare_annotation_source(
    source: Any, *, include_annotations: bool = False
) -> _AnnotationSource:
    """Make an in-memory raster source with no unchecked annotation objects."""
    try:
        pikepdf = importlib.import_module("pikepdf")
    except ImportError as exc:
        raise MissingDependencyError(
            dependency="pikepdf", instruction=_PDFPLUMBER_HINT
        ) from exc
    parts: list[str] = []
    spans: list[SourceSpan] = []
    reports: list[Mapping[str, Any]] = []
    cursor = 0
    _rewind(source)
    try:
        with pikepdf.open(source) as pdf:
            for page_index, page in enumerate(pdf.pages):
                annotations = tuple(page.obj.get("/Annots", ()))
                if len(annotations) > 10_000:
                    raise ValueError("annotation_appearance_unmappable")
                counts = Counter()
                for annotation_index, annotation in enumerate(annotations):
                    subtype = str(annotation.get("/Subtype", "")).lstrip("/")
                    subtype = subtype if subtype in _ANNOTATION_SUBTYPES else "Other"
                    counts[subtype] += 1
                    if not include_annotations:
                        continue
                    text, bbox = _annotation_text_and_bbox(
                        pdf, page_index, annotation_index, pikepdf
                    )
                    if parts:
                        parts.append("\n")
                        cursor += 1
                    start = cursor
                    parts.append(text)
                    cursor += len(text)
                    spans.append(
                        SourceSpan(
                            start=start,
                            end=cursor,
                            page=page_index,
                            bbox=bbox,
                            metadata={"block_type": "annotation", "subtype": subtype},
                        )
                    )
                reports.append(
                    {
                        "page": page_index,
                        "annotation_count": len(annotations),
                        "omitted_annotation_count": 0
                        if include_annotations
                        else len(annotations),
                        "subtypes": dict(sorted(counts.items())),
                    }
                )
                if not include_annotations and "/Annots" in page.obj:
                    del page.obj["/Annots"]
            if include_annotations:
                pdf.flatten_annotations(mode="all")
                if any(page.obj.get("/Annots") for page in pdf.pages):
                    raise ValueError("annotation_appearance_unmappable")
            if "/AcroForm" in pdf.Root:
                del pdf.Root["/AcroForm"]
            output = BytesIO()
            pdf.save(output)
    except MissingDependencyError:
        raise
    except Exception:
        raise ValueError("annotation_appearance_unmappable") from None
    return _AnnotationSource(
        content=output.getvalue(),
        report=tuple(reports),
        document=ExtractedDocument(text="".join(parts), spans=tuple(spans)),
    )


def _annotation_text_and_bbox(
    pdf: Any, page_index: int, annotation_index: int, pikepdf: Any
) -> tuple[str, tuple[float, float, float, float]]:
    """Map one normal appearance through the PDF engine before trusting it."""
    with pikepdf.Pdf.new() as probe:
        probe.pages.append(pdf.pages[page_index])
        page = probe.pages[0]
        annotation = page.Annots[annotation_index]
        appearance = annotation.get("/AP", {}).get("/N")
        if isinstance(appearance, pikepdf.Dictionary):
            appearance = appearance.get(annotation.get("/AS"))
        if not isinstance(appearance, pikepdf.Stream):
            raise ValueError("annotation_appearance_unmappable")
        bounds = tuple(float(value) for value in appearance.get("/BBox", ()))
        matrix = tuple(
            float(value) for value in appearance.get("/Matrix", (1, 0, 0, 1, 0, 0))
        )
        if (
            len(bounds) != 4
            or len(matrix) != 6
            or not all(math.isfinite(value) for value in (*bounds, *matrix))
            or bounds[2] <= bounds[0]
            or bounds[3] <= bounds[1]
            or matrix[0] * matrix[3] == matrix[1] * matrix[2]
        ):
            raise ValueError("annotation_appearance_unmappable")
        # The normal appearance is the only state retained in a flattened PDF.
        # Image/nested-form appearances need OCR, not text-only authorization.
        resources = appearance.get("/Resources", {})
        if resources.get("/XObject") or any(
            str(font.get("/Subtype")) == "/Type3"
            for font in resources.get("/Font", {}).values()
        ):
            raise ValueError("annotation_appearance_unmappable")
        # Unknown paint operators can carry vectorized or inline-image PHI
        # invisible to text extraction. Only mapped text appearances qualify.
        if any(
            str(instruction.operator) not in _ANNOTATION_TEXT_OPERATORS
            for instruction in pikepdf.parse_content_stream(appearance)
        ):
            raise ValueError("annotation_appearance_unmappable")
        page.Annots = pikepdf.Array([annotation])
        page.Contents = probe.make_stream(b"")
        raw = BytesIO()
        probe.save(raw)
        raw.seek(0)
        pdfplumber = _import_pdfplumber()
        with pdfplumber.open(raw) as measured:
            mapped = measured.pages[0].annots
            if len(mapped) != 1:
                raise ValueError("annotation_appearance_unmappable")
            measured_page = measured.pages[0]
            bbox = _word_bbox(mapped[0])
            width, height = measured_page.width, measured_page.height
            # Both raster paths use the full-page top-origin coordinate map.
            # A distinct crop or shifted media origin needs a separate mapping
            # contract; never guess or retain an unchecked cropped appearance.
            if measured_page.cropbox != measured_page.mediabox or tuple(
                measured_page.mediabox[:2]
            ) != (0, 0):
                raise ValueError("annotation_appearance_unmappable")
        if (
            not all(math.isfinite(value) for value in bbox)
            or not 0 <= bbox[0] < bbox[2] <= width
            or not 0 <= bbox[1] < bbox[3] <= height
        ):
            raise ValueError("annotation_appearance_unmappable")
        contents = str(annotation.get("/Contents", ""))
        probe.flatten_annotations(mode="all")
        if page.obj.get("/Annots"):
            raise ValueError("annotation_appearance_unmappable")
        flattened = BytesIO()
        probe.save(flattened)
        flattened.seek(0)
        with pdfplumber.open(flattened) as measured:
            words = _extract_page_words(measured.pages[0])
            characters = tuple(measured.pages[0].chars)
        if (
            not words
            or not characters
            or any(
                not all(math.isfinite(value) for value in _word_bbox(item))
                or item["x0"] < bbox[0]
                or item["top"] < bbox[1]
                or item["x1"] > bbox[2]
                or item["bottom"] > bbox[3]
                or "(cid:" in str(item.get("text", ""))
                for item in (*words, *characters)
            )
        ):
            raise ValueError("annotation_appearance_unmappable")
        appearance_text = " ".join(str(word["text"]) for word in words)
        # Rotated/mirrored glyphs can be split into individual visual words.
        # Also expose their PDF text-operation order so a detector still sees
        # complete identifiers, while both views map to the full extent.
        operation_text = "".join(str(char["text"]) for char in characters)
        text_views = dict.fromkeys((contents, appearance_text, operation_text))
        return "\n".join(text for text in text_views if text), bbox


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


def _same_text_line(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float],
) -> bool:
    if first is None:
        return False
    overlap = min(first[3], second[3]) - max(first[1], second[1])
    return overlap >= 0.5 * min(first[3] - first[1], second[3] - second[1])


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


def _annotation_entities(
    document: ExtractedDocument,
    models: Any,
    lang: str | None,
    *,
    max_detections: int = 10_000,
) -> tuple[Any, ...]:
    """Require an explicit detector result and a valid map for every span."""
    try:
        detector = _resolve_detector(models)
        signature = inspect.signature(detector)
        try:
            signature.bind(document.text, lang=lang)
        except TypeError:
            signature.bind(document.text)
            kwargs = {}
        else:
            kwargs = {"lang": lang}
        # Signature selection happens before invocation. An internal TypeError
        # is a failed inspection, never permission to retry a different call.
        result = detector(document.text, **kwargs)
        if isinstance(result, Mapping):
            keys = [
                key for key in ("entities", "pii_entities", "spans") if key in result
            ]
            if len(keys) != 1:
                raise ValueError
            result = result[keys[0]]
        elif hasattr(result, "entities"):
            result = result.entities
        elif hasattr(result, "pii_entities"):
            result = result.pii_entities
        if not isinstance(result, Iterable) or isinstance(
            result, (str, bytes, bytearray, Mapping)
        ):
            raise ValueError
        entities = tuple(islice(result, max_detections + 1))
        if len(entities) > max_detections:
            raise ValueError
        for entity in entities:
            if isinstance(entity, Sequence) and not isinstance(
                entity, (str, bytes, bytearray)
            ):
                if len(entity) < 2:
                    raise ValueError
                start, end = entity[:2]
            elif isinstance(entity, Mapping):
                start, end = entity.get("start"), entity.get("end")
            else:
                start, end = (
                    getattr(entity, "start", None),
                    getattr(entity, "end", None),
                )
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, Integral)
                or not isinstance(end, Integral)
                or not 0 <= start < end <= len(document.text)
                or not project_text_spans(document, (entity,))
            ):
                raise ValueError
        return entities
    except Exception:
        raise ValueError("annotation_detection_failed") from None


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
    path: str | Path | BinaryIO,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    include_annotations = _policy_value(policy, "include_annotations")
    if include_annotations is not None and type(include_annotations) is not bool:
        raise ValueError("include_annotations must be a boolean")
    if include_annotations and _resolve_detector(models) is None:
        raise ValueError("annotation_detector_required")
    content = _snapshot_pdf_source(path)
    document = extract_pdf(
        BytesIO(content), include_annotations=bool(include_annotations)
    )
    # Keep this bbox-preserving extraction as the canonical text/offset map
    # while adding structured boxes for table cells and caption lines.
    from .documents_pdf_tables import extract_pdf_regions, project_structured_spans

    regions = extract_pdf_regions(BytesIO(content), document=document)
    entities = (
        _annotation_entities(document, models, lang)
        if include_annotations
        else _iter_entities(_detect_entities(document, models, lang))
    )
    rectangles = project_structured_spans(document, regions, entities)
    if include_annotations:
        # A table/caption may overlap an annotation geometrically. Its narrower
        # box must never replace the complete annotation extent selected by a
        # detector, including a span crossing page content and annotation text.
        annotation_document = ExtractedDocument(
            text=document.text,
            spans=tuple(
                span
                for span in document.spans
                if span.metadata.get("block_type") == "annotation"
            ),
        )
        annotation_rectangles = project_text_spans(annotation_document, entities)
        rectangles += tuple(
            rectangle
            for rectangle in annotation_rectangles
            if not any(
                prior.page == rectangle.page and prior.bbox == rectangle.bbox
                for prior in rectangles
            )
        )
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
    output_path = _policy_value(
        policy,
        "output_path",
        "redacted_path",
        "destination_path",
    )
    if output_path is not None or bool(_policy_value(policy, "return_bytes")):
        annotation_report: list[Mapping[str, Any]] = []
        redacted_pdf = _render_redacted_pdf(
            BytesIO(content),
            rectangles,
            include_annotations=bool(include_annotations),
            annotation_report=annotation_report,
        )
        if output_path is not None:
            _write_pdf_output(output_path, redacted_pdf)
        metadata.update(
            {
                "detected_span_count": len(entities),
                "pdf_rasterized": True,
                "redacted_pdf_sha256": hashlib.sha256(redacted_pdf).hexdigest(),
                "redacted_pdf_bytes": redacted_pdf,
                "annotations": annotation_report,
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


def _render_redacted_pdf(
    source: Any,
    rectangles: Sequence[ProjectedRectangle],
    *,
    resolution: int = 150,
    include_annotations: bool = False,
    annotation_report: list[Mapping[str, Any]] | None = None,
) -> bytes:
    """Rasterize a PDF and burn opaque boxes into a fresh image-only PDF."""
    pdfplumber = _import_pdfplumber()
    try:
        image_draw = importlib.import_module("PIL.ImageDraw")
    except ImportError as exc:  # pragma: no cover - covered by extra checks.
        raise MissingDependencyError(
            dependency="Pillow", instruction=_PDFPLUMBER_HINT
        ) from exc

    by_page: dict[int, list[ProjectedRectangle]] = {}
    for rectangle in rectangles:
        by_page.setdefault(rectangle.page, []).append(rectangle)

    images: list[Any] = []
    prepared = _prepare_annotation_source(
        source, include_annotations=include_annotations
    )
    if annotation_report is not None:
        annotation_report.extend(prepared.report)
    with pdfplumber.open(BytesIO(prepared.content)) as pdf:
        for page_index, page in enumerate(getattr(pdf, "pages", ())):
            rendered = page.to_image(
                resolution=resolution,
                antialias=True,
            ).original.convert("RGB")
            width = max(float(getattr(page, "width", rendered.width)), 1.0)
            height = max(float(getattr(page, "height", rendered.height)), 1.0)
            x_scale = rendered.width / width
            y_scale = rendered.height / height
            drawer = image_draw.Draw(rendered)
            for rectangle in by_page.get(page_index, ()):
                x0, top, x1, bottom = rectangle.bbox
                drawer.rectangle(
                    (
                        int(x0 * x_scale),
                        int(top * y_scale),
                        int(x1 * x_scale) + 1,
                        int(bottom * y_scale) + 1,
                    ),
                    fill="black",
                )
            images.append(rendered)

    if not images:
        raise UnsupportedDocumentError("Cannot emit a clean PDF with no pages.")
    output = BytesIO()
    first, *remaining = images
    try:
        first.save(
            output,
            format="PDF",
            save_all=bool(remaining),
            append_images=remaining,
            resolution=resolution,
        )
        return output.getvalue()
    finally:
        for image in images:
            image.close()


def _write_pdf_output(output: Any, payload: bytes) -> None:
    if hasattr(output, "write"):
        try:
            output.seek(0)
            output.truncate()
        except (AttributeError, OSError):
            pass
        output.write(payload)
        try:
            output.seek(0)
        except (AttributeError, OSError):
            pass
        return
    path = Path(output)
    if path.suffix.lower() != ".pdf":
        raise ValueError("redacted PDF output must use the .pdf extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _policy_value(policy: Any, *names: str) -> Any:
    if isinstance(policy, Mapping):
        for name in names:
            if name in policy:
                return policy[name]
        return None
    for name in names:
        value = getattr(policy, name, None)
        if value is not None:
            return value
    return None


def _rewind(source: Any) -> None:
    try:
        source.seek(0)
    except (AttributeError, OSError):
        pass


register_handler(".pdf", _pdf_handler)


__all__ = [
    "ProjectedRectangle",
    "extract_pdf",
    "project_text_spans",
]
