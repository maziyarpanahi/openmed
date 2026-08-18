"""PPTX slide, table, and speaker-notes extraction with write-back.

The ingester imports ``python-pptx`` lazily so importing
``openmed.multimodal`` stays lightweight. Extraction emits one normalized text
stream with per-slide offsets and one source span per non-empty PowerPoint run.
Detected spans can then be projected back to those runs and optionally written
to a copy of the presentation without storing plaintext PHI in provenance.
"""

from __future__ import annotations

import hashlib
import importlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan, register_handler
from .exceptions import MissingDependencyError

_PPTX_HINT = 'Install with: pip install "openmed[multimodal]".'
_BLOCK_SEPARATOR = "\n"


@dataclass(frozen=True)
class PptxRunRange:
    """A local PowerPoint run range covered by a detected text span."""

    document_run_index: int
    run_start: int
    run_end: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-safe metadata representation."""
        return {
            "document_run_index": self.document_run_index,
            "run_start": self.run_start,
            "run_end": self.run_end,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PptxRedaction:
    """A detected PHI span projected to one or more PowerPoint runs."""

    start: int
    end: int
    replacement: str
    label: str | None = None
    confidence: float | None = None
    run_ranges: tuple[PptxRunRange, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-safe metadata representation."""
        payload: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "replacement": self.replacement,
            "run_ranges": [run_range.to_dict() for run_range in self.run_ranges],
        }
        if self.label is not None:
            payload["label"] = self.label
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class _TextFrameContext:
    text_frame: Any
    slide_index: int
    part: str
    block_type: str
    shape_index: int | None = None
    shape_path: str | None = None
    row_index: int | None = None
    cell_index: int | None = None


@dataclass
class _ExtractionState:
    parts: list[str] = field(default_factory=list)
    spans: list[SourceSpan] = field(default_factory=list)
    cursor: int = 0
    block_index: int = 0
    document_run_index: int = 0
    text_run_count: int = 0
    paragraph_count: int = 0
    slide_cursor: int = 0
    slide_has_text: bool = False


@dataclass(frozen=True)
class _RunEdit:
    start: int
    end: int
    replacement: str


def _import_pptx() -> Any:
    try:
        return importlib.import_module("pptx")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="python-pptx", instruction=_PPTX_HINT
        ) from exc


def extract_pptx(path: str | Path) -> ExtractedDocument:
    """Extract slide, table, and speaker-notes text with offset provenance.

    Paragraphs are separated with newlines. Each non-empty PowerPoint run
    becomes a :class:`SourceSpan`; ``page`` is the zero-based slide index and
    span metadata records both document-wide and per-slide character offsets.

    Args:
        path: Source PowerPoint presentation.

    Returns:
        Normalized text, source spans, and per-slide offset metadata.
    """
    pptx = _import_pptx()
    presentation = pptx.Presentation(path)
    state = _ExtractionState()
    slide_offsets: list[dict[str, int]] = []

    for slide_index, slide in enumerate(presentation.slides):
        state.slide_cursor = 0
        state.slide_has_text = False
        first_span_index = len(state.spans)

        for context in _iter_slide_text_frames(slide, slide_index):
            _append_text_frame(context, state)

        slide_spans = state.spans[first_span_index:]
        if slide_spans:
            start = slide_spans[0].start
            end = slide_spans[-1].end
        else:
            start = state.cursor
            end = state.cursor
        slide_offsets.append(
            {
                "slide_index": slide_index,
                "start": start,
                "end": end,
                "length": state.slide_cursor,
            }
        )

    return ExtractedDocument(
        text="".join(state.parts),
        spans=tuple(state.spans),
        metadata={
            "format": "pptx",
            "slide_count": len(presentation.slides),
            "slide_offsets": slide_offsets,
            "paragraph_count": state.paragraph_count,
            "text_run_count": state.text_run_count,
            "document_run_count": state.document_run_index,
        },
    )


def map_text_spans_to_pptx_runs(
    document: ExtractedDocument,
    spans: Iterable[Any] | Mapping[str, Any] | Any,
    *,
    replacement: str | None = None,
) -> tuple[PptxRedaction, ...]:
    """Project detected normalized-text spans to PowerPoint run ranges.

    Args:
        document: Extracted PPTX document returned by :func:`extract_pptx`.
        spans: Detector spans containing ``start`` and ``end`` offsets.
        replacement: Optional replacement used when a span has none.

    Returns:
        PHI-safe redaction records with global and per-slide provenance.
    """
    redactions: list[PptxRedaction] = []
    for raw_entity in _iter_entity_inputs(spans):
        entity = _coerce_entity(raw_entity, default_replacement=replacement)
        if entity is None:
            continue
        start, end, label, confidence, entity_replacement = entity
        start = max(0, start)
        end = min(len(document.text), end)
        if end <= start:
            continue

        covered = tuple(
            source
            for source in document.spans
            if source.metadata.get("format") == "pptx"
            and source.end > start
            and source.start < end
        )
        if not covered:
            continue

        run_ranges = tuple(
            _source_span_to_run_range(source, start=start, end=end)
            for source in covered
        )
        redactions.append(
            PptxRedaction(
                start=start,
                end=end,
                label=label,
                confidence=confidence,
                replacement=entity_replacement,
                run_ranges=run_ranges,
                metadata={
                    "text_sha256": hashlib.sha256(
                        document.text[start:end].encode("utf-8")
                    ).hexdigest(),
                    "source_span_count": len(run_ranges),
                    "slide_indices": sorted({source.page for source in covered}),
                },
            )
        )
    return tuple(redactions)


def write_redacted_pptx(
    source_path: str | Path,
    output_path: str | Path,
    spans: Iterable[Any] | Mapping[str, Any] | Any,
    *,
    replacement: str | None = None,
) -> Path:
    """Write a redacted copy of a PPTX using normalized-text spans.

    Args:
        source_path: Source PowerPoint presentation.
        output_path: Destination for the redacted copy.
        spans: Detector spans using offsets from :func:`extract_pptx`.
        replacement: Optional replacement used when a span has none.

    Returns:
        The output path.
    """
    document = extract_pptx(source_path)
    redactions = map_text_spans_to_pptx_runs(document, spans, replacement=replacement)
    _write_pptx_redactions(source_path, output_path, redactions)
    return Path(output_path)


def _append_text_frame(context: _TextFrameContext, state: _ExtractionState) -> None:
    for paragraph_index, paragraph in enumerate(context.text_frame.paragraphs):
        paragraph_had_text = False
        block_index = state.block_index

        for run_index, run in enumerate(paragraph.runs):
            document_run_index = state.document_run_index
            state.document_run_index += 1
            text = str(run.text)
            if not text:
                continue

            if not paragraph_had_text:
                if state.parts:
                    state.parts.append(_BLOCK_SEPARATOR)
                    state.cursor += len(_BLOCK_SEPARATOR)
                if state.slide_has_text:
                    state.slide_cursor += len(_BLOCK_SEPARATOR)
                paragraph_had_text = True
                state.slide_has_text = True

            start = state.cursor
            slide_start = state.slide_cursor
            state.parts.append(text)
            state.cursor += len(text)
            state.slide_cursor += len(text)
            state.spans.append(
                SourceSpan(
                    start=start,
                    end=state.cursor,
                    page=context.slide_index,
                    metadata=_source_metadata(
                        context,
                        block_index=block_index,
                        paragraph_index=paragraph_index,
                        run_index=run_index,
                        document_run_index=document_run_index,
                        slide_start=slide_start,
                        slide_end=state.slide_cursor,
                    ),
                )
            )
            state.text_run_count += 1

        if paragraph_had_text:
            state.block_index += 1
            state.paragraph_count += 1


def _source_metadata(
    context: _TextFrameContext,
    *,
    block_index: int,
    paragraph_index: int,
    run_index: int,
    document_run_index: int,
    slide_start: int,
    slide_end: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format": "pptx",
        "part": context.part,
        "block_type": context.block_type,
        "slide_index": context.slide_index,
        "slide_start": slide_start,
        "slide_end": slide_end,
        "block_index": block_index,
        "paragraph_index": paragraph_index,
        "run_index": run_index,
        "document_run_index": document_run_index,
    }
    optional_values = {
        "shape_index": context.shape_index,
        "shape_path": context.shape_path,
        "row_index": context.row_index,
        "cell_index": context.cell_index,
    }
    metadata.update(
        {key: value for key, value in optional_values.items() if value is not None}
    )
    return metadata


def _iter_slide_text_frames(
    slide: Any, slide_index: int
) -> Iterable[_TextFrameContext]:
    for shape_path, shape in _iter_shapes(slide.shapes):
        shape_index = shape_path[0]
        path_text = ".".join(str(index) for index in shape_path)
        if bool(getattr(shape, "has_table", False)):
            for row_index, row in enumerate(shape.table.rows):
                for cell_index, cell in enumerate(row.cells):
                    yield _TextFrameContext(
                        text_frame=cell.text_frame,
                        slide_index=slide_index,
                        part="slide",
                        block_type="table_cell",
                        shape_index=shape_index,
                        shape_path=path_text,
                        row_index=row_index,
                        cell_index=cell_index,
                    )
        elif bool(getattr(shape, "has_text_frame", False)):
            yield _TextFrameContext(
                text_frame=shape.text_frame,
                slide_index=slide_index,
                part="slide",
                block_type="shape",
                shape_index=shape_index,
                shape_path=path_text,
            )

    if bool(getattr(slide, "has_notes_slide", False)):
        notes_text_frame = slide.notes_slide.notes_text_frame
        if notes_text_frame is not None:
            yield _TextFrameContext(
                text_frame=notes_text_frame,
                slide_index=slide_index,
                part="notes",
                block_type="speaker_notes",
            )


def _iter_shapes(
    shapes: Iterable[Any], path: tuple[int, ...] = ()
) -> Iterable[tuple[tuple[int, ...], Any]]:
    for shape_index, shape in enumerate(shapes):
        shape_path = (*path, shape_index)
        nested_shapes = getattr(shape, "shapes", None)
        if nested_shapes is not None:
            yield from _iter_shapes(nested_shapes, shape_path)
        else:
            yield shape_path, shape


def _source_span_to_run_range(
    source: SourceSpan,
    *,
    start: int,
    end: int,
) -> PptxRunRange:
    metadata = dict(source.metadata)
    run_start = max(start, source.start) - source.start
    run_end = min(end, source.end) - source.start
    slide_run_start = int(metadata["slide_start"])
    return PptxRunRange(
        document_run_index=int(metadata["document_run_index"]),
        run_start=run_start,
        run_end=run_end,
        metadata={
            key: value
            for key, value in metadata.items()
            if key
            in {
                "part",
                "block_type",
                "slide_index",
                "block_index",
                "paragraph_index",
                "run_index",
                "shape_index",
                "shape_path",
                "row_index",
                "cell_index",
            }
        }
        | {
            "slide_start": slide_run_start + run_start,
            "slide_end": slide_run_start + run_end,
        },
    )


def _write_pptx_redactions(
    source_path: str | Path,
    output_path: str | Path,
    redactions: Sequence[PptxRedaction],
) -> None:
    pptx = _import_pptx()
    presentation = pptx.Presentation(source_path)
    runs = _collect_runs(presentation)
    edits_by_run: dict[int, list[_RunEdit]] = defaultdict(list)

    for redaction in _non_overlapping_redactions(redactions):
        replacement_inserted = False
        for run_range in redaction.run_ranges:
            replacement = "" if replacement_inserted else redaction.replacement
            replacement_inserted = True
            edits_by_run[run_range.document_run_index].append(
                _RunEdit(
                    start=run_range.run_start,
                    end=run_range.run_end,
                    replacement=replacement,
                )
            )

    for document_run_index, edits in edits_by_run.items():
        run = runs.get(document_run_index)
        if run is None:
            continue
        text = str(run.text)
        for edit in sorted(edits, key=lambda item: item.start, reverse=True):
            text = text[: edit.start] + edit.replacement + text[edit.end :]
        run.text = text

    presentation.save(output_path)


def _collect_runs(presentation: Any) -> dict[int, Any]:
    runs: dict[int, Any] = {}
    document_run_index = 0
    for slide_index, slide in enumerate(presentation.slides):
        for context in _iter_slide_text_frames(slide, slide_index):
            for paragraph in context.text_frame.paragraphs:
                for run in paragraph.runs:
                    runs[document_run_index] = run
                    document_run_index += 1
    return runs


def _non_overlapping_redactions(
    redactions: Sequence[PptxRedaction],
) -> tuple[PptxRedaction, ...]:
    selected: list[PptxRedaction] = []
    cursor = -1
    for redaction in sorted(
        redactions, key=lambda item: (item.start, -(item.end - item.start), item.end)
    ):
        if redaction.start < cursor:
            continue
        selected.append(redaction)
        cursor = redaction.end
    return tuple(selected)


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


def _iter_entity_inputs(spans: Any) -> tuple[Any, ...]:
    if spans is None:
        return ()
    entities = getattr(spans, "entities", None)
    if entities is not None:
        return tuple(entities)
    pii_entities = getattr(spans, "pii_entities", None)
    if pii_entities is not None:
        return tuple(pii_entities)
    if isinstance(spans, Mapping):
        for key in ("entities", "pii_entities", "spans"):
            entities = spans.get(key)
            if entities is not None:
                return tuple(entities)
        if "start" in spans and "end" in spans:
            return (spans,)
    if _looks_like_sequence_entity(spans):
        return (spans,)
    if isinstance(spans, Iterable) and not isinstance(spans, (str, bytes, bytearray)):
        return tuple(spans)
    return ()


def _coerce_entity(
    span: Any,
    *,
    default_replacement: str | None,
) -> tuple[int, int, str | None, float | None, str] | None:
    if _looks_like_sequence_entity(span):
        label = _coerce_optional_str(span[2]) if len(span) >= 3 else None
        return (
            int(span[0]),
            int(span[1]),
            label,
            None,
            default_replacement or _mask_for_label(label),
        )

    if isinstance(span, Mapping):
        start = span.get("start")
        end = span.get("end")
        label = span.get("label", span.get("entity_type", span.get("entity_group")))
        confidence = span.get("confidence", span.get("score"))
        replacement = span.get("replacement", span.get("redacted_text"))
    else:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        label = getattr(
            span,
            "label",
            getattr(span, "entity_type", getattr(span, "entity_group", None)),
        )
        confidence = getattr(span, "confidence", getattr(span, "score", None))
        replacement = getattr(span, "replacement", getattr(span, "redacted_text", None))

    if start is None or end is None:
        return None
    label_text = _coerce_optional_str(label)
    replacement_text = _coerce_optional_str(replacement)
    return (
        int(start),
        int(end),
        label_text,
        _coerce_confidence(confidence),
        replacement_text or default_replacement or _mask_for_label(label_text),
    )


def _looks_like_sequence_entity(value: Any) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 2:
            try:
                int(value[0])
                int(value[1])
            except (TypeError, ValueError):
                return False
            return True
    return False


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


def _mask_for_label(label: str | None) -> str:
    safe_label = "".join(
        character if character.isalnum() else "_"
        for character in (label or "PHI").upper()
    ).strip("_")
    return f"[{safe_label or 'PHI'}]"


def _policy_value(policy: Any, *names: str) -> Any:
    if policy is None:
        return None
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


def _pptx_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    document = extract_pptx(path)
    entities = _iter_entity_inputs(_detect_entities(document, models, lang))
    replacement = _coerce_optional_str(_policy_value(policy, "replacement"))
    redactions = map_text_spans_to_pptx_runs(
        document, entities, replacement=replacement
    )
    if not redactions:
        return document

    output_path = _policy_value(
        policy, "output_path", "redacted_path", "destination_path"
    )
    if output_path is not None:
        _write_pptx_redactions(path, output_path, redactions)

    metadata = dict(document.metadata)
    metadata.update(
        {
            "detected_span_count": len(entities),
            "pptx_redactions": [redaction.to_dict() for redaction in redactions],
        }
    )
    if output_path is not None:
        metadata["redacted_pptx_path"] = str(output_path)
    if policy is not None:
        metadata["policy"] = policy

    return ExtractedDocument(
        text=document.text,
        spans=document.spans,
        metadata=metadata,
    )


register_handler(".pptx", _pptx_handler)


__all__ = [
    "PptxRedaction",
    "PptxRunRange",
    "extract_pptx",
    "map_text_spans_to_pptx_runs",
    "write_redacted_pptx",
]
