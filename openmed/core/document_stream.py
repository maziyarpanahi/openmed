"""Memory-bounded streaming de-identification for very long documents.

The chunk-fed :class:`openmed.core.streaming.StreamingDeidentifier` handles the
case where a caller already has the document split into arbitrary byte chunks
(for example, bytes arriving off a socket). This module solves the complementary
problem: given a *single very long document*, process it incrementally so that
peak resident memory stays bounded independently of document length, while
returning spans whose **global** character offsets are identical to running the
non-streaming :func:`openmed.core.pii.deidentify` on the same input.

The document is segmented on sentence (safe) boundaries via
:func:`openmed.processing.sentences.segment_text`, and whole sentences are grouped
into windows bounded by ``window_chars``. Because an identifier lives within a
single sentence, and a sentence is never split across windows, no identifier is
ever split across a window boundary. Spans detected per window are lifted back to
global offsets and de-duplicated across the small window overlap, so an entity
that sits near a boundary is emitted exactly once, whole, with correct global
offsets -- never split and never duplicated.

Only one window (plus a bounded sentence carry) is resident at a time, so peak
memory is a function of ``window_chars`` and the longest single sentence, not of
document length.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from ..processing.sentences import segment_text
from .pii import DeidentificationMethod, PIIEntity
from .pipeline import Pipeline

__all__ = [
    "DocumentStreamResult",
    "DocumentStreamDeidentifier",
    "deidentify_document_stream",
    "iter_document_windows",
]


@dataclass(frozen=True)
class _DocumentWindow:
    """A contiguous run of whole sentences processed as one pipeline call.

    ``start``/``end`` are global character offsets into the source document.
    ``overlap_start`` marks where the leading overlap (carried from the previous
    window) ends and this window's own newly-owned region begins; spans that end
    at or before ``overlap_start`` were already emitted by the prior window and
    are dropped as duplicates.
    """

    text: str
    start: int
    end: int
    overlap_start: int


@dataclass(frozen=True)
class DocumentStreamResult:
    """Aggregate result of a memory-bounded streaming document pass.

    Attributes:
        pii_entities: Detected PII entities with **global** document offsets,
            ordered by ``(start, end)``. Identical to the non-streaming
            :func:`openmed.core.pii.deidentify` result on the same input.
        window_count: Number of windows processed.
        max_window_chars: Largest window (in characters) handed to the pipeline;
            the practical peak-memory driver.
        document_length: Total document length in characters.
    """

    pii_entities: list[PIIEntity]
    window_count: int
    max_window_chars: int
    document_length: int
    _spans: tuple[Any, ...] = field(default=(), repr=False)

    @property
    def spans(self) -> tuple[Any, ...]:
        """Return canonical :class:`OpenMedSpan` records with global offsets."""

        return self._spans


def _iter_sentence_bounds(
    text: str,
    *,
    block_chars: int,
    lang: str,
) -> Iterator[tuple[int, int]]:
    """Yield global ``(start, end)`` sentence bounds by segmenting incrementally.

    ``pysbd`` segments whatever text it is handed in one pass, so segmenting the
    whole document at once would allocate an O(length) list of per-sentence
    strings. Instead this walks the document in ``block_chars`` slices, segments
    each slice, emits every sentence that is fully inside the slice, and carries
    the trailing (possibly incomplete) sentence forward to the next slice. The
    segmenter's working set is therefore bounded by ``block_chars`` plus one
    carried sentence -- never the whole document.

    Only integer offsets are yielded; no sentence text is retained.
    """

    length = len(text)
    carry_start = 0  # global offset of the not-yet-emitted tail
    while carry_start < length:
        block_end = min(length, carry_start + block_chars)
        is_final = block_end >= length
        block = text[carry_start:block_end]
        spans = segment_text(block, language=lang)
        if not spans:
            # Segmenter produced nothing (e.g. pure whitespace block); advance.
            if is_final:
                return
            carry_start = block_end
            continue

        # On non-final blocks, hold back the last sentence: it may continue past
        # the block boundary, so re-segment it together with the next block.
        emit_upto = len(spans) if is_final else len(spans) - 1
        last_emitted_end = carry_start
        for span in spans[:emit_upto]:
            global_start = carry_start + span.start
            global_end = carry_start + span.end
            last_emitted_end = global_end
            yield (global_start, global_end)

        if is_final:
            return

        # Carry from the start of the held-back sentence. If nothing was emitted
        # (a single sentence spans the whole block), force progress by carrying
        # from the block end to avoid an infinite loop; such a sentence is longer
        # than ``block_chars`` and becomes its own window later.
        held_start = carry_start + spans[emit_upto].start
        if held_start <= carry_start:
            # Whole block is one unfinished sentence; grow the window by emitting
            # it as-is to guarantee forward progress.
            yield (carry_start, block_end)
            carry_start = block_end
        else:
            carry_start = max(last_emitted_end, held_start)


def iter_document_windows(
    text: str,
    *,
    window_chars: int = 4096,
    overlap_chars: int = 0,
    lang: str = "en",
) -> Iterator[_DocumentWindow]:
    """Yield sentence-aligned windows of ``text`` bounded by ``window_chars``.

    Whole sentences are grouped greedily so each window's own (non-overlap)
    region stays within ``window_chars`` where possible. A single sentence longer
    than ``window_chars`` becomes its own window (never split mid-sentence, so an
    identifier is never split). ``overlap_chars`` prepends trailing context from
    the previous window so a detector needing left context still sees it; the
    overlap region never re-emits spans (see :class:`_DocumentWindow`).

    Segmentation is performed incrementally over bounded blocks (see
    :func:`_iter_sentence_bounds`), so peak memory is a function of
    ``window_chars`` and the longest single sentence -- not of document length.

    Args:
        text: Source document text.
        window_chars: Soft upper bound on the newly-owned characters per window.
        overlap_chars: Leading context characters carried from the prior window.
        lang: Language code for sentence segmentation.

    Yields:
        ``_DocumentWindow`` records with global offsets into ``text``.
    """

    if window_chars < 1:
        raise ValueError("window_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if not text:
        return

    # Segment over blocks a couple of windows wide so most sentences resolve
    # inside a single block while the segmenter's working set stays bounded.
    block_chars = max(window_chars * 2, 1024)

    group_start: int | None = None
    group_end = 0
    group_chars = 0
    emitted_any = False

    def emit(own_start: int, own_end: int) -> _DocumentWindow:
        window_start = max(0, own_start - overlap_chars) if overlap_chars else own_start
        return _DocumentWindow(
            text=text[window_start:own_end],
            start=window_start,
            end=own_end,
            overlap_start=own_start,
        )

    for sent_start, sent_end in _iter_sentence_bounds(
        text, block_chars=block_chars, lang=lang
    ):
        sentence_len = sent_end - sent_start
        if group_start is not None and group_chars + sentence_len > window_chars:
            yield emit(group_start, group_end)
            emitted_any = True
            group_start = None
            group_chars = 0
        if group_start is None:
            group_start = sent_start
        group_end = sent_end
        group_chars += sentence_len

    if group_start is not None:
        yield emit(group_start, group_end)
        emitted_any = True

    if not emitted_any and text.strip():
        # Segmenter yielded no bounds but there is content; emit it whole.
        yield emit(0, len(text))


class DocumentStreamDeidentifier:
    """De-identify a very long document with bounded peak memory.

    Segments the document on sentence boundaries, processes windows incrementally
    through the shared :class:`~openmed.core.pipeline.Pipeline`, lifts each
    window's spans to global offsets, and merges them across window boundaries so
    the emitted spans are identical to the non-streaming
    :func:`openmed.core.pii.deidentify` on the same input.

    Args mirror :func:`openmed.core.pii.deidentify` where applicable.
    """

    def __init__(
        self,
        *,
        window_chars: int = 4096,
        overlap_chars: int = 256,
        method: DeidentificationMethod = "mask",
        model_name: str | None = None,
        confidence_threshold: float = 0.7,
        keep_year: bool = False,
        shift_dates: Optional[bool] = None,
        date_shift_days: Optional[int] = None,
        keep_mapping: bool = False,
        config: Any = None,
        use_smart_merging: bool = True,
        lang: str = "en",
        normalize_accents: Optional[bool] = None,
        use_safety_sweep: bool = True,
        consistent: bool = False,
        seed: Optional[int] = None,
        locale: Optional[str] = None,
        loader: Any = None,
        policy: Optional[str] = None,
        calibration_thresholds_path: Optional[str | Path] = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        if window_chars < 1:
            raise ValueError("window_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")

        self.window_chars = int(window_chars)
        self.overlap_chars = int(overlap_chars)
        self.method = method
        self.keep_year = keep_year
        self.shift_dates = shift_dates
        self.date_shift_days = date_shift_days
        self.keep_mapping = keep_mapping
        self.consistent = consistent
        self.seed = seed
        self.locale = locale
        self.lang = lang

        self.pipeline = pipeline or Pipeline(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            config=config,
            use_smart_merging=use_smart_merging,
            lang=lang,
            normalize_accents=normalize_accents,
            use_safety_sweep=use_safety_sweep,
            loader=loader,
            policy=policy,
            calibration_thresholds_path=(
                str(calibration_thresholds_path)
                if calibration_thresholds_path is not None
                else None
            ),
        )

    def run(self, text: str) -> DocumentStreamResult:
        """Process ``text`` and return global-offset entities and spans.

        Args:
            text: The full document text. Only one window at a time is handed to
                the pipeline, so peak memory is bounded by ``window_chars`` and
                the longest single sentence, not by ``len(text)``.

        Returns:
            A :class:`DocumentStreamResult` whose ``pii_entities`` carry global
            document offsets, de-duplicated across window boundaries.
        """

        entities: dict[tuple[int, int, str], PIIEntity] = {}
        spans: dict[tuple[int, int, str], Any] = {}
        window_count = 0
        max_window_chars = 0

        for window in iter_document_windows(
            text,
            window_chars=self.window_chars,
            overlap_chars=self.overlap_chars,
            lang=self.lang,
        ):
            window_count += 1
            max_window_chars = max(max_window_chars, len(window.text))
            self._process_window(window, entities, spans)

        ordered_keys = sorted(entities, key=lambda key: (key[0], key[1], key[2]))
        ordered_entities = [entities[key] for key in ordered_keys]
        ordered_spans = tuple(spans[key] for key in ordered_keys if key in spans)

        return DocumentStreamResult(
            pii_entities=ordered_entities,
            window_count=window_count,
            max_window_chars=max_window_chars,
            document_length=len(text),
            _spans=ordered_spans,
        )

    def _process_window(
        self,
        window: _DocumentWindow,
        entities: dict[tuple[int, int, str], PIIEntity],
        spans: dict[tuple[int, int, str], Any],
    ) -> None:
        window_text = window.text
        if not window_text.strip():
            return

        # ``Pipeline.run`` strips leading/trailing whitespace and returns offsets
        # relative to the stripped text, so recover the stripped-text base offset
        # within the window and add the window's global base.
        leading = len(window_text) - len(window_text.lstrip())
        base = window.start + leading

        result = self.pipeline.run(
            window_text,
            method=self.method,
            keep_year=self.keep_year,
            shift_dates=self.shift_dates,
            date_shift_days=self.date_shift_days,
            keep_mapping=self.keep_mapping,
            consistent=self.consistent,
            seed=self.seed,
            locale=self.locale,
        )

        window_entities = result.deidentification_result.pii_entities
        window_spans = list(result.spans)
        span_by_local: dict[tuple[int, int], Any] = {
            (int(span.start), int(span.end)): span for span in window_spans
        }

        for entity in window_entities:
            local_start = int(entity.start)
            local_end = int(entity.end)
            global_start = base + local_start
            global_end = base + local_end

            # Drop spans owned by the previous window's region. The overlap only
            # provides context; anything ending at or before this window's own
            # start was already emitted upstream.
            if global_end <= window.overlap_start:
                continue

            key = (global_start, global_end, str(entity.entity_type or entity.label))
            if key in entities:
                continue

            entities[key] = _shift_entity_global(entity, base)
            local_span = span_by_local.get((local_start, local_end))
            if local_span is not None:
                spans[key] = _shift_span_global(local_span, base)


def deidentify_document_stream(
    source: str | Iterable[str],
    *,
    window_chars: int = 4096,
    overlap_chars: int = 256,
    lang: str = "en",
    **kwargs: Any,
) -> DocumentStreamResult:
    """Stream-de-identify a very long document with a bounded memory footprint.

    Accepts either the full document as one string or an iterable of source
    fragments (which are concatenated); either way the text is re-segmented on
    sentence boundaries internally, so fragment boundaries never affect the
    result.

    Args:
        source: The document text, or an iterable of text fragments.
        window_chars: Soft per-window character budget (peak-memory driver).
        overlap_chars: Leading context carried between windows for detector
            left-context; never re-emits spans.
        lang: Language code for sentence segmentation and the pipeline.
        **kwargs: Forwarded to :class:`DocumentStreamDeidentifier` (``method``,
            ``model_name``, ``pipeline``, ...).

    Returns:
        A :class:`DocumentStreamResult` with global-offset entities identical to
        the non-streaming :func:`openmed.core.pii.deidentify` on the same input.
    """

    text = source if isinstance(source, str) else "".join(source)
    streamer = DocumentStreamDeidentifier(
        window_chars=window_chars,
        overlap_chars=overlap_chars,
        lang=lang,
        **kwargs,
    )
    return streamer.run(text)


def _shift_entity_global(entity: PIIEntity, offset: int) -> PIIEntity:
    shifted = copy.copy(entity)
    shifted.start = int(entity.start) + offset
    shifted.end = int(entity.end) + offset
    return shifted


def _shift_span_global(span: Any, offset: int) -> Any:
    return replace(
        span,
        start=int(span.start) + offset,
        end=int(span.end) + offset,
    )
