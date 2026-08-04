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
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from ..processing.sentences import segment_text
from .pii import DeidentificationMethod, PIIEntity
from .pipeline import Pipeline
from .schemas.span import OpenMedSpan

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
class _DocumentSentence:
    """One source-preserving sentence or other safe text segment."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class DocumentStreamResult:
    """Aggregate result of a memory-bounded streaming document pass.

    Attributes:
        pii_entities: Detected PII entities with **global** document offsets,
            ordered by ``(start, end)``. Identical to the non-streaming
            :func:`openmed.core.pii.deidentify` result on the same input.
        redacted_text: Source document with detected entities replaced according
            to the configured de-identification method.
        window_count: Number of windows processed.
        max_window_chars: Largest window (in characters) handed to the pipeline;
            the practical peak-memory driver.
        document_length: Total document length in characters.
    """

    pii_entities: list[PIIEntity]
    window_count: int
    max_window_chars: int
    document_length: int
    _spans: tuple[OpenMedSpan, ...] = field(default=(), repr=False)
    _source_text: str | None = field(default=None, repr=False)
    _redacted_chunks: tuple[str, ...] = field(default=(), repr=False)

    @property
    def spans(self) -> tuple[OpenMedSpan, ...]:
        """Return canonical :class:`OpenMedSpan` records with global offsets."""

        return self._spans

    @property
    def redacted_text(self) -> str:
        """Return the redacted document without retaining another source copy."""

        if self._source_text is not None:
            leading = len(self._source_text) - len(self._source_text.lstrip())
            stripped = self._source_text.strip()
            return _render_redacted_region(
                stripped,
                region_start=leading,
                entities=self.pii_entities,
            )
        return "".join(self._redacted_chunks).strip()

    @property
    def deidentified_text(self) -> str:
        """Alias :attr:`redacted_text` for parity with de-identification results."""

        return self.redacted_text


def _iter_source_blocks(
    source: str | Iterable[str],
    *,
    block_chars: int,
) -> Iterator[str]:
    """Coalesce arbitrary source fragments into bounded segmentation blocks."""

    fragments: Iterable[str] = (source,) if isinstance(source, str) else source
    pending = ""
    for fragment in fragments:
        if not isinstance(fragment, str):
            raise TypeError("source fragments must be strings")
        fragment_offset = 0
        while fragment_offset < len(fragment):
            take = min(block_chars - len(pending), len(fragment) - fragment_offset)
            pending += fragment[fragment_offset : fragment_offset + take]
            fragment_offset += take
            if len(pending) == block_chars:
                yield pending
                pending = ""
    if pending:
        yield pending


def _iter_sentence_segments(
    source: str | Iterable[str],
    *,
    block_chars: int,
    lang: str,
) -> Iterator[_DocumentSentence]:
    """Yield exact source segments while retaining only one unfinished sentence."""

    pending = ""
    pending_start = 0
    for block in _iter_source_blocks(source, block_chars=block_chars):
        pending += block
        spans = segment_text(pending, language=lang)

        if not spans:
            if pending and not pending.strip():
                yield _DocumentSentence(
                    text=pending,
                    start=pending_start,
                    end=pending_start + len(pending),
                )
                pending_start += len(pending)
                pending = ""
            continue

        # Retain the last sentence because it may be incomplete at the source
        # block boundary. A sentence longer than ``block_chars`` therefore grows
        # the carry until a real sentence boundary appears; it is never split at
        # an arbitrary character or token offset.
        if len(spans) < 2:
            continue

        cutoff = int(spans[-1].start)
        if cutoff <= 0:
            continue
        cursor = 0
        for index in range(len(spans) - 1):
            segment_end = int(spans[index + 1].start)
            segment_end = min(cutoff, max(cursor, segment_end))
            if segment_end <= cursor:
                continue
            yield _DocumentSentence(
                text=pending[cursor:segment_end],
                start=pending_start + cursor,
                end=pending_start + segment_end,
            )
            cursor = segment_end
        if cursor < cutoff:
            yield _DocumentSentence(
                text=pending[cursor:cutoff],
                start=pending_start + cursor,
                end=pending_start + cutoff,
            )
        pending = pending[cutoff:]
        pending_start += cutoff

    if not pending:
        return

    spans = segment_text(pending, language=lang)
    if not spans:
        yield _DocumentSentence(
            text=pending,
            start=pending_start,
            end=pending_start + len(pending),
        )
        return

    cursor = 0
    for index in range(len(spans)):
        segment_end = (
            int(spans[index + 1].start) if index + 1 < len(spans) else len(pending)
        )
        segment_end = max(cursor, min(len(pending), segment_end))
        if segment_end <= cursor:
            continue
        yield _DocumentSentence(
            text=pending[cursor:segment_end],
            start=pending_start + cursor,
            end=pending_start + segment_end,
        )
        cursor = segment_end
    if cursor < len(pending):
        yield _DocumentSentence(
            text=pending[cursor:],
            start=pending_start + cursor,
            end=pending_start + len(pending),
        )


def _iter_sentence_bounds(
    text: str,
    *,
    block_chars: int,
    lang: str,
) -> Iterator[tuple[int, int]]:
    """Yield global sentence bounds using the incremental segment iterator."""

    for sentence in _iter_sentence_segments(
        text,
        block_chars=block_chars,
        lang=lang,
    ):
        yield (sentence.start, sentence.end)


def iter_document_windows(
    source: str | Iterable[str],
    *,
    window_chars: int = 4096,
    overlap_chars: int = 0,
    lang: str = "en",
) -> Iterator[_DocumentWindow]:
    """Yield sentence-aligned windows of ``source`` bounded by ``window_chars``.

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
        source: Source document text or an iterable of source fragments.
        window_chars: Soft upper bound on the newly-owned characters per window.
        overlap_chars: Leading context characters carried from the prior window.
        lang: Language code for sentence segmentation.

    Yields:
        ``_DocumentWindow`` records with global source offsets.
    """

    if window_chars < 1:
        raise ValueError("window_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    # Segment over blocks a couple of windows wide so most sentences resolve
    # inside a single block while the segmenter's working set stays bounded.
    block_chars = max(window_chars * 2, 1024)

    group: list[_DocumentSentence] = []
    group_chars = 0
    overlap: tuple[_DocumentSentence, ...] = ()

    def emit(sentences: Sequence[_DocumentSentence]) -> _DocumentWindow:
        own_start = sentences[0].start
        own_end = sentences[-1].end
        overlap_text = "".join(sentence.text for sentence in overlap)
        own_text = "".join(sentence.text for sentence in sentences)
        window_start = overlap[0].start if overlap else own_start
        return _DocumentWindow(
            text=f"{overlap_text}{own_text}",
            start=window_start,
            end=own_end,
            overlap_start=own_start,
        )

    for sentence in _iter_sentence_segments(
        source,
        block_chars=block_chars,
        lang=lang,
    ):
        sentence_len = sentence.end - sentence.start
        if group and group_chars + sentence_len > window_chars:
            yield emit(group)
            overlap = _select_overlap_sentences(group, overlap_chars)
            group = []
            group_chars = 0
        group.append(sentence)
        group_chars += sentence_len

    if group:
        yield emit(group)


def _select_overlap_sentences(
    sentences: Sequence[_DocumentSentence],
    overlap_chars: int,
) -> tuple[_DocumentSentence, ...]:
    """Return whole trailing sentences for safe cross-window context."""

    if overlap_chars == 0 or not sentences:
        return ()

    selected: list[_DocumentSentence] = []
    selected_chars = 0
    for sentence in reversed(sentences):
        sentence_chars = sentence.end - sentence.start
        if selected and selected_chars + sentence_chars > overlap_chars:
            break
        selected.append(sentence)
        selected_chars += sentence_chars
        if selected_chars >= overlap_chars:
            break
    selected.reverse()
    return tuple(selected)


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

    def run(self, source: str | Iterable[str]) -> DocumentStreamResult:
        """Process ``source`` and return global-offset entities and spans.

        Args:
            source: Full document text or iterable of source fragments. Only one
                window at a time is handed to the pipeline, so processing memory
                is bounded by ``window_chars`` and the longest single sentence.

        Returns:
            A :class:`DocumentStreamResult` whose ``pii_entities`` carry global
            document offsets, de-duplicated across window boundaries.
        """

        entities: dict[tuple[int, int, str], PIIEntity] = {}
        spans: dict[tuple[int, int, str], OpenMedSpan] = {}
        window_count = 0
        max_window_chars = 0
        document_length = 0
        redacted_chunks: list[str] | None = [] if not isinstance(source, str) else None
        pending_region: tuple[str, int] | None = None

        for window in iter_document_windows(
            source,
            window_chars=self.window_chars,
            overlap_chars=self.overlap_chars,
            lang=self.lang,
        ):
            window_count += 1
            max_window_chars = max(max_window_chars, len(window.text))
            document_length = window.end
            self._process_window(window, entities, spans)

            if redacted_chunks is not None:
                own_offset = window.overlap_start - window.start
                own_text = window.text[own_offset:]
                if pending_region is not None:
                    region_text, region_start = pending_region
                    crossing_ends = [
                        int(entity.end)
                        for entity in entities.values()
                        if int(entity.start) < window.overlap_start < int(entity.end)
                    ]
                    if crossing_ends:
                        combined = f"{region_text}{own_text}"
                        finalized_end = max(crossing_ends)
                        finalized_chars = finalized_end - region_start
                        redacted_chunks.append(
                            _render_redacted_region(
                                combined[:finalized_chars],
                                region_start=region_start,
                                entities=entities.values(),
                            )
                        )
                        pending_region = (
                            combined[finalized_chars:],
                            finalized_end,
                        )
                    else:
                        redacted_chunks.append(
                            _render_redacted_region(
                                region_text,
                                region_start=region_start,
                                entities=entities.values(),
                            )
                        )
                        pending_region = (own_text, window.overlap_start)
                else:
                    pending_region = (own_text, window.overlap_start)

        if redacted_chunks is not None and pending_region is not None:
            region_text, region_start = pending_region
            redacted_chunks.append(
                _render_redacted_region(
                    region_text,
                    region_start=region_start,
                    entities=entities.values(),
                )
            )

        ordered_keys = sorted(entities, key=lambda key: (key[0], key[1], key[2]))
        ordered_entities = [entities[key] for key in ordered_keys]
        ordered_spans = tuple(spans[key] for key in ordered_keys if key in spans)

        return DocumentStreamResult(
            pii_entities=ordered_entities,
            window_count=window_count,
            max_window_chars=max_window_chars,
            document_length=(
                len(source) if isinstance(source, str) else document_length
            ),
            _spans=ordered_spans,
            _source_text=source if isinstance(source, str) else None,
            _redacted_chunks=tuple(redacted_chunks or ()),
        )

    def _process_window(
        self,
        window: _DocumentWindow,
        entities: dict[tuple[int, int, str], PIIEntity],
        spans: dict[tuple[int, int, str], OpenMedSpan],
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
        span_by_local: dict[tuple[int, int], OpenMedSpan] = {
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

            # An overlap window may reveal that a provisional span emitted by
            # the prior window was only part of an identifier. Prefer the span
            # that crosses the safe boundary and remove overlapping fragments.
            if global_start < window.overlap_start < global_end:
                overlapping_keys = [
                    existing_key
                    for existing_key in entities
                    if existing_key[0] < global_end and global_start < existing_key[1]
                ]
                for existing_key in overlapping_keys:
                    entities.pop(existing_key, None)
                    spans.pop(existing_key, None)

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
    fragments. Fragments are consumed incrementally and re-segmented on sentence
    boundaries, so fragment boundaries never affect the result and no complete
    raw-document copy is created for iterable input.

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

    streamer = DocumentStreamDeidentifier(
        window_chars=window_chars,
        overlap_chars=overlap_chars,
        lang=lang,
        **kwargs,
    )
    return streamer.run(source)


def _render_redacted_region(
    text: str,
    *,
    region_start: int,
    entities: Iterable[PIIEntity],
) -> str:
    """Render global-offset entities into one contiguous source region."""

    applicable = sorted(
        (
            entity
            for entity in entities
            if int(entity.start) >= region_start
            and int(entity.end) <= region_start + len(text)
        ),
        key=lambda entity: (int(entity.start), int(entity.end)),
    )
    if not applicable:
        return text

    rendered: list[str] = []
    cursor = 0
    for entity in applicable:
        local_start = int(entity.start) - region_start
        local_end = int(entity.end) - region_start
        if local_start < cursor:
            continue
        rendered.append(text[cursor:local_start])
        rendered.append(entity.redacted_text or "")
        cursor = local_end
    rendered.append(text[cursor:])
    return "".join(rendered)


def _shift_entity_global(entity: PIIEntity, offset: int) -> PIIEntity:
    shifted = copy.copy(entity)
    shifted.start = int(entity.start) + offset
    shifted.end = int(entity.end) + offset
    return shifted


def _shift_span_global(span: OpenMedSpan, offset: int) -> OpenMedSpan:
    return replace(
        span,
        start=int(span.start) + offset,
        end=int(span.end) + offset,
    )
