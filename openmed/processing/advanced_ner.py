"""Advanced NER processing with proven filtering techniques from OpenMed Gradio app."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List

from openmed.core.decoding import (
    TokenClassificationSpan,
    TokenClassificationStreamEvent,
    coerce_token_classification_spans,
    reconcile_stream_spans,
)

logger = logging.getLogger(__name__)


@dataclass
class EntitySpan:
    """Represents a single entity span with position information."""

    text: str
    label: str
    start: int
    end: int
    score: float

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "EntitySpan":
        """Create an entity span from a mapping-like model output."""

        return cls(
            text=str(data.get("text", data.get("word", ""))),
            label=str(data.get("label", data.get("entity", "")))
            .replace("B-", "")
            .replace("I-", ""),
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            score=float(data.get("score", 1.0)),
        )

    def offset_key(self) -> tuple[int, int]:
        """Return the source character-offset identity for this span."""

        return self.start, self.end

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


TokenClassifier = Callable[[str], Sequence[object] | Any]


@dataclass(frozen=True)
class StreamingReplayResult:
    """Deterministic replay comparison for streaming token classification."""

    final_spans: tuple[TokenClassificationSpan, ...]
    batch_spans: tuple[TokenClassificationSpan, ...]
    events: tuple[TokenClassificationStreamEvent, ...]
    audit_log: tuple[dict[str, object], ...]
    latency: dict[str, float]

    @property
    def span_diff(self) -> list[tuple[str, TokenClassificationSpan]]:
        """Return spans present on only one side of replay comparison."""
        streamed = {_span_signature(span): span for span in self.final_spans}
        batch = {_span_signature(span): span for span in self.batch_spans}
        diff: list[tuple[str, TokenClassificationSpan]] = []
        for signature, span in sorted(batch.items()):
            if signature not in streamed:
                diff.append(("missing", span))
        for signature, span in sorted(streamed.items()):
            if signature not in batch:
                diff.append(("extra", span))
        return diff


class StreamingTokenClassifier:
    """Incrementally classify appended text with a bounded recomputation tail."""

    def __init__(
        self,
        classifier: TokenClassifier,
        *,
        window_chars: int = 4096,
        tokenizer_context_chars: int = 128,
        max_entity_chars: int = 512,
        confidence_threshold: float = 0.0,
    ) -> None:
        if window_chars < 1:
            raise ValueError("window_chars must be positive")
        if tokenizer_context_chars < 0:
            raise ValueError("tokenizer_context_chars must be non-negative")
        if max_entity_chars < 1:
            raise ValueError("max_entity_chars must be positive")

        self.classifier = classifier
        self.window_chars = int(window_chars)
        self.tokenizer_context_chars = int(tokenizer_context_chars)
        self.max_entity_chars = int(max_entity_chars)
        self.confidence_threshold = float(confidence_threshold)
        self.recomputation_tail_chars = max(
            1,
            min(
                self.window_chars,
                self.tokenizer_context_chars + self.max_entity_chars,
            ),
        )

        self._tail_text = ""
        self._tail_start = 0
        self._tail_byte_start = 0
        self._document_length = 0
        self._document_byte_length = 0
        self._active_spans: dict[str, TokenClassificationSpan] = {}
        self._final_spans: list[TokenClassificationSpan] = []
        self._final_span_ids: set[str] = set()
        self._audit_log: list[dict[str, object]] = []
        self._append_latencies: list[float] = []
        self._window_lengths: list[int] = []
        self._closed = False

    @property
    def final_spans(self) -> tuple[TokenClassificationSpan, ...]:
        """Return committed final spans."""
        return tuple(self._final_spans)

    @property
    def audit_log(self) -> tuple[dict[str, object], ...]:
        """Return PHI-safe audit events for the stream."""
        return tuple(self._audit_log)

    @property
    def max_observed_window_chars(self) -> int:
        """Return the largest model input window used so far."""
        return max(self._window_lengths, default=0)

    def append(self, chunk: str) -> tuple[TokenClassificationStreamEvent, ...]:
        """Append text and emit span changes for the affected tail window."""
        if self._closed:
            raise RuntimeError("cannot append after finish")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if not chunk:
            return ()

        events: list[TokenClassificationStreamEvent] = []
        step = max(1, self.window_chars - self.recomputation_tail_chars)
        for offset in range(0, len(chunk), step):
            part = chunk[offset : offset + step]
            self._tail_text += part
            self._document_length += len(part)
            self._document_byte_length += len(part.encode("utf-8"))
            events.extend(self._recompute(final=False))
        return tuple(events)

    def finish(self) -> tuple[TokenClassificationStreamEvent, ...]:
        """Flush the remaining tail and emit a final reconciliation event."""
        if self._closed:
            return ()
        self._closed = True
        events = list(self._recompute(final=True))
        final_event = TokenClassificationStreamEvent(
            type="final",
            final_spans=tuple(self._final_spans),
            latency_ms=self.latency_summary()["p99_append_latency_ms"],
            window_chars=self.max_observed_window_chars,
        )
        events.append(final_event)
        self._audit_log.append(final_event.to_audit_dict())
        return tuple(events)

    def latency_summary(self) -> dict[str, float]:
        """Return deterministic latency/window metrics for acceptance tests."""
        return {
            "append_count": float(len(self._append_latencies)),
            "p99_append_latency_ms": _percentile(self._append_latencies, 99) * 1000.0,
            "max_append_latency_ms": (
                max(self._append_latencies, default=0.0) * 1000.0
            ),
            "p99_window_chars": _percentile(
                [float(value) for value in self._window_lengths],
                99,
            ),
            "max_window_chars": float(self.max_observed_window_chars),
            "single_window_chars": float(self.window_chars),
        }

    def _recompute(
        self,
        *,
        final: bool,
    ) -> tuple[TokenClassificationStreamEvent, ...]:
        if not self._tail_text:
            return ()

        window_text = self._tail_text[-self.window_chars :]
        trim_chars = len(self._tail_text) - len(window_text)
        window_start = self._tail_start + trim_chars
        window_byte_start = self._tail_byte_start + len(
            self._tail_text[:trim_chars].encode("utf-8")
        )

        start_time = time.perf_counter()
        predictions = _normalize_classifier_output(self.classifier(window_text))
        spans = coerce_token_classification_spans(
            list(predictions),
            window_text,
            base_offset=window_start,
            base_byte_offset=window_byte_start,
            confidence_threshold=self.confidence_threshold,
        )
        latency = time.perf_counter() - start_time
        self._append_latencies.append(latency)
        self._window_lengths.append(len(window_text))

        tracked_spans = [span for span in spans if span.end > self._tail_start]
        events, next_active = reconcile_stream_spans(
            self._active_spans,
            tracked_spans,
        )
        self._active_spans = next_active

        commit_cutoff = self._commit_cutoff(spans, final=final)
        self._commit_prefix(commit_cutoff)

        for event in events:
            self._audit_log.append(event.to_audit_dict())
        return tuple(events)

    def _commit_cutoff(
        self,
        spans: Sequence[TokenClassificationSpan],
        *,
        final: bool,
    ) -> int:
        if final:
            cutoff = self._document_length
        else:
            cutoff = max(
                self._tail_start,
                self._document_length - self.recomputation_tail_chars,
            )
        for span in spans:
            if span.start < cutoff < span.end:
                cutoff = span.start
        return max(self._tail_start, min(cutoff, self._document_length))

    def _commit_prefix(self, cutoff: int) -> None:
        if cutoff <= self._tail_start:
            return

        for span in sorted(
            self._active_spans.values(), key=lambda item: (item.start, item.end)
        ):
            if span.end <= cutoff and span.id not in self._final_span_ids:
                self._final_spans.append(span)
                self._final_span_ids.add(span.id)

        self._active_spans = {
            entity_id: span
            for entity_id, span in self._active_spans.items()
            if span.end > cutoff
        }

        drop_chars = cutoff - self._tail_start
        dropped = self._tail_text[:drop_chars]
        self._tail_text = self._tail_text[drop_chars:]
        self._tail_start = cutoff
        self._tail_byte_start += len(dropped.encode("utf-8"))


async def stream_token_classifier(
    classifier: TokenClassifier,
    chunks: Iterable[str] | AsyncIterable[str],
    **kwargs: Any,
):
    """Yield streaming token-classification events for sync or async chunks."""
    streamer = StreamingTokenClassifier(classifier, **kwargs)
    async for chunk in _aiter_chunks(chunks):
        for event in streamer.append(chunk):
            yield event
    for event in streamer.finish():
        yield event


def replay_token_classifier(
    classifier: TokenClassifier,
    text: str,
    chunks: Iterable[str],
    **kwargs: Any,
) -> StreamingReplayResult:
    """Replay chunks through the streaming classifier and compare to batch."""
    batch_spans = tuple(
        coerce_token_classification_spans(
            list(_normalize_classifier_output(classifier(text))),
            text,
            confidence_threshold=float(kwargs.get("confidence_threshold", 0.0)),
        )
    )
    streamer = StreamingTokenClassifier(classifier, **kwargs)
    events: list[TokenClassificationStreamEvent] = []
    for chunk in chunks:
        events.extend(streamer.append(chunk))
    events.extend(streamer.finish())
    return StreamingReplayResult(
        final_spans=streamer.final_spans,
        batch_spans=batch_spans,
        events=tuple(events),
        audit_log=streamer.audit_log,
        latency=streamer.latency_summary(),
    )


async def _aiter_chunks(chunks: Iterable[str] | AsyncIterable[str]):
    if hasattr(chunks, "__aiter__"):
        async for chunk in chunks:  # type: ignore[union-attr]
            yield chunk
        return
    for chunk in chunks:  # type: ignore[union-attr]
        yield chunk


def _normalize_classifier_output(result: Any) -> Sequence[object]:
    if isinstance(result, Awaitable):
        raise TypeError("StreamingTokenClassifier requires a synchronous classifier")
    if result is None:
        return ()
    entities = getattr(result, "entities", None)
    if entities is not None:
        return tuple(entities)
    if isinstance(result, dict):
        return (result,)
    return tuple(result)


def _span_signature(
    span: TokenClassificationSpan,
) -> tuple[str, int, int, str, int | None, int | None]:
    return (
        span.label,
        span.start,
        span.end,
        span.text,
        span.byte_start,
        span.byte_end,
    )


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class AdvancedNERProcessor:
    """
    Advanced NER processor implementing proven filtering techniques.

    Based on the OpenMed Gradio app's successful approach:
    - Smart BIO token grouping
    - Confidence threshold filtering
    - Content validation
    - Edge case handling
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        min_length: int = 1,
        remove_punctuation: bool = True,
        strip_edges: bool = True,
        merge_adjacent: bool = True,
        max_merge_gap: int = 10,
    ):
        """
        Initialize the advanced NER processor.

        Args:
            min_confidence: Minimum confidence threshold for entities
            min_length: Minimum length for valid entities
            remove_punctuation: Whether to filter punctuation-only entities
            strip_edges: Whether to strip punctuation from entity edges
            merge_adjacent: Whether to merge adjacent entities of same type
            max_merge_gap: Maximum character gap for merging adjacent entities
        """
        self.min_confidence = min_confidence
        self.min_length = min_length
        self.remove_punctuation = remove_punctuation
        self.strip_edges = strip_edges
        self.merge_adjacent = merge_adjacent
        self.max_merge_gap = max_merge_gap

        # Regex for content detection
        self.has_content = (
            re.compile(r"[A-Za-z0-9]") if remove_punctuation else re.compile(r".")
        )

        # Patterns to exclude (known false positives)
        self.exclude_patterns = [
            r"^[\s\-.,!?;:()[\]{}\"'_]+$",  # Only punctuation/whitespace
            r"^\d{1,2}$",  # Single/double digits only
            r"^[.,!?;:]+$",  # Only punctuation
        ]

    def ner_filtered(
        self, text: str, pipeline_result: List[Dict[str, Any]]
    ) -> List[EntitySpan]:
        """
        Apply confidence and punctuation filtering to NER pipeline results.
        This is the proven filtering approach that eliminates spurious predictions.

        Args:
            text: Original input text
            pipeline_result: Raw output from HuggingFace NER pipeline

        Returns:
            List of filtered EntitySpan objects
        """
        logger.debug("Processing %d raw entities", len(pipeline_result))

        filtered_entities = []

        for entity in pipeline_result:
            # Confidence filter
            if entity.get("score", 0) < self.min_confidence:
                continue

            word = entity.get("word", "")

            # Length filter
            if len(word.strip()) < self.min_length:
                continue

            # Content filter - must have actual content
            if self.remove_punctuation and not self.has_content.search(word):
                continue

            # Exclude pattern filter
            if any(re.match(pattern, word) for pattern in self.exclude_patterns):
                continue

            # Create EntitySpan
            span = EntitySpan(
                text=word,
                label=entity.get("entity", "").replace("B-", "").replace("I-", ""),
                start=entity.get("start", 0),
                end=entity.get("end", 0),
                score=entity.get("score", 0.0),
            )

            filtered_entities.append(span)

        logger.debug("After filtering: %d entities", len(filtered_entities))
        return filtered_entities

    def smart_group_entities(
        self, tokens: List[Dict[str, Any]], text: str
    ) -> List[EntitySpan]:
        """
        Smart entity grouping that properly merges sub-tokens into complete entities.

        This fixes the issue where aggregation_strategy="simple" creates overlapping spans
        by implementing proper BIO tag handling.

        Args:
            tokens: Raw token-level predictions from transformer model
            text: Original input text

        Returns:
            List of properly grouped EntitySpan objects
        """
        if not tokens:
            return []

        entities = []
        current_entity = None

        for token in tokens:
            label = token.get("entity", "O")
            score = token.get("score", 0.0)
            start = token.get("start", 0)
            end = token.get("end", 0)

            # Skip O (Outside) tags
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            # Clean the label (remove B- and I- prefixes)
            clean_label = label.replace("B-", "").replace("I-", "")
            crosses_hard_line = bool(
                current_entity
                and any(
                    char in "\r\n\v\f\x85\u2028\u2029"
                    for char in text[current_entity.end : start]
                )
            )

            # Start new entity (B- tag or different entity type)
            if (
                label.startswith("B-")
                or current_entity is None
                or current_entity.label != clean_label
                or crosses_hard_line
            ):
                if current_entity:
                    entities.append(current_entity)

                current_entity = EntitySpan(
                    text=text[start:end],
                    label=clean_label,
                    start=start,
                    end=end,
                    score=score,
                )

            # Continue current entity (I- tag)
            elif current_entity and clean_label == current_entity.label:
                # Extend the current entity
                current_entity.end = end
                current_entity.text = text[current_entity.start : end]
                current_entity.score = (
                    current_entity.score + score
                ) / 2  # Average scores

        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)

        logger.debug(
            "Smart grouping created %d entities from %d tokens",
            len(entities),
            len(tokens),
        )
        return entities

    def advanced_filter(
        self, entities: List[EntitySpan], text: str
    ) -> List[EntitySpan]:
        """
        Advanced filtering with edge stripping and additional quality checks.

        Args:
            entities: List of EntitySpan objects to filter
            text: Original input text

        Returns:
            List of filtered EntitySpan objects
        """
        filtered = []

        for entity in entities:
            # Skip if below confidence threshold
            if entity.score < self.min_confidence:
                continue

            original_text = entity.text

            # Strip punctuation from edges if enabled
            if self.strip_edges:
                stripped = original_text.strip(".,!?;:()[]{}\"'-_")
                if not stripped:
                    continue

                # Update entity text and positions if stripped
                if stripped != original_text:
                    # Find new start/end positions
                    start_offset = original_text.find(stripped)
                    entity.text = stripped
                    entity.start += start_offset
                    entity.end = entity.start + len(stripped)

            # Final content validation
            if not re.search(r"[A-Za-z0-9]", entity.text):
                continue

            # Length check after stripping
            if len(entity.text.strip()) < self.min_length:
                continue

            filtered.append(entity)

        return filtered

    def merge_adjacent_entities(
        self, entities: List[EntitySpan], original_text: str
    ) -> List[EntitySpan]:
        """
        Merge adjacent entities of the same type that are separated by small gaps.

        Useful for handling cases like "BRCA1 and BRCA2" or "HER2-positive".

        Args:
            entities: List of EntitySpan objects to potentially merge
            original_text: Original input text

        Returns:
            List of EntitySpan objects with adjacent entities merged
        """
        if len(entities) < 2:
            return entities

        merged = []
        current = entities[0]

        for next_entity in entities[1:]:
            # Check if same entity type and close proximity
            if (
                current.label == next_entity.label
                and next_entity.start - current.end <= self.max_merge_gap
            ):
                # Check what's between them
                gap_text = original_text[current.end : next_entity.start]

                # Merge if gap contains only connecting words/punctuation
                if not any(
                    char in "\r\n\v\f\x85\u2028\u2029" for char in gap_text
                ) and re.fullmatch(
                    r"(?:[ \t]*[-,/][ \t]*|[ \t]+(?:and|or)[ \t]+|[ \t]*)",
                    gap_text,
                    re.IGNORECASE,
                ):
                    # Extend current entity to include the next one
                    current.text = original_text[current.start : next_entity.end]
                    current.end = next_entity.end
                    current.score = (current.score + next_entity.score) / 2
                    continue

            # No merge, add current and move to next
            merged.append(current)
            current = next_entity

        # Don't forget the last entity
        merged.append(current)

        logger.debug("Merge process: %d -> %d entities", len(entities), len(merged))
        return merged

    def process_pipeline_output(
        self,
        text: str,
        pipeline_output: List[Dict[str, Any]],
        use_smart_grouping: bool = True,
    ) -> List[EntitySpan]:
        """
        Complete processing pipeline for NER output.

        Args:
            text: Original input text
            pipeline_output: Raw output from HuggingFace pipeline
            use_smart_grouping: Whether to use smart BIO token grouping

        Returns:
            List of processed EntitySpan objects
        """
        logger.info(
            "Processing pipeline output with %d raw predictions",
            len(pipeline_output),
        )

        # Step 1: Smart grouping if requested and we have token-level output
        if use_smart_grouping and pipeline_output:
            # Check if we have token-level data (no aggregation_strategy used)
            first_item = pipeline_output[0]
            if "entity" in first_item and not "entity_group" in first_item:
                entities = self.smart_group_entities(pipeline_output, text)
            else:
                # Already grouped, convert to EntitySpan format
                entities = []
                for item in pipeline_output:
                    span = EntitySpan(
                        text=item.get("word", ""),
                        label=item.get("entity_group", item.get("entity", "")),
                        start=item.get("start", 0),
                        end=item.get("end", 0),
                        score=item.get("score", 0.0),
                    )
                    entities.append(span)
        else:
            # Use basic filtering approach
            entities = self.ner_filtered(text, pipeline_output)

        # Step 2: Advanced filtering
        entities = self.advanced_filter(entities, text)

        # Step 3: Merge adjacent entities if enabled
        if self.merge_adjacent:
            entities = self.merge_adjacent_entities(entities, text)

        logger.info("Final result: %d high-quality entities", len(entities))
        return entities

    def create_entity_summary(self, entities: List[EntitySpan]) -> Dict[str, Any]:
        """
        Create a summary of detected entities.

        Args:
            entities: List of EntitySpan objects

        Returns:
            Dictionary with entity statistics and examples
        """
        if not entities:
            return {"total": 0, "by_type": {}, "confidence_stats": {}}

        by_type = {}
        scores = []

        for entity in entities:
            label = entity.label
            if label not in by_type:
                by_type[label] = {
                    "count": 0,
                    "examples": [],
                    "avg_confidence": 0.0,
                    "scores": [],
                }

            by_type[label]["count"] += 1
            by_type[label]["scores"].append(entity.score)
            scores.append(entity.score)

            # Keep unique examples (up to 5)
            if (
                entity.text not in by_type[label]["examples"]
                and len(by_type[label]["examples"]) < 5
            ):
                by_type[label]["examples"].append(entity.text)

        # Calculate average confidences
        for label in by_type:
            by_type[label]["avg_confidence"] = sum(by_type[label]["scores"]) / len(
                by_type[label]["scores"]
            )
            del by_type[label]["scores"]  # Remove raw scores from output

        confidence_stats = {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
        }

        return {
            "total": len(entities),
            "by_type": by_type,
            "confidence_stats": confidence_stats,
            "filter_settings": {
                "min_confidence": self.min_confidence,
                "min_length": self.min_length,
                "remove_punctuation": self.remove_punctuation,
                "strip_edges": self.strip_edges,
                "merge_adjacent": self.merge_adjacent,
            },
        }


def create_advanced_processor(
    confidence_threshold: float = 0.60, **kwargs
) -> AdvancedNERProcessor:
    """
    Convenience function to create an AdvancedNERProcessor with recommended settings.

    Args:
        confidence_threshold: Minimum confidence for entity predictions
        **kwargs: Additional arguments for AdvancedNERProcessor

    Returns:
        Configured AdvancedNERProcessor instance
    """
    return AdvancedNERProcessor(
        min_confidence=confidence_threshold,
        min_length=1,
        remove_punctuation=True,
        strip_edges=True,
        merge_adjacent=True,
        max_merge_gap=10,
        **kwargs,
    )
