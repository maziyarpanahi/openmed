"""Span post-processing helpers shared across privacy-filter backends.

When token classifiers emit slightly-too-greedy spans (e.g. "alice@hospital.org and"
absorbs the trailing "and"), these helpers tighten the boundaries before the
span reaches downstream redaction logic. Pure-Python; no array-framework
dependencies.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, Final


def trim_span_whitespace(start: int, end: int, text: str) -> tuple[int, int]:
    """Strip leading and trailing whitespace from ``text[start:end]``.

    Returns the inclusive ``[start, end)`` indices into ``text`` after
    trimming. ``start`` and ``end`` are clamped so ``start <= end``.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


_PRIVACY_FILTER_SPAN_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("url", re.compile(r"\b(?:https?://|www\.)[^\s,;)\]]+")),
    ("phone", re.compile(r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")),
)


def refine_privacy_filter_span(
    label: str,
    start: int,
    end: int,
    text: str,
) -> tuple[int, int]:
    """Tighten obvious structured-PII spans when the model absorbs glue words.

    For email / URL / phone labels, locate the strict regex match inside
    the model-suggested span and shrink to that. For any label, drop a
    trailing ``" and"`` or ``" or"`` that the model often grabs because
    it sat next to the entity in training data.
    """
    start, end = trim_span_whitespace(start, end, text)
    span_text = text[start:end]
    normalized = label.lower()

    for label_hint, pattern in _PRIVACY_FILTER_SPAN_PATTERNS:
        if label_hint not in normalized:
            continue
        match = pattern.search(span_text)
        if match:
            return start + match.start(), start + match.end()

    for suffix in (" and", " or"):
        if span_text.lower().endswith(suffix):
            end -= len(suffix)
            break
    return trim_span_whitespace(start, end, text)


def _byte_offset(text: str, char_offset: int) -> int:
    return len(text[: max(0, char_offset)].encode("utf-8"))


def stable_span_id(label: str, start: int) -> str:
    """Return a deterministic PHI-free id for a streamed entity anchor."""
    digest = hashlib.sha256(f"{label}\0{int(start)}".encode("utf-8")).hexdigest()
    return f"ent_{digest[:16]}"


@dataclass(frozen=True)
class TokenClassificationSpan:
    """Entity span emitted by incremental token-classification streaming."""

    id: str
    label: str
    start: int
    end: int
    score: float
    text: str = ""
    byte_start: int | None = None
    byte_end: int | None = None

    def to_dict(self, *, include_text: bool = True) -> dict[str, object]:
        """Return a JSON-serializable span payload."""
        payload: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "score": self.score,
        }
        if include_text:
            payload["text"] = self.text
        return payload

    def to_audit_dict(self) -> dict[str, object]:
        """Return a PHI-safe audit payload with hashes instead of raw text."""
        payload = self.to_dict(include_text=False)
        if self.text:
            payload["text_hash"] = (
                "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            )
        return payload


@dataclass(frozen=True)
class TokenClassificationStreamEvent:
    """Emit/retract/final event for streaming token classification."""

    type: str
    entity_id: str | None = None
    span: TokenClassificationSpan | None = None
    reason: str | None = None
    final_spans: tuple[TokenClassificationSpan, ...] = ()
    latency_ms: float | None = None
    window_chars: int | None = None

    def to_dict(self, *, include_text: bool = True) -> dict[str, object]:
        """Return a JSON-serializable event payload."""
        payload: dict[str, object] = {"type": self.type}
        if self.entity_id is not None:
            payload["entity_id"] = self.entity_id
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.span is not None:
            payload["span"] = self.span.to_dict(include_text=include_text)
        if self.final_spans:
            payload["final_spans"] = [
                span.to_dict(include_text=include_text) for span in self.final_spans
            ]
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.window_chars is not None:
            payload["window_chars"] = self.window_chars
        return payload

    def to_audit_dict(self) -> dict[str, object]:
        """Return a PHI-safe event payload for logs and audit trails."""
        payload: dict[str, object] = {"type": self.type}
        if self.entity_id is not None:
            payload["entity_id"] = self.entity_id
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.span is not None:
            payload["span"] = self.span.to_audit_dict()
        if self.final_spans:
            payload["final_spans"] = [span.to_audit_dict() for span in self.final_spans]
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.window_chars is not None:
            payload["window_chars"] = self.window_chars
        return payload


def coerce_token_classification_spans(
    predictions: list[object],
    text: str,
    *,
    base_offset: int = 0,
    base_byte_offset: int = 0,
    confidence_threshold: float = 0.0,
) -> list[TokenClassificationSpan]:
    """Convert backend entity dicts/objects to absolute streaming spans."""
    spans: list[TokenClassificationSpan] = []
    for item in predictions:
        if isinstance(item, TokenClassificationSpan):
            span = item
            if base_offset or base_byte_offset:
                span = replace(
                    span,
                    start=span.start + base_offset,
                    end=span.end + base_offset,
                    byte_start=(
                        None
                        if span.byte_start is None
                        else span.byte_start + base_byte_offset
                    ),
                    byte_end=(
                        None
                        if span.byte_end is None
                        else span.byte_end + base_byte_offset
                    ),
                )
            spans.append(span)
            continue

        getter = (
            item.get
            if isinstance(item, dict)
            else lambda key, default=None: getattr(item, key, default)
        )
        raw_start = getter("start")
        raw_end = getter("end")
        if raw_start is None or raw_end is None:
            continue
        start = int(raw_start)
        end = int(raw_end)
        if end <= start:
            continue
        score = float(
            getter(
                "score",
                getter("confidence", 0.0),
            )
            or 0.0
        )
        if score < confidence_threshold:
            continue
        label = str(
            getter(
                "entity_group",
                getter("entity", getter("label", getter("entity_type", "UNKNOWN"))),
            )
            or "UNKNOWN"
        )
        label = (
            label.removeprefix("B-")
            .removeprefix("I-")
            .removeprefix("E-")
            .removeprefix("S-")
        )
        local_text = str(getter("word", getter("text", text[start:end])) or "")
        absolute_start = base_offset + start
        absolute_end = base_offset + end
        byte_start = base_byte_offset + _byte_offset(text, start)
        byte_end = base_byte_offset + _byte_offset(text, end)
        spans.append(
            TokenClassificationSpan(
                id=stable_span_id(label, absolute_start),
                label=label,
                start=absolute_start,
                end=absolute_end,
                byte_start=byte_start,
                byte_end=byte_end,
                score=score,
                text=local_text or text[start:end],
            )
        )

    return sorted(spans, key=lambda span: (span.start, span.end, span.label, span.id))


def reconcile_stream_spans(
    active_spans: dict[str, TokenClassificationSpan],
    current_spans: list[TokenClassificationSpan],
) -> tuple[list[TokenClassificationStreamEvent], dict[str, TokenClassificationSpan]]:
    """Compute retract/emit events needed to reach ``current_spans``."""
    events: list[TokenClassificationStreamEvent] = []
    next_active = {span.id: span for span in current_spans}

    for entity_id, previous in sorted(
        active_spans.items(), key=lambda item: (item[1].start, item[1].end, item[0])
    ):
        current = next_active.get(entity_id)
        if current is None:
            events.append(
                TokenClassificationStreamEvent(
                    type="retract",
                    entity_id=entity_id,
                    span=previous,
                    reason="span_removed",
                )
            )
        elif _span_changed(previous, current):
            events.append(
                TokenClassificationStreamEvent(
                    type="retract",
                    entity_id=entity_id,
                    span=previous,
                    reason="span_updated",
                )
            )

    for span in current_spans:
        previous = active_spans.get(span.id)
        if previous is None or _span_changed(previous, span):
            events.append(
                TokenClassificationStreamEvent(
                    type="emit",
                    entity_id=span.id,
                    span=span,
                )
            )

    return events, next_active


def _span_changed(
    previous: TokenClassificationSpan,
    current: TokenClassificationSpan,
) -> bool:
    return (
        previous.label != current.label
        or previous.start != current.start
        or previous.end != current.end
        or previous.byte_start != current.byte_start
        or previous.byte_end != current.byte_end
        or previous.text != current.text
    )


def stable_span_key(span: Any) -> tuple[int, int, str, str]:
    """Return a deterministic ordering key for span-like objects.

    The key intentionally depends only on source offsets plus optional label and
    text fields, so downstream decoders can make stable tie-break decisions
    without depending on object identity or model output order.
    """

    start = int(getattr(span, "start", 0))
    end = int(getattr(span, "end", start))
    label = str(getattr(span, "label", ""))
    span_text = str(getattr(span, "text", ""))
    return start, end, label.casefold(), span_text.casefold()


__all__ = [
    "TokenClassificationSpan",
    "TokenClassificationStreamEvent",
    "coerce_token_classification_spans",
    "reconcile_stream_spans",
    "refine_privacy_filter_span",
    "stable_span_id",
    "stable_span_key",
    "trim_span_whitespace",
]
