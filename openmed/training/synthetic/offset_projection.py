"""Span offset projection helpers for synthetic translation augmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

SPAN_CONTAINER_KEYS: tuple[str, ...] = (
    "gold_spans",
    "labels",
    "entities",
    "spans",
)


class SpanProjectionError(ValueError):
    """Raised when translated text cannot be aligned to source spans."""


class SpanTranslator(Protocol):
    """Minimal translator contract required for span projection."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate ``text`` from ``source_lang`` to ``target_lang``."""


@dataclass(frozen=True)
class TokenSpan:
    """One token and its character offsets."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SpanAnnotation:
    """A normalized span record with stable JSONL serialization."""

    start: int
    end: int
    label: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready span mapping."""

        return {
            "end": self.end,
            "label": self.label,
            "metadata": dict(self.metadata),
            "start": self.start,
            "text": self.text,
        }


def tokenize_for_alignment(text: str) -> tuple[TokenSpan, ...]:
    """Retokenize text into word-or-punctuation spans for boundary checks."""

    return tuple(
        TokenSpan(match.start(), match.end(), match.group(0))
        for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    )


def normalize_span_annotations(
    data: Mapping[str, Any] | Iterable[Any],
    *,
    source_text: str | None = None,
    span_keys: Sequence[str] = SPAN_CONTAINER_KEYS,
) -> tuple[SpanAnnotation, ...]:
    """Normalize span-like mappings or dataclasses into ``SpanAnnotation`` rows."""

    raw_spans: Iterable[Any]
    if isinstance(data, Mapping):
        if source_text is None:
            source_text = str(data.get("text", ""))
        raw_spans = ()
        for key in span_keys:
            candidate = data.get(key)
            if candidate is not None:
                raw_spans = candidate
                break
    else:
        raw_spans = data

    spans = tuple(
        _normalize_one_span(span, source_text=source_text) for span in raw_spans
    )
    return tuple(sorted(spans, key=lambda span: (span.start, span.end, span.label)))


def validate_span_integrity(
    text: str,
    spans: Iterable[SpanAnnotation],
    *,
    allow_overlap: bool = False,
) -> None:
    """Validate that spans point at their serialized text."""

    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    previous_end = -1
    for span in ordered:
        if span.start < 0 or span.end > len(text) or span.start >= span.end:
            raise SpanProjectionError(
                f"invalid span bounds for {span.label}: {span.start}:{span.end}"
            )
        if text[span.start : span.end] != span.text:
            raise SpanProjectionError(
                f"span text mismatch for {span.label}: {span.start}:{span.end}"
            )
        if not allow_overlap and span.start < previous_end:
            raise SpanProjectionError(
                f"overlapping projected span for {span.label}: {span.start}:{span.end}"
            )
        previous_end = max(previous_end, span.end)


def span_corruption_count(text: str, spans: Iterable[SpanAnnotation]) -> int:
    """Return the number of spans whose offsets no longer match their text."""

    corrupt = 0
    for span in spans:
        if (
            span.start < 0
            or span.end > len(text)
            or span.start >= span.end
            or text[span.start : span.end] != span.text
        ):
            corrupt += 1
    return corrupt


def realign_translated_spans(
    *,
    source_text: str,
    translated_text: str,
    source_spans: Iterable[SpanAnnotation],
    translator: SpanTranslator,
    source_language: str,
    target_language: str,
) -> tuple[SpanAnnotation, ...]:
    """Translate and re-locate entity spans in already translated text.

    Each source span is translated independently, then found in the translated
    text after retokenizing the target string. If any span cannot be recovered,
    the caller should drop the augmented example rather than emit noisy labels.
    """

    del source_text
    tokenize_for_alignment(translated_text)
    projected: list[SpanAnnotation] = []
    cursor = 0
    for source_span in sorted(source_spans, key=lambda span: (span.start, span.end)):
        target_value = translator.translate(
            source_span.text,
            source_language,
            target_language,
        ).strip()
        if not target_value:
            raise SpanProjectionError(f"empty translated span for {source_span.label}")
        start = locate_text_span(translated_text, target_value, start_at=cursor)
        if start is None:
            raise SpanProjectionError(
                f"could not realign {source_span.label} span {source_span.start}:"
                f"{source_span.end}"
            )
        end = start + len(target_value)
        metadata = {
            **dict(source_span.metadata),
            "projected_from_end": source_span.end,
            "projected_from_start": source_span.start,
        }
        projected.append(
            SpanAnnotation(
                start=start,
                end=end,
                label=source_span.label,
                text=translated_text[start:end],
                metadata=metadata,
            )
        )
        cursor = end
    validate_span_integrity(translated_text, projected)
    return tuple(projected)


def locate_text_span(text: str, needle: str, *, start_at: int = 0) -> int | None:
    """Locate ``needle`` in ``text`` with conservative token boundary handling."""

    if not needle:
        return None

    start = max(start_at, 0)
    while True:
        index = text.find(needle, start)
        if index == -1:
            break
        if _boundary_match(text, index, index + len(needle)):
            return index
        start = index + 1

    folded_text = text.casefold()
    folded_needle = needle.casefold()
    start = max(start_at, 0)
    while True:
        index = folded_text.find(folded_needle, start)
        if index == -1:
            return None
        if _boundary_match(text, index, index + len(needle)):
            return index
        start = index + 1


def _normalize_one_span(
    span: Any,
    *,
    source_text: str | None,
) -> SpanAnnotation:
    if isinstance(span, SpanAnnotation):
        normalized = span
    elif isinstance(span, Mapping):
        start = _coerce_int(span.get("start"), "start")
        end = _coerce_int(span.get("end"), "end")
        label = str(span.get("label") or span.get("entity") or span.get("type") or "")
        if not label:
            raise SpanProjectionError("span label is required")
        text = str(span.get("text") or span.get("word") or "")
        if not text and source_text is not None:
            text = source_text[start:end]
        metadata = span.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise SpanProjectionError("span metadata must be a mapping")
        normalized = SpanAnnotation(
            start=start,
            end=end,
            label=label,
            text=text,
            metadata=dict(metadata),
        )
    else:
        start = _coerce_int(getattr(span, "start"), "start")
        end = _coerce_int(getattr(span, "end"), "end")
        label = str(getattr(span, "label"))
        text = str(getattr(span, "text", ""))
        if not text and source_text is not None:
            text = source_text[start:end]
        metadata = getattr(span, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            raise SpanProjectionError("span metadata must be a mapping")
        normalized = SpanAnnotation(
            start=start,
            end=end,
            label=label,
            text=text,
            metadata=dict(metadata),
        )

    if source_text is not None:
        if normalized.start < 0 or normalized.end > len(source_text):
            raise SpanProjectionError(
                f"span bounds outside source text: {normalized.start}:{normalized.end}"
            )
        if normalized.start >= normalized.end:
            raise SpanProjectionError(
                f"empty source span: {normalized.start}:{normalized.end}"
            )
        if source_text[normalized.start : normalized.end] != normalized.text:
            raise SpanProjectionError(
                f"source span text mismatch for {normalized.label}: "
                f"{normalized.start}:{normalized.end}"
            )
    return normalized


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SpanProjectionError(f"span {field_name} must be an integer") from exc


def _boundary_match(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not _is_word_char(before) and not _is_word_char(after)


def _is_word_char(value: str) -> bool:
    return bool(value) and (value.isalnum() or value == "_")


__all__ = [
    "SPAN_CONTAINER_KEYS",
    "SpanAnnotation",
    "SpanProjectionError",
    "SpanTranslator",
    "TokenSpan",
    "locate_text_span",
    "normalize_span_annotations",
    "realign_translated_spans",
    "span_corruption_count",
    "tokenize_for_alignment",
    "validate_span_integrity",
]
