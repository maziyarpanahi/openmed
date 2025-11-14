"""Sentence segmentation utilities built on top of pySBD."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Python 3.12 emits SyntaxWarnings for old-style regex escapes in pysbd.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

_SEGMENTER_CACHE: Dict[Tuple[str, bool], Any] = {}


@dataclass(frozen=True)
class SentenceSpan:
    """Represents a sentence and its character boundaries within the source."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("SentenceSpan requires 0 <= start <= end")


def _get_segmenter(
    *,
    language: str,
    clean: bool,
    segmenter: Optional[Any] = None,
) -> Any:
    """Return a cached pySBD segmenter instance."""
    if segmenter is not None:
        return segmenter

    cache_key = (language, clean)
    if cache_key in _SEGMENTER_CACHE:
        return _SEGMENTER_CACHE[cache_key]

    try:
        from pysbd import Segmenter  # type: ignore import
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "pySBD is required for sentence detection. "
            "Install it with `pip install pysbd` or add the `pysbd` dependency."
        ) from exc

    segmenter = Segmenter(
        language=language,
        clean=clean,
        char_span=True,
    )
    _SEGMENTER_CACHE[cache_key] = segmenter
    return segmenter


def _fallback_spans(text: str, sentences: Iterable[str]) -> List[SentenceSpan]:
    """Generate spans when pySBD does not provide char offsets."""
    spans: List[SentenceSpan] = []
    cursor = 0
    for sentence in sentences:
        if not sentence:
            continue

        start = text.find(sentence, cursor)
        if start == -1:
            stripped = sentence.strip()
            if stripped:
                start = text.find(stripped, cursor)
            if start == -1:
                start = cursor
        end = start + len(sentence)
        spans.append(SentenceSpan(sentence, start, end))
        cursor = end
    return spans


def segment_text(
    text: str,
    *,
    language: str = "en",
    clean: bool = False,
    segmenter: Optional[Any] = None,
) -> List[SentenceSpan]:
    """Split ``text`` into sentences using pySBD and capture character spans."""
    if not text:
        return []

    seg = _get_segmenter(language=language, clean=clean, segmenter=segmenter)
    sentences = seg.segment(text)

    spans: List[SentenceSpan] = []

    if sentences and hasattr(sentences[0], "start") and hasattr(sentences[0], "end"):
        for sentence in sentences:
            sent_text = getattr(sentence, "sent", None)
            if sent_text is None:
                sent_text = text[sentence.start:sentence.end]
            spans.append(
                SentenceSpan(
                    sent_text,
                    int(sentence.start),
                    int(sentence.end),
                )
            )
    else:
        spans = _fallback_spans(text, sentences)

    return spans


__all__ = ["SentenceSpan", "segment_text"]
