"""Language-aware sentence segmentation utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.script_detect import is_han_dominant

# Python 3.12 emits SyntaxWarnings for old-style regex escapes in pysbd.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

_SEGMENTER_CACHE: Dict[Tuple[str, bool], Any] = {}

_CHINESE_TERMINATORS = frozenset({"。", "！", "？", "；", "．", "｡"})
_CHINESE_OPEN_TO_CLOSE = {
    "「": "」",
    "『": "』",
    "《": "》",
    "（": "）",
    "〔": "〕",
}
_CHINESE_CLOSERS = frozenset(_CHINESE_OPEN_TO_CLOSE.values())


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


def _uses_chinese_segmenter(text: str, language: str) -> bool:
    normalized_language = language.casefold().replace("_", "-")
    return (
        normalized_language == "zh"
        or normalized_language.startswith("zh-")
        or is_han_dominant(text)
    )


def _is_decimal_point(text: str, index: int) -> bool:
    if text[index] != "．" or index == 0 or index + 1 >= len(text):
        return False
    return text[index - 1].isdigit() and text[index + 1].isdigit()


def _continues_chinese_sentence(char: str) -> bool:
    return char.isspace() or char in _CHINESE_TERMINATORS or char in _CHINESE_CLOSERS


def _chinese_spans(text: str) -> List[SentenceSpan]:
    spans: List[SentenceSpan] = []
    stack: List[str] = []
    start = 0
    boundary_ready = False
    deferred_boundary = False

    for index, char in enumerate(text):
        if deferred_boundary and stack and not _continues_chinese_sentence(char):
            deferred_boundary = False

        if boundary_ready and not _continues_chinese_sentence(char):
            spans.append(SentenceSpan(text[start:index], start, index))
            start = index
            boundary_ready = False

        if char in _CHINESE_OPEN_TO_CLOSE:
            stack.append(char)
            continue

        if char in _CHINESE_CLOSERS:
            if stack and _CHINESE_OPEN_TO_CLOSE[stack[-1]] == char:
                stack.pop()
                if deferred_boundary and not stack:
                    boundary_ready = True
                    deferred_boundary = False
            continue

        if char not in _CHINESE_TERMINATORS or _is_decimal_point(text, index):
            continue

        if stack:
            deferred_boundary = True
        else:
            boundary_ready = True

    if start < len(text):
        spans.append(SentenceSpan(text[start:], start, len(text)))

    if all(text[span.start : span.end] == span.text for span in spans):
        return spans
    return _fallback_spans(text, (span.text for span in spans))


def segment_chinese_text(text: str) -> List[SentenceSpan]:
    """Split Chinese text while preserving exact source-code-point offsets."""
    if not text:
        return []
    return _chinese_spans(text)


def segment_text(
    text: str,
    *,
    language: str = "en",
    clean: bool = False,
    segmenter: Optional[Any] = None,
) -> List[SentenceSpan]:
    """Split ``text`` into sentences and capture exact character spans.

    Chinese and Han-dominant text use the built-in CJK-aware path. Other
    languages retain the existing pySBD behavior.
    """
    if not text:
        return []

    if _uses_chinese_segmenter(text, language):
        return segment_chinese_text(text)

    seg = _get_segmenter(language=language, clean=clean, segmenter=segmenter)
    sentences = seg.segment(text)

    spans: List[SentenceSpan] = []

    if sentences and hasattr(sentences[0], "start") and hasattr(sentences[0], "end"):
        for sentence in sentences:
            sent_text = getattr(sentence, "sent", None)
            if sent_text is None:
                sent_text = text[sentence.start : sentence.end]
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


__all__ = ["SentenceSpan", "segment_chinese_text", "segment_text"]
