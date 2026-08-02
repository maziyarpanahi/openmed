"""Language-aware sentence segmentation utilities."""

from __future__ import annotations

import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from ..core.decoding.spans import is_grapheme_boundary, is_indic_text
from ..core.script_detect import is_han_dominant

# Python 3.12 emits SyntaxWarnings for old-style regex escapes in pysbd.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

_SEGMENTER_CACHE: Dict[Tuple[str, str, bool], Any] = {}
_SentenceBackend = Literal["auto", "yasbd"]

_CHINESE_TERMINATORS = frozenset({"。", "！", "？", "；", "．", "｡", "!", "?", ";"})
_CHINESE_OPEN_TO_CLOSE = {
    "「": "」",
    "『": "』",
    "《": "》",
    "（": "）",
    "〔": "〕",
}
_CHINESE_QUOTE_OPENERS = frozenset({"「", "『"})
_CHINESE_CLOSERS = frozenset(_CHINESE_OPEN_TO_CLOSE.values())
_CHINESE_CONTINUATION_PUNCTUATION = frozenset({",", ":", "、", "，", "："})
_COMMON_LATIN_ABBREVIATIONS = frozenset(
    {
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "vs",
    }
)
_INDIC_TERMINATORS = frozenset({".", "!", "?", "।", "॥"})
_INDIC_SENTENCE_CONTINUATIONS = _INDIC_TERMINATORS | frozenset(
    {
        "'",
        '"',
        ")",
        "]",
        "}",
        "»",
        "’",
        "”",
        "›",
        "」",
        "』",
    }
)
_INDIC_HONORIFICS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "डॉ",
        "डा",
        "श्री",
        "श्रीमती",
        "कु",
        "চি",
        "ডা",
        "ডাঃ",
        "ডঃ",
        "డా",
        "శ్రీ",
    }
)


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
    backend: _SentenceBackend = "auto",
) -> Any:
    """Return a cached segmenter instance for the selected backend."""
    if segmenter is not None:
        return segmenter

    cache_key = (backend, language, clean)
    if cache_key in _SEGMENTER_CACHE:
        return _SEGMENTER_CACHE[cache_key]

    if backend == "yasbd":
        try:
            from yasbd.utils.pysbd_adapter import (
                Segmenter,  # type: ignore[import-not-found]
            )
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "yasbd-lib is required for sentence detection when `backend='yasbd'`. "
                "Install the optional extra with `pip install 'openmed[yasbd]'`."
            ) from exc
    else:
        try:
            from pysbd import Segmenter  # type: ignore[import-not-found]
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


def _normalize_yasbd_spans(
    text: str,
    spans: List[SentenceSpan],
) -> List[SentenceSpan]:
    """Normalize YASBD offsets to OpenMed's exact contiguous span contract.

    The YASBD adapter can assign inter-sentence whitespace to the following
    sentence and omit trailing whitespace. OpenMed historically assigns that
    whitespace to the preceding sentence. Validate the adapter offsets before
    moving only whitespace boundaries; non-whitespace gaps fail closed.
    """
    if not spans:
        return []

    raw_cursor = 0
    for span in spans:
        if span.end > len(text) or span.start < raw_cursor:
            raise ValueError("yasbd-lib returned invalid or overlapping offsets")
        if text[raw_cursor : span.start].strip():
            raise ValueError("yasbd-lib returned a non-whitespace span gap")
        if span.text != text[span.start : span.end]:
            raise ValueError("yasbd-lib returned text that does not match its offsets")
        raw_cursor = span.end

    if text[raw_cursor:].strip():
        raise ValueError("yasbd-lib did not cover the complete source text")

    boundaries: List[int] = []
    for span in spans:
        boundary = span.end
        while boundary < len(text) and text[boundary].isspace():
            boundary += 1
        if not boundaries or boundary > boundaries[-1]:
            boundaries.append(boundary)

    normalized: List[SentenceSpan] = []
    start = 0
    for end in boundaries:
        if end > start and not text[start:end].isspace():
            normalized.append(SentenceSpan(text[start:end], start, end))
        elif normalized:
            previous = normalized[-1]
            normalized[-1] = SentenceSpan(
                text[previous.start : end],
                previous.start,
                end,
            )
        start = end

    return normalized


def _uses_chinese_segmenter(text: str, language: str) -> bool:
    normalized_language = language.casefold().replace("_", "-")
    return (
        normalized_language == "zh"
        or normalized_language.startswith("zh-")
        or is_han_dominant(text)
    )


def _is_non_boundary_fullwidth_period(text: str, index: int) -> bool:
    if text[index] != "．" or index == 0:
        return False

    previous = text[index - 1]
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return True
    if (
        previous.isascii()
        and previous.isalpha()
        and following.isascii()
        and following.isalpha()
    ):
        return True

    token_start = index - 1
    while token_start >= 0:
        char = text[token_start]
        if not ((char.isascii() and char.isalpha()) or char in {".", "．"}):
            break
        token_start -= 1
    token = text[token_start + 1 : index].replace("．", ".").casefold().strip(".")
    return token in _COMMON_LATIN_ABBREVIATIONS


def _continues_chinese_sentence(char: str) -> bool:
    return char.isspace() or char in _CHINESE_TERMINATORS or char in _CHINESE_CLOSERS


def _chinese_spans(
    text: str,
    terminators: frozenset[str] = _CHINESE_TERMINATORS,
) -> List[SentenceSpan]:
    spans: List[SentenceSpan] = []
    stack: List[str] = []
    start = 0
    boundary_ready = False
    deferred_boundary = False

    for index, char in enumerate(text):
        if deferred_boundary and stack and not _continues_chinese_sentence(char):
            deferred_boundary = False

        if boundary_ready:
            if char in _CHINESE_CONTINUATION_PUNCTUATION:
                boundary_ready = False
            elif not _continues_chinese_sentence(char):
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

        if char not in terminators or _is_non_boundary_fullwidth_period(
            text,
            index,
        ):
            continue

        if stack and stack[-1] in _CHINESE_QUOTE_OPENERS:
            deferred_boundary = True
        elif not stack:
            boundary_ready = True

    if start < len(text):
        spans.append(SentenceSpan(text[start:], start, len(text)))

    if all(text[span.start : span.end] == span.text for span in spans):
        return spans
    return _fallback_spans(text, (span.text for span in spans))


def _split_yasbd_chinese_semicolons(
    text: str,
    spans: List[SentenceSpan],
    language: str,
) -> List[SentenceSpan]:
    """Add OpenMed's Chinese-semicolon boundary without mutating YASBD globals."""
    if "；" not in text or not _uses_chinese_segmenter(text, language):
        return spans

    refined: List[SentenceSpan] = []
    for span in spans:
        for local in _chinese_spans(span.text, frozenset({"；"})):
            refined.append(
                SentenceSpan(
                    local.text,
                    span.start + local.start,
                    span.start + local.end,
                )
            )
    return refined


def segment_chinese_text(text: str) -> List[SentenceSpan]:
    """Split Chinese text while preserving exact source-code-point offsets."""
    if not text:
        return []
    return _chinese_spans(text)


def _previous_word(text: str, terminator_index: int) -> str:
    cursor = terminator_index
    while cursor > 0:
        category = unicodedata.category(text[cursor - 1])
        if category[0] not in {"L", "M"}:
            break
        cursor -= 1
    return text[cursor:terminator_index].casefold()


def _next_nonspace(text: str, index: int) -> str:
    cursor = index + 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return text[cursor] if cursor < len(text) else ""


def _is_guarded_terminator(text: str, index: int) -> bool:
    char = text[index]
    if (
        char == "."
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isdecimal()
        and text[index + 1].isdecimal()
    ):
        return True

    previous_word = _previous_word(text, index)
    if not previous_word:
        return False
    if char not in {".", "।"}:
        return False
    next_char = _next_nonspace(text, index)
    next_is_word = bool(next_char) and unicodedata.category(next_char)[0] in {
        "L",
        "M",
    }
    return next_is_word and (
        previous_word in _INDIC_HONORIFICS or (char == "." and len(previous_word) == 1)
    )


def _continues_indic_sentence(char: str) -> bool:
    return char.isspace() or char in _INDIC_SENTENCE_CONTINUATIONS


def segment_indic_text(text: str) -> List[SentenceSpan]:
    """Split Indic text on script-aware terminators with exact offsets.

    Danda and double-danda are treated as first-class sentence terminators.
    Common Indic and Latin honorifics, initials, and decimal points are guarded
    so embedded punctuation does not create a false boundary.
    """
    if not text:
        return []

    spans: List[SentenceSpan] = []
    start = 0
    boundary_ready = False

    for index, char in enumerate(text):
        if boundary_ready and not _continues_indic_sentence(char):
            if not (
                is_grapheme_boundary(start, text) and is_grapheme_boundary(index, text)
            ):
                raise ValueError("Indic sentence boundary splits a grapheme cluster")
            spans.append(SentenceSpan(text[start:index], start, index))
            start = index
            boundary_ready = False

        if char in _INDIC_TERMINATORS and not _is_guarded_terminator(text, index):
            boundary_ready = True

    if start < len(text):
        if not (
            is_grapheme_boundary(start, text) and is_grapheme_boundary(len(text), text)
        ):
            raise ValueError("Indic sentence boundary splits a grapheme cluster")
        spans.append(SentenceSpan(text[start:], start, len(text)))

    return spans


def segment_text(
    text: str,
    *,
    language: str = "en",
    clean: bool = False,
    segmenter: Optional[Any] = None,
    backend: _SentenceBackend = "auto",
) -> List[SentenceSpan]:
    """Split ``text`` into sentences and capture exact character spans.

    Indic text uses the built-in danda-aware path, while Chinese and
    Han-dominant text uses the built-in CJK-aware path. Other text retains the
    existing pySBD behavior.

    ``backend`` selects the engine: ``"auto"`` keeps that routing (default),
    and ``"yasbd"`` opts into the experimental yasbd-lib adapter for faster
    segmentation.
    See https://github.com/maziyarpanahi/openmed/issues/1848#issuecomment-5037658538
    """
    if backend not in {"auto", "yasbd"}:
        raise ValueError(
            f"Unknown segmentation backend {backend!r}. Choose from 'auto' or 'yasbd'."
        )
    if segmenter is not None and backend != "auto":
        raise ValueError(
            "A preconstructed segmenter cannot be combined with a non-'auto' backend."
        )
    if not text:
        return []

    if backend == "auto" and segmenter is None:
        if is_indic_text(text):
            return segment_indic_text(text)
        if _uses_chinese_segmenter(text, language):
            return segment_chinese_text(text)

    seg = _get_segmenter(
        language=language, clean=clean, segmenter=segmenter, backend=backend
    )
    segment_input = text
    span_offset = 0
    if backend == "yasbd":
        # YASBD 0.13.x reports offsets relative to the first non-whitespace
        # paragraph when the source begins with blank lines. Strip that prefix
        # before segmentation, then restore it to OpenMed's source offsets.
        span_offset = len(text) - len(text.lstrip())
        segment_input = text[span_offset:]
    sentences = seg.segment(segment_input)

    spans: List[SentenceSpan] = []

    if sentences and hasattr(sentences[0], "start") and hasattr(sentences[0], "end"):
        for sentence in sentences:
            sent_text = getattr(sentence, "sent", None)
            if sent_text is None:
                sent_text = segment_input[sentence.start : sentence.end]
            spans.append(
                SentenceSpan(
                    sent_text,
                    int(sentence.start) + span_offset,
                    int(sentence.end) + span_offset,
                )
            )
    else:
        spans = _fallback_spans(segment_input, sentences)
        if span_offset:
            spans = [
                SentenceSpan(
                    span.text, span.start + span_offset, span.end + span_offset
                )
                for span in spans
            ]

    if backend == "yasbd":
        normalized = _normalize_yasbd_spans(text, spans)
        return _split_yasbd_chinese_semicolons(text, normalized, language)
    return spans


__all__ = [
    "SentenceSpan",
    "segment_chinese_text",
    "segment_indic_text",
    "segment_text",
]
