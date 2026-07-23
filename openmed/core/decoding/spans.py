"""Span post-processing helpers shared across privacy-filter backends.

When token classifiers emit slightly-too-greedy spans (e.g. "alice@hospital.org and"
absorbs the trailing "and"), these helpers tighten the boundaries before the
span reaches downstream redaction logic. Pure-Python; no array-framework
dependencies.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Final

from ..labels import supports_name_boundary_refinement
from ..script_detect import (
    CJK_SCRIPTS,
    INDIC_SCRIPTS,
    UNKNOWN_SCRIPT,
    segment_by_script,
)

_GRAPHEME_CONTROL_CATEGORIES: Final = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_INDIC_LINKERS: Final = frozenset(
    {
        "\u094d",  # Devanagari sign virama
        "\u09cd",  # Bengali sign virama
        "\u0a4d",  # Gurmukhi sign virama
        "\u0acd",  # Gujarati sign virama
        "\u0b4d",  # Odia sign virama
        "\u0bcd",  # Tamil sign virama
        "\u0c4d",  # Telugu sign virama
        "\u0ccd",  # Kannada sign virama
        "\u0d4d",  # Malayalam sign virama
    }
)
_INDIC_CONSONANT_RANGES: Final = (
    (0x0915, 0x0939),
    (0x0958, 0x095F),
    (0x0978, 0x097F),
    (0x0995, 0x09B9),
    (0x09DC, 0x09DF),
    (0x09F0, 0x09F1),
    (0x0A15, 0x0A39),
    (0x0A59, 0x0A5E),
    (0x0A95, 0x0AB9),
    (0x0B15, 0x0B39),
    (0x0B5C, 0x0B5D),
    (0x0B5F, 0x0B5F),
    (0x0B95, 0x0BB9),
    (0x0C15, 0x0C39),
    (0x0C58, 0x0C5A),
    (0x0C95, 0x0CB9),
    (0x0CDC, 0x0CDE),
    (0x0D15, 0x0D3A),
)
_SCRIPT_REFINEMENT_GUARDS: Final = CJK_SCRIPTS | INDIC_SCRIPTS
_PREPEND_RANGES: Final = (
    (0x0600, 0x0605),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x0890, 0x0891),
    (0x08E2, 0x08E2),
    (0x0D4E, 0x0D4E),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x111C2, 0x111C3),
    (0x1193F, 0x1193F),
    (0x11941, 0x11941),
    (0x11A3A, 0x11A3A),
    (0x11A84, 0x11A89),
    (0x11D46, 0x11D46),
    (0x11F02, 0x11F02),
)


def iter_grapheme_cluster_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield UAX #29-style extended grapheme-cluster offsets for ``text``.

    The implementation is stdlib-only and preserves Python code-point offsets.
    It covers combining and spacing marks, Hangul syllables, regional-indicator
    pairs, emoji modifiers and ZWJ sequences, and Indic virama conjuncts.

    Args:
        text: Original, unnormalized Unicode text.

    Yields:
        ``(start, end)`` code-point offsets for each whole grapheme cluster.
    """
    if not text:
        return

    cluster_start = 0
    for index in range(1, len(text)):
        if _has_grapheme_break(text, index):
            yield cluster_start, index
            cluster_start = index
    yield cluster_start, len(text)


def snap_span_to_grapheme_boundaries(
    start: int,
    end: int,
    text: str,
) -> tuple[int, int]:
    """Clamp a span and snap its non-empty boundaries outward by cluster.

    Empty spans remain empty and are moved to the preceding cluster boundary.
    Returned offsets always index the original ``text`` directly.
    """
    text_length = len(text)
    safe_start = max(0, min(int(start), text_length))
    safe_end = max(safe_start, min(int(end), text_length))
    snapped_start = safe_start
    while 0 < snapped_start < text_length and not _has_grapheme_break(
        text, snapped_start
    ):
        snapped_start -= 1
    if safe_start == safe_end:
        return snapped_start, snapped_start

    snapped_end = safe_end
    while snapped_end < text_length and not _has_grapheme_break(text, snapped_end):
        snapped_end += 1
    return snapped_start, snapped_end


def iter_grapheme_clusters(text: str) -> Iterator[tuple[int, int]]:
    """Yield extended grapheme-cluster offsets for ``text``.

    Args:
        text: Original, unnormalized Unicode text.

    Yields:
        Half-open ``(start, end)`` code-point offsets for each whole cluster.
    """

    yield from iter_grapheme_cluster_spans(text)


def is_grapheme_boundary(index: int, text: str) -> bool:
    """Return whether ``index`` is a grapheme boundary in ``text``.

    Args:
        index: Candidate Python code-point offset.
        text: Original, unnormalized Unicode text.

    Returns:
        ``True`` when ``index`` is the start or end of a whole cluster.
    """

    if index < 0 or index > len(text):
        return False
    if index in {0, len(text)}:
        return True
    return any(end == index for _, end in iter_grapheme_cluster_spans(text))


def is_indic_text(text: str) -> bool:
    """Return whether ``text`` contains a supported Indic script run.

    Args:
        text: Text to inspect without normalization.

    Returns:
        ``True`` when the script detector finds one of the supported Indic
        scripts.
    """

    return any(script in INDIC_SCRIPTS for _, _, script in segment_by_script(text))


def snap_span_to_graphemes(start: int, end: int, text: str) -> tuple[int, int]:
    """Snap a span outward using the canonical grapheme-boundary engine.

    Args:
        start: Inclusive Python code-point offset.
        end: Exclusive Python code-point offset.
        text: Original text referenced by the offsets.

    Returns:
        Clamped ``(start, end)`` offsets that do not bisect a cluster.
    """

    return snap_span_to_grapheme_boundaries(start, end, text)


def trim_span_whitespace(start: int, end: int, text: str) -> tuple[int, int]:
    """Strip whole whitespace clusters from ``text[start:end]``.

    Input boundaries are first snapped outward, so the returned ``[start, end)``
    offsets never bisect a combining sequence, Indic aksara, or emoji sequence.
    """
    start, end = snap_span_to_grapheme_boundaries(start, end, text)
    if start == end:
        return start, end

    clusters = list(iter_grapheme_cluster_spans(text))
    selected = [
        cluster for cluster in clusters if cluster[0] >= start and cluster[1] <= end
    ]

    while selected and _cluster_is_whitespace(text[slice(*selected[0])]):
        start = selected.pop(0)[1]
    while selected and _cluster_is_whitespace(text[slice(*selected[-1])]):
        end = selected.pop()[0]
    return start, end


def remap_normalized_span(
    start: int,
    end: int,
    original_text: str,
    normalization: Any,
) -> tuple[int, int, str]:
    """Project a decoded normalized span onto the original source text.

    ``normalization`` follows the :class:`DetectionNormalization` contract and
    is intentionally duck-typed to keep the backend-agnostic decoder free of a
    dependency on the higher-level PII module.
    """

    original_start, original_end = normalization.remap_span(start, end)
    return (
        original_start,
        original_end,
        original_text[original_start:original_end],
    )


_PRIVACY_FILTER_SPAN_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("url", re.compile(r"\b(?:https?://|www\.)[^\s,;)\]]+")),
    ("phone", re.compile(r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")),
)


@dataclass(frozen=True)
class IndicSpanRefinement:
    """An optional name-boundary refinement with absolute source offsets."""

    original_start: int
    original_end: int
    start: int
    end: int
    applied: bool
    reason: str
    offset_map: tuple[tuple[int, int], ...]
    grapheme_origins: tuple[tuple[int, int], ...]
    rule: str | None = None

    def to_source_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a span relative to the refined text back to source offsets."""

        if not (0 <= start <= end <= len(self.offset_map)):
            raise ValueError("span must fit within the refined output")
        if start == end:
            if start < len(self.offset_map):
                anchor = self.offset_map[start][0]
            elif self.offset_map:
                anchor = self.offset_map[-1][1]
            else:
                anchor = self.start
            return anchor, anchor
        return self.offset_map[start][0], self.offset_map[end - 1][1]


def refine_indic_name_span(
    label: str,
    start: int,
    end: int,
    text: str,
    *,
    enabled: bool = False,
    language: str | None = None,
    confidence: float = 0.0,
    allowed_stems: Iterable[str] = (),
    minimum_confidence: float = 0.9,
    minimum_stem_graphemes: int = 2,
) -> IndicSpanRefinement:
    """Optionally tighten an Indic person-name span to an allow-listed stem.

    The disabled path returns the input offsets exactly. When enabled, the
    span changes only if its canonical label is name-like, the confidence and
    exact allow-list gates pass, and the source and proposed boundaries are
    whole grapheme boundaries.
    """

    if not (0 <= start <= end <= len(text)):
        raise ValueError("span must satisfy 0 <= start <= end <= len(text)")

    unchanged = _indic_refinement_result(
        text,
        original_start=start,
        original_end=end,
        start=start,
        end=end,
        applied=False,
        reason="disabled" if not enabled else "guard_rejected",
    )
    if not enabled:
        return unchanged
    if language is None or not supports_name_boundary_refinement(label):
        return unchanged

    # Imported lazily to keep decoding helpers independent while the processing
    # package initializes.
    from openmed.processing.morphology import stem_token

    source_graphemes = tuple(iter_grapheme_cluster_spans(text))
    source_boundaries = {0, *(cluster_end for _, cluster_end in source_graphemes)}
    if start not in source_boundaries or end not in source_boundaries:
        return _indic_refinement_result(
            text,
            original_start=start,
            original_end=end,
            start=start,
            end=end,
            applied=False,
            reason="unaligned_source_span",
        )

    analysis = stem_token(
        text[start:end],
        language,
        confidence=confidence,
        allowed_stems=allowed_stems,
        minimum_confidence=minimum_confidence,
        minimum_stem_graphemes=minimum_stem_graphemes,
    )
    if not analysis.applied:
        return unchanged

    local_start, local_end = analysis.stem_span
    refined_start = start + local_start
    refined_end = start + local_end
    mapped = analysis.offset_map.to_source_span(0, len(analysis.stem))
    if (
        local_start != 0
        or mapped != analysis.stem_span
        or refined_start not in source_boundaries
        or refined_end not in source_boundaries
        or text[refined_start:refined_end] != analysis.stem
    ):
        return _indic_refinement_result(
            text,
            original_start=start,
            original_end=end,
            start=start,
            end=end,
            applied=False,
            reason="offset_safety_rejected",
        )

    return _indic_refinement_result(
        text,
        original_start=start,
        original_end=end,
        start=refined_start,
        end=refined_end,
        applied=True,
        reason="allowlisted_suffix",
        rule=analysis.rule,
    )


def _indic_refinement_result(
    text: str,
    *,
    original_start: int,
    original_end: int,
    start: int,
    end: int,
    applied: bool,
    reason: str,
    rule: str | None = None,
) -> IndicSpanRefinement:
    return IndicSpanRefinement(
        original_start=original_start,
        original_end=original_end,
        start=start,
        end=end,
        applied=applied,
        reason=reason,
        offset_map=tuple((index, index + 1) for index in range(start, end)),
        grapheme_origins=tuple(
            (cluster_start, cluster_end)
            for cluster_start, cluster_end in iter_grapheme_cluster_spans(text)
            if start <= cluster_start and cluster_end <= end
        ),
        rule=rule,
    )


def refine_privacy_filter_span(
    label: str,
    start: int,
    end: int,
    text: str,
    *,
    indic_morphology: bool = False,
    language: str | None = None,
    confidence: float = 0.0,
    morphology_allowlist: Iterable[str] = (),
    minimum_morphology_confidence: float = 0.9,
) -> tuple[int, int]:
    """Tighten a PII span without crossing grapheme or script boundaries.

    For email / URL / phone labels, locate the strict regex match inside
    Latin or neutral script runs and shrink to it. The Latin-only trailing
    ``" and"`` / ``" or"`` heuristic is disabled for spans containing CJK
    or other scripts. Every returned boundary is snapped to a whole grapheme.
    """
    start, end = trim_span_whitespace(start, end, text)
    span_text = text[start:end]
    normalized = label.lower()
    script_runs = list(segment_by_script(span_text))

    for label_hint, pattern in _PRIVACY_FILTER_SPAN_PATTERNS:
        if label_hint not in normalized:
            continue
        match_offsets = _find_structured_match(span_text, pattern, script_runs)
        if match_offsets is not None:
            match_start, match_end = match_offsets
            return trim_span_whitespace(
                start + match_start,
                start + match_end,
                text,
            )

    scripts = {script for _, _, script in script_runs}
    if scripts <= {"Latin", UNKNOWN_SCRIPT} or scripts & INDIC_SCRIPTS:
        for suffix in (" and", " or"):
            if span_text.lower().endswith(suffix):
                end -= len(suffix)
                break
    start, end = trim_span_whitespace(start, end, text)
    if indic_morphology:
        refinement = refine_indic_name_span(
            label,
            start,
            end,
            text,
            enabled=True,
            language=language,
            confidence=confidence,
            allowed_stems=morphology_allowlist,
            minimum_confidence=minimum_morphology_confidence,
        )
        return refinement.start, refinement.end
    return start, end


def _find_structured_match(
    span_text: str,
    pattern: re.Pattern[str],
    script_runs: list[tuple[int, int, str]],
) -> tuple[int, int] | None:
    guarded = any(script in _SCRIPT_REFINEMENT_GUARDS for _, _, script in script_runs)
    if not guarded:
        match = pattern.search(span_text)
        if match is not None:
            return match.start(), match.end()
        return None

    # A leading CJK/Indic code point defeats the regex ``\b`` before an
    # adjacent Latin email or URL. Searching the Latin run independently also
    # prevents a URL path from greedily absorbing an adjacent ideograph.
    for run_start, run_end, script in script_runs:
        if script != "Latin":
            continue
        run_match = pattern.search(span_text[run_start:run_end])
        if run_match is not None:
            return run_start + run_match.start(), run_start + run_match.end()

    # Phone spans can be neutral-only digits attached to a surrounding script
    # run. Retain that legacy match only when the matched text itself has no
    # CJK/Indic code points.
    match = pattern.search(span_text)
    if match is not None:
        match_scripts = {script for _, _, script in segment_by_script(match.group(0))}
        if not (match_scripts & _SCRIPT_REFINEMENT_GUARDS):
            return match.start(), match.end()
    return None


def _cluster_is_whitespace(cluster: str) -> bool:
    saw_whitespace = False
    for char in cluster:
        if char.isspace():
            saw_whitespace = True
            continue
        if _grapheme_break_class(char) in {"EXTEND", "ZWJ"}:
            continue
        return False
    return saw_whitespace


def _has_grapheme_break(text: str, index: int) -> bool:
    previous = text[index - 1]
    current = text[index]
    previous_class = _grapheme_break_class(previous)
    current_class = _grapheme_break_class(current)

    if previous_class == "CR" and current_class == "LF":
        return False
    if previous_class in {"CR", "LF", "CONTROL"}:
        return True
    if current_class in {"CR", "LF", "CONTROL"}:
        return True
    if previous_class == "L" and current_class in {"L", "V", "LV", "LVT"}:
        return False
    if previous_class in {"LV", "V"} and current_class in {"V", "T"}:
        return False
    if previous_class in {"LVT", "T"} and current_class == "T":
        return False
    if current_class in {"EXTEND", "ZWJ", "SPACING_MARK"}:
        return False
    if previous_class == "PREPEND":
        return False
    if _continues_indic_conjunct(text, index):
        return False
    if (
        current_class == "EXTENDED_PICTOGRAPHIC"
        and previous_class == "ZWJ"
        and _extended_pictographic_before_zwj(text, index - 1)
    ):
        return False
    if previous_class == current_class == "REGIONAL_INDICATOR":
        preceding_indicators = 0
        cursor = index - 1
        while cursor >= 0 and _grapheme_break_class(text[cursor]) == (
            "REGIONAL_INDICATOR"
        ):
            preceding_indicators += 1
            cursor -= 1
        return preceding_indicators % 2 == 0
    return True


def _grapheme_break_class(char: str) -> str:
    codepoint = ord(char)
    if char == "\r":
        return "CR"
    if char == "\n":
        return "LF"
    if char == "\u200d":
        return "ZWJ"
    hangul_class = _hangul_grapheme_class(codepoint)
    if hangul_class is not None:
        return hangul_class
    if 0x1F1E6 <= codepoint <= 0x1F1FF:
        return "REGIONAL_INDICATOR"
    if _in_ranges(codepoint, _PREPEND_RANGES):
        return "PREPEND"
    if _is_grapheme_extend(char):
        return "EXTEND"
    if unicodedata.category(char) == "Mc":
        return "SPACING_MARK"
    if unicodedata.category(char) in _GRAPHEME_CONTROL_CATEGORIES:
        return "CONTROL"
    if _is_extended_pictographic(codepoint):
        return "EXTENDED_PICTOGRAPHIC"
    return "OTHER"


def _hangul_grapheme_class(codepoint: int) -> str | None:
    if 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97C:
        return "L"
    if 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6:
        return "V"
    if 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB:
        return "T"
    if 0xAC00 <= codepoint <= 0xD7A3:
        return "LV" if (codepoint - 0xAC00) % 28 == 0 else "LVT"
    return None


def _is_grapheme_extend(char: str) -> bool:
    codepoint = ord(char)
    return (
        unicodedata.category(char) in {"Mn", "Me"}
        or char == "\u200c"
        or 0x1F3FB <= codepoint <= 0x1F3FF
        or 0xE0020 <= codepoint <= 0xE007F
    )


def _continues_indic_conjunct(text: str, index: int) -> bool:
    if not _is_indic_consonant(text[index]):
        return False

    saw_linker = False
    cursor = index - 1
    while cursor >= 0:
        char = text[cursor]
        if char in _INDIC_LINKERS:
            saw_linker = True
            cursor -= 1
            continue
        if _grapheme_break_class(char) in {"EXTEND", "ZWJ", "SPACING_MARK"}:
            cursor -= 1
            continue
        return saw_linker and _is_indic_consonant(char)
    return False


def _is_indic_consonant(char: str) -> bool:
    return _in_ranges(ord(char), _INDIC_CONSONANT_RANGES)


def _extended_pictographic_before_zwj(text: str, zwj_index: int) -> bool:
    cursor = zwj_index - 1
    while cursor >= 0 and _grapheme_break_class(text[cursor]) == "EXTEND":
        cursor -= 1
    return cursor >= 0 and _grapheme_break_class(text[cursor]) == (
        "EXTENDED_PICTOGRAPHIC"
    )


def _is_extended_pictographic(codepoint: int) -> bool:
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2300 <= codepoint <= 0x23FF
        or 0x2600 <= codepoint <= 0x27BF
    )


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


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
    "IndicSpanRefinement",
    "TokenClassificationSpan",
    "TokenClassificationStreamEvent",
    "coerce_token_classification_spans",
    "iter_grapheme_cluster_spans",
    "reconcile_stream_spans",
    "refine_indic_name_span",
    "remap_normalized_span",
    "refine_privacy_filter_span",
    "stable_span_id",
    "stable_span_key",
    "snap_span_to_grapheme_boundaries",
    "trim_span_whitespace",
]
