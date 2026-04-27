"""Span post-processing helpers shared across privacy-filter backends.

When token classifiers emit slightly-too-greedy spans (e.g. "alice@hospital.org and"
absorbs the trailing "and"), these helpers tighten the boundaries before the
span reaches downstream redaction logic. Pure-Python; no array-framework
dependencies.
"""

from __future__ import annotations

import re
from typing import Final


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


__all__ = ["trim_span_whitespace", "refine_privacy_filter_span"]
