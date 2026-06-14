"""ConText engine: uncertainty axis for clinical span modification.

resolve_uncertainty() classifies a clinical span as certain or uncertain based
on hedging, hypothetical, and conditional language from the ConText lexicon.
Uncertain spans are *flagged* (not dropped) so grounding layers can downweight
or annotate unconfirmed conditions.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence
import re

Certainty = Literal["certain", "uncertain"]

# ConText hypothetical/uncertainty trigger lexicon.
# Multi-word phrases are listed before single-word cues; order is irrelevant
# for detection (any match returns uncertain) but aids readability.
_UNCERTAINTY_CUES: tuple[str, ...] = (
    "cannot exclude",
    "can't exclude",
    "concern for",
    "concerning for",
    "suspicious for",
    "suspicion for",
    "worrisome for",
    "question of",
    "rule out",
    "to rule out",
    "versus",
    "probable",
    "probably",
    "possible",
    "possibly",
    "suspected",
    "suspect",
    "unlikely",
    "likely",
    "might",
    "could",
    "may",
    "r/o",
    "vs",
    "if",
)

_CUE_RE: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in _UNCERTAINTY_CUES) + r")\b",
    re.IGNORECASE,
)


def _span_text(span: Any) -> str:
    if isinstance(span, str):
        return span
    return str(getattr(span, "text", span))


def resolve_uncertainty(
    span: Any,
    modifier_hits: Sequence[str],
) -> Certainty:
    """Classify a clinical span as certain or uncertain/hypothetical.

    Checks both *modifier_hits* (ConText trigger words from surrounding context)
    and the span text itself for hedging, hypothetical, or conditional cues.

    Uncertain spans are flagged, not dropped; grounding layers may downweight
    or annotate them accordingly.

    Args:
        span: The clinical span — a plain string or any object with a ``text``
            attribute.
        modifier_hits: Modifier strings identified by the ConText engine for
            this span (may be empty).

    Returns:
        ``"uncertain"`` when any hedging/hypothetical/conditional cue is
        detected; ``"certain"`` otherwise.
    """
    for hit in modifier_hits:
        if _CUE_RE.search(hit):
            return "uncertain"

    if _CUE_RE.search(_span_text(span)):
        return "uncertain"

    return "certain"


__all__ = [
    "Certainty",
    "resolve_uncertainty",
]
