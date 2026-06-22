"""Deterministic post-detector safety sweep for structured PII."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..processing.outputs import EntityPrediction
from .anonymizer.providers.clinical_ids import id_subtype_for_entity_type
from .pii_entity_merger import PII_PATTERNS, PIIPattern, find_context_words
from .quality_gates import resolve_overlapping_entities

SAFETY_SWEEP_SOURCE = "safety_sweep"
SAFETY_SWEEP_PATTERNS_VERSION = "safety-sweep-v1"


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    label: str
    text: str
    confidence: float
    priority: int
    pattern: PIIPattern


def _span_value(span: Any, key: str) -> Any:
    if isinstance(span, Mapping):
        return span.get(key)
    return getattr(span, key, None)


def _span_bounds(span: Any) -> tuple[int | None, int | None]:
    start = _span_value(span, "start")
    end = _span_value(span, "end")
    return start, end


def _overlaps(start: int, end: int, spans: Sequence[Any]) -> bool:
    for span in spans:
        span_start, span_end = _span_bounds(span)
        if span_start is None or span_end is None:
            continue
        if start < span_end and end > span_start:
            return True
    return False


def _patterns_for_language(lang: str) -> list[PIIPattern]:
    if lang == "en":
        return list(PII_PATTERNS)

    from .pii_i18n import get_patterns_for_language

    return list(get_patterns_for_language(lang))


def _validated(pattern: PIIPattern, text: str) -> bool:
    if pattern.validator is None:
        return True
    try:
        return bool(pattern.validator(text))
    except Exception:
        return False


def _confidence(text: str, start: int, end: int, pattern: PIIPattern) -> float:
    score = pattern.base_score
    if pattern.context_words and find_context_words(
        text,
        start,
        end,
        pattern.context_words,
    ):
        score = min(1.0, score + pattern.context_boost)
    return score


def _candidate_metadata(candidate: _Candidate) -> dict[str, Any]:
    id_subtype = id_subtype_for_entity_type(candidate.label)
    safety_sweep_metadata: dict[str, Any] = {
        "entity_type": candidate.label,
        "confidence": candidate.confidence,
        "pattern": candidate.pattern.pattern,
    }
    if id_subtype is not None:
        safety_sweep_metadata["id_subtype"] = id_subtype
    return {
        "source": SAFETY_SWEEP_SOURCE,
        "patterns_version": SAFETY_SWEEP_PATTERNS_VERSION,
        "safety_sweep": safety_sweep_metadata,
        **({"id_subtype": id_subtype} if id_subtype is not None else {}),
    }


def _collect_candidates(text: str, patterns: Sequence[PIIPattern]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for pattern in patterns:
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            start, end = match.span()
            if start >= end:
                continue

            matched_text = text[start:end]
            if not _validated(pattern, matched_text):
                continue

            candidates.append(
                _Candidate(
                    start=start,
                    end=end,
                    label=pattern.entity_type,
                    text=matched_text,
                    confidence=_confidence(text, start, end, pattern),
                    priority=pattern.priority,
                    pattern=pattern,
                )
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate.confidence,
            -candidate.priority,
            candidate.start,
            -(candidate.end - candidate.start),
        )
    )
    return candidates


def safety_sweep(
    text: str,
    spans: Sequence[Any],
    *,
    lang: str = "en",
    patterns: Sequence[PIIPattern] | None = None,
) -> list[Any]:
    """Add deterministic structured identifier spans not covered by ML spans.

    Existing spans always win. Sweep candidates that overlap any supplied span,
    or any higher-ranked sweep candidate, are discarded. Existing overlapping
    spans are resolved by the quality-gate policy so callers receive a
    non-overlapping span set.
    """
    existing = list(spans)
    selected: list[_Candidate] = []
    active_spans: list[Any] = list(existing)
    sweep_patterns = (
        list(patterns) if patterns is not None else _patterns_for_language(lang)
    )

    for candidate in _collect_candidates(text, sweep_patterns):
        if _overlaps(candidate.start, candidate.end, active_spans):
            continue

        entity = EntityPrediction(
            text=candidate.text,
            label=candidate.label,
            start=candidate.start,
            end=candidate.end,
            confidence=candidate.confidence,
            metadata=_candidate_metadata(candidate),
        )
        selected.append(candidate)
        active_spans.append(entity)
        existing.append(entity)

    ordered = sorted(
        existing,
        key=lambda span: (
            _span_value(span, "start") is None,
            _span_value(span, "start") or 0,
            _span_value(span, "end") or 0,
        ),
    )
    return resolve_overlapping_entities(ordered)


__all__ = [
    "SAFETY_SWEEP_PATTERNS_VERSION",
    "SAFETY_SWEEP_SOURCE",
    "safety_sweep",
]
