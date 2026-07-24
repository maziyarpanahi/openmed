"""Deterministic post-detector safety sweep for structured PII."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..processing.outputs import EntityPrediction
from .anonymizer.providers.clinical_ids import id_subtype_for_entity_type
from .pii_entity_merger import (
    PII_PATTERNS,
    PIIPattern,
    find_context_words,
    iter_pattern_spans,
)
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


def _patterns_for_language(lang: str, locale: str | None = None) -> list[PIIPattern]:
    if lang == "en" and locale is None:
        from .pii_i18n import AADHAAR_PII_PATTERNS

        return [*PII_PATTERNS, *AADHAAR_PII_PATTERNS]

    from .pii_i18n import get_patterns_for_language

    return list(get_patterns_for_language(lang, locale=locale))


def _validated(pattern: PIIPattern, text: str) -> bool:
    if pattern.validator is None:
        return True
    try:
        return bool(pattern.validator(text))
    except Exception:
        return False


def _has_context(
    text: str,
    start: int,
    end: int,
    pattern: PIIPattern,
    *,
    require_boundaries: bool = False,
) -> bool:
    return bool(
        pattern.context_words
        and find_context_words(
            text,
            start,
            end,
            pattern.context_words,
            require_boundaries=require_boundaries,
        )
    )


def _confidence(text: str, start: int, end: int, pattern: PIIPattern) -> float:
    score = pattern.base_score
    if _has_context(text, start, end, pattern):
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
        for start, end in iter_pattern_spans(text, pattern):
            matched_text = text[start:end]
            if not _validated(pattern, matched_text):
                continue
            if pattern.requires_context and not _has_context(
                text,
                start,
                end,
                pattern,
                require_boundaries=True,
            ):
                continue
            if (
                pattern.context_required or pattern.safety_sweep_requires_context
            ) and not _has_context(
                text,
                start,
                end,
                pattern,
            ):
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
    locale: str | None = None,
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
        list(patterns)
        if patterns is not None
        else _patterns_for_language(lang, locale=locale)
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


def safety_sweep_code_mixed(
    text: str,
    spans: Sequence[Any],
    *,
    token_language_tags: Sequence[Any],
    base_lang: str = "en",
    locale: str | None = None,
) -> list[Any]:
    """Sweep merged model/rule spans with explicitly gated Hinglish patterns."""
    from .pii_i18n import get_patterns_for_code_mixed_tags

    patterns = get_patterns_for_code_mixed_tags(
        text,
        token_language_tags,
        base_lang=base_lang,
        locale=locale,
    )
    return safety_sweep(
        text,
        spans,
        lang=base_lang,
        locale=locale,
        patterns=patterns,
    )


def hashed_span_surface(
    text: str,
    start: int,
    end: int,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Return offset-keyed, raw-text-free evidence for a sensitive span."""
    bounded_start = max(0, min(int(start), len(text)))
    bounded_end = max(bounded_start, min(int(end), len(text)))
    surface = text[bounded_start:bounded_end]
    payload: dict[str, Any] = {
        "start": bounded_start,
        "end": bounded_end,
        "length": len(surface),
        "text_hash": _hash_surface(surface),
    }
    if label is not None:
        payload["label"] = str(label)
    return payload


def _hash_surface(surface: str) -> str:
    return f"sha256:{hashlib.sha256(surface.encode('utf-8')).hexdigest()}"


__all__ = [
    "SAFETY_SWEEP_PATTERNS_VERSION",
    "SAFETY_SWEEP_SOURCE",
    "hashed_span_surface",
    "safety_sweep",
    "safety_sweep_code_mixed",
]
