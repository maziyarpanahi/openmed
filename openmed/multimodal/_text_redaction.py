"""Shared detector and replacement helpers for text-backed ingesters."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import ExtractedDocument

TextReplacement = tuple[int, int, str]


def detect_replacements(
    document: ExtractedDocument,
    models: Any,
    lang: str | None,
    policy: Any,
) -> tuple[TextReplacement, ...]:
    """Run a supplied detector and normalize its entities into replacements."""
    detector = _resolve_detector(models)
    if detector is None:
        return ()
    detected = _call_detector(detector, document.text, lang)
    default_replacement = policy_value(policy, "replacement")
    replacements = tuple(
        replacement
        for entity in _iter_entity_inputs(detected)
        if (
            replacement := _coerce_entity(
                entity,
                default_replacement=default_replacement,
            )
        )
        is not None
    )
    return validate_replacements(document, replacements)


def validate_replacements(
    document: ExtractedDocument,
    replacements: Iterable[TextReplacement],
) -> tuple[TextReplacement, ...]:
    """Return sorted, de-duplicated, non-overlapping logical replacements."""
    unique = {
        (int(start), int(end), str(replacement))
        for start, end, replacement in replacements
    }
    ordered = tuple(sorted(unique, key=lambda item: (item[0], item[1], item[2])))
    cursor = 0
    for start, end, _ in ordered:
        if start < 0 or end > len(document.text) or start >= end:
            raise ValueError("replacement range is outside extracted document text")
        if start < cursor:
            raise ValueError("replacement ranges overlap")
        cursor = end
    return ordered


def policy_value(policy: Any, *names: str) -> Any:
    """Read the first configured policy field from a mapping or object."""
    if policy is None:
        return None
    if isinstance(policy, Mapping):
        for name in names:
            if name in policy:
                return policy[name]
        return None
    for name in names:
        value = getattr(policy, name, None)
        if value is not None:
            return value
    return None


def validate_distinct_paths(source: Path, output: Path) -> None:
    """Reject in-place writes and existing aliases of the source file."""
    if source.resolve() == output.resolve():
        raise ValueError("output path must differ from source path")
    if output.exists() and os.path.samefile(source, output):
        raise ValueError("output path must not alias source path")


def _call_detector(detector: Any, text: str, lang: str | None) -> Any:
    try:
        signature = inspect.signature(detector)
    except (TypeError, ValueError):
        call_shape = "text_only"
    else:
        parameters = signature.parameters
        lang_parameter = parameters.get("lang")
        if lang_parameter is not None and lang_parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            signature.bind(text, lang)
            call_shape = "positional_lang"
        elif lang_parameter is not None and lang_parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            signature.bind(text, lang=lang)
            call_shape = "keyword_lang"
        elif any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            signature.bind(text, lang=lang)
            call_shape = "keyword_lang"
        else:
            signature.bind(text)
            call_shape = "text_only"
    try:
        if call_shape == "positional_lang":
            return detector(text, lang)
        if call_shape == "keyword_lang":
            return detector(text, lang=lang)
        return detector(text)
    except Exception:
        raise RuntimeError("text document detector failed") from None


def _resolve_detector(models: Any) -> Any:
    if models is None:
        return None
    if callable(models):
        return models
    if isinstance(models, Mapping):
        for key in ("detector", "extract_pii", "analyze_text", "predict_entities"):
            candidate = models.get(key)
            if callable(candidate):
                return candidate
        return None
    for name in (
        "detect",
        "extract_pii",
        "analyze_text",
        "predict_entities",
        "predict",
    ):
        candidate = getattr(models, name, None)
        if callable(candidate):
            return candidate
    return None


def _iter_entity_inputs(spans: Any) -> tuple[Any, ...]:
    if spans is None:
        return ()
    for name in ("entities", "pii_entities"):
        entities = getattr(spans, name, None)
        if entities is not None:
            return tuple(entities)
    if isinstance(spans, Mapping):
        for key in ("entities", "pii_entities", "spans"):
            entities = spans.get(key)
            if entities is not None:
                return tuple(entities)
        if "start" in spans and "end" in spans:
            return (spans,)
    if _looks_like_sequence_entity(spans):
        return (spans,)
    if isinstance(spans, Iterable) and not isinstance(spans, (str, bytes, bytearray)):
        return tuple(spans)
    return (spans,)


def _coerce_entity(
    span: Any,
    *,
    default_replacement: str | None,
) -> TextReplacement | None:
    if isinstance(span, Sequence) and not isinstance(span, (str, bytes, bytearray)):
        if len(span) < 2:
            return None
        label = str(span[2]) if len(span) >= 3 and span[2] is not None else None
        return (
            _coerce_offset(span[0]),
            _coerce_offset(span[1]),
            default_replacement or _mask_for_label(label),
        )
    if isinstance(span, Mapping):
        start = span.get("start")
        end = span.get("end")
        label = span.get("label", span.get("entity_type", span.get("entity_group")))
        replacement = span.get("replacement", span.get("redacted_text"))
    else:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        label = getattr(
            span,
            "label",
            getattr(span, "entity_type", getattr(span, "entity_group", None)),
        )
        replacement = getattr(span, "replacement", getattr(span, "redacted_text", None))
    if start is None or end is None:
        return None
    return (
        _coerce_offset(start),
        _coerce_offset(end),
        str(replacement)
        if replacement is not None
        else default_replacement or _mask_for_label(label),
    )


def _coerce_offset(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("invalid detector entity offsets") from None


def _looks_like_sequence_entity(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    if len(value) < 2:
        return False
    return all(
        not isinstance(candidate, Mapping)
        and not all(hasattr(candidate, name) for name in ("start", "end"))
        and not (
            isinstance(candidate, Sequence)
            and not isinstance(candidate, (str, bytes, bytearray))
        )
        for candidate in value[:2]
    )


def _mask_for_label(label: Any) -> str:
    safe = "".join(
        character if character.isalnum() else "_"
        for character in str(label or "PHI").upper()
    ).strip("_")
    return f"[{safe or 'PHI'}]"


__all__ = [
    "TextReplacement",
    "detect_replacements",
    "policy_value",
    "validate_distinct_paths",
    "validate_replacements",
]
