"""Deterministic, rules-first clinical document-type classification."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TypedDict

DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE = "data/doctype_signatures.json"
UNKNOWN_DOCUMENT_TYPE = "unknown"
_TOKEN_RE = re.compile(r"\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}-]\w+)*", re.UNICODE)


class DocumentClassification(TypedDict):
    """Public document-type prediction with an abstention-safe confidence."""

    type: str
    confidence: float


@dataclass(frozen=True)
class _SignatureRule:
    phrases: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class _DocumentTypeRules:
    document_type: str
    rules: tuple[_SignatureRule, ...]


@dataclass(frozen=True)
class _SignatureTable:
    max_tokens: int
    ambiguity_margin: float
    unknown_confidence: float
    document_types: tuple[_DocumentTypeRules, ...]


def classify_document(text: str) -> DocumentClassification:
    """Classify a clinical note from deterministic first-window signatures.

    The bundled signatures cover discharge summaries, progress notes,
    radiology reports, pathology reports, operative notes, and consult notes.
    Classification is local and deterministic. When no signature wins clearly,
    the function returns ``unknown`` rather than guessing.

    Args:
        text: Clinical note text. Only the first configured token window is
            inspected.

    Returns:
        A mapping with ``type`` and a rule-strength ``confidence`` from zero to
        one. Ambiguous and unrecognized notes use the ``unknown`` type.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    table = _load_signature_table()
    window = _normalized_token_window(text, max_tokens=table.max_tokens)
    if not window:
        return _unknown_classification(table)

    matches: list[tuple[float, str]] = []
    for document_type in table.document_types:
        matched_confidences = [
            rule.confidence
            for rule in document_type.rules
            if all(_contains_phrase(window, phrase) for phrase in rule.phrases)
        ]
        if matched_confidences:
            matches.append((max(matched_confidences), document_type.document_type))

    if not matches:
        return _unknown_classification(table)

    matches.sort(key=lambda item: (-item[0], item[1]))
    best_confidence, best_type = matches[0]
    if len(matches) > 1:
        runner_up_confidence = matches[1][0]
        if best_confidence - runner_up_confidence <= table.ambiguity_margin:
            return _unknown_classification(table)

    return {
        "type": best_type,
        "confidence": round(best_confidence, 6),
    }


def _normalized_token_window(text: str, *, max_tokens: int) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = _TOKEN_RE.findall(normalized)
    return " ".join(tokens[:max_tokens])


def _contains_phrase(window: str, phrase: str) -> bool:
    return f" {phrase} " in f" {window} "


def _unknown_classification(table: _SignatureTable) -> DocumentClassification:
    return {
        "type": UNKNOWN_DOCUMENT_TYPE,
        "confidence": round(table.unknown_confidence, 6),
    }


@lru_cache(maxsize=1)
def _load_signature_table() -> _SignatureTable:
    resource = resources.files("openmed.clinical").joinpath(
        DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE
    )
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _validate_signature_table(payload)


def _validate_signature_table(payload: object) -> _SignatureTable:
    if not isinstance(payload, Mapping):
        raise ValueError("document type signature table must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("document type signature schema_version must be 1")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("document type signatures require provenance metadata")
    if provenance.get("restricted_data") is not False:
        raise ValueError("document type signatures must not use restricted data")
    if provenance.get("synthetic") is not True:
        raise ValueError("document type signatures must declare synthetic=true")

    max_tokens = payload.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 1
    ):
        raise ValueError("document type max_tokens must be a positive integer")
    ambiguity_margin = _probability(
        payload.get("ambiguity_margin"), field="ambiguity_margin", maximum=0.99
    )
    unknown_confidence = _probability(
        payload.get("unknown_confidence"), field="unknown_confidence"
    )

    raw_document_types = payload.get("document_types")
    if not isinstance(raw_document_types, list) or len(raw_document_types) < 6:
        raise ValueError("document type signatures must cover at least six types")

    document_types: list[_DocumentTypeRules] = []
    seen_types: set[str] = set()
    for raw_document_type in raw_document_types:
        document_type = _validate_document_type(raw_document_type)
        if document_type.document_type in seen_types:
            raise ValueError(f"duplicate document type {document_type.document_type!r}")
        seen_types.add(document_type.document_type)
        document_types.append(document_type)

    return _SignatureTable(
        max_tokens=max_tokens,
        ambiguity_margin=ambiguity_margin,
        unknown_confidence=unknown_confidence,
        document_types=tuple(document_types),
    )


def _validate_document_type(raw: object) -> _DocumentTypeRules:
    if not isinstance(raw, Mapping):
        raise ValueError("each document type signature must be an object")
    document_type = raw.get("type")
    if not isinstance(document_type, str) or not document_type.strip():
        raise ValueError("document type names must be non-empty strings")
    document_type = document_type.strip()
    if document_type == UNKNOWN_DOCUMENT_TYPE:
        raise ValueError("unknown is reserved for classifier abstention")

    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"document type {document_type!r} requires rules")
    rules = tuple(
        _validate_signature_rule(rule, document_type=document_type)
        for rule in raw_rules
    )
    return _DocumentTypeRules(document_type=document_type, rules=rules)


def _validate_signature_rule(raw: object, *, document_type: str) -> _SignatureRule:
    if not isinstance(raw, Mapping):
        raise ValueError(f"rules for {document_type!r} must be objects")
    raw_phrases = raw.get("phrases")
    if not isinstance(raw_phrases, list) or not raw_phrases:
        raise ValueError(f"rules for {document_type!r} require phrases")
    phrases = tuple(
        normalized
        for phrase in raw_phrases
        if isinstance(phrase, str)
        and (normalized := _normalized_token_window(phrase, max_tokens=32))
    )
    if len(phrases) != len(raw_phrases):
        raise ValueError(f"rules for {document_type!r} contain invalid phrases")
    confidence = _probability(raw.get("confidence"), field="confidence", minimum=0.01)
    return _SignatureRule(phrases=phrases, confidence=confidence)


def _probability(
    raw: object,
    *,
    field: str,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"document type {field} must be numeric")
    value = float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"document type {field} must be between {minimum} and {maximum}"
        )
    return value


__all__ = [
    "DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE",
    "UNKNOWN_DOCUMENT_TYPE",
    "DocumentClassification",
    "classify_document",
]
