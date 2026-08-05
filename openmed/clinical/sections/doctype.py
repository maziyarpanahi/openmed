"""Deterministic, rules-first clinical document-type classification."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import TypedDict

from .detect import detect_sections

DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE = "data/doctype_signatures.json"
UNKNOWN_DOCUMENT_TYPE = "unknown"
GENERIC_DOCUMENT_TYPE = UNKNOWN_DOCUMENT_TYPE
DOCUMENT_TYPES = (
    "progress_note",
    "discharge_summary",
    "radiology_report",
    "pathology_report",
    "consult_note",
    "operative_note",
)
DOCUMENT_TYPE_CONFIDENCE_THRESHOLD = 0.5
DOCUMENT_TYPE_MAX_HEADER_TOKENS = 32

# LOINC's document ontology uses kind-of-document and setting axes. These
# compact lexical hints are reference-only: no LOINC table, codes, or release
# content is bundled here. Callers that need a code table must supply it
# through the existing vocabulary boundary.
LOINC_DOCUMENT_ONTOLOGY_AXES = ("kind-of-document", "setting")
LOINC_DOCUMENT_TYPE_HINTS = MappingProxyType(
    {
        "progress_note": ("progress", "daily", "follow-up", "outpatient"),
        "discharge_summary": ("discharge", "hospital course", "inpatient"),
        "radiology_report": ("radiology", "imaging", "x-ray", "ct", "mri"),
        "pathology_report": ("pathology", "histology", "specimen", "biopsy"),
        "consult_note": ("consultation", "specialist", "referral"),
        "operative_note": ("operative", "surgical", "operating room"),
    }
)

_DOCUMENT_TYPE_HEADER_PHRASES = MappingProxyType(
    {
        "progress_note": ("progress note", "daily progress note"),
        "discharge_summary": (
            "discharge summary",
            "hospital discharge summary",
        ),
        "radiology_report": (
            "radiology report",
            "imaging report",
            "x-ray report",
            "ct report",
            "mri report",
        ),
        "pathology_report": (
            "pathology report",
            "surgical pathology",
            "histopathology report",
        ),
        "consult_note": (
            "consultation note",
            "consult note",
            "specialist consultation",
        ),
        "operative_note": (
            "operative note",
            "operation note",
            "surgical operative note",
        ),
    }
)
_DOCUMENT_TYPE_KEYWORD_CUES = MappingProxyType(
    {
        "progress_note": (
            "subjective",
            "objective",
            "interval history",
            "assessment",
            "plan",
        ),
        "discharge_summary": (
            "hospital course",
            "discharge diagnoses",
            "disposition",
            "follow-up instructions",
        ),
        "radiology_report": (
            "technique",
            "comparison",
            "findings",
            "impression",
            "radiograph",
        ),
        "pathology_report": (
            "specimen",
            "gross description",
            "microscopic description",
            "final diagnosis",
            "histologic",
        ),
        "consult_note": (
            "reason for consultation",
            "consulting service",
            "referring provider",
            "recommendations",
            "clinical opinion",
        ),
        "operative_note": (
            "preoperative diagnosis",
            "postoperative diagnosis",
            "procedure performed",
            "estimated blood loss",
            "anesthesia",
        ),
    }
)
_SECTION_CUES_BY_DOCUMENT_TYPE = MappingProxyType(
    {
        "progress_note": (
            "history_of_present_illness",
            "assessment_and_plan",
            "medications",
        ),
        "discharge_summary": (
            "history_of_present_illness",
            "assessment_and_plan",
            "impression",
        ),
        "radiology_report": ("findings", "impression"),
        "pathology_report": ("findings", "impression"),
        "consult_note": (
            "history_of_present_illness",
            "assessment",
            "plan",
        ),
        "operative_note": ("findings", "assessment_and_plan"),
    }
)
_TOKEN_RE = re.compile(r"\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}-]\w+)*", re.UNICODE)


class DocumentClassification(TypedDict):
    """Public document-type prediction with an abstention-safe confidence."""

    type: str
    confidence: float


class DocumentTypeFeatures(TypedDict):
    """Stable, JSON-friendly signals used by the document-type baseline."""

    max_tokens: int
    token_count: int
    header_hits: dict[str, int]
    section_histogram: dict[str, int]
    keyword_cues: dict[str, int]
    loinc_hints: dict[str, int]


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
    """Classify a clinical note from deterministic features and signatures.

    The bundled signatures cover discharge summaries, progress notes,
    radiology reports, pathology reports, operative notes, and consult notes.
    Classification is local and deterministic. A score below
    :data:`DOCUMENT_TYPE_CONFIDENCE_THRESHOLD`, or an ambiguous top score,
    returns ``unknown`` rather than guessing.

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
    features = extract_doctype_features(text, max_tokens=table.max_tokens)
    window = _normalized_token_window(text, max_tokens=table.max_tokens)
    if not features["token_count"]:
        return _unknown_classification(table)

    matches: list[tuple[float, str]] = []
    for document_type in table.document_types:
        signature_confidences = [
            rule.confidence
            for rule in document_type.rules
            if all(_contains_phrase(window, phrase) for phrase in rule.phrases)
        ]
        signature_confidence = max(signature_confidences, default=0.0)
        feature_confidence = _feature_confidence(
            document_type.document_type,
            features,
        )
        confidence = max(signature_confidence, feature_confidence)
        if confidence >= DOCUMENT_TYPE_CONFIDENCE_THRESHOLD:
            matches.append((confidence, document_type.document_type))

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


def extract_doctype_features(
    text: str,
    *,
    max_tokens: int | None = None,
) -> DocumentTypeFeatures:
    """Extract deterministic document-type signals from a clinical note.

    The first ``max_tokens`` normalized tokens provide the bounded header and
    keyword window. Section counts are computed from :func:`detect_sections`
    over the complete note, and the LOINC-related values are lexical hints for
    the ``kind-of-document``/``setting`` axes rather than a terminology table.
    No text is returned in the feature vector, which keeps the result suitable
    for local audit and training-label plumbing.

    Args:
        text: Clinical note text.
        max_tokens: Optional first-token window override. The classifier's
            configured window is used by default.

    Returns:
        A stable mapping with fixed document-type signal keys and a sorted
        section-label histogram.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    configured_max_tokens = _load_signature_table().max_tokens
    window_size = (
        configured_max_tokens
        if max_tokens is None
        else _validate_max_tokens(max_tokens)
    )
    tokens = _normalized_tokens(text)[:window_size]
    window = " ".join(tokens)
    header_window = " ".join(tokens[:DOCUMENT_TYPE_MAX_HEADER_TOKENS])
    section_histogram = Counter(
        str(section["label"]) for section in detect_sections(text)
    )

    return {
        "max_tokens": window_size,
        "token_count": len(tokens),
        "header_hits": {
            document_type: _count_phrases(
                header_window,
                _DOCUMENT_TYPE_HEADER_PHRASES[document_type],
            )
            for document_type in DOCUMENT_TYPES
        },
        "section_histogram": dict(sorted(section_histogram.items())),
        "keyword_cues": {
            document_type: _count_phrases(
                window,
                _DOCUMENT_TYPE_KEYWORD_CUES[document_type],
            )
            for document_type in DOCUMENT_TYPES
        },
        "loinc_hints": {
            document_type: _count_phrases(
                window,
                LOINC_DOCUMENT_TYPE_HINTS[document_type],
            )
            for document_type in DOCUMENT_TYPES
        },
    }


def _normalized_token_window(text: str, *, max_tokens: int) -> str:
    return " ".join(_normalized_tokens(text)[:max_tokens])


def _normalized_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def _count_phrases(window: str, phrases: Sequence[str]) -> int:
    return sum(_contains_phrase(window, phrase) for phrase in phrases)


def _feature_confidence(
    document_type: str,
    features: DocumentTypeFeatures,
) -> float:
    header_strength = min(features["header_hits"].get(document_type, 0), 1)
    keyword_strength = min(
        features["keyword_cues"].get(document_type, 0) / 3.0,
        1.0,
    )
    loinc_strength = min(
        features["loinc_hints"].get(document_type, 0) / 2.0,
        1.0,
    )
    section_count = sum(
        features["section_histogram"].get(section, 0)
        for section in _SECTION_CUES_BY_DOCUMENT_TYPE.get(document_type, ())
    )
    section_strength = min(section_count / 2.0, 1.0)

    if header_strength:
        return round(
            min(
                0.99,
                0.68
                + (0.12 * keyword_strength)
                + (0.1 * loinc_strength)
                + (0.1 * section_strength),
            ),
            6,
        )

    return round(
        (0.4 * keyword_strength) + (0.3 * loinc_strength) + (0.3 * section_strength),
        6,
    )


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

    missing_types = set(DOCUMENT_TYPES).difference(seen_types)
    if missing_types:
        raise ValueError(
            "document type signatures are missing labels: "
            + ", ".join(sorted(missing_types))
        )

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


def _validate_max_tokens(max_tokens: int) -> int:
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 1
    ):
        raise ValueError("document type max_tokens must be a positive integer")
    return max_tokens


__all__ = [
    "DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE",
    "DOCUMENT_TYPE_CONFIDENCE_THRESHOLD",
    "DOCUMENT_TYPE_MAX_HEADER_TOKENS",
    "DOCUMENT_TYPES",
    "UNKNOWN_DOCUMENT_TYPE",
    "GENERIC_DOCUMENT_TYPE",
    "DocumentClassification",
    "DocumentTypeFeatures",
    "LOINC_DOCUMENT_ONTOLOGY_AXES",
    "LOINC_DOCUMENT_TYPE_HINTS",
    "classify_document",
    "extract_doctype_features",
]
