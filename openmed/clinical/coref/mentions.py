"""Canonical clinical mention normalization for document-level coreference."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from openmed.clinical.context import (
    AFFIRMED,
    CERTAIN,
    FAMILY_EXPERIENCER,
    NEGATION_VALUES,
    PATIENT_EXPERIENCER,
    RECENT,
    TEMPORALITY_VALUES,
    ClinicalAssertion,
    Negation,
    apply_section_context,
    canonical_section_name,
)

SpanOffset = tuple[int, int]

DEFAULT_DOCUMENT_ID = "document"

DEFAULT_ABBREVIATION_EXPANSIONS: dict[str, str] = {
    "af": "atrial fibrillation",
    "afib": "atrial fibrillation",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "ckd": "chronic kidney disease",
    "copd": "chronic obstructive pulmonary disease",
    "dm": "diabetes mellitus",
    "gerd": "gastroesophageal reflux disease",
    "hld": "hyperlipidemia",
    "htn": "hypertension",
    "mi": "myocardial infarction",
    "t2dm": "diabetes mellitus",
    "uti": "urinary tract infection",
}

DEFAULT_CANONICAL_ALIASES: dict[str, str] = {
    "blood glucose": "diabetes mellitus",
    "blood sugar": "diabetes mellitus",
    "blood sugars": "diabetes mellitus",
    "diabetes": "diabetes mellitus",
    "elevated blood glucose": "diabetes mellitus",
    "elevated blood sugar": "diabetes mellitus",
    "elevated sugar": "diabetes mellitus",
    "elevated sugars": "diabetes mellitus",
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "kidney disease": "chronic kidney disease",
    "sugar": "diabetes mellitus",
    "sugars": "diabetes mellitus",
}

_TEXT_KEYS = ("text", "surface", "mention", "label", "name", "term", "literal")
_DOCUMENT_ID_KEYS = ("document_id", "doc_id", "source_doc_id")
_START_KEYS = ("start", "start_char", "start_offset", "begin", "offset_start")
_END_KEYS = ("end", "end_char", "end_offset", "stop", "offset_end")
_OCCURRENCE_KEYS = ("occurrence", "text_occurrence", "occurrence_index")
_SEMANTIC_TYPE_KEYS = (
    "semantic_type",
    "entity_type",
    "type",
    "category",
    "canonical_label",
)
_SECTION_KEYS = ("section", "section_label", "section_name")
_SYSTEM_KEYS = ("system", "coding_system", "code_system")
_CODE_KEYS = ("code", "concept_code", "coding_code")
_COREF_ENTITY_KEYS = ("coref_entity_id", "entity_id", "cluster_id")
_NEGATION_KEYS = ("negation", "polarity")
_TEMPORALITY_KEYS = ("temporality", "temporal_status")
_EXPERIENCER_KEYS = ("experiencer",)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^a-z0-9/+]+")
_POSSESSIVE_RE = re.compile(r"\b([a-z]+)'s\b")
_LEADING_CONTEXT_RE = re.compile(
    r"^(?:"
    r"past medical history of|family history of|history of|hx of|h/o|"
    r"status post|s/p|patient has|patient with|the patient has|"
    r"the patient with|mother with|father with|mother had|father had"
    r")\s+"
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "her",
    "his",
    "of",
    "patient",
    "patients",
    "the",
    "their",
    "with",
}
_SINGULAR_EXCEPTIONS = {
    "asbestosis",
    "diabetes",
    "fibrosis",
    "hyperlipidemia",
    "necrosis",
    "sepsis",
    "status",
}


@dataclass(frozen=True)
class CanonicalMention:
    """Clinical mention normalized for deterministic document coreference.

    ``text``, ``start``, and ``end`` preserve the original surface form and
    document offsets. ``canonical_text`` is the comparison key after
    abbreviation expansion, alias hooks, and conservative morphological
    normalization.
    """

    document_id: str
    source_index: int
    text: str
    start: int
    end: int
    normalized_text: str
    canonical_text: str
    semantic_type: str | None = None
    section: str | None = None
    canonical_section: str | None = None
    negation: Negation = AFFIRMED
    temporality: str = RECENT
    experiencer: str = PATIENT_EXPERIENCER
    system: str | None = None
    code: str | None = None
    coref_entity_id: str | None = None

    @property
    def offset(self) -> SpanOffset:
        """Return the preserved source span offsets."""

        return self.start, self.end

    @property
    def stable_key(self) -> tuple[str, int, int, str, str]:
        """Return a deterministic key independent of input iteration order."""

        return (
            self.document_id,
            self.start,
            self.end,
            self.canonical_text,
            self.text.casefold(),
        )


def canonicalize_mentions(
    mentions: Iterable[Any],
    *,
    document_text: str | None = None,
    abbreviation_expansions: Mapping[str, str] | None = None,
    canonical_aliases: Mapping[str, str] | None = None,
) -> tuple[CanonicalMention, ...]:
    """Canonicalize candidate mentions while preserving provenance offsets.

    Args:
        mentions: Candidate mention mappings or objects. Each mention must
            expose a text-like field plus ``start``/``end`` offsets, an
            ``offset`` pair, or enough information to be located in
            ``document_text``.
        document_text: Optional source text used to locate mentions without
            explicit offsets and to validate provided offsets.
        abbreviation_expansions: Optional hook for project- or caller-specific
            abbreviation expansion. Values override the default clinical
            abbreviations for matching keys.
        canonical_aliases: Optional hook for surface-form aliases that should
            normalize to one canonical clinical concept.

    Returns:
        Canonical mentions sorted by document id and source offsets. Sorting
        makes downstream clustering deterministic and order-invariant after
        canonicalization.
    """

    expansions = _normalized_mapping(DEFAULT_ABBREVIATION_EXPANSIONS)
    if abbreviation_expansions:
        expansions.update(_normalized_mapping(abbreviation_expansions))

    aliases = _normalized_mapping(DEFAULT_CANONICAL_ALIASES)
    if canonical_aliases:
        aliases.update(_normalized_mapping(canonical_aliases))

    canonicalized = [
        _coerce_mention(
            mention,
            source_index=index,
            document_text=document_text,
            abbreviation_expansions=expansions,
            canonical_aliases=aliases,
        )
        for index, mention in enumerate(mentions)
    ]
    return tuple(sorted(canonicalized, key=lambda mention: mention.stable_key))


def canonicalize_text(
    text: str,
    *,
    abbreviation_expansions: Mapping[str, str] | None = None,
    canonical_aliases: Mapping[str, str] | None = None,
) -> str:
    """Return the canonical comparison key for a mention surface form."""

    if not isinstance(text, str):
        raise TypeError("mention text must be a string")
    normalized = _normalize_surface(text)

    expansions = _normalized_mapping(DEFAULT_ABBREVIATION_EXPANSIONS)
    if abbreviation_expansions:
        expansions.update(_normalized_mapping(abbreviation_expansions))

    aliases = _normalized_mapping(DEFAULT_CANONICAL_ALIASES)
    if canonical_aliases:
        aliases.update(_normalized_mapping(canonical_aliases))

    return _canonicalize_normalized_text(normalized, expansions, aliases)


def _coerce_mention(
    raw: Any,
    *,
    source_index: int,
    document_text: str | None,
    abbreviation_expansions: Mapping[str, str],
    canonical_aliases: Mapping[str, str],
) -> CanonicalMention:
    text = _clean_text(_field_value(raw, _TEXT_KEYS))
    start, end = _offsets_for(raw, text, document_text)
    if document_text is not None and document_text[start:end] != text:
        raise ValueError("mention offsets must slice the original mention text")

    document_id = _clean_optional_text(_field_value(raw, _DOCUMENT_ID_KEYS))
    semantic_type = _clean_optional_text(_field_value(raw, _SEMANTIC_TYPE_KEYS))
    section = _clean_optional_text(_field_value(raw, _SECTION_KEYS))
    system = _clean_optional_text(_field_value(raw, _SYSTEM_KEYS))
    code = _clean_optional_text(_field_value(raw, _CODE_KEYS))
    coref_entity_id = _clean_optional_text(_field_value(raw, _COREF_ENTITY_KEYS))
    negation = _normalize_negation(_field_value(raw, _NEGATION_KEYS))
    temporality = _normalize_temporality(_field_value(raw, _TEMPORALITY_KEYS))
    explicit_experiencer = _clean_optional_text(_field_value(raw, _EXPERIENCER_KEYS))

    assertion = apply_section_context(
        raw if not isinstance(raw, str) else {"text": text, "section": section},
        section,
        ClinicalAssertion(
            temporality=temporality,
            certainty=CERTAIN,
            negation=negation,
            experiencer=_normalize_experiencer(explicit_experiencer)
            if explicit_experiencer
            else None,
        ),
    )
    experiencer = assertion.experiencer or PATIENT_EXPERIENCER
    normalized_text = _normalize_surface(text)

    return CanonicalMention(
        document_id=document_id or DEFAULT_DOCUMENT_ID,
        source_index=source_index,
        text=text,
        start=start,
        end=end,
        normalized_text=normalized_text,
        canonical_text=_canonicalize_normalized_text(
            normalized_text,
            abbreviation_expansions,
            canonical_aliases,
        ),
        semantic_type=_normalize_optional_label(semantic_type),
        section=section,
        canonical_section=canonical_section_name(section),
        negation=negation,
        temporality=assertion.temporality,
        experiencer=_normalize_experiencer(experiencer) or PATIENT_EXPERIENCER,
        system=_normalize_optional_label(system),
        code=_normalize_optional_label(code),
        coref_entity_id=coref_entity_id,
    )


def _field_value(raw: Any, keys: tuple[str, ...]) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw if keys == _TEXT_KEYS else None
    if isinstance(raw, Mapping):
        for key in keys:
            value = raw.get(key)
            if value is not None:
                return value
        return None
    for key in keys:
        value = getattr(raw, key, None)
        if value is not None:
            return value
    return None


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("mention text must be a string")
    text = _WHITESPACE_RE.sub(" ", value.strip())
    if not text:
        raise ValueError("mention text must not be empty")
    return text


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("mention metadata values must be strings when provided")
    cleaned = _WHITESPACE_RE.sub(" ", value.strip())
    return cleaned or None


def _offsets_for(raw: Any, text: str, document_text: str | None) -> SpanOffset:
    explicit_offset = _field_value(raw, ("offset", "span", "char_span"))
    if explicit_offset is not None:
        return _validate_offset(explicit_offset, document_text)

    start = _field_value(raw, _START_KEYS)
    end = _field_value(raw, _END_KEYS)
    if start is not None or end is not None:
        return _validate_offset((start, end), document_text)

    if document_text is None:
        raise ValueError("mention offsets are required when document_text is absent")

    occurrence = _field_value(raw, _OCCURRENCE_KEYS)
    return _locate_text(document_text, text, occurrence)


def _validate_offset(value: Any, document_text: str | None) -> SpanOffset:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], int)
        or not isinstance(value[1], int)
    ):
        raise TypeError("mention offset must be a (start, end) integer pair")
    start, end = value
    if start < 0 or end < start:
        raise ValueError("mention offset must satisfy 0 <= start <= end")
    if document_text is not None and end > len(document_text):
        raise ValueError("mention offset end must be within document_text")
    return start, end


def _locate_text(document_text: str, text: str, occurrence: Any) -> SpanOffset:
    if occurrence is None:
        wanted_occurrence = 0
    elif isinstance(occurrence, int) and not isinstance(occurrence, bool):
        wanted_occurrence = occurrence
    else:
        raise TypeError("mention occurrence must be an integer when provided")
    if wanted_occurrence < 0:
        raise ValueError("mention occurrence must be non-negative")

    haystack = document_text.casefold()
    needle = text.casefold()
    cursor = 0
    seen = 0
    while True:
        start = haystack.find(needle, cursor)
        if start == -1:
            raise ValueError("mention text is not present in document_text")
        if seen == wanted_occurrence:
            return start, start + len(text)
        seen += 1
        cursor = start + 1


def _normalize_surface(text: str) -> str:
    normalized = text.casefold().replace("&", " and ")
    normalized = _POSSESSIVE_RE.sub(r"\1", normalized)
    normalized = normalized.replace("h/o", "history of")
    normalized = normalized.replace("s/p", "status post")
    normalized = _PUNCTUATION_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    normalized = _LEADING_CONTEXT_RE.sub("", normalized)
    tokens = [
        _singularize(token)
        for token in normalized.split()
        if token and token not in _STOPWORDS
    ]
    return " ".join(tokens)


def _canonicalize_normalized_text(
    normalized: str,
    abbreviation_expansions: Mapping[str, str],
    canonical_aliases: Mapping[str, str],
) -> str:
    if not normalized:
        raise ValueError("mention canonical text must not be empty")

    candidate = abbreviation_expansions.get(normalized, normalized)
    candidate = canonical_aliases.get(candidate, candidate)
    tokens = [
        abbreviation_expansions.get(token, token)
        for token in candidate.split()
        if token and token not in _STOPWORDS
    ]
    candidate = " ".join(tokens)
    candidate = _WHITESPACE_RE.sub(" ", candidate).strip()
    return canonical_aliases.get(candidate, candidate)


def _singularize(token: str) -> str:
    if token in _SINGULAR_EXCEPTIONS or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ses") or token.endswith("xes") or token.endswith("ches"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "itis")):
        return token[:-1]
    return token


def _normalized_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in mapping.items():
        normalized_key = _normalize_surface(key)
        normalized_value = _normalize_surface(value)
        if normalized_key and normalized_value:
            normalized[normalized_key] = normalized_value
    return normalized


def _normalize_negation(value: Any) -> Negation:
    if value is None:
        return AFFIRMED
    if not isinstance(value, str):
        raise TypeError("mention negation must be a string")
    normalized = value.strip().casefold()
    if normalized not in NEGATION_VALUES:
        raise ValueError(
            f"mention negation must be one of {', '.join(NEGATION_VALUES)}"
        )
    return cast(Negation, normalized)


def _normalize_temporality(value: Any) -> str:
    if value is None:
        return RECENT
    if not isinstance(value, str):
        raise TypeError("mention temporality must be a string")
    normalized = value.strip().casefold()
    if normalized not in TEMPORALITY_VALUES:
        raise ValueError(
            f"mention temporality must be one of {', '.join(TEMPORALITY_VALUES)}"
        )
    return normalized


def _normalize_experiencer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold().replace("-", "_")
    if normalized in {"relative", "mother", "father", "parent", "sibling"}:
        return FAMILY_EXPERIENCER
    if normalized in {"self", "subject"}:
        return PATIENT_EXPERIENCER
    return normalized or None


def _normalize_optional_label(value: str | None) -> str | None:
    if value is None:
        return None
    return _WHITESPACE_RE.sub(" ", value.strip()).casefold() or None


__all__ = [
    "DEFAULT_ABBREVIATION_EXPANSIONS",
    "DEFAULT_CANONICAL_ALIASES",
    "DEFAULT_DOCUMENT_ID",
    "CanonicalMention",
    "SpanOffset",
    "canonicalize_mentions",
    "canonicalize_text",
]
