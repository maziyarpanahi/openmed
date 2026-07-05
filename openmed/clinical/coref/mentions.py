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

_TEXT_KEYS = ("text", "surface", "mention", "label", "name", "term", "literal", "value")
_DOCUMENT_ID_KEYS = ("document_id", "doc_id", "source_doc_id")
_START_KEYS = ("start", "start_char", "start_offset", "begin", "offset_start")
_END_KEYS = ("end", "end_char", "end_offset", "stop", "offset_end")
_OCCURRENCE_KEYS = ("occurrence", "text_occurrence", "occurrence_index")
_SEMANTIC_TYPE_KEYS = (
    "semantic_type",
    "entity_type",
    "label",
    "role",
    "type",
    "category",
    "canonical_label",
)
_EVENT_TEXT_KEYS = ("text", "surface", "mention", "name", "term", "literal", "value")
_SECTION_KEYS = ("section", "section_label", "section_name")
_SYSTEM_KEYS = ("system", "coding_system", "code_system")
_CODE_KEYS = ("code", "concept_code", "coding_code")
_COREF_ENTITY_KEYS = ("coref_entity_id", "entity_id", "cluster_id")
_CANONICAL_TEXT_KEYS = (
    "canonical_text",
    "canonical_mention",
    "referent_text",
    "antecedent_text",
)
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
EVENT_COREFERENCE_SEMANTIC_TYPES = frozenset(
    {
        "analyte",
        "condition",
        "diagnosis",
        "disease",
        "drug",
        "finding",
        "lab",
        "laboratory",
        "medication",
        "medication_name",
        "medicine",
        "problem",
        "procedure",
        "test",
        "treatment",
    }
)
_EVENT_SEMANTIC_ALIASES = {
    "analyte": "test",
    "condition": "problem",
    "diagnosis": "problem",
    "disease": "problem",
    "drug": "treatment",
    "finding": "problem",
    "lab": "test",
    "laboratory": "test",
    "med": "treatment",
    "medication": "treatment",
    "medication_name": "treatment",
    "medicine": "treatment",
    "problem": "problem",
    "procedure": "treatment",
    "test": "test",
    "treatment": "treatment",
}
_DEFINITE_EVENT_NP_RE = re.compile(
    r"\b(?P<det>the|this|that|these|those)\s+"
    r"(?P<head>"
    r"antibiotics?|conditions?|coughs?|diseases?|drugs?|findings?|"
    r"infections?|labs?|lesions?|masses|medications?|medicines?|"
    r"nodules?|pneumonia|problems?|results?|studies|study|tests?|"
    r"treatments?|tumou?rs?"
    r")\b",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(r"\b(?:it|this|that|they|them|these|those)\b", re.IGNORECASE)
_TIME_ANAPHORA_FOLLOWERS = {
    "afternoon",
    "am",
    "evening",
    "morning",
    "night",
    "pm",
    "week",
}
_ANAPHORA_WINDOW_CHARS = 700


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


def event_coreference_mentions(
    document_text: str,
    mentions: Iterable[Any],
    *,
    document_id: str = DEFAULT_DOCUMENT_ID,
    include_anaphora: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    """Return event-coreference mention candidates from typed clinical spans.

    The returned records keep raw text only as an internal input to
    canonicalization. Public coreference output is sanitized by
    :func:`openmed.clinical.coref.resolve_coreference`.
    """

    if not isinstance(document_text, str):
        raise TypeError("document_text must be a string")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")

    candidates: list[dict[str, Any]] = []
    seen_offsets: set[SpanOffset] = set()
    for raw in mentions:
        candidate = _event_mention_mapping(
            raw,
            document_text=document_text,
            document_id=document_id,
        )
        if candidate is None:
            continue
        offset = (candidate["start"], candidate["end"])
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        candidates.append(candidate)

    if include_anaphora and candidates:
        anchors = canonicalize_mentions(candidates, document_text=document_text)
        for candidate in _anaphoric_event_mentions(document_text, anchors):
            offset = (candidate["start"], candidate["end"])
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)
            candidates.append(candidate)

    return tuple(sorted(candidates, key=lambda item: (item["start"], item["end"])))


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
    canonical_override = _clean_optional_text(_field_value(raw, _CANONICAL_TEXT_KEYS))
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
    canonical_source = (
        _normalize_surface(canonical_override)
        if canonical_override is not None
        else normalized_text
    )

    return CanonicalMention(
        document_id=document_id or DEFAULT_DOCUMENT_ID,
        source_index=source_index,
        text=text,
        start=start,
        end=end,
        normalized_text=normalized_text,
        canonical_text=_canonicalize_normalized_text(
            canonical_source,
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


def _event_mention_mapping(
    raw: Any,
    *,
    document_text: str,
    document_id: str,
) -> dict[str, Any] | None:
    if raw is None:
        return None
    raw_semantic_type = _field_value(raw, _SEMANTIC_TYPE_KEYS)
    semantic_type = _event_semantic_type(raw_semantic_type)
    if semantic_type is None:
        return None

    text, start, end = _event_text_and_offsets(raw, document_text)
    payload: dict[str, Any] = {
        "document_id": _clean_optional_text(_field_value(raw, _DOCUMENT_ID_KEYS))
        or document_id,
        "text": text,
        "start": start,
        "end": end,
        "semantic_type": semantic_type,
    }
    for output_key, keys in (
        ("section", _SECTION_KEYS),
        ("system", _SYSTEM_KEYS),
        ("code", _CODE_KEYS),
        ("coref_entity_id", _COREF_ENTITY_KEYS),
        ("canonical_text", _CANONICAL_TEXT_KEYS),
        ("negation", _NEGATION_KEYS),
        ("temporality", _TEMPORALITY_KEYS),
        ("experiencer", _EXPERIENCER_KEYS),
    ):
        value = _field_value(raw, keys)
        if value is not None:
            payload[output_key] = value
    return payload


def _event_text_and_offsets(raw: Any, document_text: str) -> tuple[str, int, int]:
    raw_text = _field_value(raw, _EVENT_TEXT_KEYS)
    if raw_text is not None:
        text = _clean_text(raw_text)
        start, end = _offsets_for(raw, text, document_text)
        return text, start, end

    explicit_offset = _field_value(raw, ("offset", "span", "char_span"))
    if explicit_offset is not None:
        start, end = _validate_offset(explicit_offset, document_text)
    else:
        start = _field_value(raw, _START_KEYS)
        end = _field_value(raw, _END_KEYS)
        start, end = _validate_offset((start, end), document_text)
    return _clean_text(document_text[start:end]), start, end


def _event_semantic_type(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("event mention semantic type must be a string")
    normalized = _normalize_optional_label(value)
    if normalized is None:
        return None
    if normalized not in EVENT_COREFERENCE_SEMANTIC_TYPES:
        return None
    return _EVENT_SEMANTIC_ALIASES.get(normalized, normalized)


def _anaphoric_event_mentions(
    document_text: str,
    anchors: tuple[CanonicalMention, ...],
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    dynamic_anchors = list(anchors)
    occupied_offsets = {anchor.offset for anchor in anchors}
    definite_matches = _iter_definite_np_matches(document_text)
    definite_offsets = [(start, end) for start, end, _ in definite_matches]
    matches = [
        *definite_matches,
        *_iter_pronoun_matches(document_text, excluded_offsets=definite_offsets),
    ]
    matches.sort(key=lambda item: (item[0], item[1]))
    for start, end, semantic_hint in matches:
        if any(
            _spans_overlap(start, end, left, right) for left, right in occupied_offsets
        ):
            continue
        antecedent = _nearest_antecedent(
            tuple(dynamic_anchors),
            start=start,
            semantic_hint=semantic_hint,
        )
        if antecedent is None:
            continue
        mention = {
            "document_id": antecedent.document_id,
            "text": document_text[start:end],
            "start": start,
            "end": end,
            "semantic_type": antecedent.semantic_type,
            "section": antecedent.section,
            "canonical_text": antecedent.canonical_text,
            "negation": antecedent.negation,
            "temporality": antecedent.temporality,
            "experiencer": antecedent.experiencer,
        }
        mentions.append(mention)
        dynamic_anchors.append(
            canonicalize_mentions((mention,), document_text=document_text)[0]
        )
        occupied_offsets.add((start, end))
    return mentions


def _iter_definite_np_matches(document_text: str) -> list[tuple[int, int, str | None]]:
    matches: list[tuple[int, int, str | None]] = []
    for match in _DEFINITE_EVENT_NP_RE.finditer(document_text):
        matches.append(
            (
                match.start(),
                match.end(),
                _semantic_hint_for_head(match.group("head")),
            )
        )
    return matches


def _iter_pronoun_matches(
    document_text: str,
    *,
    excluded_offsets: list[SpanOffset],
) -> list[tuple[int, int, None]]:
    matches: list[tuple[int, int, None]] = []
    for match in _PRONOUN_RE.finditer(document_text):
        if _is_temporal_anaphor(document_text, match.end()):
            continue
        if any(
            _spans_overlap(match.start(), match.end(), start, end)
            for start, end in excluded_offsets
        ):
            continue
        matches.append((match.start(), match.end(), None))
    return matches


def _semantic_hint_for_head(head: str) -> str | None:
    normalized = _singularize(_normalize_surface(head))
    if normalized in {
        "condition",
        "cough",
        "disease",
        "finding",
        "infection",
        "lesion",
        "mass",
        "nodule",
        "pneumonia",
        "problem",
        "tumor",
        "tumour",
    }:
        return "problem"
    if normalized in {"lab", "result", "study", "test"}:
        return "test"
    if normalized in {
        "antibiotic",
        "drug",
        "medication",
        "medicine",
        "treatment",
    }:
        return "treatment"
    return None


def _nearest_antecedent(
    anchors: tuple[CanonicalMention, ...],
    *,
    start: int,
    semantic_hint: str | None,
) -> CanonicalMention | None:
    compatible = [
        anchor
        for anchor in anchors
        if anchor.end <= start
        and start - anchor.end <= _ANAPHORA_WINDOW_CHARS
        and _semantic_types_match(anchor.semantic_type, semantic_hint)
    ]
    if not compatible:
        return None
    return max(compatible, key=lambda anchor: (anchor.end, -anchor.start))


def _semantic_types_match(candidate: str | None, semantic_hint: str | None) -> bool:
    return semantic_hint is None or candidate is None or candidate == semantic_hint


def _is_temporal_anaphor(document_text: str, end: int) -> bool:
    remainder = document_text[end : end + 16].casefold().strip()
    if not remainder:
        return False
    next_word = remainder.split(maxsplit=1)[0].strip(".,;:")
    return next_word in _TIME_ANAPHORA_FOLLOWERS


def _spans_overlap(
    left_start: int, left_end: int, right_start: int, right_end: int
) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


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
    "EVENT_COREFERENCE_SEMANTIC_TYPES",
    "CanonicalMention",
    "SpanOffset",
    "canonicalize_mentions",
    "canonicalize_text",
    "event_coreference_mentions",
]
