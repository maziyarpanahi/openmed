"""Deterministic span-native clinical coreference resolution.

The resolver links already-detected :class:`~openmed.core.schemas.OpenMedSpan`
mentions within one document.  It uses only local lexical and structural
features: mention form, head noun, entity type, section, sentence distance, and
experiencer.  It never performs network or model calls and does not log source
text.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from openmed.core.labels import (
    CONDITION,
    MEDICATION,
    OTHER,
    PERSON,
)
from openmed.core.schemas import OpenMedSpan

from .context import FAMILY_EXPERIENCER, PATIENT_EXPERIENCER
from .coref import canonicalize_text
from .experiencer import resolve_experiencer

SpanChainKey = tuple[str, tuple[int, int]]

COREFERENCE_RESOLUTION_ADVISORY = (
    "Clinical coreference chains are deterministic assistive annotations. "
    "Review them before clinical use."
)

COREFERENCE_FEATURES = (
    "section",
    "sentence_distance",
    "head_noun",
    "entity_type",
)

DEFAULT_COREFERENCE_THRESHOLD = 0.70

_WORD_RE = re.compile(
    r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)?", re.UNICODE
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?]+(?=\s|$)|\n+)")
_SPACE_RE = re.compile(r"\s+")

_PRONOUNS = {
    "he",
    "her",
    "hers",
    "him",
    "his",
    "it",
    "its",
    "she",
    "that",
    "their",
    "theirs",
    "them",
    "these",
    "they",
    "this",
    "those",
}
_PERSON_PRONOUNS = {"he", "her", "hers", "him", "his", "she"}
_NEUTRAL_PRONOUNS = {"it", "its", "that", "this"}
_PATIENT_ANCHORS = {
    "patient",
    "the patient",
    "this patient",
}
_DETERMINERS = {
    "a",
    "an",
    "another",
    "that",
    "the",
    "these",
    "this",
    "those",
}
_POSSESSIVES = {"her", "his", "its", "their"}

_CONDITION_HEADS = {
    "condition",
    "diagnosis",
    "disease",
    "finding",
    "lesion",
    "problem",
    "symptom",
}
_MEDICATION_HEADS = {
    "agent",
    "drug",
    "medication",
    "medicine",
    "prescription",
    "therapy",
}
_PERSON_HEADS = {"individual", "person", "patient"}
_GENERIC_TYPE_HEADS = (_CONDITION_HEADS | _MEDICATION_HEADS | _PERSON_HEADS) - {
    "lesion"
}
_FAMILY_ANCHORS = {
    "aunt",
    "brother",
    "daughter",
    "father",
    "mother",
    "parent",
    "relative",
    "sibling",
    "sister",
    "son",
}

_TYPE_ALIASES = {
    "condition": CONDITION,
    "diagnosis": CONDITION,
    "disease": CONDITION,
    "finding": CONDITION,
    "lesion": CONDITION,
    "problem": CONDITION,
    "symptom": CONDITION,
    "drug": MEDICATION,
    "med": MEDICATION,
    "medication": MEDICATION,
    "medicine": MEDICATION,
    "person": PERSON,
    "patient": PERSON,
}
_UNINFORMATIVE_TYPES = {
    "clinical entity",
    "entity",
    "mention",
    "nominal",
    "pronoun",
    "unknown",
}
_CONDITION_LABELS = {
    CONDITION,
    "CKD_STAGE",
    "CLINICAL_SIGNIFICANCE",
    "DYSPNEA_GRADE",
    "ENDOSCOPIC_FINDING",
    "GI_SYMPTOM",
    "POLYP_DESCRIPTOR",
    "RESPIRATORY_FINDING",
    "URINE_FINDING",
}
_MEDICATION_LABELS = {
    MEDICATION,
    "ANESTHETIC_AGENT",
    "ANTIBIOTIC",
    "INSULIN_REGIMEN",
    "VACCINE_NAME",
}
_PERSON_LABELS = {
    PERSON,
    "FIRST_NAME",
    "LAST_NAME",
    "MIDDLE_NAME",
}


@dataclass(frozen=True)
class CoreferenceChain:
    """One resolved entity chain with source-span provenance.

    ``members`` contains the original immutable spans in document order.
    ``representative`` is the most informative non-anaphoric member when one is
    available. ``confidence`` is the mean confidence of the deterministic links
    that formed the chain; singleton chains have confidence ``1.0``.
    """

    chain_id: str
    members: tuple[OpenMedSpan, ...]
    representative: OpenMedSpan
    confidence: float
    advisory: str = COREFERENCE_RESOLUTION_ADVISORY

    @property
    def member_spans(self) -> tuple[OpenMedSpan, ...]:
        """Return ``members`` under the explicit issue-facing field name."""

        return self.members

    @property
    def representative_mention(self) -> OpenMedSpan:
        """Return the representative source mention."""

        return self.representative


@dataclass(frozen=True)
class _Mention:
    span: OpenMedSpan
    surface: str
    canonical_text: str
    head_noun: str
    form: str
    entity_class: str | None
    section: str | None
    sentence_index: int
    experiencer: str

    @property
    def key(self) -> SpanChainKey:
        return self.span.doc_id, (self.span.start, self.span.end)


def resolve_coreference(
    spans: Iterable[OpenMedSpan],
    text: str,
    *,
    threshold: float = DEFAULT_COREFERENCE_THRESHOLD,
) -> tuple[tuple[CoreferenceChain, ...], dict[SpanChainKey, str]]:
    """Resolve clinical mention coreference within ``text``.

    Args:
        spans: ``OpenMedSpan`` mentions whose offsets index ``text``. Pronoun and
            nominal spans may use ``canonical_label=OTHER``; the resolver infers
            compatible antecedent types from their surface form when possible.
        text: Source text for the single-document resolution pass.
        threshold: Minimum deterministic compatibility score for a link.

    Returns:
        ``(chains, span_to_chain)``. The index maps
        ``(doc_id, (start, end))`` to a stable chain id, keeping raw source text
        out of the index and preserving offsets for review UIs.

    Raises:
        TypeError: If ``text`` or a span has the wrong type.
        ValueError: If offsets are invalid, duplicate span keys are supplied,
            spans refer to more than one document, or ``threshold`` is outside
            ``[0, 1]``.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("coreference threshold must be between 0 and 1")

    span_list = list(spans)
    if not span_list:
        return (), {}

    sentence_starts = _sentence_starts(text)
    mentions = tuple(
        sorted(
            (_build_mention(span, text, sentence_starts) for span in span_list),
            key=lambda mention: (
                mention.span.doc_id,
                mention.span.start,
                mention.span.end,
                mention.span.canonical_label,
                mention.span.entity_type,
            ),
        )
    )
    _validate_unique_keys(mentions)
    _validate_single_document(mentions)

    parents = list(range(len(mentions)))
    link_scores: dict[int, float] = {}
    for current_index, current in enumerate(mentions):
        candidate = _best_antecedent(
            mentions,
            parents,
            current_index,
            threshold,
        )
        if candidate is None:
            continue
        antecedent_index, score = candidate
        parents[current_index] = _find(parents, antecedent_index)
        link_scores[current_index] = score

    chains = _build_chains(mentions, parents, link_scores)
    span_to_chain = {
        (span.doc_id, (span.start, span.end)): chain.chain_id
        for chain in chains
        for span in chain.members
    }
    return chains, span_to_chain


def _build_mention(
    span: OpenMedSpan,
    text: str,
    sentence_starts: tuple[int, ...],
) -> _Mention:
    if not isinstance(span, OpenMedSpan):
        raise TypeError("spans must contain OpenMedSpan instances")
    if span.start == span.end:
        raise ValueError("coreference spans must not be empty")
    if span.end > len(text):
        raise ValueError("span offsets must be within text")

    surface = text[span.start : span.end]
    if not surface.strip():
        raise ValueError("coreference spans must contain non-whitespace text")

    words = _words(surface)
    if not words:
        raise ValueError("coreference spans must contain at least one word")
    normalized_surface = " ".join(words)
    form = _mention_form(normalized_surface, words)
    lexical_words = tuple(
        word
        for index, word in enumerate(words)
        if not (index == 0 and word in _DETERMINERS)
        and not (index == 0 and word in _POSSESSIVES)
    )
    canonical_text = _canonical_text(lexical_words or words)
    head_noun = canonical_text.split()[-1]
    entity_class = _entity_class(span, form, normalized_surface, head_noun)
    section = _section_for(span)
    experiencer = _experiencer_for(
        span,
        text,
        section,
        form,
        normalized_surface,
    )

    return _Mention(
        span=span,
        surface=surface,
        canonical_text=canonical_text,
        head_noun=head_noun,
        form=form,
        entity_class=entity_class,
        section=section,
        sentence_index=_sentence_index(sentence_starts, span.start),
        experiencer=experiencer,
    )


def _best_antecedent(
    mentions: tuple[_Mention, ...],
    parents: list[int],
    current_index: int,
    threshold: float,
) -> tuple[int, float] | None:
    current = mentions[current_index]
    candidates: list[tuple[float, int, int]] = []
    members_by_root: dict[int, list[_Mention]] = {}
    for index in range(current_index):
        members_by_root.setdefault(_find(parents, index), []).append(mentions[index])
    representatives = {
        root: _representative_mention(tuple(members))
        for root, members in members_by_root.items()
    }
    for antecedent_index in range(current_index):
        antecedent = mentions[antecedent_index]
        if antecedent.span.doc_id != current.span.doc_id:
            continue
        root = _find(parents, antecedent_index)
        representative = representatives[root]
        score = _link_score(antecedent, representative, current)
        if score is not None and score >= threshold:
            candidates.append((score, antecedent.span.end, antecedent_index))

    if not candidates:
        return None
    score, _end, antecedent_index = max(candidates)
    return antecedent_index, score


def _link_score(
    antecedent: _Mention,
    representative: _Mention,
    current: _Mention,
) -> float | None:
    if antecedent.span.end > current.span.start:
        return None
    if representative.form == "pronoun":
        return None
    if antecedent.experiencer != current.experiencer:
        return None

    sentence_distance = current.sentence_index - antecedent.sentence_index
    if sentence_distance < 0:
        return None
    if current.form == "pronoun" and sentence_distance > 2:
        return None
    if current.form in {"nominal", "patient_anchor"} and sentence_distance > 3:
        return None

    candidate_class = representative.entity_class or antecedent.entity_class
    if not _types_compatible(candidate_class, current.entity_class):
        return None

    lexical_score: float
    if current.form == "lexical":
        if current.canonical_text != representative.canonical_text:
            return None
        lexical_score = 0.62
    elif current.form == "nominal":
        if current.canonical_text == representative.canonical_text:
            lexical_score = 0.62
        elif current.head_noun == representative.head_noun:
            lexical_score = 0.54
        elif (
            current.head_noun in _GENERIC_TYPE_HEADS
            and current.entity_class
            and current.entity_class == candidate_class
        ):
            lexical_score = 0.42
        else:
            return None
    elif current.form == "patient_anchor":
        if candidate_class != PERSON:
            return None
        lexical_score = 0.46
    else:
        if current.canonical_text in _PERSON_PRONOUNS and candidate_class != PERSON:
            return None
        if current.canonical_text in _NEUTRAL_PRONOUNS and candidate_class == PERSON:
            return None
        lexical_score = 0.50

    type_score = (
        0.18
        if current.entity_class and current.entity_class == candidate_class
        else 0.08
    )
    section_score = _section_score(antecedent.section, current.section)
    distance_score = _sentence_distance_score(sentence_distance)
    recency_score = _recency_score(antecedent, current)
    return round(
        min(
            1.0,
            lexical_score + type_score + section_score + distance_score + recency_score,
        ),
        6,
    )


def _types_compatible(left: str | None, right: str | None) -> bool:
    return left is None or right is None or left == right


def _section_score(left: str | None, right: str | None) -> float:
    if left is None or right is None:
        return 0.0
    return 0.08 if left == right else -0.03


def _sentence_distance_score(distance: int) -> float:
    if distance == 0:
        return 0.12
    if distance == 1:
        return 0.09
    if distance == 2:
        return 0.05
    if distance == 3:
        return 0.02
    return 0.0


def _recency_score(antecedent: _Mention, current: _Mention) -> float:
    gap = max(0, current.span.start - antecedent.span.end)
    if gap <= 40:
        return 0.08
    if gap <= 120:
        return 0.06
    if gap <= 300:
        return 0.03
    return 0.0


def _build_chains(
    mentions: tuple[_Mention, ...],
    parents: list[int],
    link_scores: Mapping[int, float],
) -> tuple[CoreferenceChain, ...]:
    grouped: dict[int, list[int]] = {}
    for index in range(len(mentions)):
        grouped.setdefault(_find(parents, index), []).append(index)

    chains = []
    for indexes in grouped.values():
        chain_mentions = tuple(mentions[index] for index in indexes)
        representative = _representative_mention(chain_mentions)
        scores = [link_scores[index] for index in indexes if index in link_scores]
        confidence = 1.0 if not scores else round(sum(scores) / len(scores), 6)
        members = tuple(mention.span for mention in chain_mentions)
        chains.append(
            CoreferenceChain(
                chain_id=_chain_id(chain_mentions),
                members=members,
                representative=representative.span,
                confidence=confidence,
            )
        )

    return tuple(
        sorted(
            chains,
            key=lambda chain: (
                chain.members[0].doc_id,
                chain.members[0].start,
                chain.members[0].end,
            ),
        )
    )


def _representative_mention(mentions: tuple[_Mention, ...]) -> _Mention:
    return min(
        mentions,
        key=lambda mention: (
            _form_rank(mention.form),
            -len(mention.canonical_text.split()),
            -len(mention.canonical_text),
            mention.span.start,
            mention.span.end,
        ),
    )


def _form_rank(form: str) -> int:
    return {
        "lexical": 0,
        "nominal": 1,
        "patient_anchor": 2,
        "pronoun": 3,
    }[form]


def _chain_id(mentions: tuple[_Mention, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(mentions[0].span.doc_id.encode("utf-8"))
    for mention in mentions:
        digest.update(
            (
                f"|{mention.span.start}:{mention.span.end}:"
                f"{mention.span.canonical_label}"
            ).encode("utf-8")
        )
    return f"{mentions[0].span.doc_id}:coref:{digest.hexdigest()[:12]}"


def _mention_form(normalized_surface: str, words: tuple[str, ...]) -> str:
    if normalized_surface in _PATIENT_ANCHORS:
        return "patient_anchor"
    if normalized_surface in _PRONOUNS:
        return "pronoun"
    if len(words) > 1 and words[0] in _DETERMINERS:
        return "nominal"
    return "lexical"


def _entity_class(
    span: OpenMedSpan,
    form: str,
    normalized_surface: str,
    head_noun: str,
) -> str | None:
    if form == "patient_anchor":
        return PERSON
    if normalized_surface in _PERSON_PRONOUNS:
        return PERSON

    head_type = _head_type(head_noun)
    if form == "nominal" and head_type:
        return head_type
    if span.canonical_label != OTHER:
        return _canonical_entity_class(span.canonical_label)

    entity_type = _normalize_label(span.entity_type)
    if entity_type in _UNINFORMATIVE_TYPES:
        return head_type
    return _TYPE_ALIASES.get(entity_type, head_type)


def _canonical_entity_class(label: str) -> str:
    if label in _CONDITION_LABELS:
        return CONDITION
    if label in _MEDICATION_LABELS:
        return MEDICATION
    if label in _PERSON_LABELS:
        return PERSON
    return label


def _head_type(head_noun: str) -> str | None:
    if head_noun in _CONDITION_HEADS:
        return CONDITION
    if head_noun in _MEDICATION_HEADS:
        return MEDICATION
    if head_noun in _PERSON_HEADS:
        return PERSON
    return None


def _experiencer_for(
    span: OpenMedSpan,
    text: str,
    section: str | None,
    form: str,
    normalized_surface: str,
) -> str:
    explicit = _metadata_value(span.metadata, "experiencer")
    if explicit is not None:
        return _normalize_experiencer(explicit)
    if form == "patient_anchor":
        return PATIENT_EXPERIENCER
    if set(normalized_surface.split()) & _FAMILY_ANCHORS:
        return FAMILY_EXPERIENCER

    section_experiencer = (
        FAMILY_EXPERIENCER if section is not None and "family" in section else None
    )
    assignment = resolve_experiencer(
        text,
        {"start": span.start, "end": span.end},
        section_experiencer=section_experiencer,
    )
    return _normalize_experiencer(assignment.experiencer)


def _metadata_value(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value
    context = metadata.get("context")
    if isinstance(context, Mapping):
        nested = context.get(key)
        if isinstance(nested, str) and nested.strip():
            return nested
    return None


def _normalize_experiencer(value: str) -> str:
    normalized = _normalize_label(value)
    if normalized in {
        "family member",
        "father",
        "mother",
        "parent",
        "relative",
        "sibling",
    }:
        return FAMILY_EXPERIENCER
    if normalized in {"self", "subject"}:
        return PATIENT_EXPERIENCER
    return normalized


def _section_for(span: OpenMedSpan) -> str | None:
    value = span.section
    if value is None:
        value = _metadata_value(span.metadata, "section")
    return _normalize_label(value) if value else None


def _canonical_text(words: tuple[str, ...]) -> str:
    surface = " ".join(words)
    try:
        return canonicalize_text(surface)
    except ValueError:
        return surface


def _words(surface: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _WORD_RE.finditer(surface))


def _normalize_label(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip().casefold().replace("_", " "))


def _sentence_starts(text: str) -> tuple[int, ...]:
    starts = [0]
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start < len(text) and start != starts[-1]:
            starts.append(start)
    return tuple(starts)


def _sentence_index(starts: tuple[int, ...], position: int) -> int:
    low = 0
    high = len(starts)
    while low < high:
        middle = (low + high) // 2
        if starts[middle] <= position:
            low = middle + 1
        else:
            high = middle
    return max(0, low - 1)


def _validate_unique_keys(mentions: tuple[_Mention, ...]) -> None:
    keys = [mention.key for mention in mentions]
    if len(keys) != len(set(keys)):
        raise ValueError("coreference spans must have unique document offsets")


def _validate_single_document(mentions: tuple[_Mention, ...]) -> None:
    if len({mention.span.doc_id for mention in mentions}) != 1:
        raise ValueError("coreference spans must belong to one document")


def _find(parents: list[int], item: int) -> int:
    while parents[item] != item:
        parents[item] = parents[parents[item]]
        item = parents[item]
    return item


__all__ = [
    "COREFERENCE_FEATURES",
    "COREFERENCE_RESOLUTION_ADVISORY",
    "DEFAULT_COREFERENCE_THRESHOLD",
    "CoreferenceChain",
    "SpanChainKey",
    "resolve_coreference",
]
