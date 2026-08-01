"""Deterministic offline lexical matching for clinical vocabularies.

The matcher indexes caller-supplied term-to-concept mappings.  It does not
download or bundle terminology data, which keeps the lookup path fully local
and lets vocabulary-specific loaders control how their snapshots are parsed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

__all__ = [
    "AbbreviationMap",
    "ConceptInput",
    "ConceptMatch",
    "LexicalConcept",
    "LexicalMatcher",
    "MatchType",
    "VocabularyTerms",
    "normalize_term",
]


MatchType: TypeAlias = Literal["exact", "normalized", "abbreviation"]
AbbreviationMap: TypeAlias = Mapping[str, str | Sequence[str]]
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_MATCH_SCORES: Mapping[MatchType, float] = {
    "exact": 1.0,
    "normalized": 0.95,
    "abbreviation": 0.90,
}


@dataclass(frozen=True)
class LexicalConcept:
    """One coded concept stored in a :class:`LexicalMatcher`.

    Args:
        system_uri: Canonical URI identifying the vocabulary.
        code: Identifier within the vocabulary.
        display: Human-readable preferred term.
        metadata: Vocabulary-specific, non-PHI fields such as a term type or
            hierarchy flag.
    """

    system_uri: str
    code: str
    display: str
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_uri", _validate_system_uri(self.system_uri))
        object.__setattr__(self, "code", _nonempty_text(self.code, "code"))
        object.__setattr__(self, "display", _nonempty_text(self.display, "display"))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable vocabulary URI and code identity."""

        return (self.system_uri, self.code)


ConceptInput: TypeAlias = LexicalConcept | str | Mapping[str, object]
VocabularyTerms: TypeAlias = Mapping[str, ConceptInput | Sequence[ConceptInput]]


@dataclass(frozen=True)
class ConceptMatch:
    """A ranked lexical match returned for a query.

    ``match_type`` records whether the original term matched exactly, only
    after normalization, or through an abbreviation.  ``matched_term`` is the
    vocabulary term, not the caller's query, so match results do not duplicate
    arbitrary source text.
    """

    system_uri: str
    code: str
    display: str
    score: float
    match_type: MatchType
    matched_term: str
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_uri", _validate_system_uri(self.system_uri))
        object.__setattr__(self, "code", _nonempty_text(self.code, "code"))
        object.__setattr__(self, "display", _nonempty_text(self.display, "display"))
        object.__setattr__(
            self, "matched_term", _nonempty_text(self.matched_term, "matched_term")
        )
        if self.match_type not in _MATCH_SCORES:
            raise ValueError(
                "match_type must be 'exact', 'normalized', or 'abbreviation'"
            )
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be finite and between 0.0 and 1.0")
        object.__setattr__(self, "score", score)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable vocabulary URI and code identity."""

        return (self.system_uri, self.code)


@dataclass(frozen=True)
class _IndexedConcept:
    concept: LexicalConcept
    term: str
    order: int


class LexicalMatcher:
    """Index arbitrary vocabulary terms for deterministic lexical lookup.

    Values in ``terms`` may be :class:`LexicalConcept` objects, concept
    mappings, code strings, or sequences of those values.  ``system_uri`` is
    used as the default for mappings and is required for bare code strings.

    Args:
        terms: Vocabulary term to one-or-more-concepts mapping.
        system_uri: Default canonical vocabulary URI.
        abbreviations: Optional abbreviation-to-expanded-term mapping.  Common
            initialisms such as ``"cbc"`` for ``"complete blood count"`` are
            also indexed automatically.
    """

    def __init__(
        self,
        terms: VocabularyTerms,
        *,
        system_uri: str | None = None,
        abbreviations: AbbreviationMap | None = None,
    ) -> None:
        if not isinstance(terms, Mapping):
            raise TypeError("terms must be a term-to-concept mapping")
        self.system_uri = (
            _validate_system_uri(system_uri) if system_uri is not None else None
        )
        exact: dict[str, list[_IndexedConcept]] = defaultdict(list)
        normalized: dict[str, list[_IndexedConcept]] = defaultdict(list)
        exact_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
        normalized_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
        concept_keys: set[tuple[str, str]] = set()
        order = 0

        for raw_term, raw_concepts in terms.items():
            term = _nonempty_text(raw_term, "term")
            normalized_term = normalize_term(term)
            if not normalized_term:
                raise ValueError(f"term {term!r} has no indexable characters")
            for raw_concept in _concept_values(raw_concepts):
                concept = _coerce_concept(
                    raw_concept,
                    term=term,
                    default_system_uri=self.system_uri,
                )
                entry = _IndexedConcept(concept=concept, term=term, order=order)
                order += 1
                if concept.key not in exact_keys[term]:
                    exact[term].append(entry)
                    exact_keys[term].add(concept.key)
                if concept.key not in normalized_keys[normalized_term]:
                    normalized[normalized_term].append(entry)
                    normalized_keys[normalized_term].add(concept.key)
                concept_keys.add(concept.key)

        self._exact_index = _freeze_index(exact)
        self._normalized_index = _freeze_index(normalized)
        self._abbreviation_index = self._build_abbreviation_index(abbreviations or {})
        self._concept_count = len(concept_keys)

    @property
    def term_count(self) -> int:
        """Number of distinct source terms in the index."""

        return len(self._exact_index)

    @property
    def concept_count(self) -> int:
        """Number of distinct ``(system_uri, code)`` concepts in the index."""

        return self._concept_count

    def lookup(
        self, query: str, *, limit: int | None = None
    ) -> tuple[ConceptMatch, ...]:
        """Return deterministically ranked concepts for ``query``.

        Exact matches rank ahead of normalized matches, which rank ahead of
        abbreviation matches.  A concept reachable through multiple terms is
        returned once at its best score.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        normalized_query = normalize_term(query)
        if not normalized_query:
            return ()

        best: dict[tuple[str, str], ConceptMatch] = {}
        order_by_key: dict[tuple[str, str], int] = {}
        self._offer_matches(
            best,
            order_by_key,
            self._exact_index.get(query, ()),
            "exact",
        )
        self._offer_matches(
            best,
            order_by_key,
            self._normalized_index.get(normalized_query, ()),
            "normalized",
        )
        self._offer_matches(
            best,
            order_by_key,
            self._abbreviation_index.get(normalized_query, ()),
            "abbreviation",
        )

        matches = sorted(
            best.values(),
            key=lambda match: (
                -match.score,
                match.system_uri,
                match.code,
                match.display,
                order_by_key[match.key],
            ),
        )
        if limit is not None:
            matches = matches[:limit]
        return tuple(matches)

    def match(
        self, query: str, *, limit: int | None = None
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`lookup`."""

        return self.lookup(query, limit=limit)

    def _build_abbreviation_index(
        self, abbreviations: AbbreviationMap
    ) -> dict[str, tuple[_IndexedConcept, ...]]:
        abbreviation_index: dict[str, list[_IndexedConcept]] = defaultdict(list)
        abbreviation_keys: dict[str, set[tuple[tuple[str, str], str]]] = defaultdict(
            set
        )
        for normalized_term, entries in self._normalized_index.items():
            acronym = _acronym(normalized_term)
            if acronym:
                _extend_unique(
                    abbreviation_index[acronym],
                    abbreviation_keys[acronym],
                    entries,
                )

        for raw_abbreviation, raw_expansions in abbreviations.items():
            abbreviation = normalize_term(
                _nonempty_text(raw_abbreviation, "abbreviation")
            )
            expansions = (
                (raw_expansions,)
                if isinstance(raw_expansions, str)
                else tuple(raw_expansions)
            )
            if not expansions:
                raise ValueError(
                    f"abbreviation {raw_abbreviation!r} must have an expansion"
                )
            for raw_expansion in expansions:
                expansion = normalize_term(
                    _nonempty_text(raw_expansion, "abbreviation expansion")
                )
                expansion_entries = self._normalized_index.get(expansion)
                if expansion_entries is None:
                    raise ValueError(
                        f"abbreviation expansion {raw_expansion!r} does not match "
                        "an indexed vocabulary term"
                    )
                _extend_unique(
                    abbreviation_index[abbreviation],
                    abbreviation_keys[abbreviation],
                    expansion_entries,
                )
        return _freeze_index(abbreviation_index)

    @staticmethod
    def _offer_matches(
        best: dict[tuple[str, str], ConceptMatch],
        order_by_key: dict[tuple[str, str], int],
        entries: Sequence[_IndexedConcept],
        match_type: MatchType,
    ) -> None:
        score = _MATCH_SCORES[match_type]
        for entry in entries:
            key = entry.concept.key
            candidate = ConceptMatch(
                system_uri=entry.concept.system_uri,
                code=entry.concept.code,
                display=entry.concept.display,
                score=score,
                match_type=match_type,
                matched_term=entry.term,
                metadata=entry.concept.metadata,
            )
            existing = best.get(key)
            if existing is None or candidate.score > existing.score:
                best[key] = candidate
                order_by_key[key] = entry.order


def normalize_term(value: str) -> str:
    """Normalize a lexical surface without consulting external resources.

    Unicode compatibility normalization and case-folding are followed by
    punctuation/symbol folding and whitespace collapse.  Letters and numbers
    from every script are retained.
    """

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    folded = unicodedata.normalize("NFKC", value).casefold()
    characters = (
        character if _is_term_character(character) else " " for character in folded
    )
    return " ".join("".join(characters).split())


def _concept_values(
    value: ConceptInput | Sequence[ConceptInput],
) -> tuple[ConceptInput, ...]:
    if isinstance(value, (LexicalConcept, str, Mapping)):
        return (value,)
    if not isinstance(value, Sequence):
        raise TypeError(
            "concept values must be a concept, mapping, code string, or sequence"
        )
    concepts = tuple(value)
    if not concepts:
        raise ValueError("concept sequences must not be empty")
    return concepts


def _coerce_concept(
    value: ConceptInput,
    *,
    term: str,
    default_system_uri: str | None,
) -> LexicalConcept:
    if isinstance(value, LexicalConcept):
        return value
    if isinstance(value, str):
        if default_system_uri is None:
            raise ValueError("system_uri is required when concept values are codes")
        return LexicalConcept(default_system_uri, value, term)
    if not isinstance(value, Mapping):
        raise TypeError(
            "each concept must be a LexicalConcept, mapping, or code string"
        )

    raw_system_uri = (
        value.get("system_uri") or default_system_uri or value.get("system")
    )
    raw_code = value.get("code") or value.get("id")
    raw_display = value.get("display") or value.get("preferred_term") or term
    if raw_system_uri is None:
        raise ValueError("concept mappings require system_uri or a matcher default")
    if raw_code is None:
        raise ValueError("concept mappings require a code")
    raw_metadata = value.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("concept metadata must be a mapping")
    metadata = dict(raw_metadata)
    reserved = {
        "system_uri",
        "system",
        "code",
        "id",
        "display",
        "preferred_term",
        "metadata",
    }
    for key, item in value.items():
        if key not in reserved:
            metadata.setdefault(str(key), item)
    return LexicalConcept(
        system_uri=str(raw_system_uri),
        code=str(raw_code),
        display=str(raw_display),
        metadata=metadata,
    )


def _acronym(normalized_term: str) -> str:
    tokens = normalized_term.split()
    if len(tokens) < 2:
        return ""
    acronym = "".join(token[0] for token in tokens if token)
    return acronym if len(acronym) >= 2 else ""


def _is_term_character(character: str) -> bool:
    return character.isalnum() or unicodedata.category(character).startswith("M")


def _extend_unique(
    target: list[_IndexedConcept],
    existing: set[tuple[tuple[str, str], str]],
    entries: Sequence[_IndexedConcept],
) -> None:
    for entry in entries:
        identity = (entry.concept.key, entry.term)
        if identity not in existing:
            target.append(entry)
            existing.add(identity)


def _freeze_index(
    index: Mapping[str, Sequence[_IndexedConcept]],
) -> dict[str, tuple[_IndexedConcept, ...]]:
    return {key: tuple(entries) for key, entries in index.items()}


def _validate_system_uri(value: object) -> str:
    text = _nonempty_text(value, "system_uri")
    if not _URI_SCHEME_RE.match(text) or any(character.isspace() for character in text):
        raise ValueError("system_uri must be an absolute URI with no whitespace")
    return text


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text
