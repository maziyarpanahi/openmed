"""Deterministic candidate generation, ranking, and synthetic evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from .backend import (
    SYNTHETIC_CONCEPTS,
    BackendIdentity,
    TerminologyBackend,
    TerminologyConcept,
    normalize_surface,
    validate_backend_identity,
)
from .cache import ConceptNormalizationCache

__all__ = [
    "CandidateProvenance",
    "ConceptNormalizer",
    "NormalizationEvaluationResult",
    "NormalizationGoldCase",
    "RankedConcept",
    "SYNTHETIC_GOLD_SET",
    "evaluate_normalization_gold",
    "generate_query_variants",
]


DEFAULT_ABBREVIATION_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "af": ("aster fever",),
    "bc": ("beryl cough",),
    "esp": ("elin sugar panel",),
    "fbs": ("faren breath score",),
    "grcc": ("galen red cell count",),
    "hps": ("halo pain scale",),
    "isc": ("iona sleep coaching",),
    "jbt": ("juno blue tablet",),
}


@dataclass(frozen=True)
class CandidateProvenance:
    """Provenance carried with one ranked concept candidate."""

    mention_start: int | None
    mention_end: int | None
    matched_term_hash: str
    backend_name: str
    backend_version: str
    query_variant_count: int


@dataclass(frozen=True)
class RankedConcept:
    """A coded concept ranked for a mention."""

    concept: TerminologyConcept
    confidence: float
    score: float
    features: tuple[tuple[str, float], ...]
    provenance: CandidateProvenance

    @property
    def feature_map(self) -> dict[str, float]:
        """Return ranking features as a JSON-friendly mapping."""

        return dict(self.features)


@dataclass(frozen=True)
class NormalizationGoldCase:
    """Synthetic gold case used by CI-gated normalization evaluation."""

    mention: str
    expected_system_uri: str
    expected_code: str
    start: int | None = None
    end: int | None = None

    @property
    def expected_key(self) -> tuple[str, str]:
        return (self.expected_system_uri, self.expected_code)


@dataclass(frozen=True)
class NormalizationEvaluationResult:
    """Accuracy and cache metrics for a concept-normalization run."""

    case_count: int
    top1_accuracy: float
    top5_accuracy: float
    cache_hit_rate: float


SYNTHETIC_GOLD_SET: tuple[NormalizationGoldCase, ...] = (
    NormalizationGoldCase(
        mention="Aster pyrexia",
        expected_system_uri=SYNTHETIC_CONCEPTS[0].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[0].code,
        start=4,
        end=17,
    ),
    NormalizationGoldCase(
        mention="AF",
        expected_system_uri=SYNTHETIC_CONCEPTS[0].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[0].code,
        start=21,
        end=23,
    ),
    NormalizationGoldCase(
        mention="beryl cough",
        expected_system_uri=SYNTHETIC_CONCEPTS[1].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[1].code,
        start=8,
        end=19,
    ),
    NormalizationGoldCase(
        mention="skin flare corin",
        expected_system_uri=SYNTHETIC_CONCEPTS[2].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[2].code,
        start=0,
        end=16,
    ),
    NormalizationGoldCase(
        mention="dax ankle sprain",
        expected_system_uri=SYNTHETIC_CONCEPTS[3].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[3].code,
        start=5,
        end=21,
    ),
    NormalizationGoldCase(
        mention="ESP",
        expected_system_uri=SYNTHETIC_CONCEPTS[4].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[4].code,
        start=2,
        end=5,
    ),
    NormalizationGoldCase(
        mention="faren breathing score",
        expected_system_uri=SYNTHETIC_CONCEPTS[5].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[5].code,
        start=11,
        end=33,
    ),
    NormalizationGoldCase(
        mention="galen rcc",
        expected_system_uri=SYNTHETIC_CONCEPTS[6].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[6].code,
        start=12,
        end=21,
    ),
    NormalizationGoldCase(
        mention="halo pain rating",
        expected_system_uri=SYNTHETIC_CONCEPTS[7].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[7].code,
        start=0,
        end=16,
    ),
    NormalizationGoldCase(
        mention="jbt",
        expected_system_uri=SYNTHETIC_CONCEPTS[9].system_uri,
        expected_code=SYNTHETIC_CONCEPTS[9].code,
        start=40,
        end=43,
    ),
)


class ConceptNormalizer:
    """Normalize clinical mention strings to ranked coded concepts."""

    def __init__(
        self,
        backend: TerminologyBackend,
        *,
        cache: ConceptNormalizationCache | None = None,
        abbreviation_expansions: Mapping[str, str | Sequence[str]] | None = None,
        max_candidates: int = 10,
        max_ngram: int = 4,
    ) -> None:
        self.backend = backend
        self.identity = validate_backend_identity(backend.identity)
        self.cache = cache
        self.max_candidates = max_candidates
        self.max_ngram = max_ngram
        self.abbreviation_expansions = _normalize_expansions(
            abbreviation_expansions or DEFAULT_ABBREVIATION_EXPANSIONS
        )

    def normalize(
        self,
        mention: str,
        *,
        start: int | None = None,
        end: int | None = None,
        use_cache: bool = True,
    ) -> tuple[RankedConcept, ...]:
        """Return ranked coded candidates for ``mention``."""

        _validate_offsets(start, end)
        normalized = normalize_surface(mention)
        if self.cache is not None and use_cache:
            cached = self.cache.get(normalized, self.backend)
            if cached is not None:
                return cached

        ranked = self._rank_uncached(
            normalized=normalized,
            start=start,
            end=end,
        )
        if self.cache is not None and use_cache:
            self.cache.set(normalized, self.backend, ranked)
        return ranked

    def _rank_uncached(
        self,
        *,
        normalized: str,
        start: int | None,
        end: int | None,
    ) -> tuple[RankedConcept, ...]:
        query_variants = generate_query_variants(
            normalized,
            abbreviation_expansions=self.abbreviation_expansions,
            max_ngram=self.max_ngram,
        )
        candidates: dict[tuple[str, str], TerminologyConcept] = {}
        for query in query_variants:
            for concept in self.backend.lookup(query):
                candidates.setdefault(concept.key, concept)
        for query in query_variants:
            for concept in self.backend.candidates(query.split()):
                candidates.setdefault(concept.key, concept)

        ranked = [
            _rank_candidate(
                concept=concept,
                normalized_mention=normalized,
                identity=self.identity,
                start=start,
                end=end,
                query_variants=query_variants,
            )
            for concept in candidates.values()
        ]
        return tuple(sorted(ranked, key=_rank_sort_key)[: self.max_candidates])


def generate_query_variants(
    mention: str,
    *,
    abbreviation_expansions: Mapping[str, Sequence[str]] | None = None,
    max_ngram: int = 4,
) -> tuple[str, ...]:
    """Return exact, abbreviation-expanded, and n-gram query variants."""

    normalized = normalize_surface(mention)
    tokens = normalized.split()
    variants: list[str] = [normalized] if normalized else []
    expansions = abbreviation_expansions or {}

    expanded_phrases = _expanded_phrases(tokens, expansions)
    variants.extend(expanded_phrases)

    for phrase in (normalized, *expanded_phrases):
        phrase_tokens = phrase.split()
        for ngram in _ngrams(phrase_tokens, max_ngram=max_ngram):
            variants.append(ngram)

    return _unique_ordered(variants)


def evaluate_normalization_gold(
    normalizer: ConceptNormalizer,
    gold_cases: Sequence[NormalizationGoldCase] = SYNTHETIC_GOLD_SET,
    *,
    repeated_workload_repeats: int = 2,
) -> NormalizationEvaluationResult:
    """Evaluate top-k accuracy and cache hit-rate on synthetic gold cases."""

    if not gold_cases:
        raise ValueError("gold_cases must not be empty")

    top1_hits = 0
    top5_hits = 0
    for case in gold_cases:
        ranked = normalizer.normalize(case.mention, start=case.start, end=case.end)
        keys = [candidate.concept.key for candidate in ranked]
        if keys[:1] == [case.expected_key]:
            top1_hits += 1
        if case.expected_key in keys[:5]:
            top5_hits += 1

    for _ in range(repeated_workload_repeats):
        for case in gold_cases:
            normalizer.normalize(case.mention, start=case.start, end=case.end)

    cache_hit_rate = 0.0
    if normalizer.cache is not None:
        cache_hit_rate = normalizer.cache.stats().hit_rate

    return NormalizationEvaluationResult(
        case_count=len(gold_cases),
        top1_accuracy=top1_hits / len(gold_cases),
        top5_accuracy=top5_hits / len(gold_cases),
        cache_hit_rate=cache_hit_rate,
    )


def _rank_candidate(
    *,
    concept: TerminologyConcept,
    normalized_mention: str,
    identity: BackendIdentity,
    start: int | None,
    end: int | None,
    query_variants: tuple[str, ...],
) -> RankedConcept:
    terms = concept.normalized_terms
    best_term = max(
        terms,
        key=lambda term: _term_score(normalized_mention, query_variants, term),
    )
    features = _feature_values(normalized_mention, query_variants, best_term)
    score = _weighted_score(features)
    confidence = round(max(0.0, min(1.0, score)), 6)
    return RankedConcept(
        concept=concept,
        confidence=confidence,
        score=round(score, 6),
        features=tuple(sorted(features.items())),
        provenance=CandidateProvenance(
            mention_start=start,
            mention_end=end,
            matched_term_hash=_hash_text(best_term),
            backend_name=identity.name,
            backend_version=identity.version,
            query_variant_count=len(query_variants),
        ),
    )


def _feature_values(
    normalized_mention: str,
    query_variants: Sequence[str],
    term: str,
) -> dict[str, float]:
    mention_tokens = set(normalized_mention.split())
    term_tokens = set(term.split())
    overlap = 0.0
    if mention_tokens or term_tokens:
        overlap = len(mention_tokens & term_tokens) / max(
            len(mention_tokens | term_tokens),
            1,
        )

    exact = float(term in query_variants or normalized_mention == term)
    char_similarity = SequenceMatcher(None, normalized_mention, term).ratio()
    acronym = float(_acronym(term) == normalized_mention and bool(normalized_mention))
    expanded_exact = float(term in query_variants and normalized_mention != term)
    length_fit = 1.0 - (
        abs(len(normalized_mention.split()) - len(term.split()))
        / max(len(normalized_mention.split()), len(term.split()), 1)
    )
    return {
        "acronym": acronym,
        "char_similarity": char_similarity,
        "exact": exact,
        "expanded_exact": expanded_exact,
        "length_fit": length_fit,
        "token_overlap": overlap,
    }


def _weighted_score(features: Mapping[str, float]) -> float:
    return (
        0.72 * features["exact"]
        + 0.12 * features["token_overlap"]
        + 0.08 * features["char_similarity"]
        + 0.04 * features["length_fit"]
        + 0.08 * features["expanded_exact"]
        + 0.04 * features["acronym"]
    )


def _term_score(
    normalized_mention: str,
    query_variants: Sequence[str],
    term: str,
) -> tuple[float, str]:
    features = _feature_values(normalized_mention, query_variants, term)
    return (_weighted_score(features), term)


def _rank_sort_key(candidate: RankedConcept) -> tuple[float, str, str, str]:
    return (
        -candidate.score,
        candidate.concept.system_uri,
        candidate.concept.code,
        candidate.concept.display,
    )


def _normalize_expansions(
    expansions: Mapping[str, str | Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for abbreviation, values in expansions.items():
        key = normalize_surface(abbreviation)
        if isinstance(values, str):
            raw_values = (values,)
        else:
            raw_values = tuple(values)
        normalized[key] = tuple(
            value for value in (normalize_surface(item) for item in raw_values) if value
        )
    return normalized


def _expanded_phrases(
    tokens: Sequence[str],
    expansions: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    phrases: list[str] = []
    for index, token in enumerate(tokens):
        for expansion in expansions.get(token, ()):
            replaced = [*tokens]
            replaced[index : index + 1] = expansion.split()
            phrases.append(" ".join(replaced))
    return _unique_ordered(phrases)


def _ngrams(tokens: Sequence[str], *, max_ngram: int) -> tuple[str, ...]:
    result: list[str] = []
    max_size = min(max_ngram, len(tokens))
    for size in range(max_size, 0, -1):
        for start in range(0, len(tokens) - size + 1):
            result.append(" ".join(tokens[start : start + size]))
    return tuple(result)


def _acronym(term: str) -> str:
    return "".join(token[:1] for token in term.split() if token)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique_ordered(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _validate_offsets(start: int | None, end: int | None) -> None:
    if start is None and end is None:
        return
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("start and end offsets must be provided together")
    if start < 0 or end < start:
        raise ValueError("start/end offsets must be non-negative and ordered")
