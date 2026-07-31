"""Deterministic candidate generation, ranking, and synthetic evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from .backend import (
    SYNTHETIC_CONCEPTS,
    BackendIdentity,
    TerminologyBackend,
    TerminologyConcept,
    normalize_surface,
    validate_backend_identity,
)
from .cache import ConceptNormalizationCache, RankedCandidateCache

if TYPE_CHECKING:
    from openmed.clinical.context import RerankContext
    from openmed.clinical.grounding.types import Candidate

__all__ = [
    "CandidateProvenance",
    "ConceptNormalizer",
    "DEFAULT_RRF_K",
    "DEFAULT_SOURCE_WEIGHTS",
    "NormalizationEvaluationResult",
    "NormalizationGoldCase",
    "RankedCandidate",
    "RankedConcept",
    "SYNTHETIC_GOLD_SET",
    "SourceContribution",
    "evaluate_normalization_gold",
    "generate_query_variants",
    "rank_candidates",
]

#: Reciprocal-rank-fusion damping constant. The documented default follows the
#: standard RRF recommendation and keeps single-rank differences small so a
#: concept ranked well by two sources outranks one ranked highly by only one.
DEFAULT_RRF_K = 60
#: Default per-source fusion weights. Sparse (lexical) and dense (semantic)
#: channels contribute equally; callers may reweight per source.
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {"sparse": 1.0, "dense": 1.0}
#: Additive bonus applied when a candidate matches the section's preferred
#: concepts. Sized as one RRF step so it resolves same-surface collisions
#: (near-tied fused scores) without overriding a clear multi-rank lead.
DEFAULT_SECTION_WEIGHT = 1.0 / DEFAULT_RRF_K
#: Additive weight for the assertion-axis feature. The feature is uniform across
#: a mention's candidates, so it is recorded for audit without reordering them.
DEFAULT_ASSERTION_WEIGHT = 0.01
#: Source tag used when a candidate carries no explicit ``source``.
_UNKNOWN_SOURCE = "unknown"


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


@dataclass(frozen=True)
class SourceContribution:
    """One source's contribution to a fused candidate score.

    ``source`` names the generator ("sparse" or "dense"), ``rank`` is the
    candidate's zero-based position within that source's list, ``weight`` is the
    source's fusion weight, and ``rrf`` is the reciprocal-rank-fusion term
    ``weight / (rrf_k + rank + 1)`` that this source added to the fused score.
    """

    source: str
    rank: int
    weight: float
    rrf: float


@dataclass(frozen=True)
class RankedCandidate:
    """A reranked grounding candidate with fused score and source attribution.

    ``candidate`` is the winning :class:`~openmed.clinical.grounding.types.Candidate`
    for its ``(system, code)`` key, ``fused_score`` is the reciprocal-rank-fusion
    score plus context adjustments, ``contributions`` records each source's
    per-rank term, ``features`` carries the context feature values folded into
    the score, and ``sources`` lists which retrieval sources returned the
    concept (its cross-source attribution).
    """

    candidate: "Candidate"
    fused_score: float
    contributions: tuple[SourceContribution, ...]
    features: tuple[tuple[str, float], ...]
    sources: tuple[str, ...]

    @property
    def concept_key(self) -> tuple[str, str]:
        """Return the ``(system, code)`` identity of the ranked concept."""

        return (self.candidate.system, self.candidate.code)

    @property
    def feature_map(self) -> dict[str, float]:
        """Return the context features as a JSON-friendly mapping."""

        return dict(self.features)


def _rerank_fingerprint(
    context: "RerankContext | None",
    weights: Mapping[str, float],
    rrf_k: int,
    section_weight: float,
    assertion_weight: float,
) -> dict[str, Any]:
    """Identity of everything besides the mention/vocab that determines a rerank
    result: the context (section, assertion, preferred concepts) and the fusion
    parameters. Folded into the cache key so a shared cache never serves a
    ranking computed under a different section or parameter set. Sorted
    throughout to stay byte-for-byte deterministic across runs."""

    ctx: dict[str, Any] | None = None
    if context is not None:
        ctx = {
            "section": context.canonical_section,
            "assertion": (
                context.assertion.to_dict() if context.assertion is not None else None
            ),
            "preferred": sorted(context.preferred_concepts),
        }
    return {
        "context": ctx,
        "rrf_k": rrf_k,
        "weights": sorted((str(key), float(value)) for key, value in weights.items()),
        "section_weight": float(section_weight),
        "assertion_weight": float(assertion_weight),
    }


def rank_candidates(
    mention: str,
    context: "RerankContext | None",
    candidates: "Iterable[Candidate]",
    *,
    rrf_k: int = DEFAULT_RRF_K,
    source_weights: Mapping[str, float] | None = None,
    section_weight: float = DEFAULT_SECTION_WEIGHT,
    assertion_weight: float = DEFAULT_ASSERTION_WEIGHT,
    cache: RankedCandidateCache | None = None,
    vocab_version: str | None = None,
) -> tuple[RankedCandidate, ...]:
    """Fuse sparse and dense candidates into a reranked concept list.

    Candidates are partitioned by their ``source`` tag ("sparse"/"dense"),
    ranked within each source by the order given, and fused with reciprocal-rank
    fusion: a concept's fused score is the sum over the sources that returned it
    of ``weight / (rrf_k + rank + 1)``. Two optional context features from
    ``context`` are added: a section-match bonus (``section_weight``) when the
    concept is preferred in the current section — the signal that resolves a
    same-surface collision to the section-appropriate sense — and a uniform
    assertion-present adjustment (``assertion_weight``) recorded for audit.

    Concepts are de-duplicated per ``(system, code)`` keeping the strongest
    original candidate, then ordered deterministically by fused score descending
    with a stable ``(system, code)`` tie-break, so the output is byte-for-byte
    reproducible across runs. When no dense candidates are present the result
    reduces to the sparse-only order: fusion over a single source preserves that
    source's ranking, so ranking degrades gracefully to the lexical baseline.

    Args:
        mention: Surface span text being grounded (used only for cache keying).
        context: Optional section/assertion payload; ``None`` disables the
            context features (pure sparse+dense fusion).
        candidates: Merged sparse and/or dense candidates. Each source's
            candidates are consumed in the order given as that source's ranking.
        rrf_k: Reciprocal-rank-fusion damping constant.
        source_weights: Per-source fusion weights; merged over the defaults.
        section_weight: Weight of the section-match feature.
        assertion_weight: Weight of the assertion-present feature.
        cache: Optional cache; a hit for the same mention, vocab version,
            context, and fusion parameters is returned without recomputation.
        vocab_version: Vocabulary version for cache keying; required to use
            ``cache``.

    Returns:
        A deterministically ordered tuple of :class:`RankedCandidate`.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    weights = {**DEFAULT_SOURCE_WEIGHTS, **(source_weights or {})}

    use_cache = cache is not None and vocab_version is not None
    fingerprint = (
        _rerank_fingerprint(context, weights, rrf_k, section_weight, assertion_weight)
        if use_cache
        else None
    )
    if use_cache:
        cached = cache.get(mention, vocab_version, fingerprint)  # type: ignore[union-attr, arg-type]
        if cached is not None:
            return cached

    # Per-source rank (first occurrence wins) and the strongest candidate per key.
    per_source_rank: dict[str, dict[tuple[str, str], int]] = {}
    best_candidate: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.system, candidate.code)
        source = candidate.source or _UNKNOWN_SOURCE
        ranks = per_source_rank.setdefault(source, {})
        if key not in ranks:
            ranks[key] = len(ranks)
        previous = best_candidate.get(key)
        if previous is None or _candidate_preference(candidate) > _candidate_preference(
            previous
        ):
            best_candidate[key] = candidate

    assertion_feature = context.assertion_present() if context is not None else 1.0

    ranked: list[RankedCandidate] = []
    for key, candidate in best_candidate.items():
        contributions: list[SourceContribution] = []
        rrf_total = 0.0
        for source in sorted(per_source_rank):
            ranks = per_source_rank[source]
            if key not in ranks:
                continue
            weight = weights.get(source, 1.0)
            rank = ranks[key]
            rrf = weight / (rrf_k + rank + 1)
            rrf_total += rrf
            contributions.append(
                SourceContribution(
                    source=source,
                    rank=rank,
                    weight=weight,
                    rrf=round(rrf, 9),
                )
            )
        section_feature = (
            context.section_match(candidate.system, candidate.code)
            if context is not None
            else 0.0
        )
        fused = (
            rrf_total
            + section_weight * section_feature
            + assertion_weight * assertion_feature
        )
        features = (
            ("assertion_present", assertion_feature),
            ("rrf", round(rrf_total, 9)),
            ("section_match", section_feature),
        )
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                fused_score=round(fused, 9),
                contributions=tuple(contributions),
                features=features,
                sources=tuple(contribution.source for contribution in contributions),
            )
        )

    ranked.sort(key=_rerank_sort_key)
    result = tuple(ranked)
    if use_cache:
        cache.set(mention, vocab_version, result, fingerprint)  # type: ignore[union-attr, arg-type]
    return result


def _candidate_preference(candidate: "Candidate") -> tuple[float, int, int]:
    """Order key selecting the representative candidate for a concept key."""

    exact = 1 if candidate.match_kind == "exact" else 0
    sparse = 1 if candidate.source == "sparse" else 0
    return (candidate.score, exact, sparse)


def _rerank_sort_key(candidate: RankedCandidate) -> tuple[float, str, str]:
    return (
        -candidate.fused_score,
        candidate.candidate.system,
        candidate.candidate.code,
    )
