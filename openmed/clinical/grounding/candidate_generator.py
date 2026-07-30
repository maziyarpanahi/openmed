"""Sparse alias candidate generator over free clinical vocabularies.

Turns a surface mention into ranked :class:`Candidate` objects by unioning
exact normalized-alias hits with character n-gram TF-IDF fuzzy hits over a
user-loaded free vocabulary (RxNorm, ICD-10-CM, LOINC, HPO, MeSH). No
terminology content is bundled in the wheel; the caller supplies vocabularies
through :class:`~openmed.clinical.grounding.vocab.VocabLoader`, and every
emitted candidate carries provenance (``source='sparse'``, the matched alias,
the match kind, and a content hash of the vocabulary snapshot) so downstream
rankers and exporters can audit the source.

The fuzzy stage is a self-contained, dependency-free character trigram TF-IDF
cosine matcher, which keeps candidate generation fully offline and byte-for-byte
deterministic across runs.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from .registry import register_linker
from .types import Candidate
from .vocab import VocabLoader, VocabularyIndex, normalize_alias

__all__ = [
    "SparseCandidateGenerator",
    "generate_candidates",
]

#: Provenance tag stamped on every candidate this generator emits.
SPARSE_SOURCE = "sparse"
#: Registry key the generator registers under (resolved via ``get_linker``).
SPARSE_LINKER_KEY = "sparse"

_DEFAULT_TOP_K = 10
_DEFAULT_NGRAM_SIZE = 3
_DEFAULT_MIN_SIMILARITY = 0.3


def _char_ngrams(text: str, ngram_size: int) -> tuple[str, ...]:
    """Return boundary-padded character n-grams for a normalized surface."""

    padded = f"#{text}#"
    if len(padded) <= ngram_size:
        return (padded,)
    return tuple(
        padded[index : index + ngram_size]
        for index in range(len(padded) - ngram_size + 1)
    )


class _AliasNgramIndex:
    """Character n-gram TF-IDF cosine index over one vocabulary's aliases."""

    def __init__(self, aliases: Sequence[str], ngram_size: int) -> None:
        self._ngram_size = ngram_size
        self._aliases = tuple(aliases)
        document_frequency: Counter[str] = Counter()
        per_alias_grams: list[tuple[str, ...]] = []
        for alias in self._aliases:
            grams = _char_ngrams(alias, ngram_size)
            per_alias_grams.append(grams)
            document_frequency.update(set(grams))
        document_count = len(self._aliases)
        self._idf = {
            gram: math.log((1.0 + document_count) / (1.0 + frequency)) + 1.0
            for gram, frequency in document_frequency.items()
        }
        self._vectors = tuple(self._vectorize(grams) for grams in per_alias_grams)

    def _vectorize(self, grams: Iterable[str]) -> dict[str, float]:
        weights = {
            gram: count * self._idf.get(gram, 0.0)
            for gram, count in Counter(grams).items()
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        if norm == 0.0:
            return {}
        return {gram: weight / norm for gram, weight in weights.items()}

    def query(self, text: str, *, min_similarity: float) -> list[tuple[str, float]]:
        """Return ``(alias, cosine_similarity)`` pairs above ``min_similarity``."""

        query_vector = self._vectorize(_char_ngrams(text, self._ngram_size))
        if not query_vector:
            return []
        results: list[tuple[str, float]] = []
        for alias, vector in zip(self._aliases, self._vectors):
            similarity = sum(
                weight * vector[gram]
                for gram, weight in query_vector.items()
                if gram in vector
            )
            if similarity >= min_similarity:
                results.append((alias, similarity))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results


class SparseCandidateGenerator:
    """Emit provenance-carrying ``Candidate`` objects for a clinical mention.

    Candidate generation unions two deterministic stages over each requested
    free-vocabulary system: exact normalized-alias hits (score ``1.0``,
    ``match_kind='exact'``) and character n-gram TF-IDF fuzzy hits (cosine
    similarity, ``match_kind='fuzzy'``). Results are merged across systems,
    de-duplicated per ``(system, code)`` keeping the strongest score, and
    ordered by a stable tie-break (score descending, then system priority, then
    concept id).

    Note:
        This is a candidate *generator* — its shape (a ``VocabLoader`` constructor
        and a ``generate(mention, systems, k)`` method) intentionally differs from
        the ``VocabLinker`` registry entries (constructed with a single
        ``VocabularyIndex`` and exposing ``.link(...)``). Consume it via
        ``generate_candidates`` or ``.generate``; it is not compatible with the
        uniform ``get_linker(system)(get_index(system)).link(...)`` call path.

    Args:
        loader: Vocabulary loader supplying alias indexes. A default
            :class:`~openmed.clinical.grounding.vocab.VocabLoader` is used when
            omitted; the wheel ships no vocabulary content, so a loader without
            a user-supplied source raises rather than returning bundled data.
        ngram_size: Character n-gram size for the fuzzy stage (default ``3``).
        min_similarity: Minimum cosine similarity for a fuzzy hit.
    """

    #: Provenance tag emitted on every candidate.
    source = SPARSE_SOURCE

    def __init__(
        self,
        loader: VocabLoader | None = None,
        *,
        ngram_size: int = _DEFAULT_NGRAM_SIZE,
        min_similarity: float = _DEFAULT_MIN_SIMILARITY,
    ) -> None:
        if ngram_size < 1:
            raise ValueError("ngram_size must be a positive integer")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")
        self._loader = loader if loader is not None else VocabLoader()
        self._ngram_size = ngram_size
        self._min_similarity = min_similarity
        self._ngram_indexes: dict[str, _AliasNgramIndex] = {}

    def generate(
        self,
        mention: str,
        systems: Sequence[str],
        k: int = _DEFAULT_TOP_K,
        *,
        min_similarity: float | None = None,
    ) -> list[Candidate]:
        """Return up to ``k`` ranked candidates for ``mention``.

        Args:
            mention: Surface span text to ground.
            systems: Ordered free-vocabulary systems to search; earlier systems
                win ties. Must be non-empty.
            k: Maximum number of candidates to return.
            min_similarity: Override for the instance fuzzy cutoff.

        Raises:
            ValueError: If ``systems`` is empty or ``k`` is not positive.
            VocabLoaderError: If a requested system is unknown or has no source.
        """

        if not isinstance(mention, str):
            raise TypeError("mention must be a string")
        ordered_systems = tuple(systems)
        if not ordered_systems:
            raise ValueError("systems must contain at least one vocabulary system")
        if k <= 0:
            raise ValueError("k must be a positive integer")
        cutoff = self._min_similarity if min_similarity is None else min_similarity

        query = normalize_alias(mention)
        if not query:
            return []

        priority = {system: rank for rank, system in enumerate(ordered_systems)}
        best: dict[tuple[str, str], Candidate] = {}
        for system in ordered_systems:
            index = self._loader.get_index(system)
            for candidate in self._candidates_for_system(index, query, cutoff):
                key = (candidate.system, candidate.code)
                existing = best.get(key)
                if existing is None or candidate.score > existing.score:
                    best[key] = candidate

        ranked = sorted(
            best.values(),
            key=lambda candidate: (
                -candidate.score,
                priority.get(_priority_system(candidate.system, ordered_systems), 0),
                candidate.code,
            ),
        )
        return ranked[:k]

    def _candidates_for_system(
        self, index: VocabularyIndex, query: str, cutoff: float
    ) -> list[Candidate]:
        system = index.system.upper()
        version = index.content_hash
        best: dict[str, Candidate] = {}

        for concept in index.lookup_all(query):
            best[concept.code] = Candidate(
                system=system,
                code=concept.code,
                display=concept.preferred_term,
                score=1.0,
                source=self.source,
                matched_alias=query,
                match_kind="exact",
                vocab_version=version,
            )

        for alias, similarity in self._ngram_index(index).query(
            query, min_similarity=cutoff
        ):
            score = round(float(similarity), 6)
            for concept in index.lookup_all(alias):
                existing = best.get(concept.code)
                if existing is not None and existing.score >= score:
                    continue
                best[concept.code] = Candidate(
                    system=system,
                    code=concept.code,
                    display=concept.preferred_term,
                    score=score,
                    source=self.source,
                    matched_alias=alias,
                    match_kind="fuzzy",
                    vocab_version=version,
                )
        return list(best.values())

    def _ngram_index(self, index: VocabularyIndex) -> _AliasNgramIndex:
        cached = self._ngram_indexes.get(index.system)
        if cached is None:
            cached = _AliasNgramIndex(index.aliases, self._ngram_size)
            self._ngram_indexes[index.system] = cached
        return cached


def _priority_system(candidate_system: str, ordered_systems: Sequence[str]) -> str:
    """Resolve an emitted (upper-cased) system back to its request key."""

    for system in ordered_systems:
        if system.upper() == candidate_system.upper():
            return system
    return candidate_system


def generate_candidates(
    mention: str,
    systems: Sequence[str],
    k: int = _DEFAULT_TOP_K,
    *,
    loader: VocabLoader | None = None,
    ngram_size: int = _DEFAULT_NGRAM_SIZE,
    min_similarity: float = _DEFAULT_MIN_SIMILARITY,
) -> list[Candidate]:
    """Generate ranked sparse ``Candidate`` objects for a clinical mention.

    Convenience wrapper around :class:`SparseCandidateGenerator`. See
    :meth:`SparseCandidateGenerator.generate` for the ranking contract.
    """

    generator = SparseCandidateGenerator(
        loader,
        ngram_size=ngram_size,
        min_similarity=min_similarity,
    )
    return generator.generate(mention, systems, k)


register_linker(SPARSE_LINKER_KEY, SparseCandidateGenerator)
