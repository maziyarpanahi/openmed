"""Context-aware candidate ranking stage: generate, merge, and rerank.

This is the orchestration layer that wires the sparse and dense candidate
generators to the hybrid reranker. Given a mention (and optional section /
assertion context) it runs the two-stage retriever, fuses the unioned
candidates through
:func:`openmed.clinical.normalization.ranker.rank_candidates`, and returns the
deterministically ordered, source-attributed
:class:`~openmed.clinical.normalization.ranker.RankedCandidate` list.

The stage does not reimplement scoring: the reciprocal-rank fusion, the
section-match / assertion features, and the per-``(mention, vocab-version)``
cache all live in the reranker module; this stage only composes them. Its
configuration surface -- candidate depth ``k``, ``rerank`` on/off, and the
local ``encoder_path`` -- is exposed through :class:`RankingConfig` so callers
tune the grounding facade without changing its call signature.

Graceful degradation is the invariant: when the dense encoder weights are
absent the dense channel is empty, so fusion runs over the sparse candidates
alone -- but the section / assertion context features still refine that ranking
(only the dense reranking degrades, the context does not). Solely turning
``rerank`` off drops the dense channel *and* the context to yield the pure
sparse lexical order. Either way the stage degrades gracefully rather than
erroring, and retrieval and reranking run fully offline and deterministically.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openmed.clinical.normalization.cache import RankedCandidateCache
from openmed.clinical.normalization.ranker import (
    DEFAULT_RRF_K,
    RankedCandidate,
    rank_candidates,
)

from .embeddings import AliasEncoder, load_encoder
from .retrieval import DEFAULT_RETRIEVAL_K, TwoStageRetriever
from .types import Candidate
from .vocab import FREE_VOCAB_SYSTEMS, VocabLoader, normalize_language

if TYPE_CHECKING:
    from openmed.clinical.context import RerankContext

__all__ = [
    "CandidateRankingStage",
    "RankingConfig",
    "rank_mention",
]


@dataclass(frozen=True)
class RankingConfig:
    """Configuration for the context-aware ranking stage.

    The three roadmap knobs -- candidate depth, rerank on/off, and the local
    encoder path -- are surfaced here so the grounding facade exposes ranker
    configuration without changing its call signature.

    Args:
        k: Per-channel candidate depth requested from each retrieval channel.
        rerank: When ``False`` the stage skips the dense channel and the context
            features, returning the pure sparse-only baseline order; when
            ``True`` it unions dense candidates and folds in section/assertion
            context (subject to an encoder being available).
        encoder_path: Local path to dense encoder weights. Loaded through
            :func:`~openmed.clinical.grounding.embeddings.load_encoder`, which
            returns ``None`` (sparse-only fallback) when the path is unset or the
            weights/backend are unavailable. Never triggers a download.
        systems: Default ordered free-vocabulary systems to search.
        rrf_k: Reciprocal-rank-fusion damping constant forwarded to the reranker.
        source_weights: Optional per-source fusion weights for the reranker.
        backend: Dense index backend selector forwarded to the retriever.
    """

    k: int = DEFAULT_RETRIEVAL_K
    rerank: bool = True
    encoder_path: str | None = None
    systems: tuple[str, ...] = FREE_VOCAB_SYSTEMS
    rrf_k: int = DEFAULT_RRF_K
    source_weights: Mapping[str, float] | None = None
    backend: str = "auto"


class CandidateRankingStage:
    """Run generate -> merge -> rerank for a mention with optional context.

    The stage composes a :class:`~openmed.clinical.grounding.retrieval.TwoStageRetriever`
    with :func:`~openmed.clinical.normalization.ranker.rank_candidates`. It
    resolves the dense encoder once (an explicit ``encoder`` wins over
    ``config.encoder_path``); when neither yields an encoder the dense channel is
    a no-op and ranking degrades to the sparse-only baseline. Ranked lists are
    cached per ``(mention, vocab-version)`` so a document reranks each mention
    once; the vocabulary version is derived from the retrieved candidates'
    content hashes, so a changed terminology edition invalidates the entry.

    Args:
        loader: Vocabulary loader supplying alias indexes; a default
            :class:`VocabLoader` is used when omitted.
        encoder: Explicit dense encoder; overrides ``config.encoder_path``.
        config: Ranking configuration (candidate ``k``, ``rerank`` on/off,
            ``encoder_path``, and reranker/backend options).
        cache: Optional ranked-candidate cache shared across a document.
    """

    def __init__(
        self,
        loader: VocabLoader | None = None,
        *,
        encoder: AliasEncoder | None = None,
        config: RankingConfig | None = None,
        cache: RankedCandidateCache | None = None,
    ) -> None:
        self._config = config if config is not None else RankingConfig()
        self._loader = loader if loader is not None else VocabLoader()
        if encoder is not None:
            self._encoder: AliasEncoder | None = encoder
        elif self._config.encoder_path is not None:
            self._encoder = load_encoder(self._config.encoder_path)
        else:
            self._encoder = None
        self._cache = cache
        self._retriever = TwoStageRetriever(
            self._loader,
            encoder=self._encoder,
            backend=self._config.backend,
        )

    @property
    def config(self) -> RankingConfig:
        """Return the effective ranking configuration."""

        return self._config

    @property
    def rerank_enabled(self) -> bool:
        """Return whether the dense + context rerank path is active.

        ``True`` only when ``config.rerank`` is set and a dense encoder is
        available. When it is ``False`` because no encoder is present, the dense
        channel is skipped but the section / assertion context still refines the
        sparse ranking; only turning ``rerank`` off drops the context too.
        """

        return self._config.rerank and self._encoder is not None

    def rank(
        self,
        mention: str,
        systems: Sequence[str] | None = None,
        *,
        context: "RerankContext | None" = None,
        source_language: str | None = None,
    ) -> tuple[RankedCandidate, ...]:
        """Return the reranked candidates for ``mention``.

        Args:
            mention: Surface span text to ground.
            systems: Ordered free-vocabulary systems to search; defaults to
                ``config.systems``.
            context: Optional section/assertion payload. It is honoured whenever
                ``config.rerank`` is set -- including when no dense encoder is
                present, where it still refines the sparse ranking. With rerank
                off the stage returns the pure sparse baseline and ignores it.
            source_language: Source language selected by the calling router or
                pipeline. Defaults to English and is preserved on serialized
                ranked candidates.

        Returns:
            A deterministically ordered tuple of
            :class:`~openmed.clinical.normalization.ranker.RankedCandidate`.
        """

        ordered_systems = (
            tuple(systems) if systems is not None else self._config.systems
        )
        rerank = self._config.rerank
        resolved_language = normalize_language(source_language)

        if rerank:
            candidates = self._retriever.retrieve(
                mention,
                ordered_systems,
                self._config.k,
                source_language=resolved_language,
            )
            rerank_context = context
        else:
            # Sparse-only baseline: bypass the dense channel and drop context so
            # the fusion reduces to the lexical order.
            candidates = self._retriever.retrieve(
                mention,
                ordered_systems,
                self._config.k,
                include_dense=False,
                source_language=resolved_language,
            )
            rerank_context = None

        vocab_version = _vocab_version(candidates)
        return rank_candidates(
            mention,
            rerank_context,
            candidates,
            rrf_k=self._config.rrf_k,
            source_weights=self._config.source_weights,
            cache=self._cache if vocab_version is not None else None,
            vocab_version=vocab_version,
            source_language=resolved_language,
        )


def rank_mention(
    mention: str,
    systems: Sequence[str] | None = None,
    *,
    context: "RerankContext | None" = None,
    source_language: str | None = None,
    loader: VocabLoader | None = None,
    encoder: AliasEncoder | None = None,
    config: RankingConfig | None = None,
    cache: RankedCandidateCache | None = None,
) -> tuple[RankedCandidate, ...]:
    """Generate, merge, and rerank candidates for one clinical mention.

    Convenience wrapper around :class:`CandidateRankingStage`. See
    :meth:`CandidateRankingStage.rank` for the ordering and degradation
    contract; with no encoder (and no ``config.encoder_path``) the result is the
    sparse-only baseline.
    """

    stage = CandidateRankingStage(
        loader,
        encoder=encoder,
        config=config,
        cache=cache,
    )
    return stage.rank(
        mention,
        systems,
        context=context,
        source_language=source_language,
    )


def _vocab_version(candidates: Sequence[Candidate]) -> str | None:
    """Derive a deterministic vocab version from candidate content hashes.

    Returns a stable hash over the distinct per-candidate ``vocab_version``
    values so the ranked-candidate cache key changes when any contributing
    terminology edition changes. Returns ``None`` when no candidate carries a
    version (nothing to key a cache entry on).
    """

    versions = sorted({c.vocab_version for c in candidates if c.vocab_version})
    if not versions:
        return None
    if len(versions) == 1:
        return versions[0]
    digest = hashlib.sha256("\n".join(versions).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
