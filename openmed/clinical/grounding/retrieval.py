"""Two-stage candidate retrieval: sparse union dense over free vocabularies.

The first stage draws lexical candidates from the character n-gram TF-IDF /
alias generator (``source="sparse"``); the second draws semantic neighbours
from the offline alias embedding index (``source="dense"``). Their outputs are
unioned into one :class:`~openmed.clinical.grounding.types.Candidate` list that
preserves each source's own ranking, so a downstream reranker can fuse the two
channels by reciprocal rank.

Retrieval is fully offline and deterministic: the sparse stage is dependency
free, and the dense stage degrades to an empty channel when no encoder weights
are present, so the union reduces to the sparse-only baseline rather than
raising. No vocabulary content is bundled in the wheel; the caller supplies
terminologies through a :class:`~openmed.clinical.grounding.vocab.VocabLoader`.
"""

from __future__ import annotations

from collections.abc import Sequence

from .candidate_generator import SparseCandidateGenerator
from .embeddings import AliasEncoder
from .index import AliasEmbeddingIndex, DenseCandidateGenerator
from .types import Candidate
from .vocab import FREE_VOCAB_SYSTEMS, VocabLoader

__all__ = [
    "DEFAULT_RETRIEVAL_K",
    "TwoStageRetriever",
    "retrieve_candidates",
]

#: Default per-source candidate depth requested from each retrieval channel.
DEFAULT_RETRIEVAL_K = 10


class TwoStageRetriever:
    """Union sparse lexical and dense semantic candidates for a mention.

    The retriever owns one :class:`SparseCandidateGenerator` and one
    :class:`DenseCandidateGenerator` over a shared vocabulary loader. Each
    :meth:`retrieve` call asks both channels for up to ``k`` candidates and
    concatenates them sparse-first, keeping each channel's internal ranking
    intact (the reranker recovers per-source ranks from that order). When no
    encoder is configured the dense channel yields nothing, so the union is the
    sparse-only baseline.

    Args:
        loader: Vocabulary loader supplying alias indexes; a default
            :class:`VocabLoader` is used when omitted. The wheel ships no
            vocabulary content, so a loader without a user-supplied source
            raises rather than returning bundled data.
        encoder: Alias encoder for the dense channel; ``None`` disables it and
            the retriever degrades to sparse-only.
        index: Pre-built dense index to query; built lazily from ``loader`` and
            ``encoder`` when omitted.
        ngram_size: Character n-gram size for the sparse fuzzy stage.
        min_similarity: Minimum cosine similarity for a sparse fuzzy hit.
        backend: Dense index backend selector forwarded to the index builder.
    """

    def __init__(
        self,
        loader: VocabLoader | None = None,
        *,
        encoder: AliasEncoder | None = None,
        index: AliasEmbeddingIndex | None = None,
        ngram_size: int = 3,
        min_similarity: float = 0.3,
        backend: str = "auto",
    ) -> None:
        self._loader = loader if loader is not None else VocabLoader()
        self._encoder = encoder
        self._sparse = SparseCandidateGenerator(
            self._loader,
            ngram_size=ngram_size,
            min_similarity=min_similarity,
        )
        self._dense = DenseCandidateGenerator(
            encoder,
            index=index,
            loader=self._loader,
            backend=backend,
        )

    @property
    def has_dense_channel(self) -> bool:
        """Return whether a dense encoder is configured for this retriever."""

        return self._encoder is not None

    def retrieve(
        self,
        mention: str,
        systems: Sequence[str] = FREE_VOCAB_SYSTEMS,
        k: int = DEFAULT_RETRIEVAL_K,
        *,
        sparse_k: int | None = None,
        dense_k: int | None = None,
        include_dense: bool = True,
    ) -> list[Candidate]:
        """Return the sparse-then-dense candidate union for ``mention``.

        Args:
            mention: Surface span text to ground.
            systems: Ordered free-vocabulary systems to search.
            k: Per-channel candidate depth used when a channel override is unset.
            sparse_k: Optional override for the sparse channel depth.
            dense_k: Optional override for the dense channel depth.
            include_dense: When ``False`` the dense channel is skipped entirely,
                yielding the sparse-only baseline even if an encoder is
                configured.

        Returns:
            The concatenation of the sparse candidate list followed by the dense
            candidate list. The dense list is empty when no encoder is
            configured (or ``include_dense`` is ``False``), so the union is the
            sparse-only baseline.
        """

        ordered_systems = tuple(systems)
        sparse = self._sparse.generate(
            mention,
            ordered_systems,
            sparse_k if sparse_k is not None else k,
        )
        if self._encoder is None or not include_dense:
            return sparse
        dense = self._dense.generate(
            mention,
            ordered_systems,
            dense_k if dense_k is not None else k,
        )
        return [*sparse, *dense]


def retrieve_candidates(
    mention: str,
    systems: Sequence[str] = FREE_VOCAB_SYSTEMS,
    k: int = DEFAULT_RETRIEVAL_K,
    *,
    loader: VocabLoader | None = None,
    encoder: AliasEncoder | None = None,
    index: AliasEmbeddingIndex | None = None,
    ngram_size: int = 3,
    min_similarity: float = 0.3,
    backend: str = "auto",
    sparse_k: int | None = None,
    dense_k: int | None = None,
) -> list[Candidate]:
    """Union sparse and dense candidates for a clinical mention.

    Convenience wrapper around :class:`TwoStageRetriever`. See
    :meth:`TwoStageRetriever.retrieve` for the union contract; with ``encoder``
    ``None`` the result is the sparse-only baseline.
    """

    retriever = TwoStageRetriever(
        loader,
        encoder=encoder,
        index=index,
        ngram_size=ngram_size,
        min_similarity=min_similarity,
        backend=backend,
    )
    return retriever.retrieve(
        mention,
        systems,
        k,
        sparse_k=sparse_k,
        dense_k=dense_k,
    )
