"""Content-addressed in-memory cache for concept normalization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from openmed.core.result_cache import ResultCache, freeze_value

from .backend import BackendIdentity, TerminologyBackend, normalize_surface

__all__ = [
    "ConceptNormalizationCache",
    "IndexBoundCache",
    "NormalizationCacheStats",
    "RankedCandidateCache",
    "make_index_cache_key",
    "make_normalization_cache_key",
    "make_rerank_cache_key",
]


@runtime_checkable
class IndexBoundCache(Protocol):
    """Cache that can evict entries when its grounding index changes."""

    def bind_index(self, index_key: str) -> bool:
        """Bind to ``index_key`` and return whether stale entries were evicted."""
        ...


@dataclass(frozen=True)
class NormalizationCacheStats:
    """Hit/miss counters for a concept normalization cache."""

    hits: int
    misses: int
    writes: int
    size: int

    @property
    def hit_rate(self) -> float:
        """Return hits divided by cache reads."""

        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class ConceptNormalizationCache:
    """Bounded in-memory cache keyed by mention text and backend identity."""

    def __init__(self, max_entries: int = 1024) -> None:
        self._store = ResultCache(max_entries=max_entries)
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._index_key: str | None = None
        self._invalidations = 0

    @property
    def index_key(self) -> str | None:
        """Grounding index key currently governing the cached results."""

        return self._index_key

    @property
    def invalidation_count(self) -> int:
        """Number of index changes that evicted cached results."""

        return self._invalidations

    def bind_index(self, index_key: str) -> bool:
        """Bind this cache to an index and evict entries when the key drifts."""

        resolved = str(index_key).strip()
        if not resolved:
            raise ValueError("index_key must not be empty")
        if self._index_key == resolved:
            return False

        should_invalidate = self._index_key is not None or len(self._store) > 0
        self._index_key = resolved
        if not should_invalidate:
            return False

        self.clear()
        self._invalidations += 1
        return True

    def get(self, normalized_mention: str, backend: TerminologyBackend) -> Any | None:
        """Return a cached ranking tuple, if present."""

        key = self.key_for(normalized_mention, backend)
        value = self._store.get(key)
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        return value

    def set(
        self,
        normalized_mention: str,
        backend: TerminologyBackend,
        value: Any,
    ) -> None:
        """Store ``value`` for ``normalized_mention`` and ``backend``."""

        key = self.key_for(normalized_mention, backend)
        self._store.set(key, value)
        self._writes += 1

    def key_for(self, normalized_mention: str, backend: TerminologyBackend) -> str:
        """Return the content-addressed cache key for a backend lookup."""

        return make_normalization_cache_key(normalized_mention, backend.identity)

    def clear(self) -> None:
        """Drop cached rankings and reset counters."""

        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._writes = 0

    def stats(self) -> NormalizationCacheStats:
        """Return current cache counters."""

        return NormalizationCacheStats(
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            size=len(self._store),
        )


class RankedCandidateCache:
    """Bounded in-memory cache of reranked candidate lists.

    Keyed by the normalized mention text, the vocabulary version, and a
    ``fingerprint`` of the context (section / assertion / preferred concepts)
    and fusion parameters, so a document reranks each distinct configuration
    once and reuses the result for repeated occurrences — while the same surface
    under a different section or parameter set gets its own entry rather than a
    stale one. Raw mention text is never embedded in the key.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        self._store = ResultCache(max_entries=max_entries)
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._index_key: str | None = None
        self._invalidations = 0

    @property
    def index_key(self) -> str | None:
        """Grounding index key currently governing the cached rankings."""

        return self._index_key

    @property
    def invalidation_count(self) -> int:
        """Number of index changes that evicted cached rankings."""

        return self._invalidations

    def bind_index(self, index_key: str) -> bool:
        """Bind this cache to an index and evict entries when the key drifts."""

        resolved = str(index_key).strip()
        if not resolved:
            raise ValueError("index_key must not be empty")
        if self._index_key == resolved:
            return False

        should_invalidate = self._index_key is not None or len(self._store) > 0
        self._index_key = resolved
        if not should_invalidate:
            return False

        self.clear()
        self._invalidations += 1
        return True

    def get(
        self, mention: str, vocab_version: str, fingerprint: Any = None
    ) -> Any | None:
        """Return a cached ranking for ``(mention, vocab_version, fingerprint)``."""

        value = self._store.get(self.key_for(mention, vocab_version, fingerprint))
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        return value

    def set(
        self, mention: str, vocab_version: str, value: Any, fingerprint: Any = None
    ) -> None:
        """Store ``value`` for ``(mention, vocab_version, fingerprint)``."""

        self._store.set(self.key_for(mention, vocab_version, fingerprint), value)
        self._writes += 1

    def key_for(self, mention: str, vocab_version: str, fingerprint: Any = None) -> str:
        """Return the content-addressed cache key for a rerank result."""

        return make_rerank_cache_key(mention, vocab_version, fingerprint)

    def clear(self) -> None:
        """Drop cached rankings and reset counters."""

        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._writes = 0

    def stats(self) -> NormalizationCacheStats:
        """Return current cache counters."""

        return NormalizationCacheStats(
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            size=len(self._store),
        )


def make_rerank_cache_key(
    mention: str, vocab_version: str, fingerprint: Any = None
) -> str:
    """Return a hashed rerank cache key.

    The key covers the normalized mention, the vocabulary version, and a
    ``fingerprint`` of everything else that determines the ranking (the section
    / assertion / preferred-concept context and the fusion parameters), so a
    shared cache never serves a ranking computed under a different section or
    parameter set for the same surface.
    """

    payload = {
        "normalized_mention": normalize_surface(mention),
        "vocab_version": str(vocab_version),
        "fingerprint": fingerprint,
    }
    serialized = json.dumps(
        freeze_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"grounding-rerank:{digest}"


def make_normalization_cache_key(
    normalized_mention: str,
    backend_identity: BackendIdentity,
) -> str:
    """Return a hashed cache key that never embeds raw mention text."""

    identity = backend_identity.cache_payload()
    payload = {
        "backend": identity,
        "normalized_mention": normalize_surface(normalized_mention),
    }
    serialized = json.dumps(
        freeze_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"concept-normalization:{digest}"


def make_index_cache_key(
    vocab_versions: Mapping[str, str] | Sequence[str],
    encoder_id: str,
    *,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Return a content-addressed cache key for an alias embedding index.

    The key folds together each vocabulary system's content hash (its
    edition/version fingerprint), the encoder id, and the index build
    parameters. Any change to the vocabulary rows, the terminology edition, the
    encoder, or the build parameters yields a different key, so a persisted
    index is rebuilt rather than silently serving codes from a stale edition.
    """

    if isinstance(vocab_versions, Mapping):
        versions = {
            str(system): str(version) for system, version in vocab_versions.items()
        }
    else:
        versions = {str(version): str(version) for version in vocab_versions}
    payload = {
        "vocab_versions": versions,
        "encoder_id": str(encoder_id),
        "params": dict(params or {}),
    }
    serialized = json.dumps(
        freeze_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"grounding-index:{digest}"
