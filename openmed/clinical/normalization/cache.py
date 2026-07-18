"""Content-addressed in-memory cache for concept normalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from openmed.core.result_cache import ResultCache, freeze_value

from .backend import BackendIdentity, TerminologyBackend, normalize_surface

__all__ = [
    "ConceptNormalizationCache",
    "NormalizationCacheStats",
    "make_normalization_cache_key",
]


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
