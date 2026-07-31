"""Offline embedding index over concept aliases keyed by vocabulary version.

Dense candidate generation needs an on-device nearest-neighbor index over the
alias embeddings of a free clinical vocabulary. This module builds that index
from a caller-supplied :class:`~openmed.clinical.grounding.vocab.VocabLoader`
and an :class:`~openmed.clinical.grounding.embeddings.AliasEncoder`, then serves
top-k neighbors as provenance-carrying :class:`Candidate` objects
(``source='dense'``).

Key properties:

* **Offline and deterministic.** No network I/O and no bundled terminology; the
  brute-force reference path and the persisted payload are byte-for-byte
  reproducible across runs.
* **Version-keyed.** The persisted index carries a content hash of the
  vocabulary editions and the encoder id. On load, a drift in any input changes
  the key and forces a rebuild, so a stale terminology edition is never served
  silently.
* **Optional ANN backend.** An HNSW backend (``hnswlib``) is used when
  available; otherwise a pure brute-force cosine search returns exact neighbors.
* **Graceful fallback.** When no encoder weights are present, :func:`build_index`
  returns ``None`` and :func:`query_index` returns ``[]`` so callers fall back to
  sparse-only retrieval rather than raising.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openmed.clinical.normalization.cache import make_index_cache_key

from .embeddings import AliasEncoder
from .registry import register_linker
from .types import Candidate
from .vocab import VocabLoader, VocabularyIndex, normalize_alias, normalize_language

__all__ = [
    "AliasEmbeddingIndex",
    "DenseCandidateGenerator",
    "IndexBackendUnavailableError",
    "brute_force_neighbors",
    "build_index",
    "build_or_load_index",
    "load_index",
    "query_index",
]

#: Provenance tag stamped on every candidate this index emits.
DENSE_SOURCE = "dense"
#: Match kind recorded on dense neighbors.
DENSE_MATCH_KIND = "dense"
#: Registry key the dense generator registers under.
DENSE_LINKER_KEY = "dense"
#: Persisted payload schema version.
INDEX_SCHEMA_VERSION = 1
#: Persisted payload filename inside a cache directory.
INDEX_FILENAME = "alias_index.json"

_DEFAULT_TOP_K = 10
_HNSW_EF_CONSTRUCTION = 200
_HNSW_M = 16


class IndexBackendUnavailableError(RuntimeError):
    """Raised when an ANN backend is required but not installed."""


@dataclass(frozen=True)
class _AliasRecord:
    """One embedded (concept, alias) pair with its provenance."""

    system: str
    code: str
    display: str
    matched_alias: str
    vocab_version: str


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two equal-length vectors."""

    return sum(a * b for a, b in zip(left, right))


def brute_force_neighbors(
    vectors: Sequence[Sequence[float]],
    query: Sequence[float],
    k: int,
) -> list[tuple[int, float]]:
    """Return the exact top-``k`` ``(row, similarity)`` pairs by cosine.

    This is the pure reference path against which the ANN backend is measured.
    Ties break on the lower row index so ordering is deterministic.
    """

    scored = [(index, _cosine(vector, query)) for index, vector in enumerate(vectors)]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:k]


class AliasEmbeddingIndex:
    """Nearest-neighbor index over vocabulary alias embeddings.

    Instances are built by :func:`build_index` and queried with a mention vector
    to yield ranked dense :class:`Candidate` objects. The index carries its
    provenance (per-system vocabulary content hashes and the encoder id) and a
    content-addressed ``index_key`` so persisted copies can be validated against
    the current vocabulary edition on load.
    """

    def __init__(
        self,
        *,
        encoder_id: str,
        dimension: int,
        backend: str,
        index_key: str,
        vocab_versions: dict[str, str],
        records: Sequence[_AliasRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self.encoder_id = encoder_id
        self.dimension = dimension
        self.backend = backend
        self.index_key = index_key
        self.vocab_versions = dict(sorted(vocab_versions.items()))
        self._records = tuple(records)
        self._vectors = tuple(
            tuple(float(value) for value in vector) for vector in vectors
        )
        self._ann = _maybe_build_ann(backend, dimension, self._vectors)

    @property
    def record_count(self) -> int:
        """Number of embedded (concept, alias) rows in the index."""

        return len(self._records)

    @property
    def provenance(self) -> dict[str, object]:
        """Audit record persisted alongside the index."""

        return {
            "index_key": self.index_key,
            "encoder_id": self.encoder_id,
            "dimension": self.dimension,
            "backend": self.backend,
            "vocab_versions": dict(self.vocab_versions),
            "record_count": self.record_count,
            "source": DENSE_SOURCE,
        }

    def query(
        self,
        vector: Sequence[float],
        k: int = _DEFAULT_TOP_K,
        *,
        source_language: str | None = None,
    ) -> list[Candidate]:
        """Return up to ``k`` ranked dense candidates for a mention ``vector``.

        Neighbors are de-duplicated per ``(system, code)`` keeping the strongest
        similarity, then ordered by score descending with a stable tie-break on
        the concept code.
        """

        if k <= 0:
            raise ValueError("k must be a positive integer")
        if len(vector) != self.dimension:
            raise ValueError(
                f"query vector has dimension {len(vector)}, expected {self.dimension}"
            )
        if not self._records:
            return []

        # Over-fetch neighbors so per-concept de-duplication can still fill k.
        fetch = min(len(self._records), max(k * 4, k))
        neighbors = self._neighbors(vector, fetch)

        resolved_language = normalize_language(source_language)
        best: dict[tuple[str, str], Candidate] = {}
        for row, similarity in neighbors:
            record = self._records[row]
            score = round(max(0.0, float(similarity)), 6)
            key = (record.system, record.code)
            existing = best.get(key)
            if existing is not None and existing.score >= score:
                continue
            best[key] = Candidate(
                system=record.system,
                code=record.code,
                display=record.display,
                score=score,
                source_language=resolved_language,
                source=DENSE_SOURCE,
                matched_alias=record.matched_alias,
                match_kind=DENSE_MATCH_KIND,
                vocab_version=record.vocab_version,
            )
        ranked = sorted(
            best.values(),
            key=lambda candidate: (-candidate.score, candidate.code),
        )
        return ranked[:k]

    def query_text(
        self,
        mention: str,
        encoder: AliasEncoder,
        k: int = _DEFAULT_TOP_K,
        *,
        source_language: str | None = None,
    ) -> list[Candidate]:
        """Encode ``mention`` with ``encoder`` and return dense candidates."""

        (vector,) = encoder.encode([mention])
        return self.query(vector, k, source_language=source_language)

    def _neighbors(
        self, vector: Sequence[float], fetch: int
    ) -> list[tuple[int, float]]:
        if self._ann is not None:
            return self._ann_neighbors(vector, fetch)
        return brute_force_neighbors(self._vectors, vector, fetch)

    def _ann_neighbors(
        self, vector: Sequence[float], fetch: int
    ) -> list[tuple[int, float]]:  # pragma: no cover - exercised only with hnswlib
        import numpy as np

        labels, distances = self._ann.knn_query(
            np.asarray([list(vector)], dtype="float32"), k=fetch
        )
        return [
            (int(label), 1.0 - float(distance))
            for label, distance in zip(labels[0], distances[0])
        ]

    def save(self, directory: str | Path) -> Path:
        """Persist the index and its provenance under ``directory``.

        Returns the payload path. The payload is deterministic JSON so an
        unchanged vocabulary and encoder round-trip byte-for-byte.
        """

        target_dir = Path(directory).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "provenance": self.provenance,
            "records": [
                {
                    "system": record.system,
                    "code": record.code,
                    "display": record.display,
                    "matched_alias": record.matched_alias,
                    "vocab_version": record.vocab_version,
                }
                for record in self._records
            ],
            "vectors": [list(vector) for vector in self._vectors],
        }
        path = target_dir / INDEX_FILENAME
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AliasEmbeddingIndex":
        """Rebuild an index from a persisted payload mapping."""

        provenance = payload["provenance"]  # type: ignore[index]
        records = [
            _AliasRecord(
                system=str(row["system"]),
                code=str(row["code"]),
                display=str(row["display"]),
                matched_alias=str(row["matched_alias"]),
                vocab_version=str(row["vocab_version"]),
            )
            for row in payload["records"]  # type: ignore[union-attr]
        ]
        vectors = [
            tuple(float(value) for value in vector)
            for vector in payload["vectors"]  # type: ignore[union-attr]
        ]
        return cls(
            encoder_id=str(provenance["encoder_id"]),  # type: ignore[index]
            dimension=int(provenance["dimension"]),  # type: ignore[index]
            backend=str(provenance["backend"]),  # type: ignore[index]
            index_key=str(provenance["index_key"]),  # type: ignore[index]
            vocab_versions={
                str(system): str(version)
                for system, version in dict(
                    provenance["vocab_versions"]  # type: ignore[index]
                ).items()
            },
            records=records,
            vectors=vectors,
        )


def _maybe_build_ann(
    backend: str, dimension: int, vectors: Sequence[Sequence[float]]
):  # pragma: no cover - exercised only with hnswlib installed
    if backend != "hnsw" or not vectors:
        return None
    import hnswlib
    import numpy as np

    ann = hnswlib.Index(space="cosine", dim=dimension)
    ann.init_index(
        max_elements=len(vectors),
        ef_construction=_HNSW_EF_CONSTRUCTION,
        M=_HNSW_M,
    )
    ann.add_items(
        np.asarray([list(vector) for vector in vectors], dtype="float32"),
        list(range(len(vectors))),
    )
    ann.set_ef(max(_HNSW_EF_CONSTRUCTION, len(vectors)))
    return ann


def _resolve_backend(backend: str) -> str:
    """Return the concrete backend name, honoring availability and requests."""

    if backend not in {"auto", "hnsw", "brute"}:
        raise ValueError(f"unknown index backend {backend!r}")
    if backend == "brute":
        return "brute"
    try:
        import hnswlib  # noqa: F401
    except ImportError as exc:
        if backend == "hnsw":
            raise IndexBackendUnavailableError(
                "Install openmed[grounding] with hnswlib to use the HNSW backend."
            ) from exc
        return "brute"
    return "hnsw"


def _collect_records(
    index: VocabularyIndex,
) -> tuple[list[_AliasRecord], list[str]]:
    system = index.system.upper()
    version = index.content_hash
    records: list[_AliasRecord] = []
    surfaces: list[str] = []
    seen: set[tuple[str, str]] = set()
    for concept in index.concepts:
        for alias in concept.aliases:
            normalized = normalize_alias(alias)
            if not normalized:
                continue
            key = (concept.code, normalized)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                _AliasRecord(
                    system=system,
                    code=concept.code,
                    display=concept.preferred_term,
                    matched_alias=normalized,
                    vocab_version=version,
                )
            )
            surfaces.append(normalized)
    return records, surfaces


def _collect_index_inputs(
    vocab: VocabLoader, systems: Sequence[str] | None
) -> tuple[list[_AliasRecord], list[str], dict[str, str]]:
    """Collect alias records/surfaces and per-system content hashes (no encoding)."""

    from .vocab import FREE_VOCAB_SYSTEMS, VocabLoaderError

    requested = tuple(systems) if systems is not None else FREE_VOCAB_SYSTEMS
    records: list[_AliasRecord] = []
    surfaces: list[str] = []
    vocab_versions: dict[str, str] = {}
    for system in requested:
        try:
            index = vocab.get_index(system)
        except VocabLoaderError:
            if systems is not None:
                raise
            continue
        system_records, system_surfaces = _collect_records(index)
        records.extend(system_records)
        surfaces.extend(system_surfaces)
        vocab_versions[index.system.upper()] = index.content_hash
    return records, surfaces, vocab_versions


def _index_key(vocab_versions: dict[str, str], encoder: AliasEncoder) -> str:
    """Content-address an index from vocab editions + encoder identity (no encoding)."""

    return make_index_cache_key(
        vocab_versions,
        encoder.encoder_id,
        params={"dimension": encoder.dimension, "schema_version": INDEX_SCHEMA_VERSION},
    )


def build_index(
    vocab: VocabLoader,
    encoder: AliasEncoder | None,
    *,
    systems: Sequence[str] | None = None,
    backend: str = "auto",
) -> AliasEmbeddingIndex | None:
    """Build an alias embedding index, or ``None`` for the no-op fallback.

    Args:
        vocab: Loader supplying free-vocabulary alias indexes.
        encoder: Alias encoder. When ``None`` (no encoder weights present), this
            returns ``None`` so callers fall back to sparse-only retrieval.
        systems: Vocabulary systems to index; defaults to every free system with
            a resolvable source. Systems with no configured source are skipped.
        backend: ``"auto"`` (HNSW when available, else brute-force), ``"hnsw"``
            (require ANN backend), or ``"brute"`` (force the reference path).

    Returns:
        A built :class:`AliasEmbeddingIndex`, or ``None`` when ``encoder`` is
        ``None``.
    """

    if encoder is None:
        return None

    resolved_backend = _resolve_backend(backend)
    records, surfaces, vocab_versions = _collect_index_inputs(vocab, systems)
    vectors = list(encoder.encode(surfaces)) if surfaces else []
    return AliasEmbeddingIndex(
        encoder_id=encoder.encoder_id,
        dimension=encoder.dimension,
        backend=resolved_backend,
        index_key=_index_key(vocab_versions, encoder),
        vocab_versions=vocab_versions,
        records=records,
        vectors=vectors,
    )


def query_index(
    index: AliasEmbeddingIndex | None,
    vector: Sequence[float],
    k: int = _DEFAULT_TOP_K,
) -> list[Candidate]:
    """Return dense candidates for a mention ``vector``.

    When ``index`` is ``None`` (the no-op fallback for a missing encoder) this
    returns ``[]`` so callers degrade to sparse-only retrieval.
    """

    if index is None:
        return []
    return index.query(vector, k)


def load_index(directory: str | Path) -> AliasEmbeddingIndex | None:
    """Load a persisted index from ``directory``, or ``None`` when absent."""

    path = Path(directory).expanduser() / INDEX_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AliasEmbeddingIndex.from_payload(payload)


def build_or_load_index(
    vocab: VocabLoader,
    encoder: AliasEncoder | None,
    *,
    cache_dir: str | Path,
    systems: Sequence[str] | None = None,
    backend: str = "auto",
) -> AliasEmbeddingIndex | None:
    """Return a version-validated index, rebuilding on vocabulary drift.

    A persisted index under ``cache_dir`` is reused only when its ``index_key``
    matches the key recomputed from the current vocabulary editions and encoder;
    any drift forces a rebuild and overwrites the stale payload. Returns ``None``
    when ``encoder`` is ``None`` (sparse-only fallback).
    """

    if encoder is None:
        return None

    # Compute the content-addressed key from vocabulary editions + encoder
    # identity WITHOUT encoding, so a cache hit skips re-encoding entirely.
    cached = load_index(cache_dir)
    if cached is not None:
        _, _, vocab_versions = _collect_index_inputs(vocab, systems)
        if cached.index_key == _index_key(vocab_versions, encoder):
            return cached

    fresh = build_index(vocab, encoder, systems=systems, backend=backend)
    if fresh is None:  # pragma: no cover - encoder is not None here
        return None
    fresh.save(cache_dir)
    return fresh


class DenseCandidateGenerator:
    """Emit dense ``Candidate`` objects by querying an alias embedding index.

    The generator sits behind the grounding registry (key ``"dense"``) so the
    retrieval stage consumes ANN candidates without importing the index backend
    directly. When no encoder is configured it degrades to an empty result,
    letting callers fall back to sparse-only retrieval.

    Args:
        encoder: Alias encoder; ``None`` yields the sparse-only no-op.
        index: Pre-built index to query. When omitted it is built lazily from
            ``loader`` and ``encoder`` on first use and cached per system set.
        loader: Vocabulary loader used to build the index when ``index`` is
            omitted. Defaults to a fresh :class:`VocabLoader`.
        backend: Index backend selector forwarded to :func:`build_index`.
    """

    source = DENSE_SOURCE

    def __init__(
        self,
        encoder: AliasEncoder | None = None,
        *,
        index: AliasEmbeddingIndex | None = None,
        loader: VocabLoader | None = None,
        backend: str = "auto",
    ) -> None:
        self._encoder = encoder
        self._index = index
        self._loader = loader if loader is not None else VocabLoader()
        self._backend = backend
        self._built: dict[tuple[str, ...], AliasEmbeddingIndex | None] = {}

    def generate(
        self,
        mention: str,
        systems: Sequence[str],
        k: int = _DEFAULT_TOP_K,
        *,
        language: str | None = None,
    ) -> list[Candidate]:
        """Return up to ``k`` dense candidates for ``mention``.

        Returns ``[]`` (the sparse-only fallback) when no encoder is configured
        or the mention is empty.
        """

        if not isinstance(mention, str):
            raise TypeError("mention must be a string")
        ordered_systems = tuple(systems)
        if not ordered_systems:
            raise ValueError("systems must contain at least one vocabulary system")
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if self._encoder is None or not normalize_alias(mention):
            return []

        index = self._resolve_index(ordered_systems)
        if index is None:
            return []
        return index.query_text(
            mention,
            self._encoder,
            k,
            source_language=language,
        )

    def _resolve_index(self, systems: Sequence[str]) -> AliasEmbeddingIndex | None:
        if self._index is not None:
            return self._index
        key = tuple(systems)
        if key not in self._built:
            self._built[key] = build_index(
                self._loader,
                self._encoder,
                systems=systems,
                backend=self._backend,
            )
        return self._built[key]


register_linker(DENSE_LINKER_KEY, DenseCandidateGenerator)
