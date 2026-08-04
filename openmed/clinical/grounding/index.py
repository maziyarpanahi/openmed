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
* **Incremental.** Each vocabulary system is stored as a separate embedding
  shard. Edition deltas rebuild only the affected shards, additions reuse every
  unchanged shard, and removals prune only the departed systems.
* **Optional ANN backend.** An HNSW backend (``hnswlib``) is used when
  available; otherwise a pure brute-force cosine search returns exact neighbors.
* **Graceful fallback.** When no encoder weights are present, :func:`build_index`
  returns ``None`` and :func:`query_index` returns ``[]`` so callers fall back to
  sparse-only retrieval rather than raising.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openmed.clinical.normalization.cache import IndexBoundCache, make_index_cache_key

from .embeddings import AliasEncoder
from .registry import register_linker
from .types import Candidate
from .vocab import VocabLoader, VocabularyIndex, normalize_alias, normalize_language

__all__ = [
    "AliasEmbeddingIndex",
    "DenseCandidateGenerator",
    "IndexBackendUnavailableError",
    "IndexUpdateSummary",
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
#: Persisted manifest schema version.
INDEX_SCHEMA_VERSION = 2
#: Persisted payload filename inside a cache directory.
INDEX_FILENAME = "alias_index.json"
#: Directory containing independently replaceable vocabulary shards.
INDEX_SHARD_DIRECTORY = "shards"
#: Persisted shard schema version.
INDEX_SHARD_SCHEMA_VERSION = 1

_DEFAULT_TOP_K = 10
_HNSW_EF_CONSTRUCTION = 200
_HNSW_M = 16
_NO_INDEX_KEY = "grounding-index:none"


class IndexBackendUnavailableError(RuntimeError):
    """Raised when an ANN backend is required but not installed."""


@dataclass(frozen=True)
class IndexUpdateSummary:
    """Vocabulary shards reused, rebuilt, and removed by an index update."""

    reused_shards: tuple[str, ...] = ()
    rebuilt_shards: tuple[str, ...] = ()
    removed_shards: tuple[str, ...] = ()


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
        system_order: Sequence[str] | None = None,
        update_summary: IndexUpdateSummary | None = None,
        storage_schema_version: int = INDEX_SCHEMA_VERSION,
        anns: dict[str, object | None] | None = None,
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
        record_systems = tuple(dict.fromkeys(record.system for record in self._records))
        requested_order = tuple(system.upper() for system in (system_order or ()))
        self._system_order = tuple(
            dict.fromkeys((*requested_order, *record_systems, *self.vocab_versions))
        )
        self._update_summary = update_summary or IndexUpdateSummary(
            reused_shards=self._system_order
        )
        self._storage_schema_version = storage_schema_version
        provided_anns = anns or {}
        self._anns: dict[str, object | None] = {}
        for system in self.systems:
            if backend != "hnsw":
                self._anns[system] = None
            elif system in provided_anns:
                self._anns[system] = provided_anns[system]
            else:
                self._anns[system] = _maybe_build_ann(
                    backend,
                    dimension,
                    self._shard_data(system)[1],
                )

    @property
    def record_count(self) -> int:
        """Number of embedded (concept, alias) rows in the index."""

        return len(self._records)

    @property
    def systems(self) -> tuple[str, ...]:
        """Vocabulary systems represented by independently persisted shards."""

        return self._system_order

    @property
    def update_summary(self) -> IndexUpdateSummary:
        """Shard-level work performed while constructing this instance."""

        return self._update_summary

    @property
    def shard_keys(self) -> dict[str, str]:
        """Content-addressed keys for the independently persisted shards."""

        return {
            system: _shard_key(
                system,
                self.vocab_versions[system],
                encoder_id=self.encoder_id,
                dimension=self.dimension,
            )
            for system in self.systems
        }

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
            "shards": {
                system: {
                    "shard_key": self.shard_keys[system],
                    "vocab_version": self.vocab_versions[system],
                    "record_count": len(self._shard_data(system)[0]),
                }
                for system in self.systems
            },
        }

    def _shard_data(
        self, system: str
    ) -> tuple[tuple[_AliasRecord, ...], tuple[tuple[float, ...], ...]]:
        """Return records and vectors belonging to one vocabulary system."""

        pairs = [
            (self._records[index], self._vectors[index])
            for index in self._shard_positions(system)
        ]
        return (
            tuple(record for record, _ in pairs),
            tuple(vector for _, vector in pairs),
        )

    def _shard_positions(self, system: str) -> tuple[int, ...]:
        resolved = system.upper()
        return tuple(
            index
            for index, record in enumerate(self._records)
            if record.system == resolved
        )

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
        neighbors: list[tuple[int, float]] = []
        for system in self.systems:
            positions = self._shard_positions(system)
            if not positions:
                continue
            shard_fetch = min(len(positions), fetch)
            ann = self._anns[system]
            if ann is None:
                local_neighbors = brute_force_neighbors(
                    tuple(self._vectors[position] for position in positions),
                    vector,
                    shard_fetch,
                )
            else:
                local_neighbors = self._ann_neighbors(ann, vector, shard_fetch)
            neighbors.extend(
                (positions[local_row], similarity)
                for local_row, similarity in local_neighbors
            )
        neighbors.sort(key=lambda item: (-item[1], item[0]))
        return neighbors[:fetch]

    def _ann_neighbors(
        self, ann: object, vector: Sequence[float], fetch: int
    ) -> list[tuple[int, float]]:  # pragma: no cover - exercised only with hnswlib
        import numpy as np

        labels, distances = ann.knn_query(  # type: ignore[attr-defined]
            np.asarray([list(vector)], dtype="float32"), k=fetch
        )
        return [
            (int(label), 1.0 - float(distance))
            for label, distance in zip(labels[0], distances[0])
        ]

    def save(self, directory: str | Path) -> Path:
        """Persist every vocabulary shard and the deterministic manifest."""

        return self._save(directory, shard_systems=None)

    def _save(
        self,
        directory: str | Path,
        *,
        shard_systems: Sequence[str] | None = None,
    ) -> Path:
        """Persist a manifest and independently replaceable vocabulary shards.

        Args:
            directory: Local index cache directory.
            shard_systems: Systems whose shard files must be rewritten. When
                omitted, every shard is written. Existing valid shard files for
                other systems are left byte-for-byte untouched, while files for
                removed systems are deleted.

        Returns:
            Path to the deterministic index manifest.
        """

        target_dir = Path(directory).expanduser()
        shard_dir = target_dir / INDEX_SHARD_DIRECTORY
        shard_dir.mkdir(parents=True, exist_ok=True)
        requested = (
            set(self.systems)
            if shard_systems is None
            else {system.upper() for system in shard_systems}
        )

        descriptors: dict[str, dict[str, object]] = {}
        expected_files: set[str] = set()
        for system in self.systems:
            records, vectors = self._shard_data(system)
            filename = _shard_filename(system, self.shard_keys[system])
            expected_files.add(filename)
            shard_path = shard_dir / filename
            if system in requested or not shard_path.exists():
                payload = {
                    "schema_version": INDEX_SHARD_SCHEMA_VERSION,
                    "system": system,
                    "encoder_id": self.encoder_id,
                    "dimension": self.dimension,
                    "shard_key": self.shard_keys[system],
                    "vocab_version": self.vocab_versions[system],
                    "records": [_record_payload(record) for record in records],
                    "vectors": [list(vector) for vector in vectors],
                }
                shard_path.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
            descriptors[system] = {
                "file": f"{INDEX_SHARD_DIRECTORY}/{filename}",
                "shard_key": self.shard_keys[system],
                "vocab_version": self.vocab_versions[system],
                "record_count": len(records),
            }
            ann = self._anns[system]
            if ann is not None:
                ann_filename = _ann_filename(system, self.shard_keys[system])
                expected_files.add(ann_filename)
                ann_path = shard_dir / ann_filename
                if system in requested or not ann_path.exists():
                    ann.save_index(str(ann_path))  # type: ignore[attr-defined]
                descriptors[system]["ann_file"] = (
                    f"{INDEX_SHARD_DIRECTORY}/{ann_filename}"
                )

        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "provenance": self.provenance,
            "system_order": list(self.systems),
            "shards": descriptors,
        }
        path = target_dir / INDEX_FILENAME
        path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        for pattern in ("*.json", "*.hnsw"):
            for stale_path in shard_dir.glob(pattern):
                if stale_path.name not in expected_files:
                    stale_path.unlink()
        self._storage_schema_version = INDEX_SCHEMA_VERSION
        return path

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        *,
        directory: str | Path | None = None,
    ) -> "AliasEmbeddingIndex":
        """Rebuild an index from a manifest or legacy monolithic payload."""

        schema_version = int(payload.get("schema_version", 1))
        provenance = _mapping(payload, "provenance")
        vocab_versions = {
            str(system): str(version)
            for system, version in _mapping(provenance, "vocab_versions").items()
        }
        expected_index_key = _index_key_for_identity(
            vocab_versions,
            encoder_id=str(provenance["encoder_id"]),
            dimension=int(provenance["dimension"]),
            schema_version=schema_version,
        )
        if str(provenance["index_key"]) != expected_index_key:
            raise ValueError("persisted grounding index key failed validation")
        loaded_anns: dict[str, object | None] = {}

        if schema_version == 1:
            records = [
                _record_from_payload(row)
                for row in _mapping_sequence(payload, "records")
            ]
            vectors = [
                tuple(float(value) for value in vector)
                for vector in _sequence(payload, "vectors")
            ]
            system_order = tuple(dict.fromkeys(record.system for record in records))
        elif schema_version == INDEX_SCHEMA_VERSION:
            if directory is None:
                raise ValueError("directory is required to load a sharded index")
            root = Path(directory).expanduser()
            descriptors = _mapping(payload, "shards")
            raw_order = payload.get("system_order", tuple(descriptors))
            if not isinstance(raw_order, (list, tuple)):
                raise ValueError("index system_order must be a list")
            system_order = tuple(str(system).upper() for system in raw_order)
            if (
                len(system_order) != len(set(system_order))
                or set(system_order) != set(vocab_versions)
                or set(system_order) != set(descriptors)
            ):
                raise ValueError("index manifest systems do not match provenance")
            records = []
            vectors = []
            for system in system_order:
                descriptor = _mapping(descriptors, system)
                relative_path = Path(str(descriptor["file"]))
                expected_shard_key = _shard_key(
                    system,
                    vocab_versions[system],
                    encoder_id=str(provenance["encoder_id"]),
                    dimension=int(provenance["dimension"]),
                )
                if (
                    str(descriptor["vocab_version"]) != vocab_versions[system]
                    or str(descriptor["shard_key"]) != expected_shard_key
                ):
                    raise ValueError(
                        f"index shard descriptor {system!r} failed validation"
                    )
                if (
                    relative_path.is_absolute()
                    or len(relative_path.parts) != 2
                    or relative_path.parts[0] != INDEX_SHARD_DIRECTORY
                    or relative_path.name != _shard_filename(system, expected_shard_key)
                ):
                    raise ValueError(
                        "index shard path must stay inside cache directory"
                    )
                shard_payload = json.loads(
                    (root / relative_path).read_text(encoding="utf-8")
                )
                _validate_shard_payload(
                    shard_payload,
                    system=system,
                    descriptor=descriptor,
                    encoder_id=str(provenance["encoder_id"]),
                    dimension=int(provenance["dimension"]),
                )
                shard_records = [
                    _record_from_payload(row)
                    for row in _mapping_sequence(shard_payload, "records")
                ]
                shard_vectors = [
                    tuple(float(value) for value in vector)
                    for vector in _sequence(shard_payload, "vectors")
                ]
                records.extend(shard_records)
                vectors.extend(shard_vectors)
                if str(provenance["backend"]) == "hnsw" and shard_vectors:
                    ann_relative = Path(str(descriptor.get("ann_file", "")))
                    if (
                        ann_relative.is_absolute()
                        or len(ann_relative.parts) != 2
                        or ann_relative.parts[0] != INDEX_SHARD_DIRECTORY
                        or ann_relative.name
                        != _ann_filename(system, expected_shard_key)
                    ):
                        raise ValueError(
                            f"index ANN shard path {system!r} failed validation"
                        )
                    loaded_anns[system] = _load_ann(
                        root / ann_relative,
                        dimension=int(provenance["dimension"]),
                        record_count=len(shard_vectors),
                    )
        else:
            raise ValueError(f"unsupported grounding index schema {schema_version}")

        if int(provenance["record_count"]) != len(records):
            raise ValueError("persisted grounding index record count is invalid")
        if schema_version == INDEX_SCHEMA_VERSION:
            provenance_shards = _mapping(provenance, "shards")
            for system in system_order:
                descriptor = _mapping(descriptors, system)
                shard_provenance = _mapping(provenance_shards, system)
                for field in ("shard_key", "vocab_version", "record_count"):
                    if shard_provenance.get(field) != descriptor.get(field):
                        raise ValueError(
                            f"persisted grounding provenance {system!r} is invalid"
                        )

        return cls(
            encoder_id=str(provenance["encoder_id"]),
            dimension=int(provenance["dimension"]),
            backend=str(provenance["backend"]),
            index_key=str(provenance["index_key"]),
            vocab_versions=vocab_versions,
            records=records,
            vectors=vectors,
            system_order=system_order,
            update_summary=IndexUpdateSummary(reused_shards=system_order),
            storage_schema_version=schema_version,
            anns=loaded_anns,
        )


def _mapping(payload: object, key: str) -> dict[str, object]:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), dict):
        raise ValueError(f"index payload field {key!r} must be a mapping")
    return payload[key]  # type: ignore[return-value]


def _sequence(payload: object, key: str) -> list[object]:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ValueError(f"index payload field {key!r} must be a list")
    return payload[key]  # type: ignore[return-value]


def _mapping_sequence(payload: object, key: str) -> list[dict[str, object]]:
    values = _sequence(payload, key)
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"index payload field {key!r} must contain mappings")
    return values  # type: ignore[return-value]


def _record_payload(record: _AliasRecord) -> dict[str, str]:
    return {
        "system": record.system,
        "code": record.code,
        "display": record.display,
        "matched_alias": record.matched_alias,
        "vocab_version": record.vocab_version,
    }


def _record_from_payload(row: dict[str, object]) -> _AliasRecord:
    return _AliasRecord(
        system=str(row["system"]),
        code=str(row["code"]),
        display=str(row["display"]),
        matched_alias=str(row["matched_alias"]),
        vocab_version=str(row["vocab_version"]),
    )


def _validate_shard_payload(
    payload: object,
    *,
    system: str,
    descriptor: dict[str, object],
    encoder_id: str,
    dimension: int,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("index shard payload must be a mapping")
    expected = {
        "schema_version": INDEX_SHARD_SCHEMA_VERSION,
        "system": system,
        "encoder_id": encoder_id,
        "dimension": dimension,
        "shard_key": str(descriptor["shard_key"]),
        "vocab_version": str(descriptor["vocab_version"]),
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"persisted grounding shard {system!r} failed validation")
    records = _mapping_sequence(payload, "records")
    vectors = _sequence(payload, "vectors")
    if len(records) != len(vectors) or len(records) != int(descriptor["record_count"]):
        raise ValueError(f"persisted grounding shard {system!r} has invalid rows")
    if any(
        not isinstance(vector, list) or len(vector) != dimension for vector in vectors
    ):
        raise ValueError(f"persisted grounding shard {system!r} has invalid vectors")
    if any(
        str(record.get("system", "")).upper() != system
        or str(record.get("vocab_version", "")) != str(descriptor["vocab_version"])
        for record in records
    ):
        raise ValueError(f"persisted grounding shard {system!r} has invalid records")


def _shard_filename(system: str, shard_key: str) -> str:
    digest = hashlib.sha256(shard_key.encode("utf-8")).hexdigest()[:16]
    return f"{system.casefold()}-{digest}.json"


def _ann_filename(system: str, shard_key: str) -> str:
    return Path(_shard_filename(system, shard_key)).with_suffix(".hnsw").name


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


def _load_ann(
    path: Path, *, dimension: int, record_count: int
):  # pragma: no cover - exercised only with hnswlib installed
    import hnswlib

    ann = hnswlib.Index(space="cosine", dim=dimension)
    ann.load_index(str(path), max_elements=record_count)
    ann.set_ef(max(_HNSW_EF_CONSTRUCTION, record_count))
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


def _resolve_vocab_indexes(
    vocab: VocabLoader, systems: Sequence[str] | None
) -> tuple[VocabularyIndex, ...]:
    """Resolve requested vocabulary indexes without encoding their aliases."""

    from .vocab import FREE_VOCAB_SYSTEMS, VocabLoaderError

    requested = tuple(systems) if systems is not None else FREE_VOCAB_SYSTEMS
    indexes: list[VocabularyIndex] = []
    seen: set[str] = set()
    for system in requested:
        try:
            index = vocab.get_index(system)
        except VocabLoaderError:
            if systems is not None:
                raise
            continue
        normalized_system = index.system.upper()
        if normalized_system in seen:
            continue
        seen.add(normalized_system)
        indexes.append(index)
    return tuple(indexes)


def _index_key(vocab_versions: dict[str, str], encoder: AliasEncoder) -> str:
    """Content-address an index from vocab editions + encoder identity (no encoding)."""

    return _index_key_for_identity(
        vocab_versions,
        encoder_id=encoder.encoder_id,
        dimension=encoder.dimension,
        schema_version=INDEX_SCHEMA_VERSION,
    )


def _index_key_for_identity(
    vocab_versions: dict[str, str],
    *,
    encoder_id: str,
    dimension: int,
    schema_version: int,
) -> str:
    """Content-address an index from explicit persisted identity fields."""

    return make_index_cache_key(
        vocab_versions,
        encoder_id,
        params={"dimension": dimension, "schema_version": schema_version},
    )


def _shard_key(
    system: str,
    vocab_version: str,
    *,
    encoder_id: str,
    dimension: int,
) -> str:
    """Content-address one independently persisted vocabulary shard."""

    return make_index_cache_key(
        {system: vocab_version},
        encoder_id,
        params={
            "dimension": dimension,
            "schema_version": INDEX_SHARD_SCHEMA_VERSION,
            "shard": system,
        },
    )


def _encode_surfaces(
    encoder: AliasEncoder, surfaces: Sequence[str]
) -> tuple[tuple[float, ...], ...]:
    vectors = tuple(encoder.encode(surfaces)) if surfaces else ()
    if len(vectors) != len(surfaces):
        raise ValueError("encoder must return one vector per alias")
    if any(len(vector) != encoder.dimension for vector in vectors):
        raise ValueError(
            f"encoder vectors must have configured dimension {encoder.dimension}"
        )
    return vectors


def _build_from_indexes(
    indexes: Sequence[VocabularyIndex],
    encoder: AliasEncoder,
    *,
    backend: str,
    cached: AliasEmbeddingIndex | None = None,
) -> AliasEmbeddingIndex:
    """Build an index, reusing unchanged per-system embedding shards."""

    records: list[_AliasRecord] = []
    vectors: list[tuple[float, ...]] = []
    vocab_versions = {index.system.upper(): index.content_hash for index in indexes}
    system_order = tuple(index.system.upper() for index in indexes)
    reusable = (
        cached is not None
        and cached.encoder_id == encoder.encoder_id
        and cached.dimension == encoder.dimension
    )
    reused_shards: list[str] = []
    rebuilt_shards: list[str] = []
    reused_anns: dict[str, object | None] = {}

    for index in indexes:
        system = index.system.upper()
        if reusable and cached.vocab_versions.get(system) == index.content_hash:
            shard_records, shard_vectors = cached._shard_data(system)
            records.extend(shard_records)
            vectors.extend(shard_vectors)
            reused_shards.append(system)
            if cached.backend == backend:
                reused_anns[system] = cached._anns[system]
            continue

        shard_records, surfaces = _collect_records(index)
        shard_vectors = _encode_surfaces(encoder, surfaces)
        records.extend(shard_records)
        vectors.extend(shard_vectors)
        rebuilt_shards.append(system)

    removed_shards = tuple(
        system
        for system in (cached.systems if cached is not None else ())
        if system not in vocab_versions
    )
    return AliasEmbeddingIndex(
        encoder_id=encoder.encoder_id,
        dimension=encoder.dimension,
        backend=backend,
        index_key=_index_key(vocab_versions, encoder),
        vocab_versions=vocab_versions,
        records=records,
        vectors=vectors,
        system_order=system_order,
        update_summary=IndexUpdateSummary(
            reused_shards=tuple(reused_shards),
            rebuilt_shards=tuple(rebuilt_shards),
            removed_shards=removed_shards,
        ),
        anns=reused_anns,
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
    indexes = _resolve_vocab_indexes(vocab, systems)
    return _build_from_indexes(
        indexes,
        encoder,
        backend=resolved_backend,
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

    root = Path(directory).expanduser()
    path = root / INDEX_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AliasEmbeddingIndex.from_payload(payload, directory=root)


def build_or_load_index(
    vocab: VocabLoader,
    encoder: AliasEncoder | None,
    *,
    cache_dir: str | Path,
    systems: Sequence[str] | None = None,
    backend: str = "auto",
    dependent_caches: Sequence[IndexBoundCache] = (),
) -> AliasEmbeddingIndex | None:
    """Return a version-validated index, updating only changed vocab shards.

    A persisted index under ``cache_dir`` is reused only when its ``index_key``
    matches the key recomputed from the current vocabulary editions and encoder;
    any drift re-encodes new or changed vocabulary shards, reuses byte-identical
    unchanged shards, and removes shards no longer requested. Index-bound
    candidate/rerank caches are evicted when that key changes. Returns ``None``
    when ``encoder`` is ``None`` (sparse-only fallback).
    """

    if encoder is None:
        for dependent_cache in dependent_caches:
            dependent_cache.bind_index(_NO_INDEX_KEY)
        return None

    resolved_backend = _resolve_backend(backend)
    indexes = _resolve_vocab_indexes(vocab, systems)
    vocab_versions = {index.system.upper(): index.content_hash for index in indexes}
    expected_key = _index_key(vocab_versions, encoder)
    for dependent_cache in dependent_caches:
        dependent_cache.bind_index(expected_key)

    try:
        cached = load_index(cache_dir)
    except (ImportError, OSError, KeyError, RuntimeError, TypeError, ValueError):
        cached = None
    if (
        cached is not None
        and cached.index_key == expected_key
        and cached.backend == resolved_backend
    ):
        return cached

    fresh = _build_from_indexes(
        indexes,
        encoder,
        backend=resolved_backend,
        cached=cached,
    )
    rewrite_shards: Sequence[str] | None
    if cached is None or cached._storage_schema_version != INDEX_SCHEMA_VERSION:
        rewrite_shards = None
    else:
        rewrite_shards = fresh.update_summary.rebuilt_shards
    fresh._save(cache_dir, shard_systems=rewrite_shards)
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
