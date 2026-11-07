"""Versioned, content-addressed cache for local terminology indexes.

The cache stores the result of parsing a free vocabulary, not the source
release or any caller text.  Each entry contains a canonical JSON index and a
small manifest that pins the vocabulary system URI, release version, and
content hash.  Loading verifies both the artifact digest and the reconstructed
``VocabularyIndex`` content hash before returning it.

Restricted terminology is never written unless ``allow_restricted=True`` is
explicitly configured.  Cache misses are built inside OpenMed's offline socket
guard, so a caller can use a local builder while ``OPENMED_OFFLINE`` prevents
an accidental network fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openmed.core.offline import is_local_only, network_blocked_if_offline

from .vocab import VocabConcept, VocabularyIndex

__all__ = [
    "DEFAULT_CACHE_ENV",
    "DEFAULT_GROUNDING_CACHE_ENV",
    "INDEX_ARTIFACT_FILENAME",
    "MANIFEST_FILENAME",
    "MAX_ARTIFACT_BYTES",
    "MAX_MANIFEST_BYTES",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotCache",
    "SnapshotCacheStats",
    "SnapshotIntegrityError",
    "SnapshotManifest",
    "SnapshotPolicyError",
    "TerminologySnapshot",
    "TerminologySnapshotCache",
    "VersionedVocabularyLoader",
    "clear_cache",
    "default_cache_dir",
    "load_or_build_snapshot",
    "load_snapshot",
    "snapshot_path",
    "store_snapshot",
]

SNAPSHOT_SCHEMA_VERSION = 1
INDEX_ARTIFACT_FILENAME = "index.json"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_CACHE_ENV = "OPENMED_TERMINOLOGY_CACHE_DIR"
DEFAULT_GROUNDING_CACHE_ENV = "OPENMED_CACHE_DIR"
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024

_DIGEST_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_CACHE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_RESTRICTED_SYSTEM_URIS = frozenset(
    {
        "http://terminology.hl7.org/CodeSystem/umls",
        "http://snomed.info/sct",
    }
)
_SYSTEM_URI_ALIASES = {
    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "rx-norm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "icd10": "http://hl7.org/fhir/sid/icd-10-cm",
    "icd10cm": "http://hl7.org/fhir/sid/icd-10-cm",
    "icd-10": "http://hl7.org/fhir/sid/icd-10-cm",
    "icd-10-cm": "http://hl7.org/fhir/sid/icd-10-cm",
    "icd11": "http://id.who.int/icd/release/11/mms",
    "icd-11-mms": "http://id.who.int/icd/release/11/mms",
    "loinc": "http://loinc.org",
    "snomed": "http://snomed.info/sct",
    "snomed-ct": "http://snomed.info/sct",
    "snomedct": "http://snomed.info/sct",
    "umls": "http://terminology.hl7.org/CodeSystem/umls",
    "hpo": "http://human-phenotype-ontology.org",
    "hp": "http://human-phenotype-ontology.org",
    "mesh": "https://www.nlm.nih.gov/mesh",
}


class SnapshotIntegrityError(ValueError):
    """Raised when a terminology snapshot or manifest fails validation."""


class SnapshotPolicyError(ValueError):
    """Raised when cache policy would persist restricted terminology."""


@dataclass(frozen=True)
class SnapshotManifest:
    """Deterministic metadata describing one persisted terminology snapshot."""

    system_uri: str
    release_version: str
    content_hash: str
    artifact_sha256: str
    index_system: str
    concept_count: int
    snapshot_key: str
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    artifact_file: str = INDEX_ARTIFACT_FILENAME

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible manifest representation."""

        return {
            "artifact_file": self.artifact_file,
            "artifact_sha256": self.artifact_sha256,
            "concept_count": self.concept_count,
            "content_hash": self.content_hash,
            "index_system": self.index_system,
            "release_version": self.release_version,
            "schema_version": self.schema_version,
            "snapshot_key": self.snapshot_key,
            "system_uri": self.system_uri,
        }


@dataclass(frozen=True)
class TerminologySnapshot:
    """A validated vocabulary index and its release provenance.

    ``cache_hit`` is runtime-only metadata and is never serialized.  The
    ``index`` is exposed directly through attribute delegation so callers can
    use ``snapshot.lookup(...)`` or ``snapshot.content_hash`` naturally.
    """

    index: VocabularyIndex
    system_uri: str
    release_version: str
    content_hash: str
    artifact_path: Path | None = None
    manifest_path: Path | None = None
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.index, VocabularyIndex):
            raise TypeError("snapshot index must be a VocabularyIndex")
        object.__setattr__(self, "system_uri", _normalize_system_uri(self.system_uri))
        object.__setattr__(
            self,
            "release_version",
            _normalize_release_version(self.release_version),
        )
        normalized_hash = _normalize_digest(self.content_hash, "content_hash")
        if self.index.content_hash != normalized_hash:
            raise SnapshotIntegrityError(
                "snapshot content_hash does not match its vocabulary index"
            )
        object.__setattr__(self, "content_hash", normalized_hash)

    @property
    def snapshot_key(self) -> str:
        """Return the content-addressed key for this snapshot."""

        return _snapshot_key(
            self.system_uri,
            self.release_version,
            self.content_hash,
        )

    @property
    def concept_count(self) -> int:
        """Return the number of concepts in the indexed vocabulary."""

        return self.index.concept_count

    @property
    def hit(self) -> bool:
        """Alias for :attr:`cache_hit` used by cache callers."""

        return self.cache_hit

    def __getattr__(self, name: str) -> Any:
        """Delegate vocabulary-index methods and properties to ``index``."""

        return getattr(self.index, name)


@dataclass(frozen=True)
class SnapshotCacheStats:
    """Observable cache hit/miss and integrity counters."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    corruptions: int = 0

    @property
    def hit_rate(self) -> float:
        """Return the hit ratio for attempted cache loads."""

        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class TerminologySnapshotCache:
    """Persist and validate version-pinned free vocabulary indexes.

    Args:
        cache_dir: Directory containing content-addressed snapshot entries. If
            omitted, ``OPENMED_TERMINOLOGY_CACHE_DIR`` is preferred, then
            ``OPENMED_CACHE_DIR``/``XDG_CACHE_HOME``, and finally the user's
            local OpenMed cache.
        allow_restricted: Explicitly permit persistence for restricted systems.
            The default keeps restricted concept text out of the cache.
        local_only: Override ``OPENMED_OFFLINE`` for builder execution. Local
            builders remain usable in offline mode, but outbound sockets are
            blocked while they run.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        allow_restricted: bool = False,
        local_only: bool | None = None,
    ) -> None:
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else default_cache_dir()
        )
        self.allow_restricted = bool(allow_restricted)
        self._local_only_override = local_only
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._corruptions = 0

    @property
    def local_only(self) -> bool:
        """Return whether builders must run under the offline network guard."""

        if self._local_only_override is not None:
            return bool(self._local_only_override)
        return is_local_only()

    def stats(self) -> SnapshotCacheStats:
        """Return a stable snapshot of hit/miss counters."""

        return SnapshotCacheStats(
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            corruptions=self._corruptions,
        )

    def reset_stats(self) -> None:
        """Reset hit/miss counters without removing persisted snapshots."""

        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._corruptions = 0

    def load(
        self,
        system_uri: str,
        release_version: str,
        *,
        content_hash: str | None = None,
        restricted: bool | None = None,
    ) -> TerminologySnapshot | None:
        """Load a matching verified snapshot, or return ``None`` on a miss.

        When ``content_hash`` is omitted, the cache searches the pinned system
        and release for a valid entry. Supplying it gives callers a strict
        content-addressed lookup and rejects a changed source release.
        """

        normalized_uri = _normalize_system_uri(system_uri)
        normalized_release = _normalize_release_version(release_version)
        expected_hash = (
            _normalize_digest(content_hash, "content_hash")
            if content_hash is not None
            else None
        )
        is_restricted = _restricted_system(normalized_uri, restricted)
        if is_restricted and not self.allow_restricted:
            self._misses += 1
            return None

        candidates = self._candidate_directories(
            normalized_uri,
            normalized_release,
            expected_hash,
        )
        for directory in candidates:
            try:
                snapshot = load_snapshot(
                    directory,
                    expected_system_uri=normalized_uri,
                    expected_release_version=normalized_release,
                    expected_content_hash=expected_hash,
                )
            except SnapshotIntegrityError:
                self._corruptions += 1
                continue
            self._hits += 1
            return replace(snapshot, cache_hit=True)

        self._misses += 1
        return None

    get = load

    def store(
        self,
        snapshot_or_index: TerminologySnapshot | VocabularyIndex,
        system_uri: str | None = None,
        release_version: str | None = None,
        *,
        content_hash: str | None = None,
        restricted: bool | None = None,
    ) -> TerminologySnapshot:
        """Persist a validated index and return its cache metadata.

        ``source`` fields from :class:`VocabConcept` are intentionally omitted
        from the artifact, so local paths and caller provenance are not copied
        into the cache.
        """

        snapshot = _coerce_snapshot(
            snapshot_or_index,
            system_uri=system_uri,
            release_version=release_version,
            content_hash=content_hash,
        )
        is_restricted = _restricted_system(snapshot.system_uri, restricted)
        if is_restricted and not self.allow_restricted:
            raise SnapshotPolicyError(
                "restricted terminology snapshots require allow_restricted=True"
            )

        artifact = _index_payload(snapshot.index)
        artifact_bytes = _canonical_json_bytes(artifact)
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        manifest = SnapshotManifest(
            system_uri=snapshot.system_uri,
            release_version=snapshot.release_version,
            content_hash=snapshot.content_hash,
            artifact_sha256=artifact_sha256,
            index_system=snapshot.index.system,
            concept_count=snapshot.index.concept_count,
            snapshot_key=snapshot.snapshot_key,
        )
        directory = self.cache_dir / _cache_key_directory(snapshot.snapshot_key)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(directory / INDEX_ARTIFACT_FILENAME, artifact_bytes)
        _atomic_write(
            directory / MANIFEST_FILENAME, _canonical_json_bytes(manifest.to_dict())
        )
        self._writes += 1
        return replace(
            snapshot,
            artifact_path=directory / INDEX_ARTIFACT_FILENAME,
            manifest_path=directory / MANIFEST_FILENAME,
            cache_hit=False,
        )

    put = store

    def load_or_build(
        self,
        system_uri: str,
        release_version: str,
        builder: Callable[[], VocabularyIndex | TerminologySnapshot],
        *,
        content_hash: str | None = None,
        restricted: bool | None = None,
    ) -> TerminologySnapshot:
        """Return a cache hit or build, validate, and persist a new snapshot."""

        if not callable(builder):
            raise TypeError("builder must be callable")
        normalized_uri = _normalize_system_uri(system_uri)
        normalized_release = _normalize_release_version(release_version)
        is_restricted = _restricted_system(normalized_uri, restricted)
        cached = self.load(
            normalized_uri,
            normalized_release,
            content_hash=content_hash,
            restricted=is_restricted,
        )
        if cached is not None:
            return cached

        with network_blocked_if_offline(local_only=self.local_only):
            built = builder()
        snapshot = _coerce_snapshot(
            built,
            system_uri=normalized_uri,
            release_version=normalized_release,
            content_hash=content_hash,
        )
        if is_restricted and not self.allow_restricted:
            return snapshot
        return self.store(snapshot, restricted=is_restricted)

    def clear(self) -> int:
        """Remove only snapshot entries owned by this cache directory."""

        if not self.cache_dir.is_dir():
            return 0
        removed = 0
        for child in tuple(self.cache_dir.iterdir()):
            if not child.is_dir() or not _CACHE_KEY_RE.fullmatch(child.name):
                continue
            if not any(
                (child / filename).is_file()
                for filename in (INDEX_ARTIFACT_FILENAME, MANIFEST_FILENAME)
            ):
                continue
            shutil.rmtree(child)
            removed += 1
        return removed

    def _candidate_directories(
        self,
        system_uri: str,
        release_version: str,
        content_hash: str | None,
    ) -> tuple[Path, ...]:
        if content_hash is not None:
            key = _snapshot_key(system_uri, release_version, content_hash)
            return (self.cache_dir / _cache_key_directory(key),)
        if not self.cache_dir.is_dir():
            return ()
        return tuple(
            sorted(
                child
                for child in self.cache_dir.iterdir()
                if child.is_dir() and _CACHE_KEY_RE.fullmatch(child.name)
            )
        )


SnapshotCache = TerminologySnapshotCache


class VersionedVocabularyLoader:
    """Adapt a local vocabulary builder to the snapshot cache.

    The adapter deliberately requires an explicit release version. It can be
    used anywhere a small loader object with ``get_index()`` is convenient.
    """

    redistributable = True
    restricted_license = False

    def __init__(
        self,
        system_uri: str,
        release_version: str,
        builder: Callable[[], VocabularyIndex | TerminologySnapshot],
        *,
        cache: TerminologySnapshotCache | None = None,
        cache_dir: str | Path | None = None,
        content_hash: str | None = None,
        restricted: bool | None = None,
        allow_restricted: bool = False,
    ) -> None:
        if cache is not None and cache_dir is not None:
            raise ValueError("pass either cache or cache_dir, not both")
        self.system_uri = _normalize_system_uri(system_uri)
        self.release_version = _normalize_release_version(release_version)
        if not callable(builder):
            raise TypeError("builder must be callable")
        self.builder = builder
        self.content_hash = content_hash
        self.restricted = restricted
        self.cache = cache or TerminologySnapshotCache(
            cache_dir,
            allow_restricted=allow_restricted,
        )
        self._snapshot: TerminologySnapshot | None = None

    def load_snapshot(self) -> TerminologySnapshot:
        """Load or build the pinned snapshot."""

        self._snapshot = self.cache.load_or_build(
            self.system_uri,
            self.release_version,
            self.builder,
            content_hash=self.content_hash,
            restricted=self.restricted,
        )
        return self._snapshot

    def get_index(self) -> VocabularyIndex:
        """Return the validated vocabulary index for the pinned release."""

        return self.load_snapshot().index

    load = get_index


def default_cache_dir() -> Path:
    """Return the local terminology snapshot cache directory."""

    configured = os.getenv(DEFAULT_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    root = os.getenv(DEFAULT_GROUNDING_CACHE_ENV)
    if root:
        return Path(root).expanduser() / "grounding" / "snapshots"
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "openmed" / "grounding" / "snapshots"
    return Path.home() / ".cache" / "openmed" / "grounding" / "snapshots"


def snapshot_path(
    system_uri: str,
    release_version: str,
    content_hash: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the deterministic artifact path for a snapshot identity."""

    normalized_uri = _normalize_system_uri(system_uri)
    normalized_release = _normalize_release_version(release_version)
    normalized_hash = _normalize_digest(content_hash, "content_hash")
    key = _snapshot_key(normalized_uri, normalized_release, normalized_hash)
    root = (
        Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    )
    return root / _cache_key_directory(key) / INDEX_ARTIFACT_FILENAME


def load_snapshot(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected_system_uri: str | None = None,
    expected_release_version: str | None = None,
    expected_content_hash: str | None = None,
) -> TerminologySnapshot:
    """Load and integrity-check one local terminology snapshot.

    ``path`` may point at the artifact or its containing content-addressed
    directory. This function raises on malformed or mismatched files; the
    cache's ``load`` method converts that failure into a miss so a caller can
    rebuild the entry.
    """

    artifact_file = Path(path).expanduser()
    if artifact_file.is_dir():
        directory = artifact_file
        artifact_file = directory / INDEX_ARTIFACT_FILENAME
    else:
        directory = artifact_file.parent
    manifest_file = (
        Path(manifest_path).expanduser()
        if manifest_path is not None
        else directory / MANIFEST_FILENAME
    )
    try:
        artifact_bytes = _read_limited_file(
            artifact_file,
            max_bytes=MAX_ARTIFACT_BYTES,
            label="terminology index artifact",
        )
        manifest_bytes = _read_limited_file(
            manifest_file,
            max_bytes=MAX_MANIFEST_BYTES,
            label="terminology snapshot manifest",
        )
    except SnapshotIntegrityError:
        raise
    except OSError as exc:
        raise SnapshotIntegrityError(
            "terminology snapshot files must be readable"
        ) from exc

    try:
        artifact = json.loads(artifact_bytes)
        manifest_payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SnapshotIntegrityError(
            "terminology snapshot contains invalid JSON"
        ) from exc
    if not isinstance(artifact, Mapping) or not isinstance(manifest_payload, Mapping):
        raise SnapshotIntegrityError("terminology snapshot files must contain objects")

    expected_manifest_keys = {
        "artifact_file",
        "artifact_sha256",
        "concept_count",
        "content_hash",
        "index_system",
        "release_version",
        "schema_version",
        "snapshot_key",
        "system_uri",
    }
    if set(manifest_payload) != expected_manifest_keys:
        raise SnapshotIntegrityError("terminology snapshot manifest fields are invalid")
    if manifest_payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotIntegrityError("unsupported terminology snapshot schema version")
    if manifest_payload.get("artifact_file") != INDEX_ARTIFACT_FILENAME:
        raise SnapshotIntegrityError(
            "terminology snapshot artifact filename is invalid"
        )
    if artifact_file.name != INDEX_ARTIFACT_FILENAME:
        raise SnapshotIntegrityError(
            "terminology snapshot artifact filename is invalid"
        )

    actual_artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if manifest_payload.get("artifact_sha256") != actual_artifact_sha256:
        raise SnapshotIntegrityError(
            "terminology artifact sha256 does not match its manifest"
        )

    system_uri = _normalize_system_uri(manifest_payload.get("system_uri"))
    release_version = _normalize_release_version(
        manifest_payload.get("release_version")
    )
    content_hash = _normalize_digest(
        manifest_payload.get("content_hash"), "content_hash"
    )
    index_system = manifest_payload.get("index_system")
    if not isinstance(index_system, str) or not index_system.strip():
        raise SnapshotIntegrityError(
            "terminology snapshot index_system must be non-empty"
        )
    snapshot_key = manifest_payload.get("snapshot_key")
    if not isinstance(snapshot_key, str):
        raise SnapshotIntegrityError("terminology snapshot key must be a string")
    expected_key = _snapshot_key(system_uri, release_version, content_hash)
    if snapshot_key != expected_key:
        raise SnapshotIntegrityError("terminology snapshot key failed validation")
    if _CACHE_KEY_RE.fullmatch(
        directory.name
    ) and directory.name != _cache_key_directory(snapshot_key):
        raise SnapshotIntegrityError(
            "terminology snapshot directory is not content-addressed"
        )

    if expected_system_uri is not None and system_uri != _normalize_system_uri(
        expected_system_uri
    ):
        raise SnapshotIntegrityError(
            "terminology snapshot system URI does not match request"
        )
    if (
        expected_release_version is not None
        and release_version != _normalize_release_version(expected_release_version)
    ):
        raise SnapshotIntegrityError(
            "terminology snapshot release version does not match request"
        )
    if expected_content_hash is not None and content_hash != _normalize_digest(
        expected_content_hash, "content_hash"
    ):
        raise SnapshotIntegrityError(
            "terminology snapshot content hash does not match request"
        )

    index = _index_from_payload(artifact)
    if index.system != index_system:
        raise SnapshotIntegrityError(
            "terminology snapshot index system does not match its manifest"
        )
    raw_count = manifest_payload.get("concept_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        raise SnapshotIntegrityError("terminology snapshot concept_count is invalid")
    if raw_count != index.concept_count:
        raise SnapshotIntegrityError(
            "terminology snapshot concept count does not match its manifest"
        )
    if index.content_hash != content_hash:
        raise SnapshotIntegrityError(
            "terminology snapshot content hash failed validation"
        )

    return TerminologySnapshot(
        index=index,
        system_uri=system_uri,
        release_version=release_version,
        content_hash=content_hash,
        artifact_path=artifact_file,
        manifest_path=manifest_file,
        cache_hit=True,
    )


def store_snapshot(
    snapshot_or_index: TerminologySnapshot | VocabularyIndex,
    system_uri: str | None = None,
    release_version: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    content_hash: str | None = None,
    restricted: bool | None = None,
    allow_restricted: bool = False,
) -> TerminologySnapshot:
    """Store one snapshot through a short-lived :class:`SnapshotCache`."""

    return TerminologySnapshotCache(
        cache_dir,
        allow_restricted=allow_restricted,
    ).store(
        snapshot_or_index,
        system_uri,
        release_version,
        content_hash=content_hash,
        restricted=restricted,
    )


def load_or_build_snapshot(
    system_uri: str,
    release_version: str,
    builder: Callable[[], VocabularyIndex | TerminologySnapshot],
    *,
    cache: TerminologySnapshotCache | None = None,
    cache_dir: str | Path | None = None,
    content_hash: str | None = None,
    restricted: bool | None = None,
    allow_restricted: bool = False,
) -> TerminologySnapshot:
    """Load a pinned snapshot or build and persist it on a cache miss."""

    if cache is not None and cache_dir is not None:
        raise ValueError("pass either cache or cache_dir, not both")
    resolved_cache = cache or TerminologySnapshotCache(
        cache_dir,
        allow_restricted=allow_restricted,
    )
    return resolved_cache.load_or_build(
        system_uri,
        release_version,
        builder,
        content_hash=content_hash,
        restricted=restricted,
    )


def clear_cache(*, cache_dir: str | Path | None = None) -> int:
    """Remove persisted terminology snapshots from one cache directory."""

    return TerminologySnapshotCache(cache_dir).clear()


def _coerce_snapshot(
    value: TerminologySnapshot | VocabularyIndex,
    *,
    system_uri: str | None,
    release_version: str | None,
    content_hash: str | None,
) -> TerminologySnapshot:
    if isinstance(value, TerminologySnapshot):
        if system_uri is not None and value.system_uri != _normalize_system_uri(
            system_uri
        ):
            raise SnapshotIntegrityError(
                "built snapshot system URI does not match request"
            )
        if (
            release_version is not None
            and value.release_version != _normalize_release_version(release_version)
        ):
            raise SnapshotIntegrityError(
                "built snapshot release version does not match request"
            )
        if content_hash is not None and value.content_hash != _normalize_digest(
            content_hash, "content_hash"
        ):
            raise SnapshotIntegrityError(
                "built snapshot content hash does not match request"
            )
        return value
    if not isinstance(value, VocabularyIndex):
        raise TypeError("builder must return a VocabularyIndex or TerminologySnapshot")
    if system_uri is None or release_version is None:
        raise ValueError("system_uri and release_version are required for an index")
    normalized_hash = value.content_hash
    if content_hash is not None:
        expected_hash = _normalize_digest(content_hash, "content_hash")
        if normalized_hash != expected_hash:
            raise SnapshotIntegrityError(
                "built vocabulary content hash does not match the requested pin"
            )
    return TerminologySnapshot(
        index=value,
        system_uri=system_uri,
        release_version=release_version,
        content_hash=normalized_hash,
    )


def _index_payload(index: VocabularyIndex) -> dict[str, Any]:
    concepts = sorted(
        (_concept_payload(concept) for concept in index.concepts),
        key=lambda row: (
            row["system"],
            row["code"],
            row["preferred_term"],
            tuple(row["synonyms"]),
            tuple(sorted(row["language_aliases"].items())),
        ),
    )
    return {
        "concepts": concepts,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "system": index.system,
    }


def _concept_payload(concept: VocabConcept) -> dict[str, Any]:
    return {
        "code": concept.code,
        "language_aliases": {
            language: list(aliases)
            for language, aliases in sorted(concept.language_aliases.items())
        },
        "preferred_term": concept.preferred_term,
        "synonyms": list(concept.synonyms),
        "system": concept.system,
    }


def _index_from_payload(payload: Mapping[str, Any]) -> VocabularyIndex:
    if set(payload) != {"concepts", "schema_version", "system"}:
        raise SnapshotIntegrityError("terminology index artifact fields are invalid")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotIntegrityError("unsupported terminology index schema version")
    system = payload.get("system")
    concepts_payload = payload.get("concepts")
    if not isinstance(system, str) or not system.strip():
        raise SnapshotIntegrityError("terminology index system must be non-empty")
    if not isinstance(concepts_payload, list) or not concepts_payload:
        raise SnapshotIntegrityError("terminology index concepts must be non-empty")

    concepts: list[VocabConcept] = []
    for row in concepts_payload:
        if not isinstance(row, Mapping) or set(row) != {
            "code",
            "language_aliases",
            "preferred_term",
            "synonyms",
            "system",
        }:
            raise SnapshotIntegrityError("terminology concept fields are invalid")
        row_system = row.get("system")
        code = row.get("code")
        preferred_term = row.get("preferred_term")
        synonyms = row.get("synonyms")
        language_aliases = row.get("language_aliases")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                row_system,
                code,
                preferred_term,
            )
        ):
            raise SnapshotIntegrityError("terminology concept text fields are invalid")
        if not isinstance(synonyms, list) or not all(
            isinstance(value, str) for value in synonyms
        ):
            raise SnapshotIntegrityError("terminology concept synonyms are invalid")
        if not isinstance(language_aliases, Mapping):
            raise SnapshotIntegrityError(
                "terminology concept language aliases are invalid"
            )
        normalized_languages: dict[str, tuple[str, ...]] = {}
        for language, aliases in language_aliases.items():
            if not isinstance(language, str) or not isinstance(aliases, list):
                raise SnapshotIntegrityError("terminology language aliases are invalid")
            if not all(isinstance(value, str) for value in aliases):
                raise SnapshotIntegrityError("terminology language aliases are invalid")
            normalized_languages[language] = tuple(aliases)
        if row_system != system:
            raise SnapshotIntegrityError(
                "terminology concept system does not match index"
            )
        concepts.append(
            VocabConcept(
                system=row_system,
                code=code,
                preferred_term=preferred_term,
                synonyms=tuple(synonyms),
                language_aliases=normalized_languages,
            )
        )
    try:
        return VocabularyIndex(system, concepts)
    except (TypeError, ValueError) as exc:
        raise SnapshotIntegrityError(
            "terminology index could not be reconstructed"
        ) from exc


def _normalize_system_uri(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotIntegrityError("system_uri must be non-empty text")
    normalized = value.strip()
    if any(character.isspace() for character in normalized):
        raise SnapshotIntegrityError("system_uri must not contain whitespace")
    if _URI_SCHEME_RE.match(normalized):
        return normalized
    alias = _SYSTEM_URI_ALIASES.get(normalized.casefold())
    if alias is None:
        raise SnapshotIntegrityError(
            "system_uri must be an absolute URI or known vocabulary id"
        )
    return alias


def _normalize_release_version(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotIntegrityError("release_version must be non-empty text")
    normalized = value.strip()
    if any(ord(character) < 32 for character in normalized):
        raise SnapshotIntegrityError(
            "release_version must not contain control characters"
        )
    return normalized


def _normalize_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise SnapshotIntegrityError(f"{field_name} must be a SHA-256 digest")
    match = _DIGEST_RE.fullmatch(value.strip())
    if match is None:
        raise SnapshotIntegrityError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _restricted_system(system_uri: str, restricted: bool | None) -> bool:
    if restricted is not None:
        return bool(restricted)
    return system_uri.rstrip("/").casefold() in {
        uri.casefold() for uri in _RESTRICTED_SYSTEM_URIS
    }


def _snapshot_key(system_uri: str, release_version: str, content_hash: str) -> str:
    payload = {
        "content_hash": content_hash,
        "release_version": release_version,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "system_uri": system_uri,
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def _cache_key_directory(snapshot_key: str) -> str:
    match = _DIGEST_RE.fullmatch(snapshot_key)
    if match is None:
        raise SnapshotIntegrityError("snapshot key must be a SHA-256 digest")
    return match.group(1).lower()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_limited_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise SnapshotIntegrityError(f"{label} exceeds {max_bytes} bytes")
    return content


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
