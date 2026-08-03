"""Free clinical vocabulary loader and alias index.

This module provides the shared substrate for concept grounding. It accepts
normalized vocabulary files for deterministic local use and has a checksum
guarded download path for public artifacts. UMLS and SNOMED CT are deliberately
outside this loader because they require a separate user-key-gated workflow.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from openmed.core.offline import is_local_only, raise_offline_error

from .systems import canonical_system, system_uri

FREE_VOCAB_SYSTEMS = ("rxnorm", "icd10cm", "loinc", "hpo", "mesh")
RESTRICTED_VOCAB_SYSTEMS = ("umls", "snomed")
DEFAULT_CACHE_DIR = Path("~/.cache/openmed/grounding")
SUPPORTED_DATA_SUFFIXES = {".csv", ".jsonl", ".obo", ".rrf", ".tsv", ".txt", ".xml"}
_CHECKSUM_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")
_LANGUAGE_ALIAS_KEY_RE = re.compile(
    r"^(?:(?:aliases?|synonyms?|alt_labels?)_"
    r"([a-z]{2,3}(?:[-_][a-z0-9]{2,8})*)|"
    r"([a-z]{2,3}(?:[-_][a-z0-9]{2,8})*)_"
    r"(?:aliases?|synonyms?|alt_labels?))$"
)
_LANGUAGE_ALIAS_MAP_KEYS = (
    "language_aliases",
    "aliases_by_language",
    "localized_aliases",
    "multilingual_aliases",
)

_SYSTEM_ALIASES = {
    "rxnorm": "rxnorm",
    "rx-norm": "rxnorm",
    "rx_norm": "rxnorm",
    "icd10": "icd10cm",
    "icd10cm": "icd10cm",
    "icd-10": "icd10cm",
    "icd-10-cm": "icd10cm",
    "icd_10_cm": "icd10cm",
    "loinc": "loinc",
    "hpo": "hpo",
    "hp": "hpo",
    "mesh": "mesh",
    "ms": "mesh",
}

_RESTRICTED_ALIASES = {
    "umls": "umls",
    "snomed": "snomed",
    "snomedct": "snomed",
    "snomed-ct": "snomed",
    "snomed_ct": "snomed",
    "sct": "snomed",
}


class VocabLoaderError(RuntimeError):
    """Base error raised by the vocabulary loader."""


class VocabularyNotFoundError(VocabLoaderError):
    """Raised when no cached/local/downloadable vocabulary artifact exists."""


class VocabularyChecksumError(VocabLoaderError):
    """Raised when a vocabulary artifact checksum is missing or invalid."""


class RestrictedVocabularyError(VocabLoaderError):
    """Raised when a restricted vocabulary is requested from this loader."""


@dataclass(frozen=True)
class VocabSource:
    """Source metadata for one vocabulary artifact.

    Args:
        system: Vocabulary system key such as ``"rxnorm"`` or ``"loinc"``.
        url: Optional public artifact URL used when the cache is empty.
        sha256: Expected SHA-256 for a local or downloaded artifact.
        checksum_url: Optional URL containing a SHA-256 checksum.
        path: Optional local normalized artifact path, primarily for tests or
            pre-staged deployments.
        archive_member: Optional archive member to extract from a zip file.
        artifact_name: Cached filename for non-archive downloads.
        license_note: Human-readable provenance and redistribution note.
        version: Optional release/version label for the snapshot.
    """

    system: str
    url: str | None = None
    sha256: str | None = None
    checksum_url: str | None = None
    path: str | Path | None = None
    archive_member: str | None = None
    artifact_name: str = "concepts.tsv"
    license_note: str = ""
    version: str | None = None


@dataclass(frozen=True)
class SnapshotManifest:
    """Immutable metadata for one locally imported terminology snapshot."""

    system: str
    version: str
    artifact: str
    sha256: str
    system_uri: str
    format: str
    license_note: str = ""

    def __post_init__(self) -> None:
        normalized = canonical_system(self.system)
        if normalized not in FREE_VOCAB_SYSTEMS:
            raise ValueError("snapshot manifests are limited to free vocabularies")
        if not self.version.strip():
            raise ValueError("snapshot version must not be blank")
        if not self.artifact.strip():
            raise ValueError("snapshot artifact must not be blank")
        if not _CHECKSUM_RE.fullmatch(self.sha256.removeprefix("sha256:")):
            raise VocabularyChecksumError(
                f"Invalid snapshot SHA-256 checksum for {self.artifact!r}."
            )
        object.__setattr__(self, "system", normalized)
        object.__setattr__(self, "sha256", _digest_label(self.sha256))
        object.__setattr__(self, "system_uri", system_uri(normalized) or "")

    def to_dict(self) -> dict[str, str]:
        """Return deterministic JSON metadata without source content."""

        return {
            "artifact": self.artifact,
            "format": self.format,
            "license_note": self.license_note,
            "sha256": self.sha256,
            "system": self.system,
            "system_uri": self.system_uri,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SnapshotManifest":
        """Parse and validate a manifest mapping."""

        required = ("system", "version", "artifact", "sha256", "format")
        missing = [key for key in required if not payload.get(key)]
        if missing:
            raise VocabularyChecksumError(
                "snapshot manifest is missing required fields: " + ", ".join(missing)
            )
        return cls(
            system=str(payload["system"]),
            version=str(payload["version"]),
            artifact=str(payload["artifact"]),
            sha256=str(payload["sha256"]),
            system_uri=str(payload.get("system_uri") or ""),
            format=str(payload["format"]),
            license_note=str(payload.get("license_note") or ""),
        )


@dataclass(frozen=True)
class VocabConcept:
    """One vocabulary concept with preferred text and aliases."""

    system: str
    code: str
    preferred_term: str
    synonyms: tuple[str, ...] = ()
    language_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "synonyms", _unique_text_values(self.synonyms))
        object.__setattr__(
            self,
            "language_aliases",
            _normalize_language_aliases(self.language_aliases),
        )

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return preferred term followed by unique synonyms."""

        values: list[str] = []
        for value in (self.preferred_term, *self.synonyms):
            if value and value not in values:
                values.append(value)
        return tuple(values)

    def aliases_for_language(self, language: object | None = None) -> tuple[str, ...]:
        """Return aliases for ``language``, falling back to English aliases."""

        normalized = normalize_language(language)
        if normalized == "en":
            return self.aliases
        aliases = self.language_aliases.get(normalized, ())
        return aliases or self.aliases


class VocabularyIndex:
    """Mapping-like alias index for one vocabulary system."""

    def __init__(self, system: str, concepts: Iterable[VocabConcept]) -> None:
        self.system = _normalize_system(system)
        self._concepts = tuple(concepts)
        alias_map: dict[str, list[VocabConcept]] = defaultdict(list)
        language_maps: dict[str, dict[str, list[VocabConcept]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for concept in self._concepts:
            for alias in concept.aliases:
                normalized = normalize_alias(alias)
                if normalized:
                    alias_map[normalized].append(concept)
                    language_maps["en"][normalized].append(concept)
            for language, aliases in concept.language_aliases.items():
                for alias in aliases:
                    normalized = normalize_alias(alias)
                    if normalized:
                        language_maps[language][normalized].append(concept)
        self._alias_map = {key: tuple(value) for key, value in alias_map.items()}
        self._language_alias_maps = {
            language: {key: tuple(value) for key, value in aliases.items()}
            for language, aliases in language_maps.items()
        }

    @property
    def concepts(self) -> tuple[VocabConcept, ...]:
        """Concepts in source order."""

        return self._concepts

    @property
    def aliases(self) -> tuple[str, ...]:
        """Normalized aliases available in the index."""

        return tuple(self._alias_map)

    def aliases_for_language(self, language: object | None = None) -> tuple[str, ...]:
        """Normalized aliases available for ``language`` with English fallback."""

        return tuple(self._alias_map_for_language(language))

    @property
    def concept_count(self) -> int:
        """Number of concepts in the index."""

        return len(self._concepts)

    @property
    def content_hash(self) -> str:
        """Return a stable SHA-256 hash of the indexed concept content.

        The digest is computed over each concept's system, code, preferred term,
        normalized English aliases, and language-tagged aliases in canonical
        (sorted) order, so two indexes built from the same vocabulary content
        hash identically regardless of row order. English-only snapshots retain
        the original digest contract. Downstream consumers store this as a
        candidate's ``vocab_version`` so a grounding result can be traced to its
        complete multilingual snapshot.
        """

        cached = getattr(self, "_content_hash", None)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        rows = sorted(
            (
                concept.system,
                concept.code,
                concept.preferred_term,
                "\x1f".join(
                    sorted(normalize_alias(alias) for alias in concept.aliases)
                ),
            )
            for concept in self._concepts
        )
        for row in rows:
            digest.update("\x1e".join(row).encode("utf-8"))
            digest.update(b"\x1d")
        localized_rows = sorted(
            (
                concept.system,
                concept.code,
                normalize_language(language),
                "\x1f".join(sorted(normalize_alias(alias) for alias in aliases)),
            )
            for concept in self._concepts
            for language, aliases in concept.language_aliases.items()
        )
        if localized_rows:
            digest.update(b"\x1cLANGUAGE_ALIASES\x1c")
            for row in localized_rows:
                digest.update("\x1e".join(row).encode("utf-8"))
                digest.update(b"\x1d")
        content_hash = f"sha256:{digest.hexdigest()}"
        self._content_hash = content_hash
        return content_hash

    def lookup(
        self, alias: object, *, language: object | None = None
    ) -> VocabConcept | None:
        """Return the first concept matching an alias, if present."""

        matches = self.lookup_all(alias, language=language)
        return matches[0] if matches else None

    def lookup_all(
        self, alias: object, *, language: object | None = None
    ) -> tuple[VocabConcept, ...]:
        """Return all concepts matching an alias."""

        normalized_alias = normalize_alias(alias)
        if not normalized_alias:
            return ()
        alias_map = self._alias_map_for_language(language)
        matches = alias_map.get(normalized_alias, ())
        if matches:
            return matches
        source_language = normalize_language(language)
        if source_language != "en":
            return self._alias_map.get(normalized_alias, ())
        return ()

    def code_for(self, alias: object, *, language: object | None = None) -> str | None:
        """Return the first code matching an alias, if present."""

        concept = self.lookup(alias, language=language)
        return concept.code if concept else None

    def fuzzy_lookup(
        self,
        alias: object,
        *,
        limit: int = 5,
        score_cutoff: float = 80.0,
        language: object | None = None,
    ) -> tuple[tuple[VocabConcept, float], ...]:
        """Return approximate alias matches using the ``grounding`` extra."""

        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise VocabLoaderError(
                "Install openmed[grounding] to use fuzzy vocabulary lookup."
            ) from exc

        normalized = normalize_alias(alias)
        if not normalized:
            return ()
        alias_map = self._alias_map_for_language(language)

        matches = process.extract(
            normalized,
            alias_map.keys(),
            scorer=fuzz.WRatio,
            limit=limit,
            score_cutoff=score_cutoff,
        )
        results: list[tuple[VocabConcept, float]] = []
        seen: set[tuple[str, str]] = set()
        for matched_alias, score, _ in matches:
            for concept in alias_map[matched_alias]:
                key = (concept.system, concept.code)
                if key not in seen:
                    results.append((concept, float(score)))
                    seen.add(key)
        if not results and normalize_language(language) != "en":
            return self.fuzzy_lookup(
                alias,
                limit=limit,
                score_cutoff=score_cutoff,
                language="en",
            )
        return tuple(results)

    def get(
        self,
        alias: object,
        default: str | None = None,
        *,
        language: object | None = None,
    ) -> str | None:
        """Return the first code matching an alias or ``default``."""

        return self.code_for(alias, language=language) or default

    def __contains__(self, alias: object) -> bool:
        return normalize_alias(alias) in self._alias_map

    def __getitem__(self, alias: object) -> str:
        concept = self.lookup(alias)
        if concept is None:
            raise KeyError(alias)
        return concept.code

    def __len__(self) -> int:
        return len(self._alias_map)

    def _alias_map_for_language(
        self, language: object | None = None
    ) -> Mapping[str, tuple[VocabConcept, ...]]:
        source_language = normalize_language(language)
        if source_language == "en":
            return self._alias_map
        return self._language_alias_maps.get(source_language) or self._alias_map


DEFAULT_VOCAB_SOURCES: Mapping[str, VocabSource] = {
    "rxnorm": VocabSource(
        system="rxnorm",
        url="https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip",
        archive_member="RXNCONSO.RRF",
        license_note="NLM RxNorm public release; no UMLS metathesaurus bundle.",
    ),
    "icd10cm": VocabSource(
        system="icd10cm",
        url="https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip",
        license_note="CMS ICD-10-CM code descriptions public release.",
    ),
    "loinc": VocabSource(
        system="loinc",
        url="https://loinc.org/download/loinc-table-file-csv/",
        license_note="LOINC table file public distribution page.",
    ),
    "hpo": VocabSource(
        system="hpo",
        url="https://purl.obolibrary.org/obo/hp.obo",
        artifact_name="hp.obo",
        license_note="Human Phenotype Ontology OBO public release.",
    ),
    "mesh": VocabSource(
        system="mesh",
        url="https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml",
        artifact_name="desc2026.xml",
        license_note="NLM MeSH descriptor XML public release.",
    ),
}


class VocabLoader:
    """Load free clinical vocabularies from cache, local files, or URLs."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        local_only: bool = False,
        registry: Mapping[str, VocabSource] | None = None,
        downloader: Callable[[str, Path, float], None] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.cache_dir = _default_cache_dir(cache_dir)
        self.local_only = local_only
        self.registry = _normalize_registry(registry or DEFAULT_VOCAB_SOURCES)
        self.downloader = downloader or _download_url
        self.timeout = timeout
        self._indexes: dict[str, VocabularyIndex] = {}

    def get_index(self, system: str) -> VocabularyIndex:
        """Return a queryable alias index for a free vocabulary system."""

        normalized = _normalize_system(system)
        cached = self._indexes.get(normalized)
        if cached is not None:
            return cached

        source = self.registry.get(normalized)
        if source is None:
            raise VocabularyNotFoundError(
                f"No free vocabulary source is registered for {system!r}."
            )

        data_path = self._resolve_data_path(normalized, source)
        concepts = tuple(_read_concepts(normalized, data_path))
        if not concepts:
            raise VocabularyNotFoundError(
                f"No concepts could be read for {normalized!r} from {data_path}."
            )
        index = VocabularyIndex(normalized, concepts)
        self._indexes[normalized] = index
        return index

    def import_snapshot(
        self,
        system: str,
        path: str | Path,
        *,
        version: str,
        sha256: str | None = None,
        license_note: str = "",
        replace: bool = False,
    ) -> SnapshotManifest:
        """Import and checksum-pin a local free-vocabulary snapshot.

        The source is copied into the loader cache and paired with a small
        deterministic manifest. Import never opens a network connection and
        never stores caller credentials or source-note content.
        """

        normalized = _normalize_system(system)
        source_path = Path(path).expanduser()
        data_path = _resolve_local_source(normalized, source_path, sha256)
        actual_sha256 = _file_sha256(data_path)
        expected = _digest_label(sha256) if sha256 is not None else actual_sha256
        if expected != actual_sha256:
            raise VocabularyChecksumError(
                f"Checksum mismatch for {data_path}: expected {expected}, "
                f"got {actual_sha256}."
            )

        cache_root = self.cache_dir / normalized
        cache_root.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_root / "manifest.json"
        if manifest_path.exists() and not replace:
            raise VocabLoaderError(
                f"A {normalized!r} snapshot is already imported; pass replace=True."
            )

        artifact_name = f"concepts{data_path.suffix.lower()}"
        destination = cache_root / artifact_name
        shutil.copy2(data_path, destination)
        manifest = SnapshotManifest(
            system=normalized,
            version=version,
            artifact=artifact_name,
            sha256=actual_sha256,
            system_uri=system_uri(normalized) or "",
            format=data_path.suffix.lower().lstrip(".") or "text",
            license_note=license_note,
        )
        _write_manifest(manifest_path, manifest)
        self._indexes.pop(normalized, None)
        return manifest

    def download_snapshot(self, system: str) -> SnapshotManifest:
        """Download, verify, materialize, and manifest one configured snapshot."""

        normalized = _normalize_system(system)
        source = self.registry.get(normalized)
        if source is None:
            raise VocabularyNotFoundError(
                f"No free vocabulary source is registered for {normalized!r}."
            )
        if self.local_only or is_local_only():
            raise_offline_error(f"vocabulary download for {normalized}")
        if source.url is None:
            raise VocabularyNotFoundError(
                f"No download URL is configured for {normalized!r}."
            )
        expected_sha256 = source.sha256 or _fetch_checksum(source.checksum_url)
        if not expected_sha256:
            raise VocabularyChecksumError(
                f"Refusing to download {normalized!r} without a SHA-256 checksum."
            )
        cache_root = self.cache_dir / normalized
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"openmed-{normalized}-") as tmp:
            download_path = Path(tmp) / (
                source.artifact_name or f"{normalized}.download"
            )
            self.downloader(source.url, download_path, self.timeout)
            _verify_sha256(download_path, expected_sha256)
            data_path = _materialize_artifact(
                normalized, download_path, cache_root, source
            )
        manifest = SnapshotManifest(
            system=normalized,
            version=source.version or _digest_label(expected_sha256),
            artifact=_artifact_name(cache_root, data_path),
            sha256=_file_sha256(data_path),
            system_uri=system_uri(normalized) or "",
            format=data_path.suffix.lower().lstrip(".") or "text",
            license_note=source.license_note,
        )
        _write_manifest(cache_root / "manifest.json", manifest)
        self._indexes.pop(normalized, None)
        return manifest

    def snapshot_manifest(self, system: str) -> SnapshotManifest:
        """Return the verified manifest for a loaded local snapshot."""

        normalized = _normalize_system(system)
        source = self.registry.get(normalized)
        data_path = self._resolve_data_path(normalized, source)
        manifest = _read_manifest(data_path.parent / "manifest.json")
        if manifest is not None:
            if manifest.system != normalized:
                raise VocabularyChecksumError(
                    f"Snapshot manifest system does not match {normalized!r}."
                )
            _verify_sha256(data_path, manifest.sha256)
            return manifest

        digest = _file_sha256(data_path)
        return SnapshotManifest(
            system=normalized,
            version=(source.version if source is not None else None)
            or _digest_label(digest),
            artifact=data_path.name,
            sha256=digest,
            system_uri=system_uri(normalized) or "",
            format=data_path.suffix.lower().lstrip(".") or "text",
            license_note=source.license_note if source is not None else "",
        )

    def snapshot_provenance(self, systems: Sequence[str]) -> dict[str, dict[str, str]]:
        """Return stable, PHI-free provenance for requested snapshots."""

        result: dict[str, dict[str, str]] = {}
        for system in systems:
            normalized = _normalize_system(system)
            index = self.get_index(normalized)
            manifest = self.snapshot_manifest(normalized)
            result[normalized] = {
                "system": normalized,
                "system_uri": manifest.system_uri,
                "version": manifest.version,
                "sha256": manifest.sha256,
                "content_hash": index.content_hash,
                "artifact": manifest.artifact,
            }
        return result

    def list_snapshots(self) -> tuple[SnapshotManifest, ...]:
        """List imported snapshots in deterministic system order."""

        manifests: list[SnapshotManifest] = []
        for system in sorted(self.registry):
            manifest_path = self.cache_dir / system / "manifest.json"
            manifest = _read_manifest(manifest_path)
            if manifest is not None:
                manifests.append(manifest)
        return tuple(manifests)

    def _resolve_data_path(self, system: str, source: VocabSource) -> Path:
        cache_root = self.cache_dir / system
        manifest = _read_manifest(cache_root / "manifest.json")
        if manifest is not None:
            data_path = (cache_root / manifest.artifact).resolve()
            if not data_path.is_file() or not data_path.is_relative_to(
                cache_root.resolve()
            ):
                raise VocabularyChecksumError(
                    f"Snapshot artifact for {system!r} is missing or unsafe."
                )
            _verify_sha256(data_path, manifest.sha256)
            return data_path

        cached = _find_index_file(cache_root, system)
        if cached is not None:
            if source.sha256 is not None:
                _verify_sha256(cached, source.sha256)
            return cached

        if source.path is not None:
            data_path = _resolve_local_source(system, Path(source.path), source.sha256)
            return data_path

        if self.local_only or is_local_only():
            raise_offline_error(
                "vocabulary download for "
                f"{system}; place a normalized artifact under {cache_root}"
            )

        if source.url is None:
            raise VocabularyNotFoundError(
                f"No cached {system!r} vocabulary exists at {cache_root}, and no "
                "download URL is configured. Provide VocabSource(path=...) or "
                "pre-stage concepts.tsv/concepts.csv/concepts.jsonl in the cache."
            )

        expected_sha256 = source.sha256 or _fetch_checksum(source.checksum_url)
        if not expected_sha256:
            raise VocabularyChecksumError(
                f"Refusing to download {system!r} without a SHA-256 checksum. "
                "Pass VocabSource(sha256=...) or checksum_url=..."
            )

        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"openmed-{system}-") as tmp:
            download_path = Path(tmp) / (source.artifact_name or f"{system}.download")
            self.downloader(source.url, download_path, self.timeout)
            _verify_sha256(download_path, expected_sha256)
            data_path = _materialize_artifact(system, download_path, cache_root, source)
            manifest = SnapshotManifest(
                system=system,
                version=source.version or _digest_label(expected_sha256),
                artifact=_artifact_name(cache_root, data_path),
                sha256=_file_sha256(data_path),
                system_uri=system_uri(system) or "",
                format=data_path.suffix.lower().lstrip(".") or "text",
                license_note=source.license_note,
            )
            _write_manifest(cache_root / "manifest.json", manifest)
            return data_path


def get_index(
    system: str,
    *,
    cache_dir: str | Path | None = None,
    local_only: bool = False,
) -> VocabularyIndex:
    """Convenience accessor for ``VocabLoader(...).get_index(system)``."""

    return VocabLoader(cache_dir=cache_dir, local_only=local_only).get_index(system)


def normalize_alias(value: object) -> str:
    """Normalize aliases for case-insensitive exact matching."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"\s+", " ", text.casefold()).strip()
    return text


def normalize_language(value: object | None, *, default: str = "en") -> str:
    """Normalize a source-language tag for alias selection."""

    if value in (None, ""):
        return default
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = text.replace("_", "-")
    text = re.sub(r"[^a-z0-9-]+", "", text)
    language = text.split("-", 1)[0]
    return language or default


def _normalize_system(system: str) -> str:
    key = normalize_alias(system).replace(" ", "-")
    if key in _RESTRICTED_ALIASES:
        restricted = _RESTRICTED_ALIASES[key]
        raise RestrictedVocabularyError(
            f"{restricted!r} is not available from the free vocabulary loader. "
            "Use the separate user-key-gated terminology path for UMLS/SNOMED CT."
        )
    if key in _SYSTEM_ALIASES:
        return _SYSTEM_ALIASES[key]
    raise VocabularyNotFoundError(
        f"Unsupported free vocabulary system {system!r}. Expected one of "
        f"{', '.join(FREE_VOCAB_SYSTEMS)}."
    )


def _normalize_registry(registry: Mapping[str, VocabSource]) -> dict[str, VocabSource]:
    normalized: dict[str, VocabSource] = {}
    for key, source in registry.items():
        system = _normalize_system(source.system or key)
        normalized[system] = source
    return normalized


def _default_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env_cache = os.getenv("OPENMED_CACHE_DIR")
    root = Path(env_cache).expanduser() if env_cache else DEFAULT_CACHE_DIR.expanduser()
    return root / "grounding" if root.name != "grounding" else root


def _resolve_local_source(
    system: str,
    source_path: Path,
    expected_sha256: str | None,
) -> Path:
    expanded = source_path.expanduser()
    if not expanded.exists():
        raise VocabularyNotFoundError(f"Vocabulary source does not exist: {expanded}")
    data_path = _find_index_file(expanded, system) if expanded.is_dir() else expanded
    if data_path is None:
        raise VocabularyNotFoundError(
            f"No supported vocabulary file found under {expanded}."
        )
    if expected_sha256 is not None:
        _verify_sha256(data_path, expected_sha256)
    return data_path


def _download_url(url: str, target: Path, timeout: float) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _fetch_checksum(url: str | None) -> str | None:
    if not url:
        return None
    if is_local_only():
        raise_offline_error(f"checksum lookup for {url}")
    with urllib.request.urlopen(url, timeout=30.0) as response:
        text = response.read().decode("utf-8", errors="replace")
    match = _CHECKSUM_RE.search(text)
    return match.group(0).lower() if match else None


def _verify_sha256(path: Path, expected_sha256: str) -> None:
    expected = expected_sha256.strip().lower().removeprefix("sha256:")
    if not _CHECKSUM_RE.fullmatch(expected):
        raise VocabularyChecksumError(
            f"Invalid SHA-256 checksum for {path}: {expected_sha256!r}"
        )

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise VocabularyChecksumError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}."
        )


def _file_sha256(path: Path) -> str:
    """Return a content digest in the manifest's stable labeled form."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return _digest_label(digest.hexdigest())


def _artifact_name(root: Path, path: Path) -> str:
    """Return a safe manifest-relative artifact path."""

    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise VocabularyChecksumError(
            f"Snapshot artifact {path.name!r} is outside its cache directory."
        ) from exc
    return relative.as_posix()


def _digest_label(value: str) -> str:
    """Normalize a bare or labeled SHA-256 digest."""

    digest = value.strip().lower().removeprefix("sha256:")
    if not _CHECKSUM_RE.fullmatch(digest):
        raise VocabularyChecksumError(f"Invalid SHA-256 checksum: {value!r}")
    return f"sha256:{digest}"


def _read_manifest(path: Path) -> SnapshotManifest | None:
    """Read a manifest if present, rejecting malformed metadata."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VocabularyChecksumError(
            f"Unable to read snapshot manifest {path.name!r}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise VocabularyChecksumError("snapshot manifest must contain an object")
    return SnapshotManifest.from_dict(payload)


def _write_manifest(path: Path, manifest: SnapshotManifest) -> None:
    """Write a canonical manifest atomically."""

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _materialize_artifact(
    system: str,
    artifact: Path,
    cache_root: Path,
    source: VocabSource,
) -> Path:
    if zipfile.is_zipfile(artifact):
        _extract_zip(artifact, cache_root, source.archive_member)
        data_path = _find_index_file(cache_root, system)
        if data_path is None:
            raise VocabularyNotFoundError(
                f"Downloaded archive for {system!r} did not contain a supported "
                "vocabulary file."
            )
        return data_path

    target = cache_root / (source.artifact_name or f"concepts{artifact.suffix}")
    shutil.copy2(artifact, target)
    return target


def _extract_zip(artifact: Path, target_dir: Path, archive_member: str | None) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(artifact) as archive:
        members = archive.infolist()
        if archive_member is not None:
            members = [
                member
                for member in members
                if member.filename == archive_member
                or member.filename.endswith("/" + archive_member)
            ]
            if not members:
                raise VocabularyNotFoundError(
                    f"Archive member {archive_member!r} was not found in {artifact}."
                )
        for member in members:
            destination = (target_dir / member.filename).resolve()
            if not destination.is_relative_to(target_root):
                raise VocabularyNotFoundError(
                    f"Unsafe archive path {member.filename!r} in {artifact}."
                )
            archive.extract(member, target_dir)


def _find_index_file(root: Path, system: str) -> Path | None:
    if not root.exists():
        return None
    if root.is_file():
        return root if root.suffix.lower() in SUPPORTED_DATA_SUFFIXES else None

    candidates = [
        "concepts.tsv",
        "concepts.csv",
        "concepts.jsonl",
        f"{system}.tsv",
        f"{system}.csv",
        f"{system}.jsonl",
        "vocab.tsv",
        "vocab.csv",
        "vocab.jsonl",
        "hp.obo",
        "desc2026.xml",
        "RXNCONSO.RRF",
    ]
    for name in candidates:
        path = root / name
        if path.exists() and path.is_file():
            return path

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_SUFFIXES:
            return path
    return None


def _read_concepts(system: str, path: Path) -> Iterable[VocabConcept]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _read_jsonl(system, path)
    elif suffix in {".csv", ".tsv"}:
        yield from _read_delimited(system, path)
    elif suffix == ".obo":
        yield from _read_obo(system, path)
    elif suffix == ".xml":
        yield from _read_mesh_xml(system, path)
    elif suffix == ".rrf" or path.name.upper() == "RXNCONSO.RRF":
        yield from _read_rxnorm_rrf(path)
    elif suffix == ".txt":
        yield from _read_text_table(system, path)
    else:
        raise VocabularyNotFoundError(f"Unsupported vocabulary file type: {path}")


def _read_jsonl(system: str, path: Path) -> Iterable[VocabConcept]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            concept = _concept_from_mapping(system, row, source=f"{path}:{line_number}")
            if concept is not None:
                yield concept


def _read_delimited(system: str, path: Path) -> Iterable[VocabConcept]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row_number, row in enumerate(reader, start=2):
            concept = _concept_from_mapping(system, row, source=f"{path}:{row_number}")
            if concept is not None:
                yield concept


def _read_obo(system: str, path: Path) -> Iterable[VocabConcept]:
    current: dict[str, Any] = {}
    in_term = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                concept = _obo_concept(system, current, path)
                if concept is not None:
                    yield concept
                current = {"synonyms": []}
                in_term = True
                continue
            if line.startswith("["):
                concept = _obo_concept(system, current, path)
                if concept is not None:
                    yield concept
                current = {}
                in_term = False
                continue
            if not in_term or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if key == "id":
                current["code"] = value
            elif key == "name":
                current["preferred_term"] = value
            elif key == "synonym":
                match = re.match(r'"(.+?)"', value)
                if match:
                    current.setdefault("synonyms", []).append(match.group(1))

    concept = _obo_concept(system, current, path)
    if concept is not None:
        yield concept


def _obo_concept(
    system: str, row: Mapping[str, Any], path: Path
) -> VocabConcept | None:
    if not row:
        return None
    return _concept_from_mapping(system, row, source=str(path))


def _read_mesh_xml(system: str, path: Path) -> Iterable[VocabConcept]:
    for _, element in ElementTree.iterparse(path, events=("end",)):
        tag = _strip_xml_namespace(element.tag)
        if tag != "DescriptorRecord":
            continue
        code = _find_xml_text(element, "DescriptorUI")
        preferred = _find_xml_text(element, "DescriptorName/String")
        synonyms = [
            text
            for text in _find_xml_texts(
                element, "ConceptList/Concept/TermList/Term/String"
            )
            if text and text != preferred
        ]
        if code and preferred:
            yield VocabConcept(
                system=system,
                code=code,
                preferred_term=preferred,
                synonyms=tuple(synonyms),
                source=str(path),
            )
        element.clear()


def _read_rxnorm_rrf(path: Path) -> Iterable[VocabConcept]:
    by_code: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 15:
                continue
            language = fields[1]
            is_preferred = fields[6] == "Y"
            code = fields[7]
            term = fields[14]
            suppress = fields[16] if len(fields) > 16 else ""
            if language != "ENG" or not code or not term or suppress == "O":
                continue
            entry = by_code.setdefault(code, {"synonyms": []})
            if is_preferred and not entry.get("preferred_term"):
                entry["preferred_term"] = term
            elif term not in entry["synonyms"]:
                entry["synonyms"].append(term)

    for code, row in by_code.items():
        preferred = row.get("preferred_term") or next(iter(row["synonyms"]), "")
        if preferred:
            yield VocabConcept(
                system="rxnorm",
                code=code,
                preferred_term=preferred,
                synonyms=tuple(term for term in row["synonyms"] if term != preferred),
                source=str(path),
            )


def _read_text_table(system: str, path: Path) -> Iterable[VocabConcept]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.readline()
        handle.seek(0)
        if "\t" in sample:
            reader = csv.DictReader(handle, delimiter="\t")
            for row_number, row in enumerate(reader, start=2):
                concept = _concept_from_mapping(
                    system,
                    row,
                    source=f"{path}:{row_number}",
                )
                if concept is not None:
                    yield concept
            return

        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = re.split(r"\s{2,}", stripped, maxsplit=2)
            if len(parts) >= 2:
                code = parts[1] if parts[0].isdigit() and len(parts) > 2 else parts[0]
                preferred = parts[-1]
                yield VocabConcept(
                    system=system,
                    code=code,
                    preferred_term=preferred,
                    source=f"{path}:{line_number}",
                )


def _concept_from_mapping(
    system: str,
    row: Mapping[str, Any],
    *,
    source: str,
) -> VocabConcept | None:
    normalized_row = {str(key).strip().lower(): value for key, value in row.items()}
    row_system = _first_value(normalized_row, "system", "vocabulary", "code_system")
    if row_system and _normalize_system(str(row_system)) != system:
        return None
    code = _first_value(
        normalized_row,
        "code",
        "concept_id",
        "concept_code",
        "id",
        "loinc_num",
        "descriptorui",
        "rxcui",
    )
    preferred = _first_value(
        normalized_row,
        "preferred_term",
        "canonical_term",
        "preferred",
        "term",
        "name",
        "display",
        "long_common_name",
        "shortname",
        "description",
    )
    if not code or not preferred:
        return None
    synonyms = _split_synonyms(
        _first_value(normalized_row, "synonyms", "aliases", "alias", "alt_labels")
    )
    language_aliases = _language_aliases_from_mapping(normalized_row)
    return VocabConcept(
        system=system,
        code=str(code).strip(),
        preferred_term=str(preferred).strip(),
        synonyms=synonyms,
        language_aliases=language_aliases,
        source=source,
    )


def _first_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _split_synonyms(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = re.split(r"\s*[|;]\s*", value)
    elif isinstance(value, Sequence):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    return tuple(item.strip() for item in values if item and item.strip())


def _language_aliases_from_mapping(
    row: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, list[str]] = defaultdict(list)
    for key in _LANGUAGE_ALIAS_MAP_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            value = _parse_json_mapping(value)
        if isinstance(value, Mapping):
            for language, language_values in value.items():
                aliases[normalize_language(language)].extend(
                    _split_synonyms(language_values)
                )

    for key, value in row.items():
        match = _LANGUAGE_ALIAS_KEY_RE.match(key)
        if not match:
            continue
        language = normalize_language(match.group(1) or match.group(2))
        aliases[language].extend(_split_synonyms(value))

    return {
        language: _unique_text_values(values)
        for language, values in aliases.items()
        if _unique_text_values(values)
    }


def _parse_json_mapping(value: str) -> Mapping[str, Any] | None:
    stripped = value.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _normalize_language_aliases(
    aliases: Mapping[str, Sequence[str] | str],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for language, values in aliases.items():
        normalized_values = _unique_text_values(_split_synonyms(values))
        if normalized_values:
            normalized[normalize_language(language)] = normalized_values
    return normalized


def _unique_text_values(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _strip_xml_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_xml_text(element: ElementTree.Element, path: str) -> str | None:
    texts = _find_xml_texts(element, path)
    return texts[0] if texts else None


def _find_xml_texts(element: ElementTree.Element, path: str) -> list[str]:
    parts = path.split("/")
    current = [element]
    for part in parts:
        next_elements: list[ElementTree.Element] = []
        for candidate in current:
            next_elements.extend(
                child
                for child in list(candidate)
                if _strip_xml_namespace(child.tag) == part
            )
        current = next_elements
    return [node.text.strip() for node in current if node.text and node.text.strip()]
