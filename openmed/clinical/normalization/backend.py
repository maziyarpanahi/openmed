"""Terminology backend contracts for clinical concept normalization.

OpenMed ships only a permissively licensed synthetic backend here. Real
terminology catalogs stay outside the package and are accessed through the
``TerminologyBackend`` protocol implemented by the caller.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "BackendIdentity",
    "CodeSystemMetadata",
    "EmbeddingBackendStatus",
    "EmbeddingFallbackResult",
    "MultilingualEmbedder",
    "MultilingualEmbeddingBackend",
    "SYNTHETIC_CODE_SYSTEMS",
    "SYNTHETIC_CONCEPTS",
    "SyntheticTerminologyBackend",
    "TerminologyBackend",
    "TerminologyConcept",
    "embedding_fallback",
    "normalize_surface",
    "validate_backend_identity",
]

#: Optional dependency imported lazily by the multilingual embedding backend.
#: The default install never imports it; it is resolved only when an enabled
#: backend actually needs to load an offline model.
DEFAULT_EMBEDDING_DEPENDENCY = "sentence_transformers"


_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalize_surface(text: str) -> str:
    """Return a deterministic, case-insensitive surface form.

    The normalization is intentionally mechanical and local: Unicode
    compatibility normalization, case-folding, punctuation folding to spaces,
    and whitespace collapse. It never consults a terminology catalog.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    value = unicodedata.normalize("NFKC", text).casefold()
    value = _NON_ALNUM_RE.sub(" ", value)
    return " ".join(value.split())


def _normalize_source_language(value: object | None, *, default: str = "en") -> str:
    """Return a normalized source-language tag for fallback provenance.

    Mirrors the grounding loader's language normalization so a source language
    threaded into the embedding fallback resolves to the same code the sparse
    candidate generator stamps on its candidates: Unicode NFKC, case folding,
    region-subtag stripping, and an English default when no language is given.
    Keeping the rule local preserves this module's import-light, self-contained
    design (it never imports the grounding package to normalize a tag).
    """

    if value in (None, ""):
        return default
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = text.replace("_", "-")
    text = re.sub(r"[^a-z0-9-]+", "", text)
    language = text.split("-", 1)[0]
    return language or default


@dataclass(frozen=True)
class BackendIdentity:
    """Stable identity used to isolate cache entries by backend version."""

    name: str
    version: str

    def __post_init__(self) -> None:
        validate_backend_identity(self)

    def cache_payload(self) -> dict[str, str]:
        """Return the JSON-safe identity payload included in cache keys."""

        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class CodeSystemMetadata:
    """Metadata for one code system exposed by a terminology backend."""

    system_id: str
    uri: str
    version: str
    license: str = "synthetic"


@dataclass(frozen=True)
class TerminologyConcept:
    """One coded concept returned by a terminology backend."""

    system_id: str
    system_uri: str
    code: str
    display: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    version: str | None = None
    source: str = "synthetic"

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable dedupe key for this concept."""

        return (self.system_uri, self.code)

    @property
    def normalized_terms(self) -> tuple[str, ...]:
        """Return normalized display and alias surfaces for matching."""

        return _unique_ordered(
            normalize_surface(term) for term in (self.display, *self.aliases)
        )


@runtime_checkable
class TerminologyBackend(Protocol):
    """Protocol implemented by caller-supplied terminology adapters."""

    @property
    def identity(self) -> BackendIdentity:
        """Return a non-empty backend name and version for cache isolation."""

    def lookup(self, normalized: str) -> Sequence[TerminologyConcept]:
        """Return concepts whose display or alias exactly matches ``normalized``."""

    def candidates(self, tokens: Sequence[str]) -> Sequence[TerminologyConcept]:
        """Return candidate concepts sharing at least one blocking token."""

    def code_systems(self) -> Sequence[CodeSystemMetadata]:
        """Return code-system metadata exposed by the backend."""


SYNTHETIC_CODE_SYSTEMS: tuple[CodeSystemMetadata, ...] = (
    CodeSystemMetadata(
        system_id="synthetic-condition",
        uri="https://openmed.ai/fhir/CodeSystem/synthetic-condition",
        version="2026.06-synthetic",
    ),
    CodeSystemMetadata(
        system_id="synthetic-observation",
        uri="https://openmed.ai/fhir/CodeSystem/synthetic-observation",
        version="2026.06-synthetic",
    ),
    CodeSystemMetadata(
        system_id="synthetic-treatment",
        uri="https://openmed.ai/fhir/CodeSystem/synthetic-treatment",
        version="2026.06-synthetic",
    ),
)

_SYSTEM_BY_ID = {system.system_id: system for system in SYNTHETIC_CODE_SYSTEMS}

SYNTHETIC_CONCEPTS: tuple[TerminologyConcept, ...] = (
    TerminologyConcept(
        system_id="synthetic-condition",
        system_uri=_SYSTEM_BY_ID["synthetic-condition"].uri,
        code="SYN-COND-001",
        display="Aster fever",
        aliases=("aster pyrexia", "a fever pattern"),
        version=_SYSTEM_BY_ID["synthetic-condition"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-condition",
        system_uri=_SYSTEM_BY_ID["synthetic-condition"].uri,
        code="SYN-COND-002",
        display="Beryl cough pattern",
        aliases=("beryl cough", "bc pattern"),
        version=_SYSTEM_BY_ID["synthetic-condition"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-condition",
        system_uri=_SYSTEM_BY_ID["synthetic-condition"].uri,
        code="SYN-COND-003",
        display="Corin skin flare",
        aliases=("corin rash", "skin flare corin"),
        version=_SYSTEM_BY_ID["synthetic-condition"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-condition",
        system_uri=_SYSTEM_BY_ID["synthetic-condition"].uri,
        code="SYN-COND-004",
        display="Dax ankle strain",
        aliases=("dax ankle sprain", "ankle strain dax"),
        version=_SYSTEM_BY_ID["synthetic-condition"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-observation",
        system_uri=_SYSTEM_BY_ID["synthetic-observation"].uri,
        code="SYN-OBS-001",
        display="Elin sugar panel",
        aliases=("elin glucose panel", "esp"),
        version=_SYSTEM_BY_ID["synthetic-observation"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-observation",
        system_uri=_SYSTEM_BY_ID["synthetic-observation"].uri,
        code="SYN-OBS-002",
        display="Faren breath score",
        aliases=("faren breathing score", "fbs"),
        version=_SYSTEM_BY_ID["synthetic-observation"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-observation",
        system_uri=_SYSTEM_BY_ID["synthetic-observation"].uri,
        code="SYN-OBS-003",
        display="Galen red cell count",
        aliases=("galen rcc", "grcc"),
        version=_SYSTEM_BY_ID["synthetic-observation"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-observation",
        system_uri=_SYSTEM_BY_ID["synthetic-observation"].uri,
        code="SYN-OBS-004",
        display="Halo pain scale",
        aliases=("halo pain rating", "hps"),
        version=_SYSTEM_BY_ID["synthetic-observation"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-treatment",
        system_uri=_SYSTEM_BY_ID["synthetic-treatment"].uri,
        code="SYN-TREAT-001",
        display="Iona sleep coaching",
        aliases=("iona sleep plan", "isc"),
        version=_SYSTEM_BY_ID["synthetic-treatment"].version,
    ),
    TerminologyConcept(
        system_id="synthetic-treatment",
        system_uri=_SYSTEM_BY_ID["synthetic-treatment"].uri,
        code="SYN-TREAT-002",
        display="Juno blue tablet",
        aliases=("juno tablet blue", "jbt"),
        version=_SYSTEM_BY_ID["synthetic-treatment"].version,
    ),
)


class SyntheticTerminologyBackend:
    """Small in-memory backend using synthetic concepts only."""

    def __init__(
        self,
        *,
        identity: BackendIdentity | None = None,
        concepts: Sequence[TerminologyConcept] = SYNTHETIC_CONCEPTS,
        systems: Sequence[CodeSystemMetadata] = SYNTHETIC_CODE_SYSTEMS,
    ) -> None:
        self._identity = identity or BackendIdentity(
            name="openmed-synthetic-terminology",
            version="2026.06",
        )
        self._concepts = tuple(concepts)
        self._systems = tuple(systems)
        self._lookup_index: dict[str, list[TerminologyConcept]] = defaultdict(list)
        self._token_index: dict[str, list[TerminologyConcept]] = defaultdict(list)
        for concept in self._concepts:
            for normalized in concept.normalized_terms:
                self._lookup_index[normalized].append(concept)
                for token in normalized.split():
                    self._token_index[token].append(concept)

    @property
    def identity(self) -> BackendIdentity:
        return self._identity

    def lookup(self, normalized: str) -> tuple[TerminologyConcept, ...]:
        normalized = normalize_surface(normalized)
        return tuple(self._lookup_index.get(normalized, ()))

    def candidates(self, tokens: Sequence[str]) -> tuple[TerminologyConcept, ...]:
        keys = []
        for token in tokens:
            normalized = normalize_surface(token)
            if normalized:
                keys.append(normalized)
        seen: set[tuple[str, str]] = set()
        results: list[TerminologyConcept] = []
        for token in keys:
            for concept in self._token_index.get(token, ()):
                if concept.key in seen:
                    continue
                seen.add(concept.key)
                results.append(concept)
        return tuple(results)

    def code_systems(self) -> tuple[CodeSystemMetadata, ...]:
        return self._systems


@runtime_checkable
class MultilingualEmbedder(Protocol):
    """Offline sentence embedder used by the optional multilingual fallback.

    An embedder maps a sequence of surface texts to fixed-width dense vectors.
    Implementations must be fully local and never perform network access. Tests
    and callers may inject any object exposing this single method, which keeps
    the fallback usable without the heavyweight optional dependency.
    """

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one dense vector per input text, in input order."""


@dataclass(frozen=True)
class EmbeddingBackendStatus:
    """Availability of the optional multilingual embedding backend.

    ``available`` is False whenever the backend is disabled, its optional
    dependency is missing, or no offline model is configured; ``reason`` carries
    a short human-readable explanation for provenance and logs.
    """

    available: bool
    reason: str


@dataclass(frozen=True)
class EmbeddingFallbackResult:
    """Outcome of an embedding-fallback ranking attempt.

    ``used_embedding`` is True only when an enabled, available backend actually
    re-ranked the candidate pool by multilingual similarity. When it is False
    the ``concepts`` tuple holds the unchanged lexical ordering, so callers
    degrade to lexical-only ranking without any exception during grounding.

    ``source_language`` records the resolved source-language code the fallback
    was invoked for, so a downstream grounded record can expose the mention's
    language consistently with the sparse candidate generator. It defaults to
    ``"en"``, keeping existing English results and their serialized output
    unchanged when no source language is threaded through.
    """

    concepts: tuple[TerminologyConcept, ...]
    used_embedding: bool
    status: EmbeddingBackendStatus
    source_language: str = "en"


class MultilingualEmbeddingBackend:
    """Opt-in, offline multilingual similarity fallback for zero-alias mentions.

    This backend is disabled by default: construct it with ``enabled=True`` (and
    optionally an injected offline ``embedder`` or a local ``model_path``) to
    turn it on. The optional embedding dependency is imported lazily the first
    time a model is loaded, so importing OpenMed and running lexical grounding
    never imports it. When the backend is disabled, when the optional dependency
    is not installed, when no offline model is configured, or when embedding
    itself fails, ranking degrades gracefully to the supplied lexical order and
    never raises.

    The fallback is intended to trigger only when no target-language alias
    exists: :meth:`fallback` returns the lexical candidates unchanged whenever
    they are non-empty, and only consults the embedding model for the
    zero-alias case.

    Args:
        enabled: Whether the backend may load a model and re-rank. Defaults to
            ``False`` so the default install stays lexical and dependency-light.
        embedder: An offline :class:`MultilingualEmbedder` to use directly. When
            provided the optional dependency is never imported.
        model_path: Local filesystem path to an offline model, loaded through
            the optional dependency. Required (in the absence of ``embedder``)
            for the backend to be available; no remote identifiers are accepted,
            which keeps loading offline.
        dependency: Import name of the optional embedding library. Defaults to
            :data:`DEFAULT_EMBEDDING_DEPENDENCY`.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        embedder: MultilingualEmbedder | None = None,
        model_path: str | None = None,
        dependency: str = DEFAULT_EMBEDDING_DEPENDENCY,
    ) -> None:
        self._enabled = bool(enabled)
        self._embedder = embedder
        self._model_path = model_path
        self._dependency = dependency
        self._loaded: MultilingualEmbedder | None = None
        self._load_attempted = False

    @property
    def enabled(self) -> bool:
        """Return whether the backend is opted in."""

        return self._enabled

    def status(self) -> EmbeddingBackendStatus:
        """Return the current availability status without raising."""

        if not self._enabled:
            return EmbeddingBackendStatus(False, "embedding fallback disabled")
        if self._embedder is not None:
            return EmbeddingBackendStatus(True, "ready")
        if importlib.util.find_spec(self._dependency) is None:
            return EmbeddingBackendStatus(
                False,
                f"optional dependency '{self._dependency}' is not installed",
            )
        if not self._model_path:
            return EmbeddingBackendStatus(False, "no offline model configured")
        return EmbeddingBackendStatus(True, "ready")

    @property
    def available(self) -> bool:
        """Return whether an enabled backend can currently embed."""

        return self.status().available

    def _load_embedder(self) -> MultilingualEmbedder | None:
        if not self._enabled:
            return None
        if self._embedder is not None:
            return self._embedder
        if self._loaded is not None or self._load_attempted:
            return self._loaded
        self._load_attempted = True
        if not self._model_path:
            return None
        if not os.path.exists(self._model_path):
            # Offline-only, enforced in code: a non-local path (e.g. a remote
            # model id) is treated as unavailable rather than triggering a
            # download. Never import the dependency for a non-local path.
            return None
        try:
            module = importlib.import_module(self._dependency)
            model_factory = getattr(module, "SentenceTransformer", None)
            if model_factory is None:
                return None
            # Load strictly from the local path so no download is triggered.
            self._loaded = model_factory(self._model_path)
        except Exception:  # pragma: no cover - exercised via unavailable paths
            self._loaded = None
        return self._loaded

    def rank(
        self,
        mention: str,
        concepts: Sequence[TerminologyConcept],
        *,
        source_language: str | None = None,
    ) -> EmbeddingFallbackResult:
        """Re-rank ``concepts`` by multilingual similarity to ``mention``.

        Returns the concepts unchanged (``used_embedding=False``) whenever the
        backend is disabled, unavailable, or the embedding step fails, so this
        never raises during normal grounding.

        Args:
            mention: Surface span text whose language the fallback matches.
            concepts: Candidate pool to re-rank.
            source_language: Source language of the mention. The normalized code
                is stamped on the result's ``source_language`` for provenance;
                it defaults to English so existing results are unchanged.
        """

        resolved_language = _normalize_source_language(source_language)
        pool = tuple(concepts)
        embedder = self._load_embedder()
        if embedder is None or not pool:
            return EmbeddingFallbackResult(
                pool, False, self.status(), resolved_language
            )
        try:
            texts = [normalize_surface(mention), *(_concept_text(c) for c in pool)]
            vectors = [
                tuple(float(value) for value in row) for row in embedder.encode(texts)
            ]
            if len(vectors) != len(pool) + 1:
                # A ragged embedder output would silently drop concepts via zip;
                # degrade to lexical instead of returning a truncated pool.
                return EmbeddingFallbackResult(
                    pool, False, self.status(), resolved_language
                )
            mention_vector = vectors[0]
            scored = sorted(
                zip(pool, vectors[1:]),
                key=lambda item: (-_cosine(mention_vector, item[1]), item[0].key),
            )
        except Exception:  # pragma: no cover - defensive graceful degradation
            return EmbeddingFallbackResult(
                pool, False, self.status(), resolved_language
            )
        return EmbeddingFallbackResult(
            tuple(concept for concept, _ in scored),
            True,
            EmbeddingBackendStatus(True, "ready"),
            resolved_language,
        )

    def fallback(
        self,
        mention: str,
        concepts: Sequence[TerminologyConcept],
        *,
        lexical_candidates: Sequence[TerminologyConcept] = (),
        source_language: str | None = None,
    ) -> EmbeddingFallbackResult:
        """Apply the embedding fallback only when no lexical alias matched.

        When ``lexical_candidates`` is non-empty a target-language alias already
        exists, so those candidates are returned unchanged. Only the zero-alias
        case consults the embedding model, via :meth:`rank`. The resolved
        ``source_language`` is stamped on the result in both paths so a grounded
        record exposes the mention's language consistently; it defaults to
        English, leaving existing English results unchanged.
        """

        resolved_language = _normalize_source_language(source_language)
        lexical = tuple(lexical_candidates)
        if lexical:
            return EmbeddingFallbackResult(
                lexical,
                False,
                EmbeddingBackendStatus(self.available, "lexical alias present"),
                resolved_language,
            )
        return self.rank(mention, concepts, source_language=resolved_language)


def embedding_fallback(
    mention: str,
    concepts: Sequence[TerminologyConcept],
    *,
    lexical_candidates: Sequence[TerminologyConcept] = (),
    backend: MultilingualEmbeddingBackend | None = None,
    source_language: str | None = None,
) -> EmbeddingFallbackResult:
    """Rank ``concepts`` by the optional multilingual fallback, or lexically.

    Convenience wrapper around :class:`MultilingualEmbeddingBackend`. When no
    backend is supplied a disabled backend is used, so the default behavior is
    lexical-only and imports no embedding dependency. ``source_language`` threads
    the mention's language through to the result's provenance, defaulting to
    English. See :meth:`MultilingualEmbeddingBackend.fallback` for the zero-alias
    contract.
    """

    active = backend if backend is not None else MultilingualEmbeddingBackend()
    return active.fallback(
        mention,
        concepts,
        lexical_candidates=lexical_candidates,
        source_language=source_language,
    )


def _concept_text(concept: TerminologyConcept) -> str:
    terms = concept.normalized_terms
    return terms[0] if terms else normalize_surface(concept.display)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    size = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def validate_backend_identity(identity: object) -> BackendIdentity:
    """Return a validated backend identity or raise a cache-safety error."""

    if isinstance(identity, BackendIdentity):
        name = identity.name
        version = identity.version
    elif isinstance(identity, Mapping):
        name = identity.get("name")
        version = identity.get("version")
    else:
        name = getattr(identity, "name", None)
        version = getattr(identity, "version", None)

    if not isinstance(name, str) or not name.strip():
        raise ValueError("terminology backend identity must include a name")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("terminology backend identity must include a version")
    return (
        identity
        if isinstance(identity, BackendIdentity)
        else BackendIdentity(name=name.strip(), version=version.strip())
    )


def _unique_ordered(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)
