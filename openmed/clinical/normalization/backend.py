"""Terminology backend contracts for clinical concept normalization.

OpenMed ships only a permissively licensed synthetic backend here. Real
terminology catalogs stay outside the package and are accessed through the
``TerminologyBackend`` protocol implemented by the caller.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "BackendIdentity",
    "CodeSystemMetadata",
    "SYNTHETIC_CODE_SYSTEMS",
    "SYNTHETIC_CONCEPTS",
    "SyntheticTerminologyBackend",
    "TerminologyBackend",
    "TerminologyConcept",
    "normalize_surface",
    "validate_backend_identity",
]


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
