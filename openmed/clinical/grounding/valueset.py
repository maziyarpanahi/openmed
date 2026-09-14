"""Local-first FHIR ValueSet and delegated ECL expansion.

The engine expands extensional FHIR ``ValueSet`` resources entirely in memory
over caller-loaded, redistributable vocabularies.  Canonical ValueSet URLs and
ECL expressions are sent only when the caller explicitly configures a
terminology endpoint.  OpenMed bundles no terminology content and never
persists expansion members unless a caller supplies an expansion cache.

Remote results that contain restricted terminology are not cached unless the
cache was created with ``allow_restricted=True``.  Cache identities contain a
hash of the canonical URL or ECL expression, not the raw expression, and every
result records deterministic version and digest provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from openmed.clinical.exporters.valueset import ValueSetSource, load_valueset

from .ecl import validate_ecl_syntax
from .matcher import LexicalConcept
from .registry import VocabularyLoaderRegistry
from .restricted import RESTRICTED_SYSTEM_URIS
from .snapshot_cache import TerminologySnapshot, TerminologySnapshotCache
from .vocab import VocabConcept, VocabularyIndex

__all__ = [
    "EXPANSION_ARTIFACT_FILENAME",
    "EXPANSION_MANIFEST_FILENAME",
    "EXPANSION_SCHEMA_VERSION",
    "ExpansionProvenance",
    "ExpansionSourceKind",
    "ValueSetExpansion",
    "ValueSetExpansionCache",
    "ValueSetExpansionConfigurationError",
    "ValueSetExpansionEngine",
    "ValueSetExpansionError",
    "ValueSetExpansionPolicyError",
    "ValueSetExpansionResponseError",
    "ValueSetExpansionResult",
    "ValueSetExpansionUnsupportedError",
    "ValueSetExpander",
    "ValueSetMember",
    "expand_valueset",
]

EXPANSION_SCHEMA_VERSION = 1
EXPANSION_ARTIFACT_FILENAME = "expansion.json"
EXPANSION_MANIFEST_FILENAME = "manifest.json"

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_CACHE_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_CACHE_MANIFEST_BYTES = 64 * 1024
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_PAGE_SIZE = 1_000
_DEFAULT_MAX_PAGES = 100
_DEFAULT_MAX_MEMBERS = 1_000_000
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_CACHE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_SNOMED_SYSTEM_URI = RESTRICTED_SYSTEM_URIS["snomed"]
_RESTRICTED_URIS = frozenset(
    value.rstrip("/").casefold() for value in RESTRICTED_SYSTEM_URIS.values()
)

ExpansionSourceKind: TypeAlias = Literal["local", "valueset-url", "ecl"]
LocalVocabulary: TypeAlias = (
    TerminologySnapshot | VocabularyIndex | Mapping[str, object] | object
)


class ValueSetExpansionError(RuntimeError):
    """Base error for ValueSet or ECL expansion."""


class ValueSetExpansionConfigurationError(ValueSetExpansionError, ValueError):
    """Raised when remote expansion was requested without safe configuration."""


class ValueSetExpansionResponseError(ValueSetExpansionError):
    """Raised when a terminology response is invalid or incomplete."""


class ValueSetExpansionUnsupportedError(ValueSetExpansionError):
    """Raised for an intensional local clause that requires delegation."""


class ValueSetExpansionPolicyError(ValueSetExpansionError):
    """Raised when restricted terminology would cross a disallowed boundary."""


@dataclass(frozen=True, order=True)
class ValueSetMember:
    """One code-system member in an expansion.

    Args:
        system: Canonical code-system URI.
        code: Code within ``system``.
        version: Optional code-system version supplied by the ValueSet,
            terminology server, or loaded terminology snapshot.
    """

    system: str
    code: str
    version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "system", _absolute_uri(self.system, "system"))
        object.__setattr__(self, "code", _nonempty_text(self.code, "code"))
        if self.version is not None:
            object.__setattr__(
                self,
                "version",
                _nonempty_text(self.version, "member version"),
            )

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable code identity without its release version."""

        return (self.system, self.code)

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic FHIR Coding subset."""

        result = {"system": self.system, "code": self.code}
        if self.version is not None:
            result["version"] = self.version
        return result


@dataclass(frozen=True)
class ExpansionProvenance:
    """Version and digest evidence for one expansion.

    Raw ECL is deliberately excluded.  ``request_sha256`` binds the result to
    the exact canonical URL, local ValueSet JSON, or ECL expression without
    copying that input into reports or cache metadata.
    """

    source_kind: ExpansionSourceKind
    method: str
    version: str
    request_sha256: str
    response_sha256: str
    valueset_url: str | None = None
    endpoint: str | None = None
    expansion_identifier: str | None = None
    vocabulary_versions: tuple[tuple[str, str], ...] = ()
    restricted: bool = False
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if self.source_kind not in {"local", "valueset-url", "ecl"}:
            raise ValueError("source_kind must be local, valueset-url, or ecl")
        object.__setattr__(self, "method", _nonempty_text(self.method, "method"))
        object.__setattr__(self, "version", _version_text(self.version))
        object.__setattr__(
            self,
            "request_sha256",
            _sha256_text(self.request_sha256, "request_sha256"),
        )
        object.__setattr__(
            self,
            "response_sha256",
            _sha256_text(self.response_sha256, "response_sha256"),
        )
        if self.valueset_url is not None:
            object.__setattr__(
                self,
                "valueset_url",
                _absolute_uri(self.valueset_url, "valueset_url"),
            )
        if self.endpoint is not None:
            object.__setattr__(self, "endpoint", _endpoint(self.endpoint))
        if self.expansion_identifier is not None:
            object.__setattr__(
                self,
                "expansion_identifier",
                _nonempty_text(
                    self.expansion_identifier,
                    "expansion_identifier",
                ),
            )
        normalized_versions = tuple(
            sorted(
                (
                    _absolute_uri(system, "vocabulary system"),
                    _version_text(version),
                )
                for system, version in self.vocabulary_versions
            )
        )
        object.__setattr__(self, "vocabulary_versions", normalized_versions)
        if not isinstance(self.restricted, bool) or not isinstance(
            self.cache_hit, bool
        ):
            raise TypeError("restricted and cache_hit must be booleans")

    def to_dict(self) -> dict[str, Any]:
        """Return stable, raw-query-free provenance."""

        return {
            "cache_hit": self.cache_hit,
            "endpoint": self.endpoint,
            "expansion_identifier": self.expansion_identifier,
            "method": self.method,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "restricted": self.restricted,
            "source_kind": self.source_kind,
            "valueset_url": self.valueset_url,
            "version": self.version,
            "vocabulary_versions": [
                {"system": system, "version": version}
                for system, version in self.vocabulary_versions
            ],
        }


@dataclass(frozen=True)
class ValueSetExpansion(AbstractSet[str]):
    """A deterministic member-code set with expansion provenance.

    Iteration and membership operate on the code-only set requested by the
    public API.  ``codings`` retains system URI and code-system version so
    callers can safely handle ValueSets spanning multiple code systems.
    """

    codings: tuple[ValueSetMember, ...]
    provenance: ExpansionProvenance
    _codes: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(
                set(self.codings),
                key=lambda member: (member.system, member.code, member.version or ""),
            )
        )
        object.__setattr__(self, "codings", normalized)
        object.__setattr__(
            self,
            "_codes",
            frozenset(member.code for member in normalized),
        )

    @property
    def members(self) -> frozenset[str]:
        """Return the immutable member-code set."""

        return self._codes

    @property
    def codes(self) -> frozenset[str]:
        """Alias for :attr:`members`."""

        return self._codes

    @property
    def member_codes(self) -> frozenset[str]:
        """Alias for :attr:`members`."""

        return self._codes

    @property
    def version(self) -> str:
        """Return the resolved ValueSet or expansion version stamp."""

        return self.provenance.version

    def __contains__(self, value: object) -> bool:
        return value in self._codes

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._codes))

    def __len__(self) -> int:
        return len(self._codes)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "codings": [member.to_dict() for member in self.codings],
            "members": sorted(self._codes),
            "provenance": self.provenance.to_dict(),
        }


ValueSetExpansionResult = ValueSetExpansion


class ValueSetExpansionCache:
    """User-controlled, versioned cache for code-only expansion results.

    Args:
        cache_dir: Existing terminology snapshot cache root or another
            caller-selected directory.  Expansion entries use its dedicated
            ``expansions`` namespace.
        allow_restricted: Permit raw restricted member codes to be persisted.
            This is false by default even when the caller supplies a directory.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        allow_restricted: bool = False,
    ) -> None:
        if not isinstance(cache_dir, (str, PathLike)):
            raise TypeError("cache_dir must be a caller-controlled path")
        self.cache_dir = Path(cache_dir).expanduser() / "expansions"
        self.allow_restricted = bool(allow_restricted)

    @classmethod
    def from_snapshot_cache(
        cls,
        cache: TerminologySnapshotCache,
    ) -> ValueSetExpansionCache:
        """Share the configured root and policy of a terminology cache."""

        if not isinstance(cache, TerminologySnapshotCache):
            raise TypeError("cache must be a TerminologySnapshotCache")
        return cls(
            cache.cache_dir,
            allow_restricted=cache.allow_restricted,
        )

    def load(
        self,
        source: str,
        version: str,
        *,
        source_kind: ExpansionSourceKind,
    ) -> ValueSetExpansion | None:
        """Load and verify a pinned expansion, or return ``None``."""

        normalized_version = _version_text(version)
        source_hash = _digest_text(source)
        key = _cache_key(source_kind, source_hash, normalized_version)
        directory = self.cache_dir / key
        try:
            manifest_bytes = _read_limited(
                directory / EXPANSION_MANIFEST_FILENAME,
                _MAX_CACHE_MANIFEST_BYTES,
            )
        except (OSError, ValueError):
            return None

        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            return None
        if not isinstance(manifest, Mapping):
            return None
        if set(manifest) != {
            "artifact_file",
            "artifact_sha256",
            "member_count",
            "restricted",
            "schema_version",
            "source_kind",
            "source_sha256",
            "version",
        }:
            return None
        if (
            manifest.get("schema_version") != EXPANSION_SCHEMA_VERSION
            or manifest.get("artifact_file") != EXPANSION_ARTIFACT_FILENAME
            or manifest.get("source_kind") != source_kind
            or manifest.get("source_sha256") != source_hash
            or manifest.get("version") != normalized_version
        ):
            return None
        restricted = manifest.get("restricted")
        if not isinstance(restricted, bool):
            return None
        if restricted and not self.allow_restricted:
            return None
        try:
            artifact_bytes = _read_limited(
                directory / EXPANSION_ARTIFACT_FILENAME,
                _MAX_CACHE_ARTIFACT_BYTES,
            )
            artifact = json.loads(artifact_bytes)
        except (
            OSError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
        ):
            return None
        if (
            not isinstance(artifact, Mapping)
            or manifest.get("artifact_sha256")
            != hashlib.sha256(artifact_bytes).hexdigest()
        ):
            return None
        try:
            expansion = _expansion_from_payload(artifact)
        except (TypeError, ValueError):
            return None
        if (
            len(expansion.codings) != manifest.get("member_count")
            or expansion.version != normalized_version
            or expansion.provenance.restricted != restricted
            or (
                not self.allow_restricted
                and any(_is_restricted(member.system) for member in expansion.codings)
            )
        ):
            return None
        return replace(
            expansion,
            provenance=replace(expansion.provenance, cache_hit=True),
        )

    get = load

    def store(
        self,
        source: str,
        expansion: ValueSetExpansion,
    ) -> ValueSetExpansion:
        """Persist a verified expansion under its source and version."""

        if not isinstance(expansion, ValueSetExpansion):
            raise TypeError("expansion must be a ValueSetExpansion")
        if (
            expansion.provenance.restricted
            or any(_is_restricted(member.system) for member in expansion.codings)
        ) and not self.allow_restricted:
            raise ValueSetExpansionPolicyError(
                "restricted expansion caching requires allow_restricted=True"
            )
        source_hash = _digest_text(source)
        version = expansion.version
        key = _cache_key(expansion.provenance.source_kind, source_hash, version)
        directory = self.cache_dir / key
        directory.mkdir(parents=True, exist_ok=True)
        stored = replace(
            expansion,
            provenance=replace(expansion.provenance, cache_hit=False),
        )
        artifact_bytes = _canonical_json_bytes(stored.to_dict())
        manifest = {
            "artifact_file": EXPANSION_ARTIFACT_FILENAME,
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "member_count": len(stored.codings),
            "restricted": stored.provenance.restricted,
            "schema_version": EXPANSION_SCHEMA_VERSION,
            "source_kind": stored.provenance.source_kind,
            "source_sha256": source_hash,
            "version": version,
        }
        _atomic_write(directory / EXPANSION_ARTIFACT_FILENAME, artifact_bytes)
        _atomic_write(
            directory / EXPANSION_MANIFEST_FILENAME,
            _canonical_json_bytes(manifest),
        )
        return stored

    put = store

    def clear(self) -> int:
        """Remove only complete expansion entries from this cache."""

        if not self.cache_dir.is_dir():
            return 0
        removed = 0
        for directory in self.cache_dir.iterdir():
            if not directory.is_dir() or not _CACHE_KEY_RE.fullmatch(directory.name):
                continue
            artifact = directory / EXPANSION_ARTIFACT_FILENAME
            manifest = directory / EXPANSION_MANIFEST_FILENAME
            if not artifact.is_file() and not manifest.is_file():
                continue
            for path in (artifact, manifest):
                path.unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                continue
            removed += 1
        return removed


@dataclass(frozen=True)
class _LocalConcept:
    code: str
    display: str | None = None


@dataclass(frozen=True)
class _ResolvedVocabulary:
    system: str
    concepts: tuple[_LocalConcept, ...]
    version: str
    declared_version: bool


class ValueSetExpansionEngine:
    """Expand local ValueSets and explicitly delegated URLs or ECL.

    Args:
        endpoint: Caller-supplied FHIR terminology base URL.  No network client
            is created and no request is made when this is omitted.
        vocabularies: Loaded free vocabularies keyed by canonical system URI.
            Values may be ``TerminologySnapshot``, ``VocabularyIndex``, loader
            objects, or synthetic term-to-concept mappings.
        vocabulary_registry: Optional free-vocabulary loader registry consulted
            for systems not present in ``vocabularies``.
        cache: Explicit expansion cache or terminology snapshot cache whose
            user-controlled root and restricted-content policy should be used.
        cache_dir: Convenience path for an opt-in expansion cache.
        allow_restricted_cache: Allow restricted member codes in ``cache_dir``.
        client: Optional HTTPX-compatible client used for testing or custom
            transport.  The standard library is used when omitted.
        headers: Caller-supplied request headers.  Values are never logged.
        bearer_token: Optional caller-supplied bearer token, excluded from
            object representations and errors.
        timeout: Remote request timeout in seconds.
        page_size: Requested FHIR expansion page size.
        max_pages: Maximum remote pages accepted for one expansion.
        max_members: Maximum unique system/code members accepted.
        ecl_system_uri: SNOMED CT edition base used to construct the FHIR
            implicit-ValueSet URL for ECL delegation.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        vocabularies: Mapping[str, LocalVocabulary] | None = None,
        vocabulary_registry: VocabularyLoaderRegistry | None = None,
        cache: ValueSetExpansionCache | TerminologySnapshotCache | None = None,
        cache_dir: str | Path | None = None,
        allow_restricted_cache: bool = False,
        client: Any | None = None,
        headers: Mapping[str, str] | None = None,
        bearer_token: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int = _DEFAULT_MAX_PAGES,
        max_members: int = _DEFAULT_MAX_MEMBERS,
        ecl_system_uri: str = _SNOMED_SYSTEM_URI,
    ) -> None:
        if cache is not None and cache_dir is not None:
            raise ValueError("pass either cache or cache_dir, not both")
        self.endpoint = _endpoint(endpoint) if endpoint is not None else None
        self._vocabularies = dict(vocabularies or {})
        for system in self._vocabularies:
            _absolute_uri(system, "vocabulary system")
        if vocabulary_registry is not None and not isinstance(
            vocabulary_registry, VocabularyLoaderRegistry
        ):
            raise TypeError("vocabulary_registry must be a VocabularyLoaderRegistry")
        self._vocabulary_registry = vocabulary_registry
        if isinstance(cache, TerminologySnapshotCache):
            self.cache = ValueSetExpansionCache.from_snapshot_cache(cache)
        elif cache is None and cache_dir is not None:
            self.cache = ValueSetExpansionCache(
                cache_dir,
                allow_restricted=allow_restricted_cache,
            )
        elif cache is None or isinstance(cache, ValueSetExpansionCache):
            self.cache = cache
        else:
            raise TypeError("cache must be an expansion or terminology snapshot cache")
        self._client = client
        self._headers = _headers(headers or {})
        self._bearer_token = _optional_secret(bearer_token, "bearer_token")
        self.timeout = _positive_number(timeout, "timeout")
        self.page_size = _positive_integer(page_size, "page_size")
        self.max_pages = _positive_integer(max_pages, "max_pages")
        self.max_members = _positive_integer(max_members, "max_members")
        self.ecl_system_uri = _absolute_uri(ecl_system_uri, "ECL system URI")

    def expand_valueset(
        self,
        url_or_ecl: ValueSetSource,
        *,
        version: str | None = None,
    ) -> ValueSetExpansion:
        """Expand a local ValueSet, canonical URL, or ECL expression.

        Local mapping, inline JSON, and path inputs never use the endpoint.
        Absolute canonical URLs use FHIR ``ValueSet/$expand``.  Other non-path
        strings are treated as ECL after conservative syntax validation.
        Remote calls require ``endpoint``; there is no implicit fallback.
        """

        normalized_version = _version_text(version) if version is not None else None
        if isinstance(url_or_ecl, Mapping) or isinstance(url_or_ecl, PathLike):
            return self._expand_local(url_or_ecl, version=normalized_version)
        if not isinstance(url_or_ecl, str):
            raise TypeError("url_or_ecl must be a ValueSet mapping, path, URL, or ECL")

        stripped = url_or_ecl.strip()
        if not stripped:
            raise ValueError("url_or_ecl must not be blank")
        if stripped.lstrip().startswith(("{", "[")) or _is_local_file(stripped):
            return self._expand_local(stripped, version=normalized_version)
        if _URI_SCHEME_RE.match(stripped):
            canonical_url, canonical_version = _canonical_parts(stripped)
            if normalized_version is not None and canonical_version is not None:
                if normalized_version != canonical_version:
                    raise ValueError("canonical and requested ValueSet versions differ")
            return self._expand_remote(
                canonical_url,
                source_kind="valueset-url",
                version=normalized_version or canonical_version,
            )

        validate_ecl_syntax(stripped)
        return self._expand_remote(
            stripped,
            source_kind="ecl",
            version=normalized_version,
        )

    expand = expand_valueset

    def _expand_local(
        self,
        source: ValueSetSource,
        *,
        version: str | None,
    ) -> ValueSetExpansion:
        payload = load_valueset(source)
        payload_bytes = _canonical_json_bytes(payload)
        payload_hash = _digest_bytes(payload_bytes)
        valueset_url = _optional_uri(payload.get("url"), "ValueSet.url")
        declared_version = _optional_version(payload.get("version"))
        if version is not None and declared_version is not None:
            if version != declared_version:
                raise ValueError("requested version does not match ValueSet.version")
        resolved_version = version or declared_version or payload_hash
        restricted_hint = _payload_contains_restricted_system(payload)
        codings, vocabulary_versions, expansion_identifier = self._local_members(
            payload
        )
        if len(codings) > self.max_members:
            raise ValueSetExpansionResponseError(
                "local expansion exceeded the configured member limit"
            )
        restricted = restricted_hint or any(
            _is_restricted(member.system) for member in codings
        )
        response_hash = _members_digest(codings, resolved_version)
        # A canonical/version pair alone does not bind caller-loaded content.
        source_identity = f"urn:openmed:valueset:{payload_hash}:{response_hash}"
        cached = self._cache_load(
            source_identity, resolved_version, source_kind="local"
        )
        if cached is not None:
            return cached
        expansion = ValueSetExpansion(
            codings=codings,
            provenance=ExpansionProvenance(
                source_kind="local",
                method="local-extensional",
                version=resolved_version,
                request_sha256=payload_hash,
                response_sha256=response_hash,
                valueset_url=valueset_url,
                expansion_identifier=expansion_identifier,
                vocabulary_versions=vocabulary_versions,
                restricted=restricted,
            ),
        )
        return self._cache_store(source_identity, expansion)

    def _local_members(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[tuple[ValueSetMember, ...], tuple[tuple[str, str], ...], str | None]:
        expansion = payload.get("expansion")
        if isinstance(expansion, Mapping) and "contains" in expansion:
            members, item_count = _members_from_contains(expansion.get("contains"))
            _require_complete_expansion(expansion, item_count)
            identifier = _optional_text(expansion.get("identifier"))
            return (
                tuple(
                    sorted(
                        members,
                        key=lambda member: (
                            member.system,
                            member.code,
                            member.version or "",
                        ),
                    )
                ),
                (),
                identifier,
            )

        compose = payload.get("compose")
        if not isinstance(compose, Mapping):
            raise ValueSetExpansionUnsupportedError(
                "local ValueSet requires expansion.contains or compose.include"
            )
        includes = _mapping_sequence(compose.get("include"), "compose.include")
        if not includes:
            raise ValueSetExpansionUnsupportedError(
                "local ValueSet compose.include must not be empty"
            )
        excludes = _mapping_sequence(compose.get("exclude"), "compose.exclude")

        versions: dict[str, str] = {}
        included: dict[tuple[str, str], ValueSetMember] = {}
        for clause in includes:
            for member, vocabulary_version in self._expand_local_clause(clause):
                included[member.key] = member
                if vocabulary_version is not None:
                    versions[member.system] = vocabulary_version
        excluded: set[tuple[str, str]] = set()
        for clause in excludes:
            for member, vocabulary_version in self._expand_local_clause(clause):
                excluded.add(member.key)
                if vocabulary_version is not None:
                    versions[member.system] = vocabulary_version
        return (
            tuple(
                sorted(
                    member for key, member in included.items() if key not in excluded
                )
            ),
            tuple(sorted(versions.items())),
            None,
        )

    def _expand_local_clause(
        self,
        clause: Mapping[str, Any],
    ) -> tuple[tuple[ValueSetMember, str | None], ...]:
        if clause.get("valueSet") is not None:
            raise ValueSetExpansionUnsupportedError(
                "nested ValueSet references require terminology-server delegation"
            )
        system = _absolute_uri(clause.get("system"), "compose clause system")
        clause_version = _optional_version(clause.get("version"))
        concepts = _mapping_sequence(clause.get("concept"), "compose concept")
        filters = _mapping_sequence(clause.get("filter"), "compose filter")

        resolved: _ResolvedVocabulary | None = None
        needs_vocabulary = not concepts or bool(filters)
        if needs_vocabulary:
            if _is_restricted(system):
                raise ValueSetExpansionPolicyError(
                    "restricted vocabulary expansion must use the configured "
                    "out-of-process terminology endpoint"
                )
            resolved = self._resolve_vocabulary(system)
            if resolved is None:
                raise ValueSetExpansionUnsupportedError(
                    "local expansion requires a loaded free vocabulary for the "
                    "declared code system"
                )
            if (
                clause_version is not None
                and resolved.declared_version
                and clause_version != resolved.version
            ):
                raise ValueSetExpansionUnsupportedError(
                    "loaded vocabulary version does not match the ValueSet clause"
                )

        member_version = clause_version or (resolved.version if resolved else None)
        candidates: list[_LocalConcept]
        if concepts:
            candidates = []
            for concept in concepts:
                if concept.get("abstract") is True:
                    continue
                code = _nonempty_text(concept.get("code"), "compose concept code")
                display = _optional_text(concept.get("display"))
                candidates.append(_LocalConcept(code, display))
        else:
            candidates = list(cast(_ResolvedVocabulary, resolved).concepts)

        if filters:
            if resolved is None:
                resolved = self._resolve_vocabulary(system)
            if resolved is None:
                raise ValueSetExpansionUnsupportedError(
                    "local filters require a loaded free vocabulary"
                )
            known = {concept.code: concept for concept in resolved.concepts}
            candidates = [
                known.get(concept.code, concept)
                for concept in candidates
                if concept.code in known
            ]
            candidates = [
                concept
                for concept in candidates
                if _local_filters_match(concept, system=system, filters=filters)
            ]

        vocabulary_version = resolved.version if resolved is not None else None
        return tuple(
            (
                ValueSetMember(
                    system=system, code=concept.code, version=member_version
                ),
                vocabulary_version,
            )
            for concept in candidates
        )

    def _resolve_vocabulary(self, system: str) -> _ResolvedVocabulary | None:
        value = self._vocabularies.get(system)
        if value is None and self._vocabulary_registry is not None:
            try:
                value = self._vocabulary_registry.get(system)
            except KeyError:
                return None
        if value is None:
            return None
        return _coerce_vocabulary(system, value)

    def _expand_remote(
        self,
        source: str,
        *,
        source_kind: Literal["valueset-url", "ecl"],
        version: str | None,
    ) -> ValueSetExpansion:
        cache_identity = json.dumps(
            {
                "source": source,
                "endpoint": self.endpoint,
                "ecl_system": self.ecl_system_uri if source_kind == "ecl" else None,
            },
            sort_keys=True,
        )
        if version is not None:
            cached = self._cache_load(cache_identity, version, source_kind=source_kind)
            if cached is not None:
                return cached
        if self.endpoint is None:
            raise ValueSetExpansionConfigurationError(
                "remote ValueSet or ECL expansion requires a caller-supplied "
                "terminology endpoint"
            )

        if source_kind == "valueset-url":
            base_params = {"url": source}
            if version is not None:
                base_params["valueSetVersion"] = version
        else:
            base_params = {
                "url": _ecl_implicit_valueset_url(self.ecl_system_uri, source)
            }
            if version is not None:
                base_params["system-version"] = f"{self.ecl_system_uri}|{version}"
        base_params["count"] = str(self.page_size)

        members: dict[tuple[str, str], ValueSetMember] = {}
        response_version: str | None = None
        response_url: str | None = None
        expansion_identifier: str | None = None
        total: int | None = None
        next_offset = 0
        page = 0
        while True:
            if page >= self.max_pages:
                raise ValueSetExpansionResponseError(
                    "terminology expansion exceeded the configured page limit"
                )
            params = dict(base_params)
            if page:
                params["offset"] = str(next_offset)
            payload = self._request_json(params)
            page += 1
            page_members, item_count = _members_from_remote_payload(payload)
            if (
                len(set(members) | {member.key for member in page_members})
                > self.max_members
            ):
                raise ValueSetExpansionResponseError(
                    "terminology expansion exceeded the configured member limit"
                )
            for member in page_members:
                members[member.key] = member

            current_version = _remote_version(payload, source_kind=source_kind)
            if current_version is not None:
                if response_version is not None and current_version != response_version:
                    raise ValueSetExpansionResponseError(
                        "terminology expansion version changed between pages"
                    )
                response_version = current_version
            current_url = _optional_uri(payload.get("url"), "response ValueSet.url")
            if current_url is not None:
                if source_kind == "valueset-url" and current_url != source:
                    raise ValueSetExpansionResponseError(
                        "terminology response ValueSet URL does not match the request"
                    )
                if response_url is not None and current_url != response_url:
                    raise ValueSetExpansionResponseError(
                        "terminology ValueSet URL changed between pages"
                    )
                response_url = current_url
            expansion = cast(Mapping[str, Any], payload["expansion"])
            current_identifier = _optional_text(expansion.get("identifier"))
            if current_identifier is not None:
                if (
                    expansion_identifier is not None
                    and current_identifier != expansion_identifier
                ):
                    raise ValueSetExpansionResponseError(
                        "terminology expansion identifier changed between pages"
                    )
                expansion_identifier = current_identifier
            current_total = _optional_nonnegative_integer(
                expansion.get("total"),
                "expansion.total",
            )
            if current_total is not None:
                if total is not None and current_total != total:
                    raise ValueSetExpansionResponseError(
                        "terminology expansion total changed between pages"
                    )
                total = current_total
            returned_offset = _optional_nonnegative_integer(
                expansion.get("offset"), "expansion.offset"
            )
            if returned_offset is not None and returned_offset != next_offset:
                raise ValueSetExpansionResponseError(
                    "terminology expansion returned an unexpected page offset"
                )
            if total is not None and len(members) > total:
                raise ValueSetExpansionResponseError(
                    "terminology expansion exceeds its declared total"
                )
            if total is None or len(members) == total:
                break
            if item_count == 0:
                raise ValueSetExpansionResponseError(
                    "terminology expansion ended before its declared total"
                )
            current_offset = _optional_nonnegative_integer(
                expansion.get("offset"),
                "expansion.offset",
            )
            if current_offset is None:
                current_offset = next_offset
            candidate_offset = current_offset + item_count
            if candidate_offset <= next_offset:
                raise ValueSetExpansionResponseError(
                    "terminology expansion pagination did not advance"
                )
            next_offset = candidate_offset

        resolved_version = response_version or version
        if version is not None and response_version is not None:
            if version != response_version:
                raise ValueSetExpansionResponseError(
                    "terminology response does not match the requested version"
                )
        if resolved_version is None:
            resolved_version = _members_digest(
                tuple(members.values()), _digest_text(source)
            )
        valueset_url = source if source_kind == "valueset-url" else None
        restricted = source_kind == "ecl" or any(
            _is_restricted(member.system) for member in members.values()
        )
        request_hash = _digest_text(source)
        response_hash = _members_digest(tuple(members.values()), resolved_version)
        result = ValueSetExpansion(
            codings=tuple(members.values()),
            provenance=ExpansionProvenance(
                source_kind=source_kind,
                method=(
                    "remote-fhir-expand"
                    if source_kind == "valueset-url"
                    else "remote-ecl-delegation"
                ),
                version=resolved_version,
                request_sha256=request_hash,
                response_sha256=response_hash,
                valueset_url=valueset_url,
                endpoint=self.endpoint,
                expansion_identifier=expansion_identifier,
                vocabulary_versions=_member_version_pins(tuple(members.values())),
                restricted=restricted,
            ),
        )
        return self._cache_store(cache_identity, result)

    def _cache_load(
        self,
        source: str,
        version: str,
        *,
        source_kind: ExpansionSourceKind,
    ) -> ValueSetExpansion | None:
        if self.cache is None:
            return None
        return self.cache.load(source, version, source_kind=source_kind)

    def _cache_store(
        self,
        source: str,
        expansion: ValueSetExpansion,
    ) -> ValueSetExpansion:
        if self.cache is None:
            return expansion
        if expansion.provenance.restricted and not self.cache.allow_restricted:
            return expansion
        return self.cache.store(source, expansion)

    def _request_json(self, params: Mapping[str, str]) -> dict[str, Any]:
        operation_url = _expand_operation_url(cast(str, self.endpoint))
        headers = {
            "Accept": "application/fhir+json, application/json",
            **self._headers,
        }
        if self._bearer_token is not None:
            headers.setdefault("Authorization", f"Bearer {self._bearer_token}")
        if self._client is not None:
            try:
                response = self._client.get(
                    operation_url,
                    params=dict(params),
                    headers=headers,
                )
            except Exception:
                raise ValueSetExpansionResponseError(
                    "terminology expansion request failed"
                ) from None
            status = int(getattr(response, "status_code", 200))
            if status >= 300:
                raise ValueSetExpansionResponseError(
                    f"terminology expansion returned HTTP {status}"
                )
            try:
                payload = response.json()
            except (TypeError, ValueError):
                raise ValueSetExpansionResponseError(
                    "terminology expansion response was not valid JSON"
                ) from None
            return _validate_remote_payload(payload)

        request_url = f"{operation_url}?{urlparse.urlencode(params)}"
        request = urlrequest.Request(request_url, method="GET", headers=headers)
        try:
            with _open_without_redirects(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200))
                if status >= 300:
                    raise ValueSetExpansionResponseError(
                        f"terminology expansion returned HTTP {status}"
                    )
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except ValueSetExpansionResponseError:
            raise
        except urlerror.HTTPError as exc:
            raise ValueSetExpansionResponseError(
                f"terminology expansion returned HTTP {exc.code}"
            ) from None
        except (urlerror.URLError, TimeoutError, OSError):
            raise ValueSetExpansionResponseError(
                "terminology expansion request failed"
            ) from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueSetExpansionResponseError(
                "terminology expansion response is too large"
            )
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise ValueSetExpansionResponseError(
                "terminology expansion response was not valid JSON"
            ) from None
        return _validate_remote_payload(payload)


ValueSetExpander = ValueSetExpansionEngine


def expand_valueset(
    url_or_ecl: ValueSetSource,
    *,
    version: str | None = None,
    endpoint: str | None = None,
    vocabularies: Mapping[str, LocalVocabulary] | None = None,
    vocabulary_registry: VocabularyLoaderRegistry | None = None,
    cache: ValueSetExpansionCache | TerminologySnapshotCache | None = None,
    cache_dir: str | Path | None = None,
    allow_restricted_cache: bool = False,
    client: Any | None = None,
    headers: Mapping[str, str] | None = None,
    bearer_token: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    ecl_system_uri: str = _SNOMED_SYSTEM_URI,
) -> ValueSetExpansion:
    """Expand a ValueSet URL or ECL and return member codes with provenance.

    Mapping, inline JSON, and local-path inputs select offline extensional
    expansion.  An absolute URL or ECL string uses the endpoint only when the
    caller explicitly supplies one.  Pass ``cache`` or ``cache_dir`` to opt in
    to persistence; restricted results additionally require
    ``allow_restricted_cache=True`` or an equivalently configured cache.
    """

    engine = ValueSetExpansionEngine(
        endpoint,
        vocabularies=vocabularies,
        vocabulary_registry=vocabulary_registry,
        cache=cache,
        cache_dir=cache_dir,
        allow_restricted_cache=allow_restricted_cache,
        client=client,
        headers=headers,
        bearer_token=bearer_token,
        timeout=timeout,
        ecl_system_uri=ecl_system_uri,
    )
    return engine.expand_valueset(url_or_ecl, version=version)


def _coerce_vocabulary(system: str, value: object) -> _ResolvedVocabulary:
    declared_version = False
    if isinstance(value, TerminologySnapshot):
        index = value.index
        version = value.release_version
        declared_version = True
    elif isinstance(value, VocabularyIndex):
        index = value
        version = value.content_hash
    else:
        load_snapshot = getattr(value, "load_snapshot", None)
        get_index = getattr(value, "get_index", None)
        load = getattr(value, "load", None)
        if callable(load_snapshot):
            loaded = load_snapshot()
        elif callable(get_index):
            loaded = get_index()
        elif callable(load):
            loaded = load()
        else:
            loaded = value
        if loaded is value:
            if not isinstance(loaded, Mapping):
                raise TypeError("loaded vocabulary must be an index or term mapping")
            concepts = _concepts_from_term_mapping(system, loaded)
            version = _object_version(value) or _concepts_digest(concepts)
            declared_version = _object_version(value) is not None
            return _ResolvedVocabulary(system, concepts, version, declared_version)
        resolved = _coerce_vocabulary(system, loaded)
        object_version = _object_version(value)
        if object_version is None:
            return resolved
        return replace(resolved, version=object_version, declared_version=True)

    concepts = tuple(
        _LocalConcept(concept.code, concept.preferred_term)
        for concept in index.concepts
    )
    return _ResolvedVocabulary(system, concepts, version, declared_version)


def _concepts_from_term_mapping(
    system: str,
    terms: Mapping[str, object],
) -> tuple[_LocalConcept, ...]:
    concepts: dict[str, _LocalConcept] = {}
    for term, raw in terms.items():
        values: Sequence[object]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)):
            values = raw
        else:
            values = (raw,)
        for value in values:
            if isinstance(value, LexicalConcept):
                if value.system_uri != system:
                    raise ValueError(
                        "loaded vocabulary concept system does not match key"
                    )
                concept = _LocalConcept(value.code, value.display)
            elif isinstance(value, VocabConcept):
                concept = _LocalConcept(value.code, value.preferred_term)
            elif isinstance(value, Mapping):
                code = _nonempty_text(value.get("code"), "vocabulary concept code")
                display = _optional_text(value.get("display")) or str(term)
                concept = _LocalConcept(code, display)
            else:
                concept = _LocalConcept(
                    _nonempty_text(value, "vocabulary concept code"),
                    str(term),
                )
            concepts[concept.code] = concept
    return tuple(sorted(concepts.values(), key=lambda item: item.code))


def _object_version(value: object) -> str | None:
    for name in ("release_version", "version", "content_hash"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and candidate.strip():
            return _version_text(candidate)
    return None


def _members_from_remote_payload(
    payload: Mapping[str, Any],
) -> tuple[set[ValueSetMember], int]:
    expansion = cast(Mapping[str, Any], payload["expansion"])
    return _members_from_contains(expansion.get("contains"))


def _members_from_contains(value: object) -> tuple[set[ValueSetMember], int]:
    contains = _mapping_sequence(value, "expansion.contains")
    result: set[ValueSetMember] = set()
    count = 0

    def visit(
        items: Sequence[Mapping[str, Any]],
        inherited_system: str | None = None,
        inherited_version: str | None = None,
    ) -> None:
        nonlocal count
        for item in items:
            count += 1
            raw_system = item.get("system")
            system = (
                _absolute_uri(raw_system, "expansion member system")
                if raw_system is not None
                else inherited_system
            )
            version = _optional_version(item.get("version")) or inherited_version
            code = item.get("code")
            if code is not None and item.get("abstract") is not True:
                if system is None:
                    raise ValueSetExpansionResponseError(
                        "expansion member code is missing a code-system URI"
                    )
                result.add(
                    ValueSetMember(system, _nonempty_text(code, "code"), version)
                )
            nested = _mapping_sequence(item.get("contains"), "nested contains")
            visit(nested, system, version)

    visit(contains)
    return result, count


def _require_complete_expansion(
    expansion: Mapping[str, Any],
    item_count: int,
) -> None:
    offset = _optional_nonnegative_integer(expansion.get("offset"), "expansion.offset")
    total = _optional_nonnegative_integer(expansion.get("total"), "expansion.total")
    if offset not in (None, 0) or (total is not None and total > item_count):
        raise ValueSetExpansionUnsupportedError(
            "local ValueSet expansion is paged or incomplete"
        )


def _validate_remote_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueSetExpansionResponseError(
            "terminology expansion response must be a FHIR JSON object"
        )
    if payload.get("resourceType") == "OperationOutcome":
        raise ValueSetExpansionResponseError(
            "terminology server returned a FHIR OperationOutcome"
        )
    if payload.get("resourceType") not in (None, "ValueSet"):
        raise ValueSetExpansionResponseError(
            "terminology expansion response must be a ValueSet"
        )
    if not isinstance(payload.get("expansion"), Mapping):
        raise ValueSetExpansionResponseError(
            "terminology ValueSet response has no expansion"
        )
    return dict(payload)


def _remote_version(
    payload: Mapping[str, Any],
    *,
    source_kind: Literal["valueset-url", "ecl"],
) -> str | None:
    expansion = cast(Mapping[str, Any], payload["expansion"])
    parameters = _mapping_sequence(expansion.get("parameter"), "expansion.parameter")
    parameter_names = (
        ("system-version", "version", "valueSetVersion")
        if source_kind == "ecl"
        else ("valueSetVersion", "version")
    )
    for parameter_name in parameter_names:
        for parameter in parameters:
            if parameter.get("name") != parameter_name:
                continue
            for key in ("valueString", "valueUri", "valueCanonical"):
                value = _optional_version(parameter.get(key))
                if value is not None:
                    return (
                        value.rsplit("|", 1)[-1]
                        if parameter_name == "system-version"
                        else value
                    )
    version = _optional_version(payload.get("version"))
    if version is not None:
        return version
    return None


def _local_filters_match(
    concept: _LocalConcept,
    *,
    system: str,
    filters: Sequence[Mapping[str, Any]],
) -> bool:
    for definition in filters:
        property_name = definition.get("property")
        operator = definition.get("op")
        raw_value = definition.get("value")
        if not isinstance(property_name, str) or not isinstance(operator, str):
            raise ValueSetExpansionUnsupportedError(
                "local ValueSet filter requires property and op"
            )
        if property_name in {"code", "concept"}:
            actual = concept.code
        elif property_name == "system":
            actual = system
        elif property_name == "display":
            actual = concept.display
        else:
            raise ValueSetExpansionUnsupportedError(
                "local hierarchy and terminology-property filters require delegation"
            )
        if operator == "exists":
            expected = str(raw_value).casefold() in {"true", "1", "yes"}
            if (actual is not None) != expected:
                return False
            continue
        if actual is None or not isinstance(raw_value, (str, bool, int, float)):
            return False
        if operator in {"=", "=="}:
            if actual != str(raw_value):
                return False
            continue
        if operator == "regex":
            try:
                if re.search(str(raw_value), actual) is None:
                    return False
            except re.error:
                raise ValueSetExpansionUnsupportedError(
                    "local ValueSet filter contains an invalid regular expression"
                ) from None
            continue
        raise ValueSetExpansionUnsupportedError(
            "local ValueSet filter operator requires terminology-server delegation"
        )
    return True


def _mapping_sequence(value: object, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an object or array of objects")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{field_name} must contain only objects")
    return tuple(cast(Sequence[Mapping[str, Any]], value))


def _payload_contains_restricted_system(payload: Mapping[str, Any]) -> bool:
    systems: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            system = value.get("system")
            if isinstance(system, str):
                systems.add(system)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                collect(nested)

    collect(payload.get("compose"))
    collect(payload.get("expansion"))
    return any(_is_restricted(system) for system in systems)


def _member_version_pins(
    members: Sequence[ValueSetMember],
) -> tuple[tuple[str, str], ...]:
    versions: dict[str, str] = {}
    for member in members:
        if member.version is None:
            continue
        previous = versions.get(member.system)
        if previous is not None and previous != member.version:
            continue
        versions[member.system] = member.version
    return tuple(sorted(versions.items()))


def _members_digest(members: Sequence[ValueSetMember], version: str) -> str:
    payload = {
        "members": [
            member.to_dict()
            for member in sorted(
                set(members),
                key=lambda member: (member.system, member.code, member.version or ""),
            )
        ],
        "version": version,
    }
    return _digest_bytes(_canonical_json_bytes(payload))


def _concepts_digest(concepts: Sequence[_LocalConcept]) -> str:
    payload = {
        "concepts": [
            {"code": concept.code, "display": concept.display}
            for concept in sorted(concepts, key=lambda item: item.code)
        ]
    }
    return _digest_bytes(_canonical_json_bytes(payload))


def _expansion_from_payload(payload: Mapping[str, Any]) -> ValueSetExpansion:
    if set(payload) != {"codings", "members", "provenance"}:
        raise ValueError("cached expansion fields are invalid")
    raw_codings = payload.get("codings")
    raw_members = payload.get("members")
    provenance = payload.get("provenance")
    if not isinstance(raw_codings, list) or not isinstance(raw_members, list):
        raise ValueError("cached expansion members are invalid")
    if not isinstance(provenance, Mapping):
        raise ValueError("cached expansion provenance is invalid")
    codings = tuple(
        ValueSetMember(
            system=_mapping_value(item, "system"),
            code=_mapping_value(item, "code"),
            version=(
                cast(Mapping[str, Any], item).get("version")
                if isinstance(item, Mapping)
                else None
            ),
        )
        for item in raw_codings
    )
    vocabulary_versions = provenance.get("vocabulary_versions")
    if not isinstance(vocabulary_versions, list):
        raise ValueError("cached vocabulary versions are invalid")
    parsed_versions = tuple(
        (
            _mapping_value(item, "system"),
            _mapping_value(item, "version"),
        )
        for item in vocabulary_versions
    )
    result = ValueSetExpansion(
        codings=codings,
        provenance=ExpansionProvenance(
            source_kind=cast(ExpansionSourceKind, provenance.get("source_kind")),
            method=_mapping_value(provenance, "method"),
            version=_mapping_value(provenance, "version"),
            request_sha256=_mapping_value(provenance, "request_sha256"),
            response_sha256=_mapping_value(provenance, "response_sha256"),
            valueset_url=cast(str | None, provenance.get("valueset_url")),
            endpoint=cast(str | None, provenance.get("endpoint")),
            expansion_identifier=cast(
                str | None,
                provenance.get("expansion_identifier"),
            ),
            vocabulary_versions=parsed_versions,
            restricted=provenance.get("restricted") is True,
            cache_hit=False,
        ),
    )
    if sorted(result.members) != sorted(raw_members):
        raise ValueError("cached code set does not match cached codings")
    return result


def _mapping_value(value: object, key: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("cached expansion entry must be an object")
    return _nonempty_text(value.get(key), key)


def _canonical_parts(value: str) -> tuple[str, str | None]:
    canonical = value
    version: str | None = None
    if "|" in value:
        canonical, raw_version = value.rsplit("|", 1)
        version = _version_text(raw_version)
    return _absolute_uri(canonical, "ValueSet URL"), version


def _expand_operation_url(endpoint: str) -> str:
    stripped = endpoint.rstrip("/")
    if stripped.endswith("/$expand"):
        return stripped
    return f"{stripped}/ValueSet/$expand"


def _ecl_implicit_valueset_url(system_uri: str, ecl: str) -> str:
    return f"{system_uri.rstrip('/')}?fhir_vs=ecl/{ecl}"


def _endpoint(value: object) -> str:
    try:
        endpoint = _absolute_uri(value, "terminology endpoint")
    except ValueError:
        raise ValueSetExpansionConfigurationError(
            "terminology endpoint must be an absolute HTTP(S) URL"
        ) from None
    parsed = urlparse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueSetExpansionConfigurationError(
            "terminology endpoint must be an absolute HTTP(S) URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueSetExpansionConfigurationError(
            "terminology endpoint credentials must use explicit auth settings"
        )
    if parsed.query or parsed.fragment:
        raise ValueSetExpansionConfigurationError(
            "terminology endpoint must not contain query or fragment components"
        )
    return endpoint.rstrip("/")


def _absolute_uri(value: object, field_name: str) -> str:
    text = _nonempty_text(value, field_name)
    if any(character.isspace() for character in text) or not _URI_SCHEME_RE.match(text):
        raise ValueError(f"{field_name} must be an absolute URI without whitespace")
    return text


def _optional_uri(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _absolute_uri(value, field_name)


def _version_text(value: object) -> str:
    text = _nonempty_text(value, "version")
    if any(ord(character) < 32 for character in text):
        raise ValueError("version must not contain control characters")
    return text


def _optional_version(value: object) -> str | None:
    if value is None:
        return None
    return _version_text(value)


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, "text")


def _optional_nonnegative_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueSetExpansionResponseError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return normalized


def _headers(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("headers must be a string mapping")
    result: dict[str, str] = {}
    for name, header_value in value.items():
        if not isinstance(name, str) or not _HEADER_NAME_RE.fullmatch(name):
            raise ValueSetExpansionConfigurationError("invalid terminology header name")
        if not isinstance(header_value, str) or any(
            character in header_value for character in "\r\n"
        ):
            raise ValueSetExpansionConfigurationError(
                "terminology header values must be strings without line breaks"
            )
        result[name] = header_value
    return result


def _optional_secret(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in "\r\n")
    ):
        raise ValueSetExpansionConfigurationError(
            f"{field_name} must be non-empty text without line breaks"
        )
    return value


def _is_restricted(system: str) -> bool:
    normalized = system.rstrip("/").casefold()
    return normalized in _RESTRICTED_URIS or "cpt" in normalized


def _is_local_file(value: str) -> bool:
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _digest_text(value: str) -> str:
    return _digest_bytes(value.encode("utf-8"))


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_text(value: object, field_name: str) -> str:
    text = _nonempty_text(value, field_name)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return text


def _cache_key(
    source_kind: ExpansionSourceKind,
    source_hash: str,
    version: str,
) -> str:
    payload = {
        "schema_version": EXPANSION_SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_sha256": source_hash,
        "version": version,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_limited(path: Path, maximum: int) -> bytes:
    with path.open("rb") as handle:
        value = handle.read(maximum + 1)
    if len(value) > maximum:
        raise ValueError("cached expansion file exceeds its size limit")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _open_without_redirects(
    request: urlrequest.Request,
    *,
    timeout: float,
) -> Any:
    class _NoRedirect(urlrequest.HTTPRedirectHandler):
        def redirect_request(
            self,
            req: urlrequest.Request,
            fp: Any,
            code: int,
            msg: str,
            headers: Any,
            newurl: str,
        ) -> None:
            return None

    opener = urlrequest.build_opener(_NoRedirect())
    return opener.open(request, timeout=timeout)
