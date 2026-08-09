"""Deterministic, provenance-aware caching for caller-supplied terminology.

The cache is deliberately in-memory and local-only.  A caller supplies the
terminology response and the identifiers for the vocabulary release that
produced it; the cache never discovers, downloads, or refreshes terminology on
its own.  Responses are canonicalized before storage so equivalent mappings
produce the same content fingerprint regardless of insertion order.

Cache metadata is safe to use in logs and reports: it contains identifiers and
SHA-256 fingerprints, while :meth:`TerminologyCacheEntry.to_dict` omits the
response by default.  Callers can retrieve the response explicitly through the
``response`` property when it is needed for a terminology operation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Final

CACHE_SCHEMA_VERSION: Final = "openmed.terminology_cache.v1"
PROVENANCE_SCHEMA_VERSION: Final = "openmed.terminology_provenance.v1"
DEFAULT_SOURCE: Final = "user-supplied"

_MISSING = object()
_SHA256_PREFIX = "sha256:"
_SHA256_HEX_LENGTH = 64


class TerminologyCacheError(ValueError):
    """Base error for invalid or unusable terminology cache data."""


class TerminologyProvenanceError(TerminologyCacheError):
    """Raised when a cached response does not match requested provenance."""


class StaleTerminologyError(TerminologyProvenanceError):
    """Raised when only an older release is cached for a vocabulary."""

    def __init__(
        self,
        *,
        requested_key: "TerminologyCacheKey",
        cached_keys: tuple["TerminologyCacheKey", ...],
    ) -> None:
        self.requested_key = requested_key
        self.cached_keys = cached_keys
        super().__init__(
            "cached terminology is stale for the requested vocabulary release"
        )


# These aliases make the failure mode discoverable without introducing
# separate exception types that callers would need to catch independently.
StaleTerminologyCacheError = StaleTerminologyError
TerminologyCacheStaleError = StaleTerminologyError


def _identifier(value: object, field_name: str) -> str:
    """Normalize an opaque identifier without echoing its value in errors."""

    if not isinstance(value, str):
        raise TerminologyCacheError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise TerminologyCacheError(f"{field_name} must be a non-empty string")
    return normalized


def _source_identifier(value: object) -> str:
    return _identifier(value, "source")


def _canonicalize(value: object, *, field_name: str = "response") -> Any:
    """Return a JSON-compatible, deterministic copy without exposing values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TerminologyCacheError(
                f"{field_name} must not contain non-finite numbers"
            )
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TerminologyCacheError(
                    f"{field_name} mappings must use string keys"
                )
            normalized[key] = _canonicalize(item, field_name=field_name)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item, field_name=field_name) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [
            _canonicalize(item, field_name=field_name) for item in value
        ]
        return sorted(normalized_items, key=_canonical_json)
    raise TerminologyCacheError(f"{field_name} must be JSON-compatible")


def _canonical_json(value: Any) -> str:
    """Encode a normalized value with one stable JSON representation."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, TypeError, ValueError):
        raise TerminologyCacheError(
            "terminology data must be JSON-compatible"
        ) from None


def _normalized_response(response: object) -> Any:
    """Return a detached response copy suitable for hashing and storage."""

    normalized = _canonicalize(response)
    # A JSON round-trip also detaches any nested objects supplied by a caller.
    try:
        return json.loads(_canonical_json(normalized))
    except json.JSONDecodeError:
        raise TerminologyCacheError("terminology response is not valid JSON") from None


def _digest(payload: Any, *, domain: str) -> str:
    encoded = _canonical_json({"domain": domain, "payload": payload}).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def _validate_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX):
        raise TerminologyCacheError(f"{field_name} must be a SHA-256 fingerprint")
    digest = value[len(_SHA256_PREFIX) :]
    if len(digest) != _SHA256_HEX_LENGTH:
        raise TerminologyCacheError(f"{field_name} must be a SHA-256 fingerprint")
    try:
        int(digest, 16)
    except ValueError:
        raise TerminologyCacheError(
            f"{field_name} must be a SHA-256 fingerprint"
        ) from None
    return value


@dataclass(frozen=True, slots=True)
class TerminologyCacheKey:
    """The stable lookup identity for one vocabulary release."""

    vocabulary: str
    release: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "vocabulary", _identifier(self.vocabulary, "vocabulary")
        )
        object.__setattr__(self, "release", _identifier(self.release, "release"))

    @property
    def vocabulary_id(self) -> str:
        """Return ``vocabulary`` under the explicit identifier spelling."""

        return self.vocabulary

    @property
    def release_id(self) -> str:
        """Return ``release`` under the explicit identifier spelling."""

        return self.release

    @property
    def version(self) -> str:
        """Return the pinned release identifier."""

        return self.release

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for the cache key."""

        return _digest(self.to_dict(), domain="terminology-cache-key-v1")

    @property
    def digest(self) -> str:
        """Alias for :attr:`fingerprint`."""

        return self.fingerprint

    def to_dict(self) -> dict[str, str]:
        """Return the safe JSON representation of this key."""

        return {"release": self.release, "vocabulary": self.vocabulary}


def terminology_response_fingerprint(response: object) -> str:
    """Return a deterministic SHA-256 fingerprint for a terminology response.

    Mapping key order does not affect the result.  The response is validated as
    JSON-compatible, but its contents are never included in an exception or in
    the safe metadata reports produced by this module.
    """

    normalized = _normalized_response(response)
    return _digest(normalized, domain="terminology-response-v1")


def compute_terminology_fingerprint(
    response: object,
    *,
    vocabulary: str,
    release: str,
    source: str = DEFAULT_SOURCE,
) -> str:
    """Return the provenance fingerprint for one cached response."""

    key = TerminologyCacheKey(vocabulary=vocabulary, release=release)
    source_id = _source_identifier(source)
    response_fingerprint = terminology_response_fingerprint(response)
    return _digest(
        {
            "key": key.to_dict(),
            "response_fingerprint": response_fingerprint,
            "source": source_id,
            "schema_version": PROVENANCE_SCHEMA_VERSION,
        },
        domain="terminology-provenance-v1",
    )


# Short aliases are useful at call sites that already use ``fingerprint`` as
# their vocabulary terminology.
terminology_fingerprint = compute_terminology_fingerprint
fingerprint_terminology_response = terminology_response_fingerprint


@dataclass(frozen=True, slots=True)
class TerminologyProvenance:
    """Safe provenance metadata attached to one terminology cache entry."""

    key: TerminologyCacheKey
    source: str
    response_fingerprint: str
    fingerprint: str
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.key, TerminologyCacheKey):
            raise TerminologyCacheError(
                "provenance key must be a terminology cache key"
            )
        object.__setattr__(self, "source", _source_identifier(self.source))
        _validate_digest(self.response_fingerprint, "response_fingerprint")
        _validate_digest(self.fingerprint, "fingerprint")
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise TerminologyCacheError("unsupported terminology provenance schema")

    @classmethod
    def from_response(
        cls,
        *,
        vocabulary: str,
        release: str,
        response: object,
        source: str = DEFAULT_SOURCE,
    ) -> "TerminologyProvenance":
        """Build provenance metadata from a response and its release identity."""

        key = TerminologyCacheKey(vocabulary=vocabulary, release=release)
        source_id = _source_identifier(source)
        response_fingerprint = terminology_response_fingerprint(response)
        fingerprint = _digest(
            {
                "key": key.to_dict(),
                "response_fingerprint": response_fingerprint,
                "source": source_id,
                "schema_version": PROVENANCE_SCHEMA_VERSION,
            },
            domain="terminology-provenance-v1",
        )
        return cls(
            key=key,
            source=source_id,
            response_fingerprint=response_fingerprint,
            fingerprint=fingerprint,
        )

    @property
    def vocabulary(self) -> str:
        """Return the vocabulary identifier carried by this provenance."""

        return self.key.vocabulary

    @property
    def release(self) -> str:
        """Return the release identifier carried by this provenance."""

        return self.key.release

    @property
    def vocabulary_version(self) -> str:
        """Return the release under the terminology-version spelling."""

        return self.key.release

    def to_dict(self) -> dict[str, str]:
        """Return PHI-safe provenance metadata for logs and reports."""

        return {
            "fingerprint": self.fingerprint,
            "release": self.release,
            "response_fingerprint": self.response_fingerprint,
            "schema_version": self.schema_version,
            "source": self.source,
            "vocabulary": self.vocabulary,
        }


@dataclass(frozen=True, slots=True)
class TerminologyCacheEntry:
    """A cached response plus metadata needed to verify its provenance."""

    key: TerminologyCacheKey
    provenance: TerminologyProvenance
    _response: Any = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        if not isinstance(self.key, TerminologyCacheKey):
            raise TerminologyCacheError(
                "cache entry key must be a terminology cache key"
            )
        if not isinstance(self.provenance, TerminologyProvenance):
            raise TerminologyCacheError("cache entry provenance is invalid")
        if self.provenance.key != self.key:
            raise TerminologyProvenanceError(
                "cache entry key does not match its provenance"
            )
        normalized = _normalized_response(self._response)
        response_fingerprint = terminology_response_fingerprint(normalized)
        if response_fingerprint != self.provenance.response_fingerprint:
            raise TerminologyProvenanceError(
                "cache entry response does not match its provenance"
            )
        expected_fingerprint = compute_terminology_fingerprint(
            normalized,
            vocabulary=self.key.vocabulary,
            release=self.key.release,
            source=self.provenance.source,
        )
        if expected_fingerprint != self.provenance.fingerprint:
            raise TerminologyProvenanceError(
                "cache entry fingerprint does not match its provenance"
            )
        object.__setattr__(self, "_response", normalized)

    @classmethod
    def from_response(
        cls,
        *,
        vocabulary: str,
        release: str,
        response: object,
        source: str = DEFAULT_SOURCE,
    ) -> "TerminologyCacheEntry":
        """Create an entry with canonical response and derived provenance."""

        normalized = _normalized_response(response)
        key = TerminologyCacheKey(vocabulary=vocabulary, release=release)
        provenance = TerminologyProvenance.from_response(
            vocabulary=key.vocabulary,
            release=key.release,
            response=normalized,
            source=source,
        )
        return cls(key=key, provenance=provenance, _response=normalized)

    @property
    def response(self) -> Any:
        """Return a detached copy of the cached terminology response."""

        return copy.deepcopy(self._response)

    @property
    def value(self) -> Any:
        """Alias for :attr:`response` for cache-style call sites."""

        return self.response

    @property
    def vocabulary(self) -> str:
        """Return the cached vocabulary identifier."""

        return self.key.vocabulary

    @property
    def release(self) -> str:
        """Return the cached release identifier."""

        return self.key.release

    @property
    def source(self) -> str:
        """Return the source identifier carried by the provenance."""

        return self.provenance.source

    @property
    def fingerprint(self) -> str:
        """Return the full provenance fingerprint."""

        return self.provenance.fingerprint

    def to_dict(self, *, include_response: bool = False) -> dict[str, Any]:
        """Return safe metadata, optionally including the explicit response.

        The default is suitable for logs and reports.  Set
        ``include_response=True`` only at a call site that explicitly needs to
        serialize the caller-supplied terminology response.
        """

        payload: dict[str, Any] = {
            "key": self.key.to_dict(),
            "provenance": self.provenance.to_dict(),
        }
        if include_response:
            payload["response"] = self.response
        return payload

    def __repr__(self) -> str:
        return (
            f"TerminologyCacheEntry(key={self.key!r}, fingerprint={self.fingerprint!r})"
        )


def _resolve_key(
    vocabulary: str | TerminologyCacheKey | None,
    release: str | None,
    *,
    vocabulary_id: str | None,
    release_id: str | None,
) -> TerminologyCacheKey:
    if isinstance(vocabulary, TerminologyCacheKey):
        if release is not None or vocabulary_id is not None or release_id is not None:
            raise TerminologyCacheError(
                "a cache key cannot be combined with identifiers"
            )
        return vocabulary
    if vocabulary is None:
        vocabulary = vocabulary_id
    elif vocabulary_id is not None:
        if _identifier(vocabulary, "vocabulary") != _identifier(
            vocabulary_id, "vocabulary"
        ):
            raise TerminologyCacheError("vocabulary identifiers do not match")
    if release is None:
        release = release_id
    elif release_id is not None:
        if _identifier(release, "release") != _identifier(release_id, "release"):
            raise TerminologyCacheError("release identifiers do not match")
    if vocabulary is None or release is None:
        raise TerminologyCacheError("vocabulary and release are required")
    return TerminologyCacheKey(vocabulary=vocabulary, release=release)


class TerminologyCache:
    """An in-memory cache keyed by exact vocabulary and release identifiers.

    The cache does not perform I/O, consult environment variables, or make
    network requests.  A request for a release that is absent while another
    release of the same vocabulary is present raises
    :class:`StaleTerminologyError`; callers must supply the requested release
    rather than silently using an older response.
    """

    def __init__(self, entries: Iterable[TerminologyCacheEntry] | None = None) -> None:
        self._entries: dict[TerminologyCacheKey, TerminologyCacheEntry] = {}
        if entries is not None:
            for entry in entries:
                if not isinstance(entry, TerminologyCacheEntry):
                    raise TerminologyCacheError(
                        "cache entries must be terminology entries"
                    )
                self._entries[entry.key] = entry

    def put(
        self,
        vocabulary: str | TerminologyCacheKey | None = None,
        release: str | None = None,
        response: object = _MISSING,
        *,
        source: str = DEFAULT_SOURCE,
        vocabulary_id: str | None = None,
        release_id: str | None = None,
    ) -> TerminologyCacheEntry:
        """Store and return a response under an exact vocabulary release key."""

        if response is _MISSING:
            raise TerminologyCacheError("response is required")
        key = _resolve_key(
            vocabulary,
            release,
            vocabulary_id=vocabulary_id,
            release_id=release_id,
        )
        entry = TerminologyCacheEntry.from_response(
            vocabulary=key.vocabulary,
            release=key.release,
            response=response,
            source=source,
        )
        self._entries[key] = entry
        return entry

    store = put
    set = put

    def get(
        self,
        vocabulary: str | TerminologyCacheKey | None = None,
        release: str | None = None,
        *,
        source: str | None = None,
        vocabulary_id: str | None = None,
        release_id: str | None = None,
    ) -> TerminologyCacheEntry | None:
        """Return an exact entry, or ``None`` for a vocabulary cache miss.

        A release mismatch is not treated as a normal miss when an older or
        newer entry for the same vocabulary exists: it raises
        :class:`StaleTerminologyError` so callers cannot accidentally reuse a
        response from a different release.
        """

        key = _resolve_key(
            vocabulary,
            release,
            vocabulary_id=vocabulary_id,
            release_id=release_id,
        )
        entry = self._entries.get(key)
        if entry is not None:
            if source is not None and _source_identifier(source) != entry.source:
                raise TerminologyProvenanceError(
                    "cached terminology source does not match the request"
                )
            return entry

        stale_keys = tuple(
            cached_key
            for cached_key in self._entries
            if cached_key.vocabulary == key.vocabulary
        )
        if stale_keys:
            raise StaleTerminologyError(
                requested_key=key,
                cached_keys=tuple(sorted(stale_keys, key=lambda item: item.release)),
            )
        return None

    lookup = get
    get_entry = get

    def get_response(
        self,
        vocabulary: str | TerminologyCacheKey | None = None,
        release: str | None = None,
        *,
        source: str | None = None,
        vocabulary_id: str | None = None,
        release_id: str | None = None,
    ) -> Any | None:
        """Return only a detached response while retaining stale checks."""

        entry = self.get(
            vocabulary,
            release,
            source=source,
            vocabulary_id=vocabulary_id,
            release_id=release_id,
        )
        return None if entry is None else entry.response

    def get_or_compute(
        self,
        vocabulary: str | TerminologyCacheKey | None,
        release: str | None,
        compute: Callable[[], object],
        *,
        source: str | None = None,
        vocabulary_id: str | None = None,
        release_id: str | None = None,
    ) -> TerminologyCacheEntry:
        """Return a hit or cache a caller-provided computation on a miss."""

        entry = self.get(
            vocabulary,
            release,
            source=source,
            vocabulary_id=vocabulary_id,
            release_id=release_id,
        )
        if entry is not None:
            return entry
        key = _resolve_key(
            vocabulary,
            release,
            vocabulary_id=vocabulary_id,
            release_id=release_id,
        )
        return self.put(
            key.vocabulary,
            key.release,
            compute(),
            source=DEFAULT_SOURCE if source is None else source,
        )

    get_or_set = get_or_compute
    load_or_compute = get_or_compute

    def invalidate(
        self,
        vocabulary: str | TerminologyCacheKey | None = None,
        release: str | None = None,
        *,
        vocabulary_id: str | None = None,
        release_id: str | None = None,
    ) -> bool:
        """Remove one exact entry and return whether it was present."""

        key = _resolve_key(
            vocabulary,
            release,
            vocabulary_id=vocabulary_id,
            release_id=release_id,
        )
        return self._entries.pop(key, None) is not None

    def clear(self) -> int:
        """Remove all entries and return the number removed."""

        count = len(self._entries)
        self._entries.clear()
        return count

    @property
    def entries(self) -> tuple[TerminologyCacheEntry, ...]:
        """Return entries in deterministic vocabulary/release order."""

        return tuple(
            self._entries[key]
            for key in sorted(
                self._entries,
                key=lambda item: (item.vocabulary, item.release),
            )
        )

    def keys(self) -> tuple[TerminologyCacheKey, ...]:
        """Return cache keys in deterministic order."""

        return tuple(entry.key for entry in self.entries)

    def report(self) -> dict[str, Any]:
        """Return deterministic metadata without cached response contents."""

        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "entry_count": len(self),
            "schema_version": CACHE_SCHEMA_VERSION,
        }

    to_dict = report

    def __contains__(self, key: object) -> bool:
        return isinstance(key, TerminologyCacheKey) and key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"TerminologyCache(entry_count={len(self)})"


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_SOURCE",
    "PROVENANCE_SCHEMA_VERSION",
    "StaleTerminologyCacheError",
    "StaleTerminologyError",
    "TerminologyCache",
    "TerminologyCacheEntry",
    "TerminologyCacheKey",
    "TerminologyCacheStaleError",
    "TerminologyCacheError",
    "TerminologyProvenance",
    "TerminologyProvenanceError",
    "compute_terminology_fingerprint",
    "fingerprint_terminology_response",
    "terminology_fingerprint",
    "terminology_response_fingerprint",
]
