"""Deterministic, privacy-safe cache invalidation for consent revocation.

The cache keeps derived outputs in memory while retaining only fingerprints for
cache keys and consent metadata.  Revocation events contain aggregate counts,
not cache keys, consent values, or cached outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Generic, TypeAlias, TypeVar

__all__ = [
    "CONSENT_CACHE_EVENT_TYPE",
    "CONSENT_CACHE_SCHEMA_VERSION",
    "ConsentCache",
    "ConsentCacheEntry",
    "ConsentCacheStats",
    "ConsentInvalidationEvent",
    "ConsentRevision",
    "ConsentScope",
    "fingerprint_cache_key",
    "fingerprint_consent_revision",
    "fingerprint_consent_scope",
]

CONSENT_CACHE_SCHEMA_VERSION = 1
CONSENT_CACHE_EVENT_TYPE = "consent_cache_invalidation"

ConsentScope: TypeAlias = str | Iterable[str]
ConsentRevision: TypeAlias = Any

T = TypeVar("T")
_MISSING = object()
_EntryKey: TypeAlias = tuple[str, str, str]
_RevocationKey: TypeAlias = tuple[str, str]


def _canonical_json(value: Any) -> str:
    """Return a stable JSON representation for fingerprint input."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_value(value: Any) -> Any:
    """Convert supported values to an unambiguous, JSON-compatible form.

    The returned value is used only while calculating a digest.  It is never
    retained by a cache entry or included in an audit event.
    """

    if isinstance(value, str):
        return ["string", value.strip()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprint input must be finite")
        return ["float", value]
    if isinstance(value, Mapping):
        items: list[list[Any]] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError("fingerprint mappings require non-empty string keys")
            items.append([key.strip(), _canonical_value(item)])
        items.sort(key=lambda item: item[0])
        return ["mapping", items]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_canonical_value(item) for item in value]]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        items.sort(key=_canonical_json)
        return ["set", items]
    raise TypeError("fingerprint input must be a supported JSON-like value")


def _fingerprint(kind: str, value: Any) -> str:
    payload = _canonical_json({"kind": kind, "value": _canonical_value(value)})
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_scope(scope: ConsentScope) -> tuple[str, ...]:
    if isinstance(scope, str):
        values = (scope,)
    else:
        try:
            values = tuple(scope)
        except TypeError as exc:
            raise TypeError(
                "consent scope must be a non-empty string or iterable of strings"
            ) from exc

    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("consent scope values must be strings")
        item = value.strip()
        if not item:
            raise ValueError("consent scope values must not be empty")
        normalized.add(item)
    if not normalized:
        raise ValueError("consent scope must not be empty")
    return tuple(sorted(normalized))


def _require_non_empty_revision(revision: ConsentRevision) -> None:
    if revision is None:
        raise ValueError("consent revision must not be empty")
    if isinstance(revision, str) and not revision.strip():
        raise ValueError("consent revision must not be empty")
    if isinstance(revision, bytes) and not revision:
        raise ValueError("consent revision must not be empty")
    if isinstance(revision, (Mapping, list, tuple, set, frozenset)) and not revision:
        raise ValueError("consent revision must not be empty")


def fingerprint_consent_scope(scope: ConsentScope) -> str:
    """Return a deterministic fingerprint for a consent scope.

    A string scope and an iterable scope are normalized to a sorted set of
    non-empty strings.  The raw scope is not returned or retained.
    """

    return _fingerprint("consent-scope", _normalize_scope(scope))


def fingerprint_consent_revision(revision: ConsentRevision) -> str:
    """Return a deterministic fingerprint for a consent revision.

    Revisions may be scalar or JSON-like structured values.  Only the digest
    is retained by :class:`ConsentCache`.
    """

    _require_non_empty_revision(revision)
    return _fingerprint("consent-revision", revision)


def fingerprint_cache_key(cache_key: Any) -> str:
    """Return a deterministic fingerprint for a cache key.

    Cache keys are fingerprinted so a source identifier supplied as a key is
    not reproduced in entry metadata or audit output.
    """

    if cache_key is None:
        raise ValueError("cache key must not be empty")
    if isinstance(cache_key, str) and not cache_key.strip():
        raise ValueError("cache key must not be empty")
    if isinstance(cache_key, bytes) and not cache_key:
        raise ValueError("cache key must not be empty")
    return _fingerprint("cache-key", cache_key)


def _resolve_alias(
    primary: Any,
    alias: Any,
    *,
    name: str,
    required: bool,
) -> Any:
    if primary is not _MISSING and alias is not _MISSING:
        raise TypeError(f"{name} was supplied more than once")
    value = alias if primary is _MISSING else primary
    if required and value is _MISSING:
        raise TypeError(f"{name} is required")
    return value


@dataclass(frozen=True, slots=True)
class ConsentCacheEntry(Generic[T]):
    """An in-memory cached output with privacy-safe metadata.

    ``value`` is the derived output held by the in-memory cache.  It is
    intentionally excluded from ``repr`` and :meth:`to_dict`; serialized
    metadata contains fingerprints only.
    """

    cache_key_fingerprint: str
    consent_scope_fingerprint: str
    consent_revision_fingerprint: str
    value: T = field(repr=False, compare=False)

    @property
    def key_fingerprint(self) -> str:
        """Alias for :attr:`cache_key_fingerprint`."""

        return self.cache_key_fingerprint

    @property
    def scope_fingerprint(self) -> str:
        """Alias for :attr:`consent_scope_fingerprint`."""

        return self.consent_scope_fingerprint

    @property
    def revision_fingerprint(self) -> str:
        """Alias for :attr:`consent_revision_fingerprint`."""

        return self.consent_revision_fingerprint

    def to_dict(self) -> dict[str, str]:
        """Return metadata without the cached output or raw input values."""

        return {
            "cache_key_fingerprint": self.cache_key_fingerprint,
            "consent_revision_fingerprint": self.consent_revision_fingerprint,
            "consent_scope_fingerprint": self.consent_scope_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ConsentInvalidationEvent:
    """Counts-only audit evidence for one consent revocation operation."""

    invalidated_count: int
    event_type: str = CONSENT_CACHE_EVENT_TYPE

    def __post_init__(self) -> None:
        if self.invalidated_count < 0:
            raise ValueError("invalidated count must not be negative")

    @property
    def count(self) -> int:
        """Alias for :attr:`invalidated_count`."""

        return self.invalidated_count

    def to_dict(self) -> dict[str, int | str]:
        """Return an audit-safe event containing aggregate data only."""

        return {
            "event_type": self.event_type,
            "invalidated_count": self.invalidated_count,
        }


@dataclass(frozen=True, slots=True)
class ConsentCacheStats:
    """Counts-only runtime statistics for a consent-aware cache."""

    hits: int
    misses: int
    writes: int
    rejected_writes: int
    invalidated: int
    size: int

    def to_dict(self) -> dict[str, int]:
        """Return cache statistics as a counts-only mapping."""

        return {
            "hits": self.hits,
            "invalidated": self.invalidated,
            "misses": self.misses,
            "rejected_writes": self.rejected_writes,
            "size": self.size,
            "writes": self.writes,
        }


class ConsentCache(Generic[T]):
    """Bounded in-memory cache whose entries can be revoked deterministically.

    Cache entries are addressed by the tuple of fingerprints for the caller's
    cache key, consent scope, and consent revision.  Passing consent metadata
    to :meth:`get` is recommended; when it is omitted, a value is returned only
    if exactly one consent context exists for that cache key.

    Revoking an exact ``(scope, revision)`` removes all matching entries and
    prevents that pair from being written again for the lifetime of the cache.
    Omitting ``revision`` removes all currently cached revisions for the scope
    and prevents those observed pairs from being written again.  A later,
    previously unseen revision can therefore represent a new consent receipt.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be a positive integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._entries: OrderedDict[_EntryKey, ConsentCacheEntry[Any]] = OrderedDict()
        self._revoked: set[_RevocationKey] = set()
        self._audit_events: list[ConsentInvalidationEvent] = []
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._rejected_writes = 0
        self._invalidated = 0
        self._lock = RLock()

    def put(
        self,
        cache_key: Any,
        value: T,
        scope: ConsentScope | object = _MISSING,
        revision: ConsentRevision | object = _MISSING,
        *,
        consent_scope: ConsentScope | object = _MISSING,
        consent_revision: ConsentRevision | object = _MISSING,
    ) -> bool:
        """Store an output unless its consent pair has been revoked.

        Returns ``True`` when the output is stored and ``False`` when the
        exact scope/revision pair is already revoked.  Consent values and the
        output are never included in an exception, metadata mapping, or audit
        event.
        """

        resolved_scope = _resolve_alias(
            scope,
            consent_scope,
            name="consent scope",
            required=True,
        )
        resolved_revision = _resolve_alias(
            revision,
            consent_revision,
            name="consent revision",
            required=True,
        )
        key_fingerprint = fingerprint_cache_key(cache_key)
        scope_fingerprint = fingerprint_consent_scope(resolved_scope)
        revision_fingerprint = fingerprint_consent_revision(resolved_revision)
        entry_key = (
            key_fingerprint,
            scope_fingerprint,
            revision_fingerprint,
        )

        with self._lock:
            if (scope_fingerprint, revision_fingerprint) in self._revoked:
                self._rejected_writes += 1
                return False
            self._entries[entry_key] = ConsentCacheEntry(
                cache_key_fingerprint=key_fingerprint,
                consent_scope_fingerprint=scope_fingerprint,
                consent_revision_fingerprint=revision_fingerprint,
                value=value,
            )
            self._entries.move_to_end(entry_key)
            self._writes += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return True

    def set(
        self,
        cache_key: Any,
        value: T,
        scope: ConsentScope | object = _MISSING,
        revision: ConsentRevision | object = _MISSING,
        *,
        consent_scope: ConsentScope | object = _MISSING,
        consent_revision: ConsentRevision | object = _MISSING,
    ) -> bool:
        """Alias for :meth:`put`."""

        return self.put(
            cache_key,
            value,
            scope,
            revision,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
        )

    def get_entry(
        self,
        cache_key: Any,
        scope: ConsentScope | object = _MISSING,
        revision: ConsentRevision | object = _MISSING,
        *,
        consent_scope: ConsentScope | object = _MISSING,
        consent_revision: ConsentRevision | object = _MISSING,
    ) -> ConsentCacheEntry[T] | None:
        """Return a matching entry, or ``None`` for a cache miss.

        If scope and revision are omitted, the lookup succeeds only when one
        entry exists for the cache key.  This prevents an ambiguous context
        from selecting an arbitrary cached output.
        """

        resolved_scope = _resolve_alias(
            scope,
            consent_scope,
            name="consent scope",
            required=False,
        )
        resolved_revision = _resolve_alias(
            revision,
            consent_revision,
            name="consent revision",
            required=False,
        )
        key_fingerprint = fingerprint_cache_key(cache_key)
        scope_fingerprint = (
            None
            if resolved_scope is _MISSING
            else fingerprint_consent_scope(resolved_scope)
        )
        revision_fingerprint = (
            None
            if resolved_revision is _MISSING
            else fingerprint_consent_revision(resolved_revision)
        )

        with self._lock:
            matches = [
                entry_key
                for entry_key in self._entries
                if entry_key[0] == key_fingerprint
                and (scope_fingerprint is None or entry_key[1] == scope_fingerprint)
                and (
                    revision_fingerprint is None or entry_key[2] == revision_fingerprint
                )
            ]
            if len(matches) != 1:
                self._misses += 1
                return None
            entry_key = matches[0]
            self._entries.move_to_end(entry_key)
            self._hits += 1
            return self._entries[entry_key]  # type: ignore[return-value]

    def get(
        self,
        cache_key: Any,
        scope: ConsentScope | object = _MISSING,
        revision: ConsentRevision | object = _MISSING,
        *,
        consent_scope: ConsentScope | object = _MISSING,
        consent_revision: ConsentRevision | object = _MISSING,
        default: T | None = None,
    ) -> T | None:
        """Return a matching cached output, or ``default`` on a miss."""

        entry = self.get_entry(
            cache_key,
            scope,
            revision,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
        )
        return default if entry is None else entry.value

    def revoke(
        self,
        scope: ConsentScope | object = _MISSING,
        revision: ConsentRevision | object = _MISSING,
        *,
        consent_scope: ConsentScope | object = _MISSING,
        consent_revision: ConsentRevision | object = _MISSING,
    ) -> ConsentInvalidationEvent:
        """Invalidate matching entries and append a counts-only audit event.

        With a revision, only the exact scope/revision pair is invalidated.
        Without one, every currently cached revision for the scope is
        invalidated.  Repeating a revocation is deterministic and returns an
        event with an invalidated count of zero when nothing remains.
        """

        resolved_scope = _resolve_alias(
            scope,
            consent_scope,
            name="consent scope",
            required=True,
        )
        resolved_revision = _resolve_alias(
            revision,
            consent_revision,
            name="consent revision",
            required=False,
        )
        scope_fingerprint = fingerprint_consent_scope(resolved_scope)
        revision_fingerprint = (
            None
            if resolved_revision is _MISSING or resolved_revision is None
            else fingerprint_consent_revision(resolved_revision)
        )

        with self._lock:
            if revision_fingerprint is None:
                revoked_revisions = {
                    entry_key[2]
                    for entry_key in self._entries
                    if entry_key[1] == scope_fingerprint
                }
            else:
                revoked_revisions = {revision_fingerprint}
            self._revoked.update(
                (scope_fingerprint, item) for item in revoked_revisions
            )
            matching_keys = [
                entry_key
                for entry_key in self._entries
                if entry_key[1] == scope_fingerprint
                and entry_key[2] in revoked_revisions
            ]
            for entry_key in matching_keys:
                del self._entries[entry_key]
            event = ConsentInvalidationEvent(len(matching_keys))
            self._audit_events.append(event)
            self._invalidated += event.invalidated_count
        return event

    def invalidate(
        self,
        scope: ConsentScope | object = _MISSING,
        revision: ConsentRevision | object = _MISSING,
        *,
        consent_scope: ConsentScope | object = _MISSING,
        consent_revision: ConsentRevision | object = _MISSING,
    ) -> ConsentInvalidationEvent:
        """Alias for :meth:`revoke`."""

        return self.revoke(
            scope,
            revision,
            consent_scope=consent_scope,
            consent_revision=consent_revision,
        )

    def is_revoked(
        self,
        scope: ConsentScope,
        revision: ConsentRevision,
    ) -> bool:
        """Return whether an exact consent scope/revision pair is revoked."""

        scope_fingerprint = fingerprint_consent_scope(scope)
        revision_fingerprint = fingerprint_consent_revision(revision)
        with self._lock:
            return (scope_fingerprint, revision_fingerprint) in self._revoked

    def entries(self) -> tuple[ConsentCacheEntry[Any], ...]:
        """Return entries in deterministic fingerprint order.

        Use :meth:`ConsentCacheEntry.to_dict` when exposing this metadata to an
        audit or report surface; it intentionally omits cached output values.
        """

        with self._lock:
            return tuple(
                sorted(
                    self._entries.values(),
                    key=lambda entry: (
                        entry.cache_key_fingerprint,
                        entry.consent_scope_fingerprint,
                        entry.consent_revision_fingerprint,
                    ),
                )
            )

    def entry_metadata(self) -> tuple[dict[str, str], ...]:
        """Return deterministic, fingerprint-only entry metadata."""

        return tuple(entry.to_dict() for entry in self.entries())

    @property
    def audit_events(self) -> tuple[ConsentInvalidationEvent, ...]:
        """Return counts-only revocation events in operation order."""

        with self._lock:
            return tuple(self._audit_events)

    def audit_log(self) -> tuple[dict[str, int | str], ...]:
        """Return the audit events as counts-only dictionaries."""

        return tuple(event.to_dict() for event in self.audit_events)

    def stats(self) -> ConsentCacheStats:
        """Return counts-only cache statistics."""

        with self._lock:
            return ConsentCacheStats(
                hits=self._hits,
                misses=self._misses,
                writes=self._writes,
                rejected_writes=self._rejected_writes,
                invalidated=self._invalidated,
                size=len(self._entries),
            )

    def clear(self) -> int:
        """Remove cached outputs while retaining revocation tombstones."""

        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            return removed

    def clear_audit_events(self) -> int:
        """Clear retained audit events and return the number removed."""

        with self._lock:
            removed = len(self._audit_events)
            self._audit_events.clear()
            return removed

    def __contains__(self, cache_key: object) -> bool:
        """Return whether one or more entries exist for ``cache_key``."""

        key_fingerprint = fingerprint_cache_key(cache_key)
        with self._lock:
            return any(entry_key[0] == key_fingerprint for entry_key in self._entries)

    def __len__(self) -> int:
        """Return the number of cached consent-context entries."""

        with self._lock:
            return len(self._entries)
