"""Pluggable, local-first providers for reversible-surrogate key material.

The existing surrogate vault owns mapping and encrypted-file semantics.  This
module defines the narrower key-custody boundary that a vault or another local
store can depend on.  Providers expose key material only through an explicit
lookup result; lifecycle metadata never contains the material itself.

The bundled provider is intentionally synthetic and deterministic.  It is a
testable local implementation, not a claim that an in-memory seed provides
production key custody or cryptographic erasure.  Deployments that need a
different custody boundary can implement :class:`SurrogateKeyProvider` and
declare only the lifecycle capabilities they can enforce.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_SYNTHETIC_SEED",
    "InMemorySurrogateKeyProvider",
    "KeyProviderCapability",
    "KeyProviderCapabilities",
    "LocalSurrogateKeyProvider",
    "LocalSyntheticSurrogateKeyProvider",
    "MissingKeyCapabilityError",
    "MissingSurrogateKeyCapabilityError",
    "SurrogateKeyCapability",
    "SurrogateKeyCapabilities",
    "SurrogateKeyDestroyedError",
    "SurrogateKeyMaterial",
    "SurrogateKeyProvider",
    "SurrogateKeyProviderError",
    "SurrogateKeyScope",
    "SyntheticSurrogateKeyProvider",
    "require_capabilities",
    "require_surrogate_key_capabilities",
    "validate_surrogate_key_provider",
]


DEFAULT_SYNTHETIC_SEED = b"openmed.synthetic-surrogate-key-provider.v1"
_KEY_ID_PREFIX = "sk"
_KEY_MATERIAL_BYTES = 32
_MAX_SCOPE_LENGTH = 128
_CAPABILITY_ORDER = ("lookup", "rotation", "scope", "destruction")


class SurrogateKeyCapability(str, Enum):
    """Lifecycle operations a surrogate-key provider may support."""

    LOOKUP = "lookup"
    ROTATION = "rotation"
    SCOPE = "scope"
    DESTRUCTION = "destruction"


# A shorter spelling is useful for adapters that do not otherwise use the
# ``Surrogate`` prefix.  Both names identify the same enum.
KeyProviderCapability = SurrogateKeyCapability


def _capability_name(value: SurrogateKeyCapability | str) -> str:
    if isinstance(value, SurrogateKeyCapability):
        return value.value
    if not isinstance(value, str):
        raise TypeError("capability names must be strings")
    name = value.strip().lower()
    aliases = {
        "key_lookup": "lookup",
        "key_rotation": "rotation",
        "key_scope": "scope",
        "key_destruction": "destruction",
        "rotate": "rotation",
        "destroy": "destruction",
    }
    name = aliases.get(name, name)
    if name not in _CAPABILITY_ORDER:
        raise ValueError("unsupported surrogate-key capability")
    return name


@dataclass(frozen=True)
class SurrogateKeyCapabilities:
    """Declared lifecycle capabilities for one provider.

    Capabilities default to ``False`` so an adapter must opt into each
    operation explicitly.  The local provider opts into all four operations.
    """

    lookup: bool = False
    rotation: bool = False
    scope: bool = False
    destruction: bool = False

    def __post_init__(self) -> None:
        for name in _CAPABILITY_ORDER:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} capability must be boolean")

    @classmethod
    def all(cls) -> "SurrogateKeyCapabilities":
        """Return capabilities for a provider with the complete contract."""

        return cls(lookup=True, rotation=True, scope=True, destruction=True)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "SurrogateKeyCapabilities":
        """Build capabilities from JSON-like metadata.

        Unknown fields are rejected so an adapter cannot silently advertise a
        misspelled lifecycle operation.
        """

        normalized: dict[str, bool] = {}
        for key, value in values.items():
            name = _capability_name(key)
            if name in normalized:
                raise ValueError("duplicate surrogate-key capability")
            normalized[name] = value
        return cls(**normalized)

    @property
    def key_lookup(self) -> bool:
        """Return whether key lookup is supported."""

        return self.lookup

    @property
    def key_rotation(self) -> bool:
        """Return whether key rotation is supported."""

        return self.rotation

    @property
    def key_scope(self) -> bool:
        """Return whether scope inspection is supported."""

        return self.scope

    @property
    def key_destruction(self) -> bool:
        """Return whether key destruction is supported."""

        return self.destruction

    @property
    def supported(self) -> tuple[str, ...]:
        """Return supported capability names in stable order."""

        return tuple(name for name in _CAPABILITY_ORDER if getattr(self, name))

    def supports(self, capability: SurrogateKeyCapability | str) -> bool:
        """Return whether one lifecycle capability is declared."""

        return bool(getattr(self, _capability_name(capability)))

    def missing(
        self,
        required: Iterable[SurrogateKeyCapability | str],
    ) -> tuple[str, ...]:
        """Return required capability names that are not declared."""

        names = {_capability_name(value) for value in required}
        return tuple(
            name
            for name in _CAPABILITY_ORDER
            if name in names and not getattr(self, name)
        )

    def as_dict(self) -> dict[str, bool]:
        """Return deterministic, JSON-compatible capability metadata."""

        return {name: bool(getattr(self, name)) for name in _CAPABILITY_ORDER}


KeyProviderCapabilities = SurrogateKeyCapabilities


class SurrogateKeyProviderError(RuntimeError):
    """Base error for provider contract and lifecycle failures."""


class MissingSurrogateKeyCapabilityError(SurrogateKeyProviderError):
    """Raised before an operation when its required capability is absent."""

    def __init__(
        self,
        missing: Iterable[SurrogateKeyCapability | str],
    ) -> None:
        missing_names = {_capability_name(value) for value in missing}
        names = tuple(name for name in _CAPABILITY_ORDER if name in missing_names)
        if not names:
            raise ValueError("at least one missing capability is required")
        self.missing = names
        super().__init__(
            "surrogate-key provider lacks required capabilities: " + ", ".join(names)
        )


MissingKeyCapabilityError = MissingSurrogateKeyCapabilityError


class SurrogateKeyDestroyedError(SurrogateKeyProviderError):
    """Raised when a destroyed scope is used for a new lifecycle operation."""


def _provider_name(provider: object) -> str:
    return type(provider).__name__


def _capabilities_for(provider: object) -> SurrogateKeyCapabilities:
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(capabilities, SurrogateKeyCapabilities):
        raise SurrogateKeyProviderError(
            "surrogate-key provider "
            f"{_provider_name(provider)!r} has no valid capability metadata"
        )
    return capabilities


def require_surrogate_key_capabilities(
    provider: object,
    required: Iterable[SurrogateKeyCapability | str]
    | SurrogateKeyCapability
    | str = (),
    *additional: SurrogateKeyCapability | str,
) -> None:
    """Fail closed unless ``provider`` declares every required capability.

    ``required`` may be one capability, an iterable, or the first item in a
    positional list.  The helper deliberately checks metadata before an
    adapter method can perform a partial operation.
    """

    if isinstance(required, (SurrogateKeyCapability, str)):
        values: tuple[SurrogateKeyCapability | str, ...] = (required, *additional)
    else:
        values = (*tuple(required), *additional)
    capabilities = _capabilities_for(provider)
    missing = capabilities.missing(values)
    if missing:
        raise MissingSurrogateKeyCapabilityError(missing)


require_capabilities = require_surrogate_key_capabilities


def validate_surrogate_key_provider(
    provider: object,
    required: Iterable[SurrogateKeyCapability | str] = (),
) -> SurrogateKeyProvider:
    """Validate a provider contract and return it for typed composition."""

    if not isinstance(provider, SurrogateKeyProvider):
        raise TypeError("provider does not implement the surrogate-key contract")
    require_surrogate_key_capabilities(provider, required)
    return provider


def _validate_scope(scope: str) -> str:
    if not isinstance(scope, str):
        raise TypeError("surrogate-key scope must be a string")
    normalized = scope.strip()
    if not normalized or len(normalized) > _MAX_SCOPE_LENGTH:
        raise ValueError("surrogate-key scope must be a non-empty bounded identifier")
    return normalized


def _validate_key_id(key_id: str) -> str:
    if not isinstance(key_id, str) or not key_id:
        raise TypeError("surrogate-key key_id must be a non-empty string")
    return key_id


@dataclass(frozen=True, repr=False)
class SurrogateKeyMaterial:
    """One versioned key and its non-sensitive lookup metadata.

    ``material`` is intentionally excluded from ``repr`` and comparisons.  A
    caller should pass it directly to the local cryptographic operation that
    needs it and use :meth:`as_metadata` for logs, reports, or audit records.
    """

    key_id: str
    scope: str
    version: int
    material: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_key_id(self.key_id)
        _validate_scope(self.scope)
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("surrogate-key version must be an integer")
        if self.version < 1:
            raise ValueError("surrogate-key version must be positive")
        if not isinstance(self.material, bytes) or not self.material:
            raise ValueError("surrogate-key material must be non-empty bytes")

    @property
    def key_material(self) -> bytes:
        """Return the key bytes for the caller's local cryptographic use."""

        return self.material

    @property
    def fingerprint(self) -> str:
        """Return a stable, non-secret fingerprint for diagnostics."""

        return "sha256:" + hashlib.sha256(self.material).hexdigest()

    def as_metadata(self) -> dict[str, Any]:
        """Return metadata that excludes raw key material."""

        return {
            "key_id": self.key_id,
            "scope": self.scope,
            "version": self.version,
            "material_bytes": len(self.material),
            "fingerprint": self.fingerprint,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(key_id={self.key_id!r}, "
            f"scope={self.scope!r}, version={self.version})"
        )


SurrogateKey = SurrogateKeyMaterial


@dataclass(frozen=True)
class SurrogateKeyScope:
    """Privacy-safe state for one provider scope."""

    name: str
    active_key_id: str | None
    active_version: int | None
    key_versions: tuple[int, ...]
    destroyed: bool
    capabilities: SurrogateKeyCapabilities

    def __post_init__(self) -> None:
        _validate_scope(self.name)
        if self.active_key_id is not None:
            _validate_key_id(self.active_key_id)
        if self.active_version is not None and self.active_version < 1:
            raise ValueError("active key version must be positive")
        if any(version < 1 for version in self.key_versions):
            raise ValueError("key versions must be positive")
        if tuple(sorted(set(self.key_versions))) != self.key_versions:
            raise ValueError("key versions must be sorted and unique")
        if not isinstance(self.destroyed, bool):
            raise TypeError("destroyed must be boolean")

    @property
    def scope(self) -> str:
        """Return the scope name (an opaque, caller-defined identifier)."""

        return self.name

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic scope metadata without key material."""

        return {
            "scope": self.name,
            "active_key_id": self.active_key_id,
            "active_version": self.active_version,
            "key_versions": list(self.key_versions),
            "destroyed": self.destroyed,
            "capabilities": self.capabilities.as_dict(),
        }


@runtime_checkable
class SurrogateKeyProvider(Protocol):
    """Contract implemented by local or user-managed key providers."""

    @property
    def capabilities(self) -> SurrogateKeyCapabilities:
        """Return declared lifecycle capabilities."""

        ...

    def lookup(
        self,
        scope: str,
        key_id: str | None = None,
    ) -> SurrogateKeyMaterial | None:
        """Look up the active or explicitly identified key in ``scope``."""

        ...

    def rotate(self, scope: str) -> SurrogateKeyMaterial:
        """Create and return the next key version for ``scope``."""

        ...

    def scope(self, scope: str) -> SurrogateKeyScope:
        """Return non-secret lifecycle metadata for ``scope``."""

        ...

    def destroy(self, scope: str, key_id: str | None = None) -> None:
        """Destroy one key or the complete scope when ``key_id`` is omitted."""

        ...


class LocalSyntheticSurrogateKeyProvider:
    """Deterministic in-memory provider for offline tests and local demos.

    The provider derives 32-byte key material from ``seed``, scope, and key
    version with HMAC-SHA256.  It never contacts a network or writes a file.
    Destruction removes the material held by this instance and permanently
    marks a destroyed scope unavailable for the rest of the instance lifetime.
    """

    def __init__(
        self,
        seed: str | bytes = DEFAULT_SYNTHETIC_SEED,
        *,
        capabilities: SurrogateKeyCapabilities | Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(seed, str):
            seed_bytes = seed.encode("utf-8")
        elif isinstance(seed, bytes):
            seed_bytes = seed
        else:
            raise TypeError("synthetic provider seed must be text or bytes")
        if not seed_bytes:
            raise ValueError("synthetic provider seed must be non-empty")
        if isinstance(capabilities, Mapping):
            capabilities = SurrogateKeyCapabilities.from_mapping(capabilities)
        elif capabilities is not None and not isinstance(
            capabilities,
            SurrogateKeyCapabilities,
        ):
            raise TypeError(
                "capabilities must be SurrogateKeyCapabilities or a mapping"
            )
        self._seed = seed_bytes
        self._capabilities = capabilities or SurrogateKeyCapabilities.all()
        self._keys: dict[str, dict[int, SurrogateKeyMaterial]] = {}
        self._active_versions: dict[str, int] = {}
        self._next_versions: dict[str, int] = {}
        self._destroyed_scopes: set[str] = set()
        self._lock = RLock()

    @property
    def capabilities(self) -> SurrogateKeyCapabilities:
        """Return the provider's declared lifecycle capabilities."""

        return self._capabilities

    @property
    def capability_metadata(self) -> dict[str, Any]:
        """Return deterministic provider metadata without seed or key bytes."""

        return {
            "provider": type(self).__name__,
            "deterministic": True,
            "network_required": False,
            "capabilities": self.capabilities.as_dict(),
        }

    def lookup(
        self,
        scope: str,
        key_id: str | None = None,
    ) -> SurrogateKeyMaterial | None:
        """Return a scope's active key, materializing version one if needed."""

        require_surrogate_key_capabilities(self, SurrogateKeyCapability.LOOKUP)
        normalized_scope = _validate_scope(scope)
        if key_id is not None:
            key_id = _validate_key_id(key_id)
        with self._lock:
            if normalized_scope in self._destroyed_scopes:
                return None
            versions = self._keys.setdefault(normalized_scope, {})
            if not versions:
                next_version = self._next_versions.get(normalized_scope, 1)
                first = self._new_key(normalized_scope, next_version)
                versions[next_version] = first
                self._active_versions[normalized_scope] = next_version
                self._next_versions[normalized_scope] = next_version + 1
            if key_id is not None:
                return next(
                    (key for key in versions.values() if key.key_id == key_id),
                    None,
                )
            return versions[self._active_versions[normalized_scope]]

    def rotate(self, scope: str) -> SurrogateKeyMaterial:
        """Create the next deterministic key version for ``scope``."""

        require_surrogate_key_capabilities(self, SurrogateKeyCapability.ROTATION)
        normalized_scope = _validate_scope(scope)
        with self._lock:
            self._ensure_scope_available(normalized_scope)
            versions = self._keys.setdefault(normalized_scope, {})
            next_version = self._next_versions.get(normalized_scope, 1)
            key = self._new_key(normalized_scope, next_version)
            versions[next_version] = key
            self._active_versions[normalized_scope] = next_version
            self._next_versions[normalized_scope] = next_version + 1
            return key

    def scope(self, scope: str) -> SurrogateKeyScope:
        """Return lifecycle metadata for ``scope`` without exposing a key."""

        require_surrogate_key_capabilities(self, SurrogateKeyCapability.SCOPE)
        normalized_scope = _validate_scope(scope)
        with self._lock:
            versions = self._keys.get(normalized_scope, {})
            active_version = self._active_versions.get(normalized_scope)
            active_key = (
                versions.get(active_version) if active_version is not None else None
            )
            return SurrogateKeyScope(
                name=normalized_scope,
                active_key_id=active_key.key_id if active_key is not None else None,
                active_version=active_version,
                key_versions=tuple(sorted(versions)),
                destroyed=normalized_scope in self._destroyed_scopes,
                capabilities=self.capabilities,
            )

    def destroy(self, scope: str, key_id: str | None = None) -> None:
        """Destroy one key, or permanently destroy all keys in a scope."""

        require_surrogate_key_capabilities(self, SurrogateKeyCapability.DESTRUCTION)
        normalized_scope = _validate_scope(scope)
        if key_id is not None:
            key_id = _validate_key_id(key_id)
        with self._lock:
            if normalized_scope in self._destroyed_scopes:
                return
            versions = self._keys.get(normalized_scope, {})
            if key_id is None:
                versions.clear()
                self._keys.pop(normalized_scope, None)
                self._active_versions.pop(normalized_scope, None)
                self._next_versions.pop(normalized_scope, None)
                self._destroyed_scopes.add(normalized_scope)
                return

            version = next(
                (version for version, key in versions.items() if key.key_id == key_id),
                None,
            )
            if version is None:
                return
            del versions[version]
            if not versions:
                self._keys.pop(normalized_scope, None)
                self._active_versions.pop(normalized_scope, None)
            elif self._active_versions.get(normalized_scope) == version:
                self._active_versions[normalized_scope] = max(versions)

    def lookup_key(
        self,
        scope: str,
        key_id: str | None = None,
    ) -> SurrogateKeyMaterial | None:
        """Alias for :meth:`lookup` for adapters using explicit key wording."""

        return self.lookup(scope, key_id)

    def get_key(
        self,
        scope: str,
        key_id: str | None = None,
    ) -> SurrogateKeyMaterial | None:
        """Alias for :meth:`lookup`."""

        return self.lookup(scope, key_id)

    def rotate_key(self, scope: str) -> SurrogateKeyMaterial:
        """Alias for :meth:`rotate`."""

        return self.rotate(scope)

    def get_scope(self, scope: str) -> SurrogateKeyScope:
        """Alias for :meth:`scope`."""

        return self.scope(scope)

    def destroy_key(self, scope: str, key_id: str | None = None) -> None:
        """Alias for :meth:`destroy`."""

        self.destroy(scope, key_id)

    def destroy_scope(self, scope: str) -> None:
        """Destroy every key in ``scope``."""

        self.destroy(scope)

    def _ensure_scope_available(self, scope: str) -> None:
        if scope in self._destroyed_scopes:
            raise SurrogateKeyDestroyedError(
                "surrogate-key scope was destroyed and cannot be recreated"
            )

    def _new_key(self, scope: str, version: int) -> SurrogateKeyMaterial:
        scope_bytes = scope.encode("utf-8")
        version_bytes = version.to_bytes(8, "big")
        domain = b"openmed:surrogate-key-provider:v1\x00"
        material = hmac.new(
            self._seed,
            domain + len(scope_bytes).to_bytes(4, "big") + scope_bytes + version_bytes,
            hashlib.sha256,
        ).digest()[:_KEY_MATERIAL_BYTES]
        key_id_digest = hmac.new(
            self._seed,
            b"openmed:surrogate-key-id:v1\x00" + material,
            hashlib.sha256,
        ).hexdigest()[:24]
        return SurrogateKeyMaterial(
            key_id=f"{_KEY_ID_PREFIX}-{version:04d}-{key_id_digest}",
            scope=scope,
            version=version,
            material=material,
        )


LocalSurrogateKeyProvider = LocalSyntheticSurrogateKeyProvider
InMemorySurrogateKeyProvider = LocalSyntheticSurrogateKeyProvider
SyntheticSurrogateKeyProvider = LocalSyntheticSurrogateKeyProvider
