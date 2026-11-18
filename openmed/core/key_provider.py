"""Caller-supplied key providers for application-layer envelope encryption.

OpenMed never generates, stores, or escrows long-lived keys for local stores.
A caller (or an out-of-process custody bridge) implements :class:`KeyProvider`
and hands OpenMed short-lived :class:`DataKey` handles at the moment one
encryption or decryption is performed.  Key identifiers are opaque, non-secret
labels that may be persisted next to ciphertext; key material is never
persisted, logged, serialized, or included in a representation or exception.

Every failure is a typed, value-free :class:`EncryptionError` carrying one
controlled ``code``.  Messages never contain key identifiers, key material,
record identifiers, or plaintext, so callers can surface them safely.

The bundled :class:`InMemoryKeyProvider` holds caller-supplied material in the
current process only.  It is suitable for tests, synthetic runs, and for
adapters that load material from their own custody system; it is not, by
itself, a production custody boundary.
"""

from __future__ import annotations

import re
import secrets
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NoReturn, Protocol, runtime_checkable

__all__ = [
    "DATA_KEY_BYTES",
    "DataKey",
    "EncryptionError",
    "InMemoryKeyProvider",
    "KeyMaterialError",
    "KeyProvider",
    "KeyProviderError",
    "KeyRetiredError",
    "KeyRevokedError",
    "KeyState",
    "KeyUnavailableError",
    "register_error_messages",
    "require_decryption_key",
    "require_encryption_key",
    "validate_key_provider",
]

DATA_KEY_BYTES = 32

# Same grammar as ``openmed.core.key_custody`` key identifiers so one custody
# inventory can describe the keys a provider serves.
_KEY_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9._:/-]{2,127}\Z")
_PURPOSE_RE = re.compile(r"[a-z][a-z0-9._-]{1,63}\Z")
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")

_MESSAGES = {
    "encryption_failed": "encryption operation failed",
    "key_provider_failed": "key provider failed",
    "key_unavailable": "encryption key is unavailable",
    "key_revoked": "encryption key was revoked",
    "key_retired": "encryption key was retired",
    "key_material_invalid": "encryption key material is invalid",
}


class EncryptionError(RuntimeError):
    """Base class for fail-closed, value-free encryption errors.

    Attributes:
        code: A controlled identifier that names the failure. It never
            contains key identifiers, key material, or record content.
    """

    default_code = "encryption_failed"

    def __init__(self, code: str | None = None) -> None:
        selected = self.default_code if code is None else code
        if not isinstance(selected, str) or _CODE_RE.fullmatch(selected) is None:
            selected = self.default_code
        self.code = selected
        message = _MESSAGES.get(selected) or _ERROR_MESSAGES.get(selected)
        super().__init__(f"{message or 'encryption operation failed'} ({selected})")


# Messages registered by modules that build on this error hierarchy.
_ERROR_MESSAGES: dict[str, str] = {}


def register_error_messages(messages: Mapping[str, str]) -> None:
    """Register fixed, value-free messages for additional error codes.

    Args:
        messages: Mapping of controlled codes to fixed messages. Messages must
            be constant text; they are never formatted with runtime values.
    """

    for code, message in messages.items():
        if _CODE_RE.fullmatch(code) is None or not isinstance(message, str):
            raise ValueError("error messages must use controlled codes")
        _ERROR_MESSAGES.setdefault(code, message)


class KeyProviderError(EncryptionError):
    """Raised when a key provider cannot serve a usable key."""

    default_code = "key_provider_failed"


class KeyUnavailableError(KeyProviderError):
    """Raised for an unknown, destroyed, or unreachable key."""

    default_code = "key_unavailable"


class KeyRevokedError(KeyProviderError):
    """Raised when a revoked key is requested for any operation."""

    default_code = "key_revoked"


class KeyRetiredError(KeyProviderError):
    """Raised when a retired key is requested for any operation."""

    default_code = "key_retired"


class KeyMaterialError(KeyProviderError):
    """Raised when a provider returns malformed key material."""

    default_code = "key_material_invalid"


class KeyState(str, Enum):
    """Lifecycle state of one data key.

    ``ACTIVE`` keys encrypt and decrypt.  ``DECRYPT_ONLY`` keys were superseded
    by a rotation and remain readable until re-encryption completes.
    ``RETIRED`` and ``REVOKED`` keys fail closed for every operation.
    """

    ACTIVE = "active"
    DECRYPT_ONLY = "decrypt_only"
    RETIRED = "retired"
    REVOKED = "revoked"


@dataclass(frozen=True, eq=False, repr=False)
class DataKey:
    """One symmetric data key handed to a single cryptographic operation.

    Args:
        key_id: Opaque, non-secret identifier persisted beside ciphertext.
        purpose: Controlled purpose label that scopes the key, for example
            ``"journey-store"``.
        state: The key's lifecycle state.
        material: Exactly 32 bytes of key material. Excluded from ``repr``,
            comparisons, copying, and pickling.

    Raises:
        KeyMaterialError: If the identifier, purpose, state, or material is
            malformed.
    """

    key_id: str
    purpose: str
    state: KeyState
    material: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key_id, str)
            or _KEY_ID_RE.fullmatch(self.key_id) is None
        ):
            raise KeyMaterialError()
        if (
            not isinstance(self.purpose, str)
            or _PURPOSE_RE.fullmatch(self.purpose) is None
        ):
            raise KeyMaterialError()
        if not isinstance(self.state, KeyState):
            raise KeyMaterialError()
        if not isinstance(self.material, bytes) or len(self.material) != DATA_KEY_BYTES:
            raise KeyMaterialError()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(key_id={self.key_id!r}, "
            f"purpose={self.purpose!r}, state={self.state.value!r})"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("key material cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("key material cannot be copied")

    def __deepcopy__(self, memo: Any) -> NoReturn:
        raise TypeError("key material cannot be copied")


@runtime_checkable
class KeyProvider(Protocol):
    """Caller-supplied source of data keys for envelope encryption.

    Implementations may consult an operating-system keychain, a hardware
    module, or an out-of-process custody bridge.  They must raise
    :class:`KeyProviderError` subclasses (never generic exceptions carrying
    values) and must not log or persist key material.
    """

    def active_key(self, purpose: str) -> DataKey:
        """Return the single key currently used to encrypt for ``purpose``.

        Args:
            purpose: Controlled purpose label.

        Returns:
            A :class:`DataKey` in the ``ACTIVE`` state.

        Raises:
            KeyUnavailableError: If no active key exists.
        """

    def resolve_key(self, key_id: str, *, purpose: str) -> DataKey:
        """Return the identified key so ciphertext can be decrypted.

        Args:
            key_id: Identifier read from an envelope header.
            purpose: Controlled purpose label the key must belong to.

        Returns:
            The identified :class:`DataKey` with its current state.

        Raises:
            KeyUnavailableError: If the key is unknown or destroyed.
            KeyRetiredError: If the key was retired.
            KeyRevokedError: If the key was revoked.
        """


def validate_key_provider(provider: object) -> KeyProvider:
    """Return ``provider`` when it implements :class:`KeyProvider`.

    Args:
        provider: Candidate provider object.

    Returns:
        The same object, typed as a key provider.

    Raises:
        TypeError: If the object lacks the provider methods.
    """

    if not isinstance(provider, KeyProvider):
        raise TypeError("object does not implement the key-provider contract")
    return provider


def require_encryption_key(key: object, *, purpose: str) -> DataKey:
    """Fail closed unless ``key`` may encrypt new data for ``purpose``.

    Args:
        key: Value returned by :meth:`KeyProvider.active_key`.
        purpose: The purpose the caller requested.

    Returns:
        The validated active key.

    Raises:
        KeyUnavailableError: If the value is not an active key for the purpose.
        KeyRetiredError: If the key is retired.
        KeyRevokedError: If the key is revoked.
    """

    if not isinstance(key, DataKey) or key.purpose != purpose:
        raise KeyUnavailableError()
    _raise_for_terminal_state(key.state)
    if key.state is not KeyState.ACTIVE:
        raise KeyUnavailableError()
    return key


def require_decryption_key(key: object, *, purpose: str, key_id: str) -> DataKey:
    """Fail closed unless ``key`` may decrypt data sealed under ``key_id``.

    Args:
        key: Value returned by :meth:`KeyProvider.resolve_key`.
        purpose: The purpose the caller requested.
        key_id: The identifier recorded in the envelope.

    Returns:
        The validated key.

    Raises:
        KeyUnavailableError: If the value does not match the requested key.
        KeyRetiredError: If the key is retired.
        KeyRevokedError: If the key is revoked.
    """

    if not isinstance(key, DataKey) or key.purpose != purpose or key.key_id != key_id:
        raise KeyUnavailableError()
    _raise_for_terminal_state(key.state)
    return key


def _raise_for_terminal_state(state: KeyState) -> None:
    if state is KeyState.REVOKED:
        raise KeyRevokedError()
    if state is KeyState.RETIRED:
        raise KeyRetiredError()


@dataclass
class _HeldKey:
    purpose: str
    state: KeyState
    material: bytes | None = field(repr=False)


class InMemoryKeyProvider:
    """Process-local provider for caller-supplied key material.

    Material is held only in this object's memory.  Revoking a key drops its
    material.  Exactly one key per purpose is active; rotating demotes the
    previous active key to ``DECRYPT_ONLY`` so existing ciphertext stays
    readable while it is re-encrypted.

    Example:
        >>> provider = InMemoryKeyProvider()
        >>> provider.add_key(
        ...     "journey-key-1", bytes(32), purpose="journey-store",
        ...     state=KeyState.ACTIVE,
        ... )
        >>> provider.active_key("journey-store").key_id
        'journey-key-1'
    """

    def __init__(self) -> None:
        self._keys: dict[str, _HeldKey] = {}
        self._active: dict[str, str] = {}
        self._lock = threading.RLock()

    @classmethod
    def generate(
        cls,
        purpose: str,
        *,
        key_id: str = "ephemeral-key-1",
    ) -> "InMemoryKeyProvider":
        """Return a provider holding one fresh random active key.

        The key exists only in this process, so data sealed with it becomes
        unreadable once the provider is discarded.  Use it for synthetic,
        throw-away stores and tests.

        Args:
            purpose: Controlled purpose label for the generated key.
            key_id: Identifier for the generated key.

        Returns:
            A provider with one active key.
        """

        provider = cls()
        provider.add_key(
            key_id,
            secrets.token_bytes(DATA_KEY_BYTES),
            purpose=purpose,
            state=KeyState.ACTIVE,
        )
        return provider

    def add_key(
        self,
        key_id: str,
        material: bytes | bytearray | memoryview,
        *,
        purpose: str,
        state: KeyState = KeyState.DECRYPT_ONLY,
    ) -> None:
        """Register caller-supplied material under an identifier.

        Adding an ``ACTIVE`` key demotes the previous active key for the same
        purpose to ``DECRYPT_ONLY``.

        Args:
            key_id: Opaque, non-secret key identifier.
            material: Exactly 32 bytes of key material; it is copied.
            purpose: Controlled purpose label.
            state: Initial lifecycle state.

        Raises:
            KeyMaterialError: If any value is malformed.
            ValueError: If the identifier is already registered.
        """

        if not isinstance(material, (bytes, bytearray, memoryview)):
            raise KeyMaterialError()
        candidate = DataKey(
            key_id=key_id,
            purpose=purpose,
            state=state,
            material=bytes(material),
        )
        with self._lock:
            if key_id in self._keys:
                raise ValueError("key identifier is already registered")
            material_copy = None if state is KeyState.REVOKED else candidate.material
            self._keys[key_id] = _HeldKey(purpose, state, material_copy)
            if state is KeyState.ACTIVE:
                self._demote_active(purpose)
                self._active[purpose] = key_id

    def rotate(
        self,
        key_id: str,
        material: bytes | bytearray | memoryview,
        *,
        purpose: str,
    ) -> DataKey:
        """Activate new material and demote the previous active key.

        Args:
            key_id: Identifier for the new active key.
            material: Exactly 32 bytes of new key material.
            purpose: Controlled purpose label.

        Returns:
            The new active key.
        """

        self.add_key(key_id, material, purpose=purpose, state=KeyState.ACTIVE)
        return self.active_key(purpose)

    def set_state(self, key_id: str, state: KeyState, *, purpose: str) -> None:
        """Change one key's lifecycle state.

        Revoked keys cannot be reactivated and their material is dropped.
        Activating a key demotes the previous active key.

        Args:
            key_id: Identifier of a registered key.
            state: The new state.
            purpose: Purpose the key belongs to.

        Raises:
            KeyUnavailableError: If the key is unknown for the purpose.
            KeyRevokedError: If a revoked key would be made usable again.
        """

        if not isinstance(state, KeyState):
            raise TypeError("state must be a KeyState")
        with self._lock:
            held = self._keys.get(key_id)
            if held is None or held.purpose != purpose:
                raise KeyUnavailableError()
            if held.state is KeyState.REVOKED and state is not KeyState.REVOKED:
                raise KeyRevokedError()
            if state is KeyState.ACTIVE and self._active.get(purpose) != key_id:
                self._demote_active(purpose)
                self._active[purpose] = key_id
            elif state is not KeyState.ACTIVE and self._active.get(purpose) == key_id:
                del self._active[purpose]
            held.state = state
            if state is KeyState.REVOKED:
                held.material = None

    def active_key(self, purpose: str) -> DataKey:
        """Return the active key for ``purpose``.

        Raises:
            KeyUnavailableError: If no key is active for the purpose.
        """

        with self._lock:
            key_id = self._active.get(purpose)
            if key_id is None:
                raise KeyUnavailableError()
            return self._handle(key_id, purpose)

    def resolve_key(self, key_id: str, *, purpose: str) -> DataKey:
        """Return a registered key for decryption.

        Raises:
            KeyUnavailableError: If the key is unknown for the purpose.
            KeyRetiredError: If the key was retired.
            KeyRevokedError: If the key was revoked.
        """

        with self._lock:
            return self._handle(key_id, purpose)

    def key_states(self, purpose: str) -> dict[str, KeyState]:
        """Return a value-free inventory of key identifiers and states."""

        with self._lock:
            return {
                key_id: held.state
                for key_id, held in sorted(self._keys.items())
                if held.purpose == purpose
            }

    def __repr__(self) -> str:
        with self._lock:
            return f"{type(self).__name__}(keys={len(self._keys)})"

    def __reduce__(self) -> NoReturn:
        raise TypeError("key material cannot be serialized")

    def _handle(self, key_id: str, purpose: str) -> DataKey:
        held = self._keys.get(key_id) if isinstance(key_id, str) else None
        if held is None or held.purpose != purpose:
            raise KeyUnavailableError()
        _raise_for_terminal_state(held.state)
        if held.material is None:
            raise KeyUnavailableError()
        return DataKey(
            key_id=key_id,
            purpose=purpose,
            state=held.state,
            material=held.material,
        )

    def _demote_active(self, purpose: str) -> None:
        previous = self._active.pop(purpose, None)
        if previous is not None:
            self._keys[previous].state = KeyState.DECRYPT_ONLY
