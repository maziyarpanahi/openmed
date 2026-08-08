"""Deterministic authenticated encryption for reversible surrogate mappings.

This module is a small storage boundary for mappings produced by reversible
redaction. It deliberately accepts caller-owned key material and keeps the
mapping plaintext in memory only while it is being serialized. The encrypted
envelope is deterministic because it uses AES-SIV, which also authenticates the
canonical serialized mapping.

The ``cryptography`` package is imported only when an operation is requested.
This keeps importing the core package local and makes the optional dependency
failure explicit at the point where encryption is used.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

SCHEMA_VERSION = 1
ENCRYPTION_SCHEME = "AES-256-SIV"
KEY_BYTES = 32

_KEY_DERIVATION_CONTEXT = b"openmed-surrogate-vault-crypto:aes-siv-key:v1"
_ENVELOPE_FIELDS = frozenset({"schema_version", "encryption_scheme", "ciphertext"})
_MIN_CIPHERTEXT_BYTES = 16  # AES-SIV's authentication tag.

KeyMaterial: TypeAlias = bytes | bytearray | memoryview
SerializedMapping: TypeAlias = bytes | bytearray | memoryview | str | Mapping[str, Any]


class SurrogateVaultCryptoError(ValueError):
    """Base class for fail-closed encrypted-mapping errors."""


class SurrogateVaultKeyError(SurrogateVaultCryptoError):
    """Raised when caller-supplied key material is missing or malformed."""


class SurrogateVaultPayloadError(SurrogateVaultCryptoError):
    """Raised when an encrypted mapping cannot be authenticated or decoded."""


class SurrogateVaultCrypto:
    """Encrypt and persist reversible mappings with caller-supplied key material.

    Args:
        key: Exactly 32 bytes of caller-managed key material. The key is copied
            into memory and is never included in an encrypted envelope.

    The AES-SIV construction is deterministic for the same key and mapping,
    while still authenticating both the ciphertext and its envelope metadata.
    Determinism intentionally leaks equality of complete encrypted mappings;
    it does not expose individual mapping keys or values.
    """

    def __init__(self, key: KeyMaterial) -> None:
        self._key = _validate_key(key)

    @property
    def encryption_scheme(self) -> str:
        """Return the versioned authenticated-encryption scheme name."""

        return ENCRYPTION_SCHEME

    def encrypt(self, mapping: Mapping[str, str]) -> bytes:
        """Return a deterministic JSON envelope containing encrypted mapping data.

        Plaintext mapping data is canonicalized in memory and is not written to
        a temporary file, logged, or included in an exception.
        """

        plaintext = _serialize_mapping(mapping)
        aessiv = _new_aessiv(self._key)
        ciphertext = aessiv.encrypt(plaintext, [_associated_data()])
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "encryption_scheme": ENCRYPTION_SCHEME,
            "ciphertext": _b64encode(ciphertext),
        }
        return _canonical_json(envelope)

    def decrypt(self, serialized: SerializedMapping) -> dict[str, str]:
        """Authenticate and return a previously encrypted surrogate mapping."""

        envelope = _parse_envelope(serialized)
        ciphertext = _decode_ciphertext(envelope["ciphertext"])
        aessiv = _new_aessiv(self._key)
        plaintext = _decrypt_ciphertext(aessiv, ciphertext)
        return _deserialize_mapping(plaintext)

    def write(self, path: str | Path, mapping: Mapping[str, str]) -> None:
        """Atomically write an encrypted mapping to ``path``.

        Only the encrypted envelope reaches the temporary file. The temporary
        file is created with owner-only permissions and removed on every exit
        path.
        """

        encrypted = self.encrypt(mapping)
        _atomic_write(Path(path), encrypted)

    def read(self, path: str | Path) -> dict[str, str]:
        """Read, authenticate, and decrypt an encrypted mapping from ``path``."""

        try:
            serialized = Path(path).read_bytes()
        except OSError:
            raise SurrogateVaultPayloadError(
                "could not read encrypted surrogate mapping"
            ) from None
        return self.decrypt(serialized)

    def save(self, path: str | Path, mapping: Mapping[str, str]) -> None:
        """Alias for :meth:`write` for file-backed vault call sites."""

        self.write(path, mapping)

    def load(self, path: str | Path) -> dict[str, str]:
        """Alias for :meth:`read` for file-backed vault call sites."""

        return self.read(path)

    def __repr__(self) -> str:
        """Return metadata only; never expose key material or mapping data."""

        return f"{self.__class__.__name__}(encryption_scheme={ENCRYPTION_SCHEME!r})"


# The descriptive alias helps callers discover the file-backed use case while
# preserving one implementation and one key-validation path.
EncryptedSurrogateVault = SurrogateVaultCrypto


def encrypt_mapping(mapping: Mapping[str, str], key: KeyMaterial) -> bytes:
    """Encrypt a mapping with caller-supplied key material."""

    return SurrogateVaultCrypto(key).encrypt(mapping)


def decrypt_mapping(serialized: SerializedMapping, key: KeyMaterial) -> dict[str, str]:
    """Decrypt and authenticate a serialized mapping with the supplied key."""

    return SurrogateVaultCrypto(key).decrypt(serialized)


def save_mapping(
    path: str | Path,
    mapping: Mapping[str, str],
    key: KeyMaterial,
) -> None:
    """Encrypt a mapping and atomically save its envelope to ``path``."""

    SurrogateVaultCrypto(key).write(path, mapping)


def load_mapping(path: str | Path, key: KeyMaterial) -> dict[str, str]:
    """Load and decrypt an encrypted mapping from ``path``."""

    return SurrogateVaultCrypto(key).read(path)


def _validate_key(key: KeyMaterial) -> bytes:
    if key is None:
        raise SurrogateVaultKeyError("surrogate vault key is required")
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise SurrogateVaultKeyError(
            "surrogate vault key must be bytes-like and exactly 32 bytes"
        )
    key_bytes = bytes(key)
    if len(key_bytes) != KEY_BYTES:
        raise SurrogateVaultKeyError("surrogate vault key must be exactly 32 bytes")
    return key_bytes


def _new_aessiv(key: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESSIV
    except ImportError:
        raise SurrogateVaultCryptoError(
            "AES-SIV requires the optional 'integrity' extra"
        ) from None

    derived_key = hmac.new(
        key,
        _KEY_DERIVATION_CONTEXT,
        hashlib.sha512,
    ).digest()
    return AESSIV(derived_key)


def _associated_data() -> bytes:
    return _canonical_json(
        {
            "encryption_scheme": ENCRYPTION_SCHEME,
            "schema_version": SCHEMA_VERSION,
        }
    )


def _decrypt_ciphertext(aessiv: Any, ciphertext: bytes) -> bytes:
    try:
        from cryptography.exceptions import InvalidTag
    except ImportError:
        raise SurrogateVaultCryptoError(
            "AES-SIV requires the optional 'integrity' extra"
        ) from None
    try:
        return aessiv.decrypt(ciphertext, [_associated_data()])
    except (InvalidTag, TypeError, ValueError):
        raise SurrogateVaultPayloadError(
            "surrogate mapping authentication failed"
        ) from None


def _serialize_mapping(mapping: Mapping[str, str]) -> bytes:
    clean_mapping = _validated_mapping(mapping)
    return _canonical_json(clean_mapping)


def _deserialize_mapping(plaintext: bytes) -> dict[str, str]:
    try:
        decoded = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise SurrogateVaultPayloadError(
            "authenticated surrogate mapping has an invalid payload"
        ) from None
    return _validated_mapping(decoded)


def _validated_mapping(mapping: Any) -> dict[str, str]:
    if not isinstance(mapping, Mapping):
        raise SurrogateVaultCryptoError("surrogate mapping must be an object")
    clean_mapping: dict[str, str] = {}
    for key, value in mapping.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or not value
        ):
            raise SurrogateVaultCryptoError(
                "surrogate mapping keys and values must be non-empty strings"
            )
        clean_mapping[key] = value
    return clean_mapping


def _parse_envelope(serialized: SerializedMapping) -> dict[str, Any]:
    if isinstance(serialized, Mapping):
        envelope: Any = serialized
    else:
        if isinstance(serialized, str):
            raw = serialized.encode("utf-8")
        elif isinstance(serialized, (bytes, bytearray, memoryview)):
            raw = bytes(serialized)
        else:
            raise SurrogateVaultPayloadError(
                "encrypted surrogate mapping must be serialized JSON"
            )
        try:
            envelope = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            raise SurrogateVaultPayloadError(
                "encrypted surrogate mapping envelope is malformed"
            ) from None

    if not isinstance(envelope, Mapping) or set(envelope) != _ENVELOPE_FIELDS:
        raise SurrogateVaultPayloadError(
            "encrypted surrogate mapping envelope is malformed"
        )
    if (
        isinstance(envelope["schema_version"], bool)
        or envelope["schema_version"] != SCHEMA_VERSION
        or envelope["encryption_scheme"] != ENCRYPTION_SCHEME
    ):
        raise SurrogateVaultPayloadError(
            "unsupported encrypted surrogate mapping envelope"
        )
    if not isinstance(envelope["ciphertext"], str):
        raise SurrogateVaultPayloadError(
            "encrypted surrogate mapping ciphertext is malformed"
        )
    return dict(envelope)


def _decode_ciphertext(encoded: str) -> bytes:
    try:
        ciphertext = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, ValueError, TypeError):
        raise SurrogateVaultPayloadError(
            "encrypted surrogate mapping ciphertext is malformed"
        ) from None
    if len(ciphertext) < _MIN_CIPHERTEXT_BYTES:
        raise SurrogateVaultPayloadError(
            "encrypted surrogate mapping ciphertext is malformed"
        )
    return ciphertext


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(handle.name, 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError:
        raise SurrogateVaultCryptoError(
            "could not write encrypted surrogate mapping"
        ) from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


__all__ = [
    "ENCRYPTION_SCHEME",
    "EncryptedSurrogateVault",
    "KEY_BYTES",
    "SCHEMA_VERSION",
    "SurrogateVaultCrypto",
    "SurrogateVaultCryptoError",
    "SurrogateVaultKeyError",
    "SurrogateVaultPayloadError",
    "decrypt_mapping",
    "encrypt_mapping",
    "load_mapping",
    "save_mapping",
]
