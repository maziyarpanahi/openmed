"""Versioned AEAD envelopes bound to a key identifier and record identity.

An envelope is a self-describing binary record::

    magic "OMAE" | version (1 byte) | algorithm (1 byte)
    | key-id length (1 byte) | key id (ASCII) | nonce (12 bytes)
    | ciphertext and 16-byte authentication tag

The header (everything before the ciphertext) and a caller-supplied
*context* are authenticated as associated data.  The context names the record
the ciphertext belongs to, for example a table and primary key, so a
ciphertext moved to another record, another purpose, or another key
identifier fails authentication instead of decrypting into the wrong place.
The context is never stored in the envelope; the reader re-derives it.

Version 1 uses AES-256-GCM with a random 96-bit nonce.  The cipher key is a
purpose-scoped HMAC-SHA256 derivation of the provider's data key, following
the derived-key pattern in :mod:`openmed.core.surrogate_vault_crypto`, so one
caller key used for two purposes never encrypts both under the same cipher
key.  Rotate keys well before 2**32 envelopes are sealed under one key.

The ``cryptography`` package (``openmed[integrity]``) is imported only when an
envelope is sealed or opened.  Every failure is a value-free
:class:`~openmed.core.key_provider.EncryptionError`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .key_provider import (
    DataKey,
    EncryptionError,
    KeyProvider,
    KeyProviderError,
    KeyUnavailableError,
    register_error_messages,
    require_decryption_key,
    require_encryption_key,
    validate_key_provider,
)

__all__ = [
    "ENVELOPE_ALGORITHM",
    "ENVELOPE_MAGIC",
    "ENVELOPE_TEXT_PREFIX",
    "ENVELOPE_VERSION",
    "EncryptionDependencyError",
    "EnvelopeAuthenticationError",
    "EnvelopeCipher",
    "EnvelopeError",
    "EnvelopeFormatError",
    "EnvelopeHeader",
    "EnvelopeVersionError",
    "envelope_from_text",
    "envelope_to_text",
    "is_envelope",
    "is_text_envelope",
    "read_envelope_header",
]

ENVELOPE_MAGIC = b"OMAE"
ENVELOPE_VERSION = 1
ENVELOPE_ALGORITHM = "aes-256-gcm"
ENVELOPE_TEXT_PREFIX = "omae:"

_ALGORITHM_IDS = {1: ENVELOPE_ALGORITHM}
_ALGORITHM_CODES = {name: code for code, name in _ALGORITHM_IDS.items()}
_NONCE_BYTES = 12
_TAG_BYTES = 16
_MIN_KEY_ID_BYTES = 3
_MAX_KEY_ID_BYTES = 128
_FIXED_HEADER_BYTES = len(ENVELOPE_MAGIC) + 3
_SUBKEY_CONTEXT = b"openmed-aead-envelope:v1:aes-256-gcm\x00"
_KEY_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9._:/-]{2,127}\Z")
_PURPOSE_RE = re.compile(r"[a-z][a-z0-9._-]{1,63}\Z")
_CONTEXT_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_MAX_CONTEXT_TEXT = 512
_MAX_CONTEXT_ITEMS = 16

register_error_messages(
    {
        "envelope_invalid": "encrypted envelope is invalid",
        "envelope_malformed": "encrypted envelope is malformed",
        "envelope_version_unsupported": "encrypted envelope version is unsupported",
        "envelope_authentication_failed": "encrypted envelope failed authentication",
        "encryption_dependency_unavailable": (
            "envelope encryption requires the optional 'integrity' extra"
        ),
    }
)


class EnvelopeError(EncryptionError):
    """Base class for envelope format and authentication failures."""

    default_code = "envelope_invalid"


class EnvelopeFormatError(EnvelopeError):
    """Raised when bytes are not a well-formed envelope."""

    default_code = "envelope_malformed"


class EnvelopeVersionError(EnvelopeError):
    """Raised for an unknown envelope version or algorithm."""

    default_code = "envelope_version_unsupported"


class EnvelopeAuthenticationError(EnvelopeError):
    """Raised for a wrong key, tampered bytes, or a mismatched context."""

    default_code = "envelope_authentication_failed"


class EncryptionDependencyError(EnvelopeError):
    """Raised when the optional ``cryptography`` dependency is missing."""

    default_code = "encryption_dependency_unavailable"


@dataclass(frozen=True, slots=True)
class EnvelopeHeader:
    """Cleartext, non-secret envelope metadata.

    Attributes:
        version: Envelope format version.
        algorithm: AEAD algorithm name.
        key_id: Identifier of the key that sealed the envelope.
    """

    version: int
    algorithm: str
    key_id: str


def is_envelope(data: object) -> bool:
    """Return whether ``data`` starts with the envelope magic and version."""

    return isinstance(data, (bytes, bytearray, memoryview)) and bytes(
        data[: len(ENVELOPE_MAGIC) + 1]
    ) == ENVELOPE_MAGIC + bytes((ENVELOPE_VERSION,))


def is_text_envelope(text: object) -> bool:
    """Return whether ``text`` uses the text envelope encoding."""

    return isinstance(text, str) and text.startswith(ENVELOPE_TEXT_PREFIX)


def envelope_to_text(envelope: bytes) -> str:
    """Encode an envelope as ASCII text for text-typed storage columns."""

    return ENVELOPE_TEXT_PREFIX + base64.b64encode(envelope).decode("ascii")


def envelope_from_text(text: str) -> bytes:
    """Decode a text envelope produced by :func:`envelope_to_text`.

    Raises:
        EnvelopeFormatError: If the text is not an encoded envelope.
    """

    if not is_text_envelope(text):
        raise EnvelopeFormatError()
    try:
        return base64.b64decode(
            text[len(ENVELOPE_TEXT_PREFIX) :].encode("ascii"),
            validate=True,
        )
    except (UnicodeError, binascii.Error, ValueError):
        raise EnvelopeFormatError() from None


def read_envelope_header(data: bytes | bytearray | memoryview) -> EnvelopeHeader:
    """Parse the cleartext header without touching any key.

    Args:
        data: Binary envelope bytes.

    Returns:
        The envelope's version, algorithm, and key identifier.

    Raises:
        EnvelopeFormatError: If the bytes are not a well-formed envelope.
        EnvelopeVersionError: If the version or algorithm is unsupported.
    """

    header, _header_bytes, _nonce, _ciphertext = _split(data)
    return header


class EnvelopeCipher:
    """Seal and open envelopes for one purpose with a caller's key provider.

    Args:
        key_provider: Caller-supplied :class:`KeyProvider`.
        purpose: Controlled purpose label, for example ``"journey-store"``.

    The cipher never caches key material between operations; each call asks
    the provider for a fresh :class:`~openmed.core.key_provider.DataKey`.
    """

    def __init__(self, key_provider: KeyProvider, *, purpose: str) -> None:
        self._provider = validate_key_provider(key_provider)
        if not isinstance(purpose, str) or _PURPOSE_RE.fullmatch(purpose) is None:
            raise ValueError("purpose must be a controlled identifier")
        self._purpose = purpose

    @property
    def purpose(self) -> str:
        """Return the purpose label bound into every envelope."""

        return self._purpose

    def active_key_id(self) -> str:
        """Return the identifier of the key that would seal new envelopes.

        Raises:
            KeyProviderError: If no usable active key is available.
        """

        return self.active_key().key_id

    def active_key(self) -> DataKey:
        """Return the provider's validated active key for this purpose.

        Raises:
            KeyProviderError: If no usable active key is available.
        """

        try:
            candidate = self._provider.active_key(self._purpose)
        except KeyProviderError:
            raise
        except Exception:
            raise KeyUnavailableError() from None
        return require_encryption_key(candidate, purpose=self._purpose)

    def seal(
        self,
        plaintext: bytes,
        *,
        context: Mapping[str, Any],
        key: DataKey | None = None,
    ) -> bytes:
        """Encrypt ``plaintext`` under the active key and bind ``context``.

        Args:
            plaintext: Bytes to protect. They are never logged or persisted.
            context: Record identity bound as associated data. Keys are
                controlled identifiers; values are short text, integers, or
                sequences of those.
            key: Optional active key already obtained from
                :meth:`active_key`; it is revalidated before use.

        Returns:
            A binary envelope.

        Raises:
            KeyProviderError: If no usable active key is available.
            EncryptionDependencyError: If ``cryptography`` is not installed.
        """

        if not isinstance(plaintext, (bytes, bytearray, memoryview)):
            raise TypeError("plaintext must be bytes")
        context_bytes = _canonical_context(self._purpose, context)
        key = (
            self.active_key()
            if key is None
            else require_encryption_key(key, purpose=self._purpose)
        )
        key_id = key.key_id.encode("ascii")
        nonce = os.urandom(_NONCE_BYTES)
        header = (
            ENVELOPE_MAGIC
            + bytes(
                (
                    ENVELOPE_VERSION,
                    _ALGORITHM_CODES[ENVELOPE_ALGORITHM],
                    len(key_id),
                )
            )
            + key_id
            + nonce
        )
        cipher = _aesgcm(_derive_subkey(key.material, self._purpose))
        try:
            ciphertext = cipher.encrypt(nonce, bytes(plaintext), header + context_bytes)
        except Exception:
            raise EnvelopeError() from None
        return header + ciphertext

    def open(
        self,
        envelope: bytes | bytearray | memoryview,
        *,
        context: Mapping[str, Any],
    ) -> bytes:
        """Authenticate and decrypt an envelope sealed for ``context``.

        Args:
            envelope: Binary envelope bytes.
            context: The record identity the envelope must be bound to.

        Returns:
            The decrypted plaintext.

        Raises:
            EnvelopeFormatError: If the bytes are malformed.
            EnvelopeVersionError: If the version or algorithm is unsupported.
            KeyProviderError: If the key is unavailable, retired, or revoked.
            EnvelopeAuthenticationError: For a wrong key, tampering, or a
                context mismatch.
        """

        context_bytes = _canonical_context(self._purpose, context)
        header, header_bytes, nonce, ciphertext = _split(envelope)
        try:
            candidate = self._provider.resolve_key(header.key_id, purpose=self._purpose)
        except KeyProviderError:
            raise
        except Exception:
            raise KeyUnavailableError() from None
        key = require_decryption_key(
            candidate,
            purpose=self._purpose,
            key_id=header.key_id,
        )
        cipher = _aesgcm(_derive_subkey(key.material, self._purpose))
        try:
            from cryptography.exceptions import InvalidTag
        except ImportError:
            raise EncryptionDependencyError() from None
        try:
            return cipher.decrypt(nonce, ciphertext, header_bytes + context_bytes)
        except InvalidTag:
            raise EnvelopeAuthenticationError() from None
        except Exception:
            raise EnvelopeAuthenticationError() from None

    def seal_text(
        self,
        plaintext: bytes,
        *,
        context: Mapping[str, Any],
        key: DataKey | None = None,
    ) -> str:
        """Seal ``plaintext`` and return the ASCII text encoding."""

        return envelope_to_text(self.seal(plaintext, context=context, key=key))

    def open_text(self, text: str, *, context: Mapping[str, Any]) -> bytes:
        """Open an ASCII text envelope produced by :meth:`seal_text`."""

        return self.open(envelope_from_text(text), context=context)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(purpose={self._purpose!r})"


def _split(
    data: bytes | bytearray | memoryview,
) -> tuple[EnvelopeHeader, bytes, bytes, bytes]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise EnvelopeFormatError()
    raw = bytes(data)
    if len(raw) < _FIXED_HEADER_BYTES or raw[: len(ENVELOPE_MAGIC)] != ENVELOPE_MAGIC:
        raise EnvelopeFormatError()
    version, algorithm_code, key_id_length = raw[
        len(ENVELOPE_MAGIC) : _FIXED_HEADER_BYTES
    ]
    if version != ENVELOPE_VERSION or algorithm_code not in _ALGORITHM_IDS:
        raise EnvelopeVersionError()
    if not _MIN_KEY_ID_BYTES <= key_id_length <= _MAX_KEY_ID_BYTES:
        raise EnvelopeFormatError()
    key_id_end = _FIXED_HEADER_BYTES + key_id_length
    nonce_end = key_id_end + _NONCE_BYTES
    if len(raw) < nonce_end + _TAG_BYTES:
        raise EnvelopeFormatError()
    try:
        key_id = raw[_FIXED_HEADER_BYTES:key_id_end].decode("ascii")
    except UnicodeDecodeError:
        raise EnvelopeFormatError() from None
    if _KEY_ID_RE.fullmatch(key_id) is None:
        raise EnvelopeFormatError()
    header = EnvelopeHeader(
        version=version,
        algorithm=_ALGORITHM_IDS[algorithm_code],
        key_id=key_id,
    )
    return header, raw[:nonce_end], raw[key_id_end:nonce_end], raw[nonce_end:]


def _canonical_context(purpose: str, context: Mapping[str, Any]) -> bytes:
    if not isinstance(context, Mapping) or len(context) > _MAX_CONTEXT_ITEMS:
        raise ValueError("envelope context must be a small mapping")
    clean: dict[str, Any] = {}
    for key, value in context.items():
        if not isinstance(key, str) or _CONTEXT_KEY_RE.fullmatch(key) is None:
            raise ValueError("envelope context keys must be controlled identifiers")
        clean[key] = _context_value(value)
    document = {"context": clean, "purpose": purpose, "version": ENVELOPE_VERSION}
    return b"\x00" + json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _context_value(value: Any, *, nested: bool = False) -> Any:
    if isinstance(value, bool):
        raise ValueError("envelope context values cannot be booleans")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_CONTEXT_TEXT:
            raise ValueError("envelope context text is too long")
        return value
    if (
        not nested
        and isinstance(value, Sequence)
        and not isinstance(value, (bytes, bytearray))
        and len(value) <= _MAX_CONTEXT_ITEMS
    ):
        return [_context_value(item, nested=True) for item in value]
    raise ValueError("envelope context values must be text, integers, or lists")


def _derive_subkey(material: bytes, purpose: str) -> bytes:
    return hmac.new(
        material,
        _SUBKEY_CONTEXT + purpose.encode("ascii"),
        hashlib.sha256,
    ).digest()


def _aesgcm(key: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise EncryptionDependencyError() from None
    return AESGCM(key)
