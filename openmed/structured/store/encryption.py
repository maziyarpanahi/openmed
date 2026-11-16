"""Application-layer encryption configuration for local Journey storage.

Local Journey stores seal every payload-bearing value before it reaches the
filesystem: artifact bytes in :class:`~openmed.structured.store.LocalArtifactStore`
and the ``payload_json`` column of each Journey record table in
:class:`~openmed.structured.store.SQLiteJourneyStore`.  Sealing uses the
versioned AEAD envelope in :mod:`openmed.core.aead_envelope` with a
caller-supplied :class:`~openmed.core.key_provider.KeyProvider`.  The envelope
binds the record identity (table and primary key, or artifact content hash)
as associated data, so ciphertext cannot be swapped between records.

Plaintext persistence is never selected implicitly.  A store must receive a
:class:`JourneyStoreEncryption`; plaintext requires
:meth:`JourneyStoreEncryption.development_plaintext` and is refused whenever
the production profile is in effect (``OPENMED_PROFILE=prod``).

Opaque identifiers, revisions, versions, content and payload digests, and the
persisted encryption state stay in cleartext so the stores can enforce
uniqueness, history, and integrity without decrypting.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from openmed.clinical.journey_contracts import sha256_digest
from openmed.core.aead_envelope import (
    ENVELOPE_VERSION,
    EnvelopeAuthenticationError,
    EnvelopeCipher,
    EnvelopeFormatError,
    envelope_from_text,
    is_envelope,
    is_text_envelope,
    read_envelope_header,
)
from openmed.core.key_provider import (
    DataKey,
    EncryptionError,
    KeyProvider,
    KeyUnavailableError,
    register_error_messages,
    validate_key_provider,
)

__all__ = [
    "ENCRYPTION_STATE_ENCRYPTED",
    "ENCRYPTION_STATE_MIGRATING",
    "ENCRYPTION_STATE_PLAINTEXT",
    "JOURNEY_KEY_PURPOSE",
    "PROFILE_ENV_VAR",
    "JourneyEncryptionError",
    "JourneyStoreEncryption",
    "LocalJourneyReencryptionReport",
    "PersistedEncryptionState",
    "ReencryptionReport",
    "require_journey_encryption",
    "resolve_journey_profile",
]

JOURNEY_KEY_PURPOSE = "journey-store"
PROFILE_ENV_VAR = "OPENMED_PROFILE"
PRODUCTION_PROFILES = frozenset({"prod", "production"})

ENCRYPTION_STATE_PLAINTEXT = "plaintext"
ENCRYPTION_STATE_MIGRATING = "migrating"
ENCRYPTION_STATE_ENCRYPTED = "encrypted"
_KEY_CHECK_PLAINTEXT = b"openmed-journey-key-check:v1"
_KEY_CHECK_CONTEXT = {"record": "journey_key_check"}
_STATES = frozenset(
    {
        ENCRYPTION_STATE_PLAINTEXT,
        ENCRYPTION_STATE_MIGRATING,
        ENCRYPTION_STATE_ENCRYPTED,
    }
)

register_error_messages(
    {
        "journey_encryption_failed": "Journey storage encryption failed",
        "encryption_required": (
            "local Journey storage requires an explicit encryption configuration"
        ),
        "plaintext_refused_in_production": (
            "plaintext Journey storage is refused under the production profile"
        ),
        "store_migration_required": (
            "plaintext Journey store must be migrated before encrypted use"
        ),
        "store_encrypted": "encrypted Journey store cannot be opened as plaintext",
        "store_not_encrypted": "Journey store is not configured for encryption",
        "key_purpose_mismatch": "Journey store was encrypted for another purpose",
        "payload_not_encrypted": "stored Journey payload is not encrypted",
        "payload_encrypted_in_plaintext_store": (
            "plaintext Journey store contains an encrypted payload"
        ),
        "encryption_state_invalid": "persisted Journey encryption state is invalid",
        "key_check_failed": "encryption key does not match its persisted verifier",
    }
)


class JourneyEncryptionError(EncryptionError):
    """Value-free configuration, state, or payload encryption failure."""

    default_code = "journey_encryption_failed"


@dataclass(frozen=True)
class JourneyStoreEncryption:
    """Encryption configuration accepted by local Journey stores.

    Build instances with :meth:`encrypted` or :meth:`development_plaintext`
    rather than the constructor.

    Attributes:
        key_provider: Caller-supplied key provider for encrypted stores.
        key_purpose: Purpose label passed to the provider and bound into
            every envelope.
        allow_plaintext: ``True`` only for the explicit development opt-in.
        profile: Optional profile asserted by the caller. A production value
            from either this field or ``OPENMED_PROFILE`` makes plaintext
            unavailable; a development value never overrides the environment.
        migrate_plaintext: When ``True``, opening a plaintext store marks it as
            migrating to encryption instead of refusing it. Prefer the stores'
            ``migrate_to_encrypted`` helpers, which set this flag.
    """

    key_provider: KeyProvider | None = field(default=None, repr=False)
    key_purpose: str = JOURNEY_KEY_PURPOSE
    allow_plaintext: bool = False
    profile: str | None = None
    migrate_plaintext: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_plaintext, bool) or not isinstance(
            self.migrate_plaintext, bool
        ):
            raise TypeError("encryption flags must be booleans")
        if self.profile is not None and not isinstance(self.profile, str):
            raise TypeError("profile must be text when supplied")
        if self.allow_plaintext:
            if self.key_provider is not None or self.migrate_plaintext:
                raise ValueError("plaintext development mode cannot use a key provider")
            return
        if self.key_provider is None:
            raise ValueError("encrypted Journey storage requires a key provider")
        validate_key_provider(self.key_provider)
        # Validates the purpose grammar once, before any store is touched.
        EnvelopeCipher(self.key_provider, purpose=self.key_purpose)

    @classmethod
    def encrypted(
        cls,
        key_provider: KeyProvider,
        *,
        key_purpose: str = JOURNEY_KEY_PURPOSE,
        profile: str | None = None,
    ) -> "JourneyStoreEncryption":
        """Return a configuration that seals payloads with ``key_provider``.

        Args:
            key_provider: Caller-supplied key provider.
            key_purpose: Purpose label for the provider's keys.
            profile: Optional caller-asserted profile.

        Returns:
            An encrypted-store configuration.
        """

        return cls(key_provider=key_provider, key_purpose=key_purpose, profile=profile)

    @classmethod
    def development_plaintext(
        cls,
        *,
        profile: str | None = None,
    ) -> "JourneyStoreEncryption":
        """Return the explicit plaintext opt-in for development and tests.

        Stores opened with this configuration persist readable payloads and
        are refused under the production profile.

        Args:
            profile: Optional caller-asserted profile.

        Returns:
            A plaintext development configuration.
        """

        return cls(allow_plaintext=True, profile=profile)

    @property
    def is_encrypted(self) -> bool:
        """Return whether payloads are sealed before persistence."""

        return not self.allow_plaintext

    def cipher(self) -> EnvelopeCipher:
        """Return the envelope cipher for this configuration.

        Raises:
            JourneyEncryptionError: If the configuration is plaintext.
        """

        if self.key_provider is None:
            raise JourneyEncryptionError("store_not_encrypted")
        return EnvelopeCipher(self.key_provider, purpose=self.key_purpose)

    def __repr__(self) -> str:
        mode = "aead" if self.is_encrypted else "plaintext"
        return f"{type(self).__name__}(mode={mode!r}, key_purpose={self.key_purpose!r})"


def resolve_journey_profile(profile: str | None = None) -> str:
    """Return ``"prod"`` or ``"dev"`` for local Journey storage decisions.

    A production value in either the explicit argument or ``OPENMED_PROFILE``
    selects ``"prod"``; the environment cannot be weakened from code.

    Args:
        profile: Optional caller-asserted profile.

    Returns:
        ``"prod"`` or ``"dev"``.
    """

    for candidate in (profile, os.environ.get(PROFILE_ENV_VAR)):
        if (
            isinstance(candidate, str)
            and candidate.strip().lower() in PRODUCTION_PROFILES
        ):
            return "prod"
    return "dev"


def require_journey_encryption(
    encryption: JourneyStoreEncryption | None,
) -> JourneyStoreEncryption:
    """Validate a store's encryption configuration and fail closed.

    Args:
        encryption: Configuration passed to a local Journey store.

    Returns:
        The validated configuration.

    Raises:
        JourneyEncryptionError: ``encryption_required`` when no configuration
            was supplied, or ``plaintext_refused_in_production`` when the
            plaintext opt-in is used under the production profile.
        TypeError: If ``encryption`` is not a :class:`JourneyStoreEncryption`.
    """

    if encryption is None:
        raise JourneyEncryptionError("encryption_required")
    if not isinstance(encryption, JourneyStoreEncryption):
        raise TypeError("encryption must be a JourneyStoreEncryption")
    if encryption.allow_plaintext:
        if resolve_journey_profile(encryption.profile) == "prod":
            raise JourneyEncryptionError("plaintext_refused_in_production")
    return encryption


@dataclass(frozen=True, slots=True)
class PersistedEncryptionState:
    """Value-free encryption metadata persisted beside a store.

    Attributes:
        state: ``plaintext``, ``migrating``, or ``encrypted``.
        key_purpose: Key purpose for encrypted states.
        envelope_version: Envelope version written by the store.
    """

    state: str
    key_purpose: str | None = None
    envelope_version: int | None = None

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise JourneyEncryptionError("encryption_state_invalid")
        if self.state == ENCRYPTION_STATE_PLAINTEXT:
            if self.key_purpose is not None or self.envelope_version is not None:
                raise JourneyEncryptionError("encryption_state_invalid")
            return
        if (
            not isinstance(self.key_purpose, str)
            or isinstance(self.envelope_version, bool)
            or not isinstance(self.envelope_version, int)
            or self.envelope_version < 1
        ):
            raise JourneyEncryptionError("encryption_state_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, key-free metadata."""

        return {
            "envelope_version": self.envelope_version,
            "key_purpose": self.key_purpose,
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class ReencryptionReport:
    """Value-free counts from one re-encryption or migration pass.

    Attributes:
        examined: Persisted values inspected.
        reencrypted: Values re-sealed from an older key to the active key.
        migrated: Plaintext values sealed for the first time.
        complete: Whether every value is now sealed under the active key.
        state: Persisted encryption state after the pass.
    """

    examined: int
    reencrypted: int
    migrated: int
    complete: bool
    state: str


@dataclass(frozen=True, slots=True)
class LocalJourneyReencryptionReport:
    """Re-encryption reports for both halves of a local Journey store."""

    artifacts: ReencryptionReport
    metadata: ReencryptionReport

    @property
    def complete(self) -> bool:
        """Return whether artifacts and metadata are both complete."""

        return self.artifacts.complete and self.metadata.complete


class JourneyPayloadCodec(Protocol):
    """Internal codec used by local stores for payload persistence."""

    @property
    def state(self) -> str:
        """Return the encryption state the codec enforces."""

    def seal_payload(
        self, table: str, identity: Sequence[str | int], payload_json: str
    ) -> str:
        """Return the stored form of one payload column value."""

    def open_payload(
        self, table: str, identity: Sequence[str | int], stored: Any
    ) -> str:
        """Return the canonical JSON for one stored payload column value."""

    def seal_blob(self, content_hash: str, content: bytes) -> bytes:
        """Return the stored form of artifact bytes."""

    def open_blob(self, content_hash: str, stored: bytes) -> bytes:
        """Return artifact bytes from their stored form."""


class PlaintextJourneyCodec:
    """Explicit development codec that stores payloads unchanged."""

    state = ENCRYPTION_STATE_PLAINTEXT

    def seal_payload(
        self, table: str, identity: Sequence[str | int], payload_json: str
    ) -> str:
        """Return ``payload_json`` unchanged."""

        return payload_json

    def open_payload(
        self, table: str, identity: Sequence[str | int], stored: Any
    ) -> str:
        """Return a plaintext payload, refusing envelopes."""

        if is_text_envelope(stored):
            raise JourneyEncryptionError("payload_encrypted_in_plaintext_store")
        return str(stored)

    def seal_blob(self, content_hash: str, content: bytes) -> bytes:
        """Return ``content`` unchanged."""

        return content

    def open_blob(self, content_hash: str, stored: bytes) -> bytes:
        """Return stored bytes; callers verify the content digest."""

        return stored


PLAINTEXT_JOURNEY_CODEC = PlaintextJourneyCodec()


class KeyVerifierStore(Protocol):
    """Persistence for per-key verifiers, which are envelopes of a constant."""

    def load_key_verifier(self, key_id: str) -> str | None:
        """Return the persisted verifier for ``key_id``, if any."""

    def save_key_verifier(self, key_id: str, verifier: str) -> None:
        """Persist a verifier for ``key_id`` unless one already exists."""


class EncryptedJourneyCodec:
    """Codec that seals payloads with record-bound AEAD envelopes.

    Before the first seal with a key identifier, the codec checks the key
    against a persisted verifier (an envelope of a fixed constant). A key
    whose identifier is known but whose material differs therefore fails
    closed with ``key_check_failed`` before it can write unreadable data.

    Args:
        cipher: Envelope cipher for the store's key purpose.
        accept_plaintext: ``True`` only while a plaintext store migrates.
        verifiers: Persistence for per-key verifiers.
    """

    def __init__(
        self,
        cipher: EnvelopeCipher,
        *,
        accept_plaintext: bool,
        verifiers: KeyVerifierStore,
        verified: set[str] | None = None,
    ) -> None:
        self._cipher = cipher
        self._accept_plaintext = accept_plaintext
        self._verifiers = verifiers
        self._verified = set() if verified is None else verified

    @property
    def state(self) -> str:
        """Return ``migrating`` or ``encrypted``."""

        return (
            ENCRYPTION_STATE_MIGRATING
            if self._accept_plaintext
            else ENCRYPTION_STATE_ENCRYPTED
        )

    @property
    def key_purpose(self) -> str:
        """Return the key purpose bound into envelopes."""

        return self._cipher.purpose

    def completed(self) -> "EncryptedJourneyCodec":
        """Return an equivalent codec that no longer accepts plaintext."""

        return EncryptedJourneyCodec(
            self._cipher,
            accept_plaintext=False,
            verifiers=self._verifiers,
            verified=self._verified,
        )

    def active_key_id(self) -> str:
        """Return the identifier of the provider's verified active key."""

        return self._verified_active_key().key_id

    def check_active_key(self) -> None:
        """Verify the active key now, if the provider has one.

        Raises:
            JourneyEncryptionError: ``key_check_failed`` when the active key's
                material does not match its persisted verifier.
        """

        try:
            key = self._cipher.active_key()
        except KeyUnavailableError:
            return
        self._verify(key)

    def seal_payload(
        self, table: str, identity: Sequence[str | int], payload_json: str
    ) -> str:
        """Seal one payload column value bound to its table and identity."""

        return self._cipher.seal_text(
            payload_json.encode("utf-8"),
            context=_payload_context(table, identity),
            key=self._verified_active_key(),
        )

    def open_payload(
        self, table: str, identity: Sequence[str | int], stored: Any
    ) -> str:
        """Open one payload column value, failing closed on plaintext."""

        if not is_text_envelope(stored):
            if self._accept_plaintext and isinstance(stored, str):
                return stored
            raise JourneyEncryptionError("payload_not_encrypted")
        plaintext = self._cipher.open_text(
            stored,
            context=_payload_context(table, identity),
        )
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError:
            raise EnvelopeFormatError() from None

    def payload_key_id(self, stored: Any) -> str | None:
        """Return the sealing key id, or ``None`` for migratable plaintext."""

        if not is_text_envelope(stored):
            if self._accept_plaintext and isinstance(stored, str):
                return None
            raise JourneyEncryptionError("payload_not_encrypted")
        return read_envelope_header(envelope_from_text(stored)).key_id

    def seal_blob(self, content_hash: str, content: bytes) -> bytes:
        """Seal artifact bytes bound to their content digest."""

        return self._cipher.seal(
            content,
            context=_blob_context(content_hash),
            key=self._verified_active_key(),
        )

    def open_blob(self, content_hash: str, stored: bytes) -> bytes:
        """Open artifact bytes, failing closed on plaintext."""

        if self._accept_plaintext and sha256_digest(stored) == content_hash:
            return stored
        if not is_envelope(stored):
            raise JourneyEncryptionError("payload_not_encrypted")
        return self._cipher.open(stored, context=_blob_context(content_hash))

    def blob_key_id(self, content_hash: str, stored: bytes) -> str | None:
        """Return the sealing key id, or ``None`` for migratable plaintext."""

        if self._accept_plaintext and sha256_digest(stored) == content_hash:
            return None
        if not is_envelope(stored):
            raise JourneyEncryptionError("payload_not_encrypted")
        return read_envelope_header(stored).key_id

    def __repr__(self) -> str:
        return f"{type(self).__name__}(state={self.state!r})"

    def _verified_active_key(self) -> DataKey:
        key = self._cipher.active_key()
        if key.key_id not in self._verified:
            self._verify(key)
        return key

    def _verify(self, key: DataKey) -> None:
        stored = self._verifiers.load_key_verifier(key.key_id)
        if stored is None:
            # Recorded, but only trusted once read back: a write that rolls
            # back with its transaction is simply recreated next time.
            self._verifiers.save_key_verifier(
                key.key_id,
                self._cipher.seal_text(
                    _KEY_CHECK_PLAINTEXT,
                    context=_KEY_CHECK_CONTEXT,
                    key=key,
                ),
            )
            return
        try:
            header = read_envelope_header(envelope_from_text(stored))
            matches = header.key_id == key.key_id and (
                self._cipher.open_text(stored, context=_KEY_CHECK_CONTEXT)
                == _KEY_CHECK_PLAINTEXT
            )
        except EnvelopeAuthenticationError:
            matches = False
        if not matches:
            raise JourneyEncryptionError("key_check_failed")
        self._verified.add(key.key_id)


@dataclass(frozen=True, slots=True)
class EncryptionBinding:
    """Decision produced when a store binds its persisted encryption state."""

    codec: PlaintextJourneyCodec | EncryptedJourneyCodec
    persist: PersistedEncryptionState | None


def bind_encryption_state(
    encryption: JourneyStoreEncryption,
    current: PersistedEncryptionState | None,
    *,
    has_data: bool,
    verifiers: KeyVerifierStore,
) -> EncryptionBinding:
    """Reconcile a configuration with a store's persisted encryption state.

    Args:
        encryption: Validated store configuration.
        current: Persisted state, or ``None`` when none was recorded.
        has_data: Whether the store already holds payload values.
        verifiers: Persistence for per-key verifiers.

    Returns:
        The codec to use and, when it changes, the state to persist.

    Raises:
        JourneyEncryptionError: When the configuration and persisted state
            are incompatible.
    """

    if encryption.allow_plaintext:
        if current is None:
            return EncryptionBinding(
                PLAINTEXT_JOURNEY_CODEC,
                PersistedEncryptionState(ENCRYPTION_STATE_PLAINTEXT),
            )
        if current.state != ENCRYPTION_STATE_PLAINTEXT:
            raise JourneyEncryptionError("store_encrypted")
        return EncryptionBinding(PLAINTEXT_JOURNEY_CODEC, None)

    cipher = encryption.cipher()
    plaintext_source = (current is None and has_data) or (
        current is not None and current.state == ENCRYPTION_STATE_PLAINTEXT
    )
    if plaintext_source:
        if not encryption.migrate_plaintext:
            raise JourneyEncryptionError("store_migration_required")
        return EncryptionBinding(
            EncryptedJourneyCodec(cipher, accept_plaintext=True, verifiers=verifiers),
            PersistedEncryptionState(
                ENCRYPTION_STATE_MIGRATING,
                key_purpose=encryption.key_purpose,
                envelope_version=ENVELOPE_VERSION,
            ),
        )
    if current is None:
        return EncryptionBinding(
            EncryptedJourneyCodec(cipher, accept_plaintext=False, verifiers=verifiers),
            PersistedEncryptionState(
                ENCRYPTION_STATE_ENCRYPTED,
                key_purpose=encryption.key_purpose,
                envelope_version=ENVELOPE_VERSION,
            ),
        )
    if current.key_purpose != encryption.key_purpose:
        raise JourneyEncryptionError("key_purpose_mismatch")
    if current.envelope_version != ENVELOPE_VERSION:
        raise JourneyEncryptionError("envelope_version_unsupported")
    return EncryptionBinding(
        EncryptedJourneyCodec(
            cipher,
            accept_plaintext=current.state == ENCRYPTION_STATE_MIGRATING,
            verifiers=verifiers,
        ),
        None,
    )


def encrypted_state(codec: EncryptedJourneyCodec) -> PersistedEncryptionState:
    """Return the completed state for an encrypted codec's purpose."""

    return PersistedEncryptionState(
        ENCRYPTION_STATE_ENCRYPTED,
        key_purpose=codec.key_purpose,
        envelope_version=ENVELOPE_VERSION,
    )


def _payload_context(table: str, identity: Sequence[str | int]) -> dict[str, Any]:
    return {
        "identity": list(identity),
        "record": "journey_payload",
        "table": table,
    }


def _blob_context(content_hash: str) -> dict[str, Any]:
    return {"content_hash": content_hash, "record": "journey_artifact"}
