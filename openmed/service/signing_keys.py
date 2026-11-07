"""Webhook signing-key generations with active, retiring, and retired keys.

A :class:`SigningKeyring` is one complete key generation: exactly one
``active`` key that signs, optional ``retiring`` keys that still verify until
the end of a bounded overlap window, and ``retired`` tombstones that keep a
rotated-out key id from being accepted or reused. Keys are caller-owned:
OpenMed reads them from environment variables, secret files, or a keyring
file, and never generates, uploads, or logs key material.

:class:`SigningKeyringFile` resolves the newest complete generation from a
keyring file, refuses rollbacks to an older generation, and rotates by writing
a whole new generation atomically, so an interrupted rotation leaves either the
previous generation or the new one in place, never a partial key set.

Errors and evidence carry controlled codes, key ids, generation numbers, states,
and timestamps only. Secrets, secret-file paths, and environment values never
appear in messages, reprs, receipts, or :meth:`SigningKeyring.describe` output.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

from ._state_files import atomic_write_private, exclusive_lock, read_bounded
from .signing import normalize_key_id

KEY_STATE_ACTIVE = "active"
KEY_STATE_RETIRING = "retiring"
KEY_STATE_RETIRED = "retired"
KEY_STATES = frozenset({KEY_STATE_ACTIVE, KEY_STATE_RETIRING, KEY_STATE_RETIRED})

KEYRING_FORMAT = "openmed.signing-keyring"
KEYRING_SCHEMA_VERSION = 1
DEFAULT_KEY_OVERLAP_SECONDS = 24 * 60 * 60
MAX_KEY_OVERLAP_SECONDS = 7 * 24 * 60 * 60
MAX_KEYRING_KEYS = 16
MAX_RETIRED_KEYS = 8
MIN_ROTATION_SECRET_BYTES = 32
MAX_KEYRING_FILE_BYTES = 256 * 1024
MAX_SECRET_FILE_BYTES = 4096

KEYRING_FILE_ENV_VAR = "OPENMED_WEBHOOK_SIGNING_KEYRING_FILE"
SIGNING_SECRET_ENV_VAR = "OPENMED_WEBHOOK_SIGNING_SECRET"
SIGNING_SECRET_FILE_ENV_VAR = "OPENMED_WEBHOOK_SIGNING_SECRET_FILE"
SIGNING_KEY_ID_ENV_VAR = "OPENMED_WEBHOOK_SIGNING_KEY_ID"

KEY_UNKNOWN = "key_unknown"
KEY_NOT_YET_ACTIVE = "key_not_yet_active"
KEY_RETIRED = "key_retired"
KEY_OVERLAP_EXPIRED = "key_overlap_expired"
KEY_ID_REUSED = "key_id_reused"
KEYRING_INVALID = "keyring_invalid"
KEYRING_UNREADABLE = "keyring_unreadable"
KEYRING_SECRET_UNAVAILABLE = "keyring_secret_unavailable"
KEYRING_ROLLBACK = "keyring_rollback"
KEYRING_WRITE_FAILED = "keyring_write_failed"
ROTATION_INVALID = "rotation_invalid"

_DERIVED_KEY_ID_LABEL = b"OPENMED-SIGNING-KEY-ID-V1"
_DERIVED_KEY_ID_PREFIX = "hk_"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_DOCUMENT_FIELDS = frozenset(
    {"format", "schema_version", "generation", "rotated_at", "keys"}
)
_SECRET_FIELDS = ("secret_b64", "secret_file", "secret_env")
_ENTRY_FIELDS = frozenset(
    {"key_id", "state", "not_before", "verify_until", *_SECRET_FIELDS}
)

SecretValue = Union[bytes, str]


class SigningKeyError(ValueError):
    """Value-free signing-key failure.

    Attributes:
        code: Controlled reason code, such as ``"key_unknown"``. Messages never
            include secrets, file paths, or environment values.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UnknownSigningKeyError(SigningKeyError):
    """Raised for an unknown key id or a key that is not active yet."""


class RetiredSigningKeyError(SigningKeyError):
    """Raised for a retired key or a retiring key past its overlap window."""


class KeyringIntegrityError(SigningKeyError):
    """Raised when a keyring is malformed, incomplete, or unreadable."""


class KeyringRollbackError(SigningKeyError):
    """Raised when a keyring would move back to an older or forked generation."""


@dataclass(frozen=True)
class SigningKey:
    """One signing key and its lifecycle state.

    Attributes:
        key_id: Public identifier bound into every signature.
        state: ``active``, ``retiring``, or ``retired``.
        secret: HMAC secret; ``None`` for retired tombstones. Excluded from
            ``repr`` and equality.
        not_before: Unix time before which the key neither signs nor verifies.
        verify_until: For ``retiring`` keys, the Unix time the overlap window
            closes; verification after it fails with ``key_overlap_expired``.
    """

    key_id: str
    state: str
    secret: Optional[bytes] = field(default=None, repr=False, compare=False)
    not_before: Optional[float] = None
    verify_until: Optional[float] = None

    def __post_init__(self) -> None:
        try:
            normalize_key_id(self.key_id)
        except (TypeError, ValueError):
            raise KeyringIntegrityError(KEYRING_INVALID) from None
        if self.state not in KEY_STATES:
            raise KeyringIntegrityError(KEYRING_INVALID)
        if self.state == KEY_STATE_RETIRED:
            if (
                self.secret is not None
                or self.not_before is not None
                or self.verify_until is not None
            ):
                raise KeyringIntegrityError(KEYRING_INVALID)
            return
        if not isinstance(self.secret, bytes) or not self.secret:
            raise KeyringIntegrityError(KEYRING_INVALID)
        for value in (self.not_before, self.verify_until):
            if value is not None and not _is_finite_number(value):
                raise KeyringIntegrityError(KEYRING_INVALID)
        if self.state == KEY_STATE_ACTIVE and self.verify_until is not None:
            raise KeyringIntegrityError(KEYRING_INVALID)
        if self.state == KEY_STATE_RETIRING:
            if self.verify_until is None:
                raise KeyringIntegrityError(KEYRING_INVALID)
            if self.not_before is not None and self.verify_until <= self.not_before:
                raise KeyringIntegrityError(KEYRING_INVALID)

    def describe(self) -> dict[str, Any]:
        """Return value-free key evidence: id, state, and validity bounds."""
        return {
            "key_id": self.key_id,
            "state": self.state,
            "not_before": self.not_before,
            "verify_until": self.verify_until,
        }


@dataclass(frozen=True)
class SigningKeyring:
    """One complete, validated generation of signing keys.

    Attributes:
        generation: Monotonic generation number, starting at 1.
        keys: Keys of this generation; exactly one is ``active``.
        rotated_at: Unix time the generation was created. Required whenever a
            ``retiring`` key is present, so the overlap window stays bounded by
            :data:`MAX_KEY_OVERLAP_SECONDS`.
    """

    generation: int
    keys: tuple[SigningKey, ...]
    rotated_at: Optional[float] = None

    def __post_init__(self) -> None:
        keys = tuple(self.keys)
        object.__setattr__(self, "keys", keys)
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise KeyringIntegrityError(KEYRING_INVALID)
        if self.rotated_at is not None and not _is_finite_number(self.rotated_at):
            raise KeyringIntegrityError(KEYRING_INVALID)
        if not 1 <= len(keys) <= MAX_KEYRING_KEYS:
            raise KeyringIntegrityError(KEYRING_INVALID)
        if any(not isinstance(key, SigningKey) for key in keys):
            raise KeyringIntegrityError(KEYRING_INVALID)
        if len({key.key_id for key in keys}) != len(keys):
            raise KeyringIntegrityError(KEYRING_INVALID)
        if sum(key.state == KEY_STATE_ACTIVE for key in keys) != 1:
            raise KeyringIntegrityError(KEYRING_INVALID)
        for key in keys:
            if key.state != KEY_STATE_RETIRING:
                continue
            assert key.verify_until is not None
            if (
                self.rotated_at is None
                or key.verify_until <= self.rotated_at
                or key.verify_until > self.rotated_at + MAX_KEY_OVERLAP_SECONDS
            ):
                raise KeyringIntegrityError(KEYRING_INVALID)

    @classmethod
    def single(
        cls,
        secret: SecretValue,
        *,
        key_id: Optional[str] = None,
        generation: int = 1,
    ) -> "SigningKeyring":
        """Build a one-key generation from a shared secret.

        Args:
            secret: Non-empty HMAC secret.
            key_id: Public key id; derived from the secret with
                :func:`derive_key_id` when omitted.
            generation: Generation number of the keyring.

        Returns:
            A keyring whose only key is active.
        """
        secret_bytes = _secret_bytes(secret)
        return cls(
            generation=generation,
            keys=(
                SigningKey(
                    key_id=key_id or derive_key_id(secret_bytes),
                    state=KEY_STATE_ACTIVE,
                    secret=secret_bytes,
                ),
            ),
        )

    def current(self) -> "SigningKeyring":
        """Return this keyring; lets a static keyring act as a resolver."""
        return self

    @property
    def active_key(self) -> SigningKey:
        """The generation's single active key."""
        return next(key for key in self.keys if key.state == KEY_STATE_ACTIVE)

    def signing_key(self, *, now: Optional[float] = None) -> SigningKey:
        """Return the key that signs new attempts.

        Args:
            now: Unix time; defaults to the current time.

        Returns:
            The active key.

        Raises:
            UnknownSigningKeyError: With ``key_not_yet_active`` when the active
                key's ``not_before`` is still in the future.
        """
        current_time = time.time() if now is None else float(now)
        key = self.active_key
        if key.not_before is not None and current_time < key.not_before:
            raise UnknownSigningKeyError(KEY_NOT_YET_ACTIVE)
        return key

    def verification_key(
        self, key_id: str, *, now: Optional[float] = None
    ) -> SigningKey:
        """Resolve the key named by a signature for verification at ``now``.

        Args:
            key_id: Key id from the signature header.
            now: Unix time; defaults to the current time.

        Returns:
            An active key, or a retiring key inside its overlap window.

        Raises:
            UnknownSigningKeyError: ``key_unknown`` for an id outside this
                generation, ``key_not_yet_active`` before ``not_before``.
            RetiredSigningKeyError: ``key_retired`` for a tombstone,
                ``key_overlap_expired`` after a retiring key's window closed.
        """
        current_time = time.time() if now is None else float(now)
        key = next((key for key in self.keys if key.key_id == key_id), None)
        if key is None:
            raise UnknownSigningKeyError(KEY_UNKNOWN)
        if key.state == KEY_STATE_RETIRED:
            raise RetiredSigningKeyError(KEY_RETIRED)
        if key.not_before is not None and current_time < key.not_before:
            raise UnknownSigningKeyError(KEY_NOT_YET_ACTIVE)
        if key.state == KEY_STATE_RETIRING:
            assert key.verify_until is not None
            if current_time > key.verify_until:
                raise RetiredSigningKeyError(KEY_OVERLAP_EXPIRED)
        return key

    def verification_keys(
        self, *, now: Optional[float] = None
    ) -> tuple[SigningKey, ...]:
        """Return every key that verifies at ``now``, active key first."""
        usable = []
        for key in self.keys:
            if key.state == KEY_STATE_RETIRED:
                continue
            try:
                usable.append(self.verification_key(key.key_id, now=now))
            except SigningKeyError:
                continue
        return tuple(sorted(usable, key=lambda key: key.state != KEY_STATE_ACTIVE))

    def describe(self) -> dict[str, Any]:
        """Return value-free generation evidence without any key material."""
        return {
            "format": KEYRING_FORMAT,
            "schema_version": KEYRING_SCHEMA_VERSION,
            "generation": self.generation,
            "rotated_at": self.rotated_at,
            "keys": [key.describe() for key in self.keys],
        }


class SigningKeyResolver(Protocol):
    """Anything that returns the current complete signing-key generation."""

    def current(self) -> SigningKeyring:
        """Return the current generation."""
        ...


@dataclass(frozen=True)
class KeyringRotationReceipt:
    """Value-free record of one completed keyring rotation.

    Attributes:
        previous_generation: Generation that was replaced; ``None`` on the
            first rotation of a new keyring file.
        generation: Generation that is now active.
        rotated_at: Unix time of the rotation.
        active_key_id: Key id that signs from now on.
        retiring_key_ids: Key ids that keep verifying until ``verify_until``.
        retired_key_ids: Key ids that no longer verify.
        verify_until: End of the overlap window, or ``None`` without retiring
            keys.
    """

    previous_generation: Optional[int]
    generation: int
    rotated_at: float
    active_key_id: str
    retiring_key_ids: tuple[str, ...]
    retired_key_ids: tuple[str, ...]
    verify_until: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe receipt."""
        return {
            "previous_generation": self.previous_generation,
            "generation": self.generation,
            "rotated_at": self.rotated_at,
            "active_key_id": self.active_key_id,
            "retiring_key_ids": list(self.retiring_key_ids),
            "retired_key_ids": list(self.retired_key_ids),
            "verify_until": self.verify_until,
        }


class SigningKeyringFile:
    """File-backed signing-key resolver with atomic rotation.

    :meth:`current` reloads the file when it changes. A reload that finds a
    malformed or incomplete generation, or a generation older than one already
    loaded, keeps the previously loaded generation and records the reason in
    :attr:`last_error`. Rotation runs under a cross-process lock and replaces
    the whole file atomically.

    Args:
        path: Keyring file path. Relative ``secret_file`` entries resolve
            against its directory.
        min_generation: Lowest generation this resolver accepts, for callers
            that persist a high-water mark across restarts.
        environ: Environment used for ``secret_env`` entries; defaults to
            :data:`os.environ`.
        clock: Source of Unix time for rotations.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        min_generation: int = 0,
        environ: Optional[Mapping[str, str]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if (
            isinstance(min_generation, bool)
            or not isinstance(min_generation, int)
            or min_generation < 0
        ):
            raise ValueError("min_generation must be a non-negative integer")
        self.path = Path(path)
        self._environ = environ
        self._clock = clock
        self._lock = threading.Lock()
        self._keyring: Optional[SigningKeyring] = None
        self._fingerprint: Optional[str] = None
        self._file_state: Optional[tuple[int, int, int]] = None
        self._high_water = min_generation
        self._last_error: Optional[str] = None

    @property
    def last_error(self) -> Optional[str]:
        """Controlled code of the most recent rejected reload, if any."""
        return self._last_error

    @property
    def generation(self) -> Optional[int]:
        """Generation currently in use, or ``None`` before the first load."""
        return None if self._keyring is None else self._keyring.generation

    def current(self) -> SigningKeyring:
        """Return the newest complete generation.

        Returns:
            The loaded keyring.

        Raises:
            SigningKeyError: If no complete generation has loaded yet.
        """
        with self._lock:
            file_state = self._stat()
            if self._keyring is not None and file_state == self._file_state:
                return self._keyring
            previous = self._keyring
            try:
                return self._load_locked(file_state)
            except SigningKeyError as exc:
                self._last_error = exc.code
                self._file_state = file_state
                if previous is None:
                    raise
                return previous

    def reload(self) -> SigningKeyring:
        """Reload the file now, raising instead of keeping the old generation.

        Returns:
            The newly loaded keyring.

        Raises:
            SigningKeyError: If the file is invalid, incomplete, or a rollback.
                The previously loaded generation stays in use.
        """
        with self._lock:
            try:
                return self._load_locked(self._stat())
            except SigningKeyError as exc:
                self._last_error = exc.code
                raise

    def rotate(
        self,
        *,
        new_secret: SecretValue,
        new_key_id: Optional[str] = None,
        overlap_seconds: float = DEFAULT_KEY_OVERLAP_SECONDS,
        now: Optional[float] = None,
    ) -> KeyringRotationReceipt:
        """Atomically activate a new key generation.

        The new key becomes active, the previous active key becomes retiring
        until ``now + overlap_seconds``, a previously retiring key becomes a
        retired tombstone, and at most :data:`MAX_RETIRED_KEYS` tombstones are
        kept. The new secret is written inline as base64 in the ``0600`` file.

        Args:
            new_secret: At least :data:`MIN_ROTATION_SECRET_BYTES` bytes, for
                example ``secrets.token_bytes(32)``.
            new_key_id: Public id for the new key; derived from the secret
                when omitted. Ids already present in the file are refused.
            overlap_seconds: How long the previous active key keeps verifying;
                greater than 0 and at most :data:`MAX_KEY_OVERLAP_SECONDS`.
            now: Unix time of the rotation; defaults to the resolver clock.

        Returns:
            A value-free rotation receipt.

        Raises:
            SigningKeyError: ``rotation_invalid`` for bad arguments,
                ``key_id_reused``, ``keyring_write_failed``, or the integrity
                and rollback codes of the current file. The file is unchanged.
        """
        secret_bytes = _secret_bytes(new_secret, code=ROTATION_INVALID)
        if len(secret_bytes) < MIN_ROTATION_SECRET_BYTES:
            raise SigningKeyError(ROTATION_INVALID)
        if (
            not _is_finite_number(overlap_seconds)
            or overlap_seconds <= 0
            or overlap_seconds > MAX_KEY_OVERLAP_SECONDS
        ):
            raise SigningKeyError(ROTATION_INVALID)
        key_id = new_key_id or derive_key_id(secret_bytes)
        try:
            normalize_key_id(key_id)
        except (TypeError, ValueError):
            raise SigningKeyError(ROTATION_INVALID) from None
        rotated_at = float(self._clock() if now is None else now)
        if not _is_finite_number(rotated_at):
            raise SigningKeyError(ROTATION_INVALID)

        lock_path = self.path.with_name(f"{self.path.name}.lock")
        with self._lock:
            try:
                with exclusive_lock(lock_path):
                    return self._rotate_locked(
                        secret_bytes=secret_bytes,
                        key_id=key_id,
                        overlap_seconds=float(overlap_seconds),
                        rotated_at=rotated_at,
                    )
            except OSError:
                raise SigningKeyError(KEYRING_WRITE_FAILED) from None

    def _rotate_locked(
        self,
        *,
        secret_bytes: bytes,
        key_id: str,
        overlap_seconds: float,
        rotated_at: float,
    ) -> KeyringRotationReceipt:
        data = self._read_file()
        previous: Optional[SigningKeyring] = None
        entries: list[Mapping[str, Any]] = []
        if data is None:
            if self._high_water > 0:
                raise KeyringRollbackError(KEYRING_ROLLBACK)
        else:
            document = _decode_document(data)
            previous = keyring_from_document(
                document, base_dir=self.path.parent, environ=self._environ
            )
            self._check_generation(previous, _fingerprint(data))
            entries = list(document["keys"])
            if previous.rotated_at is not None and rotated_at < previous.rotated_at:
                raise SigningKeyError(ROTATION_INVALID)
            if any(key.key_id == key_id for key in previous.keys):
                raise SigningKeyError(KEY_ID_REUSED)

        verify_until = rotated_at + overlap_seconds
        new_entries: list[dict[str, Any]] = [
            {
                "key_id": key_id,
                "state": KEY_STATE_ACTIVE,
                "secret_b64": base64.b64encode(secret_bytes).decode("ascii"),
                "not_before": rotated_at,
            }
        ]
        retiring: list[str] = []
        retired: list[dict[str, Any]] = []
        if previous is not None:
            for key, entry in zip(previous.keys, entries):
                if key.state == KEY_STATE_ACTIVE:
                    retiring_entry: dict[str, Any] = {
                        "key_id": key.key_id,
                        "state": KEY_STATE_RETIRING,
                        "not_before": key.not_before,
                        "verify_until": verify_until,
                    }
                    for name in _SECRET_FIELDS:
                        if name in entry:
                            retiring_entry[name] = entry[name]
                    new_entries.append(retiring_entry)
                    retiring.append(key.key_id)
            ordered = [key for key in previous.keys if key.state == KEY_STATE_RETIRING]
            ordered += [key for key in previous.keys if key.state == KEY_STATE_RETIRED]
            for key in ordered[:MAX_RETIRED_KEYS]:
                retired.append({"key_id": key.key_id, "state": KEY_STATE_RETIRED})
        new_entries.extend(retired)
        document = {
            "format": KEYRING_FORMAT,
            "schema_version": KEYRING_SCHEMA_VERSION,
            "generation": 1 if previous is None else previous.generation + 1,
            "rotated_at": rotated_at,
            "keys": new_entries,
        }
        # Validate the complete new generation before anything is written.
        keyring = keyring_from_document(
            document, base_dir=self.path.parent, environ=self._environ
        )
        encoded = _encode_document(document)
        atomic_write_private(self.path, encoded)
        self._commit(keyring, _fingerprint(encoded), self._stat())
        return KeyringRotationReceipt(
            previous_generation=None if previous is None else previous.generation,
            generation=keyring.generation,
            rotated_at=rotated_at,
            active_key_id=key_id,
            retiring_key_ids=tuple(retiring),
            retired_key_ids=tuple(entry["key_id"] for entry in retired),
            verify_until=verify_until if retiring else None,
        )

    def _load_locked(
        self, file_state: Optional[tuple[int, int, int]]
    ) -> SigningKeyring:
        data = self._read_file()
        if data is None:
            raise KeyringIntegrityError(KEYRING_UNREADABLE)
        keyring = keyring_from_document(
            _decode_document(data), base_dir=self.path.parent, environ=self._environ
        )
        fingerprint = _fingerprint(data)
        self._check_generation(keyring, fingerprint)
        self._commit(keyring, fingerprint, file_state)
        return keyring

    def _check_generation(self, keyring: SigningKeyring, fingerprint: str) -> None:
        if keyring.generation < self._high_water:
            raise KeyringRollbackError(KEYRING_ROLLBACK)
        if (
            self._keyring is not None
            and keyring.generation == self._keyring.generation
            and fingerprint != self._fingerprint
        ):
            # Same generation number, different keys: a fork or a hand edit
            # that skipped the generation bump. Never silently adopt it.
            raise KeyringRollbackError(KEYRING_ROLLBACK)

    def _commit(
        self,
        keyring: SigningKeyring,
        fingerprint: str,
        file_state: Optional[tuple[int, int, int]],
    ) -> None:
        self._keyring = keyring
        self._fingerprint = fingerprint
        self._file_state = file_state
        self._high_water = max(self._high_water, keyring.generation)
        self._last_error = None

    def _read_file(self) -> Optional[bytes]:
        try:
            return read_bounded(self.path, MAX_KEYRING_FILE_BYTES)
        except OSError:
            raise KeyringIntegrityError(KEYRING_UNREADABLE) from None

    def _stat(self) -> Optional[tuple[int, int, int]]:
        try:
            info = os.stat(self.path)
        except OSError:
            return None
        return (info.st_mtime_ns, info.st_size, info.st_ino)


def derive_key_id(secret: SecretValue) -> str:
    """Derive a stable public key id from a secret.

    The id is a truncated HMAC of a fixed label keyed by the secret, so it
    reveals nothing about the secret beyond what any signature already does,
    and a receiver holding the same secret derives the same id.

    Args:
        secret: Non-empty HMAC secret.

    Returns:
        An ``hk_``-prefixed 24-hex-character key id.
    """
    digest = hmac.new(_secret_bytes(secret), _DERIVED_KEY_ID_LABEL, hashlib.sha256)
    return f"{_DERIVED_KEY_ID_PREFIX}{digest.hexdigest()[:24]}"


def keyring_from_document(
    document: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> SigningKeyring:
    """Validate a keyring document and resolve its secrets.

    Each active or retiring entry names exactly one secret source:
    ``secret_b64`` (inline base64), ``secret_file`` (a file holding the raw
    secret; one trailing newline is ignored), or ``secret_env`` (an environment
    variable). Retired entries carry only ``key_id`` and ``state``. An entry
    without ``key_id`` uses :func:`derive_key_id`.

    Args:
        document: Parsed keyring JSON.
        base_dir: Directory that relative ``secret_file`` paths resolve
            against; defaults to the current directory.
        environ: Environment for ``secret_env``; defaults to :data:`os.environ`.

    Returns:
        The complete generation.

    Raises:
        KeyringIntegrityError: ``keyring_invalid`` for a malformed or
            incomplete generation, ``keyring_secret_unavailable`` when a
            referenced secret cannot be read.
    """
    if not isinstance(document, Mapping) or set(document) - _DOCUMENT_FIELDS:
        raise KeyringIntegrityError(KEYRING_INVALID)
    if (
        document.get("format") != KEYRING_FORMAT
        or document.get("schema_version") != KEYRING_SCHEMA_VERSION
        or isinstance(document.get("schema_version"), bool)
    ):
        raise KeyringIntegrityError(KEYRING_INVALID)
    entries = document.get("keys")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise KeyringIntegrityError(KEYRING_INVALID)
    rotated_at = document.get("rotated_at")
    if rotated_at is not None and not _is_finite_number(rotated_at):
        raise KeyringIntegrityError(KEYRING_INVALID)
    keys = tuple(
        _key_from_entry(entry, base_dir=base_dir, environ=environ) for entry in entries
    )
    return SigningKeyring(
        generation=document.get("generation"),  # type: ignore[arg-type]
        keys=keys,
        rotated_at=None if rotated_at is None else float(rotated_at),
    )


def load_signing_keyring(
    path: Union[str, os.PathLike[str]],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> SigningKeyring:
    """Load and validate one keyring file.

    Args:
        path: Keyring JSON file.
        environ: Environment for ``secret_env`` entries.

    Returns:
        The complete generation in the file.

    Raises:
        KeyringIntegrityError: If the file is missing, unreadable, malformed,
            or incomplete.
    """
    keyring_path = Path(path)
    try:
        data = read_bounded(keyring_path, MAX_KEYRING_FILE_BYTES)
    except OSError:
        raise KeyringIntegrityError(KEYRING_UNREADABLE) from None
    if data is None:
        raise KeyringIntegrityError(KEYRING_UNREADABLE)
    return keyring_from_document(
        _decode_document(data), base_dir=keyring_path.parent, environ=environ
    )


def signing_keyring_from_env(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[SigningKeyResolver]:
    """Build a signing-key resolver from environment variables.

    ``OPENMED_WEBHOOK_SIGNING_KEYRING_FILE`` selects a rotating keyring file.
    Otherwise ``OPENMED_WEBHOOK_SIGNING_SECRET_FILE`` or
    ``OPENMED_WEBHOOK_SIGNING_SECRET`` provides a single active key, named by
    ``OPENMED_WEBHOOK_SIGNING_KEY_ID`` or derived from the secret.

    Args:
        environ: Environment to read; defaults to :data:`os.environ`.

    Returns:
        A resolver, or ``None`` when no signing key is configured.

    Raises:
        SigningKeyError: If the configuration is ambiguous, incomplete, or the
            keyring file does not hold a complete generation.
    """
    env = os.environ if environ is None else environ
    keyring_file = env.get(KEYRING_FILE_ENV_VAR, "").strip()
    secret_file = env.get(SIGNING_SECRET_FILE_ENV_VAR, "").strip()
    secret_value = env.get(SIGNING_SECRET_ENV_VAR, "")
    configured = [bool(keyring_file), bool(secret_file), bool(secret_value)]
    if sum(configured) > 1:
        raise KeyringIntegrityError(KEYRING_INVALID)
    key_id = env.get(SIGNING_KEY_ID_ENV_VAR, "").strip() or None
    if keyring_file:
        if key_id is not None:
            raise KeyringIntegrityError(KEYRING_INVALID)
        resolver = SigningKeyringFile(Path(keyring_file).expanduser(), environ=env)
        resolver.current()
        return resolver
    if secret_file:
        secret = _read_secret_file(Path(secret_file).expanduser())
    elif secret_value:
        secret = secret_value.encode("utf-8")
    else:
        return None
    return SigningKeyring.single(secret, key_id=key_id)


def _key_from_entry(
    entry: Any,
    *,
    base_dir: Optional[Path],
    environ: Optional[Mapping[str, str]],
) -> SigningKey:
    if not isinstance(entry, Mapping) or set(entry) - _ENTRY_FIELDS:
        raise KeyringIntegrityError(KEYRING_INVALID)
    state = entry.get("state")
    if not isinstance(state, str) or state not in KEY_STATES:
        raise KeyringIntegrityError(KEYRING_INVALID)
    sources = [name for name in _SECRET_FIELDS if name in entry]
    secret: Optional[bytes] = None
    if state == KEY_STATE_RETIRED:
        if sources or "key_id" not in entry:
            raise KeyringIntegrityError(KEYRING_INVALID)
    else:
        if len(sources) != 1:
            raise KeyringIntegrityError(KEYRING_INVALID)
        secret = _resolve_secret(
            sources[0], entry[sources[0]], base_dir=base_dir, environ=environ
        )
    key_id = entry.get("key_id")
    if key_id is None:
        assert secret is not None
        key_id = derive_key_id(secret)
    for name in ("not_before", "verify_until"):
        value = entry.get(name)
        if value is not None and not _is_finite_number(value):
            raise KeyringIntegrityError(KEYRING_INVALID)
    not_before = entry.get("not_before")
    verify_until = entry.get("verify_until")
    return SigningKey(
        key_id=key_id,
        state=state,
        secret=secret,
        not_before=None if not_before is None else float(not_before),
        verify_until=None if verify_until is None else float(verify_until),
    )


def _resolve_secret(
    source: str,
    value: Any,
    *,
    base_dir: Optional[Path],
    environ: Optional[Mapping[str, str]],
) -> bytes:
    if not isinstance(value, str) or not value:
        raise KeyringIntegrityError(KEYRING_INVALID)
    if source == "secret_b64":
        try:
            secret = base64.b64decode(value.encode("ascii"), validate=True)
        except (binascii.Error, UnicodeEncodeError):
            raise KeyringIntegrityError(KEYRING_INVALID) from None
    elif source == "secret_file":
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        secret = _read_secret_file(path)
    else:
        if not _ENV_NAME_RE.fullmatch(value):
            raise KeyringIntegrityError(KEYRING_INVALID)
        env = os.environ if environ is None else environ
        secret = env.get(value, "").encode("utf-8")
    if not secret:
        raise KeyringIntegrityError(KEYRING_SECRET_UNAVAILABLE)
    return secret


def _read_secret_file(path: Path) -> bytes:
    try:
        data = read_bounded(path, MAX_SECRET_FILE_BYTES)
    except OSError:
        raise KeyringIntegrityError(KEYRING_SECRET_UNAVAILABLE) from None
    if data is None:
        raise KeyringIntegrityError(KEYRING_SECRET_UNAVAILABLE)
    if data.endswith(b"\r\n"):
        data = data[:-2]
    elif data.endswith(b"\n"):
        data = data[:-1]
    if not data:
        raise KeyringIntegrityError(KEYRING_SECRET_UNAVAILABLE)
    return data


def _decode_document(data: bytes) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate key")
            result[name] = value
        return result

    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, ValueError):
        raise KeyringIntegrityError(KEYRING_INVALID) from None
    if not isinstance(document, dict):
        raise KeyringIntegrityError(KEYRING_INVALID)
    return document


def _encode_document(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _fingerprint(data: bytes) -> str:
    # Internal change detection only; never exposed because the file holds
    # secrets.
    return hashlib.sha256(data).hexdigest()


def _secret_bytes(secret: SecretValue, *, code: str = KEYRING_INVALID) -> bytes:
    if isinstance(secret, str):
        value = secret.encode("utf-8")
    elif isinstance(secret, (bytes, bytearray)):
        value = bytes(secret)
    else:
        raise SigningKeyError(code)
    if not value:
        raise SigningKeyError(code)
    return value


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


__all__ = [
    "DEFAULT_KEY_OVERLAP_SECONDS",
    "KEYRING_FILE_ENV_VAR",
    "KEYRING_FORMAT",
    "KEYRING_SCHEMA_VERSION",
    "KEY_STATES",
    "KEY_STATE_ACTIVE",
    "KEY_STATE_RETIRED",
    "KEY_STATE_RETIRING",
    "KeyringIntegrityError",
    "KeyringRollbackError",
    "KeyringRotationReceipt",
    "MAX_KEYRING_KEYS",
    "MAX_KEY_OVERLAP_SECONDS",
    "MAX_RETIRED_KEYS",
    "MIN_ROTATION_SECRET_BYTES",
    "RetiredSigningKeyError",
    "SIGNING_KEY_ID_ENV_VAR",
    "SIGNING_SECRET_ENV_VAR",
    "SIGNING_SECRET_FILE_ENV_VAR",
    "SigningKey",
    "SigningKeyError",
    "SigningKeyResolver",
    "SigningKeyring",
    "SigningKeyringFile",
    "UnknownSigningKeyError",
    "derive_key_id",
    "keyring_from_document",
    "load_signing_keyring",
    "signing_keyring_from_env",
]
