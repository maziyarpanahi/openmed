"""Keyed, domain-separated digests for service records.

Service records sometimes need a stable reference to a value that must never be
stored or returned in plaintext, such as the principal that submitted a job.
Unkeyed hashes of low-entropy values can be reversed with a dictionary, so this
module computes HMAC-SHA256 digests under a deployment key and mixes a fixed
domain label into every message. A digest made for one purpose therefore never
verifies for another, and each stored digest names its algorithm and key id so
a rotated or ephemeral key fails closed instead of matching by accident.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Optional

SERVICE_DIGEST_KEY_ENV_VAR: Final = "OPENMED_SERVICE_DIGEST_KEY"
SERVICE_DIGEST_KEY_ID_ENV_VAR: Final = "OPENMED_SERVICE_DIGEST_KEY_ID"
KEYED_DIGEST_ALGORITHM: Final = "hmac-sha256"
MIN_DIGEST_KEY_BYTES: Final = 32
DEVELOPMENT_PROFILES: Final = frozenset({"dev", "test"})

_MESSAGE_PREFIX: Final = b"openmed.service.keyed-digest.v1"
_KEY_ID_LABEL: Final = b"openmed.service.keyed-digest.key-id.v1"
_DOMAIN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._:-]{0,63}\Z")
_KEY_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_VALUE_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class DigestKeyError(ValueError):
    """Value-free failure while configuring the service digest key.

    Attributes:
        code: Stable reason code safe to log or surface to operators.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class KeyedDigest:
    """One stored keyed digest with its algorithm and key markers.

    Attributes:
        alg: Digest algorithm marker, currently always ``hmac-sha256``.
        key_id: Identifier of the deployment key that produced the digest.
        value: Lowercase hexadecimal HMAC output.
    """

    alg: str
    key_id: str
    value: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-safe stored representation."""
        return {"alg": self.alg, "key_id": self.key_id, "value": self.value}

    @classmethod
    def from_dict(cls, raw: Any) -> "KeyedDigest":
        """Parse a stored digest, rejecting anything malformed.

        Args:
            raw: Mapping previously produced by :meth:`to_dict`.

        Returns:
            The parsed digest.

        Raises:
            ValueError: If the mapping is not a well-formed keyed digest.
        """
        if not isinstance(raw, Mapping) or set(raw) != {"alg", "key_id", "value"}:
            raise ValueError("keyed digest is malformed")
        alg, key_id, value = raw["alg"], raw["key_id"], raw["value"]
        if (
            type(alg) is not str
            or alg != KEYED_DIGEST_ALGORITHM
            or type(key_id) is not str
            or not _KEY_ID_RE.fullmatch(key_id)
            or type(value) is not str
            or not _VALUE_RE.fullmatch(value)
        ):
            raise ValueError("keyed digest is malformed")
        return cls(alg=alg, key_id=key_id, value=value)


@dataclass(frozen=True, eq=False)
class ServiceDigestKey:
    """Deployment secret used for keyed, domain-separated digests.

    Attributes:
        secret: Secret key material; never included in ``repr`` output.
        key_id: Public identifier recorded next to every digest.
        ephemeral: Whether the key was generated for this process only.
    """

    secret: bytes = field(repr=False)
    key_id: str
    ephemeral: bool = False

    def __post_init__(self) -> None:
        _require_secret(self.secret)
        if type(self.key_id) is not str or not _KEY_ID_RE.fullmatch(self.key_id):
            raise DigestKeyError(
                "digest_key_id_invalid",
                f"{SERVICE_DIGEST_KEY_ID_ENV_VAR} must be 1-64 letters, digits, "
                "dots, underscores or hyphens",
            )

    @classmethod
    def from_secret(
        cls,
        secret: bytes,
        *,
        key_id: Optional[str] = None,
    ) -> "ServiceDigestKey":
        """Build a configured key, deriving a key id when none is given.

        Args:
            secret: At least 32 bytes of secret key material.
            key_id: Optional operator-chosen key identifier.

        Returns:
            A non-ephemeral digest key.

        Raises:
            DigestKeyError: If the secret is too short or the key id is invalid.
        """
        _require_secret(secret)
        return cls(secret=secret, key_id=key_id or "k-" + _fingerprint(secret))

    @classmethod
    def generate(cls) -> "ServiceDigestKey":
        """Return a random key that lives only as long as this process."""
        secret = secrets.token_bytes(MIN_DIGEST_KEY_BYTES)
        return cls(
            secret=secret,
            key_id="ephemeral-" + _fingerprint(secret),
            ephemeral=True,
        )

    def digest(self, domain: str, material: bytes) -> KeyedDigest:
        """Return the keyed digest of *material* within *domain*.

        Args:
            domain: Fixed purpose label such as ``job-owner``.
            material: Canonical bytes to digest; see :func:`encode_digest_fields`.

        Returns:
            Digest carrying this key's algorithm and key id markers.

        Raises:
            ValueError: If the domain label is invalid.
            TypeError: If the material is not bytes.
        """
        mac = hmac.new(self.secret, _message(domain, material), hashlib.sha256)
        return KeyedDigest(
            alg=KEYED_DIGEST_ALGORITHM,
            key_id=self.key_id,
            value=mac.hexdigest(),
        )

    def verify(self, candidate: KeyedDigest, domain: str, material: bytes) -> bool:
        """Return whether *candidate* is this key's digest of *material*.

        A digest made by a different algorithm or key id never verifies, so
        records written before a key rotation fail closed.

        Args:
            candidate: Previously stored digest.
            domain: Purpose label the digest was created for.
            material: Canonical bytes to compare against.

        Returns:
            ``True`` only for an exact match under this key and domain.
        """
        if candidate.alg != KEYED_DIGEST_ALGORITHM or not hmac.compare_digest(
            candidate.key_id, self.key_id
        ):
            return False
        expected = self.digest(domain, material).value
        return hmac.compare_digest(candidate.value, expected)


def encode_digest_fields(*fields: Optional[str]) -> bytes:
    """Encode ordered string fields as unambiguous digest material.

    Args:
        *fields: Strings or ``None`` values, in a fixed caller-defined order.

    Returns:
        Canonical ASCII JSON array bytes.

    Raises:
        TypeError: If a field is neither a string nor ``None``.
    """
    if any(item is not None and type(item) is not str for item in fields):
        raise TypeError("digest fields must be strings or None")
    return json.dumps(list(fields), ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )


def resolve_service_digest_key(
    *,
    profile: str,
    required: bool,
    environ: Optional[Mapping[str, str]] = None,
) -> ServiceDigestKey:
    """Resolve the deployment digest key from the environment.

    ``OPENMED_SERVICE_DIGEST_KEY`` supplies the secret (its UTF-8 bytes, at
    least 32) and ``OPENMED_SERVICE_DIGEST_KEY_ID`` an optional public key id.
    Without a configured key the service uses a random per-process key, except
    when *required* is set outside the ``dev`` and ``test`` profiles.

    Args:
        profile: Active service profile name.
        required: Whether durable keyed digests are needed, for example
            because authentication is enabled.
        environ: Environment mapping; defaults to the process environment.

    Returns:
        The configured key, or an ephemeral key where that is allowed.

    Raises:
        DigestKeyError: If the configured key is invalid, or missing when
            required outside the development profiles.
    """
    env = os.environ if environ is None else environ
    raw_secret = env.get(SERVICE_DIGEST_KEY_ENV_VAR)
    if raw_secret is not None and raw_secret.strip():
        raw_key_id = env.get(SERVICE_DIGEST_KEY_ID_ENV_VAR)
        key_id = raw_key_id.strip() if raw_key_id and raw_key_id.strip() else None
        return ServiceDigestKey.from_secret(
            raw_secret.strip().encode("utf-8"),
            key_id=key_id,
        )
    if required and profile not in DEVELOPMENT_PROFILES:
        raise DigestKeyError(
            "digest_key_required",
            f"{SERVICE_DIGEST_KEY_ENV_VAR} must be set when service authentication "
            "is enabled outside the dev and test profiles",
        )
    return ServiceDigestKey.generate()


def _message(domain: str, material: bytes) -> bytes:
    if type(domain) is not str or not _DOMAIN_RE.fullmatch(domain):
        raise ValueError("digest domain is invalid")
    if type(material) is not bytes:
        raise TypeError("digest material must be bytes")
    # The domain grammar excludes NUL, so the framing below is unambiguous.
    return _MESSAGE_PREFIX + b"\x00" + domain.encode("ascii") + b"\x00" + material


def _require_secret(secret: Any) -> None:
    if type(secret) is not bytes or len(secret) < MIN_DIGEST_KEY_BYTES:
        raise DigestKeyError(
            "digest_key_too_short",
            f"{SERVICE_DIGEST_KEY_ENV_VAR} must contain at least "
            f"{MIN_DIGEST_KEY_BYTES} bytes",
        )


def _fingerprint(secret: bytes) -> str:
    return hmac.new(secret, _KEY_ID_LABEL, hashlib.sha256).hexdigest()[:16]


__all__ = [
    "DEVELOPMENT_PROFILES",
    "KEYED_DIGEST_ALGORITHM",
    "MIN_DIGEST_KEY_BYTES",
    "SERVICE_DIGEST_KEY_ENV_VAR",
    "SERVICE_DIGEST_KEY_ID_ENV_VAR",
    "DigestKeyError",
    "KeyedDigest",
    "ServiceDigestKey",
    "encode_digest_fields",
    "resolve_service_digest_key",
]
