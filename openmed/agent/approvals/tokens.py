"""Single-use human approval tokens for high-impact local agent actions.

Tokens bind signed, metadata-only claims to one exact action digest, one
reviewer role, one exclusive expiry, and one random nonce. Verification claims
the nonce atomically before authorizing dispatch. This module never accepts or
retains action payloads, clinical values, reviewer identities, or credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeVar, cast

APPROVAL_TOKEN_SCHEMA_VERSION: Final = "openmed.agent.approval_token.v1"
APPROVAL_RECEIPT_SCHEMA_VERSION: Final = "openmed.agent.approval_receipt.v1"
APPROVAL_TOKEN_SIGNATURE_ALGORITHM: Final = "hmac-sha256"
APPROVAL_NONCE_BYTES: Final = 16

_SIGNATURE_PREFIX = f"{APPROVAL_TOKEN_SIGNATURE_ALGORITHM}:"
_SIGNATURE_RE = re.compile(r"hmac-sha256:[0-9a-f]{64}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_NONCE_RE = re.compile(r"nonce_[0-9a-f]{32}")
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE = rf"{_LABEL}(?:\.{_LABEL})+"
_LOCAL_NAME = r"[a-z][a-z0-9-]{0,63}"
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
_REVIEWER_ROLE_RE = re.compile(rf"role:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "action_digest",
        "reviewer_role",
        "expires_at",
        "nonce",
        "signature",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "action_digest",
        "reviewer_role",
        "token_digest",
        "consumed_at",
        "expires_at",
    }
)

_T = TypeVar("_T")
_NonceSource = Callable[[int], bytes]


class ApprovalTokenError(ValueError):
    """Base class for value-free approval-token failures.

    Args:
        code: Stable machine-readable failure code.
        field_name: Optional fixed public field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


class ApprovalTokenValidationError(ApprovalTokenError):
    """Raised when token, receipt, key, or expected metadata is malformed."""


class ApprovalSignatureError(ApprovalTokenError):
    """Raised when a token signature is invalid."""


class ApprovalExpiredError(ApprovalTokenError):
    """Raised when a token has reached its exclusive expiry time."""


class ApprovalReplayError(ApprovalTokenError):
    """Raised when a nonce has already been presented."""


class ApprovalActionMismatchError(ApprovalTokenError):
    """Raised after consuming a token presented for a changed action."""


class ApprovalReviewerRoleMismatchError(ApprovalTokenError):
    """Raised after consuming a token presented under another reviewer role."""


class ApprovalNonceStoreError(ApprovalTokenError):
    """Raised when the nonce store cannot make an atomic claim."""


class ApprovalNonceStore(Protocol):
    """Atomically record value-free nonce digests for replay protection."""

    def claim(
        self,
        nonce_digest: str,
        *,
        expires_at: int,
        now: int,
    ) -> bool:
        """Return ``True`` only for the first atomic claim of a nonce digest."""


@dataclass(frozen=True, slots=True, repr=False)
class ApprovalToken:
    """Signed approval claims for one exact high-impact action.

    Args:
        action_digest: SHA-256 digest of the exact reviewed action or preview.
        reviewer_role: Canonical policy role, never a reviewer identity.
        expires_at: Exclusive Unix expiry timestamp.
        nonce: Opaque 128-bit single-use nonce.
        signature: HMAC-SHA-256 signature over every other token field.
        schema_version: Stable token schema version.
    """

    action_digest: str
    reviewer_role: str
    expires_at: int
    nonce: str
    signature: str
    schema_version: str = APPROVAL_TOKEN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPROVAL_TOKEN_SCHEMA_VERSION:
            raise ApprovalTokenValidationError(
                "unsupported_schema_version", "schema_version"
            )
        _validate_digest(self.action_digest, "action_digest")
        _validate_reviewer_role(self.reviewer_role)
        _validate_timestamp(self.expires_at, "expires_at")
        _validate_nonce(self.nonce)
        if type(self.signature) is not str or (
            self.signature != "" and _SIGNATURE_RE.fullmatch(self.signature) is None
        ):
            raise ApprovalTokenValidationError("invalid_signature_format", "signature")

    def signing_payload(self) -> dict[str, str | int]:
        """Return every signed field except the signature itself."""

        return {
            "schema_version": self.schema_version,
            "action_digest": self.action_digest,
            "reviewer_role": self.reviewer_role,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
        }

    def to_dict(self) -> dict[str, str | int]:
        """Return the complete canonical JSON-compatible token."""

        payload = self.signing_payload()
        payload["signature"] = self.signature
        return payload

    def to_json(self) -> str:
        """Serialize the token as compact canonical JSON."""

        return _canonical_json(self.to_dict(), "token")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ApprovalToken":
        """Restore a token while rejecting omitted or unsigned fields."""

        values = _read_exact_mapping(payload, _TOKEN_FIELDS, "token")
        return cls(
            action_digest=cast(str, values["action_digest"]),
            reviewer_role=cast(str, values["reviewer_role"]),
            expires_at=cast(int, values["expires_at"]),
            nonce=cast(str, values["nonce"]),
            signature=cast(str, values["signature"]),
            schema_version=cast(str, values["schema_version"]),
        )

    @classmethod
    def from_json(cls, serialized: str | bytes | bytearray) -> "ApprovalToken":
        """Restore a token from strict JSON without normalizing claims."""

        return cls.from_dict(_parse_json(serialized, "token"))

    def __repr__(self) -> str:
        """Return a representation that does not expose the bearer token."""

        return "ApprovalToken(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ApprovalReceipt:
    """Value-free proof that one approval token was successfully consumed."""

    action_digest: str
    reviewer_role: str
    token_digest: str
    consumed_at: int
    expires_at: int
    schema_version: str = APPROVAL_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPROVAL_RECEIPT_SCHEMA_VERSION:
            raise ApprovalTokenValidationError(
                "unsupported_receipt_schema_version", "schema_version"
            )
        _validate_digest(self.action_digest, "action_digest")
        _validate_reviewer_role(self.reviewer_role)
        _validate_digest(self.token_digest, "token_digest")
        _validate_timestamp(self.consumed_at, "consumed_at")
        _validate_timestamp(self.expires_at, "expires_at")
        if self.consumed_at >= self.expires_at:
            raise ApprovalTokenValidationError("invalid_receipt_expiry", "expires_at")

    def to_dict(self) -> dict[str, str | int]:
        """Return deterministic metadata-only receipt fields."""

        return {
            "schema_version": self.schema_version,
            "action_digest": self.action_digest,
            "reviewer_role": self.reviewer_role,
            "token_digest": self.token_digest,
            "consumed_at": self.consumed_at,
            "expires_at": self.expires_at,
        }

    def to_json(self) -> str:
        """Serialize the receipt as compact canonical JSON."""

        return _canonical_json(self.to_dict(), "receipt")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ApprovalReceipt":
        """Restore a receipt from an exact metadata-only mapping."""

        values = _read_exact_mapping(payload, _RECEIPT_FIELDS, "receipt")
        return cls(
            action_digest=cast(str, values["action_digest"]),
            reviewer_role=cast(str, values["reviewer_role"]),
            token_digest=cast(str, values["token_digest"]),
            consumed_at=cast(int, values["consumed_at"]),
            expires_at=cast(int, values["expires_at"]),
            schema_version=cast(str, values["schema_version"]),
        )

    @classmethod
    def from_json(cls, serialized: str | bytes | bytearray) -> "ApprovalReceipt":
        """Restore a receipt from strict JSON."""

        return cls.from_dict(_parse_json(serialized, "receipt"))

    def __repr__(self) -> str:
        """Return a value-free representation safe for diagnostics."""

        return "ApprovalReceipt(<metadata-only>)"


class InMemoryApprovalNonceStore:
    """Thread-safe process-local nonce store for local workflows and tests.

    Applications with multiple processes must inject a store whose ``claim``
    operation is atomic across every approval consumer.
    """

    def __init__(self) -> None:
        self._claimed: dict[str, int] = {}
        self._lock = threading.Lock()

    def claim(
        self,
        nonce_digest: str,
        *,
        expires_at: int,
        now: int,
    ) -> bool:
        """Atomically retain a nonce digest through its token expiry."""

        _validate_digest(nonce_digest, "nonce_digest")
        _validate_timestamp(expires_at, "expires_at")
        _validate_timestamp(now, "now")
        with self._lock:
            expired = [
                digest
                for digest, retained_until in self._claimed.items()
                if retained_until <= now
            ]
            for digest in expired:
                del self._claimed[digest]
            if nonce_digest in self._claimed:
                return False
            self._claimed[nonce_digest] = expires_at
            return True

    def __repr__(self) -> str:
        """Return metadata without nonce digests."""

        with self._lock:
            count = len(self._claimed)
        return f"InMemoryApprovalNonceStore(claimed={count})"


class ApprovalTokenSigner:
    """Issue signed tokens after the application authenticates a reviewer."""

    def __init__(self, key: bytes) -> None:
        self._key = _validate_key(key)

    def issue(
        self,
        *,
        action_digest: str,
        reviewer_role: str,
        expires_at: int,
        nonce: str | None = None,
        nonce_source: _NonceSource | None = None,
    ) -> ApprovalToken:
        """Sign exact approval claims with a fresh or explicit nonce.

        ``nonce`` is intended for restoring application-owned issuance state;
        ``nonce_source`` is intended for deterministic tests. Runtime callers
        should normally omit both and use the secure local random source.
        """

        if nonce is not None and nonce_source is not None:
            raise ApprovalTokenValidationError("ambiguous_nonce", "nonce")
        resolved_nonce = (
            _generate_nonce(nonce_source) if nonce is None else _validate_nonce(nonce)
        )
        unsigned = ApprovalToken(
            action_digest=action_digest,
            reviewer_role=reviewer_role,
            expires_at=expires_at,
            nonce=resolved_nonce,
            signature="",
        )
        signature = _sign(unsigned.signing_payload(), self._key)
        return ApprovalToken(
            action_digest=unsigned.action_digest,
            reviewer_role=unsigned.reviewer_role,
            expires_at=unsigned.expires_at,
            nonce=unsigned.nonce,
            signature=signature,
            schema_version=unsigned.schema_version,
        )

    def __repr__(self) -> str:
        """Return a representation that never exposes key material."""

        return "ApprovalTokenSigner(<redacted>)"


class ApprovalTokenVerifier:
    """Verify and atomically consume approval tokens before dispatch."""

    def __init__(
        self,
        key: bytes,
        nonce_store: ApprovalNonceStore,
        *,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self._key = _validate_key(key)
        if not callable(getattr(nonce_store, "claim", None)):
            raise ApprovalTokenValidationError("invalid_nonce_store", "nonce_store")
        if not callable(clock):
            raise ApprovalTokenValidationError("invalid_clock", "clock")
        self._nonce_store = nonce_store
        self._clock = clock

    def consume(
        self,
        token: ApprovalToken | Mapping[str, Any] | str | bytes | bytearray,
        *,
        action_digest: str,
        reviewer_role: str,
        now: int | None = None,
    ) -> ApprovalReceipt:
        """Verify exact claims, consume the nonce, and return a safe receipt.

        A valid signed token is claimed before action and role comparison. A
        changed action or wrong role therefore fails and permanently consumes
        that presentation, forcing a fresh human approval.
        """

        candidate = _coerce_token(token)
        _validate_digest(action_digest, "action_digest")
        _validate_reviewer_role(reviewer_role)

        expected_signature = _sign(candidate.signing_payload(), self._key)
        if not hmac.compare_digest(expected_signature, candidate.signature):
            raise ApprovalSignatureError("invalid_signature", "token")

        current_time = _read_clock(self._clock) if now is None else now
        _validate_timestamp(current_time, "now")
        if current_time >= candidate.expires_at:
            raise ApprovalExpiredError("expired", "token")

        nonce_digest = _sha256(candidate.nonce.encode("ascii"))
        if not _claim_nonce(
            self._nonce_store,
            nonce_digest,
            expires_at=candidate.expires_at,
            now=current_time,
        ):
            raise ApprovalReplayError("replayed", "token")

        if not hmac.compare_digest(candidate.action_digest, action_digest):
            raise ApprovalActionMismatchError("action_mismatch", "action_digest")
        if not hmac.compare_digest(candidate.reviewer_role, reviewer_role):
            raise ApprovalReviewerRoleMismatchError(
                "reviewer_role_mismatch", "reviewer_role"
            )

        return ApprovalReceipt(
            action_digest=candidate.action_digest,
            reviewer_role=candidate.reviewer_role,
            token_digest=_sha256(candidate.to_json().encode("utf-8")),
            consumed_at=current_time,
            expires_at=candidate.expires_at,
        )

    def __repr__(self) -> str:
        """Return a representation that hides keys and nonce-store internals."""

        return "ApprovalTokenVerifier(<redacted>)"


def dispatch_with_approval_token(
    token: ApprovalToken | Mapping[str, Any] | str | bytes | bytearray,
    *,
    action_digest: str,
    reviewer_role: str,
    verifier: ApprovalTokenVerifier,
    dispatch: Callable[[], _T],
    now: int | None = None,
) -> tuple[_T, ApprovalReceipt]:
    """Consume an exact approval before invoking a zero-argument callback."""

    if not callable(dispatch):
        raise ApprovalTokenValidationError("invalid_dispatch", "dispatch")
    if not isinstance(verifier, ApprovalTokenVerifier):
        raise ApprovalTokenValidationError("invalid_verifier", "verifier")
    receipt = verifier.consume(
        token,
        action_digest=action_digest,
        reviewer_role=reviewer_role,
        now=now,
    )
    return dispatch(), receipt


def _coerce_token(
    value: ApprovalToken | Mapping[str, Any] | str | bytes | bytearray,
) -> ApprovalToken:
    if type(value) is ApprovalToken:
        return value
    if isinstance(value, Mapping):
        return ApprovalToken.from_dict(value)
    if isinstance(value, (str, bytes, bytearray)):
        return ApprovalToken.from_json(value)
    raise ApprovalTokenValidationError("invalid_token", "token")


def _read_exact_mapping(
    payload: Mapping[str, Any], expected_fields: frozenset[str], location: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or isinstance(payload, (str, bytes, bytearray)):
        raise ApprovalTokenValidationError("not_a_mapping", location)
    try:
        values = dict(payload)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ApprovalTokenValidationError("unreadable_mapping", location) from None
    if set(values) - expected_fields:
        raise ApprovalTokenValidationError("unknown_field", location)
    if expected_fields - set(values):
        raise ApprovalTokenValidationError("missing_field", location)
    return values


def _parse_json(
    serialized: str | bytes | bytearray, location: str
) -> Mapping[str, Any]:
    if not isinstance(serialized, (str, bytes, bytearray)):
        raise ApprovalTokenValidationError("invalid_json", location)
    try:
        decoded = json.loads(serialized, object_pairs_hook=_strict_json_object)
    except (KeyboardInterrupt, SystemExit):
        raise
    except ApprovalTokenValidationError:
        raise
    except BaseException:
        raise ApprovalTokenValidationError("invalid_json", location) from None
    if not isinstance(decoded, Mapping):
        raise ApprovalTokenValidationError("not_a_mapping", location)
    return decoded


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ApprovalTokenValidationError("duplicate_field", "json")
        result[key] = value
    return result


def _canonical_json(payload: Mapping[str, Any], location: str) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeError):
        raise ApprovalTokenValidationError("invalid_payload", location) from None


def _validate_digest(value: object, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ApprovalTokenValidationError("invalid_digest", field_name)
    return value


def _validate_reviewer_role(value: object) -> str:
    if type(value) is not str or _REVIEWER_ROLE_RE.fullmatch(value) is None:
        raise ApprovalTokenValidationError("invalid_reviewer_role", "reviewer_role")
    return value


def _validate_timestamp(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ApprovalTokenValidationError("invalid_timestamp", field_name)
    return value


def _validate_nonce(value: object) -> str:
    if type(value) is not str or _NONCE_RE.fullmatch(value) is None:
        raise ApprovalTokenValidationError("invalid_nonce", "nonce")
    return value


def _validate_key(value: object) -> bytes:
    if type(value) is not bytes or len(value) < 32:
        raise ApprovalTokenValidationError("invalid_key", "key")
    return value


def _generate_nonce(source: _NonceSource | None) -> str:
    resolved_source = secrets.token_bytes if source is None else source
    if not callable(resolved_source):
        raise ApprovalTokenValidationError("invalid_nonce_source", "nonce")
    try:
        raw = resolved_source(APPROVAL_NONCE_BYTES)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ApprovalTokenValidationError("invalid_nonce_source", "nonce") from None
    if type(raw) is not bytes or len(raw) != APPROVAL_NONCE_BYTES:
        raise ApprovalTokenValidationError("invalid_nonce_source", "nonce")
    return f"nonce_{raw.hex()}"


def _read_clock(clock: Callable[[], int]) -> int:
    try:
        value = clock()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ApprovalTokenValidationError("clock_unavailable", "clock") from None
    return _validate_timestamp(value, "now")


def _claim_nonce(
    store: ApprovalNonceStore,
    nonce_digest: str,
    *,
    expires_at: int,
    now: int,
) -> bool:
    try:
        claimed = store.claim(nonce_digest, expires_at=expires_at, now=now)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ApprovalNonceStoreError(
            "nonce_store_unavailable", "nonce_store"
        ) from None
    if type(claimed) is not bool:
        raise ApprovalNonceStoreError("invalid_nonce_store_result", "nonce_store")
    return claimed


def _sign(payload: Mapping[str, Any], key: bytes) -> str:
    encoded = _canonical_json(payload, "token").encode("utf-8")
    digest = hmac.new(key, encoded, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest}"


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


__all__ = [
    "APPROVAL_NONCE_BYTES",
    "APPROVAL_RECEIPT_SCHEMA_VERSION",
    "APPROVAL_TOKEN_SCHEMA_VERSION",
    "APPROVAL_TOKEN_SIGNATURE_ALGORITHM",
    "ApprovalActionMismatchError",
    "ApprovalExpiredError",
    "ApprovalNonceStore",
    "ApprovalNonceStoreError",
    "ApprovalReceipt",
    "ApprovalReplayError",
    "ApprovalReviewerRoleMismatchError",
    "ApprovalSignatureError",
    "ApprovalToken",
    "ApprovalTokenError",
    "ApprovalTokenSigner",
    "ApprovalTokenValidationError",
    "ApprovalTokenVerifier",
    "InMemoryApprovalNonceStore",
    "dispatch_with_approval_token",
]
