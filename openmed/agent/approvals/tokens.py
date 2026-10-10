"""Single-use human approval tokens for high-impact local agent actions.

Tokens bind signed, metadata-only claims to an exact action digest, reviewer
role, bounded lifetime, non-secret key identifier and random nonce. Verification
claims the nonce atomically before authorizing dispatch. This module never accepts or
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

APPROVAL_TOKEN_SCHEMA_VERSION: Final = "openmed.agent.approval_token.v2"
LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION: Final = "openmed.agent.approval_token.v1"
DEFAULT_APPROVAL_LIFETIME_SECONDS: Final = 900
MAX_APPROVAL_LIFETIME_SECONDS: Final = 86_400
MAX_APPROVAL_CLOCK_SKEW_SECONDS: Final = 300
APPROVAL_RECEIPT_SCHEMA_VERSION: Final = "openmed.agent.approval_receipt.v2"
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
_KEY_ID_RE = re.compile(r"[a-z][a-z0-9._-]{0,127}")
_LEGACY_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "action_digest",
        "reviewer_role",
        "expires_at",
        "nonce",
        "signature",
    }
)
_TOKEN_FIELDS = _LEGACY_TOKEN_FIELDS | {"key_id", "issued_at"}
_RECEIPT_FIELDS = frozenset({"schema_version", "action_digest", "token_digest", "code"})

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


class ApprovalNotYetValidError(ApprovalTokenError):
    """Raised when issuance is beyond the permitted clock-skew window."""


class ApprovalLifetimeError(ApprovalTokenError):
    """Raised when a signed lifetime exceeds the configured ceiling."""


class ApprovalKeyError(ApprovalTokenError):
    """Raised when a local signing key cannot be resolved safely."""


class ApprovalKeyProvider(Protocol):
    """Resolve current and retiring local keys by non-secret identifiers."""

    def get_key(self, key_id: str) -> bytes | None:
        """Return local key bytes, or ``None`` for an unknown identifier."""


@dataclass(frozen=True, slots=True, repr=False)
class MappingApprovalKeyProvider:
    """Resolve keys from an application-owned in-memory rotation mapping.

    Args:
        keys: Current and retiring keys indexed by non-secret policy labels.
    """

    keys: Mapping[str, bytes]

    def get_key(self, key_id: str) -> bytes | None:
        """Return the selected key; removing its entry ends the overlap."""
        return self.keys.get(key_id)

    def __repr__(self) -> str:
        """Hide the mapping, identifiers and key material in diagnostics."""
        return "MappingApprovalKeyProvider(<redacted>)"


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
        key_id: Non-secret, developer-authored local signing-key label.
        issued_at: Unix issuance timestamp, signed in v2.
    """

    action_digest: str
    reviewer_role: str
    expires_at: int
    nonce: str
    signature: str
    schema_version: str = APPROVAL_TOKEN_SCHEMA_VERSION
    key_id: str | None = None
    issued_at: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version == APPROVAL_TOKEN_SCHEMA_VERSION:
            _validate_key_id(self.key_id)
            _validate_timestamp(self.issued_at, "issued_at")
        elif self.schema_version == LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION:
            if self.key_id is not None or self.issued_at is not None:
                raise ApprovalTokenValidationError("unknown_field", "token")
        else:
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

        payload: dict[str, str | int] = {
            "schema_version": self.schema_version,
            "action_digest": self.action_digest,
            "reviewer_role": self.reviewer_role,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
        }
        if self.schema_version == APPROVAL_TOKEN_SCHEMA_VERSION:
            payload.update(
                key_id=cast(str, self.key_id), issued_at=cast(int, self.issued_at)
            )
        return payload

    def to_dict(self) -> dict[str, str | int]:
        """Return the complete canonical JSON-compatible token."""

        payload = self.signing_payload()
        payload["signature"] = self.signature
        return payload

    def to_json(self) -> str:
        """Serialize the token as compact canonical JSON."""

        return _canonical_json(self.to_dict(), "token")

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, allow_v1: bool = False
    ) -> "ApprovalToken":
        """Restore a token while rejecting omitted or unsigned fields."""

        _validate_compatibility_flag(allow_v1)
        values = _read_exact_mapping(payload, _TOKEN_FIELDS, "token", require_all=False)
        legacy = values.get("schema_version") == LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION
        if legacy and not allow_v1:
            raise ApprovalTokenValidationError(
                "legacy_token_disabled", "schema_version"
            )
        values = _read_exact_mapping(
            values, _LEGACY_TOKEN_FIELDS if legacy else _TOKEN_FIELDS, "token"
        )
        return cls(
            action_digest=cast(str, values["action_digest"]),
            reviewer_role=cast(str, values["reviewer_role"]),
            expires_at=cast(int, values["expires_at"]),
            nonce=cast(str, values["nonce"]),
            signature=cast(str, values["signature"]),
            schema_version=cast(str, values["schema_version"]),
            key_id=cast(str | None, values.get("key_id")),
            issued_at=cast(int | None, values.get("issued_at")),
        )

    @classmethod
    def from_json(
        cls, serialized: str | bytes | bytearray, *, allow_v1: bool = False
    ) -> "ApprovalToken":
        """Restore a token from strict JSON without normalizing claims."""

        return cls.from_dict(_parse_json(serialized, "token"), allow_v1=allow_v1)

    def __repr__(self) -> str:
        """Return a representation that does not expose the bearer token."""

        return "ApprovalToken(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ApprovalReceipt:
    """Value-free proof that one approval token was successfully consumed.

    Args:
        action_digest: Digest of the exact approved action.
        token_digest: Digest of the complete signed token.
        code: Controlled successful-consumption code, always ``approved``.
        schema_version: Codes-and-digests receipt schema version.
    """

    action_digest: str
    token_digest: str
    code: str = "approved"
    schema_version: str = APPROVAL_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPROVAL_RECEIPT_SCHEMA_VERSION:
            raise ApprovalTokenValidationError(
                "unsupported_receipt_schema_version", "schema_version"
            )
        _validate_digest(self.action_digest, "action_digest")
        _validate_digest(self.token_digest, "token_digest")
        if self.code != "approved":
            raise ApprovalTokenValidationError("invalid_receipt_code", "code")

    def to_dict(self) -> dict[str, str | int]:
        """Return only controlled codes and digests for audit storage."""
        return {
            "schema_version": self.schema_version,
            "action_digest": self.action_digest,
            "token_digest": self.token_digest,
            "code": self.code,
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
            token_digest=cast(str, values["token_digest"]),
            code=cast(str, values["code"]),
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
    """Issue signed tokens after the application authenticates a reviewer.

    Args:
        key: Local key provider, or raw bytes for the ``default`` key label.
        key_id: Non-secret label of the current signing key.
        clock: Local integer Unix clock; injectable for offline tests.
        max_lifetime_seconds: Signed lifetime ceiling, from 1 to 86,400 seconds.
        clock_skew_seconds: Edge tolerance, from 0 to 300 seconds.
    """

    def __init__(
        self,
        key: bytes | ApprovalKeyProvider,
        *,
        key_id: str = "default",
        clock: Callable[[], int] = lambda: int(time.time()),
        max_lifetime_seconds: int = DEFAULT_APPROVAL_LIFETIME_SECONDS,
        clock_skew_seconds: int = 0,
    ) -> None:
        self._provider = _validate_provider(key)
        self._key_id = _validate_key_id(key_id)
        self._clock = _validate_clock(clock)
        self._max_lifetime, self._skew = _validate_bounds(
            max_lifetime_seconds, clock_skew_seconds
        )

    def issue(
        self,
        *,
        action_digest: str,
        reviewer_role: str,
        expires_at: int,
        issued_at: int | None = None,
        nonce: str | None = None,
        nonce_source: _NonceSource | None = None,
    ) -> ApprovalToken:
        """Sign exact approval claims with a fresh or explicit nonce.

        ``nonce`` is intended for restoring application-owned issuance state;
        ``nonce_source`` is intended for deterministic tests. Runtime callers
        should normally omit both and use the secure local random source.
        """

        current_time = _read_clock(self._clock)
        issued = current_time if issued_at is None else issued_at
        _validate_timestamp(issued, "issued_at")
        _validate_timestamp(expires_at, "expires_at")
        _validate_window(
            issued, expires_at, current_time, self._max_lifetime, self._skew
        )
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
            key_id=self._key_id,
            issued_at=issued,
        )
        signature = _sign(
            unsigned.signing_payload(), _resolve_key(self._provider, self._key_id)
        )
        return ApprovalToken(
            action_digest=unsigned.action_digest,
            reviewer_role=unsigned.reviewer_role,
            expires_at=unsigned.expires_at,
            nonce=unsigned.nonce,
            signature=signature,
            schema_version=unsigned.schema_version,
            key_id=unsigned.key_id,
            issued_at=unsigned.issued_at,
        )

    def __repr__(self) -> str:
        """Return a representation that never exposes key material."""

        return "ApprovalTokenSigner(<redacted>)"


class ApprovalTokenVerifier:
    """Verify and atomically consume approval tokens before dispatch.

    Args:
        key: Local provider holding current and retiring keys, or raw bytes
            for the ``default`` key label.
        nonce_store: Atomic nonce-digest store shared by approval consumers.
        clock: Local integer Unix clock; injectable for offline tests.
        max_lifetime_seconds: Signed lifetime ceiling, from 1 to 86,400 seconds.
        clock_skew_seconds: Edge tolerance, from 0 to 300 seconds.
        allow_v1: Explicit opt-in to historical five-claim HMAC verification.
        legacy_key_id: One application-selected local key for v1 tokens.
    """

    def __init__(
        self,
        key: bytes | ApprovalKeyProvider,
        nonce_store: ApprovalNonceStore,
        *,
        clock: Callable[[], int] = lambda: int(time.time()),
        max_lifetime_seconds: int = DEFAULT_APPROVAL_LIFETIME_SECONDS,
        clock_skew_seconds: int = 0,
        allow_v1: bool = False,
        legacy_key_id: str = "default",
    ) -> None:
        self._provider = _validate_provider(key)
        self._max_lifetime, self._skew = _validate_bounds(
            max_lifetime_seconds, clock_skew_seconds
        )
        _validate_compatibility_flag(allow_v1)
        self._allow_v1 = allow_v1
        self._legacy_key_id = _validate_key_id(legacy_key_id)
        if not callable(getattr(nonce_store, "claim", None)):
            raise ApprovalTokenValidationError("invalid_nonce_store", "nonce_store")
        self._nonce_store = nonce_store
        self._clock = _validate_clock(clock)

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

        candidate = _coerce_token(token, allow_v1=self._allow_v1)
        _validate_digest(action_digest, "action_digest")
        _validate_reviewer_role(reviewer_role)

        key_id = candidate.key_id or self._legacy_key_id
        expected_signature = _sign(
            candidate.signing_payload(), _resolve_key(self._provider, key_id)
        )
        if not hmac.compare_digest(expected_signature, candidate.signature):
            raise ApprovalSignatureError("invalid_signature", "token")

        current_time = _read_clock(self._clock) if now is None else now
        _validate_timestamp(current_time, "now")
        if candidate.issued_at is None:
            # v1 has no issuance bound: restrict its remaining validity during migration.
            issued = current_time
        else:
            issued = candidate.issued_at
        _validate_window(
            issued,
            candidate.expires_at,
            current_time,
            self._max_lifetime,
            self._skew,
            legacy=candidate.issued_at is None,
        )

        nonce_digest = _sha256(candidate.nonce.encode("ascii"))
        if not _claim_nonce(
            self._nonce_store,
            nonce_digest,
            expires_at=candidate.expires_at + self._skew,
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
            token_digest=_sha256(candidate.to_json().encode("utf-8")),
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
    *,
    allow_v1: bool,
) -> ApprovalToken:
    if type(value) is ApprovalToken:
        return ApprovalToken.from_dict(value.to_dict(), allow_v1=allow_v1)
    if isinstance(value, Mapping):
        return ApprovalToken.from_dict(value, allow_v1=allow_v1)
    if isinstance(value, (str, bytes, bytearray)):
        return ApprovalToken.from_json(value, allow_v1=allow_v1)
    raise ApprovalTokenValidationError("invalid_token", "token")


def _read_exact_mapping(
    payload: Mapping[str, Any],
    expected_fields: frozenset[str],
    location: str,
    *,
    require_all: bool = True,
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
    if require_all and expected_fields - set(values):
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


def _validate_key_id(value: object) -> str:
    if type(value) is not str or _KEY_ID_RE.fullmatch(value) is None:
        raise ApprovalTokenValidationError("invalid_key_id", "key_id")
    return value


def _validate_provider(
    provider: bytes | ApprovalKeyProvider,
) -> bytes | ApprovalKeyProvider:
    if type(provider) is bytes:
        return _validate_key(provider)
    if not callable(getattr(provider, "get_key", None)):
        raise ApprovalTokenValidationError("invalid_key_provider", "key")
    return provider


def _resolve_key(provider: bytes | ApprovalKeyProvider, key_id: str) -> bytes:
    try:
        if type(provider) is bytes:
            key = provider if key_id == "default" else None
        else:
            key = cast(ApprovalKeyProvider, provider).get_key(key_id)
    except (KeyboardInterrupt, SystemExit):
        raise
    except KeyError:
        raise ApprovalKeyError("unknown_key", "key_id") from None
    except BaseException:
        raise ApprovalKeyError("key_provider_unavailable", "key_id") from None
    if key is None:
        raise ApprovalKeyError("unknown_key", "key_id")
    return _validate_key(key)


def _validate_clock(clock: Callable[[], int]) -> Callable[[], int]:
    if not callable(clock):
        raise ApprovalTokenValidationError("invalid_clock", "clock")
    return clock


def _validate_compatibility_flag(allow_v1: bool) -> None:
    if type(allow_v1) is not bool:
        raise ApprovalTokenValidationError("invalid_compatibility_flag", "allow_v1")


def _validate_bounds(lifetime: int, skew: int) -> tuple[int, int]:
    if type(lifetime) is not int or not 1 <= lifetime <= MAX_APPROVAL_LIFETIME_SECONDS:
        raise ApprovalTokenValidationError(
            "invalid_lifetime_limit", "max_lifetime_seconds"
        )
    if type(skew) is not int or not 0 <= skew <= MAX_APPROVAL_CLOCK_SKEW_SECONDS:
        raise ApprovalTokenValidationError("invalid_skew_limit", "clock_skew_seconds")
    return lifetime, skew


def _validate_window(
    issued: int,
    expires: int,
    now: int,
    lifetime: int,
    skew: int,
    *,
    legacy: bool = False,
) -> None:
    _validate_timestamp(expires + skew, "expires_at")
    if not legacy and expires <= issued:
        raise ApprovalTokenValidationError("invalid_lifetime", "expires_at")
    if expires - issued > lifetime:
        raise ApprovalLifetimeError("lifetime_exceeded", "token")
    if issued > now + skew:
        raise ApprovalNotYetValidError("not_yet_valid", "token")
    if now >= expires + skew:
        raise ApprovalExpiredError("expired", "token")


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
    "DEFAULT_APPROVAL_LIFETIME_SECONDS",
    "MAX_APPROVAL_LIFETIME_SECONDS",
    "MAX_APPROVAL_CLOCK_SKEW_SECONDS",
    "LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION",
    "ApprovalKeyError",
    "ApprovalKeyProvider",
    "ApprovalLifetimeError",
    "ApprovalNotYetValidError",
    "MappingApprovalKeyProvider",
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
