"""Signed, non-amplifying permission delegation for local clinical agents.

Delegation grants contain developer-authored governance identifiers only. Raw
record identifiers, credentials, tool arguments, and clinical values must stay
outside this layer. Derivation and verification are deterministic and local.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, cast

from openmed.agent.identifiers import GovernanceIdError, PurposeId

from .grants import CapabilityGrantConstraint, CapabilityGrantError

DELEGATION_GRANT_SCHEMA_VERSION: Final = "openmed.agent.delegation_grant.v1"
DELEGATION_AUDIT_SCHEMA_VERSION: Final = "openmed.agent.delegation_audit.v1"
DELEGATION_SIGNATURE_ALGORITHM: Final = "hmac-sha256"
DELEGATION_DIGEST_ALGORITHM: Final = "sha256"

_SIGNATURE_PREFIX = f"{DELEGATION_SIGNATURE_ALGORITHM}:"
_SIGNATURE_RE = re.compile(r"hmac-sha256:[0-9a-f]{64}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_KEY_ID_RE = re.compile(r"[a-z][a-z0-9._-]{0,127}")
_REASON_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE = rf"{_LABEL}(?:\.{_LABEL})+"
_LOCAL_NAME = r"[a-z][a-z0-9-]{0,63}"
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
_AGENT_RE = re.compile(rf"agent:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_DATA_CLASS_RE = re.compile(rf"data:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_GRANT_FIELDS = frozenset(
    {
        "schema_version",
        "principals",
        "scope",
        "expires_at",
        "remaining_depth",
        "parent_digest",
        "key_id",
        "signature",
    }
)
_SCOPE_FIELDS = frozenset({"capabilities", "data_classes", "purposes"})


class DelegationError(ValueError):
    """Base class for value-free delegation failures."""

    def __init__(
        self,
        code: str,
        *,
        parent_grant_digest: str | None = None,
        child_grant_digest: str | None = None,
    ) -> None:
        self.code = code
        self.audit_record = DelegationAuditRecord(
            reason_code=code,
            parent_grant_digest=parent_grant_digest,
            child_grant_digest=child_grant_digest,
        )
        super().__init__(code)


class DelegationValidationError(DelegationError):
    """Raised when delegation metadata or key material is malformed."""


class DelegationVerificationError(DelegationError):
    """Base class for fail-closed delegation verification outcomes."""


class DelegationSignatureError(DelegationVerificationError):
    """Raised when a delegation signature is invalid."""


class DelegationExpiredError(DelegationVerificationError):
    """Raised when a parent or child grant is not active."""


class DelegationScopeError(DelegationVerificationError):
    """Raised when delegation would have no authorized scope or is widened."""


class DelegationDepthError(DelegationVerificationError):
    """Raised when delegation exceeds the parent's remaining depth."""


class DelegationCycleError(DelegationVerificationError):
    """Raised when an agent principal repeats in a delegation chain."""


class DelegationChainError(DelegationVerificationError):
    """Raised when a child does not bind exactly to its parent grant."""


class DelegationKeyProvider(Protocol):
    """Resolve local signing key material by a non-secret key identifier."""

    def get_key(self, key_id: str) -> bytes:
        """Return local key material for ``key_id``."""


@dataclass(frozen=True, slots=True, repr=False)
class DelegationScope:
    """Canonical tools, data classes, and purposes available to an agent."""

    capabilities: tuple[CapabilityGrantConstraint, ...]
    data_classes: tuple[str, ...]
    purposes: tuple[str, ...]

    def __post_init__(self) -> None:
        capabilities = _canonical_capabilities(self.capabilities)
        data_classes = _canonical_identifiers(
            self.data_classes, "invalid_data_classes", _DATA_CLASS_RE
        )
        purposes = _canonical_purposes(self.purposes)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "data_classes", data_classes)
        object.__setattr__(self, "purposes", purposes)

    def intersection(self, requested: "DelegationScope") -> "DelegationScope":
        """Return the exact intersection with a requested task scope."""

        if type(requested) is not DelegationScope:
            raise DelegationValidationError("invalid_requested_scope")
        capabilities = tuple(
            sorted(
                set(self.capabilities).intersection(requested.capabilities),
                key=_capability_sort_key,
            )
        )
        data_classes = tuple(
            sorted(set(self.data_classes) & set(requested.data_classes))
        )
        purposes = tuple(sorted(set(self.purposes) & set(requested.purposes)))
        if not capabilities or not data_classes or not purposes:
            raise DelegationScopeError("empty_scope_intersection")
        return DelegationScope(capabilities, data_classes, purposes)

    def is_subset_of(self, other: "DelegationScope") -> bool:
        """Return whether every authority dimension is contained by ``other``."""

        return (
            set(self.capabilities).issubset(other.capabilities)
            and set(self.data_classes).issubset(other.data_classes)
            and set(self.purposes).issubset(other.purposes)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible scope metadata."""

        return {
            "capabilities": [item.to_dict() for item in self.capabilities],
            "data_classes": list(self.data_classes),
            "purposes": list(self.purposes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DelegationScope":
        """Restore a scope while rejecting omitted or extra fields."""

        values = _strict_mapping(payload, _SCOPE_FIELDS, "invalid_scope")
        raw_capabilities = _strict_sequence(values["capabilities"], "invalid_scope")
        raw_data_classes = _strict_sequence(values["data_classes"], "invalid_scope")
        raw_purposes = _strict_sequence(values["purposes"], "invalid_scope")
        try:
            capabilities = tuple(
                CapabilityGrantConstraint.from_dict(item) for item in raw_capabilities
            )
        except CapabilityGrantError:
            raise DelegationValidationError("invalid_scope") from None
        return cls(
            capabilities=capabilities,
            data_classes=tuple(raw_data_classes),
            purposes=tuple(raw_purposes),
        )

    def __repr__(self) -> str:
        """Return a representation that does not expose scope metadata."""

        return "DelegationScope(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class DelegationGrant:
    """One signed authority state in a non-amplifying delegation chain."""

    principals: tuple[str, ...]
    scope: DelegationScope
    expires_at: int
    remaining_depth: int
    parent_digest: str | None
    key_id: str
    signature: str
    schema_version: str = DELEGATION_GRANT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DELEGATION_GRANT_SCHEMA_VERSION:
            raise DelegationValidationError("unsupported_schema_version")
        principals = _canonical_principal_chain(self.principals)
        object.__setattr__(self, "principals", principals)
        if type(self.scope) is not DelegationScope:
            raise DelegationValidationError("invalid_scope")
        _validate_timestamp(self.expires_at)
        _validate_depth(self.remaining_depth)
        if len(principals) == 1:
            if self.parent_digest is not None:
                raise DelegationChainError("unexpected_parent_digest")
        elif not _is_digest(self.parent_digest):
            raise DelegationChainError("missing_parent_digest")
        _validate_key_id(self.key_id)
        if type(self.signature) is not str or (
            self.signature != "" and _SIGNATURE_RE.fullmatch(self.signature) is None
        ):
            raise DelegationValidationError("invalid_signature_format")

    @property
    def principal(self) -> str:
        """Return the agent principal that receives this grant."""

        return self.principals[-1]

    def signing_payload(self) -> dict[str, Any]:
        """Return every signed field except the signature."""

        return {
            "schema_version": self.schema_version,
            "principals": list(self.principals),
            "scope": self.scope.to_dict(),
            "expires_at": self.expires_at,
            "remaining_depth": self.remaining_depth,
            "parent_digest": self.parent_digest,
            "key_id": self.key_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete canonical JSON-compatible grant."""

        payload = self.signing_payload()
        payload["signature"] = self.signature
        return payload

    def to_json(self) -> str:
        """Serialize the grant as deterministic canonical JSON."""

        return _canonical_json(self.to_dict())

    serialize = to_json

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DelegationGrant":
        """Restore a grant while rejecting omitted or unsigned fields."""

        values = _strict_mapping(payload, _GRANT_FIELDS, "invalid_grant")
        principals = _strict_sequence(values["principals"], "invalid_grant")
        return cls(
            principals=tuple(principals),
            scope=DelegationScope.from_dict(values["scope"]),
            expires_at=cast(int, values["expires_at"]),
            remaining_depth=cast(int, values["remaining_depth"]),
            parent_digest=cast(str | None, values["parent_digest"]),
            key_id=cast(str, values["key_id"]),
            signature=cast(str, values["signature"]),
            schema_version=cast(str, values["schema_version"]),
        )

    @classmethod
    def from_json(cls, serialized: str | bytes) -> "DelegationGrant":
        """Restore a grant from strict canonicalizable JSON."""

        if not isinstance(serialized, (str, bytes)):
            raise DelegationValidationError("invalid_json")
        try:
            payload = json.loads(serialized, object_pairs_hook=_strict_json_object)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise DelegationValidationError("invalid_json") from None
        return cls.from_dict(payload)

    def digest(self) -> str:
        """Return a content digest suitable for value-free audit linkage."""

        if not self.signature:
            raise DelegationValidationError("unsigned_grant")
        value = hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
        return f"{DELEGATION_DIGEST_ALGORITHM}:{value}"

    def __repr__(self) -> str:
        """Return a representation that does not expose grant metadata."""

        return "DelegationGrant(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class DelegationRequest:
    """Requested child principal, task scope, validity, and onward depth."""

    principal: str
    scope: DelegationScope
    expires_at: int
    remaining_depth: int

    def __post_init__(self) -> None:
        _validate_principal(self.principal)
        if type(self.scope) is not DelegationScope:
            raise DelegationValidationError("invalid_requested_scope")
        _validate_timestamp(self.expires_at)
        _validate_depth(self.remaining_depth)

    def __repr__(self) -> str:
        """Return a representation that does not expose request metadata."""

        return "DelegationRequest(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class DelegationAuditRecord:
    """Content-free audit evidence containing only grant digests and a reason."""

    reason_code: str
    parent_grant_digest: str | None = None
    child_grant_digest: str | None = None
    schema_version: str = DELEGATION_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DELEGATION_AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported_audit_schema_version")
        if (
            type(self.reason_code) is not str
            or _REASON_RE.fullmatch(self.reason_code) is None
        ):
            raise ValueError("invalid_reason_code")
        for digest in (self.parent_grant_digest, self.child_grant_digest):
            if digest is not None and not _is_digest(digest):
                raise ValueError("invalid_grant_digest")

    def to_dict(self) -> dict[str, str | None]:
        """Return the fixed audit schema without scope or principal values."""

        return {
            "schema_version": self.schema_version,
            "reason_code": self.reason_code,
            "parent_grant_digest": self.parent_grant_digest,
            "child_grant_digest": self.child_grant_digest,
        }

    def to_json(self) -> str:
        """Serialize the audit record deterministically."""

        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        """Return a representation that does not expose linkable digests."""

        return "DelegationAuditRecord(<redacted>)"


@dataclass(frozen=True, slots=True)
class DelegationDecision:
    """A derived signed child grant and its content-free audit record."""

    grant: DelegationGrant
    audit_record: DelegationAuditRecord


class DelegationGrantVerifier:
    """Verify signatures and non-amplifying parent-child relationships."""

    def __init__(
        self,
        key_provider: DelegationKeyProvider | Mapping[str, bytes] | bytes,
        *,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        if not callable(clock):
            raise DelegationValidationError("invalid_clock")
        self.key_provider = key_provider
        self.clock = clock

    def verify(
        self,
        grant: DelegationGrant | Mapping[str, Any] | str | bytes,
        *,
        now: int | None = None,
    ) -> DelegationGrant:
        """Verify one signed and active grant."""

        candidate = _coerce_grant(grant)
        parent_digest = _safe_digest(candidate)
        expected = _sign(
            candidate.signing_payload(),
            _resolve_key(self.key_provider, candidate.key_id),
        )
        if not hmac.compare_digest(expected, candidate.signature):
            raise DelegationSignatureError(
                "invalid_signature", parent_grant_digest=parent_digest
            )
        current_time = self._current_time(now, parent_digest)
        if current_time >= candidate.expires_at:
            raise DelegationExpiredError("expired", parent_grant_digest=parent_digest)
        return candidate

    def verify_link(
        self,
        parent: DelegationGrant,
        child: DelegationGrant,
        *,
        now: int | None = None,
    ) -> DelegationGrant:
        """Verify that a signed child cannot amplify its signed parent."""

        parent = self.verify(parent, now=now)
        child = self.verify(child, now=now)
        parent_digest = parent.digest()
        child_digest = child.digest()
        error_digests = {
            "parent_grant_digest": parent_digest,
            "child_grant_digest": child_digest,
        }
        if child.parent_digest != parent_digest:
            raise DelegationChainError("parent_mismatch", **error_digests)
        if child.principals != parent.principals + (child.principal,):
            raise DelegationChainError("chain_mismatch", **error_digests)
        if child.principal in parent.principals:
            raise DelegationCycleError("cyclic_delegation", **error_digests)
        if not child.scope.is_subset_of(parent.scope):
            raise DelegationScopeError("scope_amplified", **error_digests)
        if child.expires_at > parent.expires_at:
            raise DelegationExpiredError("validity_amplified", **error_digests)
        if parent.remaining_depth == 0:
            raise DelegationDepthError("depth_exhausted", **error_digests)
        if child.remaining_depth > parent.remaining_depth - 1:
            raise DelegationDepthError("depth_amplified", **error_digests)
        return child

    def verify_chain(
        self,
        grants: Sequence[DelegationGrant],
        *,
        now: int | None = None,
    ) -> DelegationGrant:
        """Verify a complete root-to-leaf chain and return the active leaf."""

        if isinstance(grants, (str, bytes, bytearray)):
            raise DelegationValidationError("invalid_chain")
        try:
            chain = tuple(grants)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise DelegationValidationError("invalid_chain") from None
        if not chain or not all(type(item) is DelegationGrant for item in chain):
            raise DelegationValidationError("invalid_chain")
        self.verify(chain[0], now=now)
        for parent, child in zip(chain, chain[1:]):
            self.verify_link(parent, child, now=now)
        return chain[-1]

    def _current_time(self, now: int | None, digest: str | None) -> int:
        if now is None:
            try:
                current_time = self.clock()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                raise DelegationValidationError(
                    "clock_unavailable", parent_grant_digest=digest
                ) from None
        else:
            current_time = now
        _validate_timestamp(current_time)
        return current_time


class DelegationGrantSigner:
    """Issue roots and derive signed children with local key material."""

    def __init__(
        self,
        key_provider: DelegationKeyProvider | Mapping[str, bytes] | bytes,
        *,
        key_id: str = "default",
    ) -> None:
        self.key_provider = key_provider
        self.key_id = _validate_key_id(key_id)

    def issue_root(
        self,
        *,
        principal: str,
        scope: DelegationScope,
        expires_at: int,
        remaining_depth: int,
    ) -> DelegationGrant:
        """Issue an application-approved root authority for later delegation."""

        return self._sign_grant(
            principals=(principal,),
            scope=scope,
            expires_at=expires_at,
            remaining_depth=remaining_depth,
            parent_digest=None,
        )

    def derive_child(
        self,
        parent: DelegationGrant,
        request: DelegationRequest,
        verifier: DelegationGrantVerifier,
        *,
        now: int | None = None,
    ) -> DelegationDecision:
        """Derive a child as the intersection of active parent and task scope."""

        if type(request) is not DelegationRequest:
            raise DelegationValidationError("invalid_request")
        parent = verifier.verify(parent, now=now)
        parent_digest = parent.digest()
        if request.principal in parent.principals:
            raise DelegationCycleError(
                "cyclic_delegation", parent_grant_digest=parent_digest
            )
        if parent.remaining_depth == 0:
            raise DelegationDepthError(
                "depth_exhausted", parent_grant_digest=parent_digest
            )
        try:
            scope = parent.scope.intersection(request.scope)
        except DelegationScopeError as error:
            raise DelegationScopeError(
                error.code, parent_grant_digest=parent_digest
            ) from None
        expires_at = min(parent.expires_at, request.expires_at)
        current_time = verifier._current_time(now, parent_digest)
        if expires_at <= current_time:
            raise DelegationExpiredError("expired", parent_grant_digest=parent_digest)
        remaining_depth = min(
            request.remaining_depth,
            parent.remaining_depth - 1,
        )
        child = self._sign_grant(
            principals=parent.principals + (request.principal,),
            scope=scope,
            expires_at=expires_at,
            remaining_depth=remaining_depth,
            parent_digest=parent_digest,
        )
        verifier.verify_link(parent, child, now=current_time)
        audit_record = DelegationAuditRecord(
            reason_code="delegation_derived",
            parent_grant_digest=parent_digest,
            child_grant_digest=child.digest(),
        )
        return DelegationDecision(grant=child, audit_record=audit_record)

    def _sign_grant(
        self,
        *,
        principals: tuple[str, ...],
        scope: DelegationScope,
        expires_at: int,
        remaining_depth: int,
        parent_digest: str | None,
    ) -> DelegationGrant:
        unsigned = DelegationGrant(
            principals=principals,
            scope=scope,
            expires_at=expires_at,
            remaining_depth=remaining_depth,
            parent_digest=parent_digest,
            key_id=self.key_id,
            signature="",
        )
        signature = _sign(
            unsigned.signing_payload(),
            _resolve_key(self.key_provider, self.key_id),
        )
        return DelegationGrant(
            principals=unsigned.principals,
            scope=unsigned.scope,
            expires_at=unsigned.expires_at,
            remaining_depth=unsigned.remaining_depth,
            parent_digest=unsigned.parent_digest,
            key_id=unsigned.key_id,
            signature=signature,
        )


def _canonical_principal_chain(values: object) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise DelegationValidationError("invalid_principal_chain")
    for value in values:
        _validate_principal(value)
    if len(set(values)) != len(values):
        raise DelegationCycleError("cyclic_delegation")
    return cast(tuple[str, ...], values)


def _canonical_capabilities(values: object) -> tuple[CapabilityGrantConstraint, ...]:
    if type(values) is not tuple or not values:
        raise DelegationValidationError("invalid_capabilities")
    if not all(type(value) is CapabilityGrantConstraint for value in values):
        raise DelegationValidationError("invalid_capabilities")
    typed = cast(tuple[CapabilityGrantConstraint, ...], values)
    canonical = tuple(sorted(typed, key=_capability_sort_key))
    if len(set(canonical)) != len(canonical):
        raise DelegationValidationError("duplicate_capability")
    return canonical


def _capability_sort_key(
    value: CapabilityGrantConstraint,
) -> tuple[str, str, str, str]:
    return (value.tool, value.resource, value.action, value.policy_profile)


def _canonical_identifiers(
    values: object, code: str, pattern: re.Pattern[str]
) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise DelegationValidationError(code)
    for value in values:
        if type(value) is not str or pattern.fullmatch(value) is None:
            raise DelegationValidationError(code)
    canonical = tuple(sorted(values))
    if len(set(canonical)) != len(canonical):
        raise DelegationValidationError(code)
    return canonical


def _canonical_purposes(values: object) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise DelegationValidationError("invalid_purposes")
    for value in values:
        try:
            PurposeId.parse(value)
        except GovernanceIdError as exc:
            raise DelegationValidationError("invalid_purposes") from exc
    canonical = tuple(sorted(values))
    if len(set(canonical)) != len(canonical):
        raise DelegationValidationError("invalid_purposes")
    return cast(tuple[str, ...], canonical)


def _validate_principal(value: object) -> str:
    if type(value) is not str or _AGENT_RE.fullmatch(value) is None:
        raise DelegationValidationError("invalid_principal")
    return value


def _validate_timestamp(value: object) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise DelegationValidationError("invalid_timestamp")
    return value


def _validate_depth(value: object) -> int:
    if type(value) is not int or value < 0 or value > 255:
        raise DelegationValidationError("invalid_depth")
    return value


def _validate_key_id(value: object) -> str:
    if type(value) is not str or _KEY_ID_RE.fullmatch(value) is None:
        raise DelegationValidationError("invalid_key_id")
    return value


def _resolve_key(
    provider: DelegationKeyProvider | Mapping[str, bytes] | bytes,
    key_id: str,
) -> bytes:
    try:
        if isinstance(provider, bytes):
            key = provider
        elif isinstance(provider, Mapping):
            key = provider[key_id]
        else:
            key = provider.get_key(key_id)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise DelegationValidationError("key_unavailable") from None
    if type(key) is not bytes or len(key) < 32:
        raise DelegationValidationError("invalid_key")
    return key


def _coerce_grant(
    value: DelegationGrant | Mapping[str, Any] | str | bytes,
) -> DelegationGrant:
    if type(value) is DelegationGrant:
        return value
    if isinstance(value, Mapping):
        return DelegationGrant.from_dict(value)
    if isinstance(value, (str, bytes)):
        return DelegationGrant.from_json(value)
    raise DelegationValidationError("invalid_grant")


def _strict_mapping(
    payload: object, fields: frozenset[str], code: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DelegationValidationError(code)
    try:
        values = dict(payload)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise DelegationValidationError(code) from None
    if set(values) != fields:
        raise DelegationValidationError(code)
    return values


def _strict_sequence(value: object, code: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DelegationValidationError(code)
    try:
        return tuple(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise DelegationValidationError(code) from None


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DelegationValidationError("duplicate_field")
        result[key] = value
    return result


def _canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeError):
        raise DelegationValidationError("invalid_grant") from None


def _sign(payload: Mapping[str, Any], key: bytes) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    value = hmac.new(key, encoded, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{value}"


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def _safe_digest(grant: DelegationGrant) -> str | None:
    try:
        return grant.digest()
    except DelegationError:
        return None


__all__ = [
    "DELEGATION_AUDIT_SCHEMA_VERSION",
    "DELEGATION_DIGEST_ALGORITHM",
    "DELEGATION_GRANT_SCHEMA_VERSION",
    "DELEGATION_SIGNATURE_ALGORITHM",
    "DelegationAuditRecord",
    "DelegationChainError",
    "DelegationCycleError",
    "DelegationDecision",
    "DelegationDepthError",
    "DelegationError",
    "DelegationExpiredError",
    "DelegationGrant",
    "DelegationGrantSigner",
    "DelegationGrantVerifier",
    "DelegationKeyProvider",
    "DelegationRequest",
    "DelegationScope",
    "DelegationScopeError",
    "DelegationSignatureError",
    "DelegationValidationError",
    "DelegationVerificationError",
]
