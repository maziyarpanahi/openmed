"""Signed, deterministic capability grants for local agent dispatch.

Grant manifests contain developer-authored governance identifiers only. They
must never contain record identifiers, credentials, tool arguments, or
clinical values. Applications retain signing-key custody through an injected
provider, so issuing and verifying grants requires no network access.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

from openmed.agent.identifiers import GovernanceIdError, PolicyId, ToolId

CAPABILITY_GRANT_SCHEMA_VERSION = "openmed.agent.capability_grant.v1"
CAPABILITY_GRANT_SIGNATURE_ALGORITHM = "hmac-sha256"

_SIGNATURE_PREFIX = f"{CAPABILITY_GRANT_SIGNATURE_ALGORITHM}:"
_SIGNATURE_RE = re.compile(r"hmac-sha256:[0-9a-f]{64}")
_KEY_ID_RE = re.compile(r"[a-z][a-z0-9._-]{0,127}")
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE = rf"{_LABEL}(?:\.{_LABEL})+"
_LOCAL_NAME = r"[a-z][a-z0-9-]{0,63}"
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
_RESOURCE_RE = re.compile(rf"resource:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_ACTION_RE = re.compile(rf"action:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_MANIFEST_FIELDS = frozenset(
    {"schema_version", "constraints", "expires_at", "key_id", "signature"}
)
_CONSTRAINT_FIELDS = frozenset({"tool", "resource", "action", "policy_profile"})

_T = TypeVar("_T")


class CapabilityGrantError(ValueError):
    """Base class for value-free capability-grant failures.

    Args:
        code: Stable machine-readable failure code.
        field_name: Optional fixed field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


class CapabilityGrantValidationError(CapabilityGrantError):
    """Raised when a manifest, constraint, request, or key is malformed."""


class CapabilityGrantVerificationError(CapabilityGrantError):
    """Base class for fail-closed verification outcomes."""


class CapabilityGrantRequiredError(CapabilityGrantVerificationError):
    """Raised when dispatch is attempted without a grant manifest."""


class CapabilityGrantKeyError(CapabilityGrantVerificationError):
    """Raised when a signing key is unavailable or invalid."""


class CapabilityGrantSignatureError(CapabilityGrantVerificationError):
    """Raised when a manifest signature cannot be verified."""


class CapabilityGrantExpiredError(CapabilityGrantVerificationError):
    """Raised when a manifest has reached its exclusive expiry time."""


class CapabilityGrantScopeError(CapabilityGrantVerificationError):
    """Raised when no exact signed constraint authorizes a dispatch."""


class CapabilityGrantKeyProvider(Protocol):
    """Resolve local secret key material by a non-secret identifier."""

    def get_key(self, key_id: str) -> bytes:
        """Return key material for ``key_id``."""


@dataclass(frozen=True, slots=True, repr=False)
class StaticCapabilityGrantKeyProvider:
    """In-memory key provider for local configuration and tests."""

    key: bytes
    key_id: str = "default"

    def __post_init__(self) -> None:
        _validate_key_id(self.key_id)
        _validate_key(self.key)

    def get_key(self, key_id: str) -> bytes:
        """Return the configured key only for its configured identifier."""

        if key_id != self.key_id:
            raise KeyError(key_id)
        return self.key

    def __repr__(self) -> str:
        """Return a representation that never exposes key material."""

        return "StaticCapabilityGrantKeyProvider(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class MappingCapabilityGrantKeyProvider:
    """Resolve grant keys from an application-owned local mapping."""

    keys: Mapping[str, bytes]

    def get_key(self, key_id: str) -> bytes:
        """Return key material without exposing the mapping in diagnostics."""

        try:
            return self.keys[key_id]
        except KeyError:
            raise KeyError(key_id) from None

    def __repr__(self) -> str:
        """Return a representation that never exposes key material."""

        return "MappingCapabilityGrantKeyProvider(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class CapabilityGrantConstraint:
    """One exact tool, resource, action, and policy-profile authority tuple."""

    tool: str
    resource: str
    action: str
    policy_profile: str

    def __post_init__(self) -> None:
        _validate_governance_id(self.tool, "tool", ToolId)
        _validate_constraint_id(self.resource, "resource", _RESOURCE_RE)
        _validate_constraint_id(self.action, "action", _ACTION_RE)
        _validate_governance_id(self.policy_profile, "policy_profile", PolicyId)

    def to_dict(self) -> dict[str, str]:
        """Return the canonical JSON-compatible constraint."""

        return {
            "tool": self.tool,
            "resource": self.resource,
            "action": self.action,
            "policy_profile": self.policy_profile,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityGrantConstraint":
        """Restore a constraint while rejecting omitted or unsigned fields."""

        if not isinstance(payload, Mapping):
            raise CapabilityGrantValidationError(
                "invalid_constraint_fields", "constraints"
            )
        try:
            values = dict(payload)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise CapabilityGrantValidationError(
                "invalid_constraint_fields", "constraints"
            ) from None
        if set(values) != _CONSTRAINT_FIELDS:
            raise CapabilityGrantValidationError(
                "invalid_constraint_fields", "constraints"
            )
        return cls(
            tool=cast(str, values["tool"]),
            resource=cast(str, values["resource"]),
            action=cast(str, values["action"]),
            policy_profile=cast(str, values["policy_profile"]),
        )

    def __repr__(self) -> str:
        """Return a value-free representation for safe diagnostics."""

        return "CapabilityGrantConstraint(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class CapabilityGrantRequest:
    """Governance metadata describing one proposed local tool dispatch."""

    tool: str
    resource: str
    action: str
    policy_profile: str

    def __post_init__(self) -> None:
        CapabilityGrantConstraint(
            tool=self.tool,
            resource=self.resource,
            action=self.action,
            policy_profile=self.policy_profile,
        )

    def as_constraint(self) -> CapabilityGrantConstraint:
        """Return the exact constraint required to authorize this request."""

        return CapabilityGrantConstraint(
            tool=self.tool,
            resource=self.resource,
            action=self.action,
            policy_profile=self.policy_profile,
        )

    def __repr__(self) -> str:
        """Return a value-free representation for safe diagnostics."""

        return "CapabilityGrantRequest(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class CapabilityGrantManifest:
    """A signed set of exact capability constraints with an expiry bound."""

    constraints: tuple[CapabilityGrantConstraint, ...]
    expires_at: int
    key_id: str
    signature: str
    schema_version: str = CAPABILITY_GRANT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_GRANT_SCHEMA_VERSION:
            raise CapabilityGrantValidationError(
                "unsupported_schema_version", "schema_version"
            )
        if type(self.constraints) is not tuple or not self.constraints:
            raise CapabilityGrantValidationError(
                "constraints_must_be_non_empty_tuple", "constraints"
            )
        if not all(
            type(item) is CapabilityGrantConstraint for item in self.constraints
        ):
            raise CapabilityGrantValidationError(
                "invalid_constraint_type", "constraints"
            )
        canonical = tuple(sorted(self.constraints, key=_constraint_sort_key))
        if len(set(canonical)) != len(canonical):
            raise CapabilityGrantValidationError("duplicate_constraint", "constraints")
        object.__setattr__(self, "constraints", canonical)
        _validate_timestamp(self.expires_at, "expires_at")
        _validate_key_id(self.key_id)
        if type(self.signature) is not str or (
            self.signature != "" and _SIGNATURE_RE.fullmatch(self.signature) is None
        ):
            raise CapabilityGrantValidationError(
                "invalid_signature_format", "signature"
            )

    def signing_payload(self) -> dict[str, Any]:
        """Return every signed field except the signature itself."""

        return {
            "schema_version": self.schema_version,
            "constraints": [item.to_dict() for item in self.constraints],
            "expires_at": self.expires_at,
            "key_id": self.key_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete canonical JSON-compatible manifest."""

        payload = self.signing_payload()
        payload["signature"] = self.signature
        return payload

    def to_json(self) -> str:
        """Serialize the manifest as deterministic canonical JSON."""

        return _canonical_json(self.to_dict())

    serialize = to_json

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityGrantManifest":
        """Restore a manifest without accepting omitted or unsigned fields."""

        if not isinstance(payload, Mapping):
            raise CapabilityGrantValidationError("invalid_manifest_fields", "manifest")
        try:
            values = dict(payload)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise CapabilityGrantValidationError(
                "invalid_manifest_fields", "manifest"
            ) from None
        if set(values) != _MANIFEST_FIELDS:
            raise CapabilityGrantValidationError("invalid_manifest_fields", "manifest")
        raw_constraints = values["constraints"]
        if not isinstance(raw_constraints, Sequence) or isinstance(
            raw_constraints, (str, bytes, bytearray)
        ):
            raise CapabilityGrantValidationError("invalid_constraints", "constraints")
        constraints = tuple(
            CapabilityGrantConstraint.from_dict(item) for item in raw_constraints
        )
        return cls(
            constraints=constraints,
            expires_at=cast(int, values["expires_at"]),
            key_id=cast(str, values["key_id"]),
            signature=cast(str, values["signature"]),
            schema_version=cast(str, values["schema_version"]),
        )

    @classmethod
    def from_json(cls, serialized: str | bytes) -> "CapabilityGrantManifest":
        """Restore a manifest from JSON using strict signed-field validation."""

        if not isinstance(serialized, (str, bytes)):
            raise CapabilityGrantValidationError("invalid_json", "manifest")
        try:
            payload = json.loads(serialized, object_pairs_hook=_strict_json_object)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise CapabilityGrantValidationError("invalid_json", "manifest") from None
        return cls.from_dict(payload)

    def __repr__(self) -> str:
        """Return a value-free representation for safe diagnostics."""

        return "CapabilityGrantManifest(<redacted>)"


class CapabilityGrantSigner:
    """Create deterministic manifests with application-owned local keys."""

    def __init__(
        self,
        key_provider: CapabilityGrantKeyProvider | Mapping[str, bytes] | bytes,
        *,
        key_id: str = "default",
    ) -> None:
        self.key_provider = key_provider
        self.key_id = _validate_key_id(key_id)

    def issue(
        self,
        constraints: Sequence[CapabilityGrantConstraint],
        *,
        expires_at: int,
    ) -> CapabilityGrantManifest:
        """Sign an exact constraint set with an explicit expiry timestamp."""

        if isinstance(constraints, (str, bytes, bytearray)):
            raise CapabilityGrantValidationError("invalid_constraints", "constraints")
        try:
            constraint_tuple = tuple(constraints)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise CapabilityGrantValidationError(
                "invalid_constraints", "constraints"
            ) from None
        unsigned = CapabilityGrantManifest(
            constraints=constraint_tuple,
            expires_at=expires_at,
            key_id=self.key_id,
            signature="",
        )
        signature = _sign(
            unsigned.signing_payload(), _resolve_key(self.key_provider, self.key_id)
        )
        return CapabilityGrantManifest(
            constraints=unsigned.constraints,
            expires_at=unsigned.expires_at,
            key_id=unsigned.key_id,
            signature=signature,
            schema_version=unsigned.schema_version,
        )


class CapabilityGrantVerifier:
    """Verify signature, expiry, and exact scope before local dispatch."""

    def __init__(
        self,
        key_provider: CapabilityGrantKeyProvider | Mapping[str, bytes] | bytes,
        *,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self.key_provider = key_provider
        self.clock = clock

    def verify(
        self,
        manifest: CapabilityGrantManifest | Mapping[str, Any] | str | bytes | None,
        request: CapabilityGrantRequest,
        *,
        now: int | None = None,
    ) -> CapabilityGrantManifest:
        """Return a verified manifest or raise a value-free denial error.

        Verification checks the signature before consulting expiry or scope so
        modified fields cannot influence an authorization decision.
        """

        candidate = _coerce_manifest(manifest)
        if type(request) is not CapabilityGrantRequest:
            raise CapabilityGrantValidationError("invalid_request", "request")
        key = _resolve_key(self.key_provider, candidate.key_id)
        expected = _sign(candidate.signing_payload(), key)
        if not hmac.compare_digest(expected, candidate.signature):
            raise CapabilityGrantSignatureError("invalid_signature", "manifest")
        current_time = self.clock() if now is None else now
        _validate_timestamp(current_time, "now")
        if current_time >= candidate.expires_at:
            raise CapabilityGrantExpiredError("expired", "manifest")
        if request.as_constraint() not in candidate.constraints:
            raise CapabilityGrantScopeError("scope_mismatch", "request")
        return candidate


def dispatch_with_capability_grant(
    manifest: CapabilityGrantManifest | Mapping[str, Any] | str | bytes | None,
    request: CapabilityGrantRequest,
    verifier: CapabilityGrantVerifier,
    dispatch: Callable[[], _T],
    *,
    now: int | None = None,
) -> _T:
    """Verify a grant and only then invoke a zero-argument dispatch callback."""

    if not callable(dispatch):
        raise CapabilityGrantValidationError("invalid_dispatch", "dispatch")
    verifier.verify(manifest, request, now=now)
    return dispatch()


def _constraint_sort_key(
    constraint: CapabilityGrantConstraint,
) -> tuple[str, str, str, str]:
    return (
        constraint.tool,
        constraint.resource,
        constraint.action,
        constraint.policy_profile,
    )


def _coerce_manifest(
    value: CapabilityGrantManifest | Mapping[str, Any] | str | bytes | None,
) -> CapabilityGrantManifest:
    if value is None:
        raise CapabilityGrantRequiredError("missing_grant", "manifest")
    if type(value) is CapabilityGrantManifest:
        return value
    if isinstance(value, Mapping):
        return CapabilityGrantManifest.from_dict(value)
    if isinstance(value, (str, bytes)):
        return CapabilityGrantManifest.from_json(value)
    raise CapabilityGrantValidationError("invalid_manifest", "manifest")


def _validate_governance_id(
    value: object,
    field_name: str,
    identifier_type: type[ToolId] | type[PolicyId],
) -> str:
    try:
        identifier_type.parse(value)
    except GovernanceIdError as exc:
        raise CapabilityGrantValidationError(
            "invalid_governance_identifier", field_name
        ) from exc
    return cast(str, value)


def _validate_constraint_id(
    value: object, field_name: str, pattern: re.Pattern[str]
) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise CapabilityGrantValidationError(
            "invalid_governance_identifier", field_name
        )
    return value


def _validate_timestamp(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise CapabilityGrantValidationError("invalid_timestamp", field_name)
    return value


def _validate_key_id(value: object) -> str:
    if type(value) is not str or _KEY_ID_RE.fullmatch(value) is None:
        raise CapabilityGrantValidationError("invalid_key_id", "key_id")
    return value


def _validate_key(value: object) -> bytes:
    if type(value) is not bytes or len(value) < 32:
        raise CapabilityGrantKeyError("invalid_key", "key")
    return value


def _resolve_key(
    provider: CapabilityGrantKeyProvider | Mapping[str, bytes] | bytes,
    key_id: str,
) -> bytes:
    try:
        if type(provider) is bytes:
            key = provider
        elif isinstance(provider, Mapping):
            key = provider[key_id]
        else:
            key = provider.get_key(key_id)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise CapabilityGrantKeyError("key_unavailable", "key_id") from None
    return _validate_key(key)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CapabilityGrantValidationError("duplicate_field", "manifest")
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
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CapabilityGrantValidationError("invalid_manifest", "manifest") from exc


def _sign(payload: Mapping[str, Any], key: bytes) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    digest = hmac.new(key, encoded, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest}"


__all__ = [
    "CAPABILITY_GRANT_SCHEMA_VERSION",
    "CAPABILITY_GRANT_SIGNATURE_ALGORITHM",
    "CapabilityGrantConstraint",
    "CapabilityGrantError",
    "CapabilityGrantExpiredError",
    "CapabilityGrantKeyError",
    "CapabilityGrantKeyProvider",
    "CapabilityGrantManifest",
    "CapabilityGrantRequest",
    "CapabilityGrantRequiredError",
    "CapabilityGrantScopeError",
    "CapabilityGrantSignatureError",
    "CapabilityGrantSigner",
    "CapabilityGrantValidationError",
    "CapabilityGrantVerificationError",
    "CapabilityGrantVerifier",
    "MappingCapabilityGrantKeyProvider",
    "StaticCapabilityGrantKeyProvider",
    "dispatch_with_capability_grant",
]
