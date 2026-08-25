"""Signed, single-use consent receipts for state-changing MCP calls.

Consent receipts contain policy identifiers and a digest of the canonical tool
arguments.  They never retain the arguments themselves.  The issuer and
verifier deliberately accept injected clocks and key providers so applications
can keep key custody outside OpenMed and test expiry without wall-clock sleeps.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from numbers import Real
from threading import RLock
from typing import Any, Protocol

CONSENT_RECEIPT_SCHEMA_VERSION = "openmed.mcp.consent_receipt.v1"
CONSENT_RECEIPT_EVENT = "mcp.consent_receipt"
DEFAULT_CONSENT_POLICY_VERSION = "openmed.mcp.consent.v1"
DEFAULT_CONSENT_RESOURCE = "openmed://mcp"
DEFAULT_CONSENT_SCOPE = "mcp:state-changing"
_DIGEST_PREFIX = "sha256:"
_SIGNATURE_PREFIX = "hmac-sha256:"
_MISSING = object()

logger = logging.getLogger(__name__)


class ConsentDecision(str, Enum):
    """Decision carried by a consent receipt."""

    ALLOW = "allow"
    DENY = "deny"


class ConsentReceiptError(ValueError):
    """Base class for receipt construction and policy errors."""


class ConsentReceiptValidationError(ConsentReceiptError):
    """Raised when a receipt or argument set is malformed."""


class ConsentReceiptVerificationError(ConsentReceiptError):
    """Base class for fail-closed receipt verification errors."""


class ConsentReceiptKeyError(ConsentReceiptVerificationError):
    """Raised when a configured key provider cannot resolve a key."""


class ConsentReceiptSignatureError(ConsentReceiptVerificationError):
    """Raised when a receipt signature does not verify."""


class ConsentReceiptBindingError(ConsentReceiptVerificationError):
    """Raised when a receipt is bound to a different request."""


class ConsentReceiptExpiredError(ConsentReceiptVerificationError):
    """Raised when a receipt is no longer valid at the injected clock time."""


class ConsentReceiptNotYetValidError(ConsentReceiptVerificationError):
    """Raised when a receipt's issued time is in the future."""


class ConsentReceiptReplayError(ConsentReceiptVerificationError):
    """Raised when a valid receipt has already been consumed."""


class ConsentReceiptDeniedError(ConsentReceiptVerificationError):
    """Raised when a receipt explicitly carries a deny decision."""


class ConsentReceiptRequiredError(ConsentReceiptVerificationError):
    """Raised when a configured policy requires a receipt but none was sent."""


@dataclass(frozen=True, slots=True)
class ConsentReceiptVerificationResult:
    """Content-free outcome from non-throwing consent receipt verification."""

    verified: bool
    code: str
    receipt: ConsentReceipt | None = None


# Descriptive aliases for callers that prefer shorter exception names.
ConsentReceiptExpired = ConsentReceiptExpiredError
ConsentReceiptReplay = ConsentReceiptReplayError
ConsentReceiptRequired = ConsentReceiptRequiredError


class ConsentKeyProvider(Protocol):
    """Resolve an opaque signing key by its non-secret key identifier."""

    def get_key(self, key_id: str) -> bytes | str:
        """Return the key material for *key_id*."""


@dataclass(frozen=True, slots=True)
class StaticConsentKeyProvider:
    """Small in-memory key provider intended for local configuration and tests."""

    key: bytes | str
    key_id: str = "default"

    def __post_init__(self) -> None:
        _validate_identifier(self.key_id, "key_id")
        _key_bytes(self.key)

    def get_key(self, key_id: str) -> bytes | str:
        """Return the configured key only for its configured identifier."""

        if key_id != self.key_id:
            raise KeyError(key_id)
        return self.key


@dataclass(frozen=True, slots=True)
class MappingConsentKeyProvider:
    """Resolve receipt keys from an application-owned mapping."""

    keys: Mapping[str, bytes | str]

    def get_key(self, key_id: str) -> bytes | str:
        """Return a key without exposing the mapping in a receipt or audit field."""

        try:
            return self.keys[key_id]
        except KeyError:
            raise KeyError(key_id) from None


def canonical_argument_digest(arguments: Any) -> str:
    """Return a stable, content-free digest for JSON-compatible tool arguments.

    The canonical bytes use sorted keys, compact separators, UTF-8-safe ASCII
    escaping, and reject non-finite numbers.  Only the resulting SHA-256 digest
    is stored in a receipt or audit record.
    """

    try:
        encoded = _canonical_json(arguments).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConsentReceiptValidationError(
            "tool arguments must be JSON-compatible and finite"
        ) from exc
    return f"{_DIGEST_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


# Common shorter spelling retained as a public alias.
argument_digest = canonical_argument_digest


@dataclass(frozen=True, slots=True)
class ConsentReceipt:
    """A signed, one-time authorization for one exact MCP request.

    ``argument_digest`` is computed from the canonical JSON argument object;
    raw tool arguments, bearer values, and clinical text are intentionally not
    fields on this type.
    """

    receipt_id: str
    client: str
    tool: str
    resource: str
    scope: str
    decision: ConsentDecision | str
    issued_at: float
    expires_at: float
    argument_digest: str
    key_id: str
    signature: str
    schema_version: str = CONSENT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize and validate the receipt's signed metadata."""

        for field_name in (
            "receipt_id",
            "client",
            "tool",
            "resource",
            "scope",
            "key_id",
            "schema_version",
        ):
            _validate_identifier(getattr(self, field_name), field_name)
        if self.schema_version != CONSENT_RECEIPT_SCHEMA_VERSION:
            raise ConsentReceiptValidationError("unsupported consent receipt schema")
        try:
            decision = ConsentDecision(self.decision)
        except (TypeError, ValueError) as exc:
            raise ConsentReceiptValidationError(
                "unsupported consent receipt decision"
            ) from exc
        object.__setattr__(self, "decision", decision)
        issued_at = _timestamp(self.issued_at, "issued_at")
        expires_at = _timestamp(self.expires_at, "expires_at")
        if expires_at <= issued_at:
            raise ConsentReceiptValidationError("receipt expiry must be after issuance")
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "expires_at", expires_at)
        _validate_digest(self.argument_digest)
        if not isinstance(self.signature, str):
            raise ConsentReceiptValidationError("receipt signature must be a string")
        if self.signature and not self.signature.startswith(_SIGNATURE_PREFIX):
            raise ConsentReceiptValidationError(
                "receipt signature has an invalid format"
            )

    @property
    def canonical_argument_digest(self) -> str:
        """Return the digest under the terminology used by the policy spec."""

        return self.argument_digest

    @property
    def client_id(self) -> str:
        """Return the client identifier alias."""

        return self.client

    @property
    def tool_name(self) -> str:
        """Return the tool-name alias."""

        return self.tool

    @property
    def resource_id(self) -> str:
        """Return the resource identifier alias."""

        return self.resource

    @property
    def expiry(self) -> float:
        """Return the expiry timestamp alias."""

        return self.expires_at

    def signing_payload(self) -> dict[str, Any]:
        """Return the signed payload without the signature itself."""

        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "client": self.client,
            "tool": self.tool,
            "resource": self.resource,
            "scope": self.scope,
            "decision": self.decision.value,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "argument_digest": self.argument_digest,
            "key_id": self.key_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible receipt with no raw request content."""

        payload = self.signing_payload()
        payload["signature"] = self.signature
        return payload

    # ``as_dict`` is convenient for callers that use the project's other
    # typed artifacts, while ``to_dict`` remains the canonical method.
    as_dict = to_dict

    def to_json(self) -> str:
        """Serialize the receipt with deterministic canonical JSON."""

        return _canonical_json(self.to_dict())

    serialize = to_json

    def audit_fields(
        self,
        *,
        policy_version: str = DEFAULT_CONSENT_POLICY_VERSION,
        outcome: str = "allowed",
        reason: str = "verified",
    ) -> dict[str, Any]:
        """Return PHI-free identifiers, digest, decision, and policy metadata."""

        return consent_audit_fields(
            receipt=self,
            client=self.client,
            tool=self.tool,
            resource=self.resource,
            scope=self.scope,
            argument_digest=self.argument_digest,
            decision=self.decision,
            outcome=outcome,
            reason=reason,
            policy_version=policy_version,
        )

    to_audit_dict = audit_fields

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsentReceipt":
        """Restore a receipt from a mapping without accepting raw arguments."""

        if not isinstance(payload, Mapping):
            raise ConsentReceiptValidationError("receipt must be a mapping")
        argument_value = payload.get("argument_digest", _MISSING)
        if argument_value is _MISSING:
            argument_value = payload.get("canonical_argument_digest", _MISSING)
        expires_value = payload.get("expires_at", _MISSING)
        if expires_value is _MISSING:
            expires_value = payload.get("expiry", _MISSING)
        values = {
            "schema_version": payload.get("schema_version", _MISSING),
            "receipt_id": payload.get("receipt_id", _MISSING),
            "client": payload.get("client", payload.get("client_id", _MISSING)),
            "tool": payload.get("tool", payload.get("tool_name", _MISSING)),
            "resource": payload.get("resource", payload.get("resource_id", _MISSING)),
            "scope": payload.get("scope", _MISSING),
            "decision": payload.get("decision", _MISSING),
            "issued_at": payload.get("issued_at", _MISSING),
            "expires_at": expires_value,
            "argument_digest": argument_value,
            "key_id": payload.get("key_id", _MISSING),
            "signature": payload.get("signature", _MISSING),
        }
        if any(value is _MISSING for value in values.values()):
            raise ConsentReceiptValidationError("receipt is missing signed fields")
        return cls(**values)

    @classmethod
    def from_json(cls, serialized: str | bytes) -> "ConsentReceipt":
        """Restore a receipt from canonical or equivalent JSON."""

        if not isinstance(serialized, (str, bytes)):
            raise ConsentReceiptValidationError("serialized receipt must be JSON text")
        try:
            payload = json.loads(serialized)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ConsentReceiptValidationError(
                "serialized receipt is not valid JSON"
            ) from exc
        return cls.from_dict(payload)


class ConsentReceiptIssuer:
    """Issue signed receipts using an injected clock and key provider."""

    def __init__(
        self,
        key_provider: ConsentKeyProvider | Mapping[str, bytes | str] | bytes | str,
        *,
        key_id: str = "default",
        clock: Callable[[], Real | datetime] | Any = time.time,
        receipt_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.key_provider = key_provider
        self.key_id = _validate_identifier(key_id, "key_id")
        self.clock = clock
        self.receipt_id_factory = receipt_id_factory or (lambda: uuid.uuid4().hex)

    def issue(
        self,
        client: str,
        tool: str,
        resource: str,
        scope: str,
        arguments: Any = _MISSING,
        *,
        decision: ConsentDecision | str = ConsentDecision.ALLOW,
        ttl_seconds: Real = 300,
        expires_in: Real | None = None,
        expires_at: Real | datetime | None = None,
        expiry: Real | datetime | None = None,
        issued_at: Real | datetime | None = None,
        receipt_id: str | None = None,
        argument_digest: str | None = None,
        canonical_argument_digest: str | None = None,
        key_id: str | None = None,
    ) -> ConsentReceipt:
        """Issue one signed receipt bound to the supplied request arguments.

        Args:
            client: Stable client identifier, not a bearer credential.
            tool: Exact MCP tool name.
            resource: Stable resource identifier selected by the application.
            scope: Authorization scope required by the tool.
            arguments: JSON-compatible tool arguments, used only to compute a digest.
            decision: ``allow`` receipts can be consumed; ``deny`` receipts fail closed.
            ttl_seconds: Relative lifetime when ``expires_at`` is not supplied.
            expires_in: Alias for ``ttl_seconds``.
            expires_at: Absolute expiry timestamp, in UTC seconds or datetime form.
            issued_at: Issuance timestamp; defaults to the injected clock.
            receipt_id: Optional deterministic identifier for tests or an application ledger.
            argument_digest: Precomputed canonical digest, if arguments are unavailable.
            canonical_argument_digest: Alias for ``argument_digest``.
            key_id: Optional provider key identifier; defaults to the issuer key.
        """

        normalized_client = _validate_identifier(client, "client")
        normalized_tool = _validate_identifier(tool, "tool")
        normalized_resource = _validate_identifier(resource, "resource")
        normalized_scope = _validate_identifier(scope, "scope")
        normalized_decision = _coerce_decision(decision)
        issued = _timestamp(
            _now(self.clock) if issued_at is None else issued_at, "issued_at"
        )
        if expires_at is not None and expiry is not None:
            raise ConsentReceiptValidationError("choose one of expires_at or expiry")
        if expiry is not None:
            expires_at = expiry
        if expires_at is not None and (expires_in is not None or ttl_seconds != 300):
            raise ConsentReceiptValidationError(
                "choose one of expires_at, expires_in, or ttl_seconds"
            )
        if expires_at is None:
            lifetime = expires_in if expires_in is not None else ttl_seconds
            if isinstance(lifetime, bool) or not isinstance(lifetime, Real):
                raise ConsentReceiptValidationError("receipt lifetime must be numeric")
            if not math.isfinite(float(lifetime)) or float(lifetime) <= 0:
                raise ConsentReceiptValidationError("receipt lifetime must be positive")
            expires = issued + float(lifetime)
        else:
            expires = _timestamp(expires_at, "expires_at")
        if expires <= issued:
            raise ConsentReceiptValidationError("receipt expiry must be after issuance")

        digest = _select_digest(
            arguments,
            argument_digest=argument_digest,
            canonical_argument_digest=canonical_argument_digest,
        )
        selected_key_id = _validate_identifier(key_id or self.key_id, "key_id")
        identifier = receipt_id or self.receipt_id_factory()
        identifier = _validate_identifier(identifier, "receipt_id")
        unsigned = ConsentReceipt(
            receipt_id=identifier,
            client=normalized_client,
            tool=normalized_tool,
            resource=normalized_resource,
            scope=normalized_scope,
            decision=normalized_decision,
            issued_at=issued,
            expires_at=expires,
            argument_digest=digest,
            key_id=selected_key_id,
            signature="",
        )
        signature = _sign(
            unsigned.signing_payload(), _resolve_key(self.key_provider, selected_key_id)
        )
        return ConsentReceipt(**{**unsigned.to_dict(), "signature": signature})


class ConsentReceiptVerifier:
    """Verify and atomically consume receipts exactly once."""

    def __init__(
        self,
        key_provider: ConsentKeyProvider | Mapping[str, bytes | str] | bytes | str,
        *,
        clock: Callable[[], Real | datetime] | Any = time.time,
        clock_skew_seconds: Real = 0,
    ) -> None:
        if isinstance(clock_skew_seconds, bool) or not isinstance(
            clock_skew_seconds, Real
        ):
            raise ValueError("clock_skew_seconds must be numeric")
        if (
            not math.isfinite(float(clock_skew_seconds))
            or float(clock_skew_seconds) < 0
        ):
            raise ValueError("clock_skew_seconds must be non-negative")
        self.key_provider = key_provider
        self.clock = clock
        self.clock_skew_seconds = float(clock_skew_seconds)
        self._consumed: set[str] = set()
        self._lock = RLock()

    def verify(
        self,
        receipt: ConsentReceipt | Mapping[str, Any] | str | bytes,
        client: str,
        tool: str,
        resource: str,
        scope: str,
        arguments: Any = _MISSING,
        *,
        argument_digest: str | None = None,
        canonical_argument_digest: str | None = None,
    ) -> ConsentReceipt:
        """Verify a receipt's signature, binding, lifetime, and single-use state.

        A receipt is marked consumed only after every check succeeds.  Failed
        wrong-tool or wrong-resource attempts therefore do not burn a receipt
        that was never authorized for that request.
        """

        candidate = _coerce_receipt(receipt)
        expected_client = _validate_identifier(client, "client")
        expected_tool = _validate_identifier(tool, "tool")
        expected_resource = _validate_identifier(resource, "resource")
        expected_scope = _validate_identifier(scope, "scope")
        expected_digest = _select_digest(
            arguments,
            argument_digest=argument_digest,
            canonical_argument_digest=canonical_argument_digest,
        )
        key = _resolve_key(self.key_provider, candidate.key_id)
        expected_signature = _sign(candidate.signing_payload(), key)
        if not hmac.compare_digest(expected_signature, candidate.signature):
            raise ConsentReceiptSignatureError(
                "receipt signature could not be verified"
            )

        now = _timestamp(_now(self.clock), "clock")
        skew = self.clock_skew_seconds
        if now + skew < candidate.issued_at:
            raise ConsentReceiptNotYetValidError("receipt is not yet valid")
        if now >= candidate.expires_at:
            raise ConsentReceiptExpiredError("receipt has expired")
        if candidate.decision is not ConsentDecision.ALLOW:
            raise ConsentReceiptDeniedError("receipt decision is deny")
        if (
            candidate.client != expected_client
            or candidate.tool != expected_tool
            or candidate.resource != expected_resource
            or candidate.scope != expected_scope
            or candidate.argument_digest != expected_digest
        ):
            raise ConsentReceiptBindingError(
                "receipt does not match the requested action"
            )

        with self._lock:
            if candidate.receipt_id in self._consumed:
                raise ConsentReceiptReplayError("receipt has already been consumed")
            self._consumed.add(candidate.receipt_id)
        return candidate

    def verify_result(
        self,
        receipt: ConsentReceipt | Mapping[str, Any] | str | bytes | None,
        client: str,
        tool: str,
        resource: str,
        scope: str,
        arguments: Any = _MISSING,
        *,
        argument_digest: str | None = None,
        canonical_argument_digest: str | None = None,
    ) -> ConsentReceiptVerificationResult:
        """Verify without raising and return only a stable categorical outcome.

        Successful calls retain the one-time consumption behavior of
        :meth:`verify`. The result never includes request arguments, signatures,
        key material, or exception messages.
        """

        if receipt is None:
            return ConsentReceiptVerificationResult(False, "missing_receipt")
        try:
            verified = self.verify(
                receipt,
                client,
                tool,
                resource,
                scope,
                arguments,
                argument_digest=argument_digest,
                canonical_argument_digest=canonical_argument_digest,
            )
        except ConsentReceiptError as exc:
            return ConsentReceiptVerificationResult(False, _reason_code(exc))
        return ConsentReceiptVerificationResult(True, "verified", verified)

    verify_and_consume = verify

    def is_consumed(self, receipt_id: str) -> bool:
        """Return whether a receipt identifier has already been consumed."""

        identifier = _validate_identifier(receipt_id, "receipt_id")
        with self._lock:
            return identifier in self._consumed

    @property
    def consumed_receipt_ids(self) -> frozenset[str]:
        """Return a read-only snapshot of consumed identifiers for diagnostics."""

        with self._lock:
            return frozenset(self._consumed)


@dataclass(frozen=True, slots=True)
class ConsentReceiptPolicy:
    """Optional MCP policy hook for state-changing tool calls."""

    verifier: ConsentReceiptVerifier
    client: str
    resource: str | Mapping[str, str] | Callable[..., str] = DEFAULT_CONSENT_RESOURCE
    scope: str | Mapping[str, str] | Callable[..., str] = DEFAULT_CONSENT_SCOPE
    policy_version: str = DEFAULT_CONSENT_POLICY_VERSION
    require_receipt: bool = True
    audit_sink: Callable[[Mapping[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.verifier, ConsentReceiptVerifier):
            raise TypeError("verifier must be a ConsentReceiptVerifier")
        _validate_identifier(self.client, "client")
        _validate_identifier(self.policy_version, "policy_version")
        if not isinstance(self.require_receipt, bool):
            raise TypeError("require_receipt must be a boolean")
        if self.audit_sink is not None and not callable(self.audit_sink):
            raise TypeError("audit_sink must be callable")

    def resource_for(self, tool: str, arguments: Mapping[str, Any]) -> str:
        """Resolve the PHI-free resource identifier for a tool call."""

        return _resolve_policy_identifier(self.resource, tool, arguments, "resource")

    def scope_for(self, tool: str, arguments: Mapping[str, Any]) -> str:
        """Resolve the required scope for a tool call."""

        return _resolve_policy_identifier(self.scope, tool, arguments, "scope")

    def authorize(
        self,
        *,
        tool: str,
        arguments: Mapping[str, Any],
        receipt: ConsentReceipt | Mapping[str, Any] | str | bytes | None = None,
    ) -> ConsentReceipt | None:
        """Authorize one state-changing request and emit only safe audit data."""

        expected_tool = _validate_identifier(tool, "tool")
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        expected_client = _validate_identifier(self.client, "client")
        expected_resource = self.resource_for(expected_tool, arguments)
        expected_scope = self.scope_for(expected_tool, arguments)
        expected_digest = canonical_argument_digest(arguments)
        if receipt is None and not self.require_receipt:
            audit = consent_audit_fields(
                client=expected_client,
                tool=expected_tool,
                resource=expected_resource,
                scope=expected_scope,
                argument_digest=expected_digest,
                decision=ConsentDecision.ALLOW,
                outcome="allowed",
                reason="receipt_not_required",
                policy_version=self.policy_version,
            )
            self._emit_audit(audit)
            return None
        if receipt is None:
            audit = consent_audit_fields(
                client=expected_client,
                tool=expected_tool,
                resource=expected_resource,
                scope=expected_scope,
                argument_digest=expected_digest,
                decision=ConsentDecision.DENY,
                outcome="denied",
                reason="missing_receipt",
                policy_version=self.policy_version,
            )
            self._emit_audit(audit)
            raise ConsentReceiptRequiredError("a consent receipt is required")

        try:
            verified = self.verifier.verify(
                receipt,
                expected_client,
                expected_tool,
                expected_resource,
                expected_scope,
                arguments,
            )
        except ConsentReceiptError as error:
            candidate = _receipt_for_audit(receipt)
            audit = consent_audit_fields(
                receipt=candidate,
                client=expected_client,
                tool=expected_tool,
                resource=expected_resource,
                scope=expected_scope,
                argument_digest=expected_digest,
                decision=(
                    candidate.decision
                    if candidate is not None
                    else ConsentDecision.DENY
                ),
                outcome="denied",
                reason=_reason_code(error),
                policy_version=self.policy_version,
            )
            self._emit_audit(audit)
            raise

        audit = consent_audit_fields(
            receipt=verified,
            client=expected_client,
            tool=expected_tool,
            resource=expected_resource,
            scope=expected_scope,
            argument_digest=expected_digest,
            decision=verified.decision,
            outcome="allowed",
            reason="verified",
            policy_version=self.policy_version,
        )
        self._emit_audit(audit)
        return verified

    def _emit_audit(self, fields: Mapping[str, Any]) -> None:
        safe_fields = dict(fields)
        if self.audit_sink is not None:
            try:
                self.audit_sink(safe_fields)
            except Exception as exc:  # pragma: no cover - defensive sink boundary
                logger.warning(
                    "MCP consent audit sink failed: %s",
                    exc.__class__.__name__,
                )
        level = logging.INFO if safe_fields["outcome"] == "allowed" else logging.WARNING
        logger.log(level, "MCP consent decision: %s", safe_fields)


def consent_audit_fields(
    *,
    client: str,
    tool: str,
    resource: str,
    scope: str,
    argument_digest: str,
    decision: ConsentDecision | str,
    outcome: str,
    reason: str,
    policy_version: str,
    receipt: ConsentReceipt | None = None,
) -> dict[str, Any]:
    """Build a deterministic PHI-free audit mapping for one policy decision."""

    _validate_identifier(client, "client")
    _validate_identifier(tool, "tool")
    _validate_identifier(resource, "resource")
    _validate_identifier(scope, "scope")
    _validate_identifier(outcome, "outcome")
    _validate_identifier(reason, "reason")
    _validate_identifier(policy_version, "policy_version")
    _validate_digest(argument_digest)
    normalized_decision = _coerce_decision(decision)
    return {
        "event": CONSENT_RECEIPT_EVENT,
        "schema_version": CONSENT_RECEIPT_SCHEMA_VERSION,
        "receipt_id": receipt.receipt_id if receipt is not None else None,
        "client": client,
        "tool": tool,
        "resource": resource,
        "scope": scope,
        "argument_digest": argument_digest,
        "decision": normalized_decision.value,
        "outcome": outcome,
        "reason": reason,
        "policy_version": policy_version,
    }


def issue_consent_receipt(
    *,
    key_provider: ConsentKeyProvider | Mapping[str, bytes | str] | bytes | str,
    client: str,
    tool: str,
    resource: str,
    scope: str,
    arguments: Any = _MISSING,
    **kwargs: Any,
) -> ConsentReceipt:
    """Convenience wrapper around :class:`ConsentReceiptIssuer`."""

    return ConsentReceiptIssuer(
        key_provider,
        key_id=str(kwargs.pop("key_id", "default")),
        clock=kwargs.pop("clock", time.time),
        receipt_id_factory=kwargs.pop("receipt_id_factory", None),
    ).issue(
        client,
        tool,
        resource,
        scope,
        arguments,
        **kwargs,
    )


def verify_consent_receipt(
    receipt: ConsentReceipt | Mapping[str, Any] | str | bytes,
    *,
    key_provider: ConsentKeyProvider | Mapping[str, bytes | str] | bytes | str,
    client: str,
    tool: str,
    resource: str,
    scope: str,
    arguments: Any = _MISSING,
    **kwargs: Any,
) -> ConsentReceipt:
    """Convenience wrapper around :class:`ConsentReceiptVerifier`."""

    verifier = ConsentReceiptVerifier(
        key_provider,
        clock=kwargs.pop("clock", time.time),
        clock_skew_seconds=kwargs.pop("clock_skew_seconds", 0),
    )
    return verifier.verify(
        receipt,
        client,
        tool,
        resource,
        scope,
        arguments,
        **kwargs,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConsentReceiptValidationError(
            f"{field_name} must be a non-empty identifier"
        )
    if len(value) > 512:
        raise ConsentReceiptValidationError(f"{field_name} is too long")
    return value


def _timestamp(value: Any, field_name: str) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ConsentReceiptValidationError(
                f"{field_name} datetime must be timezone-aware"
            )
        value = value.astimezone(timezone.utc).timestamp()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConsentReceiptValidationError(f"{field_name} must be a numeric timestamp")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConsentReceiptValidationError(f"{field_name} must be finite")
    return numeric


def _now(clock: Any) -> Real | datetime:
    if callable(clock):
        return clock()
    now_method = getattr(clock, "now", None)
    if callable(now_method):
        return now_method()
    raise ConsentReceiptValidationError("clock must be callable or expose now()")


def _coerce_decision(value: ConsentDecision | str) -> ConsentDecision:
    try:
        return (
            value
            if isinstance(value, ConsentDecision)
            else ConsentDecision(str(value).lower())
        )
    except (TypeError, ValueError) as exc:
        raise ConsentReceiptValidationError(
            "unsupported consent receipt decision"
        ) from exc


def _validate_digest(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX):
        raise ConsentReceiptValidationError("argument_digest has an invalid format")
    digest = value.removeprefix(_DIGEST_PREFIX)
    if len(digest) != 64:
        raise ConsentReceiptValidationError("argument_digest has an invalid format")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ConsentReceiptValidationError(
            "argument_digest has an invalid format"
        ) from exc
    return value


def _select_digest(
    arguments: Any,
    *,
    argument_digest: str | None,
    canonical_argument_digest: str | None,
) -> str:
    selected = argument_digest or canonical_argument_digest
    if argument_digest and canonical_argument_digest:
        if _validate_digest(argument_digest) != _validate_digest(
            canonical_argument_digest
        ):
            raise ConsentReceiptValidationError("argument digest aliases do not match")
    if selected is not None:
        return _validate_digest(selected)
    if arguments is _MISSING:
        arguments = {}
    return canonical_argument_digest_fn(arguments)


# Keep the helper name separate from the public alias so the selector remains
# unambiguous when callers use ``canonical_argument_digest`` as a keyword.
def canonical_argument_digest_fn(arguments: Any) -> str:
    return canonical_argument_digest(arguments)


def _coerce_receipt(
    receipt: ConsentReceipt | Mapping[str, Any] | str | bytes,
) -> ConsentReceipt:
    if isinstance(receipt, ConsentReceipt):
        return receipt
    if isinstance(receipt, Mapping):
        return ConsentReceipt.from_dict(receipt)
    if isinstance(receipt, (str, bytes)):
        return ConsentReceipt.from_json(receipt)
    raise ConsentReceiptValidationError(
        "receipt must be a typed receipt or serialization"
    )


def _resolve_key(
    provider: ConsentKeyProvider | Mapping[str, bytes | str] | bytes | str,
    key_id: str,
) -> bytes:
    try:
        if isinstance(provider, (bytes, str)):
            value = provider
        elif isinstance(provider, Mapping):
            value = provider[key_id]
        else:
            get_key = getattr(provider, "get_key", None)
            if callable(get_key):
                value = get_key(key_id)
            elif callable(provider):
                value = provider(key_id)
            else:
                raise TypeError
        return _key_bytes(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConsentReceiptKeyError("receipt signing key is unavailable") from exc


def _key_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise ValueError("key must be non-empty bytes")
    return bytes(value)


def _sign(payload: Mapping[str, Any], key: bytes) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    digest = hmac.new(key, encoded, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest}"


def _receipt_for_audit(value: Any) -> ConsentReceipt | None:
    try:
        return _coerce_receipt(value)
    except ConsentReceiptError:
        return None


def _reason_code(error: ConsentReceiptError) -> str:
    mapping = {
        ConsentReceiptRequiredError: "missing_receipt",
        ConsentReceiptExpiredError: "expired",
        ConsentReceiptReplayError: "replay",
        ConsentReceiptBindingError: "binding_mismatch",
        ConsentReceiptDeniedError: "decision_denied",
        ConsentReceiptSignatureError: "invalid_signature",
        ConsentReceiptNotYetValidError: "not_yet_valid",
        ConsentReceiptKeyError: "key_unavailable",
    }
    for error_type, code in mapping.items():
        if isinstance(error, error_type):
            return code
    return "invalid_receipt"


def _resolve_policy_identifier(
    configured: str | Mapping[str, str] | Callable[..., str],
    tool: str,
    arguments: Mapping[str, Any],
    field_name: str,
) -> str:
    if isinstance(configured, str):
        return _validate_identifier(configured, field_name)
    if isinstance(configured, Mapping):
        value = configured.get(tool, configured.get("*"))
        if value is None:
            raise ConsentReceiptBindingError(
                f"no {field_name} is configured for the tool"
            )
        return _validate_identifier(value, field_name)
    if callable(configured):
        try:
            value = configured(tool, arguments)
        except TypeError:
            value = configured(tool)
        return _validate_identifier(value, field_name)
    raise ConsentReceiptBindingError(f"invalid {field_name} policy configuration")


__all__ = [
    "CONSENT_RECEIPT_EVENT",
    "CONSENT_RECEIPT_SCHEMA_VERSION",
    "DEFAULT_CONSENT_POLICY_VERSION",
    "DEFAULT_CONSENT_RESOURCE",
    "DEFAULT_CONSENT_SCOPE",
    "ConsentDecision",
    "ConsentKeyProvider",
    "ConsentReceipt",
    "ConsentReceiptBindingError",
    "ConsentReceiptDeniedError",
    "ConsentReceiptError",
    "ConsentReceiptExpired",
    "ConsentReceiptExpiredError",
    "ConsentReceiptIssuer",
    "ConsentReceiptKeyError",
    "ConsentReceiptNotYetValidError",
    "ConsentReceiptPolicy",
    "ConsentReceiptReplay",
    "ConsentReceiptReplayError",
    "ConsentReceiptRequired",
    "ConsentReceiptRequiredError",
    "ConsentReceiptSignatureError",
    "ConsentReceiptValidationError",
    "ConsentReceiptVerifier",
    "ConsentReceiptVerificationResult",
    "ConsentReceiptVerificationError",
    "MappingConsentKeyProvider",
    "StaticConsentKeyProvider",
    "argument_digest",
    "canonical_argument_digest",
    "consent_audit_fields",
    "issue_consent_receipt",
    "verify_consent_receipt",
]
