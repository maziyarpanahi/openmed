"""Principal-bound authority for Journey resource reads.

Network inputs to Journey read surfaces are requests, not authority. This module
resolves the authoritative role, attribute, namespace, purpose, export, and
consent context for an authenticated principal through a transport-neutral
:class:`JourneyAccessContextProvider`. Callers may narrow that context but never
widen it. Missing, unknown, expired, or withdrawn context denies the read, and
anonymous reads are refused outside the ``dev`` and ``test`` profiles. Every
decision is audited with an opaque principal digest, never the raw identity.

REST and GraphQL both call :meth:`JourneyAccessGate.list_resources`, so the
same principal receives the same decision on every transport.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Protocol, runtime_checkable

from openmed.clinical.journey_contracts import canonical_digest
from openmed.core.config import PROFILE_ENV_VAR

from .auth import AuthPrincipal, principal_digest
from .journey_resources import (
    _CONTROLLED_RE,
    _VERSION_RE,
    DEFAULT_PAGE_SIZE,
    MAX_ACCESS_ATTRIBUTES,
    JourneyAccessPolicy,
    JourneyResourceCatalog,
    JourneyResourceKind,
    JourneyResourcePage,
    JourneyResourceQuery,
)

JOURNEY_ACCESS_CONTEXT_VERSION: Final = "1.0.0"
JOURNEY_ACCESS_AUDIT_SCHEMA_VERSION: Final = "1.0.0"
SERVICE_JOURNEY_GRANTS_ENV_VAR: Final = "OPENMED_SERVICE_JOURNEY_GRANTS"
SERVICE_JOURNEY_GRANTS_FILE_ENV_VAR: Final = "OPENMED_SERVICE_JOURNEY_GRANTS_FILE"
DEVELOPMENT_PROFILES: Final = frozenset({"dev", "test"})
ANONYMOUS_CREDENTIAL_TYPE: Final = "anonymous"
PRINCIPAL_CREDENTIAL_TYPES: Final = frozenset({"api_key", "jwt", "mtls"})
UNBOUND_SELECTION: Final = "unbound"
CONSENT_STATES: Final = ("active", "unknown", "withdrawn")
JOURNEY_ACCESS_TRANSPORTS: Final = frozenset({"graphql", "grpc", "python", "rest"})
MAX_JOURNEY_GRANTS: Final = 1024
MAX_GRANT_VALUES: Final = 128
MAX_GRANT_PRINCIPAL_LENGTH: Final = 512
DEFAULT_AUDIT_CAPACITY: Final = 1024

_DEFAULT_SELECTIONS: Final = {
    "namespace": "default",
    "purpose": "care_review",
    "role": "clinician",
    "export_policy": "metadata_only",
}
_GRANT_BODY_KEYS: Final = frozenset(
    {
        "attributes",
        "consent_state",
        "expires_at",
        "export_policies",
        "namespaces",
        "purposes",
        "roles",
    }
)
_GRANT_KEYS: Final = _GRANT_BODY_KEYS | {"credential_type", "principal"}
_CONFIG_KEYS: Final = frozenset({"grants", "jwt_claim", "version"})


@dataclass(frozen=True, slots=True)
class JourneyAccessContext:
    """Authoritative Journey access context resolved for one caller.

    Attributes:
        credential_type: Credential family the context is bound to, such as
            ``api_key``, ``jwt``, ``mtls``, or ``anonymous``.
        source: Controlled name of the authority, such as ``operator_grant``.
        roles: Roles the caller may act under.
        attributes: Access attributes the caller holds.
        namespaces: Namespaces the caller may read.
        purposes: Purposes of use the caller may declare.
        export_policies: Export postures the caller may request.
        consent_state: Authoritative consent state: ``active``, ``unknown``,
            or ``withdrawn``. Anything other than ``active`` denies reads.
        expires_at: Optional Unix timestamp after which the context is invalid.
        context_version: Semantic version of the grant document or hook.
    """

    credential_type: str
    source: str
    roles: frozenset[str] = frozenset()
    attributes: frozenset[str] = frozenset()
    namespaces: frozenset[str] = frozenset()
    purposes: frozenset[str] = frozenset()
    export_policies: frozenset[str] = frozenset()
    consent_state: str = "unknown"
    expires_at: float | None = None
    context_version: str = JOURNEY_ACCESS_CONTEXT_VERSION

    def __post_init__(self) -> None:
        _require_controlled(self.credential_type, "context credential_type")
        _require_controlled(self.source, "context source")
        for name in ("roles", "namespaces", "purposes", "export_policies"):
            object.__setattr__(
                self,
                name,
                _controlled_set(getattr(self, name), name, limit=MAX_GRANT_VALUES),
            )
        object.__setattr__(
            self,
            "attributes",
            _controlled_set(self.attributes, "attributes", limit=MAX_ACCESS_ATTRIBUTES),
        )
        if (
            not isinstance(self.consent_state, str)
            or self.consent_state not in CONSENT_STATES
        ):
            raise ValueError("context consent_state is unsupported")
        object.__setattr__(self, "expires_at", _optional_timestamp(self.expires_at))
        _require_version(self.context_version, "context_version")

    def is_expired(self, now: float) -> bool:
        """Return whether the context has expired at ``now``.

        Args:
            now: Current Unix timestamp.

        Returns:
            True when an expiry is set and ``now`` is at or beyond it.
        """

        return self.expires_at is not None and now >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, identity-free representation."""

        return {
            "attributes": sorted(self.attributes),
            "consent_state": self.consent_state,
            "context_version": self.context_version,
            "credential_type": self.credential_type,
            "expires_at": self.expires_at,
            "export_policies": sorted(self.export_policies),
            "namespaces": sorted(self.namespaces),
            "purposes": sorted(self.purposes),
            "roles": sorted(self.roles),
            "source": self.source,
        }

    @property
    def digest(self) -> str:
        """Return a stable digest of the granted context."""

        return canonical_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class JourneyAccessGrant:
    """Operator-configured Journey authority for one authenticated principal.

    A grant matches only when both the credential type and the exact subject
    attached by the authentication layer match, so a JWT subject can never
    borrow an API-key or mTLS grant with the same name.

    Attributes:
        credential_type: ``api_key``, ``jwt``, or ``mtls``.
        principal: Exact authenticated subject. Never logged or echoed.
        roles: Roles the principal may act under.
        attributes: Access attributes the principal holds.
        namespaces: Namespaces the principal may read.
        purposes: Purposes of use the principal may declare.
        export_policies: Export postures the principal may request.
        consent_state: Attested consent state; defaults to ``unknown`` (deny).
        expires_at: Optional Unix timestamp or RFC 3339 string with offset.
    """

    credential_type: str
    principal: str = field(repr=False)
    roles: frozenset[str] = frozenset()
    attributes: frozenset[str] = frozenset()
    namespaces: frozenset[str] = frozenset()
    purposes: frozenset[str] = frozenset()
    export_policies: frozenset[str] = frozenset()
    consent_state: str = "unknown"
    expires_at: float | str | None = None

    def __post_init__(self) -> None:
        if self.credential_type not in PRINCIPAL_CREDENTIAL_TYPES:
            raise ValueError("Journey grant credential_type is unsupported")
        if (
            not isinstance(self.principal, str)
            or not self.principal.strip()
            or len(self.principal) > MAX_GRANT_PRINCIPAL_LENGTH
            or not self.principal.isprintable()
        ):
            raise ValueError("Journey grant principal must be a bounded subject")
        context = self.to_context(source="operator_grant")
        for name in (
            "roles",
            "attributes",
            "namespaces",
            "purposes",
            "export_policies",
            "expires_at",
        ):
            object.__setattr__(self, name, getattr(context, name))

    def to_context(
        self,
        *,
        source: str,
        context_version: str = JOURNEY_ACCESS_CONTEXT_VERSION,
    ) -> JourneyAccessContext:
        """Build the authoritative context this grant confers.

        Args:
            source: Controlled name of the authority.
            context_version: Version of the grant document.

        Returns:
            The context bound to this grant's credential type.
        """

        return JourneyAccessContext(
            credential_type=self.credential_type,
            source=source,
            roles=self.roles,
            attributes=self.attributes,
            namespaces=self.namespaces,
            purposes=self.purposes,
            export_policies=self.export_policies,
            consent_state=self.consent_state,
            expires_at=_optional_timestamp(self.expires_at),
            context_version=context_version,
        )


@dataclass(frozen=True, slots=True)
class JourneyConsentScope:
    """Value-free scope handed to an operator consent hook.

    Attributes:
        credential_type: Credential family of the caller.
        source: Authority that produced the context.
        namespaces: Namespaces the context grants.
        purposes: Purposes the context grants.
    """

    credential_type: str
    source: str
    namespaces: frozenset[str]
    purposes: frozenset[str]


JourneyConsentResolver = Callable[[JourneyConsentScope], Optional[str]]


@runtime_checkable
class JourneyAccessContextProvider(Protocol):
    """Resolve the authoritative Journey access context for a caller."""

    def resolve(
        self,
        principal: AuthPrincipal | None,
        *,
        policy: JourneyAccessPolicy,
    ) -> JourneyAccessContext | None:
        """Return the caller's context, or ``None`` when none is known.

        Args:
            principal: Authenticated principal, or ``None`` for anonymous.
            policy: Active Journey policy envelope.

        Returns:
            The authoritative context, or ``None`` to deny the read.
        """
        ...


@dataclass(frozen=True, slots=True)
class JourneyGrantConfig:
    """Parsed operator grant document.

    Attributes:
        grants: Principal grants keyed by credential type and subject.
        jwt_claim: Optional verified-JWT claim that may carry a grant.
        version: Semantic version of the grant document.
    """

    grants: tuple[JourneyAccessGrant, ...] = ()
    jwt_claim: str | None = None
    version: str = JOURNEY_ACCESS_CONTEXT_VERSION


class ConfiguredJourneyAccessContextProvider:
    """Resolve Journey context from operator grants and optional hooks.

    Resolution order for an authenticated principal: an exact operator grant
    for its credential type and subject, then (for verified JWTs only, when
    ``jwt_claim`` is configured) a grant object carried in that signed claim.
    A JWT-derived context never outlives the token's ``exp``. When a consent
    hook is configured its answer replaces the grant's consent state, and a
    failing hook yields ``unknown`` consent. Anonymous callers receive the
    policy envelope only when ``anonymous_development`` is enabled.
    """

    def __init__(
        self,
        grants: Iterable[JourneyAccessGrant] = (),
        *,
        jwt_claim: str | None = None,
        consent_resolver: JourneyConsentResolver | None = None,
        anonymous_development: bool = False,
        context_version: str = JOURNEY_ACCESS_CONTEXT_VERSION,
    ) -> None:
        """Index grants and validate hooks.

        Args:
            grants: Operator grants; duplicates are rejected.
            jwt_claim: Verified-JWT claim name that may carry a grant object.
            consent_resolver: Optional authoritative consent hook.
            anonymous_development: Grant anonymous callers the policy envelope.
            context_version: Version recorded on every resolved context.

        Raises:
            ValueError: If grants or hook settings are invalid.
        """

        index: dict[tuple[str, str], JourneyAccessGrant] = {}
        for grant in grants:
            if type(grant) is not JourneyAccessGrant:
                raise TypeError("Journey grants must be JourneyAccessGrant values")
            key = (grant.credential_type, grant.principal)
            if key in index:
                raise ValueError("Journey grants contain a duplicate principal")
            index[key] = grant
        if len(index) > MAX_JOURNEY_GRANTS:
            raise ValueError("Journey grants exceed the supported count")
        if jwt_claim is not None:
            _require_controlled(jwt_claim, "jwt_claim")
        if consent_resolver is not None and not callable(consent_resolver):
            raise TypeError("consent_resolver must be callable")
        if type(anonymous_development) is not bool:
            raise TypeError("anonymous_development must be a boolean")
        _require_version(context_version, "context_version")
        self._grants = index
        self._jwt_claim = jwt_claim
        self._consent_resolver = consent_resolver
        self._anonymous_development = anonymous_development
        self._context_version = context_version

    @classmethod
    def from_env(
        cls,
        *,
        anonymous_development: bool = False,
        consent_resolver: JourneyConsentResolver | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ConfiguredJourneyAccessContextProvider":
        """Build a provider from ``OPENMED_SERVICE_JOURNEY_GRANTS[_FILE]``.

        Args:
            anonymous_development: Grant anonymous callers the policy envelope.
            consent_resolver: Optional authoritative consent hook.
            environ: Environment mapping; defaults to ``os.environ``.

        Returns:
            A configured provider. With no grant document, every authenticated
            principal resolves to no context and is denied.
        """

        env = os.environ if environ is None else environ
        config = parse_journey_grants(
            env.get(SERVICE_JOURNEY_GRANTS_ENV_VAR),
            env.get(SERVICE_JOURNEY_GRANTS_FILE_ENV_VAR),
        )
        return cls(
            config.grants,
            jwt_claim=config.jwt_claim,
            consent_resolver=consent_resolver,
            anonymous_development=anonymous_development,
            context_version=config.version,
        )

    def resolve(
        self,
        principal: AuthPrincipal | None,
        *,
        policy: JourneyAccessPolicy,
    ) -> JourneyAccessContext | None:
        """Return the authoritative context for ``principal``.

        Args:
            principal: Authenticated principal, or ``None`` for anonymous.
            policy: Active Journey policy envelope.

        Returns:
            The resolved context, or ``None`` when no authority applies.
        """

        if principal is None:
            if not self._anonymous_development:
                return None
            context = development_access_context(
                policy, context_version=self._context_version
            )
        elif type(principal) is not AuthPrincipal:
            return None
        else:
            grant = self._grants.get((principal.credential_type, principal.subject))
            if grant is not None:
                context = grant.to_context(
                    source="operator_grant",
                    context_version=self._context_version,
                )
            elif self._jwt_claim is not None and principal.credential_type == "jwt":
                claimed = _jwt_claim_context(
                    principal.claims.get(self._jwt_claim),
                    context_version=self._context_version,
                )
                if claimed is None:
                    return None
                context = claimed
            else:
                return None
            if principal.credential_type == "jwt":
                context = _cap_expiry(context, principal.claims.get("exp"))
        if self._consent_resolver is not None:
            context = replace(
                context,
                consent_state=_resolved_consent(self._consent_resolver, context),
            )
        return context


def development_access_context(
    policy: JourneyAccessPolicy,
    *,
    context_version: str = JOURNEY_ACCESS_CONTEXT_VERSION,
) -> JourneyAccessContext:
    """Return the anonymous context used only under development profiles.

    Args:
        policy: Active policy whose envelope bounds the development context.
        context_version: Version recorded on the context.

    Returns:
        A context granting the policy's namespaces, purposes, roles, export
        postures, and required attributes with active consent.
    """

    attributes: set[str] = set()
    for required in policy.required_attributes_by_resource.values():
        attributes.update(required)
    return JourneyAccessContext(
        credential_type=ANONYMOUS_CREDENTIAL_TYPE,
        source="anonymous_development",
        roles=policy.allowed_roles,
        attributes=frozenset(attributes),
        namespaces=policy.allowed_namespaces,
        purposes=policy.allowed_purposes,
        export_policies=policy.allowed_export_policies,
        consent_state="active",
        context_version=context_version,
    )


@dataclass(frozen=True, slots=True)
class JourneyAccessRequest:
    """Caller-requested Journey read; every access field may only narrow.

    ``None`` means "use the authoritative context": a granted ``default`` /
    ``care_review`` / ``clinician`` / ``metadata_only`` selection when present,
    otherwise the single granted value. Omitted attributes mean every granted
    attribute; requested attributes must be a granted subset. Requested consent
    can only make the effective state more restrictive.

    Attributes:
        resource_type: Resource family to read.
        namespace: Optional narrower namespace selection.
        purpose: Optional narrower purpose of use.
        role: Optional narrower role.
        attributes: Optional subset of granted attributes.
        consent_state: Optional more restrictive consent state.
        export_policy: Optional narrower export posture.
        first: Page size.
        after: Opaque cursor bound to the effective query and snapshot.
        fields: Minimum-necessary field projection.
    """

    resource_type: JourneyResourceKind
    namespace: str | None = None
    purpose: str | None = None
    role: str | None = None
    attributes: tuple[str, ...] | None = None
    consent_state: str | None = None
    export_policy: str | None = None
    first: int = DEFAULT_PAGE_SIZE
    after: str | None = field(default=None, repr=False)
    fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("namespace", "purpose", "role", "export_policy"):
            value = getattr(self, name)
            if value is not None:
                _require_controlled(value, name)
        if self.attributes is not None:
            if isinstance(self.attributes, (str, bytes)):
                raise ValueError("access attributes must be a sequence")
            try:
                attributes = tuple(sorted(dict.fromkeys(self.attributes)))
            except TypeError:
                raise ValueError(
                    "access attributes must be controlled identifiers"
                ) from None
            if len(attributes) > MAX_ACCESS_ATTRIBUTES or any(
                not isinstance(item, str) or _CONTROLLED_RE.fullmatch(item) is None
                for item in attributes
            ):
                raise ValueError("access attributes must be bounded identifiers")
            object.__setattr__(self, "attributes", attributes)
        if self.consent_state is not None and (
            not isinstance(self.consent_state, str)
            or self.consent_state not in CONSENT_STATES
        ):
            raise ValueError("consent_state is unsupported")
        probe = JourneyResourceQuery(
            resource_type=self.resource_type,
            first=self.first,
            after=self.after,
            fields=self.fields,
        )
        object.__setattr__(self, "resource_type", probe.resource_type)
        object.__setattr__(self, "fields", probe.fields)


@dataclass(frozen=True, slots=True)
class JourneyAccessBinding:
    """Effective query bound to an authoritative context.

    Attributes:
        query: Query whose access fields come from the authoritative context.
        code: Value-free denial code when binding refused the request.
    """

    query: JourneyResourceQuery
    code: str | None = None


def bind_journey_access(
    request: JourneyAccessRequest,
    context: JourneyAccessContext | None,
    *,
    now: float,
) -> JourneyAccessBinding:
    """Bind a caller request to an authoritative context, narrowing only.

    Args:
        request: Caller-requested selection.
        context: Authoritative context, or ``None`` when none was resolved.
        now: Current Unix timestamp for expiry checks.

    Returns:
        The effective query, plus a denial code when the request asks for
        anything outside the context or the context is missing or expired.
    """

    if type(request) is not JourneyAccessRequest:
        raise TypeError("request must be a JourneyAccessRequest")
    if context is None:
        return refused_journey_access(request, "access_context_missing")
    if type(context) is not JourneyAccessContext:
        raise TypeError("context must be a JourneyAccessContext")
    if type(now) not in {int, float} or not math.isfinite(now):
        raise ValueError("now must be a finite timestamp")
    if context.is_expired(now):
        return refused_journey_access(request, "access_context_expired")
    codes: list[str] = []
    namespace = _select(request.namespace, context.namespaces, "namespace", codes)
    purpose = _select(request.purpose, context.purposes, "purpose", codes)
    role = _select(request.role, context.roles, "role", codes)
    if request.attributes is None:
        attributes = tuple(sorted(context.attributes))
    else:
        attributes = request.attributes
        if not set(attributes).issubset(context.attributes):
            codes.append("attribute_not_granted")
    export_policy = _select(
        request.export_policy, context.export_policies, "export_policy", codes
    )
    query = JourneyResourceQuery(
        resource_type=request.resource_type,
        namespace=namespace,
        purpose=purpose,
        role=role,
        attributes=attributes,
        consent_state=_most_restrictive(context.consent_state, request.consent_state),
        export_policy=export_policy,
        first=request.first,
        after=request.after,
        fields=request.fields,
    )
    return JourneyAccessBinding(query=query, code=codes[0] if codes else None)


def refused_journey_access(
    request: JourneyAccessRequest,
    code: str,
) -> JourneyAccessBinding:
    """Return a denial binding that echoes only the caller's own request.

    Args:
        request: Caller-requested selection.
        code: Stable, value-free denial code.

    Returns:
        A binding whose query carries the request or ``unbound`` placeholders.
    """

    _require_controlled(code, "denial code")
    query = JourneyResourceQuery(
        resource_type=request.resource_type,
        namespace=request.namespace or UNBOUND_SELECTION,
        purpose=request.purpose or UNBOUND_SELECTION,
        role=request.role or UNBOUND_SELECTION,
        attributes=request.attributes or (),
        consent_state=_most_restrictive("unknown", request.consent_state),
        export_policy=request.export_policy or UNBOUND_SELECTION,
        first=request.first,
        after=request.after,
        fields=request.fields,
    )
    return JourneyAccessBinding(query=query, code=code)


@dataclass(frozen=True, slots=True)
class JourneyAccessAuditRecord:
    """Value-free audit record of one Journey access decision.

    Attributes:
        sequence: Monotonic record number within the audit log.
        recorded_at: UTC RFC 3339 timestamp.
        transport: ``rest``, ``graphql``, ``grpc``, or ``python``.
        resource_type: Requested resource family.
        page_state: Returned page state.
        decision_state: Policy decision state.
        code: Page reason code, if any.
        decision_id: Opaque decision identifier from the response policy.
        request_digest: Digest of the effective access request.
        policy_version: Version of the Journey policy.
        credential_type: Caller credential family or ``anonymous``.
        principal_digest: Keyed principal digest, or ``None`` when anonymous.
        context_source: Authority that produced the context, if any.
        context_version: Version of the grant document or hook, if any.
        context_digest: Digest of the granted context, if any.
        schema_version: Audit record schema version.
    """

    sequence: int
    recorded_at: str
    transport: str
    resource_type: str
    page_state: str
    decision_state: str
    code: str | None
    decision_id: str
    request_digest: str
    policy_version: str
    credential_type: str
    principal_digest: str | None
    context_source: str | None
    context_version: str | None
    context_digest: str | None
    schema_version: str = JOURNEY_ACCESS_AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit record."""

        return {
            "code": self.code,
            "context_digest": self.context_digest,
            "context_source": self.context_source,
            "context_version": self.context_version,
            "credential_type": self.credential_type,
            "decision_id": self.decision_id,
            "decision_state": self.decision_state,
            "page_state": self.page_state,
            "policy_version": self.policy_version,
            "principal_digest": self.principal_digest,
            "recorded_at": self.recorded_at,
            "request_digest": self.request_digest,
            "resource_type": self.resource_type,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "transport": self.transport,
        }


class JourneyAccessAuditLog:
    """Bounded in-memory audit log with an optional durable sink.

    A sink failure is raised to the gate, which then withholds records. A lost
    record leaves a gap in ``sequence``.
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_AUDIT_CAPACITY,
        sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        """Create an empty audit log.

        Args:
            capacity: Maximum records retained in memory.
            sink: Optional callable receiving each record dictionary.
        """

        if type(capacity) is not int or capacity < 1:
            raise ValueError("audit capacity must be a positive integer")
        if sink is not None and not callable(sink):
            raise TypeError("audit sink must be callable")
        self._records: deque[JourneyAccessAuditRecord] = deque(maxlen=capacity)
        self._sink = sink
        self._sequence = 0
        self._lock = threading.Lock()

    def record(self, **values: Any) -> JourneyAccessAuditRecord:
        """Append one audit record and forward it to the sink.

        Args:
            **values: Record fields other than ``sequence``.

        Returns:
            The appended record.
        """

        with self._lock:
            self._sequence += 1
            record = JourneyAccessAuditRecord(sequence=self._sequence, **values)
            if self._sink is not None:
                self._sink(record.to_dict())
            self._records.append(record)
            return record

    @property
    def records(self) -> tuple[JourneyAccessAuditRecord, ...]:
        """Return the retained records, oldest first."""

        with self._lock:
            return tuple(self._records)


class JourneyAccessGate:
    """Single authorization path shared by every Journey read transport."""

    def __init__(
        self,
        provider: JourneyAccessContextProvider,
        *,
        profile: str,
        digest_key: bytes | None = None,
        audit: JourneyAccessAuditLog | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Create a gate around one context provider.

        Args:
            provider: Authoritative context provider.
            profile: OpenMed runtime profile. Anonymous reads are allowed only
                under ``dev`` and ``test``.
            digest_key: Principal digest key of at least 32 bytes; a random
                per-process key is used when omitted.
            audit: Audit log; a bounded in-memory log is used when omitted.
            clock: Unix-time source used for expiry and audit timestamps.
        """

        if not callable(getattr(provider, "resolve", None)):
            raise TypeError("provider must implement JourneyAccessContextProvider")
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError("profile must be a non-empty string")
        key = os.urandom(32) if digest_key is None else digest_key
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("digest_key must be at least 32 bytes")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._provider = provider
        self._profile = profile.strip()
        self._digest_key = key
        self.audit = JourneyAccessAuditLog() if audit is None else audit
        self._clock = clock

    @classmethod
    def from_env(
        cls,
        *,
        provider: JourneyAccessContextProvider | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "JourneyAccessGate":
        """Build the service gate from the process environment.

        Args:
            provider: Optional operator provider; the configured grant provider
                is used when omitted.
            environ: Environment mapping; defaults to ``os.environ``.

        Returns:
            A gate bound to ``OPENMED_PROFILE`` (default ``prod``).
        """

        env = os.environ if environ is None else environ
        profile = (env.get(PROFILE_ENV_VAR) or "").strip() or "prod"
        if provider is None:
            provider = ConfiguredJourneyAccessContextProvider.from_env(
                anonymous_development=profile in DEVELOPMENT_PROFILES,
                environ=env,
            )
        return cls(provider, profile=profile)

    @property
    def profile(self) -> str:
        """Return the runtime profile the gate enforces."""

        return self._profile

    @property
    def anonymous_allowed(self) -> bool:
        """Return whether anonymous callers may be resolved at all."""

        return self._profile in DEVELOPMENT_PROFILES

    @property
    def provider(self) -> JourneyAccessContextProvider:
        """Return the authoritative context provider."""

        return self._provider

    def list_resources(
        self,
        catalog: JourneyResourceCatalog,
        request: JourneyAccessRequest,
        *,
        principal: Any,
        policy: JourneyAccessPolicy | None = None,
        transport: str,
    ) -> JourneyResourcePage:
        """Authorize and execute one Journey read for a caller.

        Args:
            catalog: Catalog to read.
            request: Caller-requested, narrowing-only selection.
            principal: Authenticated principal from the transport, or ``None``.
                Anything other than an :class:`AuthPrincipal` is anonymous.
            policy: Active Journey policy envelope.
            transport: Calling transport for the audit record.

        Returns:
            The same typed page the catalog returns, or a typed denied page.
        """

        if type(request) is not JourneyAccessRequest:
            raise TypeError("request must be a JourneyAccessRequest")
        if transport not in JOURNEY_ACCESS_TRANSPORTS:
            raise ValueError("transport is unsupported")
        active_policy = (
            policy if isinstance(policy, JourneyAccessPolicy) else JourneyAccessPolicy()
        )
        caller = principal if type(principal) is AuthPrincipal else None
        credential_type = (
            ANONYMOUS_CREDENTIAL_TYPE
            if caller is None
            else _safe_credential_type(caller.credential_type)
        )
        now = float(self._clock())
        context: JourneyAccessContext | None = None
        if caller is None and not self.anonymous_allowed:
            binding = refused_journey_access(request, "anonymous_access_denied")
        else:
            try:
                resolved = self._provider.resolve(caller, policy=active_policy)
            except Exception:
                binding = refused_journey_access(request, "access_context_unavailable")
            else:
                if resolved is not None and type(resolved) is not JourneyAccessContext:
                    resolved = None
                if resolved is not None and resolved.credential_type != credential_type:
                    binding = refused_journey_access(request, "access_context_mismatch")
                else:
                    context = resolved
                    binding = bind_journey_access(request, context, now=now)
        if binding.code is None:
            page = catalog.list_resources(binding.query, policy=active_policy)
        else:
            page = catalog.deny(binding.query, binding.code, policy=active_policy)
        try:
            self.audit.record(
                recorded_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                transport=transport,
                resource_type=request.resource_type.value,
                page_state=page.state.value,
                decision_state=page.policy.state.value,
                code=page.code,
                decision_id=page.policy.decision_id,
                request_digest=page.policy.request_digest,
                policy_version=page.policy.policy_version,
                credential_type=credential_type,
                principal_digest=(
                    None
                    if caller is None
                    else principal_digest(caller, key=self._digest_key)
                ),
                context_source=None if context is None else context.source,
                context_version=None if context is None else context.context_version,
                context_digest=None if context is None else context.digest,
            )
        except Exception:
            return catalog.deny(
                binding.query, "audit_unavailable", policy=active_policy
            )
        return page


def parse_journey_grants(
    inline: str | None,
    path: str | None = None,
) -> JourneyGrantConfig:
    """Parse the operator Journey grant document.

    The document is ``{"version": "1.0.0", "jwt_claim": null, "grants": [...]}``
    (a bare list of grants is also accepted). Each grant names a
    ``credential_type``, an exact ``principal`` subject, and its ``roles``,
    ``attributes``, ``namespaces``, ``purposes``, ``export_policies``,
    ``consent_state``, and optional ``expires_at``.

    Args:
        inline: Inline JSON from ``OPENMED_SERVICE_JOURNEY_GRANTS``.
        path: File path from ``OPENMED_SERVICE_JOURNEY_GRANTS_FILE``.

    Returns:
        The parsed grant configuration; empty when neither value is set.

    Raises:
        ValueError: With a value-free message when the document is invalid.
    """

    raw_inline = inline if inline is not None and inline.strip() else None
    raw_path = path.strip() if path is not None and path.strip() else None
    if raw_inline is not None and raw_path is not None:
        raise ValueError(
            f"Set only one of {SERVICE_JOURNEY_GRANTS_ENV_VAR} or "
            f"{SERVICE_JOURNEY_GRANTS_FILE_ENV_VAR}"
        )
    if raw_path is not None:
        env_var = SERVICE_JOURNEY_GRANTS_FILE_ENV_VAR
        try:
            raw = Path(raw_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise ValueError(f"{env_var} must name a readable UTF-8 file") from None
    elif raw_inline is not None:
        env_var = SERVICE_JOURNEY_GRANTS_ENV_VAR
        raw = raw_inline
    else:
        return JourneyGrantConfig()
    try:
        payload = json.loads(raw, parse_constant=_reject_json_constant)
    except ValueError:
        raise ValueError(f"{env_var} must contain valid JSON") from None
    if isinstance(payload, list):
        payload = {"grants": payload}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{env_var} must be a JSON object")
    if not set(payload).issubset(_CONFIG_KEYS):
        raise ValueError(f"{env_var} contains an unsupported key")
    version = payload.get("version", JOURNEY_ACCESS_CONTEXT_VERSION)
    if not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"{env_var} version must be semantic")
    jwt_claim = payload.get("jwt_claim")
    if jwt_claim is not None and (
        not isinstance(jwt_claim, str) or _CONTROLLED_RE.fullmatch(jwt_claim) is None
    ):
        raise ValueError(f"{env_var} jwt_claim must be a controlled identifier")
    raw_grants = payload.get("grants", [])
    if not isinstance(raw_grants, list) or len(raw_grants) > MAX_JOURNEY_GRANTS:
        raise ValueError(f"{env_var} grants must be a bounded JSON list")
    grants: list[JourneyAccessGrant] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_grants):
        if not isinstance(item, Mapping) or not set(item).issubset(_GRANT_KEYS):
            raise ValueError(f"{env_var}[{index}] is not a valid Journey grant")
        try:
            grant = JourneyAccessGrant(
                credential_type=item.get("credential_type"),
                principal=item.get("principal"),
                roles=_json_values(item.get("roles")),
                attributes=_json_values(item.get("attributes")),
                namespaces=_json_values(item.get("namespaces")),
                purposes=_json_values(item.get("purposes")),
                export_policies=_json_values(item.get("export_policies")),
                consent_state=item.get("consent_state", "unknown"),
                expires_at=item.get("expires_at"),
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"{env_var}[{index}] is not a valid Journey grant"
            ) from None
        key = (grant.credential_type, grant.principal)
        if key in seen:
            raise ValueError(f"{env_var} contains a duplicate principal grant")
        seen.add(key)
        grants.append(grant)
    return JourneyGrantConfig(
        grants=tuple(grants), jwt_claim=jwt_claim, version=version
    )


def _select(
    requested: str | None,
    granted: frozenset[str],
    dimension: str,
    codes: list[str],
) -> str:
    if requested is not None:
        if requested not in granted:
            codes.append(f"{dimension}_not_granted")
        return requested
    if not granted:
        codes.append(f"{dimension}_not_granted")
        return UNBOUND_SELECTION
    default = _DEFAULT_SELECTIONS[dimension]
    if default in granted:
        return default
    if len(granted) == 1:
        return next(iter(granted))
    codes.append(f"{dimension}_selection_required")
    return UNBOUND_SELECTION


def _most_restrictive(granted: str, requested: str | None) -> str:
    if requested is None:
        return granted
    return max(granted, requested, key=CONSENT_STATES.index)


def _jwt_claim_context(
    raw: Any,
    *,
    context_version: str,
) -> JourneyAccessContext | None:
    if not isinstance(raw, Mapping) or not set(raw).issubset(_GRANT_BODY_KEYS):
        return None
    try:
        return JourneyAccessContext(
            credential_type="jwt",
            source="jwt_claim",
            roles=_json_values(raw.get("roles")),
            attributes=_json_values(raw.get("attributes")),
            namespaces=_json_values(raw.get("namespaces")),
            purposes=_json_values(raw.get("purposes")),
            export_policies=_json_values(raw.get("export_policies")),
            consent_state=raw.get("consent_state", "unknown"),
            expires_at=_optional_timestamp(raw.get("expires_at")),
            context_version=context_version,
        )
    except (TypeError, ValueError):
        return None


def _cap_expiry(context: JourneyAccessContext, raw_exp: Any) -> JourneyAccessContext:
    try:
        exp = _optional_timestamp(raw_exp)
    except ValueError:
        exp = None
    if exp is None:
        return context
    if context.expires_at is None or exp < context.expires_at:
        return replace(context, expires_at=exp)
    return context


def _resolved_consent(
    resolver: JourneyConsentResolver,
    context: JourneyAccessContext,
) -> str:
    scope = JourneyConsentScope(
        credential_type=context.credential_type,
        source=context.source,
        namespaces=context.namespaces,
        purposes=context.purposes,
    )
    try:
        value = resolver(scope)
    except Exception:
        return "unknown"
    return value if isinstance(value, str) and value in CONSENT_STATES else "unknown"


def _json_values(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, frozenset, set)):
        raise ValueError("grant values must be a list")
    return _controlled_set(value, "grant values", limit=MAX_GRANT_VALUES)


def _controlled_set(value: Any, name: str, *, limit: int) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a collection of controlled identifiers")
    items = tuple(value)
    if len(items) > limit or any(
        not isinstance(item, str) or _CONTROLLED_RE.fullmatch(item) is None
        for item in items
    ):
        raise ValueError(f"{name} must be bounded controlled identifiers")
    return frozenset(items)


def _optional_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expires_at must be a timestamp")
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("expires_at must be an RFC 3339 timestamp") from None
        if parsed.tzinfo is None:
            raise ValueError("expires_at must include a UTC offset")
        numeric = parsed.timestamp()
    else:
        raise ValueError("expires_at must be a timestamp")
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError("expires_at must be a finite timestamp")
    return numeric


def _safe_credential_type(value: Any) -> str:
    if isinstance(value, str) and _CONTROLLED_RE.fullmatch(value) is not None:
        return value
    return "unknown"


def _require_controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a controlled identifier")
    return value


def _require_version(value: Any, name: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be semantic")
    return value


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON constants are not allowed")


__all__ = [
    "ANONYMOUS_CREDENTIAL_TYPE",
    "CONSENT_STATES",
    "DEVELOPMENT_PROFILES",
    "JOURNEY_ACCESS_AUDIT_SCHEMA_VERSION",
    "JOURNEY_ACCESS_CONTEXT_VERSION",
    "SERVICE_JOURNEY_GRANTS_ENV_VAR",
    "SERVICE_JOURNEY_GRANTS_FILE_ENV_VAR",
    "UNBOUND_SELECTION",
    "ConfiguredJourneyAccessContextProvider",
    "JourneyAccessAuditLog",
    "JourneyAccessAuditRecord",
    "JourneyAccessBinding",
    "JourneyAccessContext",
    "JourneyAccessContextProvider",
    "JourneyAccessGate",
    "JourneyAccessGrant",
    "JourneyAccessRequest",
    "JourneyConsentResolver",
    "JourneyConsentScope",
    "JourneyGrantConfig",
    "bind_journey_access",
    "development_access_context",
    "parse_journey_grants",
    "refused_journey_access",
]
