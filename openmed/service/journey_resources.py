"""Versioned, policy-aware read contracts for Journey service surfaces.

The catalog in this module is deliberately transport neutral. REST, GraphQL,
SQL projections, and direct Python callers all execute the same bounded query
and receive the same serialized records. The catalog is read-only; mutation is
owned by explicit persistence workflows outside the analytics surface.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, canonical_json

JOURNEY_RESOURCE_SCHEMA_VERSION: Final = "1.0.0"
JOURNEY_RESOURCE_COMPATIBILITY: Final = "same_major"
DEFAULT_PAGE_SIZE: Final = 20
MAX_PAGE_SIZE: Final = 100
MAX_SELECTED_FIELDS: Final = 32
MAX_ACCESS_ATTRIBUTES: Final = 32

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{8,128}$")
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SENSITIVE_KEYS = frozenset(
    {
        "credential",
        "note",
        "payload",
        "phi",
        "raw",
        "raw_text",
        "secret",
        "source_text",
        "text",
        "token",
        "vault",
    }
)


class JourneyResourceKind(str, Enum):
    """Versioned Journey resource families exposed by public read surfaces."""

    ARTIFACT = "artifact"
    JOB = "job"
    FACT = "fact"
    CONFLICT = "conflict"
    JOURNEY = "journey"
    COHORT = "cohort"
    DATASET = "dataset"
    REGISTRY = "registry"
    MEASURE = "measure"
    TRIAL_REVIEW = "trial_review"
    EVIDENCE = "evidence"
    CURRENT_FACT = "current_fact"
    JOURNEY_EVENT = "journey_event"
    MAPPING = "mapping"
    COHORT_RUN = "cohort_run"
    DATASET_MANIFEST = "dataset_manifest"


class JourneyResourceState(str, Enum):
    """Non-lossy state shared by every Journey read surface."""

    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    DENIED = "denied"
    FAILURE = "failure"


RESOURCE_FIELDS: Final[Mapping[JourneyResourceKind, frozenset[str]]] = MappingProxyType(
    {
        JourneyResourceKind.ARTIFACT: frozenset(
            {"content_hash", "media_type", "byte_size", "created_at"}
        ),
        JourneyResourceKind.JOB: frozenset({"status", "recorded_at"}),
        JourneyResourceKind.FACT: frozenset(
            {
                "subject_id",
                "concept",
                "assertion",
                "confidence",
                "evidence_ids",
            }
        ),
        JourneyResourceKind.CONFLICT: frozenset(
            {"fact_ids", "status", "resolution_id"}
        ),
        JourneyResourceKind.JOURNEY: frozenset(
            {"subject_id", "snapshot_id", "event_count"}
        ),
        JourneyResourceKind.COHORT: frozenset({"cohort_id", "status", "member_count"}),
        JourneyResourceKind.DATASET: frozenset(
            {"dataset_id", "snapshot_hash", "row_count"}
        ),
        JourneyResourceKind.REGISTRY: frozenset(
            {"registry_id", "status", "record_count"}
        ),
        JourneyResourceKind.MEASURE: frozenset(
            {"measure_id", "value", "unit", "computed_at"}
        ),
        JourneyResourceKind.TRIAL_REVIEW: frozenset(
            {"trial_id", "status", "criterion_count"}
        ),
        JourneyResourceKind.EVIDENCE: frozenset(
            {"artifact_id", "locator_type", "start", "end"}
        ),
        JourneyResourceKind.CURRENT_FACT: frozenset(
            {"subject_id", "fact_id", "state", "effective_at"}
        ),
        JourneyResourceKind.JOURNEY_EVENT: frozenset(
            {"subject_id", "event_type", "effective_at", "fact_id"}
        ),
        JourneyResourceKind.MAPPING: frozenset(
            {"source_id", "target_id", "mapping_type"}
        ),
        JourneyResourceKind.COHORT_RUN: frozenset(
            {"cohort_id", "status", "member_count", "run_at"}
        ),
        JourneyResourceKind.DATASET_MANIFEST: frozenset(
            {"dataset_id", "snapshot_hash", "row_count", "license_id"}
        ),
    }
)


@dataclass(frozen=True, slots=True)
class JourneyResourceRecord:
    """One value-safe record shared by Python and all public transports."""

    resource_type: JourneyResourceKind
    resource_id: str
    namespace: str
    data: Mapping[str, Any] = field(repr=False)
    state: JourneyResourceState = JourneyResourceState.SUCCESS
    version: int = 1
    revision: int = 1
    schema_version: str = JOURNEY_RESOURCE_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_RESOURCE_COMPATIBILITY
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        resource_type = JourneyResourceKind(self.resource_type)
        state = JourneyResourceState(self.state)
        _require_version(self.schema_version)
        if self.compatibility_policy != JOURNEY_RESOURCE_COMPATIBILITY:
            raise ValueError("unsupported Journey resource compatibility policy")
        if _OPAQUE_ID_RE.fullmatch(self.resource_id) is None:
            raise ValueError("Journey resource_id must be an opaque identifier")
        _require_controlled(self.namespace, "namespace")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Journey resource version must be positive")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Journey resource revision must be positive")
        normalized_data = _freeze_mapping(self.data, "data")
        allowed_fields = RESOURCE_FIELDS[resource_type]
        if not set(normalized_data).issubset(allowed_fields):
            raise ValueError("Journey resource data contains an unsupported field")
        if state is not JourneyResourceState.SUCCESS and normalized_data:
            raise ValueError("non-success Journey resources cannot carry data")
        normalized_extensions = _freeze_mapping(self.extensions, "extensions")
        object.__setattr__(self, "resource_type", resource_type)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "data", normalized_data)
        object.__setattr__(self, "extensions", normalized_extensions)

    def to_dict(self, *, fields: Sequence[str] | None = None) -> dict[str, Any]:
        """Return a deterministic, optionally field-projected representation."""

        selected = set(self.data) if fields is None else set(fields)
        data = {
            key: _plain(value) for key, value in self.data.items() if key in selected
        }
        return {
            "compatibility_policy": self.compatibility_policy,
            "data": data,
            "extensions": _plain(self.extensions),
            "namespace": self.namespace,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "state": self.state.value,
            "version": self.version,
        }

    @property
    def canonical_hash(self) -> str:
        """Return the stable digest of the complete record."""

        return canonical_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JourneyResourceRecord":
        """Load a supported same-major record without discarding extensions."""

        known = {
            "compatibility_policy",
            "data",
            "extensions",
            "namespace",
            "resource_id",
            "resource_type",
            "revision",
            "schema_version",
            "state",
            "version",
        }
        extensions = dict(payload.get("extensions", {}))
        extensions.update(
            {key: value for key, value in payload.items() if key not in known}
        )
        try:
            return cls(
                resource_type=JourneyResourceKind(payload["resource_type"]),
                resource_id=payload["resource_id"],
                namespace=payload["namespace"],
                data=payload.get("data", {}),
                state=JourneyResourceState(payload.get("state", "success")),
                version=payload.get("version", 1),
                revision=payload.get("revision", 1),
                schema_version=payload["schema_version"],
                compatibility_policy=payload.get(
                    "compatibility_policy", JOURNEY_RESOURCE_COMPATIBILITY
                ),
                extensions=extensions,
            )
        except KeyError:
            raise ValueError("Journey resource is missing a required field") from None


@dataclass(frozen=True, slots=True)
class JourneyResourceQuery:
    """Bounded list query shared by Python, REST, GraphQL, and SQL."""

    resource_type: JourneyResourceKind
    namespace: str = "default"
    purpose: str = "care_review"
    role: str = "clinician"
    attributes: tuple[str, ...] = ()
    consent_state: str = "active"
    export_policy: str = "metadata_only"
    first: int = DEFAULT_PAGE_SIZE
    after: str | None = field(default=None, repr=False)
    fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        resource_type = JourneyResourceKind(self.resource_type)
        _require_controlled(self.namespace, "namespace")
        _require_controlled(self.purpose, "purpose")
        _require_controlled(self.role, "role")
        if isinstance(self.attributes, (str, bytes)):
            raise ValueError("access attributes must be a sequence")
        try:
            attributes = tuple(sorted(dict.fromkeys(self.attributes)))
        except TypeError:
            raise ValueError(
                "access attributes must be controlled identifiers"
            ) from None
        if len(attributes) > MAX_ACCESS_ATTRIBUTES or any(
            _CONTROLLED_RE.fullmatch(item) is None for item in attributes
        ):
            raise ValueError("access attributes must be bounded identifiers")
        if not isinstance(self.consent_state, str) or self.consent_state not in {
            "active",
            "unknown",
            "withdrawn",
        }:
            raise ValueError("consent_state is unsupported")
        _require_controlled(self.export_policy, "export_policy")
        if type(self.first) is not int or not 1 <= self.first <= MAX_PAGE_SIZE:
            raise ValueError(f"first must be between 1 and {MAX_PAGE_SIZE}")
        if self.after is not None and (
            not isinstance(self.after, str) or len(self.after) > 2048
        ):
            raise ValueError("after must be a bounded cursor")
        fields = tuple(sorted(dict.fromkeys(self.fields)))
        if len(fields) > MAX_SELECTED_FIELDS:
            raise ValueError("too many Journey resource fields were requested")
        allowed = RESOURCE_FIELDS[resource_type]
        if any(_CONTROLLED_RE.fullmatch(name) is None for name in fields):
            raise ValueError("Journey resource fields must be controlled identifiers")
        if not set(fields).issubset(allowed):
            raise ValueError("Journey resource query requested an unsupported field")
        object.__setattr__(self, "resource_type", resource_type)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "fields", fields)

    @property
    def access_request_digest(self) -> str:
        """Return a stable value-free digest of policy-relevant inputs."""

        return canonical_digest(
            {
                "attributes": list(self.attributes),
                "consent_state": self.consent_state,
                "export_policy": self.export_policy,
                "fields": list(self.fields),
                "namespace": self.namespace,
                "purpose": self.purpose,
                "resource_type": self.resource_type.value,
                "role": self.role,
            }
        )


@dataclass(frozen=True, slots=True)
class JourneyPolicyDecision:
    """Inspectible minimum-necessary response policy decision."""

    state: JourneyResourceState
    namespace: str
    purpose: str
    role: str
    attributes: tuple[str, ...]
    consent_state: str
    export_policy: str
    allowed_fields: tuple[str, ...]
    decision_id: str
    request_digest: str
    code: str | None = None
    policy_version: str = JOURNEY_RESOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        state = JourneyResourceState(self.state)
        _require_controlled(self.namespace, "namespace")
        _require_controlled(self.purpose, "purpose")
        _require_controlled(self.role, "role")
        if isinstance(self.attributes, (str, bytes)):
            raise ValueError("policy attributes must be a sequence")
        try:
            attributes = tuple(sorted(dict.fromkeys(self.attributes)))
        except TypeError:
            raise ValueError(
                "policy attributes must be controlled identifiers"
            ) from None
        if len(attributes) > MAX_ACCESS_ATTRIBUTES or any(
            _CONTROLLED_RE.fullmatch(item) is None for item in attributes
        ):
            raise ValueError("policy attributes must be bounded identifiers")
        if not isinstance(self.consent_state, str) or self.consent_state not in {
            "active",
            "unknown",
            "withdrawn",
        }:
            raise ValueError("policy consent state is unsupported")
        _require_controlled(self.export_policy, "export_policy")
        if _OPAQUE_ID_RE.fullmatch(self.decision_id) is None:
            raise ValueError("policy decision_id must be an opaque identifier")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.request_digest) is None:
            raise ValueError("policy request_digest must be a SHA-256 digest")
        _require_version(self.policy_version)
        fields = tuple(sorted(dict.fromkeys(self.allowed_fields)))
        if len(fields) > MAX_SELECTED_FIELDS or any(
            _CONTROLLED_RE.fullmatch(item) is None for item in fields
        ):
            raise ValueError("policy fields must be bounded controlled identifiers")
        if state not in {
            JourneyResourceState.SUCCESS,
            JourneyResourceState.DENIED,
        }:
            raise ValueError("policy decision state must be success or denied")
        if state is JourneyResourceState.SUCCESS and self.code is not None:
            raise ValueError("successful policy decisions cannot carry a code")
        if state is JourneyResourceState.DENIED and self.code is None:
            raise ValueError("denied policy decisions require a code")
        if state is JourneyResourceState.DENIED and fields:
            raise ValueError("denied policy decisions cannot allow fields")
        if self.code is not None:
            _require_controlled(self.code, "policy code")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "allowed_fields", fields)

    def to_dict(self) -> dict[str, Any]:
        """Return a transport-safe policy representation."""

        return {
            "allowed_fields": list(self.allowed_fields),
            "code": self.code,
            "consent_state": self.consent_state,
            "decision_id": self.decision_id,
            "export_policy": self.export_policy,
            "namespace": self.namespace,
            "policy_version": self.policy_version,
            "purpose": self.purpose,
            "request_digest": self.request_digest,
            "role": self.role,
            "state": self.state.value,
            "attributes": list(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class JourneyAccessPolicy:
    """Namespace, purpose, and field policy for read-only Journey queries."""

    allowed_namespaces: frozenset[str] = frozenset({"default"})
    allowed_purposes: frozenset[str] = frozenset(
        {"analytics", "care_review", "quality"}
    )
    allowed_roles: frozenset[str] = frozenset(
        {"clinician", "data_steward", "privacy_officer", "researcher"}
    )
    allowed_export_policies: frozenset[str] = frozenset({"metadata_only"})
    required_attributes_by_resource: Mapping[JourneyResourceKind, frozenset[str]] = (
        field(default_factory=dict)
    )
    fields_by_resource: Mapping[JourneyResourceKind, frozenset[str]] = field(
        default_factory=lambda: RESOURCE_FIELDS
    )
    policy_version: str = JOURNEY_RESOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        namespaces = frozenset(self.allowed_namespaces)
        purposes = frozenset(self.allowed_purposes)
        roles = frozenset(self.allowed_roles)
        export_policies = frozenset(self.allowed_export_policies)
        if not namespaces or any(
            _CONTROLLED_RE.fullmatch(item) is None for item in namespaces
        ):
            raise ValueError("allowed namespaces must be controlled identifiers")
        if not purposes or any(
            _CONTROLLED_RE.fullmatch(item) is None for item in purposes
        ):
            raise ValueError("allowed purposes must be controlled identifiers")
        if not roles or any(_CONTROLLED_RE.fullmatch(item) is None for item in roles):
            raise ValueError("allowed roles must be controlled identifiers")
        if not export_policies or any(
            _CONTROLLED_RE.fullmatch(item) is None for item in export_policies
        ):
            raise ValueError("allowed export policies must be controlled identifiers")
        normalized_fields: dict[JourneyResourceKind, frozenset[str]] = {}
        for raw_kind, raw_fields in self.fields_by_resource.items():
            kind = JourneyResourceKind(raw_kind)
            fields = frozenset(raw_fields)
            if not fields.issubset(RESOURCE_FIELDS[kind]):
                raise ValueError("policy contains an unsupported resource field")
            normalized_fields[kind] = fields
        _require_version(self.policy_version)
        required_attributes: dict[JourneyResourceKind, frozenset[str]] = {}
        for raw_kind, raw_attributes in self.required_attributes_by_resource.items():
            kind = JourneyResourceKind(raw_kind)
            if isinstance(raw_attributes, (str, bytes)):
                raise ValueError("required attributes must be a collection")
            attributes = frozenset(raw_attributes)
            if len(attributes) > MAX_ACCESS_ATTRIBUTES or any(
                _CONTROLLED_RE.fullmatch(item) is None for item in attributes
            ):
                raise ValueError("required attributes must be controlled identifiers")
            required_attributes[kind] = attributes
        object.__setattr__(self, "allowed_namespaces", namespaces)
        object.__setattr__(self, "allowed_purposes", purposes)
        object.__setattr__(self, "allowed_roles", roles)
        object.__setattr__(self, "allowed_export_policies", export_policies)
        object.__setattr__(
            self,
            "fields_by_resource",
            MappingProxyType(normalized_fields),
        )
        object.__setattr__(
            self,
            "required_attributes_by_resource",
            MappingProxyType(required_attributes),
        )

    def decide(self, query: JourneyResourceQuery) -> JourneyPolicyDecision:
        """Return a typed allow or deny decision without leaking record values."""

        if query.namespace not in self.allowed_namespaces:
            return self._denied(query, "namespace_denied")
        if query.purpose not in self.allowed_purposes:
            return self._denied(query, "purpose_denied")
        if query.role not in self.allowed_roles:
            return self._denied(query, "role_denied")
        if query.consent_state != "active":
            return self._denied(query, f"consent_{query.consent_state}")
        if query.export_policy not in self.allowed_export_policies:
            return self._denied(query, "export_policy_denied")
        required = self.required_attributes_by_resource.get(
            query.resource_type, frozenset()
        )
        if not required.issubset(query.attributes):
            return self._denied(query, "attribute_denied")
        allowed = self.fields_by_resource.get(query.resource_type, frozenset())
        selected = query.fields or tuple(sorted(allowed))
        if not set(selected).issubset(allowed):
            return self._denied(query, "field_denied")
        decision_id, request_digest = _policy_decision_identity(
            query,
            state=JourneyResourceState.SUCCESS,
            allowed_fields=tuple(selected),
            code=None,
            policy_version=self.policy_version,
        )
        return JourneyPolicyDecision(
            state=JourneyResourceState.SUCCESS,
            namespace=query.namespace,
            purpose=query.purpose,
            role=query.role,
            attributes=query.attributes,
            consent_state=query.consent_state,
            export_policy=query.export_policy,
            allowed_fields=tuple(selected),
            decision_id=decision_id,
            request_digest=request_digest,
            policy_version=self.policy_version,
        )

    def _denied(self, query: JourneyResourceQuery, code: str) -> JourneyPolicyDecision:
        decision_id, request_digest = _policy_decision_identity(
            query,
            state=JourneyResourceState.DENIED,
            allowed_fields=(),
            code=code,
            policy_version=self.policy_version,
        )
        return JourneyPolicyDecision(
            state=JourneyResourceState.DENIED,
            namespace=query.namespace,
            purpose=query.purpose,
            role=query.role,
            attributes=query.attributes,
            consent_state=query.consent_state,
            export_policy=query.export_policy,
            allowed_fields=(),
            decision_id=decision_id,
            request_digest=request_digest,
            code=code,
            policy_version=self.policy_version,
        )


@dataclass(frozen=True, slots=True)
class JourneyPageInfo:
    """Bounded cursor metadata shared across transports."""

    has_next_page: bool
    end_cursor: str | None
    page_size: int
    snapshot_digest: str

    def __post_init__(self) -> None:
        if type(self.page_size) is not int or not 0 <= self.page_size <= MAX_PAGE_SIZE:
            raise ValueError("Journey page size is outside the bounded range")
        if self.has_next_page != (self.end_cursor is not None):
            raise ValueError("Journey page cursor does not match has_next_page")
        if not isinstance(self.snapshot_digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.snapshot_digest
        ):
            raise ValueError("Journey page snapshot must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_cursor": self.end_cursor,
            "has_next_page": self.has_next_page,
            "page_size": self.page_size,
            "snapshot_digest": self.snapshot_digest,
        }


@dataclass(frozen=True, slots=True)
class JourneyResourcePage:
    """One bounded response that preserves non-success states."""

    state: JourneyResourceState
    resources: tuple[Mapping[str, Any], ...]
    page_info: JourneyPageInfo
    policy: JourneyPolicyDecision
    code: str | None = None
    schema_version: str = JOURNEY_RESOURCE_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_RESOURCE_COMPATIBILITY

    def __post_init__(self) -> None:
        state = JourneyResourceState(self.state)
        _require_version(self.schema_version)
        if self.compatibility_policy != JOURNEY_RESOURCE_COMPATIBILITY:
            raise ValueError("unsupported Journey page compatibility policy")
        if len(self.resources) > MAX_PAGE_SIZE:
            raise ValueError("Journey page exceeds the maximum resource count")
        if state is JourneyResourceState.SUCCESS and self.code is not None:
            raise ValueError("successful Journey pages cannot carry a code")
        if state is not JourneyResourceState.SUCCESS and self.code is None:
            raise ValueError("non-success Journey pages require a code")
        if (
            state
            in {
                JourneyResourceState.DENIED,
                JourneyResourceState.EMPTY,
                JourneyResourceState.FAILURE,
                JourneyResourceState.UNSUPPORTED,
            }
            and self.resources
        ):
            raise ValueError("this Journey page state cannot carry resources")
        if self.code is not None:
            _require_controlled(self.code, "page code")
        object.__setattr__(self, "state", state)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical cross-transport response payload."""

        return {
            "code": self.code,
            "compatibility_policy": self.compatibility_policy,
            "page_info": self.page_info.to_dict(),
            "policy": self.policy.to_dict(),
            "resources": [_plain(item) for item in self.resources],
            "schema_version": self.schema_version,
            "state": self.state.value,
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_digest(self.to_dict())


class JourneyResourceCatalog:
    """Immutable in-memory projection used identically by every read surface."""

    def __init__(self, records: Iterable[JourneyResourceRecord] = ()) -> None:
        ordered = tuple(
            sorted(
                records,
                key=lambda item: (
                    item.resource_type.value,
                    item.namespace,
                    item.resource_id,
                    item.version,
                    item.revision,
                ),
            )
        )
        identities = [
            (item.resource_type, item.namespace, item.resource_id, item.version)
            for item in ordered
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Journey resource catalog contains duplicate versions")
        self._records = ordered
        self._snapshot_digest = canonical_digest(
            [item.canonical_hash for item in ordered]
        )

    @property
    def snapshot_digest(self) -> str:
        """Return the exact immutable catalog snapshot digest."""

        return self._snapshot_digest

    def list_resources(
        self,
        query: JourneyResourceQuery,
        *,
        policy: JourneyAccessPolicy | None = None,
    ) -> JourneyResourcePage:
        """Execute one bounded read with cursor and policy enforcement."""

        active_policy = policy or JourneyAccessPolicy()
        decision = active_policy.decide(query)
        if decision.state is JourneyResourceState.DENIED:
            return self._terminal_page(
                query,
                decision,
                JourneyResourceState.DENIED,
                decision.code or "policy_denied",
            )
        try:
            offset = self._decode_cursor(query.after, query)
        except ValueError as exc:
            return self._terminal_page(
                query,
                decision,
                JourneyResourceState.FAILURE,
                str(exc),
            )
        matching = tuple(
            item
            for item in self._records
            if item.resource_type is query.resource_type
            and item.namespace == query.namespace
        )
        if offset > len(matching):
            return self._terminal_page(
                query,
                decision,
                JourneyResourceState.FAILURE,
                "cursor_offset_invalid",
            )
        selected = matching[offset : offset + query.first]
        if not selected:
            return self._terminal_page(
                query,
                decision,
                JourneyResourceState.EMPTY,
                "no_resources",
            )
        projected = tuple(
            item.to_dict(fields=decision.allowed_fields) for item in selected
        )
        state, code = _aggregate_page_state(selected)
        if state in {
            JourneyResourceState.FAILURE,
            JourneyResourceState.UNSUPPORTED,
        }:
            return self._terminal_page(
                query,
                decision,
                state,
                code or f"resource_{state.value}",
            )
        next_offset = offset + len(selected)
        has_next_page = next_offset < len(matching)
        return JourneyResourcePage(
            state=state,
            code=code,
            resources=projected,
            page_info=JourneyPageInfo(
                has_next_page=has_next_page,
                end_cursor=(
                    self.cursor_for_offset(query, next_offset)
                    if has_next_page
                    else None
                ),
                page_size=len(projected),
                snapshot_digest=self.snapshot_digest,
            ),
            policy=decision,
        )

    def cursor_for_offset(self, query: JourneyResourceQuery, offset: int) -> str:
        """Build a deterministic cursor tied to query and snapshot semantics."""

        if type(offset) is not int or offset < 0:
            raise ValueError("cursor offset must be non-negative")
        body = {
            "attributes": list(query.attributes),
            "consent_state": query.consent_state,
            "export_policy": query.export_policy,
            "fields": list(query.fields),
            "namespace": query.namespace,
            "offset": offset,
            "purpose": query.purpose,
            "resource_type": query.resource_type.value,
            "role": query.role,
            "snapshot_digest": self.snapshot_digest,
            "version": 1,
        }
        body["digest"] = canonical_digest(body)
        return (
            base64.urlsafe_b64encode(canonical_json(body).encode("ascii"))
            .decode("ascii")
            .rstrip("=")
        )

    def _decode_cursor(self, cursor: str | None, query: JourneyResourceQuery) -> int:
        if cursor is None:
            return 0
        try:
            padding = "=" * (-len(cursor) % 4)
            payload = json.loads(
                base64.urlsafe_b64decode(cursor + padding).decode("ascii")
            )
        except (UnicodeError, ValueError, json.JSONDecodeError):
            raise ValueError("cursor_invalid") from None
        if not isinstance(payload, dict):
            raise ValueError("cursor_invalid")
        digest = payload.pop("digest", None)
        if digest != canonical_digest(payload):
            raise ValueError("cursor_integrity_failed")
        expected = {
            "attributes": list(query.attributes),
            "consent_state": query.consent_state,
            "export_policy": query.export_policy,
            "fields": list(query.fields),
            "namespace": query.namespace,
            "purpose": query.purpose,
            "resource_type": query.resource_type.value,
            "role": query.role,
            "snapshot_digest": self.snapshot_digest,
            "version": 1,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                code = (
                    "cursor_snapshot_changed"
                    if key == "snapshot_digest"
                    else "cursor_query_mismatch"
                )
                raise ValueError(code)
        offset = payload.get("offset")
        if type(offset) is not int or offset < 0:
            raise ValueError("cursor_offset_invalid")
        return offset

    def _terminal_page(
        self,
        query: JourneyResourceQuery,
        decision: JourneyPolicyDecision,
        state: JourneyResourceState,
        code: str,
    ) -> JourneyResourcePage:
        del query
        return JourneyResourcePage(
            state=state,
            code=code,
            resources=(),
            page_info=JourneyPageInfo(
                has_next_page=False,
                end_cursor=None,
                page_size=0,
                snapshot_digest=self.snapshot_digest,
            ),
            policy=decision,
        )


def parse_resource_fields(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize REST, GraphQL, or client field selection."""

    if value is None:
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    fields = tuple(str(item).strip() for item in raw if str(item).strip())
    return tuple(sorted(dict.fromkeys(fields)))


def parse_access_attributes(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize REST, GraphQL, SQL, or client policy attributes."""

    if value is None:
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    attributes = tuple(str(item).strip() for item in raw if str(item).strip())
    return tuple(sorted(dict.fromkeys(attributes)))


def migrate_resource_record(
    payload: Mapping[str, Any],
    *,
    target_version: str = JOURNEY_RESOURCE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Migrate a same-major resource without losing data or extensions."""

    source_version = payload.get("schema_version")
    _require_version(source_version)
    _require_version(target_version)
    if str(source_version).split(".", 1)[0] != target_version.split(".", 1)[0]:
        raise ValueError("unsupported Journey resource schema major")
    record = JourneyResourceRecord.from_dict(payload)
    migrated = record.to_dict()
    migrated["schema_version"] = target_version
    return migrated


def _aggregate_page_state(
    records: Sequence[JourneyResourceRecord],
) -> tuple[JourneyResourceState, str | None]:
    states = {item.state for item in records}
    if states == {JourneyResourceState.SUCCESS}:
        return JourneyResourceState.SUCCESS, None
    if len(states) == 1:
        state = next(iter(states))
        return state, f"resource_{state.value}"
    return JourneyResourceState.PARTIAL, "mixed_resource_states"


def _policy_decision_identity(
    query: JourneyResourceQuery,
    *,
    state: JourneyResourceState,
    allowed_fields: tuple[str, ...],
    code: str | None,
    policy_version: str,
) -> tuple[str, str]:
    request_digest = query.access_request_digest
    digest = canonical_digest(
        {
            "allowed_fields": list(allowed_fields),
            "code": code,
            "policy_version": policy_version,
            "request_digest": request_digest,
            "state": state.value,
        }
    )
    return f"decision_{digest.removeprefix('sha256:')[:32]}", request_digest


def _require_version(value: Any) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise ValueError("Journey resource schema version must be semantic")
    if value.split(".", 1)[0] != JOURNEY_RESOURCE_SCHEMA_VERSION.split(".", 1)[0]:
        raise ValueError("unsupported Journey resource schema major")
    return value


def _require_controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a controlled identifier")
    return value


def _freeze_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or _CONTROLLED_RE.fullmatch(key) is None:
            raise ValueError(f"{name} keys must be controlled identifiers")
        if key.casefold() in _SENSITIVE_KEYS:
            raise ValueError(f"{name} contains a sensitive field")
        normalized[key] = _freeze_json(item, name)
    return MappingProxyType(dict(sorted(normalized.items())))


def _freeze_json(value: Any, name: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value, name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, name) for item in value)
    raise ValueError(f"{name} must contain JSON-compatible values")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "JOURNEY_RESOURCE_COMPATIBILITY",
    "JOURNEY_RESOURCE_SCHEMA_VERSION",
    "MAX_PAGE_SIZE",
    "MAX_ACCESS_ATTRIBUTES",
    "RESOURCE_FIELDS",
    "JourneyAccessPolicy",
    "JourneyPageInfo",
    "JourneyPolicyDecision",
    "JourneyResourceCatalog",
    "JourneyResourceKind",
    "JourneyResourcePage",
    "JourneyResourceQuery",
    "JourneyResourceRecord",
    "JourneyResourceState",
    "migrate_resource_record",
    "parse_access_attributes",
    "parse_resource_fields",
]
