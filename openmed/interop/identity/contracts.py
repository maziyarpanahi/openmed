"""Versioned, value-safe contracts for patient and encounter resolution."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any, ClassVar, TypeVar

from openmed.clinical.journey_contracts import canonical_digest, canonical_json

IDENTITY_SCHEMA_VERSION = "1.0.0"
IDENTITY_SCHEMA_PACKAGE = "openmed.core.schemas.json"
IDENTITY_SCHEMA_NAMES = (
    "source_key",
    "request",
    "evidence",
    "resolution",
    "link",
    "review_decision",
)
IDENTITY_STATES = frozenset({"matched", "unmatched", "ambiguous", "conflict"})
ENTITY_TYPES = frozenset({"patient", "encounter"})
REVIEW_ACTIONS = frozenset({"confirm_match", "confirm_unmatched", "merge", "split"})

_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+){0,2}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)

R = TypeVar("R", bound="IdentityRecord")


class IdentityContractError(ValueError):
    """Value-safe validation failure for identity-resolution records."""


class IdentityRecord:
    """Strict deterministic serialization shared by identity records."""

    _fields: ClassVar[frozenset[str]]

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible metadata."""

        raise NotImplementedError

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls: type[R], payload: Mapping[str, Any]) -> R:
        """Parse one strict mapping."""

        raise NotImplementedError

    @classmethod
    def from_json(cls: type[R], payload: str) -> R:
        """Parse strict JSON with duplicates and non-finite values rejected."""

        return cls.from_dict(_strict_json(payload, cls.__name__))


@dataclass(frozen=True, slots=True, order=True)
class SourceIdentityKey(IdentityRecord):
    """Opaque source-local key with no raw patient identifier."""

    entity_type: str
    source_id: str
    local_key: str
    schema_version: str = IDENTITY_SCHEMA_VERSION

    _fields = frozenset({"entity_type", "local_key", "schema_version", "source_id"})

    def __post_init__(self) -> None:
        _entity_type(self.entity_type)
        _opaque_id(self.source_id, "source_id")
        _opaque_id(self.local_key, "local_key")
        _schema(self.schema_version)

    @property
    def fingerprint(self) -> str:
        """Return the stable fingerprint used for storage joins."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        """Return the opaque key mapping."""

        return {
            "entity_type": self.entity_type,
            "local_key": self.local_key,
            "schema_version": self.schema_version,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceIdentityKey":
        """Parse one strict source-local key."""

        data = _strict(payload, cls._fields, "source identity key")
        try:
            return cls(**data)
        except KeyError:
            raise IdentityContractError(
                "source identity key is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class IdentityResolutionRequest(IdentityRecord):
    """One deterministic patient or encounter resolution request."""

    request_id: str
    entity_type: str
    source_keys: tuple[SourceIdentityKey, ...]
    purpose: str
    role: str
    attributes: tuple[str, ...]
    policy_id: str
    policy_version: str
    requested_at: str
    schema_version: str = IDENTITY_SCHEMA_VERSION

    _fields = frozenset(
        {
            "attributes",
            "entity_type",
            "policy_id",
            "policy_version",
            "purpose",
            "request_id",
            "requested_at",
            "role",
            "schema_version",
            "source_keys",
        }
    )

    def __post_init__(self) -> None:
        _opaque_id(self.request_id, "request_id")
        _entity_type(self.entity_type)
        if not isinstance(self.source_keys, tuple) or not self.source_keys:
            raise IdentityContractError("source_keys must be a non-empty tuple")
        for key in self.source_keys:
            if not isinstance(key, SourceIdentityKey):
                raise IdentityContractError("source_keys must contain source keys")
            if key.entity_type != self.entity_type:
                raise IdentityContractError("source key entity type differs")
        normalized_keys = tuple(sorted(set(self.source_keys)))
        if len(normalized_keys) != len(self.source_keys):
            raise IdentityContractError("source_keys must be unique")
        object.__setattr__(self, "source_keys", normalized_keys)
        _controlled(self.purpose, "purpose")
        _controlled(self.role, "role")
        object.__setattr__(
            self,
            "attributes",
            _controlled_tuple(self.attributes, "attributes"),
        )
        _controlled(self.policy_id, "policy_id")
        _version(self.policy_version, "policy_version")
        _timestamp(self.requested_at, "requested_at")
        _schema(self.schema_version)

    @property
    def request_digest(self) -> str:
        """Return deterministic resolution identity independent of request ID/time."""

        return canonical_digest(
            {
                "attributes": list(self.attributes),
                "entity_type": self.entity_type,
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "purpose": self.purpose,
                "role": self.role,
                "schema_version": self.schema_version,
                "source_keys": [key.to_dict() for key in self.source_keys],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the request without raw identifiers."""

        return {
            "attributes": list(self.attributes),
            "entity_type": self.entity_type,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "purpose": self.purpose,
            "request_id": self.request_id,
            "requested_at": self.requested_at,
            "role": self.role,
            "schema_version": self.schema_version,
            "source_keys": [key.to_dict() for key in self.source_keys],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdentityResolutionRequest":
        """Parse one strict resolution request."""

        data = _strict(payload, cls._fields, "identity resolution request")
        try:
            keys = data["source_keys"]
            attributes = data["attributes"]
            if not isinstance(keys, list) or not isinstance(attributes, list):
                raise IdentityContractError("request arrays are invalid")
            return cls(
                request_id=data["request_id"],
                entity_type=data["entity_type"],
                source_keys=tuple(SourceIdentityKey.from_dict(item) for item in keys),
                purpose=data["purpose"],
                role=data["role"],
                attributes=tuple(attributes),
                policy_id=data["policy_id"],
                policy_version=data["policy_version"],
                requested_at=data["requested_at"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IdentityContractError(
                "identity resolution request is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True, order=True)
class IdentityEvidence(IdentityRecord):
    """Digest-only provenance for one candidate link."""

    evidence_id: str
    evidence_type: str
    source_key_fingerprint: str
    candidate_key: str
    evidence_digest: str
    method: str
    policy_id: str
    policy_version: str
    schema_version: str = IDENTITY_SCHEMA_VERSION

    _fields = frozenset(
        {
            "candidate_key",
            "evidence_digest",
            "evidence_id",
            "evidence_type",
            "method",
            "policy_id",
            "policy_version",
            "schema_version",
            "source_key_fingerprint",
        }
    )

    def __post_init__(self) -> None:
        _opaque_id(self.evidence_id, "evidence_id")
        _controlled(self.evidence_type, "evidence_type")
        _digest(self.source_key_fingerprint, "source_key_fingerprint")
        _opaque_id(self.candidate_key, "candidate_key")
        _digest(self.evidence_digest, "evidence_digest")
        if self.method not in {"exact", "probabilistic"}:
            raise IdentityContractError("evidence method is unsupported")
        _controlled(self.policy_id, "policy_id")
        _version(self.policy_version, "policy_version")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, str]:
        """Return digest-only evidence metadata."""

        return {
            "candidate_key": self.candidate_key,
            "evidence_digest": self.evidence_digest,
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "method": self.method,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            "source_key_fingerprint": self.source_key_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdentityEvidence":
        """Parse one strict evidence record."""

        data = _strict(payload, cls._fields, "identity evidence")
        try:
            return cls(**data)
        except KeyError:
            raise IdentityContractError(
                "identity evidence is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class IdentityResolution(IdentityRecord):
    """Matched, unmatched, ambiguous, or conflicting resolution outcome."""

    resolution_id: str
    request_digest: str
    entity_type: str
    source_keys: tuple[SourceIdentityKey, ...]
    state: str
    canonical_key: str | None
    candidate_keys: tuple[str, ...]
    evidence: tuple[IdentityEvidence, ...]
    resolver_id: str
    resolver_version: str
    policy_id: str
    policy_version: str
    resolved_at: str
    review_required: bool
    schema_version: str = IDENTITY_SCHEMA_VERSION

    _fields = frozenset(
        {
            "candidate_keys",
            "canonical_key",
            "entity_type",
            "evidence",
            "policy_id",
            "policy_version",
            "request_digest",
            "resolution_id",
            "resolved_at",
            "resolver_id",
            "resolver_version",
            "review_required",
            "schema_version",
            "source_keys",
            "state",
        }
    )

    def __post_init__(self) -> None:
        _opaque_id(self.resolution_id, "resolution_id")
        _digest(self.request_digest, "request_digest")
        _entity_type(self.entity_type)
        if not isinstance(self.source_keys, tuple) or not self.source_keys:
            raise IdentityContractError("source_keys must be a non-empty tuple")
        for key in self.source_keys:
            if (
                not isinstance(key, SourceIdentityKey)
                or key.entity_type != self.entity_type
            ):
                raise IdentityContractError("resolution source key is invalid")
        normalized_source_keys = tuple(sorted(set(self.source_keys)))
        if len(normalized_source_keys) != len(self.source_keys):
            raise IdentityContractError("resolution source_keys must be unique")
        object.__setattr__(self, "source_keys", normalized_source_keys)
        if self.state not in IDENTITY_STATES:
            raise IdentityContractError("identity state is unsupported")
        if self.canonical_key is not None:
            _opaque_id(self.canonical_key, "canonical_key")
        object.__setattr__(
            self,
            "candidate_keys",
            _opaque_tuple(self.candidate_keys, "candidate_keys"),
        )
        if not isinstance(self.evidence, tuple):
            raise IdentityContractError("evidence must be a tuple")
        if not all(isinstance(item, IdentityEvidence) for item in self.evidence):
            raise IdentityContractError("evidence entries are invalid")
        normalized_evidence = tuple(sorted(set(self.evidence)))
        if len(normalized_evidence) != len(self.evidence):
            raise IdentityContractError("resolution evidence must be unique")
        object.__setattr__(self, "evidence", normalized_evidence)
        _controlled(self.resolver_id, "resolver_id")
        _version(self.resolver_version, "resolver_version")
        _controlled(self.policy_id, "policy_id")
        _version(self.policy_version, "policy_version")
        _timestamp(self.resolved_at, "resolved_at")
        if type(self.review_required) is not bool:
            raise IdentityContractError("review_required must be boolean")
        _schema(self.schema_version)
        if self.state == "matched":
            if self.canonical_key is None or self.candidate_keys != (
                self.canonical_key,
            ):
                raise IdentityContractError("matched resolution requires one candidate")
            if self.review_required:
                raise IdentityContractError("matched resolution cannot require review")
        elif self.state == "unmatched":
            if self.canonical_key is not None or self.candidate_keys:
                raise IdentityContractError(
                    "unmatched resolution cannot have candidates"
                )
            if self.review_required:
                raise IdentityContractError(
                    "unmatched resolution cannot require review"
                )
        else:
            if self.canonical_key is not None or not self.candidate_keys:
                raise IdentityContractError("uncertain resolution requires candidates")
            if not self.review_required:
                raise IdentityContractError("uncertain resolution requires review")

    def to_dict(self) -> dict[str, Any]:
        """Return the value-safe resolution mapping."""

        return {
            "candidate_keys": list(self.candidate_keys),
            "canonical_key": self.canonical_key,
            "entity_type": self.entity_type,
            "evidence": [item.to_dict() for item in self.evidence],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "request_digest": self.request_digest,
            "resolution_id": self.resolution_id,
            "resolved_at": self.resolved_at,
            "resolver_id": self.resolver_id,
            "resolver_version": self.resolver_version,
            "review_required": self.review_required,
            "schema_version": self.schema_version,
            "source_keys": [key.to_dict() for key in self.source_keys],
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdentityResolution":
        """Parse one strict identity resolution."""

        data = _strict(payload, cls._fields, "identity resolution")
        try:
            source_keys = data["source_keys"]
            candidates = data["candidate_keys"]
            evidence = data["evidence"]
            if not all(
                isinstance(value, list) for value in (source_keys, candidates, evidence)
            ):
                raise IdentityContractError("resolution arrays are invalid")
            return cls(
                resolution_id=data["resolution_id"],
                request_digest=data["request_digest"],
                entity_type=data["entity_type"],
                source_keys=tuple(
                    SourceIdentityKey.from_dict(item) for item in source_keys
                ),
                state=data["state"],
                canonical_key=data["canonical_key"],
                candidate_keys=tuple(candidates),
                evidence=tuple(IdentityEvidence.from_dict(item) for item in evidence),
                resolver_id=data["resolver_id"],
                resolver_version=data["resolver_version"],
                policy_id=data["policy_id"],
                policy_version=data["policy_version"],
                resolved_at=data["resolved_at"],
                review_required=data["review_required"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IdentityContractError(
                "identity resolution is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class IdentityLink(IdentityRecord):
    """Versioned exact link from a source-local key to a canonical key."""

    link_id: str
    source_key: SourceIdentityKey
    canonical_key: str
    evidence_digest: str
    policy_id: str
    policy_version: str
    recorded_at: str
    active: bool = True
    schema_version: str = IDENTITY_SCHEMA_VERSION

    _fields = frozenset(
        {
            "active",
            "canonical_key",
            "evidence_digest",
            "link_id",
            "policy_id",
            "policy_version",
            "recorded_at",
            "schema_version",
            "source_key",
        }
    )

    def __post_init__(self) -> None:
        _opaque_id(self.link_id, "link_id")
        if not isinstance(self.source_key, SourceIdentityKey):
            raise IdentityContractError("source_key is invalid")
        _opaque_id(self.canonical_key, "canonical_key")
        _digest(self.evidence_digest, "evidence_digest")
        _controlled(self.policy_id, "policy_id")
        _version(self.policy_version, "policy_version")
        _timestamp(self.recorded_at, "recorded_at")
        if type(self.active) is not bool:
            raise IdentityContractError("active must be boolean")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return exact-link metadata."""

        return {
            "active": self.active,
            "canonical_key": self.canonical_key,
            "evidence_digest": self.evidence_digest,
            "link_id": self.link_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "recorded_at": self.recorded_at,
            "schema_version": self.schema_version,
            "source_key": self.source_key.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdentityLink":
        """Parse one strict exact link."""

        data = _strict(payload, cls._fields, "identity link")
        try:
            return cls(
                link_id=data["link_id"],
                source_key=SourceIdentityKey.from_dict(data["source_key"]),
                canonical_key=data["canonical_key"],
                evidence_digest=data["evidence_digest"],
                policy_id=data["policy_id"],
                policy_version=data["policy_version"],
                recorded_at=data["recorded_at"],
                active=data["active"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise IdentityContractError(
                "identity link is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class IdentityReviewDecision(IdentityRecord):
    """Explicit review decision required before merging uncertain identities."""

    decision_id: str
    resolution_id: str
    action: str
    selected_canonical_key: str | None
    reviewer_digest: str
    evidence_digest: str
    policy_id: str
    policy_version: str
    decided_at: str
    schema_version: str = IDENTITY_SCHEMA_VERSION

    _fields = frozenset(
        {
            "action",
            "decided_at",
            "decision_id",
            "evidence_digest",
            "policy_id",
            "policy_version",
            "resolution_id",
            "reviewer_digest",
            "schema_version",
            "selected_canonical_key",
        }
    )

    def __post_init__(self) -> None:
        _opaque_id(self.decision_id, "decision_id")
        _opaque_id(self.resolution_id, "resolution_id")
        if self.action not in REVIEW_ACTIONS:
            raise IdentityContractError("review action is unsupported")
        if self.selected_canonical_key is not None:
            _opaque_id(self.selected_canonical_key, "selected_canonical_key")
        if (
            self.action == "confirm_unmatched"
            and self.selected_canonical_key is not None
        ):
            raise IdentityContractError("unmatched review cannot select a key")
        if self.action != "confirm_unmatched" and self.selected_canonical_key is None:
            raise IdentityContractError("review action requires a selected key")
        _digest(self.reviewer_digest, "reviewer_digest")
        _digest(self.evidence_digest, "evidence_digest")
        _controlled(self.policy_id, "policy_id")
        _version(self.policy_version, "policy_version")
        _timestamp(self.decided_at, "decided_at")
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return digest-only review evidence."""

        return {
            "action": self.action,
            "decided_at": self.decided_at,
            "decision_id": self.decision_id,
            "evidence_digest": self.evidence_digest,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "resolution_id": self.resolution_id,
            "reviewer_digest": self.reviewer_digest,
            "schema_version": self.schema_version,
            "selected_canonical_key": self.selected_canonical_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdentityReviewDecision":
        """Parse one strict review decision."""

        data = _strict(payload, cls._fields, "identity review decision")
        try:
            return cls(**data)
        except KeyError:
            raise IdentityContractError(
                "identity review decision is missing a required field"
            ) from None


def load_identity_schema(name: str) -> dict[str, Any]:
    """Load one bundled identity-resolution JSON Schema."""

    normalized = name.removeprefix("identity_").removesuffix(".schema.json")
    if normalized not in IDENTITY_SCHEMA_NAMES:
        raise KeyError("unknown identity schema")
    resource = resources.files(IDENTITY_SCHEMA_PACKAGE).joinpath(
        f"identity_{normalized}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_identity_schemas() -> dict[str, dict[str, Any]]:
    """Load all bundled identity-resolution JSON Schemas."""

    return {name: load_identity_schema(name) for name in IDENTITY_SCHEMA_NAMES}


def _strict(
    payload: Mapping[str, Any], fields: frozenset[str], record_name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise IdentityContractError(f"{record_name} must be an object")
    data = dict(payload)
    if set(data) != fields:
        raise IdentityContractError(f"{record_name} has missing or unknown fields")
    return data


def _strict_json(payload: str, record_name: str) -> Mapping[str, Any]:
    if not isinstance(payload, str):
        raise IdentityContractError(f"{record_name} JSON must be text")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise IdentityContractError(f"{record_name} JSON is invalid") from None
    if not isinstance(value, Mapping):
        raise IdentityContractError(f"{record_name} JSON must contain an object")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite number")


def _schema(value: Any) -> str:
    if value != IDENTITY_SCHEMA_VERSION:
        raise IdentityContractError("identity schema version is unsupported")
    return str(value)


def _opaque_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{field_name} must be an opaque identifier")
    return value


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{field_name} must be a SHA-256 digest")
    return value


def _controlled(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{field_name} must be a controlled identifier")
    return value


def _version(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise IdentityContractError(f"{field_name} must be a version")
    return value


def _timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise IdentityContractError(
            f"{field_name} must be a timezone-aware ISO timestamp"
        )
    return value


def _entity_type(value: Any) -> str:
    if value not in ENTITY_TYPES:
        raise IdentityContractError("entity type is unsupported")
    return str(value)


def _controlled_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise IdentityContractError(f"{field_name} must be a tuple")
    result = tuple(sorted({_controlled(value, field_name) for value in values}))
    if len(result) != len(values):
        raise IdentityContractError(f"{field_name} must be unique")
    return result


def _opaque_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise IdentityContractError(f"{field_name} must be a tuple")
    result = tuple(sorted({_opaque_id(value, field_name) for value in values}))
    if len(result) != len(values):
        raise IdentityContractError(f"{field_name} must be unique")
    return result
