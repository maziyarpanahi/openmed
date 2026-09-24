"""Versioned, value-free contracts for governed clinical registries."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.clinical.review_transitions import ClinicalReviewPacket

REGISTRY_SCHEMA_VERSION: Final = "1.0.0"
REGISTRY_COMPATIBILITY_POLICY: Final = "same_major"
REGISTRY_ADVISORY: Final = (
    "Registry output supports governed review and data operations only; it is not "
    "an autonomous diagnosis, treatment, enrollment, outreach, or ordering action."
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class RegistryContractError(ValueError):
    """Raised when a registry artifact violates its public contract."""


class RegistryConflictError(RegistryContractError):
    """Raised when registry custody or immutable history conflicts."""


class RegistryUnsupportedError(RegistryContractError):
    """Raised when a registry version, state, or rule is unsupported."""


class RegistryFieldState(str, Enum):
    """Evidence state for one materialized registry field."""

    PRESENT = "present"
    MISSING_REQUIRED = "missing_required"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    CORRECTED = "corrected"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"


class RegistryCaseState(str, Enum):
    """Guarded workflow state for one registry case version."""

    REVIEW_REQUIRED = "review_required"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    REVIEW_REJECTED = "review_rejected"
    ADJUDICATION_REQUIRED = "adjudication_required"
    ADJUDICATION_REJECTED = "adjudication_rejected"
    EXPORT_READY = "export_ready"
    EXPORTED = "exported"


_UNRESOLVED_FIELD_STATES = frozenset(
    {
        RegistryFieldState.MISSING_REQUIRED,
        RegistryFieldState.UNKNOWN,
        RegistryFieldState.CONFLICT,
        RegistryFieldState.UNSUPPORTED,
    }
)
_DEFAULT_REVIEW_FIELD_STATES = _UNRESOLVED_FIELD_STATES | {RegistryFieldState.CORRECTED}
_CASE_EVENT_TRANSITIONS = {
    "assign": frozenset(
        {(RegistryCaseState.REVIEW_REQUIRED, RegistryCaseState.ASSIGNED)}
    ),
    "begin_review": frozenset(
        {
            (RegistryCaseState.REVIEW_REQUIRED, RegistryCaseState.IN_REVIEW),
            (RegistryCaseState.ASSIGNED, RegistryCaseState.IN_REVIEW),
        }
    ),
    "complete_review": frozenset(
        {
            (RegistryCaseState.IN_REVIEW, RegistryCaseState.REVIEW_REJECTED),
            (RegistryCaseState.IN_REVIEW, RegistryCaseState.ADJUDICATION_REQUIRED),
            (RegistryCaseState.IN_REVIEW, RegistryCaseState.EXPORT_READY),
        }
    ),
    "adjudicate": frozenset(
        {
            (
                RegistryCaseState.ADJUDICATION_REQUIRED,
                RegistryCaseState.ADJUDICATION_REJECTED,
            ),
            (
                RegistryCaseState.ADJUDICATION_REQUIRED,
                RegistryCaseState.EXPORT_READY,
            ),
        }
    ),
    "export": frozenset({(RegistryCaseState.EXPORT_READY, RegistryCaseState.EXPORTED)}),
}


@dataclass(frozen=True, slots=True)
class RegistryFieldRule:
    """Versioned extraction and validation rule for one registry field."""

    field_id: str
    fact_type: str
    required: bool
    allowed_statuses: tuple[str, ...]
    unknown_statuses: tuple[str, ...] = ("unknown", "uncertain")
    conflict_statuses: tuple[str, ...] = ("conflict", "disputed")
    max_values: int = 1
    extraction_rule: str = "latest_supported_fact"
    validation_rule: str = "status_and_cardinality"

    def __post_init__(self) -> None:
        _controlled(self.field_id, "field_id")
        _controlled(self.fact_type, "fact_type")
        if type(self.required) is not bool:
            raise RegistryContractError("required must be boolean")
        allowed = _controlled_values(
            self.allowed_statuses, "allowed_statuses", minimum=1
        )
        unknown = _controlled_values(self.unknown_statuses, "unknown_statuses")
        conflict = _controlled_values(self.conflict_statuses, "conflict_statuses")
        if set(unknown).intersection(conflict):
            raise RegistryConflictError("unknown and conflict statuses must differ")
        if type(self.max_values) is not int or self.max_values < 1:
            raise RegistryContractError("max_values must be a positive integer")
        _controlled(self.extraction_rule, "extraction_rule")
        _controlled(self.validation_rule, "validation_rule")
        object.__setattr__(self, "allowed_statuses", allowed)
        object.__setattr__(self, "unknown_statuses", unknown)
        object.__setattr__(self, "conflict_statuses", conflict)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical field rule."""

        return {
            "allowed_statuses": list(self.allowed_statuses),
            "conflict_statuses": list(self.conflict_statuses),
            "extraction_rule": self.extraction_rule,
            "fact_type": self.fact_type,
            "field_id": self.field_id,
            "max_values": self.max_values,
            "required": self.required,
            "unknown_statuses": list(self.unknown_statuses),
            "validation_rule": self.validation_rule,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryFieldRule":
        """Parse one strict field rule."""

        data = _mapping(value, "field rule")
        _exact_keys(
            data,
            {
                "allowed_statuses",
                "conflict_statuses",
                "extraction_rule",
                "fact_type",
                "field_id",
                "max_values",
                "required",
                "unknown_statuses",
                "validation_rule",
            },
            "field rule",
        )
        return cls(
            field_id=_text(data["field_id"], "field_id"),
            fact_type=_text(data["fact_type"], "fact_type"),
            required=_boolean(data["required"], "required"),
            allowed_statuses=_text_sequence(
                data["allowed_statuses"], "allowed_statuses"
            ),
            unknown_statuses=_text_sequence(
                data["unknown_statuses"], "unknown_statuses"
            ),
            conflict_statuses=_text_sequence(
                data["conflict_statuses"], "conflict_statuses"
            ),
            max_values=_integer(data["max_values"], "max_values"),
            extraction_rule=_text(data["extraction_rule"], "extraction_rule"),
            validation_rule=_text(data["validation_rule"], "validation_rule"),
        )


@dataclass(frozen=True, slots=True)
class RegistryWorkflowPolicy:
    """Pinned completion, assignment, review, adjudication, and export rules."""

    policy_id: str
    version: str
    owner_scope_id: str
    assignment_required: bool = True
    review_field_states: tuple[RegistryFieldState, ...] = tuple(
        sorted(_DEFAULT_REVIEW_FIELD_STATES, key=lambda item: item.value)
    )
    adjudication_field_states: tuple[RegistryFieldState, ...] = (
        RegistryFieldState.CONFLICT,
        RegistryFieldState.CORRECTED,
    )
    correction_allowed: bool = True
    privacy_policy_digest: str = "sha256:" + "0" * 64
    export_policy_digest: str = "sha256:" + "0" * 64

    def __post_init__(self) -> None:
        _controlled(self.policy_id, "policy_id")
        _semantic_version(self.version, "policy version")
        _opaque_id(self.owner_scope_id, "owner_scope_id")
        if type(self.assignment_required) is not bool:
            raise RegistryContractError("assignment_required must be boolean")
        if type(self.correction_allowed) is not bool:
            raise RegistryContractError("correction_allowed must be boolean")
        review = _field_states(self.review_field_states, "review_field_states")
        adjudication = _field_states(
            self.adjudication_field_states, "adjudication_field_states"
        )
        if not set(adjudication) <= set(review) | {RegistryFieldState.CORRECTED}:
            raise RegistryConflictError("adjudication states must also require review")
        _digest(self.privacy_policy_digest, "privacy_policy_digest")
        _digest(self.export_policy_digest, "export_policy_digest")
        object.__setattr__(self, "review_field_states", review)
        object.__setattr__(self, "adjudication_field_states", adjudication)

    @property
    def digest(self) -> str:
        """Return the canonical workflow-policy digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete workflow policy."""

        return {
            "adjudication_field_states": [
                item.value for item in self.adjudication_field_states
            ],
            "assignment_required": self.assignment_required,
            "correction_allowed": self.correction_allowed,
            "export_policy_digest": self.export_policy_digest,
            "owner_scope_id": self.owner_scope_id,
            "policy_id": self.policy_id,
            "privacy_policy_digest": self.privacy_policy_digest,
            "review_field_states": [item.value for item in self.review_field_states],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryWorkflowPolicy":
        """Parse a strict workflow policy."""

        data = _mapping(value, "workflow policy")
        _exact_keys(
            data,
            {
                "adjudication_field_states",
                "assignment_required",
                "correction_allowed",
                "export_policy_digest",
                "owner_scope_id",
                "policy_id",
                "privacy_policy_digest",
                "review_field_states",
                "version",
            },
            "workflow policy",
        )
        return cls(
            policy_id=_text(data["policy_id"], "policy_id"),
            version=_text(data["version"], "policy version"),
            owner_scope_id=_text(data["owner_scope_id"], "owner_scope_id"),
            assignment_required=_boolean(
                data["assignment_required"], "assignment_required"
            ),
            review_field_states=tuple(
                _enum(item, RegistryFieldState, "review field state")
                for item in _sequence(
                    data["review_field_states"], "review_field_states"
                )
            ),
            adjudication_field_states=tuple(
                _enum(item, RegistryFieldState, "adjudication field state")
                for item in _sequence(
                    data["adjudication_field_states"], "adjudication_field_states"
                )
            ),
            correction_allowed=_boolean(
                data["correction_allowed"], "correction_allowed"
            ),
            privacy_policy_digest=_text(
                data["privacy_policy_digest"], "privacy_policy_digest"
            ),
            export_policy_digest=_text(
                data["export_policy_digest"], "export_policy_digest"
            ),
        )


@dataclass(frozen=True, slots=True)
class RegistryDefinition:
    """One configurable registry definition before content versioning."""

    registry_id: str
    cohort_definition_version_id: str
    cohort_definition_digest: str
    fields: tuple[RegistryFieldRule, ...]
    workflow: RegistryWorkflowPolicy
    definition_version: str
    completion_rule: str = "all_required_resolved"
    schema_version: str = REGISTRY_SCHEMA_VERSION
    compatibility_policy: str = REGISTRY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        _opaque_id(self.registry_id, "registry_id")
        _opaque_id(self.cohort_definition_version_id, "cohort_definition_version_id")
        _digest(self.cohort_definition_digest, "cohort_definition_digest")
        _semantic_version(self.definition_version, "definition_version")
        _controlled(self.completion_rule, "completion_rule")
        if any(not isinstance(item, RegistryFieldRule) for item in self.fields):
            raise TypeError("fields must contain RegistryFieldRule")
        fields = tuple(sorted(self.fields, key=lambda item: item.field_id))
        if not fields or len({item.field_id for item in fields}) != len(fields):
            raise RegistryConflictError("registry field identifiers must be unique")
        fact_types = [item.fact_type for item in fields]
        if len(fact_types) != len(set(fact_types)):
            raise RegistryConflictError("one fact type may populate only one field")
        if not isinstance(self.workflow, RegistryWorkflowPolicy):
            raise TypeError("workflow must be RegistryWorkflowPolicy")
        object.__setattr__(self, "fields", fields)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical registry definition."""

        return {
            "cohort_definition_digest": self.cohort_definition_digest,
            "cohort_definition_version_id": self.cohort_definition_version_id,
            "compatibility_policy": self.compatibility_policy,
            "completion_rule": self.completion_rule,
            "definition_version": self.definition_version,
            "fields": [item.to_dict() for item in self.fields],
            "registry_id": self.registry_id,
            "schema_version": self.schema_version,
            "workflow": self.workflow.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryDefinition":
        """Parse a strict registry definition."""

        data = _mapping(value, "registry definition")
        _exact_keys(
            data,
            {
                "cohort_definition_digest",
                "cohort_definition_version_id",
                "compatibility_policy",
                "completion_rule",
                "definition_version",
                "fields",
                "registry_id",
                "schema_version",
                "workflow",
            },
            "registry definition",
        )
        return cls(
            registry_id=_text(data["registry_id"], "registry_id"),
            cohort_definition_version_id=_text(
                data["cohort_definition_version_id"], "cohort_definition_version_id"
            ),
            cohort_definition_digest=_text(
                data["cohort_definition_digest"], "cohort_definition_digest"
            ),
            fields=tuple(
                RegistryFieldRule.from_dict(_mapping(item, "field rule"))
                for item in _sequence(data["fields"], "fields")
            ),
            workflow=RegistryWorkflowPolicy.from_dict(
                _mapping(data["workflow"], "workflow")
            ),
            definition_version=_text(data["definition_version"], "definition_version"),
            completion_rule=_text(data["completion_rule"], "completion_rule"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class RegistryDefinitionVersion:
    """Content-addressed registry definition version."""

    definition: RegistryDefinition
    version_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.definition, RegistryDefinition):
            raise TypeError("definition must be RegistryDefinition")
        expected = _derived_id("registryversion", self.definition.to_dict())
        if self.version_id is not None and self.version_id != expected:
            raise RegistryConflictError("registry definition version id differs")
        object.__setattr__(self, "version_id", expected)

    @property
    def definition_digest(self) -> str:
        """Return the canonical definition digest."""

        return canonical_digest(self.definition.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted definition version."""

        return {
            "artifact_type": "registry_definition_version",
            "definition": self.definition.to_dict(),
            "definition_digest": self.definition_digest,
            "version_id": self.version_id,
        }

    def to_json(self) -> str:
        """Return canonical definition-version JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryDefinitionVersion":
        """Parse and verify a definition version."""

        data = _mapping(value, "registry definition version")
        _exact_keys(
            data,
            {"artifact_type", "definition", "definition_digest", "version_id"},
            "registry definition version",
        )
        if data["artifact_type"] != "registry_definition_version":
            raise RegistryUnsupportedError("registry artifact type is unsupported")
        result = cls(
            definition=RegistryDefinition.from_dict(
                _mapping(data["definition"], "definition")
            ),
            version_id=_text(data["version_id"], "version_id"),
        )
        if data["definition_digest"] != result.definition_digest:
            raise RegistryConflictError("registry definition digest differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "RegistryDefinitionVersion":
        """Parse canonical or human-formatted definition JSON."""

        return cls.from_dict(_json_object(value, "registry definition version"))


@dataclass(frozen=True, slots=True)
class RegistryFieldEvidence:
    """Value-free fact, evidence, and digest custody for one field."""

    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    value_digests: tuple[str, ...]
    derivation_digests: tuple[str, ...]
    corrected_from_fact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("fact_ids", "evidence_ids", "corrected_from_fact_ids"):
            object.__setattr__(self, name, _opaque_values(getattr(self, name), name))
        for name in ("value_digests", "derivation_digests"):
            object.__setattr__(
                self,
                name,
                tuple(_digest(item, name) for item in getattr(self, name)),
            )
        if len(self.value_digests) != len(self.fact_ids):
            raise RegistryConflictError("value digests must cover every fact")
        if len(self.derivation_digests) != len(self.fact_ids):
            raise RegistryConflictError("derivation digests must cover every fact")

    def to_dict(self) -> dict[str, Any]:
        """Return evidence custody without fact values."""

        return {
            "corrected_from_fact_ids": list(self.corrected_from_fact_ids),
            "derivation_digests": list(self.derivation_digests),
            "evidence_ids": list(self.evidence_ids),
            "fact_ids": list(self.fact_ids),
            "value_digests": list(self.value_digests),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryFieldEvidence":
        """Parse strict field evidence."""

        data = _mapping(value, "field evidence")
        _exact_keys(
            data,
            {
                "corrected_from_fact_ids",
                "derivation_digests",
                "evidence_ids",
                "fact_ids",
                "value_digests",
            },
            "field evidence",
        )
        return cls(
            fact_ids=_text_sequence(data["fact_ids"], "fact_ids"),
            evidence_ids=_text_sequence(data["evidence_ids"], "evidence_ids"),
            value_digests=_text_sequence(data["value_digests"], "value_digests"),
            derivation_digests=_text_sequence(
                data["derivation_digests"], "derivation_digests"
            ),
            corrected_from_fact_ids=_text_sequence(
                data["corrected_from_fact_ids"], "corrected_from_fact_ids"
            ),
        )


@dataclass(frozen=True, slots=True)
class RegistryFieldResult:
    """One typed registry field state and its value-free evidence."""

    field_id: str
    state: RegistryFieldState
    evidence: RegistryFieldEvidence
    reason_code: str

    def __post_init__(self) -> None:
        _controlled(self.field_id, "field_id")
        object.__setattr__(
            self, "state", _enum(self.state, RegistryFieldState, "field state")
        )
        if not isinstance(self.evidence, RegistryFieldEvidence):
            raise TypeError("evidence must be RegistryFieldEvidence")
        _controlled(self.reason_code, "reason_code")
        if self.state in {RegistryFieldState.PRESENT, RegistryFieldState.CORRECTED}:
            if not self.evidence.fact_ids:
                raise RegistryConflictError("resolved fields require fact evidence")
        if self.state is RegistryFieldState.CORRECTED:
            if not self.evidence.corrected_from_fact_ids:
                raise RegistryConflictError("corrected fields require prior facts")

    @property
    def digest(self) -> str:
        """Return the field-result digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return one value-free field result."""

        return {
            "evidence": self.evidence.to_dict(),
            "field_id": self.field_id,
            "reason_code": self.reason_code,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryFieldResult":
        """Parse one strict field result."""

        data = _mapping(value, "field result")
        _exact_keys(
            data, {"evidence", "field_id", "reason_code", "state"}, "field result"
        )
        return cls(
            field_id=_text(data["field_id"], "field_id"),
            state=_enum(data["state"], RegistryFieldState, "field state"),
            evidence=RegistryFieldEvidence.from_dict(
                _mapping(data["evidence"], "field evidence")
            ),
            reason_code=_text(data["reason_code"], "reason_code"),
        )


@dataclass(frozen=True, slots=True)
class RegistryAssignment:
    """Owner-authorized queue assignment without reviewer identity."""

    assignment_id: str
    queue_id: str
    authorization_id: str
    authorization_digest: str
    assigned_at: str

    def __post_init__(self) -> None:
        _opaque_id(self.assignment_id, "assignment_id")
        _controlled(self.queue_id, "queue_id")
        _opaque_id(self.authorization_id, "authorization_id")
        _digest(self.authorization_digest, "authorization_digest")
        _timestamp(self.assigned_at, "assigned_at")

    def to_dict(self) -> dict[str, str]:
        """Return a value-free assignment record."""

        return {
            "assigned_at": self.assigned_at,
            "assignment_id": self.assignment_id,
            "authorization_digest": self.authorization_digest,
            "authorization_id": self.authorization_id,
            "queue_id": self.queue_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryAssignment":
        """Parse one strict assignment record."""

        data = _mapping(value, "registry assignment")
        _exact_keys(
            data,
            {
                "assigned_at",
                "assignment_id",
                "authorization_digest",
                "authorization_id",
                "queue_id",
            },
            "registry assignment",
        )
        return cls(**{key: _text(data[key], key) for key in data})


@dataclass(frozen=True, slots=True)
class RegistryCaseEvent:
    """One immutable case transition, review, adjudication, or correction event."""

    event_id: str
    case_id: str
    action: str
    from_state: RegistryCaseState
    to_state: RegistryCaseState
    occurred_at: str
    reason_code: str
    policy_digest: str
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        _opaque_id(self.event_id, "event_id")
        _opaque_id(self.case_id, "case_id")
        _controlled(self.action, "action")
        object.__setattr__(
            self, "from_state", _enum(self.from_state, RegistryCaseState, "from_state")
        )
        object.__setattr__(
            self, "to_state", _enum(self.to_state, RegistryCaseState, "to_state")
        )
        _timestamp(self.occurred_at, "occurred_at")
        _controlled(self.reason_code, "reason_code")
        _digest(self.policy_digest, "policy_digest")
        if self.artifact_digest is not None:
            _digest(self.artifact_digest, "artifact_digest")

    def to_dict(self) -> dict[str, Any]:
        """Return value-free transition metadata."""

        return {
            "action": self.action,
            "artifact_digest": self.artifact_digest,
            "case_id": self.case_id,
            "event_id": self.event_id,
            "from_state": self.from_state.value,
            "occurred_at": self.occurred_at,
            "policy_digest": self.policy_digest,
            "reason_code": self.reason_code,
            "to_state": self.to_state.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryCaseEvent":
        """Parse a strict case event."""

        data = _mapping(value, "registry case event")
        _exact_keys(
            data,
            {
                "action",
                "artifact_digest",
                "case_id",
                "event_id",
                "from_state",
                "occurred_at",
                "policy_digest",
                "reason_code",
                "to_state",
            },
            "registry case event",
        )
        return cls(
            event_id=_text(data["event_id"], "event_id"),
            case_id=_text(data["case_id"], "case_id"),
            action=_text(data["action"], "action"),
            from_state=_enum(data["from_state"], RegistryCaseState, "from_state"),
            to_state=_enum(data["to_state"], RegistryCaseState, "to_state"),
            occurred_at=_text(data["occurred_at"], "occurred_at"),
            reason_code=_text(data["reason_code"], "reason_code"),
            policy_digest=_text(data["policy_digest"], "policy_digest"),
            artifact_digest=_optional_text(data["artifact_digest"], "artifact_digest"),
        )


@dataclass(frozen=True, slots=True)
class RegistryCase:
    """Immutable registry case version with evidence and guarded history."""

    case_id: str
    definition_version_id: str
    definition_digest: str
    subject_id: str
    cohort_execution_id: str
    cohort_execution_digest: str
    source_snapshot_id: str
    source_snapshot_digest: str
    created_at: str
    fields: tuple[RegistryFieldResult, ...]
    origin_state: RegistryCaseState
    state: RegistryCaseState
    workflow_policy_digest: str
    assignment: RegistryAssignment | None = None
    review_packets: tuple[ClinicalReviewPacket, ...] = ()
    events: tuple[RegistryCaseEvent, ...] = ()
    schema_version: str = REGISTRY_SCHEMA_VERSION
    compatibility_policy: str = REGISTRY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        for name in (
            "case_id",
            "definition_version_id",
            "subject_id",
            "cohort_execution_id",
            "source_snapshot_id",
        ):
            _opaque_id(getattr(self, name), name)
        for name in (
            "definition_digest",
            "cohort_execution_digest",
            "source_snapshot_digest",
            "workflow_policy_digest",
        ):
            _digest(getattr(self, name), name)
        _timestamp(self.created_at, "created_at")
        if any(not isinstance(item, RegistryFieldResult) for item in self.fields):
            raise TypeError("fields must contain RegistryFieldResult")
        fields = tuple(sorted(self.fields, key=lambda item: item.field_id))
        if not fields or len({item.field_id for item in fields}) != len(fields):
            raise RegistryConflictError("case fields must be non-empty and unique")
        object.__setattr__(
            self,
            "origin_state",
            _enum(self.origin_state, RegistryCaseState, "origin state"),
        )
        if self.origin_state not in {
            RegistryCaseState.REVIEW_REQUIRED,
            RegistryCaseState.EXPORT_READY,
        }:
            raise RegistryConflictError("case origin state is unsupported")
        object.__setattr__(
            self, "state", _enum(self.state, RegistryCaseState, "case state")
        )
        if self.assignment is not None and not isinstance(
            self.assignment, RegistryAssignment
        ):
            raise TypeError("assignment must be RegistryAssignment")
        if any(
            not isinstance(item, ClinicalReviewPacket) for item in self.review_packets
        ):
            raise TypeError("review_packets must contain ClinicalReviewPacket")
        packets = tuple(sorted(self.review_packets, key=lambda item: item.packet_id))
        fact_ids = {
            fact_id
            for field_result in fields
            for fact_id in field_result.evidence.fact_ids
        }
        if any(not set(packet.fact_ids) <= fact_ids for packet in packets):
            raise RegistryConflictError("review packet references facts outside case")
        events = tuple(self.events)
        previous_state = self.initial_state
        previous_time = self.created_at
        seen: set[str] = set()
        for event in events:
            if not isinstance(event, RegistryCaseEvent):
                raise TypeError("events must contain RegistryCaseEvent")
            if event.case_id != self.case_id or event.event_id in seen:
                raise RegistryConflictError("case event identity conflicts")
            if event.from_state is not previous_state:
                raise RegistryConflictError("case event history is discontinuous")
            if event.action == "correct_field":
                if event.to_state not in {
                    RegistryCaseState.REVIEW_REQUIRED,
                    RegistryCaseState.EXPORT_READY,
                }:
                    raise RegistryConflictError("correction transition is invalid")
            elif (event.from_state, event.to_state) not in _CASE_EVENT_TRANSITIONS.get(
                event.action, frozenset()
            ):
                raise RegistryConflictError("case transition is invalid")
            if _timestamp_key(event.occurred_at) < _timestamp_key(previous_time):
                raise RegistryConflictError("case events must be chronological")
            seen.add(event.event_id)
            previous_state = event.to_state
            previous_time = event.occurred_at
        if self.state is not previous_state:
            raise RegistryConflictError("case state differs from event history")
        if self.state in {RegistryCaseState.ASSIGNED, RegistryCaseState.IN_REVIEW}:
            if self.assignment is None:
                raise RegistryConflictError("assigned review state requires assignment")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "review_packets", packets)
        object.__setattr__(self, "events", events)

    @property
    def initial_state(self) -> RegistryCaseState:
        """Return the deterministic workflow state before appended events."""

        return self.origin_state

    @property
    def completion_state(self) -> str:
        """Return complete or incomplete without hiding field-level reasons."""

        return (
            "incomplete"
            if any(item.state in _UNRESOLVED_FIELD_STATES for item in self.fields)
            else "complete"
        )

    @property
    def case_digest(self) -> str:
        """Return the digest of the complete case version."""

        return canonical_digest(self.identity_payload)

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return all immutable case fields covered by ``case_digest``."""

        return {
            "assignment": None
            if self.assignment is None
            else self.assignment.to_dict(),
            "cohort_execution_digest": self.cohort_execution_digest,
            "cohort_execution_id": self.cohort_execution_id,
            "compatibility_policy": self.compatibility_policy,
            "created_at": self.created_at,
            "definition_digest": self.definition_digest,
            "definition_version_id": self.definition_version_id,
            "events": [item.to_dict() for item in self.events],
            "fields": [item.to_dict() for item in self.fields],
            "origin_state": self.origin_state.value,
            "review_packets": [item.to_dict() for item in self.review_packets],
            "schema_version": self.schema_version,
            "source_snapshot_digest": self.source_snapshot_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "state": self.state.value,
            "subject_id": self.subject_id,
            "workflow_policy_digest": self.workflow_policy_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted value-free case."""

        return {
            "advisory": REGISTRY_ADVISORY,
            "artifact_type": "registry_case",
            "case_digest": self.case_digest,
            "case_id": self.case_id,
            "completion_state": self.completion_state,
            **self.identity_payload,
        }

    def to_json(self) -> str:
        """Return canonical registry-case JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryCase":
        """Parse and verify a persisted registry case."""

        data = _mapping(value, "registry case")
        expected = {
            "advisory",
            "artifact_type",
            "assignment",
            "case_digest",
            "case_id",
            "cohort_execution_digest",
            "cohort_execution_id",
            "compatibility_policy",
            "completion_state",
            "created_at",
            "definition_digest",
            "definition_version_id",
            "events",
            "fields",
            "origin_state",
            "review_packets",
            "schema_version",
            "source_snapshot_digest",
            "source_snapshot_id",
            "state",
            "subject_id",
            "workflow_policy_digest",
        }
        _exact_keys(data, expected, "registry case")
        if data["artifact_type"] != "registry_case":
            raise RegistryUnsupportedError("registry artifact type is unsupported")
        if data["advisory"] != REGISTRY_ADVISORY:
            raise RegistryConflictError("registry advisory differs")
        assignment = data["assignment"]
        result = cls(
            case_id=_text(data["case_id"], "case_id"),
            definition_version_id=_text(
                data["definition_version_id"], "definition_version_id"
            ),
            definition_digest=_text(data["definition_digest"], "definition_digest"),
            subject_id=_text(data["subject_id"], "subject_id"),
            cohort_execution_id=_text(
                data["cohort_execution_id"], "cohort_execution_id"
            ),
            cohort_execution_digest=_text(
                data["cohort_execution_digest"], "cohort_execution_digest"
            ),
            source_snapshot_id=_text(data["source_snapshot_id"], "source_snapshot_id"),
            source_snapshot_digest=_text(
                data["source_snapshot_digest"], "source_snapshot_digest"
            ),
            created_at=_text(data["created_at"], "created_at"),
            fields=tuple(
                RegistryFieldResult.from_dict(_mapping(item, "field result"))
                for item in _sequence(data["fields"], "fields")
            ),
            origin_state=_enum(data["origin_state"], RegistryCaseState, "origin state"),
            state=_enum(data["state"], RegistryCaseState, "case state"),
            workflow_policy_digest=_text(
                data["workflow_policy_digest"], "workflow_policy_digest"
            ),
            assignment=(
                None
                if assignment is None
                else RegistryAssignment.from_dict(_mapping(assignment, "assignment"))
            ),
            review_packets=tuple(
                ClinicalReviewPacket.from_dict(_mapping(item, "review packet"))
                for item in _sequence(data["review_packets"], "review_packets")
            ),
            events=tuple(
                RegistryCaseEvent.from_dict(_mapping(item, "case event"))
                for item in _sequence(data["events"], "events")
            ),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["case_digest"] != result.case_digest:
            raise RegistryConflictError("registry case digest differs")
        if data["completion_state"] != result.completion_state:
            raise RegistryConflictError("registry completion state differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "RegistryCase":
        """Parse canonical or human-formatted case JSON."""

        return cls.from_dict(_json_object(value, "registry case"))


@dataclass(frozen=True, slots=True)
class RegistryAssignmentAuthorization:
    """Owner-scope approval for one queue assignment."""

    authorization_id: str
    owner_scope_id: str
    definition_version_id: str
    workflow_policy_digest: str
    queue_id: str
    assignment_approved: bool

    def __post_init__(self) -> None:
        for name in ("authorization_id", "owner_scope_id", "definition_version_id"):
            _opaque_id(getattr(self, name), name)
        _digest(self.workflow_policy_digest, "workflow_policy_digest")
        _controlled(self.queue_id, "queue_id")
        if type(self.assignment_approved) is not bool:
            raise RegistryContractError("assignment_approved must be boolean")

    @property
    def digest(self) -> str:
        """Return the approval digest recorded on assignment."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return authorization custody without owner identity."""

        return {
            "assignment_approved": self.assignment_approved,
            "authorization_id": self.authorization_id,
            "definition_version_id": self.definition_version_id,
            "owner_scope_id": self.owner_scope_id,
            "queue_id": self.queue_id,
            "workflow_policy_digest": self.workflow_policy_digest,
        }


@dataclass(frozen=True, slots=True)
class RegistryExportAuthorization:
    """Approval bound to one definition and its exact export/privacy policies."""

    authorization_id: str
    definition_version_id: str
    privacy_policy_digest: str
    export_policy_digest: str
    export_approved: bool

    def __post_init__(self) -> None:
        _opaque_id(self.authorization_id, "authorization_id")
        _opaque_id(self.definition_version_id, "definition_version_id")
        _digest(self.privacy_policy_digest, "privacy_policy_digest")
        _digest(self.export_policy_digest, "export_policy_digest")
        if type(self.export_approved) is not bool:
            raise RegistryContractError("export_approved must be boolean")

    @property
    def digest(self) -> str:
        """Return the export authorization digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return value-free export approval custody."""

        return {
            "authorization_id": self.authorization_id,
            "definition_version_id": self.definition_version_id,
            "export_approved": self.export_approved,
            "export_policy_digest": self.export_policy_digest,
            "privacy_policy_digest": self.privacy_policy_digest,
        }


@dataclass(frozen=True, slots=True)
class RegistryExportEnvelope:
    """Privacy-safe export manifest for reviewed registry case versions."""

    export_id: str
    definition_version_id: str
    definition_digest: str
    registry_id: str
    created_at: str
    case_digests: Mapping[str, str]
    source_snapshot_ids: tuple[str, ...]
    authorization_id: str
    authorization_digest: str
    privacy_policy_digest: str
    export_policy_digest: str
    schema_version: str = REGISTRY_SCHEMA_VERSION
    compatibility_policy: str = REGISTRY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        for name in (
            "export_id",
            "definition_version_id",
            "registry_id",
            "authorization_id",
        ):
            _opaque_id(getattr(self, name), name)
        for name in (
            "definition_digest",
            "authorization_digest",
            "privacy_policy_digest",
            "export_policy_digest",
        ):
            _digest(getattr(self, name), name)
        _timestamp(self.created_at, "created_at")
        cases = {
            _opaque_id(key, "case_id"): _digest(value, "case_digest")
            for key, value in sorted(
                _mapping(self.case_digests, "case_digests").items()
            )
        }
        if not cases:
            raise RegistryContractError("registry export requires at least one case")
        snapshots = _opaque_values(self.source_snapshot_ids, "source_snapshot_ids")
        if not snapshots:
            raise RegistryContractError("registry export requires source snapshots")
        object.__setattr__(self, "case_digests", MappingProxyType(cases))
        object.__setattr__(self, "source_snapshot_ids", snapshots)

    @property
    def manifest_digest(self) -> str:
        """Return the canonical export-manifest digest."""

        return canonical_digest(self.identity_payload)

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return all export fields covered by ``manifest_digest``."""

        return {
            "authorization_digest": self.authorization_digest,
            "authorization_id": self.authorization_id,
            "case_digests": dict(self.case_digests),
            "compatibility_policy": self.compatibility_policy,
            "created_at": self.created_at,
            "definition_digest": self.definition_digest,
            "definition_version_id": self.definition_version_id,
            "export_id": self.export_id,
            "export_policy_digest": self.export_policy_digest,
            "privacy_policy_digest": self.privacy_policy_digest,
            "registry_id": self.registry_id,
            "schema_version": self.schema_version,
            "source_snapshot_ids": list(self.source_snapshot_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free registry export manifest."""

        return {
            "advisory": REGISTRY_ADVISORY,
            "artifact_type": "registry_export",
            "manifest_digest": self.manifest_digest,
            **self.identity_payload,
        }

    def to_json(self) -> str:
        """Return canonical export JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryExportEnvelope":
        """Parse and verify a persisted registry export envelope."""

        data = _mapping(value, "registry export")
        expected = {
            "advisory",
            "artifact_type",
            "authorization_digest",
            "authorization_id",
            "case_digests",
            "compatibility_policy",
            "created_at",
            "definition_digest",
            "definition_version_id",
            "export_id",
            "export_policy_digest",
            "manifest_digest",
            "privacy_policy_digest",
            "registry_id",
            "schema_version",
            "source_snapshot_ids",
        }
        _exact_keys(data, expected, "registry export")
        if data["artifact_type"] != "registry_export":
            raise RegistryUnsupportedError("registry artifact type is unsupported")
        if data["advisory"] != REGISTRY_ADVISORY:
            raise RegistryConflictError("registry advisory differs")
        result = cls(
            export_id=_text(data["export_id"], "export_id"),
            definition_version_id=_text(
                data["definition_version_id"], "definition_version_id"
            ),
            definition_digest=_text(data["definition_digest"], "definition_digest"),
            registry_id=_text(data["registry_id"], "registry_id"),
            created_at=_text(data["created_at"], "created_at"),
            case_digests={
                _text(key, "case_id"): _text(item, "case_digest")
                for key, item in _mapping(data["case_digests"], "case_digests").items()
            },
            source_snapshot_ids=_text_sequence(
                data["source_snapshot_ids"], "source_snapshot_ids"
            ),
            authorization_id=_text(data["authorization_id"], "authorization_id"),
            authorization_digest=_text(
                data["authorization_digest"], "authorization_digest"
            ),
            privacy_policy_digest=_text(
                data["privacy_policy_digest"], "privacy_policy_digest"
            ),
            export_policy_digest=_text(
                data["export_policy_digest"], "export_policy_digest"
            ),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["manifest_digest"] != result.manifest_digest:
            raise RegistryConflictError("registry export manifest digest differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "RegistryExportEnvelope":
        """Parse canonical or human-formatted registry export JSON."""

        return cls.from_dict(_json_object(value, "registry export"))


def _contract_version(schema_version: str, compatibility_policy: str) -> None:
    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise RegistryUnsupportedError("registry schema version is unsupported")
    if compatibility_policy != REGISTRY_COMPATIBILITY_POLICY:
        raise RegistryUnsupportedError("registry compatibility policy is unsupported")


def _field_states(
    value: Sequence[RegistryFieldState], name: str
) -> tuple[RegistryFieldState, ...]:
    states = tuple(
        sorted(
            {_enum(item, RegistryFieldState, name) for item in value},
            key=lambda item: item.value,
        )
    )
    return states


def _derived_id(prefix: str, *materials: Any) -> str:
    digest = canonical_digest(list(materials)).removeprefix("sha256:")
    return f"{prefix}_{digest[:32]}"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryContractError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RegistryContractError(f"{name} must be an array")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise RegistryContractError(f"{name} fields are incompatible")


def _json_object(value: str | bytes | bytearray, name: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        raise RegistryContractError(f"{name} is not valid JSON") from None
    return _mapping(parsed, name)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryContractError(f"{name} must be non-empty text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise RegistryContractError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RegistryContractError(f"{name} must be boolean")
    return value


def _controlled(value: Any, name: str) -> str:
    text = _text(value, name)
    if _CONTROLLED_RE.fullmatch(text) is None:
        raise RegistryContractError(f"{name} must be controlled")
    return text


def _opaque_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if _OPAQUE_ID_RE.fullmatch(text) is None:
        raise RegistryContractError(f"{name} must be opaque")
    return text


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if _DIGEST_RE.fullmatch(text) is None:
        raise RegistryContractError(f"{name} must be a digest")
    return text


def _semantic_version(value: Any, name: str) -> str:
    text = _text(value, name)
    if _VERSION_RE.fullmatch(text) is None:
        raise RegistryContractError(f"{name} must be semantic")
    return text


def _timestamp(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise RegistryContractError(f"{name} must be an RFC 3339 timestamp")
    return text


def _timestamp_key(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _controlled_values(
    value: Sequence[str], name: str, *, minimum: int = 0
) -> tuple[str, ...]:
    values = tuple(sorted({_controlled(item, name) for item in value}))
    if len(values) < minimum:
        raise RegistryContractError(f"{name} has too few values")
    return values


def _opaque_values(value: Sequence[str], name: str) -> tuple[str, ...]:
    return tuple(sorted({_opaque_id(item, name) for item in value}))


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError):
        raise RegistryUnsupportedError(f"{name} is unsupported") from None


__all__ = [
    "REGISTRY_ADVISORY",
    "REGISTRY_COMPATIBILITY_POLICY",
    "REGISTRY_SCHEMA_VERSION",
    "RegistryAssignment",
    "RegistryAssignmentAuthorization",
    "RegistryCase",
    "RegistryCaseEvent",
    "RegistryCaseState",
    "RegistryConflictError",
    "RegistryContractError",
    "RegistryDefinition",
    "RegistryDefinitionVersion",
    "RegistryExportAuthorization",
    "RegistryExportEnvelope",
    "RegistryFieldEvidence",
    "RegistryFieldResult",
    "RegistryFieldRule",
    "RegistryFieldState",
    "RegistryUnsupportedError",
    "RegistryWorkflowPolicy",
]
