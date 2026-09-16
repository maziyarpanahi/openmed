"""Allowlisted, metadata-only attributes for agent events.

Event producers share one closed attribute contract. Accepting arbitrary
mappings is what lets prompts, tool arguments, clinical text, credentials, or
filesystem paths ride along on an otherwise privacy-safe event, so this module
accepts a fixed set of names with a fixed kind each. Unknown and
sensitive-looking keys are refused without the submitted key or value ever
being retained or echoed.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Mapping

from .correlation import ActionId, CorrelationIdError, RunId
from .identifiers import (
    CapabilityId,
    GovernanceIdError,
    PolicyId,
    PurposeId,
    ToolId,
    WorkflowId,
)
from .outcomes import OutcomeClass, allowed_reason_codes

EVENT_ATTRIBUTES_SCHEMA_VERSION: Final[str] = "openmed.agent.event_attributes.v1"
MAX_ATTRIBUTE_COUNT: Final[int] = 32
MAX_ATTRIBUTE_JSON_BYTES: Final[int] = 4_096
MAX_COUNT_VALUE: Final[int] = 1_000_000_000
MAX_DURATION_MS: Final[float] = 31_536_000_000.0

EXECUTION_STAGES: Final[tuple[str, ...]] = (
    "planned",
    "authorized",
    "started",
    "tool_call",
    "completed",
    "failed",
    "aborted",
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SENSITIVE_KEY_RE = re.compile(
    r"address|arg|auth|bearer|content|cookie|cred|dob|email|file|input|key|"
    r"message|mrn|name|note|passw|patient|path|payload|phone|prompt|secret|"
    r"session|text|token|uri|url",
    re.IGNORECASE,
)


class AttributeKind(str, Enum):
    """Closed set of value shapes an allowlisted attribute may take.

    Values:
        RUN_ID: An opaque ``run_`` correlation identifier.
        ACTION_ID: An opaque ``act_`` correlation identifier.
        CAPABILITY_ID: A canonical ``capability:`` governance identifier.
        POLICY_ID: A canonical ``policy:`` governance identifier.
        PURPOSE_ID: A canonical ``purpose:`` governance identifier.
        TOOL_ID: A canonical ``tool:`` governance identifier.
        WORKFLOW_ID: A canonical ``workflow:`` governance identifier.
        EXECUTION_STAGE: A member of :data:`EXECUTION_STAGES`.
        OUTCOME_CLASS: A member of :class:`~openmed.agent.outcomes.OutcomeClass`.
        OUTCOME_REASON: A reason code allowed for the declared outcome class.
        DIGEST: A lowercase ``sha256:`` digest.
        COUNT: A bounded non-negative integer.
        DURATION_MS: A bounded, finite, non-negative number of milliseconds.
        FLAG: A real boolean.
    """

    RUN_ID = "run_id"
    ACTION_ID = "action_id"
    CAPABILITY_ID = "capability_id"
    POLICY_ID = "policy_id"
    PURPOSE_ID = "purpose_id"
    TOOL_ID = "tool_id"
    WORKFLOW_ID = "workflow_id"
    EXECUTION_STAGE = "execution_stage"
    OUTCOME_CLASS = "outcome_class"
    OUTCOME_REASON = "outcome_reason"
    DIGEST = "digest"
    COUNT = "count"
    DURATION_MS = "duration_ms"
    FLAG = "flag"


ALLOWED_ATTRIBUTES: Final[Mapping[str, AttributeKind]] = MappingProxyType(
    {
        "action_id": AttributeKind.ACTION_ID,
        "artifact_count": AttributeKind.COUNT,
        "artifact_digest": AttributeKind.DIGEST,
        "attempt_number": AttributeKind.COUNT,
        "capability_id": AttributeKind.CAPABILITY_ID,
        "duration_ms": AttributeKind.DURATION_MS,
        "execution_stage": AttributeKind.EXECUTION_STAGE,
        "input_digest": AttributeKind.DIGEST,
        "outcome_class": AttributeKind.OUTCOME_CLASS,
        "outcome_reason": AttributeKind.OUTCOME_REASON,
        "output_digest": AttributeKind.DIGEST,
        "parent_action_id": AttributeKind.ACTION_ID,
        "policy_id": AttributeKind.POLICY_ID,
        "purpose_id": AttributeKind.PURPOSE_ID,
        "redacted": AttributeKind.FLAG,
        "retry_count": AttributeKind.COUNT,
        "retryable": AttributeKind.FLAG,
        "run_id": AttributeKind.RUN_ID,
        "sequence_number": AttributeKind.COUNT,
        "tool_call_count": AttributeKind.COUNT,
        "tool_id": AttributeKind.TOOL_ID,
        "workflow_id": AttributeKind.WORKFLOW_ID,
    }
)

_IDENTIFIER_KINDS: Final[dict[AttributeKind, Any]] = {
    AttributeKind.RUN_ID: RunId,
    AttributeKind.ACTION_ID: ActionId,
    AttributeKind.CAPABILITY_ID: CapabilityId,
    AttributeKind.POLICY_ID: PolicyId,
    AttributeKind.PURPOSE_ID: PurposeId,
    AttributeKind.TOOL_ID: ToolId,
    AttributeKind.WORKFLOW_ID: WorkflowId,
}


class EventAttributeError(ValueError):
    """Raised when event attributes fail closed validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Allowlisted field the failure belongs to, or ``None`` when
            naming it would echo a submitted key.

    Messages and attributes carry controlled diagnostic metadata only. A
    rejected key or value is never retained or echoed.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class EventAttributes:
    """Validated, immutable, metadata-only attributes for one agent event."""

    values: Mapping[str, Any]
    schema_version: str = EVENT_ATTRIBUTES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != EVENT_ATTRIBUTES_SCHEMA_VERSION
        ):
            raise EventAttributeError("invalid_schema_version", "schema_version")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EventAttributes":
        """Validate a mapping against the allowlist and build attributes."""

        return cls(values=validate_event_attributes(data))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "EventAttributes":
        """Validate a JSON object, rejecting duplicate keys before parsing."""

        if isinstance(payload, (bytes, bytearray)):
            size = len(payload)
        elif type(payload) is str:
            size = len(payload.encode("utf-8"))
        else:
            raise EventAttributeError("malformed_json")
        if size > MAX_ATTRIBUTE_JSON_BYTES:
            raise EventAttributeError("payload_too_large")
        try:
            data = json.loads(payload, object_pairs_hook=_strict_json_object)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            pass
        except EventAttributeError:
            raise
        else:
            if not isinstance(data, dict):
                raise EventAttributeError("not_a_mapping")
            return cls.from_mapping(data)
        # Raised outside the handler so a parser message is not retained.
        raise EventAttributeError("malformed_json")

    def get(self, name: str) -> Any:
        """Return one attribute value, or ``None`` when it was not supplied."""

        return self.values.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary with attributes in key order."""

        ordered: dict[str, Any] = {"schema_version": self.schema_version}
        for key in sorted(self.values):
            ordered[key] = self.values[key]
        return ordered

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def validate_event_attributes(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a mapping against the closed attribute allowlist.

    Args:
        data: Candidate attributes keyed by allowlisted names. A
            ``schema_version`` key is accepted so a serialized payload round
            trips; it must equal
            :data:`EVENT_ATTRIBUTES_SCHEMA_VERSION` and is not stored as an
            attribute.

    Returns:
        A new dictionary of normalized, JSON-safe values.

    Raises:
        EventAttributeError: If a key is unknown or sensitive-looking, a value
            has the wrong shape, a number is non-finite or out of range, or the
            mapping is oversized. Unknown and sensitive keys are reported
            without a field name so the submitted key is not echoed.
    """

    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise EventAttributeError("not_a_mapping")
    try:
        keys = list(data)
    except Exception:
        pass
    else:
        if len(keys) > MAX_ATTRIBUTE_COUNT:
            raise EventAttributeError("too_many_attributes")
        return _validate_keys(data, keys)
    raise EventAttributeError("not_a_mapping")


def _validate_keys(data: Mapping[str, Any], keys: list[Any]) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for key in keys:
        if type(key) is not str:
            raise EventAttributeError("invalid_key_type")
        if key == "schema_version":
            if data[key] != EVENT_ATTRIBUTES_SCHEMA_VERSION:
                raise EventAttributeError("invalid_schema_version", "schema_version")
            continue
        kind = ALLOWED_ATTRIBUTES.get(key)
        if kind is None:
            if _SENSITIVE_KEY_RE.search(key):
                raise EventAttributeError("sensitive_attribute_key")
            raise EventAttributeError("unknown_attribute")
        validated[key] = _validate_value(kind, data[key], key, data)
    return validated


def _validate_value(
    kind: AttributeKind, value: Any, field_name: str, data: Mapping[str, Any]
) -> Any:
    if isinstance(value, (Mapping, list, tuple, set, frozenset, bytes, bytearray)):
        raise EventAttributeError("nested_value_not_allowed", field_name)
    if kind in _IDENTIFIER_KINDS:
        return _validate_identifier(_IDENTIFIER_KINDS[kind], value, field_name)
    if kind is AttributeKind.EXECUTION_STAGE:
        if type(value) is not str or value not in EXECUTION_STAGES:
            raise EventAttributeError("unknown_execution_stage", field_name)
        return value
    if kind is AttributeKind.OUTCOME_CLASS:
        return _validate_outcome_class(value, field_name)
    if kind is AttributeKind.OUTCOME_REASON:
        return _validate_outcome_reason(value, field_name, data)
    if kind is AttributeKind.DIGEST:
        if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
            raise EventAttributeError("invalid_digest", field_name)
        return value
    if kind is AttributeKind.COUNT:
        if type(value) is not int:
            raise EventAttributeError("invalid_count", field_name)
        if value < 0 or value > MAX_COUNT_VALUE:
            raise EventAttributeError("count_out_of_range", field_name)
        return value
    if kind is AttributeKind.DURATION_MS:
        return _validate_duration(value, field_name)
    if type(value) is not bool:
        raise EventAttributeError("invalid_flag", field_name)
    return value


def _validate_identifier(factory: Any, value: Any, field_name: str) -> str:
    try:
        factory(value)
    except (CorrelationIdError, GovernanceIdError):
        pass
    else:
        return value
    # Raised outside the handler so the rejected identifier cannot be chained.
    raise EventAttributeError("invalid_identifier", field_name)


def _validate_outcome_class(value: Any, field_name: str) -> str:
    if isinstance(value, OutcomeClass):
        return value.value
    if type(value) is str:
        try:
            return OutcomeClass(value).value
        except ValueError:
            pass
    raise EventAttributeError("unknown_outcome_class", field_name)


def _validate_outcome_reason(
    value: Any, field_name: str, data: Mapping[str, Any]
) -> str:
    if "outcome_class" not in data:
        raise EventAttributeError("outcome_class_required", field_name)
    outcome_class = _validate_outcome_class(data["outcome_class"], "outcome_class")
    if type(value) is not str or value not in allowed_reason_codes(outcome_class):
        raise EventAttributeError("unknown_outcome_reason", field_name)
    return value


def _validate_duration(value: Any, field_name: str) -> float:
    if type(value) not in (int, float):
        raise EventAttributeError("invalid_duration", field_name)
    normalized = float(value)
    if not math.isfinite(normalized):
        raise EventAttributeError("non_finite_number", field_name)
    if not 0.0 <= normalized <= MAX_DURATION_MS:
        raise EventAttributeError("duration_out_of_range", field_name)
    return normalized


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EventAttributeError("duplicate_attribute")
        result[key] = value
    return result


__all__ = [
    "ALLOWED_ATTRIBUTES",
    "EVENT_ATTRIBUTES_SCHEMA_VERSION",
    "EXECUTION_STAGES",
    "MAX_ATTRIBUTE_COUNT",
    "MAX_ATTRIBUTE_JSON_BYTES",
    "MAX_COUNT_VALUE",
    "MAX_DURATION_MS",
    "AttributeKind",
    "EventAttributeError",
    "EventAttributes",
    "validate_event_attributes",
]
