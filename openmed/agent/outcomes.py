"""Stable outcome classes and reason codes for agent workflows.

The vocabulary is closed. Callers can record success, abstention, reviewer
handoff, policy denial, or execution failure without attaching free-text
status strings. Unknown identifiers fail closed, and exception messages name
fields or stable error codes rather than submitted values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

OUTCOME_SCHEMA_VERSION = "openmed.agent.outcome.v1"

_ALLOWED_FIELDS = frozenset({"schema_version", "outcome_class", "reason_code"})
_ORDERED_FIELDS = ("schema_version", "outcome_class", "reason_code")

_REASON_CODES: dict[str, frozenset[str]] = {
    "success": frozenset({"completed"}),
    "abstained": frozenset({"insufficient_evidence", "out_of_scope", "low_confidence"}),
    "review_required": frozenset(
        {"conflicting_evidence", "safety_review", "human_gate"}
    ),
    "policy_denied": frozenset({"consent_required", "purpose_mismatch", "phi_policy"}),
    "failed": frozenset({"tool_error", "timeout", "invalid_input"}),
}


class OutcomeClass(str, Enum):
    """Closed set of workflow outcomes.

    Values:
        SUCCESS: The workflow finished the requested work.
        ABSTAINED: The workflow stopped without producing a clinical answer.
        REVIEW_REQUIRED: A reviewer must inspect the run before use.
        POLICY_DENIED: A policy or consent check blocked execution.
        FAILED: Execution failed for a non-policy reason.
    """

    SUCCESS = "success"
    ABSTAINED = "abstained"
    REVIEW_REQUIRED = "review_required"
    POLICY_DENIED = "policy_denied"
    FAILED = "failed"


class OutcomeError(ValueError):
    """Raised when an outcome payload fails closed validation."""

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        if field_name is None:
            message = code
        else:
            message = f"{field_name}: {code}"
        super().__init__(message)


def allowed_reason_codes(outcome_class: OutcomeClass | str) -> frozenset[str]:
    """Return the closed reason-code set for an outcome class."""
    if isinstance(outcome_class, OutcomeClass):
        key = outcome_class.value
    elif type(outcome_class) is str:
        key = outcome_class
    else:
        raise OutcomeError("unknown_class", "outcome_class")
    try:
        return _REASON_CODES[key]
    except KeyError:
        pass
    raise OutcomeError("unknown_class", "outcome_class")


@dataclass(frozen=True, slots=True)
class WorkflowOutcome:
    """JSON-safe workflow outcome with a class and a closed reason code."""

    outcome_class: OutcomeClass
    reason_code: str
    schema_version: str = OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if not isinstance(self.outcome_class, OutcomeClass):
            raise OutcomeError("unknown_class", "outcome_class")
        _validate_reason_code(self.outcome_class, self.reason_code)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowOutcome":
        """Build and validate an outcome from a strict mapping."""
        if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
            raise OutcomeError("not_a_mapping")

        unknown: set[Any] | None = None
        try:
            unknown = set(data) - _ALLOWED_FIELDS
        except Exception:
            pass
        if unknown is None:
            raise OutcomeError("not_a_mapping")
        if unknown:
            raise OutcomeError("unknown_field")

        if "outcome_class" not in data or "reason_code" not in data:
            raise OutcomeError("missing_field")

        outcome_class = _parse_outcome_class(data["outcome_class"])
        schema_version = data.get("schema_version", OUTCOME_SCHEMA_VERSION)
        return cls(
            outcome_class=outcome_class,
            reason_code=data["reason_code"],
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "WorkflowOutcome":
        """Build and validate an outcome from a JSON object."""
        try:
            data = json.loads(payload, object_pairs_hook=_strict_json_object)
        except (json.JSONDecodeError, OutcomeError, TypeError, UnicodeDecodeError):
            pass
        else:
            return cls.from_dict(data)
        raise OutcomeError("malformed_json")

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic dictionary in field order."""
        values = {
            "schema_version": self.schema_version,
            "outcome_class": self.outcome_class.value,
            "reason_code": self.reason_code,
        }
        return {key: values[key] for key in _ORDERED_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _parse_outcome_class(value: Any) -> OutcomeClass:
    if isinstance(value, OutcomeClass):
        return value
    if type(value) is not str:
        raise OutcomeError("unknown_class", "outcome_class")
    try:
        return OutcomeClass(value)
    except ValueError:
        pass
    raise OutcomeError("unknown_class", "outcome_class")


def _validate_schema_version(value: Any) -> None:
    if type(value) is not str or value != OUTCOME_SCHEMA_VERSION:
        raise OutcomeError("invalid_schema_version", "schema_version")


def _validate_reason_code(outcome_class: OutcomeClass, value: Any) -> None:
    if type(value) is not str:
        raise OutcomeError("unknown_reason", "reason_code")
    allowed = _REASON_CODES[outcome_class.value]
    if value not in allowed:
        raise OutcomeError("unknown_reason", "reason_code")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OutcomeError("duplicate_field")
        result[key] = value
    return result


__all__ = [
    "OUTCOME_SCHEMA_VERSION",
    "OutcomeClass",
    "OutcomeError",
    "WorkflowOutcome",
    "allowed_reason_codes",
]
