"""Deterministic, privacy-safe agent policy decision matrices.

Matrices contain governance identifiers, closed outcome reason codes, and a
review flag only. They never accept prompts, tool payloads, clinical outputs,
evidence text, credentials, paths, or arbitrary annotations.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from .identifiers import (
    CapabilityId,
    GovernanceIdError,
    PolicyId,
    PurposeId,
    ToolId,
)
from .outcomes import OutcomeClass, allowed_reason_codes

POLICY_MATRIX_SCHEMA_VERSION: Final = "openmed.agent.policy_matrix.v1"
MAX_POLICY_MATRIX_ROWS: Final = 10_000

_ROW_FIELDS = frozenset(
    {
        "policy_version",
        "capability_id",
        "purpose_id",
        "tool_id",
        "outcome_reason_code",
        "reviewer_required",
    }
)
_MATRIX_FIELDS = frozenset({"schema_version", "rows"})
_KNOWN_REASON_CODES = frozenset().union(
    *(allowed_reason_codes(outcome_class) for outcome_class in OutcomeClass)
)


class PolicyMatrixError(ValueError):
    """Raised when policy-matrix input fails closed validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional public field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PolicyDecisionRow:
    """One metadata-only policy decision.

    Args:
        policy_version: Canonical, explicitly versioned policy identifier.
        capability_id: Canonical capability identifier.
        purpose_id: Canonical purpose identifier.
        tool_id: Canonical tool identifier.
        outcome_reason_code: Closed workflow outcome reason code.
        reviewer_required: Whether a human reviewer is required.
    """

    policy_version: PolicyId
    capability_id: CapabilityId
    purpose_id: PurposeId
    tool_id: ToolId
    outcome_reason_code: str
    reviewer_required: bool

    def __post_init__(self) -> None:
        for field_name, value, expected_type in (
            ("policy_version", self.policy_version, PolicyId),
            ("capability_id", self.capability_id, CapabilityId),
            ("purpose_id", self.purpose_id, PurposeId),
            ("tool_id", self.tool_id, ToolId),
        ):
            if type(value) is not expected_type:
                raise PolicyMatrixError("wrong_identifier_kind", field_name)
        if self.policy_version.version is None:
            raise PolicyMatrixError("version_required", "policy_version")
        if (
            type(self.outcome_reason_code) is not str
            or self.outcome_reason_code not in _KNOWN_REASON_CODES
        ):
            raise PolicyMatrixError("unknown_outcome_code", "outcome_reason_code")
        if type(self.reviewer_required) is not bool:
            raise PolicyMatrixError("invalid_boolean", "reviewer_required")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyDecisionRow":
        """Build a decision row from an exact metadata-only mapping."""
        values = _read_exact_mapping(data, _ROW_FIELDS, "row")
        try:
            policy_version = PolicyId.parse(values["policy_version"])
            capability_id = CapabilityId.parse(values["capability_id"])
            purpose_id = PurposeId.parse(values["purpose_id"])
            tool_id = ToolId.parse(values["tool_id"])
        except GovernanceIdError:
            raise PolicyMatrixError("invalid_identifier", "row") from None
        return cls(
            policy_version=policy_version,
            capability_id=capability_id,
            purpose_id=purpose_id,
            tool_id=tool_id,
            outcome_reason_code=values["outcome_reason_code"],
            reviewer_required=values["reviewer_required"],
        )

    @property
    def decision_key(self) -> tuple[str, str, str, str]:
        """Return the stable key identifying this policy decision."""
        return (
            self.policy_version.serialize(),
            self.capability_id.serialize(),
            self.purpose_id.serialize(),
            self.tool_id.serialize(),
        )

    def to_dict(self) -> dict[str, str | bool]:
        """Return deterministic, metadata-only JSON-compatible data."""
        return {
            "policy_version": self.policy_version.serialize(),
            "capability_id": self.capability_id.serialize(),
            "purpose_id": self.purpose_id.serialize(),
            "tool_id": self.tool_id.serialize(),
            "outcome_reason_code": self.outcome_reason_code,
            "reviewer_required": self.reviewer_required,
        }


@dataclass(frozen=True, slots=True)
class PolicyDecisionMatrix:
    """Deterministically ordered collection of policy decision rows.

    Use :meth:`from_rows` when caller input is not already sorted. Direct
    construction accepts only a canonical sorted tuple with unique decision
    keys.
    """

    rows: tuple[PolicyDecisionRow, ...]
    schema_version: str = POLICY_MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != POLICY_MATRIX_SCHEMA_VERSION
        ):
            raise PolicyMatrixError("invalid_schema_version", "schema_version")
        if type(self.rows) is not tuple:
            raise PolicyMatrixError("invalid_sequence", "rows")
        if len(self.rows) > MAX_POLICY_MATRIX_ROWS:
            raise PolicyMatrixError("too_many_items", "rows")
        if any(type(row) is not PolicyDecisionRow for row in self.rows):
            raise PolicyMatrixError("invalid_item", "rows")

        keys = tuple(row.decision_key for row in self.rows)
        if len(set(keys)) != len(keys):
            raise PolicyMatrixError("duplicate_decision_key", "rows")
        if keys != tuple(sorted(keys)):
            raise PolicyMatrixError("not_sorted", "rows")

    @classmethod
    def from_rows(cls, rows: Iterable[PolicyDecisionRow]) -> "PolicyDecisionMatrix":
        """Build a matrix with stable row ordering and duplicate validation."""
        if isinstance(rows, (str, bytes, bytearray, Mapping)):
            raise PolicyMatrixError("invalid_iterable", "rows")
        try:
            iterator = iter(rows)
        except TypeError:
            raise PolicyMatrixError("invalid_iterable", "rows") from None

        normalized: list[PolicyDecisionRow] = []
        for index, row in enumerate(iterator):
            if index >= MAX_POLICY_MATRIX_ROWS:
                raise PolicyMatrixError("too_many_items", "rows")
            if type(row) is not PolicyDecisionRow:
                raise PolicyMatrixError("invalid_item", "rows")
            normalized.append(row)
        normalized.sort(key=lambda row: row.decision_key)
        return cls(rows=tuple(normalized))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyDecisionMatrix":
        """Build a matrix from an exact metadata-only mapping."""
        values = _read_exact_mapping(data, _MATRIX_FIELDS, "matrix")
        if (
            type(values["schema_version"]) is not str
            or values["schema_version"] != POLICY_MATRIX_SCHEMA_VERSION
        ):
            raise PolicyMatrixError("invalid_schema_version", "schema_version")
        if type(values["rows"]) not in (list, tuple):
            raise PolicyMatrixError("invalid_sequence", "rows")
        if len(values["rows"]) > MAX_POLICY_MATRIX_ROWS:
            raise PolicyMatrixError("too_many_items", "rows")
        rows = (PolicyDecisionRow.from_dict(row) for row in values["rows"])
        return cls.from_rows(rows)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "PolicyDecisionMatrix":
        """Build a matrix from JSON while rejecting duplicate object fields."""
        if not isinstance(payload, (str, bytes, bytearray)):
            raise PolicyMatrixError("invalid_json", "matrix")
        try:
            decoded = json.loads(payload, object_pairs_hook=_strict_json_object)
        except (KeyboardInterrupt, SystemExit):
            raise
        except PolicyMatrixError:
            raise
        except Exception:
            raise PolicyMatrixError("invalid_json", "matrix") from None
        return cls.from_dict(decoded)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, metadata-only JSON-compatible data."""
        return {
            "schema_version": self.schema_version,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self) -> str:
        """Return compact deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_markdown(self) -> str:
        """Return deterministic metadata-only Markdown."""
        lines = [
            "# Agent Policy Decision Matrix",
            "",
            f"Schema: `{self.schema_version}`",
            "",
            (
                "| Policy version | Capability | Purpose | Tool | "
                "Outcome reason | Reviewer required |"
            ),
            "| --- | --- | --- | --- | --- | --- |",
            *(
                "| "
                f"`{row.policy_version.serialize()}` | "
                f"`{row.capability_id.serialize()}` | "
                f"`{row.purpose_id.serialize()}` | "
                f"`{row.tool_id.serialize()}` | "
                f"`{row.outcome_reason_code}` | "
                f"{'yes' if row.reviewer_required else 'no'} |"
                for row in self.rows
            ),
        ]
        return "\n".join(lines) + "\n"


def _read_exact_mapping(
    data: Mapping[str, Any], expected_fields: frozenset[str], location: str
) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise PolicyMatrixError("not_a_mapping", location)
    try:
        fields = set(data)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise PolicyMatrixError("not_a_mapping", location) from None
    if fields - expected_fields:
        raise PolicyMatrixError("unknown_field", location)
    if expected_fields - fields:
        raise PolicyMatrixError("missing_field", location)
    try:
        return {field_name: data[field_name] for field_name in expected_fields}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise PolicyMatrixError("unreadable_mapping", location) from None


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyMatrixError("duplicate_field", "matrix")
        result[key] = value
    return result


__all__ = [
    "MAX_POLICY_MATRIX_ROWS",
    "POLICY_MATRIX_SCHEMA_VERSION",
    "PolicyDecisionMatrix",
    "PolicyDecisionRow",
    "PolicyMatrixError",
]
