"""Plan minimum-data projections before clinical tool inputs are materialized.

The planner consumes developer-authored schema and governance metadata only.
It never accepts record values, invokes a tool, or performs a network request.
Plans and denial rationales contain only controlled reason codes, schema and
field paths, and canonical data-class identifiers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, cast

from openmed.agent.identifiers import GovernanceIdError, PurposeId
from openmed.agent.tool_contract_lint import (
    MINIMUM_DATA_ANNOTATION,
    PURPOSE_ANNOTATION,
    lint_tool_contract,
)

DATA_CLASS_ANNOTATION: Final = "x-openmed-data-class"
DATA_PROJECTION_PLAN_SCHEMA_VERSION: Final = "openmed.agent.data_projection.v1"
DATA_PROJECTION_RATIONALE_SCHEMA_VERSION: Final = (
    "openmed.agent.data_projection_rationale.v1"
)

_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE = rf"{_LABEL}(?:\.{_LABEL})+"
_LOCAL_NAME = r"[a-z][a-z0-9-]{0,63}"
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
_DATA_CLASS_RE = re.compile(rf"data:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")
_MAX_SCHEMA_DEPTH = 64


class ProjectionDecision(str, Enum):
    """Whether one declared field is included in the planned projection."""

    INCLUDE = "include"
    DENY = "deny"


class ProjectionReasonCode(str, Enum):
    """Stable, value-free reasons recorded by the projection planner."""

    REQUIRED_FOR_PURPOSE = "required_for_purpose"
    DATA_CLASS_NOT_GRANTED = "data_class_not_granted"
    PURPOSE_MISMATCH = "purpose_mismatch"


class DataProjectionError(ValueError):
    """Base class for value-free projection failures."""

    def __init__(self, code: str, field_name: str) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(f"{field_name}: {code}")


class DataProjectionValidationError(DataProjectionError):
    """Raised when projection governance metadata is malformed."""


class DataProjectionDeniedError(DataProjectionError):
    """Raised when a tool's declared requirements exceed active authority."""

    def __init__(self, rationale: "DataProjectionRationale") -> None:
        if type(rationale) is not DataProjectionRationale or rationale.approved:
            raise DataProjectionValidationError("invalid_rationale", "projection")
        self.rationale = rationale
        super().__init__("projection_denied", "projection")


@dataclass(frozen=True, slots=True, repr=False)
class ProjectionRationaleEntry:
    """One value-free decision for a schema field or the root purpose."""

    decision: ProjectionDecision
    reason_code: ProjectionReasonCode
    schema_path: str
    field_path: str | None
    data_class: str | None

    def __post_init__(self) -> None:
        if type(self.decision) is not ProjectionDecision:
            raise DataProjectionValidationError("invalid_decision", "rationale")
        if type(self.reason_code) is not ProjectionReasonCode:
            raise DataProjectionValidationError("invalid_reason", "rationale")
        if type(self.schema_path) is not str or not _is_safe_schema_path(
            self.schema_path
        ):
            raise DataProjectionValidationError("invalid_path", "rationale")
        if self.field_path is not None and (
            type(self.field_path) is not str or not _is_safe_field_path(self.field_path)
        ):
            raise DataProjectionValidationError("invalid_path", "rationale")
        if self.data_class is not None:
            _validate_data_class(self.data_class, "rationale")
        if self.reason_code is ProjectionReasonCode.PURPOSE_MISMATCH:
            if (
                self.decision is not ProjectionDecision.DENY
                or self.schema_path != f"#/{PURPOSE_ANNOTATION}"
                or self.field_path is not None
                or self.data_class is not None
            ):
                raise DataProjectionValidationError(
                    "invalid_purpose_rationale", "rationale"
                )
        elif self.field_path is None or self.data_class is None:
            raise DataProjectionValidationError("incomplete_rationale", "rationale")
        elif (
            self.reason_code is ProjectionReasonCode.REQUIRED_FOR_PURPOSE
            and self.decision is not ProjectionDecision.INCLUDE
        ) or (
            self.reason_code is ProjectionReasonCode.DATA_CLASS_NOT_GRANTED
            and self.decision is not ProjectionDecision.DENY
        ):
            raise DataProjectionValidationError("inconsistent_rationale", "rationale")

    def to_dict(self) -> dict[str, str | None]:
        """Return a deterministic JSON-compatible rationale entry."""

        return {
            "data_class": self.data_class,
            "decision": self.decision.value,
            "field_path": self.field_path,
            "reason_code": self.reason_code.value,
            "schema_path": self.schema_path,
        }

    def __repr__(self) -> str:
        """Return a representation that cannot expose schema metadata."""

        return "ProjectionRationaleEntry(<value-free>)"


@dataclass(frozen=True, slots=True, repr=False)
class DataProjectionRationale:
    """Canonical value-free evidence for one projection decision."""

    entries: tuple[ProjectionRationaleEntry, ...]
    schema_version: str = DATA_PROJECTION_RATIONALE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATA_PROJECTION_RATIONALE_SCHEMA_VERSION:
            raise DataProjectionValidationError("invalid_schema_version", "rationale")
        if type(self.entries) is not tuple or not all(
            type(entry) is ProjectionRationaleEntry for entry in self.entries
        ):
            raise DataProjectionValidationError("invalid_entries", "rationale")
        ordered = tuple(
            sorted(
                set(self.entries),
                key=lambda entry: (
                    entry.schema_path,
                    entry.field_path or "",
                    entry.data_class or "",
                    entry.decision.value,
                    entry.reason_code.value,
                ),
            )
        )
        if len({entry.schema_path for entry in ordered}) != len(ordered) or len(
            {entry.field_path for entry in ordered if entry.field_path is not None}
        ) != sum(entry.field_path is not None for entry in ordered):
            raise DataProjectionValidationError("conflicting_entries", "rationale")
        object.__setattr__(self, "entries", ordered)

    @property
    def approved(self) -> bool:
        """Return whether every declared requirement is granted."""

        return all(
            entry.decision is ProjectionDecision.INCLUDE for entry in self.entries
        )

    def to_dict(self) -> dict[str, object]:
        """Return deterministic evidence containing no clinical values."""

        return {
            "approved": self.approved,
            "entries": [entry.to_dict() for entry in self.entries],
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Serialize the rationale as canonical JSON."""

        return json.dumps(
            self.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

    def __repr__(self) -> str:
        """Return a value-free summary of this rationale."""

        return f"DataProjectionRationale(entries={len(self.entries)})"


@dataclass(frozen=True, slots=True, repr=False)
class DataProjectionPlan:
    """Approved field paths and data classes for later local materialization."""

    field_paths: tuple[str, ...]
    data_classes: tuple[str, ...]
    rationale: DataProjectionRationale
    schema_version: str = DATA_PROJECTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATA_PROJECTION_PLAN_SCHEMA_VERSION:
            raise DataProjectionValidationError("invalid_schema_version", "plan")
        if type(self.field_paths) is not tuple or any(
            type(path) is not str or not path.startswith("/")
            for path in self.field_paths
        ):
            raise DataProjectionValidationError("invalid_paths", "plan")
        if tuple(sorted(set(self.field_paths))) != self.field_paths:
            raise DataProjectionValidationError("noncanonical_paths", "plan")
        if type(self.data_classes) is not tuple:
            raise DataProjectionValidationError("invalid_data_classes", "plan")
        for data_class in self.data_classes:
            _validate_data_class(data_class, "plan")
        if tuple(sorted(set(self.data_classes))) != self.data_classes:
            raise DataProjectionValidationError("noncanonical_data_classes", "plan")
        if type(self.rationale) is not DataProjectionRationale:
            raise DataProjectionValidationError("invalid_rationale", "plan")
        if not self.rationale.approved:
            raise DataProjectionValidationError("denied_rationale", "plan")
        included_paths = tuple(
            sorted(
                entry.field_path
                for entry in self.rationale.entries
                if entry.field_path is not None
            )
        )
        included_classes = tuple(
            sorted(
                {
                    entry.data_class
                    for entry in self.rationale.entries
                    if entry.data_class is not None
                }
            )
        )
        if self.field_paths != included_paths or self.data_classes != included_classes:
            raise DataProjectionValidationError("inconsistent_plan", "plan")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible plan."""

        return {
            "data_classes": list(self.data_classes),
            "field_paths": list(self.field_paths),
            "rationale": self.rationale.to_dict(),
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Serialize the plan as canonical JSON."""

        return json.dumps(
            self.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

    def __repr__(self) -> str:
        """Return a value-free summary of this plan."""

        return f"DataProjectionPlan(fields={len(self.field_paths)})"


def plan_data_projection(
    schema: Mapping[str, Any],
    *,
    workflow_purpose: str,
    granted_data_classes: tuple[str, ...],
) -> DataProjectionPlan:
    """Derive an authorized field projection from governance metadata.

    Callers should pass ``AccessTicket.permitted_data_classes`` only after the
    ticket has been verified for the active run. The resulting paths can then
    be handed to a trusted local materializer; this function deliberately has
    no argument for record values or a materialization callback.

    Args:
        schema: Reviewed JSON input schema for the proposed tool.
        workflow_purpose: Versioned purpose of the active workflow.
        granted_data_classes: Exact data classes granted to the active run.

    Returns:
        A deterministic approved projection and value-free rationale.

    Raises:
        DataProjectionValidationError: If governance metadata is malformed.
        DataProjectionDeniedError: If purpose or field requirements exceed the
            active grant.
    """

    purpose = _validate_purpose(workflow_purpose)
    grants = _validate_grants(granted_data_classes)
    lint_report = lint_tool_contract(schema)
    if not lint_report.passed:
        raise DataProjectionValidationError("invalid_tool_schema", "schema")
    tool_schema = cast(dict[str, Any], schema)
    declared_purpose = cast(str, tool_schema[PURPOSE_ANNOTATION])
    if declared_purpose != purpose:
        rationale = DataProjectionRationale(
            (
                ProjectionRationaleEntry(
                    decision=ProjectionDecision.DENY,
                    reason_code=ProjectionReasonCode.PURPOSE_MISMATCH,
                    schema_path=f"#/{PURPOSE_ANNOTATION}",
                    field_path=None,
                    data_class=None,
                ),
            )
        )
        raise DataProjectionDeniedError(rationale)

    declarations: list[tuple[str, str, str]] = []
    _collect_fields(
        tool_schema,
        schema_path="#",
        field_path="",
        declarations=declarations,
        depth=0,
        ancestors=frozenset(),
    )
    entries = tuple(
        ProjectionRationaleEntry(
            decision=(
                ProjectionDecision.INCLUDE
                if data_class in grants
                else ProjectionDecision.DENY
            ),
            reason_code=(
                ProjectionReasonCode.REQUIRED_FOR_PURPOSE
                if data_class in grants
                else ProjectionReasonCode.DATA_CLASS_NOT_GRANTED
            ),
            schema_path=schema_path,
            field_path=field_path,
            data_class=data_class,
        )
        for schema_path, field_path, data_class in declarations
    )
    rationale = DataProjectionRationale(entries)
    if not rationale.approved:
        raise DataProjectionDeniedError(rationale)
    field_paths = tuple(
        sorted(cast(str, entry.field_path) for entry in rationale.entries)
    )
    data_classes = tuple(
        sorted({cast(str, entry.data_class) for entry in rationale.entries})
    )
    return DataProjectionPlan(
        field_paths=field_paths,
        data_classes=data_classes,
        rationale=rationale,
    )


def _collect_fields(
    schema: dict[str, Any],
    *,
    schema_path: str,
    field_path: str,
    declarations: list[tuple[str, str, str]],
    depth: int,
    ancestors: frozenset[int],
) -> None:
    if depth > _MAX_SCHEMA_DEPTH or id(schema) in ancestors:
        raise DataProjectionValidationError("invalid_tool_schema", "schema")
    child_ancestors = ancestors | {id(schema)}
    properties = cast(dict[str, Any], schema.get("properties", {}))
    for name in sorted(properties):
        if _FIELD_RE.fullmatch(name) is None:
            raise DataProjectionValidationError("unsafe_field_name", "schema")
        child = cast(dict[str, Any], properties[name])
        child_schema_path = _join_schema(_join_schema(schema_path, "properties"), name)
        child_field_path = _join_field(field_path, name)
        if child.get(MINIMUM_DATA_ANNOTATION) != "required":
            raise DataProjectionValidationError("invalid_tool_schema", "schema")
        data_class = _validate_data_class(child.get(DATA_CLASS_ANNOTATION), "schema")
        declarations.append((child_schema_path, child_field_path, data_class))
        _collect_fields(
            child,
            schema_path=child_schema_path,
            field_path=child_field_path,
            declarations=declarations,
            depth=depth + 1,
            ancestors=child_ancestors,
        )
        items = child.get("items")
        if type(items) is dict:
            _collect_fields(
                items,
                schema_path=_join_schema(child_schema_path, "items"),
                field_path=f"{child_field_path}/*",
                declarations=declarations,
                depth=depth + 1,
                ancestors=child_ancestors | {id(child)},
            )


def _validate_purpose(value: object) -> str:
    try:
        purpose = PurposeId.parse(value)
    except GovernanceIdError as exc:
        raise DataProjectionValidationError(
            "invalid_governance_identifier", "workflow_purpose"
        ) from exc
    if purpose.version is None:
        raise DataProjectionValidationError(
            "unversioned_governance_identifier", "workflow_purpose"
        )
    return cast(str, value)


def _validate_grants(values: object) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise DataProjectionValidationError("invalid_grant", "granted_data_classes")
    for value in values:
        _validate_data_class(value, "granted_data_classes")
    canonical = tuple(sorted(values))
    if len(set(canonical)) != len(canonical):
        raise DataProjectionValidationError("duplicate_grant", "granted_data_classes")
    return canonical


def _validate_data_class(value: object, field_name: str) -> str:
    if type(value) is not str or _DATA_CLASS_RE.fullmatch(value) is None:
        raise DataProjectionValidationError("invalid_governance_identifier", field_name)
    return value


def _join_schema(path: str, token: str) -> str:
    escaped = token.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _join_field(path: str, token: str) -> str:
    escaped = token.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _is_safe_schema_path(path: str) -> bool:
    if path == f"#/{PURPOSE_ANNOTATION}":
        return True
    tokens = path.split("/")
    if not tokens or tokens[0] != "#" or len(tokens) < 3:
        return False
    expect_field = False
    for token in tokens[1:]:
        if expect_field:
            if _FIELD_RE.fullmatch(token) is None:
                return False
            expect_field = False
        elif token == "properties":
            expect_field = True
        elif token != "items":
            return False
    return not expect_field


def _is_safe_field_path(path: str) -> bool:
    tokens = path.split("/")
    return bool(tokens and tokens[0] == "" and len(tokens) > 1) and all(
        token == "*" or _FIELD_RE.fullmatch(token) is not None for token in tokens[1:]
    )


__all__ = [
    "DATA_CLASS_ANNOTATION",
    "DATA_PROJECTION_PLAN_SCHEMA_VERSION",
    "DATA_PROJECTION_RATIONALE_SCHEMA_VERSION",
    "DataProjectionDeniedError",
    "DataProjectionError",
    "DataProjectionPlan",
    "DataProjectionRationale",
    "DataProjectionValidationError",
    "ProjectionDecision",
    "ProjectionRationaleEntry",
    "ProjectionReasonCode",
    "plan_data_projection",
]
