"""Lint clinical agent input schemas for minimum-data declarations.

The linter inspects schema structure and developer-authored governance metadata
only. Findings contain stable codes and JSON Pointer paths; schema values,
examples, defaults, and descriptions are never retained or rendered.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from openmed.agent.identifiers import GovernanceIdError, PurposeId

TOOL_CONTRACT_LINT_SCHEMA_VERSION: Final = "openmed.agent.tool_contract_lint.v1"
PURPOSE_ANNOTATION: Final = "x-openmed-purpose"
MINIMUM_DATA_ANNOTATION: Final = "x-openmed-minimum-data"

_SUPPORTED_MINIMUM_DATA = frozenset({"required", "optional", "derived"})
_SUPPORTED_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_MAX_SCHEMA_DEPTH = 64
_UNSUPPORTED_INPUT_KEYWORDS = frozenset(
    {
        "$ref",
        "additionalItems",
        "allOf",
        "anyOf",
        "dependentSchemas",
        "else",
        "if",
        "oneOf",
        "patternProperties",
        "prefixItems",
        "then",
        "unevaluatedProperties",
    }
)


class LintSeverity(str, Enum):
    """Stable severities emitted by the minimum-data contract linter."""

    ERROR = "error"


class LintReasonCode(str, Enum):
    """Stable reasons a tool input contract fails minimum-data review."""

    INVALID_SCHEMA = "invalid_schema"
    MISSING_PURPOSE = "missing_purpose"
    INVALID_PURPOSE = "invalid_purpose"
    OVERBROAD_FOR_PURPOSE = "overbroad_for_purpose"
    MISSING_MINIMUM_DATA = "missing_minimum_data"
    INVALID_MINIMUM_DATA = "invalid_minimum_data"
    OPTIONAL_INPUT = "optional_input"
    DERIVED_INPUT = "derived_input"
    OPEN_INPUT_OBJECT = "open_input_object"
    UNSUPPORTED_SCHEMA_KEYWORD = "unsupported_schema_keyword"


@dataclass(frozen=True, slots=True, repr=False)
class ToolContractFinding:
    """One value-free contract finding at a JSON Pointer schema path."""

    severity: LintSeverity
    reason_code: LintReasonCode
    schema_path: str

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic machine-readable finding."""

        return {
            "severity": self.severity.value,
            "reason_code": self.reason_code.value,
            "schema_path": self.schema_path,
        }

    def __repr__(self) -> str:
        """Return a representation that cannot expose contract contents."""

        return "ToolContractFinding(<value-free>)"


@dataclass(frozen=True, slots=True, repr=False)
class ToolContractLintReport:
    """Deterministically ordered minimum-data contract lint results."""

    findings: tuple[ToolContractFinding, ...]
    schema_version: str = TOOL_CONTRACT_LINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                set(self.findings),
                key=lambda finding: (
                    finding.schema_path,
                    finding.reason_code.value,
                    finding.severity.value,
                ),
            )
        )
        object.__setattr__(self, "findings", ordered)

    @property
    def passed(self) -> bool:
        """Return whether the contract has no blocking findings."""

        return not self.findings

    def to_dict(self) -> dict[str, object]:
        """Return a value-free JSON-compatible report."""

        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        """Render the report as deterministic compact JSON."""

        return json.dumps(
            self.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

    def __repr__(self) -> str:
        """Return a representation that cannot expose contract contents."""

        return f"ToolContractLintReport(findings={len(self.findings)})"


def lint_tool_contract(schema: Mapping[str, Any]) -> ToolContractLintReport:
    """Lint one JSON input schema without reading or retaining example values.

    The root declares the tool purpose with ``x-openmed-purpose``. Every field
    repeats that purpose and declares ``x-openmed-minimum-data`` as
    ``required``, ``optional``, or ``derived``. Only required fields supplied
    directly for the root purpose pass review; optional, derived, or
    mismatched-purpose fields expand the accepted input surface and fail.

    Args:
        schema: JSON-Schema-like mapping for a tool's input object.

    Returns:
        A deterministic report containing only severities, reason codes, and
        JSON Pointer paths.
    """

    findings: list[ToolContractFinding] = []
    if type(schema) is not dict:
        _add(findings, LintReasonCode.INVALID_SCHEMA, "#")
        return ToolContractLintReport(tuple(findings))

    root_purpose = _purpose_at(
        schema,
        path="#",
        findings=findings,
        required_purpose=None,
    )
    if schema.get("type") != "object":
        _add(findings, LintReasonCode.INVALID_SCHEMA, "#/type")
    _walk_object_schema(
        schema,
        path="#",
        root_purpose=root_purpose,
        findings=findings,
        require_annotations=False,
        depth=0,
        ancestors=frozenset(),
    )
    return ToolContractLintReport(tuple(findings))


def _walk_object_schema(
    schema: dict[str, Any],
    *,
    path: str,
    root_purpose: str | None,
    findings: list[ToolContractFinding],
    require_annotations: bool,
    depth: int,
    ancestors: frozenset[int],
) -> None:
    if depth > _MAX_SCHEMA_DEPTH or id(schema) in ancestors:
        _add(findings, LintReasonCode.INVALID_SCHEMA, path)
        return
    child_ancestors = ancestors | {id(schema)}

    for keyword in sorted(_UNSUPPORTED_INPUT_KEYWORDS & schema.keys()):
        _add(
            findings,
            LintReasonCode.UNSUPPORTED_SCHEMA_KEYWORD,
            _join(path, keyword),
        )

    if require_annotations:
        _purpose_at(
            schema,
            path=path,
            findings=findings,
            required_purpose=root_purpose,
        )
        minimum_data = schema.get(MINIMUM_DATA_ANNOTATION)
        annotation_path = _join(path, MINIMUM_DATA_ANNOTATION)
        if minimum_data is None:
            _add(findings, LintReasonCode.MISSING_MINIMUM_DATA, annotation_path)
        elif (
            type(minimum_data) is not str or minimum_data not in _SUPPORTED_MINIMUM_DATA
        ):
            _add(findings, LintReasonCode.INVALID_MINIMUM_DATA, annotation_path)
        elif minimum_data == "optional":
            _add(findings, LintReasonCode.OPTIONAL_INPUT, path)
        elif minimum_data == "derived":
            _add(findings, LintReasonCode.DERIVED_INPUT, path)

    schema_type = schema.get("type")
    properties = schema.get("properties")
    required = schema.get("required", [])

    if require_annotations and schema_type not in _SUPPORTED_SCHEMA_TYPES:
        _add(findings, LintReasonCode.INVALID_SCHEMA, _join(path, "type"))

    if properties is not None and type(properties) is not dict:
        _add(findings, LintReasonCode.INVALID_SCHEMA, _join(path, "properties"))
        return
    if (
        type(required) is not list
        or any(type(name) is not str for name in required)
        or len(required) != len(set(required))
    ):
        _add(findings, LintReasonCode.INVALID_SCHEMA, _join(path, "required"))
        required_names: set[str] = set()
    else:
        required_names = set(required)

    if properties is not None and schema_type != "object":
        _add(findings, LintReasonCode.INVALID_SCHEMA, _join(path, "type"))
    if schema_type == "object" and schema.get("additionalProperties") is not False:
        _add(
            findings,
            LintReasonCode.OPEN_INPUT_OBJECT,
            _join(path, "additionalProperties"),
        )
    if schema_type == "array" and "items" not in schema:
        _add(findings, LintReasonCode.INVALID_SCHEMA, _join(path, "items"))

    if properties is not None:
        if any(type(name) is not str for name in properties):
            _add(findings, LintReasonCode.INVALID_SCHEMA, _join(path, "properties"))
        for name in sorted(name for name in properties if type(name) is str):
            field_path = _join(_join(path, "properties"), name)
            field_schema = properties[name]
            if type(field_schema) is not dict:
                _add(findings, LintReasonCode.INVALID_SCHEMA, field_path)
                continue
            _walk_field(
                field_schema,
                path=field_path,
                root_purpose=root_purpose,
                is_required=name in required_names,
                findings=findings,
                depth=depth + 1,
                ancestors=child_ancestors,
            )

    if required_names and properties is not None:
        undeclared = required_names - set(properties)
        if undeclared:
            _add(findings, LintReasonCode.INVALID_SCHEMA, _join(path, "required"))


def _walk_field(
    schema: dict[str, Any],
    *,
    path: str,
    root_purpose: str | None,
    is_required: bool,
    findings: list[ToolContractFinding],
    depth: int,
    ancestors: frozenset[int],
) -> None:
    _walk_object_schema(
        schema,
        path=path,
        root_purpose=root_purpose,
        findings=findings,
        require_annotations=True,
        depth=depth,
        ancestors=ancestors,
    )

    if not is_required:
        _add(findings, LintReasonCode.OPTIONAL_INPUT, path)

    items = schema.get("items")
    if items is not None:
        items_path = _join(path, "items")
        if type(items) is not dict:
            _add(findings, LintReasonCode.INVALID_SCHEMA, items_path)
        else:
            _walk_object_schema(
                items,
                path=items_path,
                root_purpose=root_purpose,
                findings=findings,
                require_annotations=False,
                depth=depth + 1,
                ancestors=ancestors | {id(schema)},
            )


def _purpose_at(
    schema: dict[str, Any],
    *,
    path: str,
    findings: list[ToolContractFinding],
    required_purpose: str | None,
) -> str | None:
    annotation_path = _join(path, PURPOSE_ANNOTATION)
    if PURPOSE_ANNOTATION not in schema:
        _add(findings, LintReasonCode.MISSING_PURPOSE, annotation_path)
        return None

    purpose = schema[PURPOSE_ANNOTATION]
    try:
        parsed = PurposeId.parse(purpose)
    except GovernanceIdError:
        _add(findings, LintReasonCode.INVALID_PURPOSE, annotation_path)
        return None

    if parsed.version is None:
        _add(findings, LintReasonCode.INVALID_PURPOSE, annotation_path)
        return None
    if required_purpose is not None and purpose != required_purpose:
        _add(findings, LintReasonCode.OVERBROAD_FOR_PURPOSE, annotation_path)
    return purpose


def _add(
    findings: list[ToolContractFinding],
    reason_code: LintReasonCode,
    schema_path: str,
) -> None:
    findings.append(
        ToolContractFinding(
            severity=LintSeverity.ERROR,
            reason_code=reason_code,
            schema_path=schema_path,
        )
    )


def _join(path: str, token: str) -> str:
    escaped = token.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


__all__ = [
    "MINIMUM_DATA_ANNOTATION",
    "PURPOSE_ANNOTATION",
    "TOOL_CONTRACT_LINT_SCHEMA_VERSION",
    "LintReasonCode",
    "LintSeverity",
    "ToolContractFinding",
    "ToolContractLintReport",
    "lint_tool_contract",
]
