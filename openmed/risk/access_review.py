"""Deterministic, value-free reviews of structured workflow access.

An access review answers a narrow governance question: do the fields a
workflow declares for reading or exporting fit the fields exposed by a
resource schema and the caller's deny policy? It does not inspect records or
schema metadata. Reports contain field names, counts, and access decisions;
they never contain field values.

The comparison is deliberately local and deterministic. Mapping values in a
resource schema are ignored, which means examples, defaults, and other schema
metadata cannot accidentally become report content. Field and workflow
identifiers are restricted to a safe structural form so a malformed input
cannot inject raw content into a report or an exception message.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

ACCESS_REVIEW_SCHEMA_VERSION = 1
READ_ACCESS = "read"
EXPORT_ACCESS = "export"
ACCESS_MODES = (READ_ACCESS, EXPORT_ACCESS)
_DEFAULT_WORKFLOW_NAME = "default"
_MISSING = object()
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}\Z")

FieldCollection: TypeAlias = Iterable[str] | Mapping[str, Any] | str | None
ResourceSchema: TypeAlias = Mapping[str, Any] | Iterable[str]

__all__ = [
    "ACCESS_MODES",
    "ACCESS_REVIEW_SCHEMA_VERSION",
    "EXPORT_ACCESS",
    "READ_ACCESS",
    "AccessModeReview",
    "AccessReviewError",
    "AccessReviewReport",
    "AccessReviewValidationError",
    "WorkflowAccessReview",
    "WorkflowRequirement",
    "access_review_report",
    "build_access_review_report",
    "render_access_review",
    "review_access",
    "review_structured_access",
]


class AccessReviewError(ValueError):
    """Base error for invalid structured access review declarations."""


class AccessReviewValidationError(AccessReviewError):
    """Raised when a review declaration cannot be interpreted safely."""


def _validation_error(message: str) -> AccessReviewValidationError:
    """Construct a validation error without interpolating input values."""

    return AccessReviewValidationError(message)


def _identifier(value: Any, *, kind: str) -> str:
    """Validate one report-visible structural identifier.

    The error intentionally describes only the input position and type. A
    caller may have supplied a sensitive value where a field name was expected,
    and echoing it in an exception would defeat the report's privacy boundary.
    """

    if not isinstance(value, str) or _SAFE_IDENTIFIER.fullmatch(value) is None:
        value_type = type(value).__name__
        raise _validation_error(
            f"{kind} must be a safe structural identifier ({value_type})"
        )
    return value


def _field_tuple(value: FieldCollection, *, kind: str) -> tuple[str, ...]:
    """Return sorted unique field names without inspecting mapping values."""

    if value is None:
        return ()
    if isinstance(value, str):
        candidates: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        # A mapping is a convenient schema/field declaration. Its values may
        # contain examples or defaults and are intentionally never traversed.
        candidates = value.keys()
    else:
        try:
            candidates = iter(value)
        except TypeError as exc:
            raise _validation_error(
                f"{kind} must be an iterable of field names"
            ) from exc

    fields: set[str] = set()
    for item in candidates:
        fields.add(_identifier(item, kind="field name"))
    return tuple(sorted(fields))


def _workflow_name(value: Any) -> str:
    return _identifier(value, kind="workflow name")


def _resource_fields(resource_schema: ResourceSchema) -> tuple[str, ...]:
    """Extract field names from common schema shapes, ignoring metadata."""

    if isinstance(resource_schema, Mapping):
        properties = resource_schema.get("properties", _MISSING)
        if properties is not _MISSING:
            if not isinstance(properties, Mapping):
                raise _validation_error("resource schema properties must be a mapping")
            return _field_tuple(properties.keys(), kind="resource schema fields")

        fields = resource_schema.get("fields", _MISSING)
        if fields is not _MISSING:
            return _field_tuple(fields, kind="resource schema fields")

        return _field_tuple(resource_schema.keys(), kind="resource schema fields")

    for attribute in ("fields", "columns", "names"):
        candidate = getattr(resource_schema, attribute, _MISSING)
        if candidate is not _MISSING:
            return _field_tuple(candidate, kind="resource schema fields")

    return _field_tuple(resource_schema, kind="resource schema fields")


def _mode_fields(declaration: Mapping[str, Any], mode: str) -> FieldCollection:
    """Read a mode declaration while accepting short and long spellings."""

    short_key = mode
    long_key = f"{mode}_fields"
    if short_key in declaration:
        return declaration[short_key]
    if long_key in declaration:
        return declaration[long_key]
    return ()


@dataclass(frozen=True)
class WorkflowRequirement:
    """Declared structured fields for one workflow.

    ``read_fields`` and ``export_fields`` are field names rather than record
    values. They are normalized to sorted tuples during construction.
    """

    name: str
    read_fields: FieldCollection = ()
    export_fields: FieldCollection = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _workflow_name(self.name))
        object.__setattr__(
            self,
            "read_fields",
            _field_tuple(self.read_fields, kind="read fields"),
        )
        object.__setattr__(
            self,
            "export_fields",
            _field_tuple(self.export_fields, kind="export fields"),
        )

    def fields_for(self, mode: str) -> tuple[str, ...]:
        """Return the declared fields for ``read`` or ``export``."""

        if mode == READ_ACCESS:
            return self.read_fields
        if mode == EXPORT_ACCESS:
            return self.export_fields
        raise _validation_error("access mode is unsupported")

    def to_dict(self) -> dict[str, Any]:
        """Return the declaration without record or schema values."""

        return {
            "workflow": self.name,
            "read_fields": list(self.read_fields),
            "export_fields": list(self.export_fields),
        }


@dataclass(frozen=True)
class AccessModeReview:
    """Value-free comparison for one workflow and one access mode."""

    mode: str
    requested_fields: tuple[str, ...]
    available_fields: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    excessive_fields: tuple[str, ...]
    denied_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.mode not in ACCESS_MODES:
            raise _validation_error("access mode is unsupported")
        for attribute in (
            "requested_fields",
            "available_fields",
            "allowed_fields",
            "missing_fields",
            "excessive_fields",
            "denied_fields",
        ):
            object.__setattr__(
                self,
                attribute,
                _field_tuple(
                    getattr(self, attribute),
                    kind=f"{self.mode} review fields",
                ),
            )

    @property
    def complete(self) -> bool:
        """Whether this mode has no missing, excessive, or denied fields."""

        return not (self.missing_fields or self.excessive_fields or self.denied_fields)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe mode review."""

        return {
            "mode": self.mode,
            "requested_fields": list(self.requested_fields),
            "available_fields": list(self.available_fields),
            "allowed_fields": list(self.allowed_fields),
            "missing_fields": list(self.missing_fields),
            "excessive_fields": list(self.excessive_fields),
            "denied_fields": list(self.denied_fields),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class WorkflowAccessReview:
    """Read and export access findings for one named workflow."""

    workflow: str
    read: AccessModeReview
    export: AccessModeReview

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow", _workflow_name(self.workflow))
        if not isinstance(self.read, AccessModeReview) or not isinstance(
            self.export, AccessModeReview
        ):
            raise _validation_error("workflow review modes must be access mode reviews")
        if self.read.mode != READ_ACCESS or self.export.mode != EXPORT_ACCESS:
            raise _validation_error("workflow review modes are in the wrong order")

    @property
    def modes(self) -> tuple[AccessModeReview, AccessModeReview]:
        """Return mode reviews in stable read-then-export order."""

        return (self.read, self.export)

    @property
    def missing_fields(self) -> tuple[str, ...]:
        """Return unique missing fields across both access modes."""

        return _union_fields(mode.missing_fields for mode in self.modes)

    @property
    def excessive_fields(self) -> tuple[str, ...]:
        """Return unique excessive fields across both access modes."""

        return _union_fields(mode.excessive_fields for mode in self.modes)

    @property
    def denied_fields(self) -> tuple[str, ...]:
        """Return unique denied fields across both access modes."""

        return _union_fields(mode.denied_fields for mode in self.modes)

    @property
    def complete(self) -> bool:
        """Whether both declared access modes have no review findings."""

        return self.read.complete and self.export.complete

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe workflow review."""

        return {
            "workflow": self.workflow,
            "read": self.read.to_dict(),
            "export": self.export.to_dict(),
            "missing_fields": list(self.missing_fields),
            "excessive_fields": list(self.excessive_fields),
            "denied_fields": list(self.denied_fields),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class AccessReviewReport:
    """Structured, deterministic access review for one resource schema."""

    resource_fields: tuple[str, ...]
    workflows: tuple[WorkflowAccessReview, ...]
    policy_denied_fields: tuple[str, ...] = ()
    schema_version: int = ACCESS_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != (
            ACCESS_REVIEW_SCHEMA_VERSION
        ):
            raise _validation_error("unsupported access review schema version")
        object.__setattr__(
            self,
            "resource_fields",
            _field_tuple(self.resource_fields, kind="resource schema fields"),
        )
        object.__setattr__(
            self,
            "policy_denied_fields",
            _field_tuple(self.policy_denied_fields, kind="denied fields"),
        )
        if not isinstance(self.workflows, Iterable) or isinstance(
            self.workflows, (str, bytes, Mapping)
        ):
            raise _validation_error("workflows must be an iterable of workflow reviews")
        workflows = tuple(self.workflows)
        if not all(isinstance(item, WorkflowAccessReview) for item in workflows):
            raise _validation_error("workflows must contain workflow reviews")
        if len({item.workflow for item in workflows}) != len(workflows):
            raise _validation_error("workflow names must be unique")
        object.__setattr__(
            self,
            "workflows",
            tuple(sorted(workflows, key=lambda item: item.workflow)),
        )

    @property
    def missing_fields(self) -> tuple[str, ...]:
        """Return unique fields missing from at least one workflow mode."""

        return _union_fields(workflow.missing_fields for workflow in self.workflows)

    @property
    def excessive_fields(self) -> tuple[str, ...]:
        """Return unique fields excessive for at least one workflow mode."""

        return _union_fields(workflow.excessive_fields for workflow in self.workflows)

    @property
    def denied_fields(self) -> tuple[str, ...]:
        """Return unique declared fields denied by policy."""

        return _union_fields(workflow.denied_fields for workflow in self.workflows)

    @property
    def workflows_with_findings(self) -> tuple[WorkflowAccessReview, ...]:
        """Return workflows with a missing, excessive, or denied field."""

        return tuple(
            workflow
            for workflow in self.workflows
            if workflow.missing_fields
            or workflow.excessive_fields
            or workflow.denied_fields
        )

    @property
    def complete(self) -> bool:
        """Whether every workflow mode has no access review findings."""

        return all(workflow.complete for workflow in self.workflows)

    @property
    def summary(self) -> dict[str, Any]:
        """Return aggregate counts without exposing schema metadata."""

        return {
            "workflow_count": len(self.workflows),
            "workflows_with_findings": len(self.workflows_with_findings),
            "resource_field_count": len(self.resource_fields),
            "missing_field_count": len(self.missing_fields),
            "excessive_field_count": len(self.excessive_fields),
            "denied_field_count": len(self.denied_fields),
            "complete": self.complete,
        }

    def workflow(self, name: str) -> WorkflowAccessReview:
        """Return one workflow review by safe name."""

        normalized = _workflow_name(name)
        for workflow in self.workflows:
            if workflow.workflow == normalized:
                return workflow
        raise KeyError("workflow review was not found")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe report.

        Only structural field names and aggregate decisions are emitted. The
        values from a resource schema mapping are never copied here.
        """

        return {
            "schema_version": self.schema_version,
            "resource_fields": list(self.resource_fields),
            "policy_denied_fields": list(self.policy_denied_fields),
            "missing_fields": list(self.missing_fields),
            "excessive_fields": list(self.excessive_fields),
            "denied_fields": list(self.denied_fields),
            "summary": self.summary,
            "workflows": [workflow.to_dict() for workflow in self.workflows],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this report to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a deterministic Markdown access review without values."""

        summary = self.summary
        lines = [
            "# Structured Access Review",
            "",
            "> This report compares declared structured-field access with a "
            "resource schema. It contains field names and decisions only; it "
            "does not contain field values.",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|---|---:|",
            f"| Workflows | {summary['workflow_count']} |",
            f"| Resource fields | {summary['resource_field_count']} |",
            f"| Missing fields | {summary['missing_field_count']} |",
            f"| Excessive fields | {summary['excessive_field_count']} |",
            f"| Denied fields | {summary['denied_field_count']} |",
            f"| Complete | {'yes' if summary['complete'] else 'no'} |",
            "",
            "## Workflow findings",
            "",
        ]

        if not self.workflows:
            lines.append("No workflow declarations were supplied.")
            return "\n".join(lines) + "\n"

        for index, workflow in enumerate(self.workflows):
            if index:
                lines.append("")
            lines.extend(
                [
                    f"### Workflow `{workflow.workflow}`",
                    "",
                    "| Access | Missing fields | Excessive fields | Denied fields | Status |",
                    "|---|---|---|---|---|",
                ]
            )
            for mode in workflow.modes:
                lines.append(
                    "| {mode} | {missing} | {excessive} | {denied} | {status} |".format(
                        mode=mode.mode.capitalize(),
                        missing=_markdown_fields(mode.missing_fields),
                        excessive=_markdown_fields(mode.excessive_fields),
                        denied=_markdown_fields(mode.denied_fields),
                        status="pass" if mode.complete else "review",
                    )
                )

        return "\n".join(lines) + "\n"


def _union_fields(groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    fields: set[str] = set()
    for group in groups:
        fields.update(group)
    return tuple(sorted(fields))


def _markdown_fields(fields: Sequence[str]) -> str:
    if not fields:
        return "—"
    return ", ".join(f"`{field}`" for field in fields)


def _requirement_from_declaration(
    name: str,
    declaration: Any,
) -> WorkflowRequirement:
    if isinstance(declaration, WorkflowRequirement):
        if declaration.name != name:
            raise _validation_error("workflow declaration name does not match its key")
        return declaration

    if isinstance(declaration, Mapping):
        mode_keys = {READ_ACCESS, EXPORT_ACCESS, "read_fields", "export_fields"}
        if mode_keys.intersection(declaration):
            return WorkflowRequirement(
                name,
                _mode_fields(declaration, READ_ACCESS),
                _mode_fields(declaration, EXPORT_ACCESS),
            )
        # A mapping keyed by fields is treated as a read declaration. Its
        # values are deliberately ignored, just as schema metadata is.
        return WorkflowRequirement(name, declaration.keys())

    return WorkflowRequirement(name, declaration)


def _normalize_workflows(
    declarations: Mapping[str, Any] | Sequence[Any] | WorkflowRequirement,
) -> tuple[WorkflowRequirement, ...]:
    if isinstance(declarations, WorkflowRequirement):
        return (declarations,)

    if isinstance(declarations, Mapping):
        keys = set(declarations)
        if keys and keys <= {
            READ_ACCESS,
            EXPORT_ACCESS,
            "read_fields",
            "export_fields",
        }:
            declarations = {_DEFAULT_WORKFLOW_NAME: declarations}
        elif {"name", "workflow"}.intersection(keys):
            name = declarations.get("name", declarations.get("workflow", _MISSING))
            if name is _MISSING:
                raise _validation_error("workflow declaration requires a name")
            declarations = {str(name): declarations}

    if isinstance(declarations, Mapping):
        workflows = tuple(
            _requirement_from_declaration(_workflow_name(name), declaration)
            for name, declaration in declarations.items()
        )
    else:
        if isinstance(declarations, (str, bytes)):
            declarations = (declarations,)
        try:
            items = tuple(declarations)
        except TypeError as exc:
            raise _validation_error(
                "workflow requirements must be a mapping or iterable"
            ) from exc

        normalized: list[WorkflowRequirement] = []
        for index, item in enumerate(items):
            if isinstance(item, WorkflowRequirement):
                normalized.append(item)
                continue
            if isinstance(item, Mapping):
                name = item.get("name", item.get("workflow", _MISSING))
                if name is _MISSING:
                    raise _validation_error(
                        f"workflow declaration at position {index} requires a name"
                    )
                normalized.append(
                    _requirement_from_declaration(_workflow_name(name), item)
                )
                continue
            raise _validation_error(
                f"workflow declaration at position {index} is unsupported"
            )
        workflows = tuple(normalized)

    names = [item.name for item in workflows]
    if len(set(names)) != len(names):
        raise _validation_error("workflow names must be unique")
    return tuple(sorted(workflows, key=lambda item: item.name))


def _normalize_denied_fields(
    denied_fields: FieldCollection | Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Return global and mode-specific deny sets."""

    if not isinstance(denied_fields, Mapping):
        return _field_tuple(denied_fields, kind="denied fields"), {}

    mode_keys = {READ_ACCESS, EXPORT_ACCESS, "all"}
    if not mode_keys.intersection(denied_fields):
        return _field_tuple(denied_fields, kind="denied fields"), {}

    global_fields = _field_tuple(denied_fields.get("all"), kind="denied fields")
    by_mode = {
        mode: _field_tuple(denied_fields.get(mode), kind=f"{mode} denied fields")
        for mode in ACCESS_MODES
        if mode in denied_fields
    }
    return global_fields, by_mode


def review_structured_access(
    workflow_requirements: Mapping[str, Any] | Sequence[Any] | WorkflowRequirement,
    resource_schema: ResourceSchema,
    *,
    denied_fields: FieldCollection | Mapping[str, Any] = (),
) -> AccessReviewReport:
    """Compare workflow field declarations with a local resource schema.

    Args:
        workflow_requirements: Mapping of workflow name to either a field
            collection (read access) or a mapping containing ``read`` and
            ``export`` collections. A sequence of :class:`WorkflowRequirement`
            objects is also accepted. A single mapping containing only
            ``read``/``export`` is treated as a workflow named ``default``.
        resource_schema: Field names, a mapping keyed by field names, a JSON
            Schema-like mapping with ``properties``, or an object exposing
            ``fields``, ``columns``, or ``names``. Mapping values are ignored.
        denied_fields: Globally denied field names, or a mapping with optional
            ``all``, ``read``, and ``export`` collections.

    Returns:
        An immutable :class:`AccessReviewReport` containing only structural
        field names, counts, and access decisions.

    The comparison is set-based and sorted, so equivalent inputs produce
    byte-for-byte identical JSON and Markdown. No network or model access is
    performed.
    """

    resource_fields = _resource_fields(resource_schema)
    global_denied, mode_denied = _normalize_denied_fields(denied_fields)
    normalized_workflows = _normalize_workflows(workflow_requirements)
    resource_set = set(resource_fields)
    global_denied_set = set(global_denied)

    reviews: list[WorkflowAccessReview] = []
    for requirement in normalized_workflows:
        mode_reviews: dict[str, AccessModeReview] = {}
        for mode in ACCESS_MODES:
            requested = requirement.fields_for(mode)
            requested_set = set(requested)
            denied = global_denied_set | set(mode_denied.get(mode, ()))
            denied_requested = requested_set & denied
            mode_reviews[mode] = AccessModeReview(
                mode=mode,
                requested_fields=requested,
                available_fields=resource_fields,
                allowed_fields=tuple(sorted((requested_set & resource_set) - denied)),
                missing_fields=tuple(sorted(requested_set - resource_set)),
                excessive_fields=tuple(sorted(resource_set - requested_set)),
                denied_fields=tuple(sorted(denied_requested)),
            )
        reviews.append(
            WorkflowAccessReview(
                workflow=requirement.name,
                read=mode_reviews[READ_ACCESS],
                export=mode_reviews[EXPORT_ACCESS],
            )
        )

    return AccessReviewReport(
        resource_fields=resource_fields,
        workflows=tuple(reviews),
        policy_denied_fields=global_denied + _union_fields(mode_denied.values()),
    )


def render_access_review(report: AccessReviewReport) -> str:
    """Render an :class:`AccessReviewReport` as deterministic Markdown."""

    if not isinstance(report, AccessReviewReport):
        raise TypeError("report must be an AccessReviewReport")
    return report.to_markdown()


# Report-oriented aliases keep the API discoverable alongside the other risk
# report builders while preserving one implementation and one output contract.
build_access_review_report = review_structured_access
access_review_report = review_structured_access
review_access = review_structured_access
