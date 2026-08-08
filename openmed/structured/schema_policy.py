"""Declarative field-level de-identification for FHIR and OMOP data.

Schema policies bind canonical field paths to one of five privacy actions:
``suppress``, ``generalize``, ``date-shift``, ``route-to-deidentify``, or
``keep``.  The runtime is deliberately local and schema-aware.  Policy lint
results contain field paths and action metadata only, never source values.

FHIR paths use resource-prefixed dotted notation and ``[]`` for repeated
elements (for example ``Patient.name[].family``).  OMOP paths use the table
name as the prefix (for example ``visit_occurrence.visit_start_date``).
Wildcard rules are supported for schema families such as
``*.*_source_value``.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from fnmatch import fnmatchcase
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Final

from openmed.core.date_shift import DEFAULT_DATE_SHIFT_MAX_DAYS, stable_offset_for
from openmed.structured.hierarchies import (
    SUPPORTED_COLUMN_TYPES,
    HierarchyError,
    generalize_value,
    max_level,
)
from openmed.structured.table_io import read_table, write_table

ACTION_SUPPRESS: Final = "suppress"
ACTION_GENERALIZE: Final = "generalize"
ACTION_DATE_SHIFT: Final = "date-shift"
ACTION_DEIDENTIFY: Final = "route-to-deidentify"
ACTION_KEEP: Final = "keep"

SUPPORTED_SCHEMA_ACTIONS: Final = frozenset(
    {
        ACTION_SUPPRESS,
        ACTION_GENERALIZE,
        ACTION_DATE_SHIFT,
        ACTION_DEIDENTIFY,
        ACTION_KEEP,
    }
)
SUPPORTED_SCHEMA_TYPES: Final = frozenset({"fhir", "omop"})
SCHEMA_POLICY_VERSION: Final = 1

_BUNDLED_POLICIES: Final = (
    "fhir_hipaa_safe_harbor",
    "fhir_research_limited_dataset",
    "omop_hipaa_safe_harbor",
    "omop_research_limited_dataset",
)
_POLICY_ALIASES: Final = {
    "fhir_safe_harbor": "fhir_hipaa_safe_harbor",
    "omop_safe_harbor": "omop_hipaa_safe_harbor",
    "fhir_limited_dataset": "fhir_research_limited_dataset",
    "omop_limited_dataset": "omop_research_limited_dataset",
}
_ACTION_ALIASES: Final = {
    "date_shift": ACTION_DATE_SHIFT,
    "route_to_deidentify": ACTION_DEIDENTIFY,
    "deidentify": ACTION_DEIDENTIFY,
}
_FIELD_TYPES: Final = frozenset(
    {
        "date",
        "free-text",
        "identifier",
        "internal-linkage",
        "quasi-identifier",
        "safe",
        "sensitive",
    }
)
_FHIR_SUBJECT_KEYS: Final = ("subject", "patient", "beneficiary", "individual")
_IDENTIFIER_SEGMENTS: Final = frozenset(
    {
        "address",
        "alias",
        "contact",
        "email",
        "emails",
        "family",
        "given",
        "identifier",
        "identifiers",
        "line",
        "mrn",
        "name",
        "names",
        "phone",
        "phones",
        "reference",
        "ssn",
        "telecom",
    }
)
_REMOVE: Final = object()
_ISO_DATE_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(.*)$", re.DOTALL)

TextDeidentifier = Callable[..., Any]


class SchemaPolicyError(ValueError):
    """Raised when a schema policy or schema-policy application is invalid."""


@dataclass(frozen=True)
class FieldRule:
    """One normalized field-path rule in a schema policy."""

    path: str
    action: str
    field_type: str | None = None
    generalization: str | None = None
    level: int = 0


@dataclass(frozen=True)
class SchemaPolicyLintFinding:
    """PHI-safe schema-policy lint finding."""

    code: str
    path: str
    message: str
    severity: str = "warning"

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-ready finding without any field value."""

        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class SchemaPolicy:
    """Validated declarative policy for one structured-data schema."""

    name: str
    schema: str
    base_policy: str
    default_action: str
    rules: tuple[FieldRule, ...]
    identifier_fields: tuple[str, ...] = ()
    known_fields: tuple[str, ...] = ()
    schema_version: int = SCHEMA_POLICY_VERSION

    def rule_for(self, path: str) -> FieldRule | None:
        """Return the most-specific rule matching ``path``."""

        normalized = _normalize_path(path)
        candidates = [
            rule for rule in self.rules if _path_matches(rule.path, normalized)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda rule: _path_specificity(rule.path))

    def covering_rule_for(self, path: str) -> FieldRule | None:
        """Return a rule matching ``path`` or one of its parent containers."""

        for candidate in _path_ancestors(path):
            rule = self.rule_for(candidate)
            if rule is not None:
                return rule
        return None

    def is_identifier_path(self, path: str) -> bool:
        """Return whether policy metadata or conservative inference marks a path."""

        normalized = _normalize_path(path)
        rule = self.covering_rule_for(normalized)
        if rule is not None and rule.field_type in {
            "identifier",
            "internal-linkage",
        }:
            return True
        if any(
            _path_matches(pattern, normalized) for pattern in self.identifier_fields
        ):
            return True
        return _looks_like_identifier_path(normalized)


PolicyInput = SchemaPolicy | Mapping[str, Any] | str | Path


def list_schema_policies() -> tuple[str, ...]:
    """Return the bundled schema-policy names.

    Returns:
        Canonical names accepted by :func:`load_schema_policy`.
    """

    return _BUNDLED_POLICIES


def load_schema_policy(policy: PolicyInput) -> SchemaPolicy:
    """Load and validate a bundled, file-backed, or in-memory schema policy.

    Args:
        policy: A :class:`SchemaPolicy`, JSON-compatible mapping, bundled policy
            name, or path to a JSON policy document.

    Returns:
        A validated immutable :class:`SchemaPolicy`.

    Raises:
        SchemaPolicyError: If the document is malformed or unsupported.
    """

    if isinstance(policy, SchemaPolicy):
        return policy
    if isinstance(policy, Mapping):
        return _policy_from_mapping(policy, source="<mapping>")
    if isinstance(policy, Path):
        return _load_policy_path(policy)
    if not isinstance(policy, str) or not policy.strip():
        raise SchemaPolicyError(
            "schema policy must be a policy name, path, mapping, or SchemaPolicy"
        )

    value = policy.strip()
    candidate = Path(value)
    if candidate.suffix.lower() == ".json" or candidate.is_absolute():
        return _load_policy_path(candidate)
    canonical = value.lower().replace("-", "_")
    canonical = _POLICY_ALIASES.get(canonical, canonical)
    if canonical not in _BUNDLED_POLICIES:
        expected = ", ".join(_BUNDLED_POLICIES)
        raise SchemaPolicyError(
            f"unknown schema policy {policy!r}; expected one of: {expected}"
        )
    return _load_bundled_policy(canonical)


def validate_schema_policy(
    policy: PolicyInput,
    schema_fields: Iterable[str] | Mapping[str, Iterable[str]] | None = None,
) -> tuple[SchemaPolicyLintFinding, ...]:
    """Validate policy paths against a supplied or declared resource schema.

    Structural policy errors raise :class:`SchemaPolicyError`.  Path-level
    issues are returned as PHI-safe findings so callers can lint optional
    fields without exceptions.

    Args:
        policy: Bundled name, JSON path, mapping, or loaded schema policy.
        schema_fields: Optional canonical paths from an external resource
            schema, either flat or grouped by resource/table name. When
            omitted, the policy document's declared fields are checked.

    Returns:
        Unknown-policy-field and uncovered-schema-field findings.

    Raises:
        SchemaPolicyError: If the policy document itself is invalid.
    """

    loaded = load_schema_policy(policy)
    declared = _declared_schema_paths(
        schema_fields if schema_fields is not None else loaded.known_fields
    )
    if not declared:
        return ()

    findings: list[SchemaPolicyLintFinding] = []
    for rule in loaded.rules:
        if not any(_patterns_overlap(rule.path, field) for field in declared):
            findings.append(
                SchemaPolicyLintFinding(
                    code="unknown-policy-field",
                    path=rule.path,
                    message="policy field is not present in the declared schema",
                )
            )
    for field in declared:
        if loaded.covering_rule_for(field) is None:
            findings.append(
                SchemaPolicyLintFinding(
                    code="uncovered-schema-field",
                    path=field,
                    message="declared schema field has no explicit policy action",
                    severity=(
                        "error" if loaded.is_identifier_path(field) else "warning"
                    ),
                )
            )
    return tuple(_deduplicate_findings(findings))


def lint_schema_policy(
    data: Any,
    policy: PolicyInput,
    *,
    schema: str | None = None,
    table_name: str | None = None,
) -> tuple[SchemaPolicyLintFinding, ...]:
    """Report observed fields that have no explicit policy action.

    Findings contain paths only.  Uncovered identifier-typed paths are errors
    because application suppresses them by default; other uncovered paths are
    warnings and follow the policy's ``default_action``.

    Args:
        data: FHIR resource/Bundle or OMOP table input to inspect.
        policy: Bundled name, JSON path, mapping, or loaded schema policy.
        schema: Optional explicit ``"fhir"`` or ``"omop"`` assertion.
        table_name: OMOP table name when it cannot be inferred from the input.

    Returns:
        PHI-safe findings ordered by canonical field path.

    Raises:
        SchemaPolicyError: If the input shape or policy is invalid.
    """

    loaded = load_schema_policy(policy)
    _validate_schema_override(loaded, schema)
    observed = _observed_paths(data, loaded, table_name=table_name)
    findings: list[SchemaPolicyLintFinding] = []
    for path in sorted(observed):
        if loaded.covering_rule_for(path) is not None:
            continue
        identifier = loaded.is_identifier_path(path)
        findings.append(
            SchemaPolicyLintFinding(
                code="uncovered-identifier" if identifier else "uncovered-field",
                path=path,
                message=(
                    "identifier-typed field is uncovered and defaults to suppression"
                    if identifier
                    else (
                        "field has no explicit policy action and uses default action "
                        f"{loaded.default_action!r}"
                    )
                ),
                severity="error" if identifier else "warning",
            )
        )
    return tuple(findings)


def apply_schema_policy(
    data: Any,
    policy: PolicyInput,
    *,
    schema: str | None = None,
    table_name: str | None = None,
    subject_key: str | bytes | None = None,
    date_shift_secret: str | bytes | None = None,
    date_shift_max_days: int = DEFAULT_DATE_SHIFT_MAX_DAYS,
    deidentifier: TextDeidentifier | None = None,
    text_policy: str | None = None,
    text_method: str = "replace",
) -> Any:
    """Apply a declarative policy to FHIR resources or OMOP tables.

    The returned object has the same high-level shape as ``data``.  Suppressed
    mapping fields and table columns are removed.  Uncovered identifier-typed
    fields are also removed, while other uncovered fields follow the policy's
    default action.  Input objects are never mutated.

    A date-shift rule requires both stable subject identity (inferred from FHIR
    references or OMOP ``person_id`` columns unless ``subject_key`` is given)
    and ``date_shift_secret``.  Neither is retained in the output or lint data.

    Args:
        data: FHIR resource/Bundle, OMOP row sequence/table mapping, local table
            path, or DataFrame-like object.
        policy: Bundled name, JSON path, mapping, or loaded schema policy.
        schema: Optional explicit ``"fhir"`` or ``"omop"`` assertion.
        table_name: OMOP table name when it cannot be inferred from the input.
        subject_key: Optional stable subject identity override.
        date_shift_secret: HMAC key material required by date-shift rules.
        date_shift_max_days: Maximum absolute patient-keyed date shift.
        deidentifier: Optional offline free-text pipeline override.
        text_policy: Optional core text-policy override. The schema policy's
            ``base_policy`` is used by default.
        text_method: Core free-text de-identification method.

    Returns:
        A deep transformed copy preserving the input's high-level shape.

    Raises:
        SchemaPolicyError: If policy validation or a field transform fails.
    """

    loaded = load_schema_policy(policy)
    _validate_schema_override(loaded, schema)
    context = _ApplicationContext(
        policy=loaded,
        subject_key=subject_key,
        date_shift_secret=date_shift_secret,
        date_shift_max_days=date_shift_max_days,
        deidentifier=deidentifier,
        text_policy=text_policy or loaded.base_policy,
        text_method=text_method,
    )
    if loaded.schema == "fhir":
        return _apply_fhir(data, context)
    return _apply_omop(data, context, table_name=table_name)


def apply_omop_file(
    in_path: str | Path,
    out_path: str | Path,
    policy: PolicyInput,
    *,
    table_name: str | None = None,
    subject_key: str | bytes | None = None,
    date_shift_secret: str | bytes | None = None,
    date_shift_max_days: int = DEFAULT_DATE_SHIFT_MAX_DAYS,
    deidentifier: TextDeidentifier | None = None,
    text_policy: str | None = None,
    text_method: str = "replace",
    overwrite: bool = False,
) -> Path:
    """Apply an OMOP schema policy to a local CSV or Parquet table file.

    Args:
        in_path: Source table path accepted by structured table I/O.
        out_path: Destination path using the desired CSV or Parquet suffix.
        policy: Bundled name, JSON path, mapping, or loaded OMOP policy.
        table_name: Explicit OMOP table name; defaults to the source stem.
        subject_key: Optional stable subject identity override.
        date_shift_secret: HMAC key material required by date-shift rules.
        date_shift_max_days: Maximum absolute patient-keyed date shift.
        deidentifier: Optional offline free-text pipeline override.
        text_policy: Optional core text-policy override.
        text_method: Core free-text de-identification method.
        overwrite: Whether an existing destination may be atomically replaced.

    Returns:
        The written destination path.

    Raises:
        SchemaPolicyError: If policy validation or a field transform fails.
    """

    source = Path(in_path)
    rows = read_table(source)
    transformed = apply_schema_policy(
        rows,
        policy,
        schema="omop",
        table_name=table_name or source.stem,
        subject_key=subject_key,
        date_shift_secret=date_shift_secret,
        date_shift_max_days=date_shift_max_days,
        deidentifier=deidentifier,
        text_policy=text_policy,
        text_method=text_method,
    )
    return write_table(out_path, transformed, overwrite=overwrite)


@dataclass(frozen=True)
class _ApplicationContext:
    policy: SchemaPolicy
    subject_key: str | bytes | None
    date_shift_secret: str | bytes | None
    date_shift_max_days: int
    deidentifier: TextDeidentifier | None
    text_policy: str
    text_method: str


def _apply_fhir(data: Any, context: _ApplicationContext) -> Any:
    if isinstance(data, Mapping):
        resource_type = data.get("resourceType")
        if resource_type == "Bundle":
            transformed = copy.deepcopy(dict(data))
            entries = transformed.get("entry")
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    resource = entry.get("resource")
                    if isinstance(resource, Mapping):
                        entry["resource"] = _apply_fhir_resource(resource, context)
            return transformed
        return _apply_fhir_resource(data, context)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        if not all(isinstance(resource, Mapping) for resource in data):
            raise SchemaPolicyError("every FHIR resource must be a mapping")
        return [_apply_fhir_resource(resource, context) for resource in data]
    raise SchemaPolicyError("FHIR policy input must be a resource, Bundle, or sequence")


def _apply_fhir_resource(
    resource: Mapping[str, Any],
    context: _ApplicationContext,
) -> dict[str, Any]:
    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type:
        raise SchemaPolicyError("FHIR resource is missing resourceType")
    subject_key = context.subject_key or _fhir_subject_key(resource)
    transformed = _transform_node(
        dict(resource),
        path=resource_type,
        context=context,
        row_subject_key=subject_key,
        root=True,
    )
    if not isinstance(transformed, dict):  # pragma: no cover - root is retained
        raise SchemaPolicyError("FHIR resource root cannot be suppressed")
    return transformed


def _apply_omop(
    data: Any,
    context: _ApplicationContext,
    *,
    table_name: str | None,
) -> Any:
    if isinstance(data, (str, Path)):
        source = Path(data)
        data = read_table(source)
        table_name = table_name or source.stem

    if isinstance(data, Mapping) and _looks_like_table_collection(data):
        return {
            str(name): _apply_omop_rows(
                rows,
                context,
                table_name=_normalize_table_name(str(name)),
            )
            for name, rows in data.items()
        }

    rows = _coerce_rows(data)
    resolved_table = _resolve_table_name(rows, context.policy, table_name)
    return _apply_omop_rows(rows, context, table_name=resolved_table)


def _apply_omop_rows(
    rows: Any,
    context: _ApplicationContext,
    *,
    table_name: str,
) -> list[dict[str, Any]]:
    materialized = _coerce_rows(rows)
    transformed_rows: list[dict[str, Any]] = []
    for row in materialized:
        row_subject_key = context.subject_key or _omop_subject_key(row)
        transformed = _transform_node(
            row,
            path=table_name,
            context=context,
            row_subject_key=row_subject_key,
            root=True,
        )
        if not isinstance(transformed, dict):  # pragma: no cover - root retained
            raise SchemaPolicyError("OMOP row root cannot be suppressed")
        transformed_rows.append(transformed)
    return transformed_rows


def _transform_node(
    value: Any,
    *,
    path: str,
    context: _ApplicationContext,
    row_subject_key: str | bytes | None,
    root: bool = False,
) -> Any:
    rule = context.policy.rule_for(path)
    if rule is not None and not root:
        return _apply_rule(value, rule, path, context, row_subject_key)

    if isinstance(value, Mapping):
        transformed: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            output = _transform_node(
                child,
                path=child_path,
                context=context,
                row_subject_key=row_subject_key,
            )
            if output is not _REMOVE:
                transformed[str(key)] = output
        return transformed
    if isinstance(value, list):
        item_path = f"{path}[]"
        items = [
            _transform_node(
                child,
                path=item_path,
                context=context,
                row_subject_key=row_subject_key,
            )
            for child in value
        ]
        return [item for item in items if item is not _REMOVE]
    if isinstance(value, tuple):
        item_path = f"{path}[]"
        items = [
            _transform_node(
                child,
                path=item_path,
                context=context,
                row_subject_key=row_subject_key,
            )
            for child in value
        ]
        return tuple(item for item in items if item is not _REMOVE)

    if context.policy.is_identifier_path(path):
        return _REMOVE
    default_rule = FieldRule(path=path, action=context.policy.default_action)
    return _apply_rule(value, default_rule, path, context, row_subject_key)


def _apply_rule(
    value: Any,
    rule: FieldRule,
    path: str,
    context: _ApplicationContext,
    row_subject_key: str | bytes | None,
) -> Any:
    if rule.action == ACTION_SUPPRESS:
        return _REMOVE
    if rule.action == ACTION_KEEP or value is None:
        return copy.deepcopy(value)
    if rule.action == ACTION_GENERALIZE:
        if isinstance(value, (Mapping, list, tuple)):
            raise SchemaPolicyError(f"generalize action requires a scalar at {path}")
        try:
            return generalize_value(
                str(rule.generalization),
                value,
                rule.level,
            )
        except (HierarchyError, TypeError, ValueError):
            raise SchemaPolicyError(f"generalization failed at {path}") from None
    if rule.action == ACTION_DATE_SHIFT:
        if isinstance(value, (Mapping, list, tuple)):
            raise SchemaPolicyError(f"date-shift action requires a scalar at {path}")
        if row_subject_key is None:
            raise SchemaPolicyError(f"date-shift field lacks a subject key at {path}")
        if context.date_shift_secret is None:
            raise SchemaPolicyError(
                f"date-shift field requires date_shift_secret at {path}"
            )
        return _shift_temporal(
            value,
            subject_key=row_subject_key,
            secret=context.date_shift_secret,
            max_days=context.date_shift_max_days,
            path=path,
        )
    if rule.action == ACTION_DEIDENTIFY:
        return _deidentify_value(value, path=path, context=context)
    raise SchemaPolicyError(f"unsupported action {rule.action!r} at {path}")


def _deidentify_value(value: Any, *, path: str, context: _ApplicationContext) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _deidentify_value(child, path=path, context=context)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_deidentify_value(child, path=path, context=context) for child in value]
    if isinstance(value, tuple):
        return tuple(
            _deidentify_value(child, path=path, context=context) for child in value
        )
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaPolicyError(f"route-to-deidentify requires text at {path}")
    deidentifier = context.deidentifier or _default_deidentifier()
    try:
        result = deidentifier(
            value,
            method=context.text_method,
            policy=context.text_policy,
        )
    except Exception:
        raise SchemaPolicyError(f"de-identification failed at {path}") from None
    output = (
        result
        if isinstance(result, str)
        else getattr(result, "deidentified_text", None)
    )
    if not isinstance(output, str):
        raise SchemaPolicyError(
            f"deidentifier returned an invalid result for field at {path}"
        )
    return output


@lru_cache(maxsize=1)
def _default_deidentifier() -> TextDeidentifier:
    from openmed.core.pii import deidentify

    return deidentify


def _shift_temporal(
    value: Any,
    *,
    subject_key: str | bytes,
    secret: str | bytes,
    max_days: int,
    path: str,
) -> Any:
    try:
        offset = stable_offset_for(subject_key, max_days=max_days, secret=secret)
    except (TypeError, ValueError):
        raise SchemaPolicyError(f"invalid date-shift configuration at {path}") from None

    if isinstance(value, datetime):
        return value + timedelta(days=offset)
    if isinstance(value, date):
        return value + timedelta(days=offset)
    if not isinstance(value, str):
        raise SchemaPolicyError(f"date-shift requires an ISO date at {path}")
    match = _ISO_DATE_PREFIX.fullmatch(value.strip())
    if match is None:
        raise SchemaPolicyError(f"date-shift requires an ISO date at {path}")
    try:
        parsed = date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        raise SchemaPolicyError(f"date-shift requires an ISO date at {path}") from None
    suffix = match[4]
    if suffix:
        candidate = f"{parsed.isoformat()}{suffix}"
        try:
            datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            raise SchemaPolicyError(
                f"date-shift requires an ISO date at {path}"
            ) from None
    return f"{(parsed + timedelta(days=offset)).isoformat()}{suffix}"


def _fhir_subject_key(resource: Mapping[str, Any]) -> str | bytes | None:
    if resource.get("resourceType") == "Patient":
        return _subject_scalar(resource.get("id"))
    for key in _FHIR_SUBJECT_KEYS:
        reference = resource.get(key)
        if not isinstance(reference, Mapping):
            continue
        value = reference.get("reference")
        if isinstance(value, str) and value:
            marker = "/Patient/"
            if marker in value:
                return value.rsplit(marker, maxsplit=1)[-1]
            if value.startswith("Patient/"):
                return value.split("/", maxsplit=1)[-1]
            return value
    return None


def _omop_subject_key(row: Mapping[str, Any]) -> str | bytes | None:
    for name in ("person_id", "subject_id", "patient_id"):
        if name in row:
            return _subject_scalar(row[name])
    return None


def _subject_scalar(value: Any) -> str | bytes | None:
    if isinstance(value, (str, bytes)) and value:
        return value
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _observed_paths(
    data: Any,
    policy: SchemaPolicy,
    *,
    table_name: str | None,
) -> set[str]:
    if policy.schema == "fhir":
        resources_to_scan: list[Mapping[str, Any]] = []
        if isinstance(data, Mapping) and data.get("resourceType") == "Bundle":
            for entry in data.get("entry") or []:
                if isinstance(entry, Mapping) and isinstance(
                    entry.get("resource"), Mapping
                ):
                    resources_to_scan.append(entry["resource"])
        elif isinstance(data, Mapping):
            resources_to_scan.append(data)
        elif isinstance(data, Sequence) and not isinstance(
            data, (str, bytes, bytearray)
        ):
            resources_to_scan.extend(
                resource for resource in data if isinstance(resource, Mapping)
            )
        else:
            raise SchemaPolicyError(
                "FHIR policy input must be a resource, Bundle, or sequence"
            )
        observed: set[str] = set()
        for resource in resources_to_scan:
            resource_type = resource.get("resourceType")
            if not isinstance(resource_type, str) or not resource_type:
                raise SchemaPolicyError("FHIR resource is missing resourceType")
            observed.update(_leaf_paths(resource, root=resource_type))
        return observed

    if isinstance(data, (str, Path)):
        source = Path(data)
        data = read_table(source)
        table_name = table_name or source.stem
    if isinstance(data, Mapping) and _looks_like_table_collection(data):
        observed = set()
        for name, rows in data.items():
            observed.update(
                _row_paths(rows, table_name=_normalize_table_name(str(name)))
            )
        return observed
    rows = _coerce_rows(data)
    resolved_table = _resolve_table_name(rows, policy, table_name)
    return _row_paths(rows, table_name=resolved_table)


def _leaf_paths(value: Any, *, root: str) -> set[str]:
    paths: set[str] = set()

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            if not node:
                paths.add(path)
            for key, child in node.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(node, (list, tuple)):
            if not node:
                paths.add(f"{path}[]")
            for child in node:
                visit(child, f"{path}[]")
            return
        paths.add(path)

    visit(value, root)
    return paths


def _row_paths(rows: Any, *, table_name: str) -> set[str]:
    paths: set[str] = set()
    for row in _coerce_rows(rows):
        paths.update(f"{table_name}.{column}" for column in row)
    return paths


def _looks_like_table_collection(data: Mapping[Any, Any]) -> bool:
    if not data:
        return False
    return all(
        isinstance(name, str)
        and isinstance(rows, Sequence)
        and not isinstance(rows, (str, bytes, bytearray))
        for name, rows in data.items()
    )


def _coerce_rows(data: Any) -> list[dict[str, Any]]:
    to_dicts = getattr(data, "to_dicts", None)
    if callable(to_dicts):
        rows = to_dicts()
    else:
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict) and not isinstance(data, Mapping):
            rows = to_dict("records")
        elif isinstance(data, Mapping):
            rows = [data]
        elif isinstance(data, Sequence) and not isinstance(
            data, (str, bytes, bytearray)
        ):
            rows = data
        else:
            raise SchemaPolicyError(
                "OMOP policy input must be a table path, row sequence, frame, "
                "or table-name mapping"
            )
    if not all(isinstance(row, Mapping) for row in rows):
        raise SchemaPolicyError("every OMOP row must be a mapping")
    return [dict(row) for row in rows]


def _resolve_table_name(
    rows: Sequence[Mapping[str, Any]],
    policy: SchemaPolicy,
    table_name: str | None,
) -> str:
    if table_name:
        return _normalize_table_name(table_name)
    if not rows:
        raise SchemaPolicyError("table_name is required for an empty OMOP table")
    columns = set(rows[0])
    prefixes = sorted({rule.path.split(".", maxsplit=1)[0] for rule in policy.rules})
    scores: list[tuple[int, str]] = []
    for prefix in prefixes:
        policy_columns = {
            rule.path.split(".", maxsplit=1)[1]
            for rule in policy.rules
            if rule.path.startswith(f"{prefix}.") and "*" not in rule.path
        }
        scores.append((len(columns & policy_columns), prefix))
    scores.sort(reverse=True)
    if not scores or scores[0][0] == 0:
        raise SchemaPolicyError("table_name is required for this OMOP schema")
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        raise SchemaPolicyError("table_name is ambiguous for this OMOP schema")
    return scores[0][1]


def _normalize_table_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise SchemaPolicyError("table_name must not be blank")
    return normalized


def _validate_schema_override(policy: SchemaPolicy, schema: str | None) -> None:
    if schema is None:
        return
    normalized = schema.strip().lower()
    if normalized != policy.schema:
        raise SchemaPolicyError(
            f"schema override {schema!r} does not match policy schema {policy.schema!r}"
        )


@lru_cache(maxsize=len(_BUNDLED_POLICIES))
def _load_bundled_policy(name: str) -> SchemaPolicy:
    resource = resources.files("openmed.core").joinpath("policies", f"{name}.json")
    try:
        with resource.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:  # pragma: no cover - packaging failure
        raise SchemaPolicyError(
            f"could not load bundled schema policy {name!r}"
        ) from exc
    return _policy_from_mapping(payload, source=f"{name}.json")


def _load_policy_path(path: Path) -> SchemaPolicy:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise SchemaPolicyError(f"could not read schema policy: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaPolicyError(f"schema policy is not valid JSON: {path}") from exc
    return _policy_from_mapping(payload, source=str(path))


def _policy_from_mapping(payload: Mapping[str, Any], *, source: str) -> SchemaPolicy:
    version = payload.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_POLICY_VERSION:
        raise SchemaPolicyError(
            f"schema policy {source} must use schema_version {SCHEMA_POLICY_VERSION}"
        )
    name = _nonempty_string(payload.get("name"), "name")
    schema = _nonempty_string(payload.get("schema"), "schema").lower()
    if schema not in SUPPORTED_SCHEMA_TYPES:
        raise SchemaPolicyError(
            f"schema must be one of {sorted(SUPPORTED_SCHEMA_TYPES)!r}"
        )
    base_policy = _nonempty_string(
        payload.get("base_policy", "hipaa_safe_harbor"),
        "base_policy",
    )
    default_action = _normalize_action(payload.get("default_action", ACTION_KEEP))

    rule_payloads: list[tuple[str, Any]] = []
    fields = payload.get("fields") or {}
    if not isinstance(fields, Mapping):
        raise SchemaPolicyError("fields must be an object")
    rule_payloads.extend((str(path), rule) for path, rule in fields.items())

    container_key = "resources" if schema == "fhir" else "tables"
    containers = payload.get(container_key) or {}
    if not isinstance(containers, Mapping):
        raise SchemaPolicyError(f"{container_key} must be an object")
    container_identifiers: list[str] = []
    container_known: list[str] = []
    for container_name, container_payload in containers.items():
        prefix = _nonempty_string(container_name, f"{container_key} name")
        if not isinstance(container_payload, Mapping):
            raise SchemaPolicyError(f"{container_key}.{prefix} must be an object")
        nested_fields = container_payload.get("fields", container_payload)
        if not isinstance(nested_fields, Mapping):
            raise SchemaPolicyError(
                f"{container_key}.{prefix}.fields must be an object"
            )
        metadata_keys = {"fields", "identifier_fields", "known_fields"}
        for path, rule in nested_fields.items():
            if path in metadata_keys and nested_fields is container_payload:
                continue
            rule_payloads.append((f"{prefix}.{path}", rule))
        container_identifiers.extend(
            f"{prefix}.{path}"
            for path in _string_sequence(
                container_payload.get("identifier_fields") or (),
                f"{container_key}.{prefix}.identifier_fields",
            )
        )
        container_known.extend(
            f"{prefix}.{path}"
            for path in _string_sequence(
                container_payload.get("known_fields") or (),
                f"{container_key}.{prefix}.known_fields",
            )
        )

    rules: list[FieldRule] = []
    seen_paths: set[str] = set()
    for raw_path, raw_rule in rule_payloads:
        path = _normalize_path(raw_path)
        if path in seen_paths:
            raise SchemaPolicyError(
                f"schema policy defines field {path!r} more than once"
            )
        seen_paths.add(path)
        rules.append(_field_rule(path, raw_rule))
    if not rules:
        raise SchemaPolicyError("schema policy must define at least one field rule")

    identifier_fields = tuple(
        _normalize_path(path)
        for path in (
            *_string_sequence(
                payload.get("identifier_fields") or (), "identifier_fields"
            ),
            *container_identifiers,
        )
    )
    explicit_known = (
        *_string_sequence(payload.get("known_fields") or (), "known_fields"),
        *container_known,
    )
    known_fields = tuple(_normalize_path(path) for path in explicit_known)
    if not known_fields:
        known_fields = tuple(rule.path for rule in rules)

    return SchemaPolicy(
        name=name,
        schema=schema,
        base_policy=base_policy,
        default_action=default_action,
        rules=tuple(rules),
        identifier_fields=identifier_fields,
        known_fields=known_fields,
        schema_version=version,
    )


def _field_rule(path: str, payload: Any) -> FieldRule:
    if isinstance(payload, str):
        return FieldRule(path=path, action=_normalize_action(payload))
    if not isinstance(payload, Mapping):
        raise SchemaPolicyError(f"field rule {path!r} must be a string or object")
    action = _normalize_action(payload.get("action"))
    field_type_value = payload.get("field_type", payload.get("type"))
    field_type = None
    if field_type_value is not None:
        field_type = _nonempty_string(field_type_value, f"{path}.field_type")
        field_type = field_type.lower().replace("_", "-")
        if field_type not in _FIELD_TYPES:
            raise SchemaPolicyError(
                f"field rule {path!r} has unsupported field_type {field_type!r}"
            )

    generalization_value = payload.get("generalization", payload.get("column_type"))
    generalization = None
    if generalization_value is not None:
        generalization = _nonempty_string(
            generalization_value,
            f"{path}.generalization",
        )
    level = payload.get("level", 0)
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise SchemaPolicyError(
            f"field rule {path!r} level must be a non-negative integer"
        )
    if action == ACTION_GENERALIZE:
        if generalization not in SUPPORTED_COLUMN_TYPES:
            raise SchemaPolicyError(
                f"field rule {path!r} generalization must be one of "
                f"{sorted(SUPPORTED_COLUMN_TYPES)!r}"
            )
        if level > max_level(generalization):
            raise SchemaPolicyError(
                f"field rule {path!r} level exceeds the {generalization!r} maximum"
            )
    elif generalization is not None:
        raise SchemaPolicyError(
            f"field rule {path!r} sets generalization for action {action!r}"
        )
    return FieldRule(
        path=path,
        action=action,
        field_type=field_type,
        generalization=generalization,
        level=level,
    )


def _normalize_action(value: Any) -> str:
    action = _nonempty_string(value, "action").lower().replace(" ", "-")
    action = _ACTION_ALIASES.get(action, action)
    if action not in SUPPORTED_SCHEMA_ACTIONS:
        raise SchemaPolicyError(
            f"action must be one of {sorted(SUPPORTED_SCHEMA_ACTIONS)!r}"
        )
    return action


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaPolicyError(f"{name} must be a non-empty string")
    return value.strip()


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SchemaPolicyError(f"{name} must be a list of field paths")
    resolved: list[str] = []
    for item in value:
        resolved.append(_nonempty_string(item, name))
    return tuple(resolved)


def _declared_schema_paths(
    value: Iterable[str] | Mapping[str, Iterable[str]],
) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for container, fields in value.items():
            prefix = _nonempty_string(container, "schema resource/table name")
            if isinstance(fields, (str, bytes, bytearray)):
                fields = (str(fields),)
            for field in fields:
                path = _nonempty_string(field, "schema field")
                paths.append(
                    path if path.startswith(f"{prefix}.") else f"{prefix}.{path}"
                )
        return tuple(_normalize_path(path) for path in paths)
    if isinstance(value, (str, bytes, bytearray)):
        return (_normalize_path(str(value)),)
    return tuple(_normalize_path(path) for path in value)


def _normalize_path(path: str) -> str:
    if not isinstance(path, str):
        raise SchemaPolicyError("field paths must be strings")
    normalized = path.strip().replace("[*]", "[]")
    normalized = re.sub(r"\[(?:\d+)\]", "[]", normalized)
    normalized = normalized.strip(".")
    if not normalized or ".." in normalized:
        raise SchemaPolicyError(f"invalid field path {path!r}")
    return normalized


def _path_matches(pattern: str, path: str) -> bool:
    return fnmatchcase(path, pattern)


def _path_specificity(path: str) -> tuple[int, int, str]:
    wildcard_count = path.count("*")
    return (-wildcard_count, len(path), path)


def _path_ancestors(path: str) -> tuple[str, ...]:
    normalized = _normalize_path(path)
    parts = normalized.split(".")
    ancestors: list[str] = []
    for length in range(len(parts), 0, -1):
        candidate = ".".join(parts[:length])
        ancestors.append(candidate)
        if candidate.endswith("[]"):
            ancestors.append(candidate[:-2])
    return tuple(ancestors)


def _patterns_overlap(left: str, right: str) -> bool:
    return _path_matches(left, right) or _path_matches(right, left)


def _looks_like_identifier_path(path: str) -> bool:
    segments = [segment.removesuffix("[]").lower() for segment in path.split(".")]
    if any(segment in _IDENTIFIER_SEGMENTS for segment in segments[1:]):
        return True
    leaf = segments[-1]
    return (
        leaf == "id"
        or leaf.endswith("_id")
        or leaf.endswith("identifier")
        or leaf.endswith("identifier_value")
    )


def _deduplicate_findings(
    findings: Iterable[SchemaPolicyLintFinding],
) -> list[SchemaPolicyLintFinding]:
    seen: set[tuple[str, str]] = set()
    unique: list[SchemaPolicyLintFinding] = []
    for finding in findings:
        key = (finding.code, finding.path)
        if key not in seen:
            unique.append(finding)
            seen.add(key)
    return unique


__all__ = [
    "ACTION_DATE_SHIFT",
    "ACTION_DEIDENTIFY",
    "ACTION_GENERALIZE",
    "ACTION_KEEP",
    "ACTION_SUPPRESS",
    "SCHEMA_POLICY_VERSION",
    "SUPPORTED_SCHEMA_ACTIONS",
    "FieldRule",
    "SchemaPolicy",
    "SchemaPolicyError",
    "SchemaPolicyLintFinding",
    "apply_omop_file",
    "apply_schema_policy",
    "lint_schema_policy",
    "list_schema_policies",
    "load_schema_policy",
    "validate_schema_policy",
]
