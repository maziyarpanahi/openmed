"""Deterministic, privacy-safe gates for tabular schema drift.

Schema contracts are compared in memory and can be used in streaming release
checks without materializing rows or making network calls.  A
:class:`SchemaDriftReport` deliberately contains counts only: column names,
stable field identifiers, and table values never leave the matcher.

The ``field_id`` on :class:`SchemaField` is an optional, caller-controlled
logical identity.  Supplying the same stable identifier in two schema
versions lets the matcher classify a name change as a rename.  Without it, a
name change is conservatively reported as one removed and one added column.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SchemaContract",
    "SchemaDriftError",
    "SchemaDriftReport",
    "SchemaField",
    "compare_schema_drift",
    "enforce_schema_contract",
]

_REPORT_SCHEMA_VERSION = 1
_PROTECTED_ROLES = frozenset({"direct_identifier", "quasi_identifier", "sensitive"})
_SAFE_ROLES = frozenset({"non_sensitive", "excluded"})
_ROLE_ALIASES = {
    "direct": "direct_identifier",
    "direct_id": "direct_identifier",
    "direct_identifier": "direct_identifier",
    "identifier": "direct_identifier",
    "quasi": "quasi_identifier",
    "quasi_id": "quasi_identifier",
    "quasi_identifier": "quasi_identifier",
    "sensitive": "sensitive",
    "non_sensitive": "non_sensitive",
    "nonsensitive": "non_sensitive",
    "excluded": "excluded",
    "unknown": "unknown",
    "unclassified": "unknown",
}
_TYPE_ALIASES = {
    "bool": "boolean",
    "boolean": "boolean",
    "bytes": "binary",
    "binary": "binary",
    "date": "date",
    "datetime": "datetime",
    "datetime64": "datetime",
    "datetime64[ns]": "datetime",
    "double": "float64",
    "float": "float64",
    "float32": "float32",
    "float64": "float64",
    "int": "integer",
    "integer": "integer",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "json": "object",
    "list": "array",
    "number": "number",
    "object": "object",
    "str": "string",
    "string": "string",
    "text": "string",
    "time": "time",
    "timestamp": "datetime",
}
_MISSING = object()


def _normalize_version(value: Any) -> str:
    if type(value) is int:
        if value < 1:
            raise ValueError("schema contract version must be a positive integer")
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise TypeError("schema contract version must be a non-empty string or integer")
    return value.strip()


def _normalize_column_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("schema field names must be non-empty strings")
    return value


def _normalize_field_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("schema field identifiers must be non-empty strings")
    return value


def _normalize_dtype(value: Any) -> str:
    if isinstance(value, type):
        value = value.__name__
    if not isinstance(value, str) or not value.strip():
        raise TypeError("schema field types must be non-empty strings or types")
    normalized = value.strip().lower().replace(" ", "")
    return _TYPE_ALIASES.get(normalized, normalized)


def _normalize_role(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("schema field roles must be non-empty strings")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _ROLE_ALIASES.get(normalized, normalized)


def _validate_nullable(value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError("schema field nullability must be a boolean")
    return value


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


@dataclass(frozen=True, slots=True)
class SchemaField:
    """Describe one column in a versioned tabular schema contract.

    ``field_id`` is not generated from ``name``.  A caller that wants rename
    detection must provide a stable logical identifier in both schemas.
    """

    name: str
    dtype: str | type[Any]
    nullable: bool = False
    role: str = "unknown"
    field_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_column_name(self.name))
        object.__setattr__(self, "dtype", _normalize_dtype(self.dtype))
        object.__setattr__(self, "nullable", _validate_nullable(self.nullable))
        object.__setattr__(self, "role", _normalize_role(self.role))
        object.__setattr__(self, "field_id", _normalize_field_id(self.field_id))

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable contract representation of this field."""

        payload: dict[str, Any] = {
            "name": self.name,
            "type": self.dtype,
            "nullable": self.nullable,
            "role": self.role,
        }
        if self.field_id is not None:
            payload["field_id"] = self.field_id
        return payload


def _coerce_field(
    value: Any,
    *,
    offset: int,
    name_hint: str | None = None,
) -> SchemaField:
    if isinstance(value, SchemaField):
        return value
    if isinstance(value, Mapping):
        name = _first_present(value, "name", "column")
        if name is _MISSING:
            name = name_hint
        if name is _MISSING or name is None:
            raise ValueError(f"schema field at offset {offset} is missing a name")
        dtype = _first_present(value, "dtype", "type", "data_type")
        if dtype is _MISSING:
            raise ValueError(f"schema field at offset {offset} is missing a type")
        nullable = _first_present(value, "nullable", "nullability")
        if nullable is _MISSING:
            nullable = False
        role = value.get("role", "unknown")
        field_id = _first_present(value, "field_id", "column_id", "id")
        if field_id is _MISSING:
            field_id = None
        return SchemaField(
            name=name,
            dtype=dtype,
            nullable=nullable,
            role=role,
            field_id=field_id,
        )
    if name_hint is not None:
        return SchemaField(name_hint, value)
    raise TypeError(f"schema field at offset {offset} must be a mapping")


def _coerce_fields(value: Any) -> tuple[SchemaField, ...]:
    if isinstance(value, Mapping):
        fields = tuple(
            _coerce_field(item, offset=index, name_hint=name)
            for index, (name, item) in enumerate(value.items())
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        fields = tuple(
            _coerce_field(item, offset=index) for index, item in enumerate(value)
        )
    else:
        raise TypeError("schema columns must be a mapping or sequence")

    names = [field.name for field in fields]
    if len(names) != len(set(names)):
        raise ValueError("schema field names must be unique")
    field_ids = [field.field_id for field in fields if field.field_id is not None]
    if len(field_ids) != len(set(field_ids)):
        raise ValueError("schema field identifiers must be unique")
    return tuple(sorted(fields, key=_field_sort_key))


def _field_sort_key(field: SchemaField) -> tuple[int, str, str]:
    if field.field_id is None:
        return (1, "", field.name)
    return (0, field.field_id, field.name)


@dataclass(frozen=True, slots=True)
class SchemaContract:
    """Immutable, versioned contract for a tabular schema."""

    version: str | int
    columns: tuple[SchemaField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _normalize_version(self.version))
        object.__setattr__(self, "columns", _coerce_fields(self.columns))

    @property
    def fields(self) -> tuple[SchemaField, ...]:
        """Alias for ``columns`` when integrating with schema libraries."""

        return self.columns

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SchemaContract:
        """Build a contract from a JSON-compatible versioned mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("schema contract must be a mapping")
        version = _first_present(value, "version", "schema_version")
        if version is _MISSING:
            raise ValueError("schema contract must declare a version")
        columns = _first_present(value, "columns", "fields")
        if columns is _MISSING:
            raise ValueError("schema contract must declare columns")
        return cls(version, _coerce_fields(columns))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete contract for storage by the caller."""

        return {
            "version": self.version,
            "columns": [field.to_dict() for field in self.columns],
        }


def _coerce_contract(value: SchemaContract | Mapping[str, Any]) -> SchemaContract:
    if isinstance(value, SchemaContract):
        return value
    if isinstance(value, Mapping):
        return SchemaContract.from_mapping(value)
    raise TypeError("contract must be a SchemaContract or mapping")


def _coerce_incoming(
    value: SchemaContract | Mapping[str, Any] | Sequence[Any],
) -> tuple[tuple[SchemaField, ...], str | None]:
    if isinstance(value, SchemaContract):
        return value.columns, value.version
    if isinstance(value, Mapping):
        version = _first_present(value, "version", "schema_version")
        columns = _first_present(value, "columns", "fields")
        if columns is not _MISSING:
            incoming_version = (
                None if version is _MISSING else _normalize_version(version)
            )
            return _coerce_fields(columns), incoming_version
        return _coerce_fields(value), None
    return _coerce_fields(value), None


def _match_fields(
    expected: tuple[SchemaField, ...],
    incoming: tuple[SchemaField, ...],
) -> tuple[
    tuple[tuple[SchemaField, SchemaField], ...],
    tuple[SchemaField, ...],
    tuple[SchemaField, ...],
]:
    expected_by_id = {
        field.field_id: field for field in expected if field.field_id is not None
    }
    incoming_by_id = {
        field.field_id: field for field in incoming if field.field_id is not None
    }
    expected_remaining = set(expected)
    incoming_remaining = set(incoming)
    matches: list[tuple[SchemaField, SchemaField]] = []

    for field_id in sorted(set(expected_by_id) & set(incoming_by_id)):
        expected_field = expected_by_id[field_id]
        incoming_field = incoming_by_id[field_id]
        matches.append((expected_field, incoming_field))
        expected_remaining.remove(expected_field)
        incoming_remaining.remove(incoming_field)

    expected_by_name = {field.name: field for field in expected_remaining}
    incoming_by_name = {field.name: field for field in incoming_remaining}
    for name in sorted(set(expected_by_name) & set(incoming_by_name)):
        expected_field = expected_by_name[name]
        incoming_field = incoming_by_name[name]
        matches.append((expected_field, incoming_field))
        expected_remaining.remove(expected_field)
        incoming_remaining.remove(incoming_field)

    matches.sort(key=lambda pair: _field_sort_key(pair[0]))
    return (
        tuple(matches),
        tuple(sorted(expected_remaining, key=_field_sort_key)),
        tuple(sorted(incoming_remaining, key=_field_sort_key)),
    )


def _role_is_unsafe(role: str) -> bool:
    return role in _PROTECTED_ROLES or role not in _SAFE_ROLES


def _role_transition_is_unsafe(before: str, after: str) -> bool:
    if before == after:
        return False
    return _role_is_unsafe(before) or _role_is_unsafe(after)


def _field_has_structural_drift(before: SchemaField, after: SchemaField) -> bool:
    return (
        before.name != after.name
        or before.dtype != after.dtype
        or before.nullable != after.nullable
    )


@dataclass(frozen=True, slots=True)
class SchemaDriftReport:
    """Counts-only result of comparing an incoming schema to a contract."""

    contract_version: str
    version_match: bool
    added: int
    removed: int
    renamed: int
    type_changed: int
    nullability_changed: int
    role_changed: int
    unsafe_role_drift: int
    unsafe_schema_drift: int
    release_blocked: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_version",
            _normalize_version(self.contract_version),
        )
        if type(self.version_match) is not bool:
            raise TypeError("schema report version_match must be a boolean")
        if type(self.release_blocked) is not bool:
            raise TypeError("schema report release_blocked must be a boolean")
        for name in (
            "added",
            "removed",
            "renamed",
            "type_changed",
            "nullability_changed",
            "role_changed",
            "unsafe_role_drift",
            "unsafe_schema_drift",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("schema report counts must be non-negative integers")

    @property
    def has_drift(self) -> bool:
        """Whether any schema or contract-version difference was observed."""

        return bool(
            self.counts["version_mismatch"]
            or any(
                self.counts[name]
                for name in (
                    "added",
                    "removed",
                    "renamed",
                    "type_changed",
                    "nullability_changed",
                    "role_changed",
                )
            )
        )

    @property
    def passed(self) -> bool:
        """Whether this report permits the privacy release gate to pass."""

        return not self.release_blocked

    @property
    def counts(self) -> dict[str, int]:
        """Return only aggregate, non-sensitive diagnostics."""

        return {
            "added": self.added,
            "removed": self.removed,
            "renamed": self.renamed,
            "type_changed": self.type_changed,
            "nullability_changed": self.nullability_changed,
            "role_changed": self.role_changed,
            "unsafe_role_drift": self.unsafe_role_drift,
            "unsafe_schema_drift": self.unsafe_schema_drift,
            "version_mismatch": int(not self.version_match),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a counts-only JSON-compatible CI evidence payload."""

        return {
            "report_schema_version": _REPORT_SCHEMA_VERSION,
            "contract_version": self.contract_version,
            "version_match": self.version_match,
            "release_blocked": self.release_blocked,
            "counts": self.counts,
        }

    def to_json(self) -> str:
        """Serialize counts-only diagnostics deterministically."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def raise_if_blocked(self) -> SchemaDriftReport:
        """Raise a counts-only error when this report blocks a release."""

        if self.release_blocked:
            raise SchemaDriftError(self)
        return self


class SchemaDriftError(ValueError):
    """Raised when unsafe tabular schema drift blocks a release."""

    def __init__(self, report: SchemaDriftReport) -> None:
        if not isinstance(report, SchemaDriftReport):
            raise TypeError("SchemaDriftError requires a SchemaDriftReport")
        self.report = report
        counts = report.counts
        super().__init__(
            "schema contract rejected release; "
            f"added={counts['added']}, removed={counts['removed']}, "
            f"renamed={counts['renamed']}, "
            f"type_changed={counts['type_changed']}, "
            f"nullability_changed={counts['nullability_changed']}, "
            f"role_changed={counts['role_changed']}, "
            f"unsafe_role_drift={counts['unsafe_role_drift']}, "
            f"unsafe_schema_drift={counts['unsafe_schema_drift']}, "
            f"version_mismatch={counts['version_mismatch']}"
        )


def compare_schema_drift(
    contract: SchemaContract | Mapping[str, Any],
    incoming: SchemaContract | Mapping[str, Any] | Sequence[Any],
) -> SchemaDriftReport:
    """Compare an incoming schema to a versioned contract.

    The returned report never contains column names or input values.  A
    release is blocked for a version mismatch, a role transition involving a
    protected or unknown role, a protected-column add/remove, or structural
    drift on a protected or unknown column.  Drift limited to
    ``non_sensitive`` and ``excluded`` columns is reported but does not block
    this privacy gate.
    """

    expected_contract = _coerce_contract(contract)
    incoming_fields, incoming_version = _coerce_incoming(incoming)
    version_match = (
        incoming_version is None or incoming_version == expected_contract.version
    )
    matches, removed_fields, added_fields = _match_fields(
        expected_contract.columns,
        incoming_fields,
    )

    added = len(added_fields)
    removed = len(removed_fields)
    renamed = 0
    type_changed = 0
    nullability_changed = 0
    role_changed = 0
    unsafe_role_drift = sum(
        _role_is_unsafe(field.role) for field in (*added_fields, *removed_fields)
    )
    unsafe_schema_drift = 0

    for expected_field, incoming_field in matches:
        renamed += expected_field.name != incoming_field.name
        type_changed += expected_field.dtype != incoming_field.dtype
        nullability_changed += expected_field.nullable != incoming_field.nullable
        role_changed += expected_field.role != incoming_field.role
        unsafe_role_drift += _role_transition_is_unsafe(
            expected_field.role,
            incoming_field.role,
        )
        if _field_has_structural_drift(expected_field, incoming_field) and (
            _role_is_unsafe(expected_field.role) or _role_is_unsafe(incoming_field.role)
        ):
            unsafe_schema_drift += 1

    release_blocked = bool(
        not version_match or unsafe_role_drift or unsafe_schema_drift
    )
    return SchemaDriftReport(
        contract_version=expected_contract.version,
        version_match=version_match,
        added=added,
        removed=removed,
        renamed=renamed,
        type_changed=type_changed,
        nullability_changed=nullability_changed,
        role_changed=role_changed,
        unsafe_role_drift=unsafe_role_drift,
        unsafe_schema_drift=unsafe_schema_drift,
        release_blocked=release_blocked,
    )


def enforce_schema_contract(
    contract: SchemaContract | Mapping[str, Any],
    incoming: SchemaContract | Mapping[str, Any] | Sequence[Any],
) -> SchemaDriftReport:
    """Compare schemas and raise a counts-only error when release is blocked."""

    return compare_schema_drift(contract, incoming).raise_if_blocked()
