"""Deterministic, value-free compatibility checks for schema snapshots.

Snapshots contain field metadata only.  They deliberately do not accept or
retain example payloads, which keeps compatibility reports suitable for local
release checks without copying sensitive values into an artifact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, Final

SCHEMA_SNAPSHOT_FORMAT_VERSION: Final = 1
COMPATIBILITY_RULES_VERSION: Final = 1

_MISSING = object()
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?(?:\.(0|[1-9][0-9]*))?$"
)
_TYPE_ALIASES: Final[Mapping[str, str]] = {
    "bool": "boolean",
    "dict": "object",
    "float": "number",
    "int": "integer",
    "list": "array",
}
_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_MAX_FIELDS: Final = 10_000
_MAX_FIELD_PATH_LENGTH: Final = 1_024
_MAX_TYPE_MEMBERS: Final = 128
_MAX_TYPE_NAME_LENGTH: Final = 128
_MAX_VERSION_LENGTH: Final = 64
_CHANGE_KINDS: Final = frozenset({"added", "removed", "changed"})
_CHANGE_REASONS: Final = frozenset(
    {
        "field_became_optional",
        "field_became_required",
        "field_removed",
        "optional_field_added",
        "required_field_added",
        "type_changed",
        "type_widened",
    }
)
_REASONS_BY_KIND: Final[Mapping[str, frozenset[str]]] = {
    "added": frozenset({"optional_field_added", "required_field_added"}),
    "removed": frozenset({"field_removed"}),
    "changed": frozenset(
        {
            "field_became_optional",
            "field_became_required",
            "type_changed",
            "type_widened",
        }
    ),
}
_BREAKING_REASONS: Final = frozenset(
    {
        "field_became_required",
        "field_removed",
        "required_field_added",
        "type_changed",
    }
)
_VIOLATIONS: Final = frozenset(
    {
        "breaking_change_requires_major_version_bump",
        "schema_version_regressed",
    }
)

__all__ = [
    "COMPATIBILITY_RULES_VERSION",
    "SCHEMA_SNAPSHOT_FORMAT_VERSION",
    "SchemaChange",
    "SchemaCompatibilityReport",
    "SchemaField",
    "SchemaSnapshot",
    "build_schema_snapshot",
    "check_schema_compatibility",
    "compare_schema_snapshots",
    "is_schema_compatible",
]


@dataclass(frozen=True, slots=True, order=True)
class _Version:
    """Parsed semantic version used for deterministic rule evaluation."""

    major: int
    minor: int
    patch: int


def _parse_version(value: Any) -> _Version:
    if type(value) is int:
        if value < 0:
            raise ValueError("schema version must be non-negative")
        return _Version(value, 0, 0)
    if type(value) is not str:
        raise TypeError("schema version must be an integer or semantic version")
    if len(value) > _MAX_VERSION_LENGTH:
        raise ValueError("schema version is too long")

    match = _SEMVER_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError("schema version must use major.minor.patch notation")
    major, minor, patch = (int(part or 0) for part in match.groups())
    return _Version(major, minor, patch)


def _version_string(version: _Version) -> str:
    return f"{version.major}.{version.minor}.{version.patch}"


def _validate_rules_version(value: Any) -> int:
    if type(value) is not int or value != COMPATIBILITY_RULES_VERSION:
        raise ValueError("unsupported schema compatibility-rules version")
    return value


def _normalize_path(value: Any) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("schema field path must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > _MAX_FIELD_PATH_LENGTH:
        raise ValueError("schema field path is too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError("schema field path contains an invalid character")
    return normalized


def _normalize_type(value: Any) -> tuple[str, bool]:
    nullable = False
    if type(value) is str:
        members = tuple(value.split("|"))
    elif isinstance(value, str):
        raise TypeError("schema field type must use plain strings")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        members = _bounded_tuple(
            value,
            limit=_MAX_TYPE_MEMBERS,
            type_error="schema field type sequence could not be read",
            limit_error="schema field type has too many members",
        )
    else:
        raise TypeError("schema field type must be a string or sequence of strings")
    if len(members) > _MAX_TYPE_MEMBERS:
        raise ValueError("schema field type has too many members")

    normalized_members: set[str] = set()
    for member in members:
        if type(member) is not str or not member.strip():
            raise ValueError("schema field type members must be non-empty strings")
        normalized = member.strip().casefold()
        if len(normalized) > _MAX_TYPE_NAME_LENGTH:
            raise ValueError("schema field type name is too long")
        if normalized == "null":
            nullable = True
            continue
        normalized = _TYPE_ALIASES.get(normalized, normalized)
        if _TYPE_PATTERN.fullmatch(normalized) is None:
            raise ValueError("schema field type contains an invalid name")
        normalized_members.add(normalized)

    if not normalized_members:
        normalized_members.add("null")
    return "|".join(sorted(normalized_members)), nullable


def _bounded_tuple(
    values: Any,
    *,
    limit: int,
    type_error: str,
    limit_error: str,
) -> tuple[Any, ...]:
    try:
        result = tuple(islice(iter(values), limit + 1))
    except Exception:
        raise TypeError(type_error) from None
    if len(result) > limit:
        raise ValueError(limit_error)
    return result


def _mapping_entry(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[bool, Any]:
    try:
        if key not in payload:
            return False, _MISSING
        return True, payload[key]
    except Exception:
        raise TypeError("schema metadata mapping could not be read") from None


@dataclass(frozen=True, slots=True)
class SchemaField:
    """Value-free metadata for one schema field.

    Args:
        path: Stable nested field path, such as ``"encounter.measurement"``.
        type: Canonical scalar, container, or union type name.  Common aliases
            such as ``int`` and ``float`` are normalized.
        optional: Whether the field may be absent.  A ``null`` member in a
            type union also marks the field optional.
    """

    path: str
    type: str
    optional: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        normalized_type, nullable = _normalize_type(self.type)
        object.__setattr__(self, "type", normalized_type)
        if type(self.optional) is not bool:
            raise TypeError("schema field optionality must be a boolean")
        object.__setattr__(self, "optional", self.optional or nullable)

    @property
    def field_type(self) -> str:
        """Return ``type`` under a descriptive alias."""

        return self.type

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | str | "SchemaField",
        *,
        path: str | None = None,
    ) -> "SchemaField":
        """Create a field from metadata, ignoring example-payload keys.

        The mapping form requires ``type`` and accepts ``optional``.  A
        mapping supplied under a field-path key may omit ``path``.  Unknown
        keys such as ``example``, ``default``, and ``description`` are never
        copied into the field or a later report.
        """

        if type(payload) is cls:
            if path is not None and payload.path != _normalize_path(path):
                raise ValueError("schema field path is inconsistent")
            return payload
        if type(payload) is str:
            if path is None:
                raise ValueError("schema field path is required")
            return cls(path=path, type=payload)
        if not isinstance(payload, Mapping):
            raise TypeError("schema field must be metadata mapping")

        has_payload_path, payload_path = _mapping_entry(payload, "path")
        if path is not None and has_payload_path:
            if _normalize_path(payload_path) != _normalize_path(path):
                raise ValueError("schema field path is inconsistent")
        field_path = payload_path if has_payload_path else path
        has_type, type_value = _mapping_entry(payload, "type")
        has_field_type, field_type_value = _mapping_entry(payload, "field_type")
        raw_type = type_value if has_type else field_type_value
        if field_path is None or raw_type is _MISSING:
            raise ValueError("schema field metadata requires path and type")
        normalized_type, type_nullable = _normalize_type(raw_type)
        if has_type and has_field_type:
            if _normalize_type(type_value) != _normalize_type(field_type_value):
                raise ValueError("schema field type aliases are inconsistent")

        optionality: list[bool] = []
        for key in ("optional", "nullable"):
            has_value, value = _mapping_entry(payload, key)
            if has_value:
                if type(value) is not bool:
                    raise TypeError("schema field optionality must be a boolean")
                optionality.append(value)
        has_required, required = _mapping_entry(payload, "required")
        if has_required:
            if type(required) is not bool:
                raise TypeError("schema field requiredness must be a boolean")
            optionality.append(not required)
        if len(set(optionality)) > 1:
            raise ValueError("schema field optionality aliases are inconsistent")
        if optionality and not optionality[0] and type_nullable:
            raise ValueError("schema field optionality aliases are inconsistent")
        optional = (optionality[0] if optionality else False) or type_nullable
        return cls(path=field_path, type=normalized_type, optional=optional)


def _normalize_fields(
    fields: Mapping[str, Any] | Sequence[Any],
) -> tuple[SchemaField, ...]:
    if isinstance(fields, Mapping):
        try:
            items = fields.items()
        except Exception:
            raise TypeError("schema fields mapping could not be read") from None
        raw_entries = _bounded_tuple(
            items,
            limit=_MAX_FIELDS,
            type_error="schema fields mapping could not be read",
            limit_error="schema snapshot has too many fields",
        )
        entries: list[tuple[Any, Any]] = []
        for item in raw_entries:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("schema fields mapping contains an invalid entry")
            entries.append((item[0], item[1]))
    elif isinstance(fields, Sequence) and not isinstance(
        fields, (str, bytes, bytearray)
    ):
        payloads = _bounded_tuple(
            fields,
            limit=_MAX_FIELDS,
            type_error="schema fields sequence could not be read",
            limit_error="schema snapshot has too many fields",
        )
        entries = [(None, payload) for payload in payloads]
    else:
        raise TypeError("schema fields must be a mapping or sequence")

    by_path: dict[str, SchemaField] = {}
    for path, payload in entries:
        field = SchemaField.from_mapping(payload, path=path)
        if field.path in by_path:
            raise ValueError("schema snapshot contains duplicate field paths")
        by_path[field.path] = field
    return tuple(by_path[path] for path in sorted(by_path))


@dataclass(frozen=True, slots=True, init=False)
class SchemaSnapshot:
    """Immutable, normalized schema metadata for a versioned release.

    ``fields`` can be a ``{path: {"type": ..., "optional": ...}}`` mapping
    or a sequence of field metadata mappings.  Only those metadata keys are
    retained; payload examples are intentionally not part of this contract.
    """

    version: str
    fields: tuple[SchemaField, ...]
    rules_version: int

    def __init__(
        self,
        version: str | int = "1.0.0",
        fields: Mapping[str, Any] | Sequence[Any] = (),
        *,
        rules_version: int = COMPATIBILITY_RULES_VERSION,
        schema_version: str | int | None = None,
    ) -> None:
        version_uses_default = type(version) is str and version == "1.0.0"
        parsed_version = _parse_version(version)
        if schema_version is not None:
            parsed_schema_version = _parse_version(schema_version)
            if not version_uses_default and parsed_version != parsed_schema_version:
                raise ValueError("schema version aliases are inconsistent")
            parsed_version = parsed_schema_version
        object.__setattr__(self, "version", _version_string(parsed_version))
        object.__setattr__(self, "fields", _normalize_fields(fields))
        object.__setattr__(
            self, "rules_version", _validate_rules_version(rules_version)
        )

    @property
    def schema_version(self) -> str:
        """Return the normalized semantic version alias."""

        return self.version

    @property
    def field_map(self) -> dict[str, SchemaField]:
        """Return fields keyed by path in deterministic order."""

        return {field.path: field for field in self.fields}

    def to_dict(self) -> dict[str, Any]:
        """Return only deterministic schema metadata, never payload values."""

        return {
            "format_version": SCHEMA_SNAPSHOT_FORMAT_VERSION,
            "version": self.version,
            "rules_version": self.rules_version,
            "fields": [
                {
                    "path": field.path,
                    "type": field.type,
                    "optional": field.optional,
                }
                for field in self.fields
            ],
        }

    as_dict = to_dict

    def to_json(self) -> str:
        """Return canonical JSON containing schema metadata only."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SchemaSnapshot":
        """Load a snapshot mapping while discarding non-schema metadata."""

        if not isinstance(payload, Mapping):
            raise TypeError("schema snapshot must be a mapping")

        has_format_version, format_version = _mapping_entry(payload, "format_version")
        if not has_format_version:
            format_version = SCHEMA_SNAPSHOT_FORMAT_VERSION
        if (
            type(format_version) is not int
            or format_version != SCHEMA_SNAPSHOT_FORMAT_VERSION
        ):
            raise ValueError("unsupported schema snapshot format version")

        has_fields, fields = _mapping_entry(payload, "fields")
        if has_fields:
            has_version, version = _mapping_entry(payload, "version")
            has_schema_version, schema_version = _mapping_entry(
                payload, "schema_version"
            )
            if not has_version:
                version = "1.0.0"
            elif has_schema_version and _parse_version(version) != _parse_version(
                schema_version
            ):
                raise ValueError("schema version aliases are inconsistent")
            has_rules_version, rules_version = _mapping_entry(payload, "rules_version")
            if not has_rules_version:
                rules_version = COMPATIBILITY_RULES_VERSION
            return cls(
                version=version,
                fields=fields,
                rules_version=rules_version,
                schema_version=schema_version if has_schema_version else None,
            )
        else:
            fields = payload
            version = "1.0.0"
            rules_version = COMPATIBILITY_RULES_VERSION
        return cls(version=version, fields=fields, rules_version=rules_version)

    from_dict = from_mapping


@dataclass(frozen=True, slots=True)
class SchemaChange:
    """One deterministic addition, removal, or field-metadata modification."""

    kind: str
    path: str
    before: SchemaField | None
    after: SchemaField | None
    reasons: tuple[str, ...]
    breaking: bool

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _CHANGE_KINDS:
            raise ValueError("schema change kind is unsupported")
        object.__setattr__(self, "path", _normalize_path(self.path))
        if type(self.before) is not SchemaField and self.before is not None:
            raise TypeError("schema change before field is invalid")
        if type(self.after) is not SchemaField and self.after is not None:
            raise TypeError("schema change after field is invalid")
        if self.kind == "added" and (self.before is not None or self.after is None):
            raise ValueError("added schema changes require only an after field")
        if self.kind == "removed" and (self.before is None or self.after is not None):
            raise ValueError("removed schema changes require only a before field")
        if self.kind == "changed" and (self.before is None or self.after is None):
            raise ValueError("changed schema changes require before and after fields")
        if self.before is not None and self.before.path != self.path:
            raise ValueError("schema change before path is inconsistent")
        if self.after is not None and self.after.path != self.path:
            raise ValueError("schema change after path is inconsistent")
        if type(self.reasons) is not tuple or not self.reasons:
            raise ValueError("schema change must have at least one reason")
        if any(
            type(reason) is not str or reason not in _CHANGE_REASONS
            for reason in self.reasons
        ):
            raise ValueError("schema change reason is unsupported")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("schema change reasons must be unique")
        if any(reason not in _REASONS_BY_KIND[self.kind] for reason in self.reasons):
            raise ValueError("schema change reason does not match its kind")
        if type(self.breaking) is not bool:
            raise TypeError("schema change breaking flag must be a boolean")
        if self.breaking != any(reason in _BREAKING_REASONS for reason in self.reasons):
            raise ValueError("schema change breaking flag is inconsistent")

    @property
    def reason(self) -> str:
        """Return a stable primary reason for simple consumers."""

        return self.reasons[0]

    @property
    def incompatible(self) -> bool:
        """Return whether the change breaks the current compatibility rules."""

        return self.breaking

    @property
    def before_type(self) -> str | None:
        return self.before.type if self.before is not None else None

    @property
    def after_type(self) -> str | None:
        return self.after.type if self.after is not None else None

    @property
    def before_optional(self) -> bool | None:
        return self.before.optional if self.before is not None else None

    @property
    def after_optional(self) -> bool | None:
        return self.after.optional if self.after is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free JSON-compatible change record."""

        result: dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "breaking": self.breaking,
        }
        if self.before is not None:
            result["before"] = {
                "type": self.before.type,
                "optional": self.before.optional,
            }
        if self.after is not None:
            result["after"] = {
                "type": self.after.type,
                "optional": self.after.optional,
            }
        return result


def _validate_change_collection(
    value: tuple[SchemaChange, ...],
    name: str,
    *,
    kind: str | None = None,
) -> None:
    if type(value) is not tuple:
        raise TypeError(f"schema report {name} must be a tuple")
    if len(value) > 2 * _MAX_FIELDS:
        raise ValueError(f"schema report {name} contains too many changes")
    if any(type(change) is not SchemaChange for change in value):
        raise TypeError(f"schema report {name} contains an invalid change")
    if kind is not None and any(change.kind != kind for change in value):
        raise ValueError(f"schema report {name} contains the wrong change kind")
    paths = [change.path for change in value]
    if len(paths) != len(set(paths)):
        raise ValueError(f"schema report {name} contains duplicate paths")
    if kind is not None and paths != sorted(paths):
        raise ValueError(f"schema report {name} must use stable path order")


@dataclass(frozen=True, slots=True)
class SchemaCompatibilityReport:
    """Deterministic compatibility evidence for two schema snapshots.

    ``compatible`` means that the versioned rules accept the transition.  A
    major-version bump permits breaking field changes, but
    ``has_breaking_changes`` remains true so release tooling can distinguish an
    intentionally breaking major release from a non-breaking edit.
    """

    before_version: str
    after_version: str
    rules_version: int
    additions: tuple[SchemaChange, ...]
    removals: tuple[SchemaChange, ...]
    changes: tuple[SchemaChange, ...]
    incompatible_changes: tuple[SchemaChange, ...]
    compatible: bool
    version_bump_satisfies_rules: bool
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        before_version = _parse_version(self.before_version)
        after_version = _parse_version(self.after_version)
        object.__setattr__(self, "before_version", _version_string(before_version))
        object.__setattr__(self, "after_version", _version_string(after_version))
        _validate_rules_version(self.rules_version)

        _validate_change_collection(self.additions, "additions", kind="added")
        _validate_change_collection(self.removals, "removals", kind="removed")
        _validate_change_collection(self.changes, "changes", kind="changed")
        _validate_change_collection(
            self.incompatible_changes,
            "incompatible changes",
        )
        expected_incompatible = tuple(
            change
            for change in (*self.additions, *self.removals, *self.changes)
            if change.breaking
        )
        if self.incompatible_changes != expected_incompatible:
            raise ValueError("incompatible schema changes are inconsistent")

        if type(self.compatible) is not bool:
            raise TypeError("schema compatibility flag must be a boolean")
        if type(self.version_bump_satisfies_rules) is not bool:
            raise TypeError("schema version-rule flag must be a boolean")
        if type(self.violations) is not tuple or any(
            type(violation) is not str or violation not in _VIOLATIONS
            for violation in self.violations
        ):
            raise ValueError("schema compatibility violation is unsupported")
        if len(set(self.violations)) != len(self.violations):
            raise ValueError("schema compatibility violations must be unique")

        expected_violations: list[str] = []
        if after_version < before_version:
            expected_violations.append("schema_version_regressed")
        if expected_incompatible and after_version.major <= before_version.major:
            expected_violations.append("breaking_change_requires_major_version_bump")
        if self.violations != tuple(expected_violations):
            raise ValueError("schema compatibility violations are inconsistent")
        expected_compatible = not expected_violations
        if (
            self.compatible != expected_compatible
            or self.version_bump_satisfies_rules != expected_compatible
        ):
            raise ValueError("schema compatibility flags are inconsistent")

    @property
    def breaking_changes(self) -> tuple[SchemaChange, ...]:
        """Return the incompatible field changes, including major releases."""

        return self.incompatible_changes

    @property
    def has_breaking_changes(self) -> bool:
        """Return whether any field change is breaking."""

        return bool(self.incompatible_changes)

    @property
    def is_compatible(self) -> bool:
        """Return ``compatible`` under a descriptive alias."""

        return self.compatible

    @property
    def added(self) -> tuple[SchemaChange, ...]:
        """Return additions under a concise alias."""

        return self.additions

    @property
    def removed(self) -> tuple[SchemaChange, ...]:
        """Return removals under a concise alias."""

        return self.removals

    @property
    def incompatible(self) -> tuple[SchemaChange, ...]:
        """Return breaking changes under a concise alias."""

        return self.incompatible_changes

    @property
    def has_violations(self) -> bool:
        """Return whether the versioned compatibility contract was violated."""

        return bool(self.violations)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic report metadata without example payloads."""

        return {
            "format_version": SCHEMA_SNAPSHOT_FORMAT_VERSION,
            "before_version": self.before_version,
            "after_version": self.after_version,
            "rules_version": self.rules_version,
            "compatible": self.compatible,
            "version_bump_satisfies_rules": self.version_bump_satisfies_rules,
            "has_breaking_changes": self.has_breaking_changes,
            "violations": list(self.violations),
            "additions": [change.to_dict() for change in self.additions],
            "removals": [change.to_dict() for change in self.removals],
            "changes": [change.to_dict() for change in self.changes],
            "incompatible_changes": [
                change.to_dict() for change in self.incompatible_changes
            ],
        }

    as_dict = to_dict

    def to_json(self) -> str:
        """Return canonical JSON suitable for a local release artifact."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def build_schema_snapshot(
    fields: Mapping[str, Any] | Sequence[Any],
    *,
    version: str | int = "1.0.0",
    rules_version: int = COMPATIBILITY_RULES_VERSION,
) -> SchemaSnapshot:
    """Build a normalized snapshot from field metadata only."""

    return SchemaSnapshot(
        version=version,
        fields=fields,
        rules_version=rules_version,
    )


def _coerce_snapshot(value: SchemaSnapshot | Mapping[str, Any]) -> SchemaSnapshot:
    if type(value) is SchemaSnapshot:
        return value
    if isinstance(value, Mapping):
        return SchemaSnapshot.from_mapping(value)
    raise TypeError("schema snapshot must be a SchemaSnapshot or mapping")


def _type_members(field_type: str) -> frozenset[str]:
    return frozenset(field_type.split("|"))


def _type_accepts(old_member: str, new_members: frozenset[str]) -> bool:
    if "any" in new_members:
        return True
    if old_member in new_members:
        return True
    return old_member == "integer" and "number" in new_members


def _is_type_widening(before: str, after: str) -> bool:
    before_members = _type_members(before)
    after_members = _type_members(after)
    return before_members != after_members and all(
        _type_accepts(member, after_members) for member in before_members
    )


def _change_reasons(
    before: SchemaField,
    after: SchemaField,
) -> tuple[tuple[str, ...], bool]:
    reasons: list[str] = []
    breaking = False
    if before.type != after.type:
        if _is_type_widening(before.type, after.type):
            reasons.append("type_widened")
        else:
            reasons.append("type_changed")
            breaking = True
    if before.optional != after.optional:
        if before.optional and not after.optional:
            reasons.append("field_became_required")
            breaking = True
        else:
            reasons.append("field_became_optional")
    return tuple(reasons), breaking


def compare_schema_snapshots(
    before: SchemaSnapshot | Mapping[str, Any],
    after: SchemaSnapshot | Mapping[str, Any],
    *,
    rules_version: int | None = None,
) -> SchemaCompatibilityReport:
    """Compare two local snapshots under the versioned compatibility rules.

    Optional fields may be added and field types may be widened.  Required
    additions, removals, type changes that narrow the accepted values, and
    making an optional field required are breaking changes.  A breaking change
    is accepted only when ``after`` has a strictly greater major version.
    """

    old_snapshot = _coerce_snapshot(before)
    new_snapshot = _coerce_snapshot(after)
    if old_snapshot.rules_version != new_snapshot.rules_version:
        raise ValueError("schema snapshots use different compatibility-rules versions")
    if rules_version is not None:
        expected_rules_version = _validate_rules_version(rules_version)
        if old_snapshot.rules_version != expected_rules_version:
            raise ValueError(
                "schema snapshot uses an unexpected compatibility-rules version"
            )

    old_fields = old_snapshot.field_map
    new_fields = new_snapshot.field_map
    additions: list[SchemaChange] = []
    removals: list[SchemaChange] = []
    changes: list[SchemaChange] = []

    for path in sorted(new_fields.keys() - old_fields.keys()):
        field = new_fields[path]
        additions.append(
            SchemaChange(
                kind="added",
                path=path,
                before=None,
                after=field,
                reasons=(
                    "optional_field_added"
                    if field.optional
                    else "required_field_added",
                ),
                breaking=not field.optional,
            )
        )

    for path in sorted(old_fields.keys() - new_fields.keys()):
        removals.append(
            SchemaChange(
                kind="removed",
                path=path,
                before=old_fields[path],
                after=None,
                reasons=("field_removed",),
                breaking=True,
            )
        )

    for path in sorted(old_fields.keys() & new_fields.keys()):
        old_field = old_fields[path]
        new_field = new_fields[path]
        if old_field == new_field:
            continue
        reasons, breaking = _change_reasons(old_field, new_field)
        changes.append(
            SchemaChange(
                kind="changed",
                path=path,
                before=old_field,
                after=new_field,
                reasons=reasons,
                breaking=breaking,
            )
        )

    all_changes = (*additions, *removals, *changes)
    incompatible_changes = tuple(change for change in all_changes if change.breaking)
    old_version = _parse_version(old_snapshot.version)
    new_version = _parse_version(new_snapshot.version)
    violations: list[str] = []
    if new_version < old_version:
        violations.append("schema_version_regressed")
    if incompatible_changes and new_version.major <= old_version.major:
        violations.append("breaking_change_requires_major_version_bump")
    version_bump_satisfies_rules = not violations

    return SchemaCompatibilityReport(
        before_version=old_snapshot.version,
        after_version=new_snapshot.version,
        rules_version=old_snapshot.rules_version,
        additions=tuple(additions),
        removals=tuple(removals),
        changes=tuple(changes),
        incompatible_changes=incompatible_changes,
        compatible=version_bump_satisfies_rules,
        version_bump_satisfies_rules=version_bump_satisfies_rules,
        violations=tuple(violations),
    )


def check_schema_compatibility(
    before: SchemaSnapshot | Mapping[str, Any],
    after: SchemaSnapshot | Mapping[str, Any],
    *,
    rules_version: int | None = None,
) -> SchemaCompatibilityReport:
    """Return the structured compatibility report for two snapshots."""

    return compare_schema_snapshots(before, after, rules_version=rules_version)


def is_schema_compatible(
    before: SchemaSnapshot | Mapping[str, Any],
    after: SchemaSnapshot | Mapping[str, Any],
    *,
    rules_version: int | None = None,
) -> bool:
    """Return whether the versioned compatibility rules accept a transition."""

    return compare_schema_snapshots(
        before,
        after,
        rules_version=rules_version,
    ).compatible
