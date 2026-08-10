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
    if isinstance(value, bool):
        raise TypeError("schema version must be an integer or semantic version")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("schema version must be non-negative")
        return _Version(value, 0, 0)
    if not isinstance(value, str):
        raise TypeError("schema version must be an integer or semantic version")

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
    if not isinstance(value, str) or not value.strip():
        raise ValueError("schema field path must be a non-empty string")
    normalized = value.strip()
    if "\x00" in normalized:
        raise ValueError("schema field path contains an invalid character")
    return normalized


def _normalize_type(value: Any) -> tuple[str, bool]:
    nullable = False
    if isinstance(value, str):
        members: Sequence[Any] = value.split("|")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        members = value
    else:
        raise TypeError("schema field type must be a string or sequence of strings")

    normalized_members: set[str] = set()
    for member in members:
        if not isinstance(member, str) or not member.strip():
            raise ValueError("schema field type members must be non-empty strings")
        normalized = member.strip().casefold()
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

        if isinstance(payload, cls):
            if path is not None and payload.path != _normalize_path(path):
                raise ValueError("schema field path is inconsistent")
            return payload
        if isinstance(payload, str):
            if path is None:
                raise ValueError("schema field path is required")
            return cls(path=path, type=payload)
        if not isinstance(payload, Mapping):
            raise TypeError("schema field must be metadata mapping")

        field_path = payload.get("path", path)
        raw_type = payload.get("type", payload.get("field_type", _MISSING))
        if field_path is None or raw_type is _MISSING:
            raise ValueError("schema field metadata requires path and type")
        if "optional" in payload:
            optional = payload["optional"]
        elif "nullable" in payload:
            optional = payload["nullable"]
        elif "required" in payload:
            required = payload["required"]
            if type(required) is not bool:
                raise TypeError("schema field requiredness must be a boolean")
            optional = not required
        else:
            optional = False
        return cls(path=field_path, type=raw_type, optional=optional)


def _normalize_fields(
    fields: Mapping[str, Any] | Sequence[Any],
) -> tuple[SchemaField, ...]:
    if isinstance(fields, Mapping):
        entries = ((path, payload) for path, payload in fields.items())
    elif isinstance(fields, Sequence) and not isinstance(
        fields, (str, bytes, bytearray)
    ):
        entries = ((None, payload) for payload in fields)
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
        if schema_version is not None:
            if version != "1.0.0" and _parse_version(version) != _parse_version(
                schema_version
            ):
                raise ValueError("schema version aliases are inconsistent")
            version = schema_version
        parsed_version = _parse_version(version)
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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SchemaSnapshot":
        """Load a snapshot mapping while discarding non-schema metadata."""

        if not isinstance(payload, Mapping):
            raise TypeError("schema snapshot must be a mapping")

        format_version = payload.get("format_version", SCHEMA_SNAPSHOT_FORMAT_VERSION)
        if (
            type(format_version) is not int
            or format_version != SCHEMA_SNAPSHOT_FORMAT_VERSION
        ):
            raise ValueError("unsupported schema snapshot format version")

        if "fields" in payload:
            fields = payload["fields"]
            version = payload.get("version", payload.get("schema_version", "1.0.0"))
            rules_version = payload.get("rules_version", COMPATIBILITY_RULES_VERSION)
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
        if self.kind not in {"added", "removed", "changed"}:
            raise ValueError("schema change kind is unsupported")
        object.__setattr__(self, "path", _normalize_path(self.path))
        if not self.reasons:
            raise ValueError("schema change must have at least one reason")
        if type(self.breaking) is not bool:
            raise TypeError("schema change breaking flag must be a boolean")

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
    if isinstance(value, SchemaSnapshot):
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
