"""Deterministic, content-free diffs of clinical agent tool catalogs.

Catalog entries are keyed by tool ID and version. The diff accepts only the
closed :mod:`openmed.agent.tool_inventory` snapshot contract, so endpoints,
credentials, arguments, examples, and other free-form metadata fail closed
before comparison or rendering.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from openmed.agent.tool_inventory import (
    SideEffectClass,
    ToolInventory,
    ToolInventoryError,
    ToolInventoryRecord,
)

TOOL_CATALOG_DIFF_SCHEMA_VERSION: Final = "openmed.agent.tool_catalog_diff.v1"

_COMPARED_FIELDS: Final = (
    "capability_class",
    "side_effect_class",
    "schema_digest",
)
_FIELD_ORDER: Final = {field: index for index, field in enumerate(_COMPARED_FIELDS)}
_VALIDATION_CAPABILITY: Final = "capability:org.openmed/catalog-diff@1.0.0"
_VALIDATION_DIGEST: Final = "sha256:" + "0" * 64

CatalogInput = ToolInventory | Mapping[str, Any]
CatalogKey = tuple[str, str]


class ToolCatalogDiffError(ValueError):
    """Raised when a catalog diff input violates the content-free contract.

    Error text contains only fixed snapshot and field names plus a stable
    reason code. Rejected values are never retained or echoed.
    """

    def __init__(self, code: str, field_name: str) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(f"{field_name}: {code}")


@dataclass(frozen=True, slots=True)
class ToolCatalogFieldChange:
    """Before and after values for one content-free catalog field."""

    field: str
    before: str
    after: str

    def __post_init__(self) -> None:
        if self.field not in _COMPARED_FIELDS:
            raise ToolCatalogDiffError("invalid_field", "change.field")
        _validate_field_value(self.field, self.before, "change.before")
        _validate_field_value(self.field, self.after, "change.after")
        if self.before == self.after:
            raise ToolCatalogDiffError("unchanged_value", "change")

    def to_dict(self) -> dict[str, str]:
        """Return the deterministic JSON-compatible field delta."""

        return {"field": self.field, "before": self.before, "after": self.after}


@dataclass(frozen=True, slots=True)
class ToolCatalogChange:
    """Safe field changes for one tool ID and version."""

    tool_id: str
    version: str
    fields: tuple[ToolCatalogFieldChange, ...]

    def __post_init__(self) -> None:
        _validate_identity(self.tool_id, self.version)
        if (
            type(self.fields) is not tuple
            or not self.fields
            or not all(type(field) is ToolCatalogFieldChange for field in self.fields)
        ):
            raise ToolCatalogDiffError("invalid_fields", "change.fields")
        field_names = tuple(field.field for field in self.fields)
        if len(field_names) != len(set(field_names)) or field_names != tuple(
            sorted(field_names, key=_FIELD_ORDER.__getitem__)
        ):
            raise ToolCatalogDiffError("invalid_fields", "change.fields")

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-compatible tool change."""

        return {
            "tool_id": self.tool_id,
            "version": self.version,
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True, slots=True, repr=False)
class ToolCatalogDiff:
    """Structured delta between two content-free tool catalog snapshots."""

    added: tuple[ToolInventoryRecord, ...]
    removed: tuple[ToolInventoryRecord, ...]
    changed: tuple[ToolCatalogChange, ...]
    unchanged: tuple[ToolInventoryRecord, ...]
    schema_version: str = TOOL_CATALOG_DIFF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_CATALOG_DIFF_SCHEMA_VERSION:
            raise ToolCatalogDiffError("unsupported_version", "schema_version")
        for field_name in ("added", "removed", "unchanged"):
            records = getattr(self, field_name)
            if type(records) is not tuple or not all(
                type(record) is ToolInventoryRecord for record in records
            ):
                raise ToolCatalogDiffError("invalid_records", field_name)
        if type(self.changed) is not tuple or not all(
            type(change) is ToolCatalogChange for change in self.changed
        ):
            raise ToolCatalogDiffError("invalid_changes", "changed")

    @property
    def is_empty(self) -> bool:
        """Return whether no tool was added, removed, or changed."""

        return not (self.added or self.removed or self.changed)

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed JSON-compatible diff projection."""

        return {
            "schema_version": self.schema_version,
            "added": [record.to_dict() for record in self.added],
            "removed": [record.to_dict() for record in self.removed],
            "changed": [change.to_dict() for change in self.changed],
            "unchanged": [record.to_dict() for record in self.unchanged],
        }

    def to_json(self) -> str:
        """Render byte-stable canonical JSON."""

        return render_tool_catalog_diff_json(self)

    def to_markdown(self) -> str:
        """Render deterministic Markdown for deployment review."""

        return render_tool_catalog_diff_markdown(self)

    def __repr__(self) -> str:
        """Return a count-only representation."""

        return (
            "ToolCatalogDiff("
            f"added={len(self.added)}, removed={len(self.removed)}, "
            f"changed={len(self.changed)}, unchanged={len(self.unchanged)})"
        )


def diff_tool_catalogs(
    baseline: CatalogInput,
    candidate: CatalogInput,
) -> ToolCatalogDiff:
    """Compare two content-free tool catalog snapshots.

    Entries are keyed by ``(tool_id, version)`` so catalogs can contain more
    than one registered version of a tool. A version replacement is therefore
    one removal and one addition. Capability, side-effect, or schema-digest
    drift for the same key is reported as a changed entry.

    Args:
        baseline: Typed inventory or its exact JSON-compatible mapping.
        candidate: Typed inventory or its exact JSON-compatible mapping.

    Returns:
        A deterministically ordered, content-free catalog delta.

    Raises:
        ToolCatalogDiffError: If either snapshot contains unsupported fields
            or otherwise violates the inventory contract.
    """

    baseline_inventory = _coerce_catalog(baseline, snapshot="baseline")
    candidate_inventory = _coerce_catalog(candidate, snapshot="candidate")
    baseline_by_key = _records_by_key(baseline_inventory)
    candidate_by_key = _records_by_key(candidate_inventory)
    baseline_keys = set(baseline_by_key)
    candidate_keys = set(candidate_by_key)

    added = tuple(
        candidate_by_key[key]
        for key in sorted(candidate_keys - baseline_keys, key=_catalog_key)
    )
    removed = tuple(
        baseline_by_key[key]
        for key in sorted(baseline_keys - candidate_keys, key=_catalog_key)
    )
    changed: list[ToolCatalogChange] = []
    unchanged: list[ToolInventoryRecord] = []
    for key in sorted(baseline_keys & candidate_keys, key=_catalog_key):
        before = baseline_by_key[key]
        after = candidate_by_key[key]
        fields = tuple(
            ToolCatalogFieldChange(
                field=field,
                before=_field_value(before, field),
                after=_field_value(after, field),
            )
            for field in _COMPARED_FIELDS
            if _field_value(before, field) != _field_value(after, field)
        )
        if fields:
            changed.append(
                ToolCatalogChange(
                    tool_id=before.tool_id,
                    version=before.version,
                    fields=fields,
                )
            )
        else:
            unchanged.append(after)

    return ToolCatalogDiff(
        added=added,
        removed=removed,
        changed=tuple(changed),
        unchanged=tuple(unchanged),
    )


def render_tool_catalog_diff_json(diff: ToolCatalogDiff) -> str:
    """Render a tool catalog diff as deterministic compact JSON."""

    _require_diff(diff)
    return json.dumps(
        diff.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def render_tool_catalog_diff_markdown(diff: ToolCatalogDiff) -> str:
    """Render a tool catalog diff as deterministic content-free Markdown."""

    _require_diff(diff)
    lines = [
        "# Agent tool catalog diff",
        "",
        f"Schema: `{diff.schema_version}`",
        "",
        f"Added: {len(diff.added)}",
        f"Removed: {len(diff.removed)}",
        f"Changed: {len(diff.changed)}",
        f"Unchanged: {len(diff.unchanged)}",
        "",
    ]
    _append_record_section(lines, "Added", diff.added)
    _append_record_section(lines, "Removed", diff.removed)
    _append_change_section(lines, diff.changed)
    _append_record_section(lines, "Unchanged", diff.unchanged)
    return "\n".join(lines) + "\n"


def _coerce_catalog(value: CatalogInput, *, snapshot: str) -> ToolInventory:
    if type(value) is ToolInventory:
        return value
    if not isinstance(value, Mapping):
        raise ToolCatalogDiffError("invalid_type", snapshot)
    try:
        return ToolInventory.from_dict(value)
    except ToolInventoryError as exc:
        raise ToolCatalogDiffError(exc.code, f"{snapshot}.{exc.field_name}") from None


def _records_by_key(inventory: ToolInventory) -> dict[CatalogKey, ToolInventoryRecord]:
    return {(record.tool_id, record.version): record for record in inventory.tools}


def _catalog_key(key: CatalogKey) -> tuple[str, tuple[int, int, int]]:
    tool_id, version = key
    major, minor, patch = (int(part) for part in version.split("."))
    return tool_id, (major, minor, patch)


def _field_value(record: ToolInventoryRecord, field: str) -> str:
    value = getattr(record, field)
    if type(value) is SideEffectClass:
        return value.value
    if type(value) is str:
        return value
    raise ToolCatalogDiffError("invalid_value", field)


def _validate_identity(tool_id: str, version: str) -> None:
    try:
        ToolInventoryRecord(
            tool_id=tool_id,
            version=version,
            capability_class=_VALIDATION_CAPABILITY,
            side_effect_class=SideEffectClass.NONE,
            schema_digest=_VALIDATION_DIGEST,
        )
    except ToolInventoryError as exc:
        raise ToolCatalogDiffError(exc.code, f"change.{exc.field_name}") from None


def _validate_field_value(field: str, value: str, field_name: str) -> None:
    capability_class = _VALIDATION_CAPABILITY
    side_effect_class = SideEffectClass.NONE
    schema_digest = _VALIDATION_DIGEST
    if field == "side_effect_class":
        try:
            side_effect_class = SideEffectClass(value)
        except (TypeError, ValueError):
            raise ToolCatalogDiffError("invalid_class", field_name) from None
    elif field == "capability_class":
        capability_class = value
    elif field == "schema_digest":
        schema_digest = value
    try:
        ToolInventoryRecord(
            tool_id="tool:org.openmed/catalog-diff",
            version="1.0.0",
            capability_class=capability_class,
            side_effect_class=side_effect_class,
            schema_digest=schema_digest,
        )
    except ToolInventoryError as exc:
        raise ToolCatalogDiffError(exc.code, field_name) from None


def _append_record_section(
    lines: list[str],
    title: str,
    records: tuple[ToolInventoryRecord, ...],
) -> None:
    lines.extend((f"## {title}", ""))
    if not records:
        lines.extend((f"_No {title.lower()} tools._", ""))
        return
    lines.extend(
        (
            "| Tool ID | Version | Capability class | Side-effect class | Schema digest |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    lines.extend(
        "| "
        f"`{record.tool_id}` | `{record.version}` | "
        f"`{record.capability_class}` | `{record.side_effect_class.value}` | "
        f"`{record.schema_digest}` |"
        for record in records
    )
    lines.append("")


def _append_change_section(
    lines: list[str], changes: tuple[ToolCatalogChange, ...]
) -> None:
    lines.extend(("## Changed", ""))
    if not changes:
        lines.extend(("_No changed tools._", ""))
        return
    lines.extend(
        (
            "| Tool ID | Version | Field | Baseline | Candidate |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    for change in changes:
        lines.extend(
            "| "
            f"`{change.tool_id}` | `{change.version}` | `{field.field}` | "
            f"`{field.before}` | `{field.after}` |"
            for field in change.fields
        )
    lines.append("")


def _require_diff(diff: ToolCatalogDiff) -> None:
    if type(diff) is not ToolCatalogDiff:
        raise ToolCatalogDiffError("invalid_type", "diff")


__all__ = [
    "TOOL_CATALOG_DIFF_SCHEMA_VERSION",
    "ToolCatalogChange",
    "ToolCatalogDiff",
    "ToolCatalogDiffError",
    "ToolCatalogFieldChange",
    "diff_tool_catalogs",
    "render_tool_catalog_diff_json",
    "render_tool_catalog_diff_markdown",
]
