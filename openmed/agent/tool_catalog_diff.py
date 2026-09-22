"""Deterministic, PHI-safe comparison of two agent tool catalogs.

Deployment review needs to see how an agent tool catalog changed without
disclosing local endpoints or credentials. This module compares content-free
:class:`~openmed.agent.tool_inventory.ToolInventory` snapshots by tool
identifier, version, schema digest, capability classes, and side-effect class,
and renders deterministic JSON and Markdown deltas.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .run_summary import _assert_safe_payload
from .tool_inventory import (
    CapabilityClass,
    SideEffectClass,
    ToolEntry,
    ToolInventory,
)

TOOL_CATALOG_DIFF_SCHEMA_VERSION = "openmed.agent.tool_catalog_diff.v1"

_TOOL_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_MAX_TOOLS = 1_024
_MAX_CAPABILITY_CLASSES = 8


class ToolCatalogDiffError(ValueError):
    """Value-free failure for invalid tool-catalog diff input."""


def _validate_tool_id(value: Any) -> str:
    if type(value) is not str or _TOOL_ID_RE.fullmatch(value) is None:
        raise ToolCatalogDiffError("tool_id: invalid_identifier")
    return value


def _validate_version(value: Any, field_name: str) -> str:
    if type(value) is not str or _VERSION_RE.fullmatch(value) is None:
        raise ToolCatalogDiffError(f"{field_name}: invalid_version")
    return value


def _validate_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ToolCatalogDiffError(f"{field_name}: invalid_digest")
    return value


def _capability_class(value: Any) -> CapabilityClass:
    if isinstance(value, CapabilityClass):
        return value
    if type(value) is not str:
        raise ToolCatalogDiffError("capability_classes: invalid_item")
    try:
        return CapabilityClass(value)
    except ValueError:
        pass
    raise ToolCatalogDiffError("capability_classes: unknown_class")


def _validate_capability_classes(
    value: Any, field_name: str
) -> tuple[CapabilityClass, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ToolCatalogDiffError(f"{field_name}: invalid_sequence")
    try:
        items = tuple(value)
    except TypeError:
        raise ToolCatalogDiffError(f"{field_name}: invalid_sequence") from None
    if len(items) > _MAX_CAPABILITY_CLASSES:
        raise ToolCatalogDiffError(f"{field_name}: too_many_items")
    normalized = tuple(_capability_class(item) for item in items)
    if normalized != tuple(sorted(normalized, key=lambda item: item.value)):
        raise ToolCatalogDiffError(f"{field_name}: not_sorted_unique")
    if len(set(normalized)) != len(normalized):
        raise ToolCatalogDiffError(f"{field_name}: duplicate_item")
    return normalized


def _side_effect_class(value: Any, field_name: str) -> SideEffectClass:
    if isinstance(value, SideEffectClass):
        return value
    if type(value) is not str:
        raise ToolCatalogDiffError(f"{field_name}: unknown_class")
    try:
        return SideEffectClass(value)
    except ValueError:
        pass
    raise ToolCatalogDiffError(f"{field_name}: unknown_class")


@dataclass(frozen=True, slots=True)
class ToolChange:
    """Per-field differences for one tool whose entry changed."""

    tool_id: str
    version_before: str
    version_after: str
    schema_digest_before: str
    schema_digest_after: str
    capability_classes_added: tuple[CapabilityClass, ...]
    capability_classes_removed: tuple[CapabilityClass, ...]
    side_effect_class_before: SideEffectClass
    side_effect_class_after: SideEffectClass

    def __post_init__(self) -> None:
        _validate_tool_id(self.tool_id)
        _validate_version(self.version_before, "version_before")
        _validate_version(self.version_after, "version_after")
        _validate_digest(self.schema_digest_before, "schema_digest_before")
        _validate_digest(self.schema_digest_after, "schema_digest_after")
        added = _validate_capability_classes(
            self.capability_classes_added, "capability_classes_added"
        )
        removed = _validate_capability_classes(
            self.capability_classes_removed, "capability_classes_removed"
        )
        if set(added) & set(removed):
            raise ToolCatalogDiffError("capability_classes: added_and_removed")
        before_side_effect = _side_effect_class(
            self.side_effect_class_before, "side_effect_class_before"
        )
        after_side_effect = _side_effect_class(
            self.side_effect_class_after, "side_effect_class_after"
        )
        if (
            self.version_before == self.version_after
            and self.schema_digest_before == self.schema_digest_after
            and not added
            and not removed
            and before_side_effect == after_side_effect
        ):
            raise ToolCatalogDiffError("change: no_field_changed")
        object.__setattr__(self, "capability_classes_added", added)
        object.__setattr__(self, "capability_classes_removed", removed)
        object.__setattr__(self, "side_effect_class_before", before_side_effect)
        object.__setattr__(self, "side_effect_class_after", after_side_effect)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic content-free JSON-compatible data."""
        return {
            "tool_id": self.tool_id,
            "version": {
                "before": self.version_before,
                "after": self.version_after,
            },
            "schema_digest": {
                "before": self.schema_digest_before,
                "after": self.schema_digest_after,
            },
            "capability_classes": {
                "added": [
                    capability.value for capability in self.capability_classes_added
                ],
                "removed": [
                    capability.value for capability in self.capability_classes_removed
                ],
            },
            "side_effect_class": {
                "before": self.side_effect_class_before.value,
                "after": self.side_effect_class_after.value,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCatalogDiff:
    """Signed, content-free differences from one tool catalog to another."""

    tool_ids_added: tuple[str, ...]
    tool_ids_removed: tuple[str, ...]
    tool_ids_unchanged: tuple[str, ...]
    changes: tuple[ToolChange, ...]
    schema_version: str = TOOL_CATALOG_DIFF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != TOOL_CATALOG_DIFF_SCHEMA_VERSION
        ):
            raise ToolCatalogDiffError("schema_version: unsupported_version")

        added = _sorted_unique_ids(self.tool_ids_added, "tool_ids_added")
        removed = _sorted_unique_ids(self.tool_ids_removed, "tool_ids_removed")
        unchanged = _sorted_unique_ids(self.tool_ids_unchanged, "tool_ids_unchanged")
        if set(added) & set(removed):
            raise ToolCatalogDiffError("tool_ids: added_and_removed")
        if set(added) & set(unchanged):
            raise ToolCatalogDiffError("tool_ids: added_and_unchanged")
        if set(removed) & set(unchanged):
            raise ToolCatalogDiffError("tool_ids: removed_and_unchanged")

        if isinstance(self.changes, (str, bytes, bytearray)):
            raise ToolCatalogDiffError("changes: invalid_sequence")
        try:
            changes = tuple(self.changes)
        except TypeError:
            raise ToolCatalogDiffError("changes: invalid_sequence") from None
        for change in changes:
            if not isinstance(change, ToolChange):
                raise ToolCatalogDiffError("changes: invalid_item")
        if changes != tuple(sorted(changes, key=lambda change: change.tool_id)):
            raise ToolCatalogDiffError("changes: not_sorted")
        change_ids = tuple(change.tool_id for change in changes)
        if len(set(change_ids)) != len(change_ids):
            raise ToolCatalogDiffError("changes: duplicate_tool_id")
        if set(change_ids) & (set(added) | set(removed) | set(unchanged)):
            raise ToolCatalogDiffError("changes: overlapping_tool_id")

        object.__setattr__(self, "tool_ids_added", added)
        object.__setattr__(self, "tool_ids_removed", removed)
        object.__setattr__(self, "tool_ids_unchanged", unchanged)
        object.__setattr__(self, "changes", changes)

    @property
    def changed(self) -> bool:
        """Return whether any tool differs between the catalogs."""
        return bool(self.tool_ids_added or self.tool_ids_removed or self.changes)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic content-free JSON-compatible data."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "changed": self.changed,
            "tool_ids": {
                "added": list(self.tool_ids_added),
                "removed": list(self.tool_ids_removed),
                "unchanged": list(self.tool_ids_unchanged),
            },
            "changes": [change.to_dict() for change in self.changes],
        }
        _assert_safe_payload(payload)
        return payload

    def to_json(self) -> str:
        """Return compact, deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_markdown(self) -> str:
        """Return deterministic content-free Markdown."""
        self.to_dict()
        lines = [
            "# Agent Tool Catalog Diff",
            "",
            f"Changed: {'yes' if self.changed else 'no'}",
            "",
            "## Tools",
            "",
            "| Change | Tool |",
            "| --- | --- |",
            *(f"| added | `{value}` |" for value in self.tool_ids_added),
            *(f"| removed | `{value}` |" for value in self.tool_ids_removed),
            *(f"| changed | `{value.tool_id}` |" for value in self.changes),
            *(f"| unchanged | `{value}` |" for value in self.tool_ids_unchanged),
        ]
        return "\n".join(lines) + "\n"


def _sorted_unique_ids(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or type(value) not in (list, tuple):
        raise ToolCatalogDiffError(f"{field_name}: invalid_sequence")
    items = tuple(value)
    if len(items) > _MAX_TOOLS:
        raise ToolCatalogDiffError(f"{field_name}: too_many_items")
    for item in items:
        _validate_tool_id(item)
    if items != tuple(sorted(items)):
        raise ToolCatalogDiffError(f"{field_name}: not_sorted_unique")
    return items


def diff_tool_catalogs(before: ToolInventory, after: ToolInventory) -> ToolCatalogDiff:
    """Compare two validated tool catalogs.

    Args:
        before: The baseline inventory.
        after: The inventory compared against the baseline.

    Returns:
        A :class:`ToolCatalogDiff` describing added, removed, changed, and
        unchanged tools by content-free fields.

    Raises:
        ToolCatalogDiffError: If either input is not a :class:`ToolInventory`.
    """
    if not isinstance(before, ToolInventory):
        raise ToolCatalogDiffError("before: invalid_catalog")
    if not isinstance(after, ToolInventory):
        raise ToolCatalogDiffError("after: invalid_catalog")

    before_by_id = {entry.tool_id: entry for entry in before.entries}
    after_by_id = {entry.tool_id: entry for entry in after.entries}

    before_ids = set(before_by_id)
    after_ids = set(after_by_id)

    added = after_ids - before_ids
    removed = before_ids - after_ids
    common = before_ids & after_ids

    changes: list[ToolChange] = []
    unchanged: list[str] = []
    for tool_id in sorted(common):
        before_entry = before_by_id[tool_id]
        after_entry = after_by_id[tool_id]
        if before_entry == after_entry:
            unchanged.append(tool_id)
            continue
        changes.append(_build_change(tool_id, before_entry, after_entry))

    return ToolCatalogDiff(
        tool_ids_added=tuple(sorted(added)),
        tool_ids_removed=tuple(sorted(removed)),
        tool_ids_unchanged=tuple(unchanged),
        changes=tuple(changes),
    )


def _build_change(
    tool_id: str, before_entry: ToolEntry, after_entry: ToolEntry
) -> ToolChange:
    before_capabilities = set(before_entry.capability_classes)
    after_capabilities = set(after_entry.capability_classes)
    return ToolChange(
        tool_id=tool_id,
        version_before=before_entry.version,
        version_after=after_entry.version,
        schema_digest_before=before_entry.schema_digest,
        schema_digest_after=after_entry.schema_digest,
        capability_classes_added=tuple(
            sorted(after_capabilities - before_capabilities, key=lambda c: c.value)
        ),
        capability_classes_removed=tuple(
            sorted(before_capabilities - after_capabilities, key=lambda c: c.value)
        ),
        side_effect_class_before=before_entry.side_effect_class,
        side_effect_class_after=after_entry.side_effect_class,
    )


__all__ = [
    "TOOL_CATALOG_DIFF_SCHEMA_VERSION",
    "ToolCatalogDiff",
    "ToolCatalogDiffError",
    "ToolChange",
    "diff_tool_catalogs",
]
