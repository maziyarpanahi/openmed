"""Deterministic, PHI-safe inventory of registered agent tools.

Operators need to inspect the local agent tool surface without revealing
endpoints, credentials, arguments, or clinical data. This module renders a
content-free inventory from each tool's identifier, version, closed capability
classes, closed side-effect class, and schema digest, and rejects any
free-form or identifying metadata before it can reach output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from .run_summary import _assert_safe_payload

TOOL_INVENTORY_SCHEMA_VERSION = "openmed.agent.tool_inventory.v1"

_TOOL_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_MAX_TOOLS = 1_024
_MAX_CAPABILITY_CLASSES = 8


class ToolInventoryError(ValueError):
    """Value-free failure for invalid tool-inventory input."""


class CapabilityClass(str, Enum):
    """Closed set of agent tool capability classes."""

    READ = "read"
    WRITE = "write"
    QUERY = "query"
    COMPUTE = "compute"
    NOTIFY = "notify"


class SideEffectClass(str, Enum):
    """Closed set of agent tool side-effect classes."""

    NONE = "none"
    READ_ONLY = "read_only"
    STATE_MUTATION = "state_mutation"
    EXTERNAL = "external"


def _validate_tool_id(value: Any) -> str:
    if type(value) is not str or _TOOL_ID_RE.fullmatch(value) is None:
        raise ToolInventoryError("tool_id: invalid_identifier")
    return value


def _validate_version(value: Any) -> str:
    if type(value) is not str or _VERSION_RE.fullmatch(value) is None:
        raise ToolInventoryError("version: invalid_version")
    return value


def _validate_schema_digest(value: Any) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ToolInventoryError("schema_digest: invalid_digest")
    return value


def _capability_class(value: Any) -> CapabilityClass:
    if isinstance(value, CapabilityClass):
        return value
    if type(value) is not str:
        raise ToolInventoryError("capability_classes: invalid_item")
    try:
        return CapabilityClass(value)
    except ValueError:
        pass
    raise ToolInventoryError("capability_classes: unknown_class")


def _validate_capability_classes(value: Any) -> tuple[CapabilityClass, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ToolInventoryError("capability_classes: invalid_sequence")
    try:
        items = tuple(value)
    except TypeError:
        raise ToolInventoryError("capability_classes: invalid_sequence") from None
    if not items:
        raise ToolInventoryError("capability_classes: empty")
    if len(items) > _MAX_CAPABILITY_CLASSES:
        raise ToolInventoryError("capability_classes: too_many_items")
    normalized = tuple(_capability_class(item) for item in items)
    if normalized != tuple(sorted(normalized, key=lambda item: item.value)):
        raise ToolInventoryError("capability_classes: not_sorted_unique")
    if len(set(normalized)) != len(normalized):
        raise ToolInventoryError("capability_classes: duplicate_item")
    return normalized


def _validate_side_effect_class(value: Any) -> SideEffectClass:
    if isinstance(value, SideEffectClass):
        return value
    if type(value) is not str:
        raise ToolInventoryError("side_effect_class: unknown_class")
    try:
        return SideEffectClass(value)
    except ValueError:
        pass
    raise ToolInventoryError("side_effect_class: unknown_class")


@dataclass(frozen=True, slots=True)
class ToolEntry:
    """Content-free description of one registered agent tool."""

    tool_id: str
    version: str
    capability_classes: tuple[CapabilityClass, ...]
    side_effect_class: SideEffectClass
    schema_digest: str

    def __post_init__(self) -> None:
        _validate_tool_id(self.tool_id)
        _validate_version(self.version)
        normalized_capabilities = _validate_capability_classes(self.capability_classes)
        normalized_side_effect = _validate_side_effect_class(self.side_effect_class)
        _validate_schema_digest(self.schema_digest)
        object.__setattr__(self, "capability_classes", normalized_capabilities)
        object.__setattr__(self, "side_effect_class", normalized_side_effect)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic content-free JSON-compatible data."""
        return {
            "tool_id": self.tool_id,
            "version": self.version,
            "capability_classes": [
                capability.value for capability in self.capability_classes
            ],
            "side_effect_class": self.side_effect_class.value,
            "schema_digest": self.schema_digest,
        }


@dataclass(frozen=True, slots=True)
class ToolInventory:
    """Deterministic, sorted inventory of content-free tool entries."""

    entries: tuple[ToolEntry, ...]
    schema_version: str = TOOL_INVENTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != TOOL_INVENTORY_SCHEMA_VERSION
        ):
            raise ToolInventoryError("schema_version: unsupported_version")
        if isinstance(self.entries, (str, bytes, bytearray)):
            raise ToolInventoryError("entries: invalid_sequence")
        try:
            entries = tuple(self.entries)
        except TypeError:
            raise ToolInventoryError("entries: invalid_sequence") from None
        if len(entries) > _MAX_TOOLS:
            raise ToolInventoryError("entries: too_many_items")
        for entry in entries:
            if not isinstance(entry, ToolEntry):
                raise ToolInventoryError("entries: invalid_item")
        if entries != tuple(sorted(entries, key=lambda entry: entry.tool_id)):
            raise ToolInventoryError("entries: not_sorted")
        tool_ids = tuple(entry.tool_id for entry in entries)
        if len(set(tool_ids)) != len(tool_ids):
            raise ToolInventoryError("entries: duplicate_tool_id")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def from_entries(cls, entries: Iterable[ToolEntry]) -> "ToolInventory":
        """Sort validated tool entries into a deterministic inventory."""
        if isinstance(entries, (str, bytes, bytearray)):
            raise ToolInventoryError("entries: invalid_sequence")
        try:
            iterator = iter(entries)
        except TypeError:
            raise ToolInventoryError("entries: invalid_sequence") from None

        collected: list[ToolEntry] = []
        for entry in iterator:
            if not isinstance(entry, ToolEntry):
                raise ToolInventoryError("entries: invalid_item")
            collected.append(entry)
            if len(collected) > _MAX_TOOLS:
                raise ToolInventoryError("entries: too_many_items")
        return cls(entries=tuple(sorted(collected, key=lambda item: item.tool_id)))

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic content-free JSON-compatible data."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "tool_count": len(self.entries),
            "tools": [entry.to_dict() for entry in self.entries],
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
            "# Agent Tool Inventory",
            "",
            "| Tool | Version | Capabilities | Side effect | Schema digest |",
            "| --- | --- | --- | --- | --- |",
        ]
        for entry in self.entries:
            capabilities = ", ".join(
                f"`{capability.value}`" for capability in entry.capability_classes
            )
            lines.append(
                f"| `{entry.tool_id}` | `{entry.version}` | {capabilities} | "
                f"`{entry.side_effect_class.value}` | `{entry.schema_digest}` |"
            )
        return "\n".join(lines) + "\n"


__all__ = [
    "TOOL_INVENTORY_SCHEMA_VERSION",
    "CapabilityClass",
    "SideEffectClass",
    "ToolEntry",
    "ToolInventory",
    "ToolInventoryError",
]
