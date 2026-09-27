"""Deterministic, content-free inventories of registered agent tools.

The inventory intentionally accepts only canonical governance identifiers,
semantic versions, a closed side-effect vocabulary, and SHA-256 schema
digests. Endpoints, credentials, arguments, examples, descriptions, and other
free-form metadata are not part of the record contract and fail validation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Any, Final

from openmed.agent.identifiers import CapabilityId, GovernanceIdError, ToolId

TOOL_INVENTORY_SCHEMA_VERSION: Final = "openmed.agent.tool_inventory.v1"
MAX_TOOL_INVENTORY_RECORDS: Final = 10_000
MAX_TOOL_VERSION_LENGTH: Final = 64

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SEMVER_RE = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
_RECORD_FIELDS = frozenset(
    {
        "tool_id",
        "version",
        "capability_class",
        "side_effect_class",
        "schema_digest",
    }
)
_INVENTORY_FIELDS = frozenset({"schema_version", "tools"})


class ToolInventoryError(ValueError):
    """Raised when inventory input violates the content-free contract.

    Error text contains only a fixed field name and stable reason code. The
    rejected value is never retained or echoed.
    """

    def __init__(self, code: str, field_name: str) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(f"{field_name}: {code}")


class SideEffectClass(str, Enum):
    """Closed mutation-risk classes for registered tools.

    ``NONE`` is for pure local computation. ``READ_ONLY`` may inspect a local
    or remote resource but cannot mutate it. Write classes distinguish replay-
    safe, replay-unsafe, and explicitly destructive operations.
    """

    NONE = "none"
    READ_ONLY = "read-only"
    IDEMPOTENT_WRITE = "idempotent-write"
    NON_IDEMPOTENT_WRITE = "non-idempotent-write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True, repr=False)
class ToolInventoryRecord:
    """Content-free metadata for one registered agent tool version."""

    tool_id: str
    version: str
    capability_class: str
    side_effect_class: SideEffectClass
    schema_digest: str

    def __post_init__(self) -> None:
        try:
            parsed_tool = ToolId.parse(self.tool_id)
        except GovernanceIdError as exc:
            raise ToolInventoryError("invalid_identifier", "tool_id") from exc
        if parsed_tool.version is not None:
            raise ToolInventoryError("version_must_be_separate", "tool_id")

        if (
            type(self.version) is not str
            or len(self.version) > MAX_TOOL_VERSION_LENGTH
            or _SEMVER_RE.fullmatch(self.version) is None
        ):
            raise ToolInventoryError("invalid_semver", "version")

        try:
            CapabilityId.parse(self.capability_class)
        except GovernanceIdError as exc:
            raise ToolInventoryError("invalid_identifier", "capability_class") from exc

        if type(self.side_effect_class) is not SideEffectClass:
            raise ToolInventoryError("invalid_class", "side_effect_class")
        if (
            type(self.schema_digest) is not str
            or _DIGEST_RE.fullmatch(self.schema_digest) is None
        ):
            raise ToolInventoryError("invalid_digest", "schema_digest")

    def to_dict(self) -> dict[str, str]:
        """Return the fixed, content-free record projection."""

        return {
            "tool_id": self.tool_id,
            "version": self.version,
            "capability_class": self.capability_class,
            "side_effect_class": self.side_effect_class.value,
            "schema_digest": self.schema_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolInventoryRecord":
        """Parse a record while rejecting every non-inventory field."""

        values = _snapshot_mapping(payload, field_name="tool")
        if set(values) != _RECORD_FIELDS:
            raise ToolInventoryError("invalid_fields", "tool")
        try:
            side_effect_class = SideEffectClass(values["side_effect_class"])
        except (TypeError, ValueError):
            raise ToolInventoryError("invalid_class", "side_effect_class") from None
        return cls(
            tool_id=values["tool_id"],
            version=values["version"],
            capability_class=values["capability_class"],
            side_effect_class=side_effect_class,
            schema_digest=values["schema_digest"],
        )

    def __repr__(self) -> str:
        """Return a representation that cannot expose rejected metadata."""

        return "ToolInventoryRecord(<content-free>)"


@dataclass(frozen=True, slots=True, repr=False)
class ToolInventory:
    """A deterministically ordered snapshot of registered agent tools."""

    tools: tuple[ToolInventoryRecord, ...]
    schema_version: str = TOOL_INVENTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_INVENTORY_SCHEMA_VERSION:
            raise ToolInventoryError("unsupported_version", "schema_version")
        if type(self.tools) is not tuple or not all(
            type(record) is ToolInventoryRecord for record in self.tools
        ):
            raise ToolInventoryError("invalid_records", "tools")

        if len(self.tools) > MAX_TOOL_INVENTORY_RECORDS:
            raise ToolInventoryError("too_many_records", "tools")
        ordered = tuple(sorted(self.tools, key=_record_sort_key))
        keys = [(record.tool_id, record.version) for record in ordered]
        if len(keys) != len(set(keys)):
            raise ToolInventoryError("duplicate_record", "tools")
        object.__setattr__(self, "tools", ordered)

    @classmethod
    def from_records(cls, records: Iterable[ToolInventoryRecord]) -> "ToolInventory":
        """Build an inventory from typed records without executing any tool."""

        try:
            materialized = tuple(islice(records, MAX_TOOL_INVENTORY_RECORDS + 1))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ToolInventoryError("unreadable_records", "tools") from None
        if len(materialized) > MAX_TOOL_INVENTORY_RECORDS:
            raise ToolInventoryError("too_many_records", "tools")
        return cls(materialized)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolInventory":
        """Parse a fixed inventory mapping and reject content-bearing fields."""

        values = _snapshot_mapping(payload, field_name="inventory")
        if set(values) != _INVENTORY_FIELDS:
            raise ToolInventoryError("invalid_fields", "inventory")
        if values["schema_version"] != TOOL_INVENTORY_SCHEMA_VERSION:
            raise ToolInventoryError("unsupported_version", "schema_version")
        tools = values["tools"]
        if type(tools) is not list:
            raise ToolInventoryError("invalid_records", "tools")
        return cls.from_records(ToolInventoryRecord.from_dict(item) for item in tools)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible inventory projection."""

        return {
            "schema_version": self.schema_version,
            "tools": [record.to_dict() for record in self.tools],
        }

    def to_json(self) -> str:
        """Render byte-stable canonical JSON."""

        return render_tool_inventory_json(self)

    def to_markdown(self) -> str:
        """Render deterministic Markdown without runtime configuration."""

        return render_tool_inventory_markdown(self)

    def __repr__(self) -> str:
        """Return a content-free representation."""

        return f"ToolInventory(tool_count={len(self.tools)})"


def render_tool_inventory_json(inventory: ToolInventory) -> str:
    """Render a tool inventory as deterministic compact JSON."""

    _require_inventory(inventory)
    return json.dumps(
        inventory.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def render_tool_inventory_markdown(inventory: ToolInventory) -> str:
    """Render a tool inventory as deterministic content-free Markdown."""

    _require_inventory(inventory)
    lines = [
        "# Agent tool inventory",
        "",
        f"Schema: `{inventory.schema_version}`",
        "",
        f"Registered tools: {len(inventory.tools)}",
        "",
    ]
    if not inventory.tools:
        lines.append("_No registered tools._")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Tool ID | Version | Capability class | Side-effect class | Schema digest |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        "| "
        f"`{record.tool_id}` | `{record.version}` | "
        f"`{record.capability_class}` | `{record.side_effect_class.value}` | "
        f"`{record.schema_digest}` |"
        for record in inventory.tools
    )
    return "\n".join(lines) + "\n"


def _snapshot_mapping(payload: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ToolInventoryError("invalid_type", field_name)
    try:
        items = list(islice(payload.items(), len(_RECORD_FIELDS) + 1))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ToolInventoryError("unreadable_mapping", field_name) from None
    if len(items) > len(_RECORD_FIELDS):
        raise ToolInventoryError("invalid_fields", field_name)

    result: dict[str, Any] = {}
    for item in items:
        if type(item) not in {tuple, list} or len(item) != 2:
            raise ToolInventoryError("invalid_fields", field_name)
        key, value = item
        if type(key) is not str or key in result:
            raise ToolInventoryError("invalid_fields", field_name)
        result[key] = value
    return result


def _require_inventory(inventory: ToolInventory) -> None:
    if type(inventory) is not ToolInventory:
        raise ToolInventoryError("invalid_type", "inventory")


def _record_sort_key(record: ToolInventoryRecord) -> tuple[str, tuple[int, int, int]]:
    major, minor, patch = (int(part) for part in record.version.split("."))
    return record.tool_id, (major, minor, patch)


__all__ = [
    "TOOL_INVENTORY_SCHEMA_VERSION",
    "MAX_TOOL_INVENTORY_RECORDS",
    "MAX_TOOL_VERSION_LENGTH",
    "SideEffectClass",
    "ToolInventory",
    "ToolInventoryError",
    "ToolInventoryRecord",
    "render_tool_inventory_json",
    "render_tool_inventory_markdown",
]
