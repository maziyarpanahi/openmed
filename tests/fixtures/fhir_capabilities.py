"""Deterministic, metadata-free FHIR R4 CapabilityStatement fixtures.

These builders model one write-preflight capability at a time. They never
include endpoints, credentials, server software, organizations, or clinical
resources, so tests can use them entirely offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, TypeAlias

FHIR_R4_VERSION = "4.0.1"
SUPPORTED_RESOURCE_TYPE = "Observation"
UNSUPPORTED_RESOURCE_TYPE = "MedicationRequest"

CapabilityStatement: TypeAlias = dict[str, Any]
PathSegment: TypeAlias = str | int

__all__ = [
    "FHIR_R4_VERSION",
    "SUPPORTED_RESOURCE_TYPE",
    "UNSUPPORTED_RESOURCE_TYPE",
    "build_capability_statement",
    "build_conditional_capability_statement",
    "build_create_capability_statement",
    "build_read_only_capability_statement",
    "build_transaction_capability_statement",
    "build_unsupported_resource_capability_statement",
    "build_update_capability_statement",
    "with_invalid_capability_field",
    "without_capability_field",
]


def build_capability_statement(
    *,
    resource_type: str | None = SUPPORTED_RESOURCE_TYPE,
    resource_interactions: Sequence[str] = (),
    conditional_create: bool = False,
    conditional_update: bool = False,
    system_interactions: Sequence[str] = (),
) -> CapabilityStatement:
    """Build a minimal synthetic FHIR R4 ``CapabilityStatement``.

    Args:
        resource_type: Resource type to declare, or ``None`` for a
            system-interaction-only statement.
        resource_interactions: Resource-level FHIR REST interaction codes.
        conditional_create: Whether conditional create is supported.
        conditional_update: Whether conditional update is supported.
        system_interactions: System-level FHIR REST interaction codes.

    Returns:
        A new mutable dictionary with deterministic ordering and values.
    """

    rest: CapabilityStatement = {"mode": "server"}
    if resource_type is not None:
        resource: CapabilityStatement = {
            "type": resource_type,
            "interaction": [
                {"code": interaction} for interaction in resource_interactions
            ],
            "conditionalCreate": conditional_create,
            "conditionalUpdate": conditional_update,
        }
        rest["resource"] = [resource]
    if system_interactions:
        rest["interaction"] = [
            {"code": interaction} for interaction in system_interactions
        ]

    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": "2000-01-01T00:00:00Z",
        "kind": "capability",
        "fhirVersion": FHIR_R4_VERSION,
        "format": ["json"],
        "rest": [rest],
    }


def build_read_only_capability_statement() -> CapabilityStatement:
    """Build a statement that accepts reads but rejects planned writes."""

    return build_capability_statement(resource_interactions=("read", "search-type"))


def build_create_capability_statement() -> CapabilityStatement:
    """Build a statement that accepts an ordinary create plan."""

    return build_capability_statement(resource_interactions=("create",))


def build_update_capability_statement() -> CapabilityStatement:
    """Build a statement that accepts an ordinary update plan."""

    return build_capability_statement(resource_interactions=("update",))


def build_conditional_capability_statement() -> CapabilityStatement:
    """Build a statement that accepts conditional create and update plans."""

    return build_capability_statement(
        resource_interactions=("create", "update"),
        conditional_create=True,
        conditional_update=True,
    )


def build_transaction_capability_statement() -> CapabilityStatement:
    """Build a statement that accepts a system-level transaction plan."""

    return build_capability_statement(
        resource_type=None,
        system_interactions=("transaction",),
    )


def build_unsupported_resource_capability_statement() -> CapabilityStatement:
    """Build a statement that rejects plans for ``UNSUPPORTED_RESOURCE_TYPE``."""

    return build_capability_statement(resource_interactions=("read",))


def without_capability_field(
    statement: CapabilityStatement,
    path: Sequence[PathSegment],
) -> CapabilityStatement:
    """Return a copy with the existing field at ``path`` removed.

    Args:
        statement: CapabilityStatement fixture to copy and corrupt.
        path: Non-empty sequence of mapping keys and list indexes.

    Returns:
        A deep copy with exactly one field removed.

    Raises:
        KeyError: If the path is empty, malformed, or does not exist.
        IndexError: If a list index is out of range.
    """

    corrupted = deepcopy(statement)
    parent, final = _resolve_parent(corrupted, path)
    if isinstance(parent, dict) and isinstance(final, str):
        del parent[final]
    elif isinstance(parent, list) and isinstance(final, int):
        del parent[final]
    else:
        raise KeyError("CapabilityStatement path does not match its container")
    return corrupted


def with_invalid_capability_field(
    statement: CapabilityStatement,
    path: Sequence[PathSegment],
    value: Any,
) -> CapabilityStatement:
    """Return a copy with the existing field at ``path`` replaced by ``value``.

    The value is copied too, which keeps subsequent test mutations isolated.

    Args:
        statement: CapabilityStatement fixture to copy and corrupt.
        path: Non-empty sequence of mapping keys and list indexes.
        value: Deliberately invalid replacement value.

    Returns:
        A deep copy with exactly one field replaced.

    Raises:
        KeyError: If the path is empty, malformed, or does not exist.
        IndexError: If a list index is out of range.
    """

    corrupted = deepcopy(statement)
    parent, final = _resolve_parent(corrupted, path)
    replacement = deepcopy(value)
    if isinstance(parent, dict) and isinstance(final, str):
        if final not in parent:
            raise KeyError(final)
        parent[final] = replacement
    elif isinstance(parent, list) and isinstance(final, int):
        parent[final] = replacement
    else:
        raise KeyError("CapabilityStatement path does not match its container")
    return corrupted


def _resolve_parent(
    statement: CapabilityStatement,
    path: Sequence[PathSegment],
) -> tuple[CapabilityStatement | list[Any], PathSegment]:
    if isinstance(path, (str, bytes)) or not path:
        raise KeyError("CapabilityStatement path must be a non-empty sequence")

    current: Any = statement
    for segment in path[:-1]:
        if isinstance(current, dict) and isinstance(segment, str):
            current = current[segment]
        elif isinstance(current, list) and isinstance(segment, int):
            current = current[segment]
        else:
            raise KeyError("CapabilityStatement path does not match its container")
    return current, path[-1]
