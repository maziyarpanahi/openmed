"""Offline SMART-on-FHIR scope comparison helpers.

The helpers in this module compare declared SMART v2 resource scopes with a
synthetic workflow's declared needs. They do not implement OAuth, contact FHIR
servers, or inspect patient records.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SmartScope",
    "SmartScopeAudit",
    "audit_smart_scopes",
    "normalize_smart_scope",
    "parse_smart_scope",
]

_CONTEXT_ORDER = {"patient": 0, "user": 1, "system": 2}
_OPERATION_ORDER = "cruds"
_OPERATION_NAMES = {
    "c": "create",
    "r": "read",
    "u": "update",
    "d": "delete",
    "s": "search",
}
_RESOURCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


@dataclass(frozen=True, order=True)
class SmartScope:
    """A normalized SMART v2 resource scope."""

    context: str
    resource_type: str
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.context not in _CONTEXT_ORDER:
            raise ValueError("SMART scope context must be patient, user, or system")
        if not _RESOURCE_RE.fullmatch(self.resource_type):
            raise ValueError("SMART scope resource type must be alphanumeric")
        if not self.operations:
            raise ValueError("SMART scope operations must not be empty")
        unknown = [op for op in self.operations if op not in _OPERATION_NAMES]
        if unknown:
            raise ValueError(
                f"unsupported SMART scope operation(s): {', '.join(unknown)}"
            )
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("SMART scope operations must be unique")

    @property
    def value(self) -> str:
        """Return the canonical ``context/Resource.ops`` representation."""

        return f"{self.context}/{self.resource_type}.{''.join(self.operations)}"

    def atoms(self) -> frozenset[tuple[str, str, str]]:
        """Return one comparable atom per context, resource, and operation."""

        return frozenset(
            (self.context, self.resource_type, operation)
            for operation in self.operations
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "scope": self.value,
            "context": self.context,
            "resource_type": self.resource_type,
            "operations": [
                {"code": operation, "name": _OPERATION_NAMES[operation]}
                for operation in self.operations
            ],
        }


@dataclass(frozen=True)
class SmartScopeAudit:
    """Comparison between workflow-required and declared SMART scopes."""

    workflow_id: str
    required_scopes: tuple[SmartScope, ...]
    declared_scopes: tuple[SmartScope, ...]
    missing_scopes: tuple[SmartScope, ...]
    excessive_scopes: tuple[SmartScope, ...]

    @property
    def status(self) -> str:
        """Return ``pass`` when declared scopes exactly cover the workflow."""

        if self.missing_scopes:
            return "missing"
        if self.excessive_scopes:
            return "excessive"
        return "pass"

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible audit report."""

        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "required_scopes": [scope.to_dict() for scope in self.required_scopes],
            "declared_scopes": [scope.to_dict() for scope in self.declared_scopes],
            "missing_scopes": [scope.to_dict() for scope in self.missing_scopes],
            "excessive_scopes": [scope.to_dict() for scope in self.excessive_scopes],
        }


def normalize_smart_scope(value: str) -> str:
    """Return a canonical SMART resource scope string."""

    return parse_smart_scope(value).value


def parse_smart_scope(value: str) -> SmartScope:
    """Parse a SMART v2 resource scope.

    Supported examples include ``patient/Observation.r`` and
    ``system/SyntheticEncounter.rs``. Non-resource scopes such as launch or
    offline access are intentionally outside this offline comparison helper.
    """

    raw = value.strip()
    if not raw:
        raise ValueError("SMART scope must not be empty")
    try:
        context_and_resource, operations = raw.split(".", 1)
        context, resource_type = context_and_resource.split("/", 1)
    except ValueError as exc:
        raise ValueError(
            "SMART scope must use context/Resource.operations format"
        ) from exc

    context = context.strip().lower()
    resource_type = resource_type.strip()
    operations = operations.strip().lower()
    if "*" in resource_type or "*" in operations:
        raise ValueError("wildcard SMART scopes are not supported by this helper")

    ordered_operations = tuple(
        operation for operation in _OPERATION_ORDER if operation in operations
    )
    if len(ordered_operations) != len(set(operations)):
        raise ValueError("SMART scope operations must be c, r, u, d, and/or s")
    return SmartScope(context, resource_type, ordered_operations)


def audit_smart_scopes(
    *,
    workflow_id: str,
    required_scopes: Iterable[str],
    declared_scopes: Iterable[str],
) -> SmartScopeAudit:
    """Compare required and declared SMART scopes for one offline workflow."""

    required = _dedupe_scopes(parse_smart_scope(scope) for scope in required_scopes)
    declared = _dedupe_scopes(parse_smart_scope(scope) for scope in declared_scopes)

    required_atoms = _scope_atoms(required)
    declared_atoms = _scope_atoms(declared)
    missing = _scopes_from_atoms(required_atoms - declared_atoms)
    excessive = _scopes_from_atoms(declared_atoms - required_atoms)

    return SmartScopeAudit(
        workflow_id=workflow_id,
        required_scopes=required,
        declared_scopes=declared,
        missing_scopes=missing,
        excessive_scopes=excessive,
    )


def _dedupe_scopes(scopes: Iterable[SmartScope]) -> tuple[SmartScope, ...]:
    return tuple(
        sorted({scope.value: scope for scope in scopes}.values(), key=_sort_key)
    )


def _scope_atoms(scopes: Iterable[SmartScope]) -> frozenset[tuple[str, str, str]]:
    atoms: set[tuple[str, str, str]] = set()
    for scope in scopes:
        atoms.update(scope.atoms())
    return frozenset(atoms)


def _scopes_from_atoms(atoms: Iterable[tuple[str, str, str]]) -> tuple[SmartScope, ...]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for context, resource_type, operation in atoms:
        grouped.setdefault((context, resource_type), set()).add(operation)

    scopes = [
        SmartScope(
            context=context,
            resource_type=resource_type,
            operations=tuple(op for op in _OPERATION_ORDER if op in operations),
        )
        for (context, resource_type), operations in grouped.items()
    ]
    return tuple(sorted(scopes, key=_sort_key))


def _sort_key(scope: SmartScope) -> tuple[int, str, str]:
    return (
        _CONTEXT_ORDER[scope.context],
        scope.resource_type,
        "".join(scope.operations),
    )
