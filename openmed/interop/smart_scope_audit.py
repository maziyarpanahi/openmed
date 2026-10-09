"""Offline SMART-on-FHIR scope comparison helpers.

The helpers in this module compare declared SMART v2 resource scopes with a
synthetic workflow's declared needs. They do not implement OAuth, contact FHIR
servers, or inspect patient records.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

__all__ = [
    "SmartScope",
    "SmartScopeAudit",
    "SmartGrantedScopeAudit",
    "audit_granted_smart_scopes",
    "smart_scopes_cover",
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


@dataclass(frozen=True)
class SmartGrantedScopeAudit:
    """Value-free comparison of requested and actually granted permissions.

    Args:
        requested_count: Number of distinct requested scope tokens.
        granted_count: Number of distinct granted scope tokens.
        narrowed: Whether the requested permission union is not fully granted.
        expanded: Whether the response grants permissions not requested.
    """

    requested_count: int
    granted_count: int
    narrowed: bool
    expanded: bool

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or not 0 <= n <= 128
            for n in (self.requested_count, self.granted_count)
        ) or any(type(flag) is not bool for flag in (self.narrowed, self.expanded)):
            raise ValueError("Invalid SMART scope finding.")

    def to_dict(self) -> dict[str, int | bool]:
        """Return only counts and closed findings, never scope/filter values."""
        return {
            "requested_count": self.requested_count,
            "granted_count": self.granted_count,
            "narrowed": self.narrowed,
            "expanded": self.expanded,
        }


_OAUTH_SCOPE_TOKEN = re.compile(r"[\x21\x23-\x5b\x5d-\x7e]{1,512}")
_SMART_GRANT_SCOPE = re.compile(
    r"(patient|user|system)/([A-Z][A-Za-z0-9]*|\*)\."
    r"(read|write|\*|[cruds]+)(\?[^\s]+)?"
)
_LEGACY_OPERATIONS = {"read": "rs", "write": "cud", "*": "cruds"}


def _granted_scope_tokens(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("Invalid SMART scope collection.")
    try:
        tokens = tuple(islice(values, 129))
    except Exception:
        raise ValueError("Invalid SMART scope collection.") from None
    if len(tokens) > 128 or any(
        type(token) is not str or _OAUTH_SCOPE_TOKEN.fullmatch(token) is None
        for token in tokens
    ):
        raise ValueError("Invalid SMART scope collection.")
    for token in tokens:
        match = _SMART_GRANT_SCOPE.fullmatch(token)
        if token.startswith(("patient/", "user/", "system/")):
            if match is None:
                raise ValueError("Invalid SMART resource scope.")
            operations = match[3]
            if operations not in _LEGACY_OPERATIONS and len(set(operations)) != len(
                operations
            ):
                raise ValueError("Invalid SMART resource scope.")
    return tuple(sorted(set(tokens)))


def _permission_covered(scope: str, declared: tuple[str, ...]) -> bool:
    requested = _SMART_GRANT_SCOPE.fullmatch(scope)
    if requested is None:
        return scope in declared
    context, resource, operations, query = requested.groups()
    wanted = set(_LEGACY_OPERATIONS.get(operations, operations))
    covered: set[str] = set()
    for token in declared:
        parent = _SMART_GRANT_SCOPE.fullmatch(token)
        if parent is None:
            continue
        parent_context, parent_resource, parent_ops, parent_query = parent.groups()
        if (
            parent_context == context
            and parent_resource in (resource, "*")
            and (parent_query is None or parent_query == query)
        ):
            covered.update(_LEGACY_OPERATIONS.get(parent_ops, parent_ops))
    return wanted <= covered


def smart_scopes_cover(required: Iterable[str], granted: Iterable[str]) -> bool:
    """Check SMART permission unions without exposing scope/filter values.

    SMART v1 read/write and v2 CRUDS forms, resource wildcards and exact query
    constraints are supported. Query predicates are never evaluated: a grant
    with a different filter cannot satisfy the requested filter. Other OAuth
    scopes require exact token equality. This check does not authorize an action.
    """
    wanted = _granted_scope_tokens(required)
    actual = _granted_scope_tokens(granted)
    return all(_permission_covered(scope, actual) for scope in wanted)


def audit_granted_smart_scopes(
    *, requested_scopes: Iterable[str], granted_scopes: Iterable[str]
) -> SmartGrantedScopeAudit:
    """Compare scope unions and return counts plus narrowing/expansion flags."""
    requested = _granted_scope_tokens(requested_scopes)
    granted = _granted_scope_tokens(granted_scopes)
    return SmartGrantedScopeAudit(
        requested_count=len(requested),
        granted_count=len(granted),
        narrowed=not smart_scopes_cover(requested, granted),
        expanded=not smart_scopes_cover(granted, requested),
    )
