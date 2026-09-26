"""Offline SMART-on-FHIR scope comparison helpers.

The helpers in this module compare declared SMART v2 resource scopes with a
synthetic workflow's declared needs. They do not implement OAuth, contact FHIR
servers, or inspect patient records.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "SmartScope",
    "SmartScopeAudit",
    "audit_smart_scopes",
    "normalize_smart_scope",
    "parse_smart_scope",
    "PreflightReason",
    "PreflightScope",
    "PreflightScopeAudit",
    "PreflightScopeFinding",
    "audit_smart_scope_preflight",
    "parse_smart_scope_preflight",
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


# This stricter preflight contract also covers launch and wildcard scopes. It is
# additive: the existing exact-resource audit above retains its public shape.
_PREFLIGHT_RESOURCE_RE = re.compile(r"[A-Z][A-Za-z0-9]{0,63}\Z")
_PREFLIGHT_ACCESS_ORDER = "cruds"
_PREFLIGHT_CONTEXTS = frozenset({"patient", "user", "system"})
_PREFLIGHT_LAUNCH_SCOPES = frozenset({"launch", "launch/patient", "launch/encounter"})


@dataclass(frozen=True, slots=True)
class PreflightScope:
    """A validated clinical or launch-context SMART scope for preflight."""

    name: str
    context: str
    resource_type: str | None
    access: frozenset[str]

    @property
    def is_launch(self) -> bool:
        """Return whether this scope requests launch context."""

        return self.name in _PREFLIGHT_LAUNCH_SCOPES


class PreflightReason(str, Enum):
    """Stable reason codes for differences in declared permissions."""

    MISSING_SCOPE = "missing_scope"
    EXCESSIVE_SCOPE = "excessive_scope"
    OVERBROAD_RESOURCE = "overbroad_resource"


@dataclass(frozen=True, slots=True)
class PreflightScopeFinding:
    """A difference containing only a scope name and optional resource type."""

    reason_code: PreflightReason
    scope: str
    resource_type: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Return content-free, JSON-compatible finding fields."""

        return {
            "reason_code": self.reason_code.value,
            "scope": self.scope,
            "resource_type": self.resource_type,
        }


@dataclass(frozen=True, slots=True)
class PreflightScopeAudit:
    """Deterministic preflight missing and excessive scope findings."""

    missing_scopes: tuple[PreflightScopeFinding, ...]
    excessive_scopes: tuple[PreflightScopeFinding, ...]

    @property
    def is_least_privilege(self) -> bool:
        """Return whether requested scopes match declared workflow needs."""

        return not (self.missing_scopes or self.excessive_scopes)

    def to_dict(self) -> dict[str, list[dict[str, str | None]]]:
        """Return findings without credentials, endpoints, or workflow data."""

        return {
            "missing_scopes": [item.to_dict() for item in self.missing_scopes],
            "excessive_scopes": [item.to_dict() for item in self.excessive_scopes],
        }


def parse_smart_scope_preflight(value: str) -> PreflightScope:
    """Normalize one SMART v2 clinical or launch-context scope.

    Raises:
        ValueError: The scope is unsupported or malformed. Input is never
            repeated in the exception, since it could contain a secret.
    """

    if not isinstance(value, str):
        raise ValueError("SMART scope must be a string")
    name = value.strip()
    if name in _PREFLIGHT_LAUNCH_SCOPES:
        return PreflightScope(name, "launch", None, frozenset())
    try:
        context, remainder = name.split("/", 1)
        resource_type, access = remainder.split(".", 1)
    except ValueError:
        raise ValueError("unsupported SMART scope format") from None
    if context not in _PREFLIGHT_CONTEXTS:
        raise ValueError("unsupported SMART scope context")
    if resource_type != "*" and not _PREFLIGHT_RESOURCE_RE.fullmatch(resource_type):
        raise ValueError("invalid SMART resource type")
    if (
        not access
        or len(set(access)) != len(access)
        or any(operation not in _PREFLIGHT_ACCESS_ORDER for operation in access)
    ):
        raise ValueError("invalid SMART v2 access operations")
    ordered = "".join(
        operation for operation in _PREFLIGHT_ACCESS_ORDER if operation in access
    )
    return PreflightScope(
        f"{context}/{resource_type}.{ordered}",
        context,
        resource_type,
        frozenset(ordered),
    )


def audit_smart_scope_preflight(
    *, required_scopes: Iterable[str], requested_scopes: Iterable[str]
) -> PreflightScopeAudit:
    """Compare a content-free workflow declaration with requested permissions.

    A wildcard satisfies specific resources for missing-scope detection, but
    is reported as overbroad when the workflow lists only specific resources.
    The caller must stop or review whenever either finding list is nonempty.
    """

    required = _parse_preflight_scopes(required_scopes)
    requested = _parse_preflight_scopes(requested_scopes)
    missing: list[PreflightScopeFinding] = []
    excessive: list[PreflightScopeFinding] = []

    for scope in required:
        if scope.is_launch:
            if not any(item.name == scope.name for item in requested):
                missing.append(_preflight_finding(PreflightReason.MISSING_SCOPE, scope))
            continue
        available = frozenset().union(
            *(
                item.access
                for item in requested
                if not item.is_launch
                and item.context == scope.context
                and (
                    item.resource_type == scope.resource_type
                    or item.resource_type == "*"
                )
            )
        )
        absent = scope.access - available
        if absent:
            missing.append(
                _preflight_finding(PreflightReason.MISSING_SCOPE, scope, access=absent)
            )

    for scope in requested:
        if scope.is_launch:
            if not any(item.name == scope.name for item in required):
                excessive.append(
                    _preflight_finding(PreflightReason.EXCESSIVE_SCOPE, scope)
                )
            continue
        matching = [
            item
            for item in required
            if not item.is_launch
            and item.context == scope.context
            and (item.resource_type == scope.resource_type or item.resource_type == "*")
        ]
        if scope.resource_type == "*" and not any(
            item.resource_type == "*" for item in matching
        ):
            excessive.append(
                _preflight_finding(PreflightReason.OVERBROAD_RESOURCE, scope)
            )
            continue
        needed = frozenset().union(*(item.access for item in matching))
        extra = scope.access - needed
        if extra:
            excessive.append(
                _preflight_finding(PreflightReason.EXCESSIVE_SCOPE, scope, access=extra)
            )

    return PreflightScopeAudit(tuple(missing), tuple(excessive))


def _parse_preflight_scopes(values: Iterable[str]) -> tuple[PreflightScope, ...]:
    if isinstance(values, str):
        raise ValueError("scopes must be an iterable of names")
    scopes = {scope.name: scope for scope in map(parse_smart_scope_preflight, values)}
    return tuple(scopes[name] for name in sorted(scopes))


def _preflight_finding(
    reason: PreflightReason,
    scope: PreflightScope,
    *,
    access: frozenset[str] | None = None,
) -> PreflightScopeFinding:
    name = scope.name
    if access is not None:
        ordered = "".join(op for op in _PREFLIGHT_ACCESS_ORDER if op in access)
        name = f"{scope.context}/{scope.resource_type}.{ordered}"
    return PreflightScopeFinding(reason, name, scope.resource_type)
