"""Offline, content-free comparison of SMART v2 scopes with workflow needs."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

_RESOURCE_RE = re.compile(r"[A-Z][A-Za-z0-9]{0,63}\Z")
_ACCESS_ORDER = "cruds"
_CONTEXTS = frozenset({"patient", "user", "system"})
_LAUNCH_SCOPES = frozenset({"launch", "launch/patient", "launch/encounter"})


@dataclass(frozen=True, slots=True)
class SmartScope:
    """A validated clinical or launch-context SMART scope."""

    name: str
    context: str
    resource_type: str | None
    access: frozenset[str]

    @property
    def is_launch(self) -> bool:
        """Return whether this scope requests launch context."""

        return self.name in _LAUNCH_SCOPES


class ScopeReason(str, Enum):
    """Stable reason codes for differences in declared permissions."""

    MISSING_SCOPE = "missing_scope"
    EXCESSIVE_SCOPE = "excessive_scope"
    OVERBROAD_RESOURCE = "overbroad_resource"


@dataclass(frozen=True, slots=True)
class ScopeFinding:
    """A difference containing only a scope name and optional resource type."""

    reason_code: ScopeReason
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
class SmartScopeAudit:
    """Deterministic missing and excessive scope findings."""

    missing_scopes: tuple[ScopeFinding, ...]
    excessive_scopes: tuple[ScopeFinding, ...]

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


def parse_smart_scope(value: str) -> SmartScope:
    """Normalize one SMART v2 clinical or launch-context scope.

    Raises:
        ValueError: The scope is unsupported or malformed. Input is never
            repeated in the exception, since it could contain a secret.
    """

    if not isinstance(value, str):
        raise ValueError("SMART scope must be a string")
    name = value.strip()
    if name in _LAUNCH_SCOPES:
        return SmartScope(name, "launch", None, frozenset())
    try:
        context, remainder = name.split("/", 1)
        resource_type, access = remainder.split(".", 1)
    except ValueError:
        raise ValueError("unsupported SMART scope format") from None
    if context not in _CONTEXTS:
        raise ValueError("unsupported SMART scope context")
    if resource_type != "*" and not _RESOURCE_RE.fullmatch(resource_type):
        raise ValueError("invalid SMART resource type")
    if (
        not access
        or len(set(access)) != len(access)
        or any(operation not in _ACCESS_ORDER for operation in access)
    ):
        raise ValueError("invalid SMART v2 access operations")
    ordered = "".join(operation for operation in _ACCESS_ORDER if operation in access)
    return SmartScope(
        f"{context}/{resource_type}.{ordered}",
        context,
        resource_type,
        frozenset(ordered),
    )


def audit_smart_scopes(
    *, required_scopes: Iterable[str], requested_scopes: Iterable[str]
) -> SmartScopeAudit:
    """Compare a content-free workflow declaration with requested permissions.

    A wildcard satisfies specific resources for missing-scope detection, but
    is reported as overbroad when the workflow lists only specific resources.
    The caller must stop or review whenever either finding list is nonempty.
    """

    required = _parse_all(required_scopes)
    requested = _parse_all(requested_scopes)
    missing: list[ScopeFinding] = []
    excessive: list[ScopeFinding] = []

    for scope in required:
        if scope.is_launch:
            if not any(item.name == scope.name for item in requested):
                missing.append(_finding(ScopeReason.MISSING_SCOPE, scope))
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
            missing.append(_finding(ScopeReason.MISSING_SCOPE, scope, access=absent))

    for scope in requested:
        if scope.is_launch:
            if not any(item.name == scope.name for item in required):
                excessive.append(_finding(ScopeReason.EXCESSIVE_SCOPE, scope))
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
            excessive.append(_finding(ScopeReason.OVERBROAD_RESOURCE, scope))
            continue
        needed = frozenset().union(*(item.access for item in matching))
        extra = scope.access - needed
        if extra:
            excessive.append(_finding(ScopeReason.EXCESSIVE_SCOPE, scope, access=extra))

    return SmartScopeAudit(tuple(missing), tuple(excessive))


def _parse_all(values: Iterable[str]) -> tuple[SmartScope, ...]:
    if isinstance(values, str):
        raise ValueError("scopes must be an iterable of names")
    scopes = {scope.name: scope for scope in map(parse_smart_scope, values)}
    return tuple(scopes[name] for name in sorted(scopes))


def _finding(
    reason: ScopeReason, scope: SmartScope, *, access: frozenset[str] | None = None
) -> ScopeFinding:
    name = scope.name
    if access is not None:
        ordered = "".join(op for op in _ACCESS_ORDER if op in access)
        name = f"{scope.context}/{scope.resource_type}.{ordered}"
    return ScopeFinding(reason, name, scope.resource_type)
