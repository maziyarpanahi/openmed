"""Offline SMART-on-FHIR scope audits using the shared scope grammar.

These helpers perform no OAuth, server discovery, or patient-record access.
"""

from __future__ import annotations

import re

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openmed.interop.smart_scope_grammar import (
    SmartScope,
    SmartScopeFinding,
    compare_smart_scopes,
    parse_smart_scope,
)

__all__ = [
    "SmartScope",
    "SmartScopeAudit",
    "SmartScopeFinding",
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


@dataclass(frozen=True)
class SmartScopeAudit:
    """Comparison between workflow-required and declared SMART scopes."""

    workflow_id: str
    required_scopes: tuple[SmartScope, ...]
    declared_scopes: tuple[SmartScope, ...]
    missing_scopes: tuple[SmartScope, ...]
    excessive_scopes: tuple[SmartScope, ...]
    findings: tuple[SmartScopeFinding, ...] = ()

    @property
    def status(self) -> str:
        """Return invalid, missing, excessive, or pass; invalid input fails closed."""
        if self.findings:
            return "invalid"
        if self.missing_scopes:
            return "missing"
        if self.excessive_scopes:
            return "excessive"
        return "pass"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence with no granular query values."""
        report = {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "required_scopes": [scope.to_dict() for scope in self.required_scopes],
            "declared_scopes": [scope.to_dict() for scope in self.declared_scopes],
            "missing_scopes": [scope.to_dict() for scope in self.missing_scopes],
            "excessive_scopes": [scope.to_dict() for scope in self.excessive_scopes],
        }
        if self.findings:
            report["findings"] = [finding.to_dict() for finding in self.findings]
        return report


def normalize_smart_scope(value: str) -> str | SmartScopeFinding:
    """Return a private canonical scope string or a value-free parse finding."""
    parsed = parse_smart_scope(value)
    return parsed.value if isinstance(parsed, SmartScope) else parsed


def audit_smart_scopes(
    *,
    workflow_id: str,
    required_scopes: Iterable[str] | str,
    declared_scopes: Iterable[str] | str,
) -> SmartScopeAudit:
    """Compare required and declared SMART scopes for one offline workflow.

    Invalid input yields findings and an invalid status, rather than raising.
    Recognized non-clinical scopes never confer clinical resource permissions.
    """
    comparison = compare_smart_scopes(
        required_scopes=required_scopes, declared_scopes=declared_scopes
    )
    return SmartScopeAudit(
        workflow_id=workflow_id,
        required_scopes=comparison.required_scopes,
        declared_scopes=comparison.declared_scopes,
        missing_scopes=comparison.missing_scopes,
        excessive_scopes=comparison.excessive_scopes,
        findings=comparison.findings,
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
