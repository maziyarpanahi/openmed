"""Offline SMART-on-FHIR scope audits using the shared scope grammar.

These helpers perform no OAuth, server discovery, or patient-record access.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from typing import Any

from openmed.interop.smart_scope_grammar import (
    SmartScope,
    SmartScopeFinding,
    compare_smart_scopes,
    parse_smart_scope,
)

__all__ = [
    "SmartGrantedScopeAudit",
    "audit_granted_smart_scopes",
    "smart_scopes_cover",
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


# Preflight retains its public difference records and uses the shared grammar.
_PREFLIGHT_LAUNCH_SCOPES = frozenset({"launch", "launch/patient", "launch/encounter"})


@dataclass(frozen=True, slots=True)
class PreflightScope:
    """Validated scope with a private name that may include query constraints."""

    name: str = field(repr=False)
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
    findings: tuple[SmartScopeFinding, ...] = ()

    @property
    def is_least_privilege(self) -> bool:
        """Return whether requested scopes match declared workflow needs."""

        return not (self.missing_scopes or self.excessive_scopes or self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Return differences and invalid-input positions without private values."""
        result: dict[str, Any] = {
            "missing_scopes": [item.to_dict() for item in self.missing_scopes],
            "excessive_scopes": [item.to_dict() for item in self.excessive_scopes],
        }
        if self.findings:
            result["findings"] = [item.to_dict() for item in self.findings]
        return result


def parse_smart_scope_preflight(value: str) -> PreflightScope:
    """Normalize a clinical, identity, launch or session scope privately.

    This strict adapter retains its controlled ValueError contract for custody
    intake. The audit APIs return parse findings instead of raising. A valid
    name can contain query values and must never be used as public evidence.

    Raises:
        ValueError: The scope is malformed or unsupported, without input text.
    """
    parsed = parse_smart_scope(value)
    if isinstance(parsed, SmartScopeFinding):
        raise ValueError("invalid SMART scope") from None
    return PreflightScope(
        parsed.value, parsed.context, parsed.resource_type, frozenset(parsed.operations)
    )


def audit_smart_scope_preflight(
    *, required_scopes: Iterable[str] | str, requested_scopes: Iterable[str] | str
) -> PreflightScopeAudit:
    """Compare workflow needs with requested permissions using one grammar.

    Granular permissions are compared conservatively. Invalid entries yield
    findings and fail least-privilege checks. Clinical wildcard grants remain
    explicitly overbroad when only specific resources are required. Every
    difference contains a constraint digest in place of query keys and values.
    """
    comparison = compare_smart_scopes(
        required_scopes=required_scopes, declared_scopes=requested_scopes
    )
    missing = tuple(
        PreflightScopeFinding(
            PreflightReason.MISSING_SCOPE, scope.evidence_value, scope.resource_type
        )
        for scope in comparison.missing_scopes
    )
    excessive = tuple(
        PreflightScopeFinding(
            PreflightReason.OVERBROAD_RESOURCE
            if scope.is_clinical
            and scope.resource_type == "*"
            and not any(
                need.context == scope.context and need.resource_type == "*"
                for need in comparison.required_scopes
            )
            else PreflightReason.EXCESSIVE_SCOPE,
            scope.evidence_value,
            scope.resource_type,
        )
        for scope in comparison.excessive_scopes
    )
    return PreflightScopeAudit(
        tuple(sorted(missing, key=lambda item: item.scope)),
        tuple(sorted(excessive, key=lambda item: item.scope)),
        comparison.findings,
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
