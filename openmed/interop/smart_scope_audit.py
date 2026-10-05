"""Offline SMART-on-FHIR scope audits using the shared scope grammar.

These helpers perform no OAuth, server discovery, or patient-record access.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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
