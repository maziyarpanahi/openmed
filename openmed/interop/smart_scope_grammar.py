"""Shared offline SMART scope grammar and conservative permission comparison.

Query values remain private comparison inputs. Reports use only their digests.
No OAuth, credential storage, server discovery, or FHIR search is performed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import parse_qsl, urlencode

__all__ = [
    "SmartScope",
    "SmartScopeComparison",
    "SmartScopeFinding",
    "compare_smart_scopes",
    "parse_smart_scope",
    "parse_smart_scope_set",
]

_OPERATION_ORDER = "cruds"
_OPERATION_NAMES = dict(
    zip(_OPERATION_ORDER, ("create", "read", "update", "delete", "search"))
)
_V1_OPERATIONS = {"read": "rs", "write": "cud", "*": "cruds"}
_CONTEXT_ORDER = {"patient": 0, "user": 1, "system": 2}
_NON_CLINICAL = {
    "openid": "identity",
    "profile": "identity",
    "fhirUser": "identity",
    "launch": "launch",
    "launch/patient": "launch",
    "launch/encounter": "launch",
    "online_access": "session",
    "offline_access": "session",
}
_RESOURCE_RE = re.compile(r"[A-Z][A-Za-z0-9]{0,63}\Z")
_QUERY_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*\Z")
_BAD_ESCAPE_RE = re.compile(r"%(?![0-9a-fA-F]{2})")


@dataclass(frozen=True, order=True)
class SmartScope:
    """Normalized clinical permission or recognized non-clinical scope.

    ``value`` is a private canonical scope string, including query values.
    Use ``to_dict`` or ``evidence_value`` for diagnostic and audit output.
    """

    context: str
    resource_type: str | None
    operations: tuple[str, ...]
    constraints: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    non_clinical_name: str | None = None

    def __post_init__(self) -> None:
        if self.non_clinical_name is not None:
            if (
                _NON_CLINICAL.get(self.non_clinical_name) != self.context
                or self.resource_type is not None
                or self.operations
                or self.constraints
            ):
                raise ValueError("invalid non-clinical SMART scope")
            return
        if self.context not in _CONTEXT_ORDER:
            raise ValueError("invalid SMART scope context")
        if self.resource_type != "*" and (
            not isinstance(self.resource_type, str)
            or not _RESOURCE_RE.fullmatch(self.resource_type)
        ):
            raise ValueError("invalid SMART resource type")
        if (
            not self.operations
            or any(op not in _OPERATION_NAMES for op in self.operations)
            or len(set(self.operations)) != len(self.operations)
        ):
            raise ValueError("invalid SMART scope operations")
        if any(
            not _QUERY_KEY_RE.fullmatch(key) or not value
            for key, value in self.constraints
        ):
            raise ValueError("invalid SMART query constraint")

    @property
    def is_clinical(self) -> bool:
        """Return whether this scope grants clinical resource operations."""
        return self.non_clinical_name is None

    @property
    def value(self) -> str:
        """Return the private canonical scope, including any query constraints."""
        if self.non_clinical_name is not None:
            return self.non_clinical_name
        base = f"{self.context}/{self.resource_type}.{''.join(self.operations)}"
        return base + ("?" + urlencode(self.constraints) if self.constraints else "")

    @property
    def evidence_value(self) -> str:
        """Return a scope label with a digest in place of query keys and values."""
        base = self.value.split("?", 1)[0]
        if not self.constraints:
            return base
        encoded = json.dumps(self.constraints, ensure_ascii=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"{base}?constraint_digest={digest}"

    def atoms(self) -> frozenset[tuple[str, str, str]]:
        """Return clinical operation labels; constraints require full comparison.

        This legacy projection is not an authorization check. Use
        ``compare_smart_scopes`` to preserve wildcard and constraint semantics.
        """
        return frozenset(
            (self.context, self.resource_type, operation)
            for operation in self.operations
            if self.resource_type is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic evidence without granular query keys or values."""
        return {
            "scope": self.evidence_value,
            "context": self.context,
            "resource_type": self.resource_type,
            "operations": [
                {"code": op, "name": _OPERATION_NAMES[op]} for op in self.operations
            ],
        }


@dataclass(frozen=True)
class SmartScopeFinding:
    """A value-free parse finding, optionally located in a scope-set input."""

    reason_code: str
    source: str | None = None
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return controlled reason codes and input positions only."""
        result: dict[str, Any] = {"reason_code": self.reason_code}
        if self.source is not None:
            result["source"] = self.source
        if self.index is not None:
            result["index"] = self.index
        return result


def parse_smart_scope(value: str) -> SmartScope | SmartScopeFinding:
    """Parse v1/v2 clinical and identity, launch, or session scopes offline.

    SMART v1 read maps to rs, write to cud (including delete), and * to cruds.
    Unsupported or malformed input returns a value-free finding, never an
    exception containing the input. Query values are not diagnostic data.
    """
    if not isinstance(value, str):
        return SmartScopeFinding("malformed_scope")
    raw = value.strip()
    if raw in _NON_CLINICAL:
        return SmartScope(_NON_CLINICAL[raw], None, (), non_clinical_name=raw)
    if not raw or any(char.isspace() or ord(char) < 32 for char in raw):
        return SmartScopeFinding("malformed_scope")
    base, separator, query = raw.partition("?")
    try:
        context, remainder = base.split("/", 1)
        resource, operations = remainder.split(".", 1)
    except ValueError:
        return SmartScopeFinding("unrecognized_scope")
    context, operations = context.lower(), operations.lower()
    operations = _V1_OPERATIONS.get(operations, operations)
    if (
        context not in _CONTEXT_ORDER
        or (resource != "*" and not _RESOURCE_RE.fullmatch(resource))
        or not operations
        or len(set(operations)) != len(operations)
        or any(op not in _OPERATION_ORDER for op in operations)
    ):
        return SmartScopeFinding("malformed_scope")
    constraints: tuple[tuple[str, str], ...] = ()
    if separator:
        if not query or "#" in query or _BAD_ESCAPE_RE.search(query):
            return SmartScopeFinding("malformed_scope")
        try:
            pairs = parse_qsl(
                query,
                keep_blank_values=True,
                strict_parsing=True,
                encoding="utf-8",
                errors="strict",
                max_num_fields=256,
            )
        except (ValueError, UnicodeError):
            return SmartScopeFinding("malformed_scope")
        if not pairs or any(
            not _QUERY_KEY_RE.fullmatch(key)
            or not val
            or any(
                ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF
                for char in val
            )
            for key, val in pairs
        ):
            return SmartScopeFinding("malformed_scope")
        # Repeated parameters are retained. No FHIR search implication is inferred.
        constraints = tuple(sorted(pairs))
    ordered = tuple(op for op in _OPERATION_ORDER if op in operations)
    return SmartScope(context, resource, ordered, constraints)


def parse_smart_scope_set(
    values: Iterable[str] | str,
    *,
    source: str,
) -> tuple[tuple[SmartScope, ...], tuple[SmartScopeFinding, ...]]:
    """Parse an OAuth space-delimited scope string or iterable of scope names.

    ``source`` must be a controlled label (required or declared). Invalid entries
    contribute findings and never clinical permissions.
    """
    if source not in {"required", "declared"}:
        raise ValueError("invalid scope input source")
    entries = values.split() if isinstance(values, str) else values
    scopes: set[SmartScope] = set()
    findings: list[SmartScopeFinding] = []
    for index, value in enumerate(entries):
        parsed = parse_smart_scope(value)
        if isinstance(parsed, SmartScopeFinding):
            findings.append(replace(parsed, source=source, index=index))
        else:
            scopes.add(parsed)
    return tuple(sorted(scopes, key=_sort_key)), tuple(findings)


@dataclass(frozen=True)
class SmartScopeComparison:
    """Shared parsed scope sets, differences, and value-free parse findings."""

    required_scopes: tuple[SmartScope, ...]
    declared_scopes: tuple[SmartScope, ...]
    missing_scopes: tuple[SmartScope, ...]
    excessive_scopes: tuple[SmartScope, ...]
    findings: tuple[SmartScopeFinding, ...]


def compare_smart_scopes(
    *, required_scopes: Iterable[str] | str, declared_scopes: Iterable[str] | str
) -> SmartScopeComparison:
    """Compare permissions conservatively for exact audits or preflight callers.

    A constrained grant covers only an identical normalized constraint set.
    An unconstrained grant covers constrained needs but is overbroad. Wildcard
    resources cover specific resources, never the reverse. Non-clinical scopes
    match only by name and cannot cover any clinical resource operation.
    """
    required, required_findings = parse_smart_scope_set(
        required_scopes, source="required"
    )
    declared, declared_findings = parse_smart_scope_set(
        declared_scopes, source="declared"
    )
    return SmartScopeComparison(
        required,
        declared,
        _difference(required, declared),
        _difference(declared, required),
        required_findings + declared_findings,
    )


def _covers(grant: SmartScope, need: SmartScope) -> bool:
    if not grant.is_clinical or not need.is_clinical:
        return (
            grant.non_clinical_name == need.non_clinical_name
            and not grant.is_clinical
            and not need.is_clinical
        )
    return (
        grant.context == need.context
        and (grant.resource_type == "*" or grant.resource_type == need.resource_type)
        and (not grant.constraints or grant.constraints == need.constraints)
    )


def _difference(
    needs: tuple[SmartScope, ...], grants: tuple[SmartScope, ...]
) -> tuple[SmartScope, ...]:
    differences: list[SmartScope] = []
    for need in needs:
        matching = [grant for grant in grants if _covers(grant, need)]
        if not need.is_clinical:
            if not matching:
                differences.append(need)
            continue
        available = {op for grant in matching for op in grant.operations}
        absent = tuple(op for op in need.operations if op not in available)
        if absent:
            differences.append(replace(need, operations=absent))
    grouped: dict[SmartScope, set[str]] = {}
    for scope in differences:
        if not scope.is_clinical:
            grouped.setdefault(scope, set())
            continue
        key = replace(scope, operations=("r",))
        grouped.setdefault(key, set()).update(scope.operations)
    merged = [
        replace(scope, operations=tuple(op for op in _OPERATION_ORDER if op in ops))
        if scope.is_clinical
        else scope
        for scope, ops in grouped.items()
    ]
    return tuple(sorted(merged, key=_sort_key))


def _sort_key(scope: SmartScope) -> tuple[int, str, str]:
    return (
        _CONTEXT_ORDER.get(scope.context, 3),
        scope.resource_type or "",
        scope.value,
    )
