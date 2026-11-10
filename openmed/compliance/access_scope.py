"""Deterministic, aggregate-only access-scope minimization evidence.

The evaluator compares the resource/action scopes a workflow requested, used,
and was approved to use. Scope identifiers are accepted only as structured
metadata and are kept out of serialized reports and exception messages. The
module is local-only: it has no clock, filesystem, telemetry, or network
dependency.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

ACCESS_SCOPE_SCHEMA_VERSION: Final = 1
ACCESS_SCOPE_REPORT_TYPE: Final = "access_scope_minimization"
ACCESS_SCOPE_ALLOW: Final = "allow"
ACCESS_SCOPE_BLOCK: Final = "block"
ACCESS_SCOPE_DECISIONS: Final = (ACCESS_SCOPE_ALLOW, ACCESS_SCOPE_BLOCK)

REASON_UNDECLARED_USE: Final = "undeclared_use"
REASON_UNAPPROVED_REQUEST: Final = "unapproved_request"
REASON_UNAPPROVED_USE: Final = "unapproved_use"
REASON_UNAPPROVED_ESCALATION: Final = "unapproved_escalation"
REASON_OVERBROAD_REQUEST: Final = "overbroad_request"
REASON_WILDCARD_REQUEST: Final = "wildcard_request"
REASON_WILDCARD_ESCALATION: Final = "wildcard_escalation"
REASON_WILDCARD_USE: Final = "wildcard_use"
ACCESS_SCOPE_REASON_CODES: Final = (
    REASON_UNDECLARED_USE,
    REASON_UNAPPROVED_REQUEST,
    REASON_UNAPPROVED_USE,
    REASON_UNAPPROVED_ESCALATION,
    REASON_OVERBROAD_REQUEST,
    REASON_WILDCARD_REQUEST,
    REASON_WILDCARD_ESCALATION,
    REASON_WILDCARD_USE,
)

_SCOPE_COMPONENT = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")

__all__ = [
    "ACCESS_SCOPE_ALLOW",
    "ACCESS_SCOPE_BLOCK",
    "ACCESS_SCOPE_DECISIONS",
    "ACCESS_SCOPE_REASON_CODES",
    "ACCESS_SCOPE_REPORT_TYPE",
    "ACCESS_SCOPE_SCHEMA_VERSION",
    "REASON_OVERBROAD_REQUEST",
    "REASON_UNAPPROVED_ESCALATION",
    "REASON_UNAPPROVED_REQUEST",
    "REASON_UNAPPROVED_USE",
    "REASON_UNDECLARED_USE",
    "REASON_WILDCARD_ESCALATION",
    "REASON_WILDCARD_REQUEST",
    "REASON_WILDCARD_USE",
    "AccessScope",
    "AccessScopeCounts",
    "AccessScopeEvaluation",
    "AccessScopePolicy",
    "AccessScopeReport",
    "AccessScopeValidationError",
    "AccessScopeViolation",
    "AccessScopeViolationError",
    "enforce_access_scope",
    "evaluate_access_scope",
    "normalize_access_scopes",
    "render_access_scope_evidence",
    "render_access_scope_markdown",
]


class AccessScopeValidationError(ValueError):
    """Raised when structured access-scope metadata is invalid."""


def _normalize_component(value: Any, *, field_name: str) -> str:
    """Normalize one safe resource/action component without echoing input."""

    if not isinstance(value, str) or not value.strip():
        raise AccessScopeValidationError(
            f"{field_name} must be a non-empty scope identifier"
        )
    normalized = value.strip().lower()
    if normalized == "*":
        return normalized
    if _SCOPE_COMPONENT.fullmatch(normalized) is None:
        raise AccessScopeValidationError(
            f"{field_name} must be an identifier or the explicit wildcard '*'"
        )
    return normalized


@dataclass(frozen=True, init=False)
class AccessScope:
    """One structured ``resource:action`` access scope.

    ``*`` is allowed as a complete resource or action component. A wildcard
    is a policy declaration, not an observed access: used scopes must be
    concrete so that evidence cannot hide an unbounded operation.
    """

    resource: str
    action: str

    def __init__(self, resource: str, action: str) -> None:
        object.__setattr__(
            self,
            "resource",
            _normalize_component(resource, field_name="resource"),
        )
        object.__setattr__(
            self,
            "action",
            _normalize_component(action, field_name="action"),
        )

    @classmethod
    def from_string(cls, value: str) -> AccessScope:
        """Parse a canonical ``resource:action`` string."""

        if not isinstance(value, str) or value.count(":") != 1:
            raise AccessScopeValidationError(
                "scope must use the canonical resource:action form"
            )
        resource, action = value.split(":")
        return cls(resource, action)

    @property
    def is_wildcard(self) -> bool:
        """Whether either component is an explicit wildcard."""

        return self.resource == "*" or self.action == "*"

    @property
    def value(self) -> str:
        """Return the canonical string form for in-memory comparisons."""

        return f"{self.resource}:{self.action}"

    def covers(self, other: AccessScope) -> bool:
        """Return whether this declaration covers ``other``.

        Coverage is component-wise. A wildcard declaration covers a concrete
        component or the same wildcard; a concrete declaration never covers a
        wildcard declaration.
        """

        if not isinstance(other, AccessScope):
            raise TypeError("scope coverage requires an AccessScope")
        return _component_covers(self.resource, other.resource) and _component_covers(
            self.action, other.action
        )

    def matches(self, actual: AccessScope) -> bool:
        """Return whether this declaration covers one concrete actual scope."""

        return not actual.is_wildcard and self.covers(actual)

    def to_dict(self) -> dict[str, str]:
        """Return the structured scope for callers that already handle metadata."""

        return {"resource": self.resource, "action": self.action}

    def __str__(self) -> str:
        return self.value


def _component_covers(declared: str, observed: str) -> bool:
    return declared == "*" or declared == observed


def _scope_item(value: Any, *, field_name: str) -> AccessScope:
    if isinstance(value, AccessScope):
        return value
    if isinstance(value, str):
        try:
            return AccessScope.from_string(value)
        except AccessScopeValidationError as exc:
            raise AccessScopeValidationError(
                f"{field_name} contains an invalid scope"
            ) from exc
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return AccessScope(value[0], value[1])
    raise AccessScopeValidationError(
        f"{field_name} must contain resource:action scopes"
    )


def normalize_access_scopes(
    scopes: Any,
    *,
    field_name: str = "scopes",
    allow_none: bool = False,
) -> tuple[AccessScope, ...]:
    """Return unique, sorted scopes from strings, pairs, or resource mappings.

    Mappings use ``resource -> actions`` shorthand. Mapping values are used
    only to create structured scope pairs and are never copied to evidence.
    """

    if scopes is None:
        if allow_none:
            return ()
        raise AccessScopeValidationError(f"{field_name} is required")

    if isinstance(scopes, (AccessScope, str)):
        candidates: Iterable[Any] = (scopes,)
    elif isinstance(scopes, Mapping):
        mapped: list[AccessScope] = []
        for resource, actions in scopes.items():
            if isinstance(actions, str):
                action_values: Iterable[Any] = (actions,)
            else:
                try:
                    action_values = tuple(actions)
                except TypeError as exc:
                    raise AccessScopeValidationError(
                        f"{field_name} mapping values must be action iterables"
                    ) from exc
            for action in action_values:
                mapped.append(AccessScope(resource, action))
        candidates = mapped
    elif isinstance(scopes, (bytes, bytearray)):
        raise AccessScopeValidationError(
            f"{field_name} must contain resource:action scopes"
        )
    elif (
        isinstance(scopes, (tuple, list))
        and len(scopes) == 2
        and all(isinstance(item, str) and ":" not in item for item in scopes)
    ):
        candidates = (scopes,)
    else:
        try:
            candidates = tuple(scopes)
        except TypeError as exc:
            raise AccessScopeValidationError(
                f"{field_name} must be an iterable of resource:action scopes"
            ) from exc

    try:
        normalized = {
            _scope_item(candidate, field_name=field_name) for candidate in candidates
        }
    except AccessScopeValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise AccessScopeValidationError(
            f"{field_name} contains an invalid scope"
        ) from exc
    return tuple(sorted(normalized, key=lambda item: (item.resource, item.action)))


@dataclass(frozen=True, init=False)
class AccessScopePolicy:
    """Explicit wildcard and escalation rules for one scope evaluation.

    Wildcards in requested scopes and escalation declarations are blocked by
    default. They become valid only when ``allow_wildcards`` is true or the
    matching wildcard is listed in ``wildcard_rules``. Approved wildcards are
    already explicit approval declarations and therefore do not need a second
    opt-in merely to approve a concrete request.
    """

    allow_wildcards: bool
    wildcard_rules: tuple[AccessScope, ...]
    escalation_rules: tuple[AccessScope, ...]

    def __init__(
        self,
        *,
        allow_wildcards: bool = False,
        wildcard_rules: Any = (),
        escalation_rules: Any = (),
    ) -> None:
        if not isinstance(allow_wildcards, bool):
            raise AccessScopeValidationError("allow_wildcards must be a boolean")
        normalized_wildcards = normalize_access_scopes(
            wildcard_rules,
            field_name="wildcard_rules",
            allow_none=True,
        )
        if any(not scope.is_wildcard for scope in normalized_wildcards):
            raise AccessScopeValidationError(
                "wildcard_rules must contain explicit wildcard scopes"
            )
        normalized_escalations = normalize_access_scopes(
            escalation_rules,
            field_name="escalation_rules",
            allow_none=True,
        )
        object.__setattr__(self, "allow_wildcards", allow_wildcards)
        object.__setattr__(self, "wildcard_rules", normalized_wildcards)
        object.__setattr__(self, "escalation_rules", normalized_escalations)

    def allows_wildcard(self, scope: AccessScope) -> bool:
        """Return whether a requested or escalation wildcard is explicit."""

        return self.allow_wildcards or any(
            rule.covers(scope) for rule in self.wildcard_rules
        )


@dataclass(frozen=True)
class AccessScopeViolation:
    """One aggregate block reason without the matching scope identifier."""

    reason: str
    count: int

    def __post_init__(self) -> None:
        if self.reason not in ACCESS_SCOPE_REASON_CODES:
            raise AccessScopeValidationError("unsupported access-scope reason")
        if not isinstance(self.count, int) or self.count <= 0:
            raise AccessScopeValidationError("access-scope violation count is invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a count-only violation record."""

        return {"reason": self.reason, "count": self.count}


@dataclass(frozen=True)
class AccessScopeCounts:
    """Aggregate counts used by the access-scope evidence report."""

    requested: int
    used: int
    approved: int
    escalation_rules: int
    wildcard_rules: int
    wildcard_requested: int
    wildcard_used: int
    wildcard_approved: int
    wildcard_escalation_rules: int
    unused_requested: int
    undeclared_used: int
    unapproved_requested: int
    unapproved_used: int
    unapproved_escalation_rules: int
    escalated_used: int

    def to_dict(self) -> dict[str, int]:
        """Return deterministic counts without any scope values."""

        return {
            "approved_count": self.approved,
            "escalated_used_count": self.escalated_used,
            "escalation_rule_count": self.escalation_rules,
            "undeclared_used_count": self.undeclared_used,
            "unapproved_escalation_rule_count": self.unapproved_escalation_rules,
            "unapproved_requested_count": self.unapproved_requested,
            "unapproved_used_count": self.unapproved_used,
            "unused_requested_count": self.unused_requested,
            "used_count": self.used,
            "wildcard_approved_count": self.wildcard_approved,
            "wildcard_escalation_rule_count": self.wildcard_escalation_rules,
            "wildcard_requested_count": self.wildcard_requested,
            "wildcard_rule_count": self.wildcard_rules,
            "wildcard_used_count": self.wildcard_used,
            "requested_count": self.requested,
        }


@dataclass(frozen=True)
class AccessScopeEvaluation:
    """Deterministic access-scope decision with aggregate-only serialization."""

    requested: tuple[AccessScope, ...]
    used: tuple[AccessScope, ...]
    approved: tuple[AccessScope, ...]
    policy: AccessScopePolicy
    unused_requested: tuple[AccessScope, ...]
    undeclared_used: tuple[AccessScope, ...]
    unapproved_requested: tuple[AccessScope, ...]
    unapproved_used: tuple[AccessScope, ...]
    unapproved_escalation_rules: tuple[AccessScope, ...]
    escalated_used: tuple[AccessScope, ...]
    wildcard_requested: tuple[AccessScope, ...]
    wildcard_used: tuple[AccessScope, ...]
    wildcard_approved: tuple[AccessScope, ...]
    wildcard_escalation_rules: tuple[AccessScope, ...]
    wildcard_request_violations: tuple[AccessScope, ...]
    wildcard_escalation_violations: tuple[AccessScope, ...]

    @property
    def allowed(self) -> bool:
        """Whether no undeclared, unapproved, overbroad, or wildcard use exists."""

        return not self.reasons

    @property
    def passed(self) -> bool:
        """Alias for :attr:`allowed` suitable for gate-oriented callers."""

        return self.allowed

    @property
    def decision(self) -> str:
        """Return the stable ``allow`` or ``block`` decision."""

        return ACCESS_SCOPE_ALLOW if self.allowed else ACCESS_SCOPE_BLOCK

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return deterministic reason codes for all blocking conditions."""

        reason_counts = self._reason_counts()
        return tuple(
            reason for reason in ACCESS_SCOPE_REASON_CODES if reason_counts[reason] > 0
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Alias for :attr:`reasons`."""

        return self.reasons

    @property
    def violations(self) -> tuple[AccessScopeViolation, ...]:
        """Return aggregate violations without scope identifiers."""

        counts = self._reason_counts()
        return tuple(
            AccessScopeViolation(reason, counts[reason])
            for reason in ACCESS_SCOPE_REASON_CODES
            if counts[reason] > 0
        )

    @property
    def overbroad_requested(self) -> tuple[AccessScope, ...]:
        """Return requested declarations that matched no observed use."""

        return self.unused_requested

    @property
    def undeclared_scopes(self) -> tuple[AccessScope, ...]:
        """Return concrete observed scopes absent from request and escalation."""

        return self.undeclared_used

    @property
    def escalated_scopes(self) -> tuple[AccessScope, ...]:
        """Return concrete observed scopes allowed by an escalation rule."""

        return self.escalated_used

    @property
    def counts(self) -> AccessScopeCounts:
        """Return the aggregate dimensions used by the serialized evidence."""

        return AccessScopeCounts(
            requested=len(self.requested),
            used=len(self.used),
            approved=len(self.approved),
            escalation_rules=len(self.policy.escalation_rules),
            wildcard_rules=len(self.policy.wildcard_rules),
            wildcard_requested=len(self.wildcard_requested),
            wildcard_used=len(self.wildcard_used),
            wildcard_approved=len(self.wildcard_approved),
            wildcard_escalation_rules=len(self.wildcard_escalation_rules),
            unused_requested=len(self.unused_requested),
            undeclared_used=len(self.undeclared_used),
            unapproved_requested=len(self.unapproved_requested),
            unapproved_used=len(self.unapproved_used),
            unapproved_escalation_rules=len(self.unapproved_escalation_rules),
            escalated_used=len(self.escalated_used),
        )

    def _reason_counts(self) -> dict[str, int]:
        return {
            REASON_UNDECLARED_USE: len(self.undeclared_used),
            REASON_UNAPPROVED_REQUEST: len(self.unapproved_requested),
            REASON_UNAPPROVED_USE: len(self.unapproved_used),
            REASON_UNAPPROVED_ESCALATION: len(self.unapproved_escalation_rules),
            REASON_OVERBROAD_REQUEST: len(self.unused_requested),
            REASON_WILDCARD_REQUEST: len(self.wildcard_request_violations),
            REASON_WILDCARD_ESCALATION: len(self.wildcard_escalation_violations),
            REASON_WILDCARD_USE: len(self.wildcard_used),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return count-only, JSON-compatible minimization evidence."""

        return {
            "decision": self.decision,
            "notice": (
                "Counts only; resource and action identifiers are omitted from "
                "serialized access-scope evidence."
            ),
            "reasons": list(self.reasons),
            "report_type": ACCESS_SCOPE_REPORT_TYPE,
            "rules": {
                "allow_wildcards": self.policy.allow_wildcards,
                "explicit_wildcard_rules": len(self.policy.wildcard_rules),
                "explicit_escalation_rules": len(self.policy.escalation_rules),
            },
            "schema_version": ACCESS_SCOPE_SCHEMA_VERSION,
            "summary": self.counts.to_dict(),
            "violations": [violation.to_dict() for violation in self.violations],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministic count-only evidence as JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render deterministic count-only evidence as Markdown."""

        counts = self.counts.to_dict()
        lines = [
            "# Access-scope minimization evidence",
            "",
            "> Counts only; resource and action identifiers are intentionally omitted.",
            "",
            f"- Decision: **{self.decision}**",
            f"- Schema version: `{ACCESS_SCOPE_SCHEMA_VERSION}`",
            f"- Report type: `{ACCESS_SCOPE_REPORT_TYPE}`",
            "- Generation: deterministic and offline; no external calls are made",
            "",
            "## Scope counts",
            "",
            "| Measure | Count |",
            "|---|---:|",
        ]
        for name, value in counts.items():
            lines.append(f"| {name.replace('_', ' ').title()} | {value} |")
        lines.extend(["", "## Blocking reasons", ""])
        if self.violations:
            lines.extend(
                f"- `{violation.reason}`: {violation.count}"
                for violation in self.violations
            )
        else:
            lines.append("- none")
        return "\n".join(lines) + "\n"


AccessScopeReport = AccessScopeEvaluation


class AccessScopeViolationError(PermissionError):
    """Raised when an access-scope evaluation must block a workflow."""

    def __init__(self, evaluation: AccessScopeEvaluation) -> None:
        if not isinstance(evaluation, AccessScopeEvaluation):
            raise TypeError("access-scope errors require an evaluation")
        self.evaluation = evaluation
        self.report = evaluation
        reason_text = ", ".join(
            f"{violation.reason}={violation.count}"
            for violation in evaluation.violations
        )
        super().__init__(f"access-scope policy blocked the workflow: {reason_text}")


def _covered(scope: AccessScope, declarations: tuple[AccessScope, ...]) -> bool:
    return any(declaration.covers(scope) for declaration in declarations)


def _build_policy(
    *,
    policy: AccessScopePolicy | None,
    allow_wildcards: bool,
    wildcard_rules: Any,
    escalation_rules: Any,
) -> AccessScopePolicy:
    if policy is not None:
        if not isinstance(policy, AccessScopePolicy):
            raise AccessScopeValidationError("policy must be an AccessScopePolicy")
        if (
            allow_wildcards
            or wildcard_rules not in ((), None)
            or escalation_rules
            not in (
                (),
                None,
            )
        ):
            raise AccessScopeValidationError(
                "provide either policy or explicit access-scope rules, not both"
            )
        return policy
    return AccessScopePolicy(
        allow_wildcards=allow_wildcards,
        wildcard_rules=wildcard_rules,
        escalation_rules=escalation_rules,
    )


def evaluate_access_scope(
    requested: Any,
    used: Any,
    approved: Any,
    *,
    escalation_rules: Any = (),
    wildcard_rules: Any = (),
    allow_wildcards: bool = False,
    policy: AccessScopePolicy | None = None,
) -> AccessScopeEvaluation:
    """Compare requested, used, and approved scopes.

    A decision is allowed only when every concrete used scope is requested or
    covered by an explicit, approved escalation rule; every request is covered
    by approval; and every requested declaration matches an observed use.
    Wildcard requests and wildcard escalation rules require explicit policy
    opt-in. Approved wildcard declarations may approve narrower concrete
    scopes, but they do not silently authorize wildcard requests.
    """

    normalized_requested = normalize_access_scopes(
        requested,
        field_name="requested",
    )
    normalized_used = normalize_access_scopes(used, field_name="used")
    normalized_approved = normalize_access_scopes(approved, field_name="approved")
    resolved_policy = _build_policy(
        policy=policy,
        allow_wildcards=allow_wildcards,
        wildcard_rules=wildcard_rules,
        escalation_rules=escalation_rules,
    )

    wildcard_requested = tuple(
        scope for scope in normalized_requested if scope.is_wildcard
    )
    wildcard_used = tuple(scope for scope in normalized_used if scope.is_wildcard)
    wildcard_approved = tuple(
        scope for scope in normalized_approved if scope.is_wildcard
    )
    wildcard_escalation_rules = tuple(
        scope for scope in resolved_policy.escalation_rules if scope.is_wildcard
    )
    wildcard_request_violations = tuple(
        scope
        for scope in wildcard_requested
        if not resolved_policy.allows_wildcard(scope)
    )
    wildcard_escalation_violations = tuple(
        scope
        for scope in wildcard_escalation_rules
        if not resolved_policy.allows_wildcard(scope)
    )

    unapproved_requested = tuple(
        scope
        for scope in normalized_requested
        if not _covered(scope, normalized_approved)
    )
    concrete_used = tuple(scope for scope in normalized_used if not scope.is_wildcard)
    unapproved_used = tuple(
        scope for scope in concrete_used if not _covered(scope, normalized_approved)
    )
    unapproved_escalation_rules = tuple(
        scope
        for scope in resolved_policy.escalation_rules
        if not _covered(scope, normalized_approved)
    )
    escalated_used = tuple(
        scope
        for scope in concrete_used
        if not _covered(scope, normalized_requested)
        and _covered(scope, resolved_policy.escalation_rules)
        and _covered(scope, normalized_approved)
    )
    undeclared_used = tuple(
        scope
        for scope in concrete_used
        if not _covered(scope, normalized_requested)
        and not _covered(scope, resolved_policy.escalation_rules)
    )
    unused_requested = tuple(
        scope
        for scope in normalized_requested
        if not any(scope.matches(observed) for observed in concrete_used)
    )

    return AccessScopeEvaluation(
        requested=normalized_requested,
        used=normalized_used,
        approved=normalized_approved,
        policy=resolved_policy,
        unused_requested=unused_requested,
        undeclared_used=undeclared_used,
        unapproved_requested=unapproved_requested,
        unapproved_used=unapproved_used,
        unapproved_escalation_rules=unapproved_escalation_rules,
        escalated_used=escalated_used,
        wildcard_requested=wildcard_requested,
        wildcard_used=wildcard_used,
        wildcard_approved=wildcard_approved,
        wildcard_escalation_rules=wildcard_escalation_rules,
        wildcard_request_violations=wildcard_request_violations,
        wildcard_escalation_violations=wildcard_escalation_violations,
    )


def enforce_access_scope(
    requested: Any,
    used: Any,
    approved: Any,
    *,
    escalation_rules: Any = (),
    wildcard_rules: Any = (),
    allow_wildcards: bool = False,
    policy: AccessScopePolicy | None = None,
) -> AccessScopeEvaluation:
    """Evaluate scopes and raise a PHI-safe error for a blocking decision."""

    evaluation = evaluate_access_scope(
        requested,
        used,
        approved,
        escalation_rules=escalation_rules,
        wildcard_rules=wildcard_rules,
        allow_wildcards=allow_wildcards,
        policy=policy,
    )
    if not evaluation.allowed:
        raise AccessScopeViolationError(evaluation)
    return evaluation


def render_access_scope_evidence(evaluation: AccessScopeEvaluation) -> str:
    """Render an access-scope evaluation as count-only Markdown evidence."""

    if not isinstance(evaluation, AccessScopeEvaluation):
        raise TypeError("access-scope evidence requires an evaluation")
    return evaluation.to_markdown()


def render_access_scope_markdown(evaluation: AccessScopeEvaluation) -> str:
    """Alias for :func:`render_access_scope_evidence`."""

    return render_access_scope_evidence(evaluation)
