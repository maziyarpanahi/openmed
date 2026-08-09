"""Deterministic, value-free composition of overlapping privacy policies.

The module models one policy rule per :class:`PrivacyPolicy`.  Rules may be
scoped to a field, resource path, or transport.  Composition is deliberately
small and local: matching is structural, deny always overrides allow, and
every serialized decision contains fingerprints and categories rather than
selectors or context values.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

POLICY_COMPOSITION_SCHEMA_VERSION = 1


class PolicyScope(str, Enum):
    """Scope at which a policy rule is evaluated."""

    FIELD = "field"
    RESOURCE = "resource"
    TRANSPORT = "transport"


class PolicyDecision(str, Enum):
    """The two decisions supported by the policy composer."""

    ALLOW = "allow"
    DENY = "deny"


class ConflictCategory(str, Enum):
    """Stable explanations for how an effective decision was selected."""

    NONE = "none"
    DEFAULT = "default"
    DENY_OVERRIDES = "deny_overrides"
    MULTIPLE_DENIES = "multiple_denies"
    PRECEDENCE = "precedence"
    INHERITED_DENY = "inherited_deny"
    INHERITED_ALLOW = "inherited_allow"


# Field selectors are the most specific, followed by resource selectors and
# then transport selectors.  This order only chooses among rules with the
# same decision; an applicable deny always wins over every applicable allow.
DEFAULT_SCOPE_PRECEDENCE: tuple[PolicyScope, ...] = (
    PolicyScope.FIELD,
    PolicyScope.RESOURCE,
    PolicyScope.TRANSPORT,
)


def _coerce_scope(value: PolicyScope | str) -> PolicyScope:
    if isinstance(value, PolicyScope):
        return value
    if isinstance(value, str):
        candidate = value.strip().lower()
        for scope in PolicyScope:
            if candidate == scope.value:
                return scope
    raise ValueError("scope must be one of field, resource, or transport")


def _coerce_decision(value: PolicyDecision | str) -> PolicyDecision:
    if isinstance(value, PolicyDecision):
        return value
    if isinstance(value, str):
        candidate = value.strip().lower()
        for decision in PolicyDecision:
            if candidate == decision.value:
                return decision
    raise ValueError("decision must be allow or deny")


def _normalise_component(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalised = unicodedata.normalize("NFC", value.strip())
    if not normalised:
        raise ValueError(f"{name} must be non-empty")
    return normalised


def _normalise_path(value: str | Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.split("/")
    elif isinstance(value, Sequence):
        parts = list(value)
    else:
        raise TypeError(f"{name} must be a path string or sequence")

    if not parts:
        raise ValueError(f"{name} must be non-empty")
    normalised = tuple(_normalise_component(part, f"{name} segment") for part in parts)
    if any(part == "" for part in normalised):
        raise ValueError(f"{name} contains an empty path segment")
    if "**" in normalised[:-1]:
        raise ValueError(f"{name} may use ** only as its final segment")
    return normalised


def _normalise_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    # Validate once at construction time so fingerprinting cannot fail during
    # evaluation and no fallback stringification can leak a sensitive value.
    canonical = _canonical_value(dict(value))
    return cast(Mapping[str, Any], _freeze_value(canonical))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return unicodedata.normalize("NFC", value) if isinstance(value, str) else value
    if isinstance(value, Mapping):
        items: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings")
            items[unicodedata.normalize("NFC", key)] = _canonical_value(item)
        return {key: items[key] for key in sorted(items)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_canonical_value(item) for item in value]
        return sorted(values, key=lambda item: _canonical_json(item))
    raise TypeError("metadata must contain JSON-compatible values")


def _freeze_value(value: Any) -> Any:
    """Recursively freeze canonical metadata before it enters a policy."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _canonical_value(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("policy values must be JSON-compatible") from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, repr=False)
class PrivacyPolicy:
    """One immutable, scoped privacy-policy rule.

    ``selector`` is an exact field or transport selector, or a slash-delimited
    resource path.  A resource selector inherits into descendants by default;
    set ``inherit=False`` to make it exact.  ``*`` matches one path component
    and a final ``**`` matches any number of descendant components.

    The optional ``effect`` and ``target`` keyword aliases make policy rules
    easy to load from configurations that use those common names.  The
    canonical public attributes remain ``decision`` and ``selector``.
    """

    scope: PolicyScope | str
    decision: PolicyDecision | str | None = None
    selector: str | Sequence[str] | None = None
    policy_id: str = "policy"
    priority: int = 0
    inherit: bool = True
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict, repr=False)
    effect: PolicyDecision | str | None = dataclass_field(default=None, repr=False)
    target: str | Sequence[str] | None = dataclass_field(default=None, repr=False)

    def __post_init__(self) -> None:
        scope = _coerce_scope(self.scope)
        decision_value = self.decision
        if decision_value is None:
            decision_value = self.effect
        elif self.effect is not None and _coerce_decision(
            decision_value
        ) != _coerce_decision(self.effect):
            raise ValueError("decision and effect must agree")
        if decision_value is None:
            raise ValueError("decision must be provided")
        decision = _coerce_decision(decision_value)

        selector_value = self.selector
        if selector_value is None:
            selector_value = self.target
        elif self.target is not None:
            if scope is PolicyScope.RESOURCE:
                selectors_agree = _normalise_path(selector_value, "selector") == (
                    _normalise_path(self.target, "target")
                )
            else:
                if not isinstance(selector_value, str) or not isinstance(
                    self.target, str
                ):
                    raise TypeError("selector must be a string for this scope")
                selectors_agree = _normalise_component(
                    selector_value, "selector"
                ) == _normalise_component(self.target, "target")
            if not selectors_agree:
                raise ValueError("selector and target must agree")
        if selector_value is None:
            raise ValueError("selector must be provided")

        if scope is PolicyScope.RESOURCE:
            selector: str | tuple[str, ...] = _normalise_path(
                selector_value, "selector"
            )
        else:
            if not isinstance(selector_value, str):
                raise TypeError("selector must be a string for this scope")
            selector = _normalise_component(selector_value, "selector")

        policy_id = _normalise_component(self.policy_id, "policy_id")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")
        if not isinstance(self.inherit, bool):
            raise TypeError("inherit must be a boolean")
        metadata = _normalise_metadata(self.metadata)

        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "selector", selector)
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "effect", decision)
        object.__setattr__(self, "target", selector)

    @classmethod
    def for_field(
        cls,
        selector: str,
        decision: PolicyDecision | str,
        *,
        policy_id: str = "policy",
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PrivacyPolicy":
        """Create a field-scoped policy rule."""

        return cls(
            PolicyScope.FIELD,
            decision,
            selector,
            policy_id=policy_id,
            priority=priority,
            inherit=False,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def for_resource(
        cls,
        selector: str | Sequence[str],
        decision: PolicyDecision | str,
        *,
        policy_id: str = "policy",
        priority: int = 0,
        inherit: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PrivacyPolicy":
        """Create a resource-scoped policy rule."""

        return cls(
            PolicyScope.RESOURCE,
            decision,
            selector,
            policy_id=policy_id,
            priority=priority,
            inherit=inherit,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def for_transport(
        cls,
        selector: str,
        decision: PolicyDecision | str,
        *,
        policy_id: str = "policy",
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PrivacyPolicy":
        """Create a transport-scoped policy rule."""

        return cls(
            PolicyScope.TRANSPORT,
            decision,
            selector,
            policy_id=policy_id,
            priority=priority,
            inherit=False,
            metadata={} if metadata is None else metadata,
        )

    @property
    def selector_parts(self) -> tuple[str, ...]:
        """Return the normalized selector as path components."""

        selector = self.selector
        if isinstance(selector, tuple):
            return selector
        if isinstance(selector, str):
            return (selector,)
        raise TypeError("policy selector is not normalized")

    @property
    def selector_fingerprint(self) -> str:
        """Return a stable fingerprint for the selector without exposing it."""

        return _fingerprint(self.selector_parts)

    @property
    def fingerprint(self) -> str:
        """Return the stable content fingerprint for this rule."""

        scope = cast(PolicyScope, self.scope)
        decision = cast(PolicyDecision, self.decision)
        return _fingerprint(
            {
                "schema_version": POLICY_COMPOSITION_SCHEMA_VERSION,
                "scope": scope.value,
                "decision": decision.value,
                "selector": self.selector_parts,
                "policy_id": self.policy_id,
                "priority": self.priority,
                "inherit": self.inherit,
                "metadata": self.metadata,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free representation suitable for audit output."""

        scope = cast(PolicyScope, self.scope)
        decision = cast(PolicyDecision, self.decision)
        return {
            "schema_version": POLICY_COMPOSITION_SCHEMA_VERSION,
            "scope": scope.value,
            "decision": decision.value,
            "selector_fingerprint": self.selector_fingerprint,
            "policy_id_fingerprint": _fingerprint(self.policy_id),
            "priority": self.priority,
            "inherit": self.inherit,
            "metadata_fingerprint": _fingerprint(self.metadata),
            "policy_fingerprint": self.fingerprint,
        }

    def __repr__(self) -> str:
        scope = cast(PolicyScope, self.scope)
        decision = cast(PolicyDecision, self.decision)
        return (
            "PrivacyPolicy("
            f"scope={scope.value!r}, decision={decision.value!r}, "
            f"fingerprint={self.fingerprint!r})"
        )


@dataclass(frozen=True, repr=False)
class PolicyContext:
    """Evaluation context for a resource, field, and transport."""

    resource: str | Sequence[str] | None = None
    field: str | None = dataclass_field(default=None, repr=False)
    transport: str | None = dataclass_field(default=None, repr=False)

    def __post_init__(self) -> None:
        resource = (
            () if self.resource is None else _normalise_path(self.resource, "resource")
        )
        field = (
            None if self.field is None else _normalise_component(self.field, "field")
        )
        transport = (
            None
            if self.transport is None
            else _normalise_component(self.transport, "transport")
        )
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "transport", transport)

    @property
    def resource_path(self) -> tuple[str, ...]:
        """Return the normalized resource path for matching."""

        return cast(tuple[str, ...], self.resource)

    @property
    def fingerprint(self) -> str:
        """Return a stable context fingerprint without exposing context values."""

        return _fingerprint(
            {
                "resource": self.resource,
                "field": self.field,
                "transport": self.transport,
            }
        )

    def to_dict(self) -> dict[str, str]:
        """Return the value-free context representation."""

        return {"context_fingerprint": self.fingerprint}

    def __repr__(self) -> str:
        return f"PolicyContext(fingerprint={self.fingerprint!r})"


@dataclass(frozen=True)
class PolicyTraceEntry:
    """Value-free evidence for one matching policy rule."""

    policy_fingerprint: str
    selector_fingerprint: str
    scope: PolicyScope
    decision: PolicyDecision
    inherited: bool
    specificity: int
    priority: int
    precedence_rank: int
    selected: bool
    shadowed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible trace entry."""

        return {
            "policy_fingerprint": self.policy_fingerprint,
            "selector_fingerprint": self.selector_fingerprint,
            "scope": self.scope.value,
            "decision": self.decision.value,
            "inherited": self.inherited,
            "specificity": self.specificity,
            "priority": self.priority,
            "precedence_rank": self.precedence_rank,
            "selected": self.selected,
            "shadowed": self.shadowed,
        }


@dataclass(frozen=True, repr=False)
class PolicyDecisionTrace:
    """Stable, value-free explanation of one composed decision."""

    decision: PolicyDecision
    conflict_category: ConflictCategory
    context_fingerprint: str
    policy_set_fingerprint: str
    policy_fingerprints: tuple[str, ...]
    selected_policy_fingerprint: str | None
    defaulted: bool
    precedence: tuple[PolicyScope, ...]
    entries: tuple[PolicyTraceEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the trace as a value-free JSON-compatible mapping."""

        return {
            "schema_version": POLICY_COMPOSITION_SCHEMA_VERSION,
            "decision": self.decision.value,
            "conflict_category": self.conflict_category.value,
            "context_fingerprint": self.context_fingerprint,
            "policy_set_fingerprint": self.policy_set_fingerprint,
            "policy_fingerprints": list(self.policy_fingerprints),
            "selected_policy_fingerprint": self.selected_policy_fingerprint,
            "defaulted": self.defaulted,
            "precedence": [scope.value for scope in self.precedence],
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self) -> str:
        """Return deterministic JSON for logs or audit storage."""

        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        return (
            "PolicyDecisionTrace("
            f"decision={self.decision.value!r}, "
            f"conflict_category={self.conflict_category.value!r}, "
            f"selected_policy_fingerprint={self.selected_policy_fingerprint!r})"
        )


@dataclass(frozen=True, repr=False)
class PolicyDecisionResult:
    """Effective decision and its value-free decision trace."""

    decision: PolicyDecision
    trace: PolicyDecisionTrace

    @property
    def effective_decision(self) -> PolicyDecision:
        """Return the composed decision."""

        return self.decision

    @property
    def allowed(self) -> bool:
        """Whether the effective decision allows the requested operation."""

        return self.decision is PolicyDecision.ALLOW

    @property
    def denied(self) -> bool:
        """Whether the effective decision denies the requested operation."""

        return not self.allowed

    @property
    def decision_trace(self) -> PolicyDecisionTrace:
        """Return the decision trace under its descriptive alias."""

        return self.trace

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free decision report."""

        return {
            "decision": self.decision.value,
            "allowed": self.allowed,
            "trace": self.trace.to_dict(),
        }

    def to_json(self) -> str:
        """Return deterministic JSON for logs or audit storage."""

        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        return (
            "PolicyDecisionResult("
            f"decision={self.decision.value!r}, trace={self.trace!r})"
        )


@dataclass(frozen=True, repr=False)
class PolicySet:
    """Immutable collection of rules and explicit composition settings."""

    policies: Iterable[PrivacyPolicy | Mapping[str, Any]] = dataclass_field(
        default_factory=tuple
    )
    default_decision: PolicyDecision | str = PolicyDecision.DENY
    precedence: Sequence[PolicyScope | str] = DEFAULT_SCOPE_PRECEDENCE

    def __post_init__(self) -> None:
        if isinstance(self.policies, (PrivacyPolicy, Mapping)):
            policy_values: tuple[PrivacyPolicy | Mapping[str, Any], ...] = (
                cast(PrivacyPolicy | Mapping[str, Any], self.policies),
            )
        else:
            try:
                policy_values = tuple(
                    cast(Iterable[PrivacyPolicy | Mapping[str, Any]], self.policies)
                )
            except TypeError as exc:
                raise TypeError("policies must be an iterable") from exc
        policies = tuple(_coerce_policy(value) for value in policy_values)
        default_decision = _coerce_decision(self.default_decision)
        try:
            precedence = tuple(_coerce_scope(value) for value in self.precedence)
        except TypeError as exc:
            raise TypeError("precedence must be an iterable of scopes") from exc
        if set(precedence) != set(PolicyScope) or len(precedence) != len(PolicyScope):
            raise ValueError("precedence must contain each policy scope exactly once")
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "default_decision", default_decision)
        object.__setattr__(self, "precedence", precedence)

    @property
    def fingerprint(self) -> str:
        """Return a stable fingerprint for the complete policy set."""

        policies = cast(tuple[PrivacyPolicy, ...], self.policies)
        precedence = cast(tuple[PolicyScope, ...], self.precedence)
        return _fingerprint(
            {
                "schema_version": POLICY_COMPOSITION_SCHEMA_VERSION,
                "default_decision": cast(PolicyDecision, self.default_decision).value,
                "precedence": [scope.value for scope in precedence],
                "policies": sorted(policy.fingerprint for policy in policies),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free policy-set summary."""

        policies = cast(tuple[PrivacyPolicy, ...], self.policies)
        precedence = cast(tuple[PolicyScope, ...], self.precedence)
        return {
            "schema_version": POLICY_COMPOSITION_SCHEMA_VERSION,
            "default_decision": cast(PolicyDecision, self.default_decision).value,
            "precedence": [scope.value for scope in precedence],
            "policy_fingerprints": sorted(policy.fingerprint for policy in policies),
            "policy_set_fingerprint": self.fingerprint,
        }

    def evaluate(
        self, context: PolicyContext | Mapping[str, Any] | None = None
    ) -> PolicyDecisionResult:
        """Evaluate this policy set against a value-bearing context locally."""

        evaluation_context = _coerce_context(context)
        policies = cast(tuple[PrivacyPolicy, ...], self.policies)
        precedence = cast(tuple[PolicyScope, ...], self.precedence)
        scope_ranks = {
            scope: len(precedence) - index for index, scope in enumerate(precedence)
        }
        candidates: list[_Candidate] = []
        for policy in policies:
            match = _match_policy(policy, evaluation_context)
            if match is None:
                continue
            inherited, specificity = match
            policy_scope = cast(PolicyScope, policy.scope)
            candidates.append(
                _Candidate(
                    policy=policy,
                    inherited=inherited,
                    specificity=specificity,
                    precedence_rank=scope_ranks[policy_scope],
                )
            )

        candidates.sort(key=_candidate_sort_key)
        denies = [
            candidate
            for candidate in candidates
            if candidate.policy.decision is PolicyDecision.DENY
        ]
        allows = [
            candidate
            for candidate in candidates
            if candidate.policy.decision is PolicyDecision.ALLOW
        ]

        if denies:
            winner = denies[0]
            decision = PolicyDecision.DENY
            if allows:
                category = ConflictCategory.DENY_OVERRIDES
            elif len(denies) > 1:
                category = ConflictCategory.MULTIPLE_DENIES
            elif winner.inherited:
                category = ConflictCategory.INHERITED_DENY
            else:
                category = ConflictCategory.NONE
        elif allows:
            winner = allows[0]
            decision = PolicyDecision.ALLOW
            if len(allows) > 1:
                category = ConflictCategory.PRECEDENCE
            elif winner.inherited:
                category = ConflictCategory.INHERITED_ALLOW
            else:
                category = ConflictCategory.NONE
        else:
            winner = None
            decision = cast(PolicyDecision, self.default_decision)
            category = ConflictCategory.DEFAULT

        entries = tuple(
            PolicyTraceEntry(
                policy_fingerprint=candidate.policy.fingerprint,
                selector_fingerprint=candidate.policy.selector_fingerprint,
                scope=cast(PolicyScope, candidate.policy.scope),
                decision=cast(PolicyDecision, candidate.policy.decision),
                inherited=candidate.inherited,
                specificity=candidate.specificity,
                priority=candidate.policy.priority,
                precedence_rank=candidate.precedence_rank,
                selected=candidate is winner,
                shadowed=candidate is not winner,
            )
            for candidate in candidates
        )
        trace = PolicyDecisionTrace(
            decision=decision,
            conflict_category=category,
            context_fingerprint=evaluation_context.fingerprint,
            policy_set_fingerprint=self.fingerprint,
            policy_fingerprints=tuple(
                candidate.policy.fingerprint for candidate in candidates
            ),
            selected_policy_fingerprint=None
            if winner is None
            else winner.policy.fingerprint,
            defaulted=winner is None,
            precedence=precedence,
            entries=entries,
        )
        return PolicyDecisionResult(decision=decision, trace=trace)

    def __repr__(self) -> str:
        return (
            "PolicySet("
            f"policy_count={len(cast(tuple[PrivacyPolicy, ...], self.policies))}, "
            f"fingerprint={self.fingerprint!r})"
        )


@dataclass(frozen=True)
class _Candidate:
    policy: PrivacyPolicy
    inherited: bool
    specificity: int
    precedence_rank: int


def _candidate_sort_key(candidate: _Candidate) -> tuple[int, int, int, str]:
    return (
        -candidate.precedence_rank,
        -candidate.specificity,
        -candidate.policy.priority,
        candidate.policy.fingerprint,
    )


def _coerce_policy(value: PrivacyPolicy | Mapping[str, Any]) -> PrivacyPolicy:
    if isinstance(value, PrivacyPolicy):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each policy must be a PrivacyPolicy or mapping")

    scope = value.get("scope")
    if scope is None:
        raise ValueError("policy scope is required")
    scope_value = _coerce_scope(scope)
    decision = value.get("decision")
    if decision is None:
        decision = value.get("effect", value.get("action"))
    selector = value.get("selector")
    if selector is None:
        selector = value.get("target", value.get(scope_value.value))
    if selector is None and scope_value is PolicyScope.RESOURCE:
        selector = value.get("path")
    if decision is None or selector is None:
        raise ValueError("policy decision and selector are required")
    return PrivacyPolicy(
        scope=scope_value,
        decision=decision,
        selector=selector,
        policy_id=value.get("policy_id", value.get("name", "policy")),
        priority=value.get("priority", 0),
        inherit=value.get("inherit", True),
        metadata=value.get("metadata", {}),
    )


def _coerce_context(
    context: PolicyContext | Mapping[str, Any] | None,
) -> PolicyContext:
    if context is None:
        return PolicyContext()
    if isinstance(context, PolicyContext):
        return context
    if isinstance(context, Mapping):
        return PolicyContext(
            resource=context.get("resource"),
            field=context.get("field"),
            transport=context.get("transport"),
        )
    raise TypeError("context must be a PolicyContext or mapping")


def _match_value(selector: str, value: str | None) -> tuple[bool, int]:
    if value is None:
        return False, 0
    if selector == "*":
        return True, 0
    return selector == value, 2


def _match_resource(
    selector: tuple[str, ...],
    resource: tuple[str, ...],
    *,
    inherit: bool,
) -> tuple[bool, bool, int]:
    if not resource or len(selector) > len(resource):
        return False, False, 0
    has_globstar = selector[-1] == "**"
    fixed_selector = selector[:-1] if has_globstar else selector
    if not inherit and (has_globstar or len(selector) != len(resource)):
        return False, False, 0
    if len(fixed_selector) > len(resource):
        return False, False, 0
    for index, part in enumerate(fixed_selector):
        if part != "*" and part != resource[index]:
            return False, False, 0
    if not has_globstar and len(selector) != len(resource) and not inherit:
        return False, False, 0
    if has_globstar and len(resource) == len(fixed_selector) and not inherit:
        return False, False, 0
    inherited = len(resource) > len(fixed_selector)
    literal_count = sum(part != "*" for part in fixed_selector)
    specificity = literal_count * 2 + len(fixed_selector)
    if not inherited and not has_globstar:
        specificity += 1
    return True, inherited, specificity


def _match_policy(
    policy: PrivacyPolicy,
    context: PolicyContext,
) -> tuple[bool, int] | None:
    if policy.scope is PolicyScope.FIELD:
        matched, specificity = _match_value(policy.selector_parts[0], context.field)
        return (False, specificity) if matched else None
    if policy.scope is PolicyScope.TRANSPORT:
        matched, specificity = _match_value(policy.selector_parts[0], context.transport)
        return (False, specificity) if matched else None
    matched, inherited, specificity = _match_resource(
        policy.selector_parts,
        context.resource_path,
        inherit=policy.inherit,
    )
    return (inherited, specificity) if matched else None


def compose_policies(
    policies: Iterable[PrivacyPolicy | Mapping[str, Any]] | PolicySet = (),
    *,
    context: PolicyContext | Mapping[str, Any] | None = None,
    resource: str | Sequence[str] | None = None,
    field: str | None = None,
    transport: str | None = None,
    default_decision: PolicyDecision | str = PolicyDecision.DENY,
    precedence: Sequence[PolicyScope | str] = DEFAULT_SCOPE_PRECEDENCE,
) -> PolicyDecisionResult:
    """Compose policies and return a deterministic, value-free decision.

    ``context`` may be supplied as :class:`PolicyContext` or as a mapping.  A
    context can instead be built from the ``resource``, ``field``, and
    ``transport`` keyword arguments.  No network, filesystem, or model access
    occurs during composition.
    """

    if context is not None and any(
        value is not None for value in (resource, field, transport)
    ):
        raise ValueError("use context or resource, field, and transport, not both")
    evaluation_context = context
    if evaluation_context is None:
        evaluation_context = PolicyContext(
            resource=resource,
            field=field,
            transport=transport,
        )

    if isinstance(policies, PolicySet):
        policy_set = policies
    else:
        policy_set = PolicySet(
            policies=policies,
            default_decision=default_decision,
            precedence=precedence,
        )
    return policy_set.evaluate(evaluation_context)


def evaluate_policy(
    policies: Iterable[PrivacyPolicy | Mapping[str, Any]] | PolicySet = (),
    **kwargs: Any,
) -> PolicyDecisionResult:
    """Alias for :func:`compose_policies` with an evaluation-oriented name."""

    return compose_policies(policies, **kwargs)


def policy_fingerprint(policy: PrivacyPolicy | Mapping[str, Any]) -> str:
    """Return the stable fingerprint of one policy rule."""

    return _coerce_policy(policy).fingerprint


# Small aliases keep the vocabulary convenient for callers that use
# ``PolicyRule``/``PolicyEffect`` terminology while retaining one implementation.
Policy = PrivacyPolicy
PolicyRule = PrivacyPolicy
PolicyEffect = PolicyDecision
Decision = PolicyDecision
DecisionTrace = PolicyDecisionTrace
PolicyComposition = PolicySet
compose_policy = compose_policies
compose_privacy_policies = compose_policies


__all__ = [
    "POLICY_COMPOSITION_SCHEMA_VERSION",
    "DEFAULT_SCOPE_PRECEDENCE",
    "ConflictCategory",
    "Decision",
    "DecisionTrace",
    "Policy",
    "PolicyComposition",
    "PolicyContext",
    "PolicyDecision",
    "PolicyDecisionResult",
    "PolicyDecisionTrace",
    "PolicyEffect",
    "PolicyRule",
    "PolicyScope",
    "PolicySet",
    "PrivacyPolicy",
    "compose_policy",
    "compose_policies",
    "compose_privacy_policies",
    "evaluate_policy",
    "policy_fingerprint",
]
