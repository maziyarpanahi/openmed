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
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

POLICY_COMPOSITION_SCHEMA_VERSION = 1

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_POLICIES = 4096
_MAX_PATH_PARTS = 64
_MAX_METADATA_ITEMS = 4096
_MAX_METADATA_DEPTH = 16
_MAX_COMPONENT_CHARS = 512
_MAX_METADATA_TEXT_CHARS = 16_384
_MAX_METADATA_INT_BITS = 4096
_MAX_PRIORITY = 2**31 - 1
_POLICY_MAPPING_KEYS = frozenset(
    {
        "scope",
        "decision",
        "effect",
        "action",
        "selector",
        "target",
        "field",
        "resource",
        "transport",
        "path",
        "policy_id",
        "name",
        "priority",
        "inherit",
        "metadata",
    }
)
_CONTEXT_MAPPING_KEYS = frozenset({"resource", "field", "transport"})


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


def _bounded_iterable(value: Iterable[Any], *, limit: int, label: str) -> list[Any]:
    try:
        iterator = iter(value)
    except MemoryError:
        raise
    except Exception:
        raise ValueError(f"{label} is invalid") from None
    result: list[Any] = []
    for _ in range(limit + 1):
        try:
            result.append(next(iterator))
        except StopIteration:
            return result
        except MemoryError:
            raise
        except Exception:
            raise ValueError(f"{label} is invalid") from None
    raise ValueError(f"{label} exceeds the item limit")


def _bounded_mapping_items(
    value: Mapping[Any, Any], *, limit: int = _MAX_METADATA_ITEMS
) -> list[tuple[Any, Any]]:
    try:
        raw_items = value.items()
    except MemoryError:
        raise
    except Exception:
        raise ValueError("policy mapping is invalid") from None
    items = _bounded_iterable(raw_items, limit=limit, label="policy mapping")
    if not all(isinstance(item, tuple) and len(item) == 2 for item in items):
        raise ValueError("policy mapping is invalid")
    return cast(list[tuple[Any, Any]], items)


def _copy_mapping(
    value: Mapping[Any, Any], *, allowed: frozenset[str], label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in _bounded_mapping_items(value, limit=len(allowed)):
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"{label} contains unsupported fields")
        if key in result:
            raise ValueError(f"{label} contains duplicate fields")
        result[key] = item
    return result


def _one_alias(value: Mapping[str, Any], *keys: str, required: bool = False) -> Any:
    present = [key for key in keys if key in value]
    if len(present) > 1:
        raise ValueError("policy mapping contains ambiguous aliases")
    if present:
        return value[present[0]]
    if required:
        raise ValueError("policy mapping is missing a required field")
    return None


def _validate_fingerprint(value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _FINGERPRINT_RE.fullmatch(value) is None:
        raise ValueError("policy fingerprint is invalid")


def _coerce_scope(value: PolicyScope | str) -> PolicyScope:
    if isinstance(value, PolicyScope):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_COMPONENT_CHARS:
            raise ValueError("scope exceeds the length limit")
        candidate = value.strip().lower()
        for scope in PolicyScope:
            if candidate == scope.value:
                return scope
    raise ValueError("scope must be one of field, resource, or transport")


def _coerce_decision(value: PolicyDecision | str) -> PolicyDecision:
    if isinstance(value, PolicyDecision):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_COMPONENT_CHARS:
            raise ValueError("decision exceeds the length limit")
        candidate = value.strip().lower()
        for decision in PolicyDecision:
            if candidate == decision.value:
                return decision
    raise ValueError("decision must be allow or deny")


def _normalise_component(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) > _MAX_COMPONENT_CHARS:
        raise ValueError(f"{name} exceeds the length limit")
    normalised = unicodedata.normalize("NFC", value.strip())
    if not normalised or len(normalised) > _MAX_COMPONENT_CHARS:
        raise ValueError(f"{name} must be non-empty")
    return normalised


def _normalise_path(value: str | Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        if len(value) > _MAX_COMPONENT_CHARS * _MAX_PATH_PARTS:
            raise ValueError(f"{name} exceeds the length limit")
        parts = value.split("/")
    elif isinstance(value, Sequence):
        parts = _bounded_iterable(value, limit=_MAX_PATH_PARTS, label=name)
    else:
        raise TypeError(f"{name} must be a path string or sequence")

    if not parts or len(parts) > _MAX_PATH_PARTS:
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
    canonical = _canonical_value(value)
    _dump_canonical(canonical)
    return cast(Mapping[str, Any], _freeze_value(canonical))


def _canonical_value(
    value: Any, *, depth: int = 0, seen: set[int] | None = None
) -> Any:
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("policy metadata exceeds the nesting limit")
    if isinstance(value, Enum):
        return _canonical_value(value.value, depth=depth + 1, seen=seen)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > _MAX_METADATA_INT_BITS:
            raise ValueError("policy metadata integer exceeds the size limit")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("policy metadata numbers must be finite")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_METADATA_TEXT_CHARS:
            raise ValueError("policy metadata text exceeds the length limit")
        return unicodedata.normalize("NFC", value)
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            raise ValueError("policy metadata contains a cycle")
        seen.add(marker)
        items: dict[str, Any] = {}
        try:
            for key, item in _bounded_mapping_items(value):
                if not isinstance(key, str):
                    raise TypeError("metadata keys must be strings")
                if len(key) > _MAX_COMPONENT_CHARS:
                    raise ValueError("metadata key exceeds the length limit")
                normalised_key = unicodedata.normalize("NFC", key)
                if normalised_key in items:
                    raise ValueError("metadata keys collide after normalization")
                items[normalised_key] = _canonical_value(
                    item, depth=depth + 1, seen=seen
                )
            return {key: items[key] for key in sorted(items)}
        finally:
            seen.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            raise ValueError("policy metadata contains a cycle")
        seen.add(marker)
        try:
            return [
                _canonical_value(item, depth=depth + 1, seen=seen)
                for item in _bounded_iterable(
                    value, limit=_MAX_METADATA_ITEMS, label="policy metadata"
                )
            ]
        finally:
            seen.remove(marker)
    if isinstance(value, (set, frozenset)):
        values = [
            _canonical_value(item, depth=depth + 1, seen=seen)
            for item in _bounded_iterable(
                value, limit=_MAX_METADATA_ITEMS, label="policy metadata"
            )
        ]
        return sorted(values, key=_dump_canonical)
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
        return _dump_canonical(_canonical_value(value))
    except MemoryError:
        raise
    except (TypeError, ValueError):
        raise TypeError("policy values must be JSON-compatible") from None


def _dump_canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


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
        if abs(self.priority) > _MAX_PRIORITY:
            raise ValueError("priority exceeds the supported range")
        if not isinstance(self.inherit, bool):
            raise TypeError("inherit must be a boolean")
        inherit = self.inherit if scope is PolicyScope.RESOURCE else False
        metadata = _normalise_metadata(self.metadata)

        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "selector", selector)
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "inherit", inherit)
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

    def __post_init__(self) -> None:
        """Reject trace entries that are unsafe or internally inconsistent."""

        _validate_fingerprint(self.policy_fingerprint)
        _validate_fingerprint(self.selector_fingerprint)
        if not isinstance(self.scope, PolicyScope):
            raise TypeError("trace entry scope must be a PolicyScope")
        if not isinstance(self.decision, PolicyDecision):
            raise TypeError("trace entry decision must be a PolicyDecision")
        if type(self.inherited) is not bool:
            raise TypeError("trace entry inherited must be a boolean")
        if type(self.selected) is not bool or type(self.shadowed) is not bool:
            raise TypeError("trace entry selection flags must be booleans")
        if self.selected is self.shadowed:
            raise ValueError("trace entry selection flags are inconsistent")
        if (
            type(self.specificity) is not int
            or not 0 <= self.specificity <= _MAX_PATH_PARTS * 3 + 1
        ):
            raise ValueError("trace entry specificity must be non-negative")
        if type(self.priority) is not int or abs(self.priority) > _MAX_PRIORITY:
            raise ValueError("trace entry priority is invalid")
        if type(
            self.precedence_rank
        ) is not int or not 1 <= self.precedence_rank <= len(PolicyScope):
            raise ValueError("trace entry precedence rank is invalid")

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


def _trace_entry_sort_key(entry: PolicyTraceEntry) -> tuple[int, int, int, str]:
    return (
        -entry.precedence_rank,
        -entry.specificity,
        -entry.priority,
        entry.policy_fingerprint,
    )


def _conflict_category_for_entries(
    entries: Sequence[PolicyTraceEntry],
) -> ConflictCategory:
    denies = [entry for entry in entries if entry.decision is PolicyDecision.DENY]
    allows = [entry for entry in entries if entry.decision is PolicyDecision.ALLOW]
    if denies and allows:
        return ConflictCategory.DENY_OVERRIDES
    if len(denies) > 1:
        return ConflictCategory.MULTIPLE_DENIES
    if len(allows) > 1:
        return ConflictCategory.PRECEDENCE
    selected = next(entry for entry in entries if entry.selected)
    if selected.inherited:
        return (
            ConflictCategory.INHERITED_DENY
            if selected.decision is PolicyDecision.DENY
            else ConflictCategory.INHERITED_ALLOW
        )
    return ConflictCategory.NONE


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

    def __post_init__(self) -> None:
        """Enforce deterministic, value-free trace invariants."""

        if not isinstance(self.decision, PolicyDecision):
            raise TypeError("trace decision must be a PolicyDecision")
        if not isinstance(self.conflict_category, ConflictCategory):
            raise TypeError("trace conflict category must be a ConflictCategory")
        _validate_fingerprint(self.context_fingerprint)
        _validate_fingerprint(self.policy_set_fingerprint)
        _validate_fingerprint(self.selected_policy_fingerprint, optional=True)
        if type(self.defaulted) is not bool:
            raise TypeError("trace defaulted must be a boolean")
        if (
            type(self.precedence) is not tuple
            or len(self.precedence) != len(PolicyScope)
            or not all(isinstance(scope, PolicyScope) for scope in self.precedence)
            or set(self.precedence) != set(PolicyScope)
        ):
            raise ValueError("trace precedence is invalid")
        if type(self.entries) is not tuple:
            raise TypeError("trace entries must be PolicyTraceEntry records")
        if len(self.entries) > _MAX_POLICIES:
            raise ValueError("trace exceeds the entry limit")
        if not all(isinstance(entry, PolicyTraceEntry) for entry in self.entries):
            raise TypeError("trace entries must be PolicyTraceEntry records")
        if type(self.policy_fingerprints) is not tuple:
            raise TypeError("trace policy fingerprints must be a tuple")
        if len(self.policy_fingerprints) > _MAX_POLICIES:
            raise ValueError("trace exceeds the fingerprint limit")
        for fingerprint in self.policy_fingerprints:
            _validate_fingerprint(fingerprint)
        entry_fingerprints = tuple(entry.policy_fingerprint for entry in self.entries)
        if self.policy_fingerprints != entry_fingerprints or len(
            set(entry_fingerprints)
        ) != len(entry_fingerprints):
            raise ValueError("trace policy fingerprints are inconsistent")
        if list(self.entries) != sorted(self.entries, key=_trace_entry_sort_key):
            raise ValueError("trace entries are not deterministically ordered")
        scope_ranks = {
            scope: len(self.precedence) - index
            for index, scope in enumerate(self.precedence)
        }
        if any(
            entry.precedence_rank != scope_ranks[entry.scope] for entry in self.entries
        ):
            raise ValueError("trace precedence ranks are inconsistent")

        selected = [entry for entry in self.entries if entry.selected]
        if self.defaulted:
            if self.entries or selected or self.selected_policy_fingerprint is not None:
                raise ValueError("defaulted trace contains selected policies")
            expected_category = ConflictCategory.DEFAULT
        else:
            if len(selected) != 1:
                raise ValueError("trace must contain exactly one selected policy")
            if self.selected_policy_fingerprint != selected[0].policy_fingerprint:
                raise ValueError("trace selected policy fingerprint is inconsistent")
            if self.decision is not selected[0].decision:
                raise ValueError("trace decision does not match selected policy")
            if (
                any(entry.decision is PolicyDecision.DENY for entry in self.entries)
                and self.decision is not PolicyDecision.DENY
            ):
                raise ValueError("trace does not apply deny-overrides")
            expected_category = _conflict_category_for_entries(self.entries)
        if self.conflict_category is not expected_category:
            raise ValueError("trace conflict category is inconsistent")

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

    def __post_init__(self) -> None:
        """Require the effective decision and trace to agree."""

        if not isinstance(self.decision, PolicyDecision):
            raise TypeError("result decision must be a PolicyDecision")
        if not isinstance(self.trace, PolicyDecisionTrace):
            raise TypeError("result trace must be a PolicyDecisionTrace")
        if self.decision is not self.trace.decision:
            raise ValueError("result decision does not match its trace")

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
            policy_values = tuple(
                cast(
                    list[PrivacyPolicy | Mapping[str, Any]],
                    _bounded_iterable(
                        cast(Iterable[Any], self.policies),
                        limit=_MAX_POLICIES,
                        label="policies",
                    ),
                )
            )
        policies = tuple(_coerce_policy(value) for value in policy_values)
        fingerprints = [policy.fingerprint for policy in policies]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("policies must not contain duplicate rules")
        default_decision = _coerce_decision(self.default_decision)
        precedence = tuple(
            _coerce_scope(value)
            for value in _bounded_iterable(
                self.precedence,
                limit=len(PolicyScope),
                label="precedence",
            )
        )
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

    policy = _copy_mapping(value, allowed=_POLICY_MAPPING_KEYS, label="policy mapping")
    scope = policy.get("scope")
    if scope is None:
        raise ValueError("policy scope is required")
    scope_value = _coerce_scope(scope)
    decision = _one_alias(policy, "decision", "effect", "action", required=True)
    selector_keys = ["selector", "target", scope_value.value]
    if scope_value is PolicyScope.RESOURCE:
        selector_keys.append("path")
    all_selector_keys = {"selector", "target", "field", "resource", "transport", "path"}
    if (set(policy) & all_selector_keys) - set(selector_keys):
        raise ValueError("policy mapping contains an invalid selector alias")
    selector = _one_alias(policy, *selector_keys, required=True)
    if decision is None or selector is None:
        raise ValueError("policy decision and selector are required")
    policy_id = _one_alias(policy, "policy_id", "name")
    if policy_id is None:
        policy_id = "policy"
    if scope_value is not PolicyScope.RESOURCE and "inherit" in policy:
        raise ValueError("inherit is only valid for resource policies")
    return PrivacyPolicy(
        scope=scope_value,
        decision=decision,
        selector=selector,
        policy_id=policy_id,
        priority=policy.get("priority", 0),
        inherit=policy.get("inherit", scope_value is PolicyScope.RESOURCE),
        metadata=policy.get("metadata", {}),
    )


def _coerce_context(
    context: PolicyContext | Mapping[str, Any] | None,
) -> PolicyContext:
    if context is None:
        return PolicyContext()
    if isinstance(context, PolicyContext):
        return context
    if isinstance(context, Mapping):
        context_value = _copy_mapping(
            context, allowed=_CONTEXT_MAPPING_KEYS, label="policy context"
        )
        return PolicyContext(
            resource=context_value.get("resource"),
            field=context_value.get("field"),
            transport=context_value.get("transport"),
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


def _compose_policies(
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
    """Compose policies into a deterministic, value-free decision trace."""

    try:
        return _compose_policies(
            policies,
            context=context,
            resource=resource,
            field=field,
            transport=transport,
            default_decision=default_decision,
            precedence=precedence,
        )
    except MemoryError:
        raise
    except Exception:
        raise ValueError("policy composition inputs are invalid") from None


def evaluate_policy(
    policies: Iterable[PrivacyPolicy | Mapping[str, Any]] | PolicySet = (),
    **kwargs: Any,
) -> PolicyDecisionResult:
    """Alias for :func:`compose_policies` with an evaluation-oriented name."""

    return compose_policies(policies, **kwargs)


def policy_fingerprint(policy: PrivacyPolicy | Mapping[str, Any]) -> str:
    """Return the stable fingerprint of one policy rule."""

    try:
        return _coerce_policy(policy).fingerprint
    except MemoryError:
        raise
    except Exception:
        raise ValueError("policy is invalid") from None


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
