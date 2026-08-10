"""Offline, aggregate-only simulation of policy-version changes.

Policy impact review must be possible before a policy is promoted.  This
module evaluates two immutable policy descriptions over typed resource counts
and reports only action, gate, and waiver transitions.  Resource identifiers,
values, and unknown policy fields are never copied into the result.

The simulator is deliberately separate from the live de-identification and
budget paths.  It reads policy-like mappings and loaded :class:`PolicyProfile`
objects, but never calls a mutating method or writes a policy or budget.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

from openmed.core.audit import stable_hash
from openmed.core.policy import PolicyName, PolicyProfile, load_policy
from openmed.core.schemas.span import ACTION_KEEP

CURRENT_POLICY_IMPACT_SCHEMA_VERSION = 1

_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_RESOURCE_TYPE_KEYS = ("resource_type", "type", "kind", "label")

GateValue: TypeAlias = bool | str | tuple[str, ...]


@dataclass(frozen=True)
class TypedResource:
    """A count of one safe, typed resource kind.

    ``count`` lets callers provide an already aggregated synthetic inventory.
    Resource identifiers and resource payloads are intentionally not fields on
    this type.
    """

    resource_type: str
    count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_type",
            _safe_token(self.resource_type, field_name="resource_type"),
        )
        object.__setattr__(
            self,
            "count",
            _count(self.count, field_name="count"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TypedResource":
        """Build a resource count from a mapping without retaining its values."""

        resource_type = _mapping_resource_type(value)
        return cls(resource_type, value.get("count", 1))


@dataclass(frozen=True)
class PolicyVersion:
    """Validated policy settings used by the offline impact simulator.

    The mappings are keyed by typed resource kind.  Missing entries use the
    corresponding default.  A waiver is reduced to an active/inactive state;
    waiver reasons are never serialized.
    """

    name: str
    actions: Mapping[str, str] = field(default_factory=dict)
    gates: Mapping[str, GateValue] = field(default_factory=dict)
    waivers: Mapping[str, bool] = field(default_factory=dict)
    default_action: str = ACTION_KEEP
    default_gate: GateValue = False
    default_waiver: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_token(self.name, field_name="name"))
        object.__setattr__(
            self,
            "actions",
            MappingProxyType(
                _normalize_value_map(
                    self.actions,
                    _normalize_action,
                    field_name="actions",
                )
            ),
        )
        object.__setattr__(
            self,
            "gates",
            MappingProxyType(
                _normalize_value_map(
                    self.gates,
                    _normalize_gate,
                    field_name="gates",
                )
            ),
        )
        object.__setattr__(
            self,
            "waivers",
            MappingProxyType(_normalize_waiver_map(self.waivers)),
        )
        object.__setattr__(
            self,
            "default_action",
            _normalize_action(self.default_action, field_name="default_action"),
        )
        object.__setattr__(
            self,
            "default_gate",
            _normalize_gate(self.default_gate, field_name="default_gate"),
        )
        object.__setattr__(
            self,
            "default_waiver",
            _normalize_waiver(self.default_waiver, field_name="default_waiver"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyVersion":
        """Build a version from a safe policy mapping.

        The preferred shape is ``actions``, ``gates``, and ``waivers`` keyed by
        resource type.  A ``resources`` mapping with per-type ``action``,
        ``gate``, and ``waiver`` fields is also accepted for compact fixtures.
        Unknown fields are ignored so sensitive policy metadata cannot enter a
        report accidentally.
        """

        if not isinstance(value, Mapping):
            raise TypeError("policy version must be a mapping")

        raw_name = value.get("name", value.get("version", "policy"))
        defaults = value.get("defaults") or {}
        if not isinstance(defaults, Mapping):
            raise ValueError("policy defaults must be a mapping")

        resource_actions, resource_gates, resource_waivers = _resource_policy_fields(
            value.get("resources")
        )
        actions = resource_actions
        gates = resource_gates
        waivers = resource_waivers

        if "actions" in value and value.get("actions") is not None:
            actions = _mapping(value.get("actions"), field_name="actions")
        if "gates" in value and value.get("gates") is not None:
            gates = _mapping(value.get("gates"), field_name="gates")
        if "waivers" in value and value.get("waivers") is not None:
            waivers = value.get("waivers")

        return cls(
            name=_policy_name(raw_name),
            actions=actions,
            gates=gates,
            waivers=waivers,
            default_action=value.get(
                "default_action",
                defaults.get("action", ACTION_KEEP),
            ),
            default_gate=value.get("default_gate", defaults.get("gate", False)),
            default_waiver=value.get(
                "default_waiver",
                defaults.get("waiver", False),
            ),
        )

    def action_for(self, resource_type: str) -> str:
        """Return the action for a normalized resource type."""

        return self.actions.get(resource_type, self.default_action)

    def gate_for(self, resource_type: str) -> GateValue:
        """Return the gate state for a normalized resource type."""

        return self.gates.get(resource_type, self.default_gate)

    def waiver_for(self, resource_type: str) -> bool:
        """Return whether a waiver is active for a normalized resource type."""

        return self.waivers.get(resource_type, self.default_waiver)

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized policy settings without waiver reasons."""

        return {
            "schema_version": CURRENT_POLICY_IMPACT_SCHEMA_VERSION,
            "name": self.name,
            "actions": dict(self.actions),
            "gates": {key: _json_gate(value) for key, value in self.gates.items()},
            "waivers": dict(self.waivers),
            "default_action": self.default_action,
            "default_gate": _json_gate(self.default_gate),
            "default_waiver": self.default_waiver,
        }


@dataclass(frozen=True)
class PolicyImpactChange:
    """A counts-only transition for one typed resource kind."""

    resource_type: str
    from_value: str | bool | tuple[str, ...]
    to_value: str | bool | tuple[str, ...]
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_type",
            _safe_token(self.resource_type, field_name="resource_type"),
        )
        object.__setattr__(
            self,
            "from_value",
            _normalize_gate(self.from_value, field_name="from_value"),
        )
        object.__setattr__(
            self,
            "to_value",
            _normalize_gate(self.to_value, field_name="to_value"),
        )
        object.__setattr__(
            self,
            "count",
            _count(self.count, field_name="count"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe transition containing no resource identity."""

        return {
            "resource_type": self.resource_type,
            "from": _json_value(self.from_value),
            "to": _json_value(self.to_value),
            "count": self.count,
        }


@dataclass(frozen=True)
class PolicyImpactDigest:
    """Aggregate action, gate, and waiver impact between two policy versions."""

    baseline_policy: str
    candidate_policy: str
    resource_count: int
    resource_type_counts: tuple[tuple[str, int], ...]
    action_deltas: tuple[PolicyImpactChange, ...]
    gate_deltas: tuple[PolicyImpactChange, ...]
    waiver_deltas: tuple[PolicyImpactChange, ...]
    unchanged_resource_count: int
    changed_resource_count: int

    @property
    def action_changed_resource_count(self) -> int:
        """Return the number of resources whose effective action changed."""

        return sum(change.count for change in self.action_deltas)

    @property
    def gate_changed_resource_count(self) -> int:
        """Return the number of resources whose effective gate changed."""

        return sum(change.count for change in self.gate_deltas)

    @property
    def waiver_changed_resource_count(self) -> int:
        """Return the number of resources whose waiver state changed."""

        return sum(change.count for change in self.waiver_deltas)

    @property
    def is_empty(self) -> bool:
        """Return whether the candidate has no effective impact."""

        return self.changed_resource_count == 0

    @property
    def digest(self) -> str:
        """Return the hash of the canonical counts-only payload."""

        return stable_hash(self.canonical_payload())

    def canonical_payload(self) -> dict[str, Any]:
        """Return the digest payload before adding its self-excluding hash."""

        return {
            "schema_version": CURRENT_POLICY_IMPACT_SCHEMA_VERSION,
            "artifact": "policy_impact_digest",
            "baseline_policy": self.baseline_policy,
            "candidate_policy": self.candidate_policy,
            "resources": {
                "total_count": self.resource_count,
                "type_counts": dict(self.resource_type_counts),
            },
            "summary": {
                "changed_resource_count": self.changed_resource_count,
                "unchanged_resource_count": self.unchanged_resource_count,
                "action_changed_resource_count": (self.action_changed_resource_count),
                "gate_changed_resource_count": self.gate_changed_resource_count,
                "waiver_changed_resource_count": (self.waiver_changed_resource_count),
            },
            "action_deltas": [change.to_dict() for change in self.action_deltas],
            "gate_deltas": [change.to_dict() for change in self.gate_deltas],
            "waiver_deltas": [change.to_dict() for change in self.waiver_deltas],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregate digest with its canonical SHA-256 hash."""

        return {**self.canonical_payload(), "digest": self.digest}

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the aggregate digest deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )


def evaluate_policy_impact(
    baseline: PolicyVersion | PolicyProfile | PolicyName | Mapping[str, Any] | str,
    candidate: PolicyVersion | PolicyProfile | PolicyName | Mapping[str, Any] | str,
    resources: Iterable[Any] | Mapping[str, Any] | TypedResource,
) -> PolicyImpactDigest:
    """Evaluate two policy versions over typed resources without side effects.

    ``resources`` may be an iterable of :class:`TypedResource` objects or
    mappings containing ``resource_type``/``type``/``kind`` and an optional
    ``count``.  A mapping of ``resource_type`` to integer counts is also
    accepted.  Only those type counts reach the returned digest.
    """

    baseline_version = _resolve_policy(baseline)
    candidate_version = _resolve_policy(candidate)
    resource_counts = _resource_counts(resources)

    action_deltas: list[PolicyImpactChange] = []
    gate_deltas: list[PolicyImpactChange] = []
    waiver_deltas: list[PolicyImpactChange] = []
    unchanged = 0
    changed = 0

    for resource_type, count in sorted(resource_counts.items()):
        before_action = baseline_version.action_for(resource_type)
        after_action = candidate_version.action_for(resource_type)
        before_gate = baseline_version.gate_for(resource_type)
        after_gate = candidate_version.gate_for(resource_type)
        before_waiver = baseline_version.waiver_for(resource_type)
        after_waiver = candidate_version.waiver_for(resource_type)

        action_changed = before_action != after_action
        gate_changed = before_gate != after_gate
        waiver_changed = before_waiver != after_waiver
        if action_changed:
            action_deltas.append(
                PolicyImpactChange(
                    resource_type,
                    before_action,
                    after_action,
                    count,
                )
            )
        if gate_changed:
            gate_deltas.append(
                PolicyImpactChange(resource_type, before_gate, after_gate, count)
            )
        if waiver_changed:
            waiver_deltas.append(
                PolicyImpactChange(
                    resource_type,
                    before_waiver,
                    after_waiver,
                    count,
                )
            )

        if action_changed or gate_changed or waiver_changed:
            changed += count
        else:
            unchanged += count

    return PolicyImpactDigest(
        baseline_policy=baseline_version.name,
        candidate_policy=candidate_version.name,
        resource_count=sum(resource_counts.values()),
        resource_type_counts=tuple(sorted(resource_counts.items())),
        action_deltas=tuple(action_deltas),
        gate_deltas=tuple(gate_deltas),
        waiver_deltas=tuple(waiver_deltas),
        unchanged_resource_count=unchanged,
        changed_resource_count=changed,
    )


def compare_policy_versions(
    baseline: PolicyVersion | PolicyProfile | PolicyName | Mapping[str, Any] | str,
    candidate: PolicyVersion | PolicyProfile | PolicyName | Mapping[str, Any] | str,
    resources: Iterable[Any] | Mapping[str, Any] | TypedResource,
) -> PolicyImpactDigest:
    """Alias for :func:`evaluate_policy_impact` used by review tooling."""

    return evaluate_policy_impact(baseline, candidate, resources)


def simulate_policy_impact(
    resources: Iterable[Any] | Mapping[str, Any] | TypedResource,
    baseline: PolicyVersion | PolicyProfile | PolicyName | Mapping[str, Any] | str,
    candidate: PolicyVersion | PolicyProfile | PolicyName | Mapping[str, Any] | str,
) -> PolicyImpactDigest:
    """Evaluate impact with resources first for pipeline-oriented callers."""

    return evaluate_policy_impact(baseline, candidate, resources)


policy_impact_digest = evaluate_policy_impact
PolicyResource = TypedResource
PolicyImpact = PolicyImpactDigest


def _resolve_policy(
    value: PolicyVersion | PolicyProfile | PolicyName | Mapping[str, Any] | str,
) -> PolicyVersion:
    if isinstance(value, PolicyVersion):
        return value
    if isinstance(value, PolicyProfile):
        return PolicyVersion(
            name=value.name,
            actions=value.actions,
            gates={
                "strict_no_leak": value.strict_no_leak,
                "safety_sweep_mandatory": value.safety_sweep_mandatory,
                "reversible_id": value.reversible_id,
            },
            default_action=value.default_action,
        )
    if isinstance(value, Mapping):
        return PolicyVersion.from_mapping(value)
    try:
        profile = load_policy(value)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("policy version could not be resolved") from None
    return _resolve_policy(profile)


def _resource_counts(
    resources: Iterable[Any] | Mapping[str, Any] | TypedResource,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for resource in _resource_items(resources):
        counts[resource.resource_type] += resource.count
    return Counter({key: count for key, count in counts.items() if count})


def _resource_items(
    resources: Iterable[Any] | Mapping[str, Any] | TypedResource,
) -> Iterable[TypedResource]:
    if isinstance(resources, TypedResource):
        return (resources,)
    if isinstance(resources, Mapping):
        if _has_resource_type_key(resources):
            return (TypedResource.from_mapping(resources),)
        return tuple(_count_mapping_items(resources))
    if isinstance(resources, str):
        return (TypedResource(resources),)
    if isinstance(resources, (bytes, bytearray)):
        raise TypeError("resources must contain typed resource records")
    try:
        return tuple(_resource_item(item) for item in resources)
    except TypeError as exc:
        raise TypeError("resources must be an iterable of typed resources") from exc


def _count_mapping_items(value: Mapping[str, Any]) -> Iterable[TypedResource]:
    items: list[TypedResource] = []
    for resource_type, raw_count in value.items():
        if isinstance(raw_count, TypedResource):
            items.append(raw_count)
            continue
        if isinstance(raw_count, Mapping):
            item = dict(raw_count)
            item.setdefault("resource_type", resource_type)
            items.append(TypedResource.from_mapping(item))
            continue
        items.append(TypedResource(resource_type, raw_count))
    return items


def _resource_item(value: Any) -> TypedResource:
    if isinstance(value, TypedResource):
        return value
    if isinstance(value, Mapping):
        return TypedResource.from_mapping(value)
    if isinstance(value, str):
        return TypedResource(value)
    resource_type = getattr(value, "resource_type", None)
    if resource_type is None:
        resource_type = getattr(value, "canonical_label", None)
    if resource_type is None:
        raise TypeError("resource items must expose a safe resource type")
    return TypedResource(resource_type, getattr(value, "count", 1))


def _mapping_resource_type(value: Mapping[str, Any]) -> Any:
    for key in _RESOURCE_TYPE_KEYS:
        if key in value and value[key] is not None:
            return value[key]
    raise ValueError("resource mapping must include a resource type")


def _has_resource_type_key(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in _RESOURCE_TYPE_KEYS)


def _resource_policy_fields(
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if value is None:
        return {}, {}, {}
    if not isinstance(value, Mapping):
        raise ValueError("policy resources must be a mapping")

    actions: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    waivers: dict[str, Any] = {}
    for resource_type, settings in value.items():
        if not isinstance(settings, Mapping):
            continue
        if "action" in settings:
            actions[resource_type] = settings["action"]
        if "gate" in settings:
            gates[resource_type] = settings["gate"]
        if "waiver" in settings:
            waivers[resource_type] = settings["waiver"]
    return actions, gates, waivers


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _normalize_value_map(
    value: Mapping[str, Any],
    normalizer: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return {
        _safe_token(key, field_name=f"{field_name} key"): normalizer(
            raw_value,
            field_name=f"{field_name} value",
        )
        for key, raw_value in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _normalize_waiver_map(value: Any) -> dict[str, bool]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            _safe_token(key, field_name="waivers key"): _normalize_waiver(
                raw_value,
                field_name="waivers value",
            )
            for key, raw_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, bytearray, Mapping),
    ):
        return {
            _safe_token(item, field_name="waiver resource type"): True for item in value
        }
    raise ValueError("waivers must be a mapping or sequence")


def _normalize_action(value: Any, *, field_name: str) -> str:
    return _safe_token(value, field_name=field_name)


def _normalize_gate(value: Any, *, field_name: str) -> GateValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _safe_token(value, field_name=field_name)
    if isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, bytearray, Mapping),
    ):
        values = tuple(
            sorted(
                {_safe_token(item, field_name=f"{field_name} item") for item in value}
            )
        )
        return values
    if value is None:
        return False
    raise ValueError(f"{field_name} must be a boolean, token, or token sequence")


def _normalize_waiver(value: Any, *, field_name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, bytearray, Mapping),
    ):
        return bool(value)
    raise ValueError(f"{field_name} must be a boolean or waiver declaration")


def _safe_token(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a safe non-empty token")
    token = value.strip()
    if not _SAFE_TOKEN_RE.fullmatch(token):
        raise ValueError(f"{field_name} must be a safe non-empty token")
    return token


def _policy_name(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    return _safe_token(value, field_name="name")


def _count(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _json_gate(value: GateValue) -> bool | str | list[str]:
    if isinstance(value, tuple):
        return list(value)
    return value


def _json_value(value: str | bool | tuple[str, ...]) -> str | bool | list[str]:
    return _json_gate(value)


__all__ = [
    "CURRENT_POLICY_IMPACT_SCHEMA_VERSION",
    "GateValue",
    "PolicyImpact",
    "PolicyImpactChange",
    "PolicyImpactDigest",
    "PolicyResource",
    "PolicyVersion",
    "TypedResource",
    "compare_policy_versions",
    "evaluate_policy_impact",
    "policy_impact_digest",
    "simulate_policy_impact",
]
