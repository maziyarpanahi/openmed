"""Versioned de-identification policy profile loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping
import json

from .arbitration import MODE_BALANCED, MODE_HIGH_RECALL_UNION
from .labels import CANONICAL_LABELS, POLICY_LABELS, normalize_label, policy_label_for
from .schemas.span import ACTION_VALUES


CURRENT_POLICY_SCHEMA_VERSION = 1


class PolicyName(str, Enum):
    """Canonical policy profile names accepted by the de-identification runtime."""

    HIPAA_SAFE_HARBOR = "hipaa_safe_harbor"
    HIPAA_EXPERT_REVIEW_ASSIST = "hipaa_expert_review_assist"
    GDPR_PSEUDONYMIZATION = "gdpr_pseudonymization"
    RESEARCH_LIMITED_DATASET = "research_limited_dataset"
    STRICT_NO_LEAK = "strict_no_leak"
    CLINICAL_MINIMAL_REDACTION = "clinical_minimal_redaction"


CANONICAL_POLICY_NAMES = tuple(policy.value for policy in PolicyName)
POLICY_ALIASES: Mapping[str, str] = {
    "gdpr": PolicyName.GDPR_PSEUDONYMIZATION.value,
}

_ARBITRATION_MODES = frozenset({MODE_BALANCED, MODE_HIGH_RECALL_UNION})


@dataclass(frozen=True)
class PolicyProfile:
    """Loaded policy profile with validated action and runtime settings."""

    name: str
    schema_version: int
    posture: str
    threshold_profile: str
    default_action: str
    default_action_bias: str
    arbitration_mode: str
    safety_sweep_mandatory: bool
    keep_mapping: bool
    reversible_id: bool
    forced_cascade_tiers: tuple[str, ...]
    actions: Mapping[str, str]
    policy_label_actions: Mapping[str, str] = field(default_factory=dict)
    strict_no_leak: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def action_for(self, label: str, *, lang: str = "en") -> str:
        """Return the action configured for a canonical or source label."""

        canonical_label = normalize_label(label, lang=lang)
        action = self.actions.get(canonical_label)
        if action is not None:
            return action

        policy_label = policy_label_for(canonical_label, lang=lang)
        action = self.policy_label_actions.get(policy_label)
        if action is not None:
            return action
        return self.default_action

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the loaded profile."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "posture": self.posture,
            "threshold_profile": self.threshold_profile,
            "default_action": self.default_action,
            "default_action_bias": self.default_action_bias,
            "arbitration_mode": self.arbitration_mode,
            "strict_no_leak": self.strict_no_leak,
            "safety_sweep_mandatory": self.safety_sweep_mandatory,
            "keep_mapping": self.keep_mapping,
            "reversible_id": self.reversible_id,
            "forced_cascade_tiers": list(self.forced_cascade_tiers),
            "actions": dict(self.actions),
            "policy_label_actions": dict(self.policy_label_actions),
            "metadata": dict(self.metadata),
        }


def canonical_policy_name(name: str | PolicyName) -> str:
    """Resolve aliases and validate a policy profile name."""

    if isinstance(name, PolicyName):
        return name.value
    normalized = str(name or "").strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("policy must not be blank")
    normalized = POLICY_ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_POLICY_NAMES:
        allowed = ", ".join((*CANONICAL_POLICY_NAMES, *sorted(POLICY_ALIASES)))
        raise ValueError(f"unknown policy {name!r}; expected one of: {allowed}")
    return normalized


def load_policy(name: str | PolicyName | PolicyProfile) -> PolicyProfile:
    """Load and validate a bundled policy profile by name."""

    if isinstance(name, PolicyProfile):
        return name
    return _load_policy(canonical_policy_name(name))


def list_policies() -> tuple[str, ...]:
    """Return the canonical policy names."""

    return CANONICAL_POLICY_NAMES


@lru_cache(maxsize=len(CANONICAL_POLICY_NAMES))
def _load_policy(canonical_name: str) -> PolicyProfile:
    resource = resources.files("openmed.core").joinpath(
        "policies",
        f"{canonical_name}.json",
    )
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _profile_from_mapping(payload, source=str(resource))


def _profile_from_mapping(payload: Mapping[str, Any], *, source: str) -> PolicyProfile:
    schema_version = _positive_int(payload.get("schema_version"), "schema_version")
    if schema_version != CURRENT_POLICY_SCHEMA_VERSION:
        raise ValueError(
            "policy profile schema_version "
            f"{schema_version} is not supported; expected {CURRENT_POLICY_SCHEMA_VERSION}"
        )

    name = canonical_policy_name(str(payload.get("name") or ""))
    posture = _non_empty_str(payload.get("posture"), "posture")
    threshold_profile = _non_empty_str(
        payload.get("threshold_profile"),
        "threshold_profile",
    )
    default_action = _action(payload.get("default_action"), "default_action")
    default_action_bias = _non_empty_str(
        payload.get("default_action_bias", default_action),
        "default_action_bias",
    )
    arbitration_mode = _non_empty_str(
        payload.get("arbitration_mode"),
        "arbitration_mode",
    )
    if arbitration_mode not in _ARBITRATION_MODES:
        raise ValueError(
            f"arbitration_mode must be one of {sorted(_ARBITRATION_MODES)!r}"
        )

    policy_label_actions = _policy_label_actions(
        payload.get("policy_label_actions") or {},
    )
    actions = _canonical_actions(payload.get("actions") or {})
    forced_cascade_tiers = _string_tuple(
        payload.get("forced_cascade_tiers") or (),
        "forced_cascade_tiers",
    )

    return PolicyProfile(
        name=name,
        schema_version=schema_version,
        posture=posture,
        threshold_profile=threshold_profile,
        default_action=default_action,
        default_action_bias=default_action_bias,
        arbitration_mode=arbitration_mode,
        strict_no_leak=bool(payload.get("strict_no_leak", name == "strict_no_leak")),
        safety_sweep_mandatory=bool(payload.get("safety_sweep_mandatory", False)),
        keep_mapping=bool(payload.get("keep_mapping", False)),
        reversible_id=bool(payload.get("reversible_id", False)),
        forced_cascade_tiers=forced_cascade_tiers,
        actions=actions,
        policy_label_actions=policy_label_actions,
        metadata={
            "source": Path(source).name,
            **dict(payload.get("metadata") or {}),
        },
    )


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _action(value: Any, field_name: str) -> str:
    action = _non_empty_str(value, field_name)
    if action not in ACTION_VALUES:
        raise ValueError(f"{field_name} must be one of {ACTION_VALUES!r}")
    return action


def _policy_label_actions(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("policy_label_actions must be an object")

    actions: dict[str, str] = {}
    for label, action in value.items():
        policy_label = str(label)
        if policy_label not in POLICY_LABELS:
            raise ValueError(f"unknown policy label {policy_label!r}")
        actions[policy_label] = _action(action, f"policy_label_actions.{policy_label}")
    return actions


def _canonical_actions(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("actions must be an object")

    actions: dict[str, str] = {}
    for label, action in value.items():
        canonical = normalize_label(str(label))
        if canonical != str(label):
            raise ValueError(f"actions must use canonical labels, got {label!r}")
        actions[canonical] = _action(action, f"actions.{canonical}")

    missing = sorted(CANONICAL_LABELS - set(actions))
    extra = sorted(set(actions) - CANONICAL_LABELS)
    if missing or extra:
        raise ValueError(
            "actions must cover CANONICAL_LABELS exactly; "
            f"missing={missing}, extra={extra}"
        )
    return actions


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    items = tuple(str(item).strip() for item in value)
    if any(not item for item in items):
        raise ValueError(f"{field_name} must not contain blank values")
    return items


__all__ = [
    "CANONICAL_POLICY_NAMES",
    "CURRENT_POLICY_SCHEMA_VERSION",
    "POLICY_ALIASES",
    "PolicyName",
    "PolicyProfile",
    "canonical_policy_name",
    "list_policies",
    "load_policy",
]
