"""Minimum-necessary redaction-strength selection."""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from .labels import (
    CANONICAL_LABELS,
    CLINICAL_CONCEPT,
    DIRECT_IDENTIFIER,
    POLICY_LABELS,
    QUASI_IDENTIFIER,
    RISK_HIGH,
    RISK_LEVELS,
    RISK_LOW,
    RISK_MEDIUM,
    normalize_label,
    policy_label_for,
    risk_level_for,
)
from .policy import PolicyName, PolicyProfile, list_policies, load_policy
from .schemas.span import ACTION_VALUES

ACTION_STRENGTH_ORDER: tuple[str, ...] = (
    "keep",
    "replace",
    "hash",
    "mask",
    "redact",
)
ACTION_STRENGTH: Mapping[str, int] = {
    action: index for index, action in enumerate(ACTION_STRENGTH_ORDER)
}

_RISK_POLICY_LABELS: Mapping[str, str] = {
    RISK_LOW: CLINICAL_CONCEPT,
    RISK_MEDIUM: QUASI_IDENTIFIER,
    RISK_HIGH: DIRECT_IDENTIFIER,
}


class ActionMapComparison(str, Enum):
    """Ordering relationship between two redaction action maps."""

    EQUAL = "equal"
    WEAKER = "weaker"
    STRONGER = "stronger"
    INCOMPARABLE = "incomparable"


def action_strength(action: str) -> int:
    """Return the ordinal strength for a redaction action."""

    normalized = str(action)
    try:
        return ACTION_STRENGTH[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown redaction action {action!r}; expected one of "
            f"{ACTION_STRENGTH_ORDER!r}"
        ) from exc


def action_meets_floor(action: str, floor: str) -> bool:
    """Return whether ``action`` is at least as strong as ``floor``."""

    return action_strength(action) >= action_strength(floor)


def compare_action_maps(
    left: Mapping[str, str],
    right: Mapping[str, str],
    *,
    labels: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
) -> ActionMapComparison:
    """Compare two action maps by per-label redaction strength.

    ``WEAKER`` means every compared action in ``left`` is less than or equal to
    the corresponding action in ``right``, with at least one strictly weaker
    action. ``STRONGER`` is the inverse. Mixed directions are ``INCOMPARABLE``.
    """

    label_set = set(labels) if labels is not None else set(left) | set(right)
    if not label_set:
        return ActionMapComparison.EQUAL
    if not label_set <= set(left) or not label_set <= set(right):
        return ActionMapComparison.INCOMPARABLE

    saw_weaker = False
    saw_stronger = False
    for label in sorted(label_set):
        left_strength = action_strength(left[label])
        right_strength = action_strength(right[label])
        saw_weaker = saw_weaker or left_strength < right_strength
        saw_stronger = saw_stronger or left_strength > right_strength
        if saw_weaker and saw_stronger:
            return ActionMapComparison.INCOMPARABLE

    if saw_weaker:
        return ActionMapComparison.WEAKER
    if saw_stronger:
        return ActionMapComparison.STRONGER
    return ActionMapComparison.EQUAL


def select_minimum_necessary(
    profile: str | PolicyName | PolicyProfile,
    *,
    target_posture: str | PolicyName | PolicyProfile,
    risk_level_by_label: Mapping[str, str] | None = None,
    name: str | None = None,
) -> PolicyProfile:
    """Derive a profile with the weakest actions meeting target posture floors.

    The target posture contributes policy-label floors such as
    ``DIRECT_IDENTIFIER -> mask`` and ``QUASI_IDENTIFIER -> keep``. Each
    canonical label receives the weakest runtime action that satisfies both its
    policy-label floor and, when supplied, its risk-level floor. The input
    profile is copied, not mutated, so runtime settings such as arbitration and
    safety-sweep behavior stay stable.
    """

    base_profile = load_policy(profile)
    posture_profile = _load_target_posture(target_posture)
    normalized_risk_levels = _normalize_risk_levels(risk_level_by_label)
    policy_floors = _policy_label_floors(posture_profile)
    actions = {
        label: _minimum_action_for_label(
            label,
            policy_floors=policy_floors,
            risk_level_by_label=normalized_risk_levels,
        )
        for label in sorted(CANONICAL_LABELS)
    }

    derived_name = (
        name or f"{base_profile.name}_minimum_necessary_{posture_profile.name}"
    )
    return base_profile.derive(
        name=derived_name,
        posture=f"minimum_necessary:{posture_profile.posture}",
        default_action=actions[normalize_label("OTHER")],
        default_action_bias="minimum_necessary",
        actions=actions,
        policy_label_actions=policy_floors,
        metadata={
            "minimum_necessary": {
                "base_profile": base_profile.name,
                "target_posture": posture_profile.name,
                "target_posture_value": posture_profile.posture,
                "comparison_to_base": compare_action_maps(
                    actions,
                    base_profile.actions,
                    labels=CANONICAL_LABELS,
                ).value,
            },
        },
    )


def _load_target_posture(
    target_posture: str | PolicyName | PolicyProfile,
) -> PolicyProfile:
    if isinstance(target_posture, PolicyProfile):
        return target_posture
    try:
        return load_policy(target_posture)
    except ValueError:
        pass

    normalized = str(target_posture or "").strip().lower().replace("-", "_")
    for policy_name in list_policies():
        profile = load_policy(policy_name)
        if profile.posture.lower().replace("-", "_") == normalized:
            return profile
    raise ValueError(
        f"unknown target_posture {target_posture!r}; pass a policy name, "
        "policy posture, or PolicyProfile"
    )


def _policy_label_floors(profile: PolicyProfile) -> dict[str, str]:
    floors: dict[str, str] = {}
    for policy_label in sorted(POLICY_LABELS):
        action = profile.policy_label_actions.get(
            policy_label,
            profile.default_action,
        )
        _validate_action(action)
        floors[policy_label] = action
    return floors


def _normalize_risk_levels(
    risk_level_by_label: Mapping[str, str] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for label, risk_level in (risk_level_by_label or {}).items():
        canonical = normalize_label(str(label))
        normalized[canonical] = _validate_risk_level(risk_level)
    return normalized


def _minimum_action_for_label(
    label: str,
    *,
    policy_floors: Mapping[str, str],
    risk_level_by_label: Mapping[str, str],
) -> str:
    policy_label = policy_label_for(label)
    risk_level = risk_level_by_label.get(label, risk_level_for(label))
    floor_policy_labels = {
        policy_label,
        _RISK_POLICY_LABELS[_validate_risk_level(risk_level)],
    }
    floor = max(
        (policy_floors[policy_label] for policy_label in floor_policy_labels),
        key=action_strength,
    )
    return _weakest_action_meeting(floor)


def _weakest_action_meeting(floor: str) -> str:
    _validate_action(floor)
    for action in ACTION_STRENGTH_ORDER:
        if action_meets_floor(action, floor):
            return action
    raise ValueError(f"no action can satisfy floor {floor!r}")


def _validate_action(action: str) -> None:
    if action not in ACTION_VALUES:
        raise ValueError(
            f"action {action!r} is not supported by the span schema; expected "
            f"one of {ACTION_VALUES!r}"
        )
    action_strength(action)


def _validate_risk_level(risk_level: str) -> str:
    normalized = str(risk_level).strip().lower()
    if normalized not in RISK_LEVELS:
        raise ValueError(f"risk_level must be one of {RISK_LEVELS!r}")
    return normalized


__all__ = [
    "ACTION_STRENGTH",
    "ACTION_STRENGTH_ORDER",
    "ActionMapComparison",
    "action_meets_floor",
    "action_strength",
    "compare_action_maps",
    "select_minimum_necessary",
]
