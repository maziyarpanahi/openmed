"""Policy profile diffing for privacy review workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .arbitration import MODE_BALANCED, MODE_HIGH_RECALL_UNION
from .labels import POLICY_LABELS
from .policy import (
    CURRENT_POLICY_SCHEMA_VERSION,
    PolicyName,
    PolicyProfile,
    load_policy,
)
from .schemas.span import ACTION_VALUES

PolicyInput = str | Path | PolicyName | PolicyProfile
DiffDirection = Literal["weaker", "stronger", "equivalent"]

_ACTION_STRENGTH = {action: index for index, action in enumerate(ACTION_VALUES)}
_ARBITRATION_MODES = frozenset({MODE_BALANCED, MODE_HIGH_RECALL_UNION})
_RUNTIME_FLAG_FIELDS = (
    "strict_no_leak",
    "safety_sweep_mandatory",
    "reversible_id",
)
_SETTING_FIELDS = ("threshold_profile", "forced_cascade_tiers")


@dataclass(frozen=True)
class ActionChange:
    """Changed action for a canonical or policy-level label."""

    label: str
    from_action: str
    to_action: str
    direction: DiffDirection

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return {
            "label": self.label,
            "from": self.from_action,
            "to": self.to_action,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class LabelAction:
    """Added or removed label action."""

    label: str
    action: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return {"label": self.label, "action": self.action}


@dataclass(frozen=True)
class FieldChange:
    """Changed non-action policy setting."""

    field: str
    from_value: Any
    to_value: Any

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "field": self.field,
            "from": _json_value(self.from_value),
            "to": _json_value(self.to_value),
        }


@dataclass(frozen=True)
class PolicyDiff:
    """Structured diff between two policy profiles."""

    base: str
    candidate: str
    changed_label_actions: tuple[ActionChange, ...]
    added_labels: tuple[LabelAction, ...]
    removed_labels: tuple[LabelAction, ...]
    changed_policy_label_actions: tuple[ActionChange, ...]
    added_policy_labels: tuple[LabelAction, ...]
    removed_policy_labels: tuple[LabelAction, ...]
    runtime_flag_changes: tuple[FieldChange, ...]
    setting_changes: tuple[FieldChange, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether the compared profiles are equivalent for diffed fields."""

        return not any(
            (
                self.changed_label_actions,
                self.added_labels,
                self.removed_labels,
                self.changed_policy_label_actions,
                self.added_policy_labels,
                self.removed_policy_labels,
                self.runtime_flag_changes,
                self.setting_changes,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the diff."""

        return {
            "base": self.base,
            "candidate": self.candidate,
            "changed_label_actions": [
                change.to_dict() for change in self.changed_label_actions
            ],
            "added_labels": [label.to_dict() for label in self.added_labels],
            "removed_labels": [label.to_dict() for label in self.removed_labels],
            "changed_policy_label_actions": [
                change.to_dict() for change in self.changed_policy_label_actions
            ],
            "added_policy_labels": [
                label.to_dict() for label in self.added_policy_labels
            ],
            "removed_policy_labels": [
                label.to_dict() for label in self.removed_policy_labels
            ],
            "runtime_flag_changes": [
                change.to_dict() for change in self.runtime_flag_changes
            ],
            "setting_changes": [change.to_dict() for change in self.setting_changes],
        }


def diff_policies(base: PolicyInput, candidate: PolicyInput) -> PolicyDiff:
    """Diff two policy profiles.

    Args:
        base: Baseline profile name, JSON path, or loaded ``PolicyProfile``.
        candidate: Candidate profile name, JSON path, or loaded ``PolicyProfile``.

    Returns:
        Structured, JSON-serializable policy diff.
    """

    base_profile = _resolve_policy(base)
    candidate_profile = _resolve_policy(candidate)

    return PolicyDiff(
        base=base_profile.name,
        candidate=candidate_profile.name,
        changed_label_actions=_changed_actions(
            base_profile.actions,
            candidate_profile.actions,
        ),
        added_labels=_added_actions(base_profile.actions, candidate_profile.actions),
        removed_labels=_removed_actions(
            base_profile.actions, candidate_profile.actions
        ),
        changed_policy_label_actions=_changed_actions(
            base_profile.policy_label_actions,
            candidate_profile.policy_label_actions,
        ),
        added_policy_labels=_added_actions(
            base_profile.policy_label_actions,
            candidate_profile.policy_label_actions,
        ),
        removed_policy_labels=_removed_actions(
            base_profile.policy_label_actions,
            candidate_profile.policy_label_actions,
        ),
        runtime_flag_changes=_field_changes(
            base_profile,
            candidate_profile,
            _RUNTIME_FLAG_FIELDS,
        ),
        setting_changes=_field_changes(
            base_profile, candidate_profile, _SETTING_FIELDS
        ),
    )


def render(
    diff: PolicyDiff, fmt: Literal["text", "dict"] = "text"
) -> str | dict[str, Any]:
    """Render a policy diff as compact text or a JSON-serializable dictionary."""

    if fmt == "dict":
        return diff.to_dict()
    if fmt != "text":
        raise ValueError("fmt must be 'text' or 'dict'")

    if diff.is_empty:
        return f"Policy diff: {diff.base} -> {diff.candidate}\nNo policy changes."

    lines = [f"Policy diff: {diff.base} -> {diff.candidate}"]
    _append_action_section(lines, "Label action changes", diff.changed_label_actions)
    _append_label_section(lines, "Added labels", diff.added_labels)
    _append_label_section(lines, "Removed labels", diff.removed_labels)
    _append_action_section(
        lines,
        "Policy-label action changes",
        diff.changed_policy_label_actions,
    )
    _append_label_section(lines, "Added policy labels", diff.added_policy_labels)
    _append_label_section(lines, "Removed policy labels", diff.removed_policy_labels)
    _append_field_section(lines, "Runtime flag changes", diff.runtime_flag_changes)
    _append_field_section(lines, "Setting changes", diff.setting_changes)
    return "\n".join(lines)


def _resolve_policy(value: PolicyInput) -> PolicyProfile:
    if isinstance(value, PolicyProfile):
        return value
    if isinstance(value, Path):
        return _load_policy_path(value)
    if isinstance(value, PolicyName):
        return load_policy(value)

    text = str(value)
    path = Path(text)
    if path.exists():
        return _load_policy_path(path)
    return load_policy(text)


def _load_policy_path(path: Path) -> PolicyProfile:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"policy profile {path} must contain a JSON object")
    return _profile_from_mapping(payload, source=str(path))


def _profile_from_mapping(payload: Mapping[str, Any], *, source: str) -> PolicyProfile:
    schema_version = _positive_int(payload.get("schema_version"), "schema_version")
    if schema_version != CURRENT_POLICY_SCHEMA_VERSION:
        raise ValueError(
            "policy profile schema_version "
            f"{schema_version} is not supported; expected "
            f"{CURRENT_POLICY_SCHEMA_VERSION}"
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

    return PolicyProfile(
        name=_non_empty_str(payload.get("name"), "name"),
        schema_version=schema_version,
        posture=_non_empty_str(payload.get("posture"), "posture"),
        threshold_profile=_non_empty_str(
            payload.get("threshold_profile"),
            "threshold_profile",
        ),
        default_action=default_action,
        default_action_bias=default_action_bias,
        arbitration_mode=arbitration_mode,
        strict_no_leak=bool(payload.get("strict_no_leak", False)),
        safety_sweep_mandatory=bool(payload.get("safety_sweep_mandatory", False)),
        keep_mapping=bool(payload.get("keep_mapping", False)),
        reversible_id=bool(payload.get("reversible_id", False)),
        forced_cascade_tiers=_string_tuple(
            payload.get("forced_cascade_tiers") or (),
            "forced_cascade_tiers",
        ),
        actions=_actions(payload.get("actions") or {}, "actions"),
        policy_label_actions=_actions(
            payload.get("policy_label_actions") or {},
            "policy_label_actions",
            allowed_labels=POLICY_LABELS,
        ),
        metadata={
            "source": Path(source).name,
            **dict(payload.get("metadata") or {}),
        },
    )


def _changed_actions(
    base_actions: Mapping[str, str],
    candidate_actions: Mapping[str, str],
) -> tuple[ActionChange, ...]:
    changes = []
    for label in sorted(set(base_actions) & set(candidate_actions)):
        from_action = base_actions[label]
        to_action = candidate_actions[label]
        if from_action != to_action:
            changes.append(
                ActionChange(
                    label=label,
                    from_action=from_action,
                    to_action=to_action,
                    direction=_direction(from_action, to_action),
                )
            )
    return tuple(changes)


def _added_actions(
    base_actions: Mapping[str, str],
    candidate_actions: Mapping[str, str],
) -> tuple[LabelAction, ...]:
    return tuple(
        LabelAction(label=label, action=candidate_actions[label])
        for label in sorted(set(candidate_actions) - set(base_actions))
    )


def _removed_actions(
    base_actions: Mapping[str, str],
    candidate_actions: Mapping[str, str],
) -> tuple[LabelAction, ...]:
    return tuple(
        LabelAction(label=label, action=base_actions[label])
        for label in sorted(set(base_actions) - set(candidate_actions))
    )


def _field_changes(
    base_profile: PolicyProfile,
    candidate_profile: PolicyProfile,
    fields: tuple[str, ...],
) -> tuple[FieldChange, ...]:
    changes = []
    for field in fields:
        from_value = getattr(base_profile, field)
        to_value = getattr(candidate_profile, field)
        if from_value != to_value:
            changes.append(FieldChange(field, from_value, to_value))
    return tuple(changes)


def _direction(from_action: str, to_action: str) -> DiffDirection:
    from_strength = _ACTION_STRENGTH[from_action]
    to_strength = _ACTION_STRENGTH[to_action]
    if to_strength > from_strength:
        return "stronger"
    if to_strength < from_strength:
        return "weaker"
    return "equivalent"


def _append_action_section(
    lines: list[str],
    title: str,
    changes: tuple[ActionChange, ...],
) -> None:
    if not changes:
        return
    lines.append(f"{title}:")
    for change in changes:
        lines.append(
            f"  - {change.label}: {change.from_action} -> {change.to_action} "
            f"({change.direction})"
        )


def _append_label_section(
    lines: list[str],
    title: str,
    labels: tuple[LabelAction, ...],
) -> None:
    if not labels:
        return
    lines.append(f"{title}:")
    for label in labels:
        lines.append(f"  - {label.label}: {label.action}")


def _append_field_section(
    lines: list[str],
    title: str,
    changes: tuple[FieldChange, ...],
) -> None:
    if not changes:
        return
    lines.append(f"{title}:")
    for change in changes:
        lines.append(
            f"  - {change.field}: "
            f"{_format_value(change.from_value)} -> {_format_value(change.to_value)}"
        )


def _actions(
    value: Mapping[str, Any],
    field_name: str,
    *,
    allowed_labels: frozenset[str] | None = None,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")

    actions: dict[str, str] = {}
    for label, action in value.items():
        label_name = _non_empty_str(label, f"{field_name} label")
        if allowed_labels is not None and label_name not in allowed_labels:
            raise ValueError(f"unknown policy label {label_name!r}")
        actions[label_name] = _action(action, f"{field_name}.{label_name}")
    return actions


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


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    items = tuple(str(item).strip() for item in value)
    if any(not item for item in items):
        raise ValueError(f"{field_name} must not contain blank values")
    return items


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _format_value(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True)


__all__ = [
    "ActionChange",
    "FieldChange",
    "LabelAction",
    "PolicyDiff",
    "diff_policies",
    "render",
]
