"""Deterministic, value-free selection of minimum-necessary fields.

Structured exports should be assembled from a declared purpose rather than
from the complete source record.  This module keeps the declaration separate
from record values: purpose mappings and policy profiles contain field names,
while :class:`FieldSelection` can project a record only after the selection has
been approved.

The selector is deliberately local and configuration-driven.  It does not
load policy data, inspect values, or make network calls at runtime.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, TypeAlias

MINIMUM_NECESSARY_SCHEMA_VERSION: Final[int] = 1

_REASON_SELECTED = "purpose_and_policy_allowlisted"
_REASON_UNKNOWN_PURPOSE = "unknown_purpose_mapping"
_REASON_UNKNOWN_POLICY = "unknown_policy_profile"
_REASON_REQUIRED_MISSING = "required_fields_unavailable"
_REASON_REQUIRED_BLOCKED = "required_fields_not_permitted"
_REASON_NO_FIELDS = "no_permitted_fields"

FieldInput: TypeAlias = Mapping[str, Any] | Iterable[str] | None
PurposeConfig: TypeAlias = "PurposeMapping | Mapping[str, Any] | Iterable[str]"
PolicyConfig: TypeAlias = "FieldPolicyProfile | Mapping[str, Any] | Iterable[str]"


def _normalize_identifier(value: str, kind: str) -> str:
    """Normalize a purpose or profile name without echoing caller input."""

    if not isinstance(value, str):
        raise TypeError(f"{kind} must be a string")
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError(f"{kind} must not be blank")
    return normalized


def _normalize_fields(value: Iterable[str] | None) -> tuple[str, ...]:
    """Return deterministic, validated field names without exposing values."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("field collections must contain field names")
    try:
        fields = tuple(value)
    except TypeError as exc:
        raise TypeError("field collections must contain field names") from exc

    normalized: set[str] = set()
    for field_name in fields:
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError("field names must be non-empty strings")
        normalized.add(field_name.strip())
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class PurposeMapping:
    """Fields that may be used for one declared purpose.

    ``fields`` are eligible for the purpose.  ``required_fields`` are the
    subset that must be present and permitted for a selection to be approved;
    this prevents a partial export from being mistaken for a complete one.
    All fields are treated as metadata and are never read from a record here.
    """

    fields: tuple[str, ...]
    required_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fields = _normalize_fields(self.fields)
        required_fields = _normalize_fields(self.required_fields)
        if not set(required_fields) <= set(fields):
            raise ValueError("required fields must be included in fields")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "required_fields", required_fields)

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free purpose declaration."""

        return {
            "fields": list(self.fields),
            "required_fields": list(self.required_fields),
        }


@dataclass(frozen=True)
class FieldPolicyProfile:
    """Allowlist and denylist for one structured-export policy profile.

    ``allowed_fields=None`` means that the profile places no additional
    allowlist restriction; the purpose mapping remains the minimum-necessary
    boundary.  An empty tuple is an explicit allowlist that permits no fields.
    A denylist always wins over an allowlist.
    """

    name: str
    allowed_fields: tuple[str, ...] | None = None
    denied_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _normalize_identifier(self.name, "policy profile")
        allowed_fields = (
            None
            if self.allowed_fields is None
            else _normalize_fields(self.allowed_fields)
        )
        denied_fields = _normalize_fields(self.denied_fields)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "allowed_fields", allowed_fields)
        object.__setattr__(self, "denied_fields", denied_fields)

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free policy declaration."""

        return {
            "name": self.name,
            "allowed_fields": (
                None if self.allowed_fields is None else list(self.allowed_fields)
            ),
            "denied_fields": list(self.denied_fields),
        }


@dataclass(frozen=True)
class SelectionExplanation:
    """Safe explanation metadata for one field-selection decision.

    The explanation contains field names, counts, and stable reason codes.  It
    intentionally has no record, cell, or source values.
    """

    allowed: bool
    reason: str
    purpose: str | None
    policy_profile: str | None
    selected_fields: tuple[str, ...]
    omitted_fields: tuple[str, ...]
    required_fields: tuple[str, ...]
    available_field_count: int
    schema_version: int = MINIMUM_NECESSARY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible, value-free explanation metadata."""

        return {
            "schema_version": self.schema_version,
            "allowed": self.allowed,
            "reason": self.reason,
            "purpose": self.purpose,
            "policy_profile": self.policy_profile,
            "selected_fields": list(self.selected_fields),
            "omitted_fields": list(self.omitted_fields),
            "required_fields": list(self.required_fields),
            "available_field_count": self.available_field_count,
        }

    def to_json(self) -> str:
        """Serialize the explanation deterministically without record data."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FieldSelection:
    """Result of a minimum-necessary field selection.

    ``selected_fields`` contains only field names.  Call :meth:`project` to
    apply an approved selection to a record; the projected values are not part
    of this result or its serialized explanation.
    """

    selected_fields: tuple[str, ...]
    explanation: SelectionExplanation

    @property
    def allowed(self) -> bool:
        """Return whether the selection may be applied."""

        return self.explanation.allowed

    @property
    def fields(self) -> tuple[str, ...]:
        """Return selected field names as a concise alias."""

        return self.selected_fields

    @property
    def reason(self) -> str:
        """Return the stable selection reason code."""

        return self.explanation.reason

    def project(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return only approved fields from ``record``.

        A denied selection always projects to an empty mapping.  The method
        never puts record values into the explanation or any exception.
        """

        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        if not self.allowed:
            return {}
        return {
            field_name: record[field_name]
            for field_name in self.selected_fields
            if field_name in record
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free selection explanation."""

        return self.explanation.to_dict()

    def to_json(self) -> str:
        """Serialize the value-free selection explanation deterministically."""

        return self.explanation.to_json()


SelectionResult: TypeAlias = FieldSelection


class MinimumNecessarySelector:
    """Select fields using caller-owned purpose and policy declarations.

    Args:
        purpose_mappings: Mapping from purpose names to either
            :class:`PurposeMapping`, a mapping with ``fields`` and optional
            ``required_fields``, or an iterable of field names.
        policy_profiles: Mapping from profile names to either
            :class:`FieldPolicyProfile`, a mapping with optional
            ``allowed_fields``/``denied_fields``, or an iterable interpreted as
            an allowlist.

    Unknown purposes and profiles are denied rather than guessed.  The input
    mappings are copied and normalized during construction, so later caller
    mutation cannot change a selection.
    """

    def __init__(
        self,
        purpose_mappings: Mapping[str, PurposeConfig],
        policy_profiles: Mapping[str, PolicyConfig],
    ) -> None:
        if not isinstance(purpose_mappings, Mapping):
            raise TypeError("purpose_mappings must be a mapping")
        if not isinstance(policy_profiles, Mapping):
            raise TypeError("policy_profiles must be a mapping")

        normalized_purposes: dict[str, PurposeMapping] = {}
        for raw_name, raw_config in purpose_mappings.items():
            name = _normalize_identifier(raw_name, "purpose")
            if name in normalized_purposes:
                raise ValueError("purpose mappings contain duplicate names")
            normalized_purposes[name] = _coerce_purpose_mapping(raw_config)

        normalized_profiles: dict[str, FieldPolicyProfile] = {}
        for raw_name, raw_config in policy_profiles.items():
            name = _normalize_identifier(raw_name, "policy profile")
            if name in normalized_profiles:
                raise ValueError("policy profiles contain duplicate names")
            normalized_profiles[name] = _coerce_policy_profile(name, raw_config)

        self._purpose_mappings = MappingProxyType(normalized_purposes)
        self._policy_profiles = MappingProxyType(normalized_profiles)

    @property
    def purpose_mappings(self) -> Mapping[str, PurposeMapping]:
        """Return the immutable normalized purpose registry."""

        return self._purpose_mappings

    @property
    def policy_profiles(self) -> Mapping[str, FieldPolicyProfile]:
        """Return the immutable normalized policy registry."""

        return self._policy_profiles

    def select(
        self,
        available_fields: FieldInput = None,
        *,
        purpose: str,
        policy_profile: str | FieldPolicyProfile,
    ) -> FieldSelection:
        """Select the minimum permitted field set for a declared request.

        ``available_fields`` may be a record mapping, an iterable of field
        names, or ``None``.  When it is ``None``, all fields in the purpose
        mapping are considered available.  Record values are never inspected.
        """

        purpose_name = _normalize_identifier(purpose, "purpose")
        purpose_mapping = self._purpose_mappings.get(purpose_name)
        if purpose_mapping is None:
            available = _available_fields(available_fields)
            return _denied_selection(
                reason=_REASON_UNKNOWN_PURPOSE,
                purpose=None,
                policy_profile=None,
                available_field_count=len(available),
            )

        available = (
            purpose_mapping.fields
            if available_fields is None
            else _available_fields(available_fields)
        )
        profile_name, profile = self._resolve_profile(policy_profile)
        if profile is None:
            return _denied_selection(
                reason=_REASON_UNKNOWN_POLICY,
                purpose=purpose_name,
                policy_profile=None,
                available_field_count=len(available),
            )

        purpose_fields = set(purpose_mapping.fields)
        available_set = set(available)
        eligible = purpose_fields & available_set
        if profile.allowed_fields is not None:
            eligible &= set(profile.allowed_fields)
        eligible -= set(profile.denied_fields)
        selected = tuple(sorted(eligible))
        omitted = tuple(sorted(available_set - set(selected)))
        required = set(purpose_mapping.required_fields)

        unavailable_required = required - available_set
        if unavailable_required:
            return _denied_selection(
                reason=_REASON_REQUIRED_MISSING,
                purpose=purpose_name,
                policy_profile=profile_name,
                selected_fields=(),
                omitted_fields=omitted,
                required_fields=purpose_mapping.required_fields,
                available_field_count=len(available),
            )

        blocked_required = required - eligible
        if blocked_required:
            return _denied_selection(
                reason=_REASON_REQUIRED_BLOCKED,
                purpose=purpose_name,
                policy_profile=profile_name,
                selected_fields=(),
                omitted_fields=omitted,
                required_fields=purpose_mapping.required_fields,
                available_field_count=len(available),
            )

        if not selected:
            return _denied_selection(
                reason=_REASON_NO_FIELDS,
                purpose=purpose_name,
                policy_profile=profile_name,
                omitted_fields=omitted,
                required_fields=purpose_mapping.required_fields,
                available_field_count=len(available),
            )

        explanation = SelectionExplanation(
            allowed=True,
            reason=_REASON_SELECTED,
            purpose=purpose_name,
            policy_profile=profile_name,
            selected_fields=selected,
            omitted_fields=omitted,
            required_fields=purpose_mapping.required_fields,
            available_field_count=len(available),
        )
        return FieldSelection(selected_fields=selected, explanation=explanation)

    def _resolve_profile(
        self,
        policy_profile: str | FieldPolicyProfile,
    ) -> tuple[str | None, FieldPolicyProfile | None]:
        if isinstance(policy_profile, FieldPolicyProfile):
            return policy_profile.name, policy_profile
        profile_name = _normalize_identifier(policy_profile, "policy profile")
        return profile_name, self._policy_profiles.get(profile_name)


def _coerce_purpose_mapping(config: PurposeConfig) -> PurposeMapping:
    if isinstance(config, PurposeMapping):
        return config
    if isinstance(config, Mapping):
        if "fields" not in config:
            raise ValueError("purpose mappings must declare fields")
        return PurposeMapping(
            fields=config["fields"],
            required_fields=config.get("required_fields") or (),
        )
    return PurposeMapping(fields=config)


def _coerce_policy_profile(
    name: str,
    config: PolicyConfig,
) -> FieldPolicyProfile:
    if isinstance(config, FieldPolicyProfile):
        return FieldPolicyProfile(
            name=name,
            allowed_fields=config.allowed_fields,
            denied_fields=config.denied_fields,
        )
    if isinstance(config, Mapping):
        allowed_fields = config.get("allowed_fields", config.get("fields"))
        return FieldPolicyProfile(
            name=name,
            allowed_fields=allowed_fields,
            denied_fields=config.get("denied_fields") or (),
        )
    return FieldPolicyProfile(name=name, allowed_fields=config)


def _available_fields(available_fields: FieldInput) -> tuple[str, ...]:
    if isinstance(available_fields, Mapping):
        return _normalize_fields(available_fields.keys())
    return _normalize_fields(available_fields)


def _denied_selection(
    *,
    reason: str,
    purpose: str | None,
    policy_profile: str | None,
    selected_fields: tuple[str, ...] = (),
    omitted_fields: tuple[str, ...] = (),
    required_fields: tuple[str, ...] = (),
    available_field_count: int = 0,
) -> FieldSelection:
    explanation = SelectionExplanation(
        allowed=False,
        reason=reason,
        purpose=purpose,
        policy_profile=policy_profile,
        selected_fields=selected_fields,
        omitted_fields=omitted_fields,
        required_fields=required_fields,
        available_field_count=available_field_count,
    )
    return FieldSelection(selected_fields=selected_fields, explanation=explanation)


def select_fields(
    available_fields: FieldInput,
    purpose: str,
    policy_profile: str | FieldPolicyProfile,
    *,
    purpose_mappings: Mapping[str, PurposeConfig],
    policy_profiles: Mapping[str, PolicyConfig],
) -> FieldSelection:
    """Select fields from caller-declared purpose and policy registries.

    This convenience wrapper constructs a selector from immutable copies of
    the supplied registries and returns a value-free :class:`FieldSelection`.
    """

    selector = MinimumNecessarySelector(purpose_mappings, policy_profiles)
    return selector.select(
        available_fields,
        purpose=purpose,
        policy_profile=policy_profile,
    )


def select_minimum_necessary_fields(
    available_fields: FieldInput,
    purpose: str,
    policy_profile: str | FieldPolicyProfile,
    *,
    purpose_mappings: Mapping[str, PurposeConfig],
    policy_profiles: Mapping[str, PolicyConfig],
) -> FieldSelection:
    """Alias with an explicit minimum-necessary name for the public API."""

    return select_fields(
        available_fields,
        purpose,
        policy_profile,
        purpose_mappings=purpose_mappings,
        policy_profiles=policy_profiles,
    )


__all__ = [
    "MINIMUM_NECESSARY_SCHEMA_VERSION",
    "FieldPolicyProfile",
    "FieldSelection",
    "MinimumNecessarySelector",
    "PurposeMapping",
    "SelectionExplanation",
    "SelectionResult",
    "select_fields",
    "select_minimum_necessary_fields",
]
