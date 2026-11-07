"""Deterministic, value-free selection of minimum-necessary fields.

Structured exports should be assembled from a declared purpose rather than
from the complete source record. This module keeps policy declarations
separate from record values and bounds all caller-controlled metadata.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, TypeAlias, cast

MINIMUM_NECESSARY_SCHEMA_VERSION: Final = 1
MAX_PURPOSE_MAPPINGS: Final = 256
MAX_POLICY_PROFILES: Final = 256
MAX_FIELDS_PER_DECLARATION: Final = 512
MAX_AVAILABLE_FIELDS: Final = 4_096

_MAX_FIELD_NAME_LENGTH: Final = 128
_SAFE_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")
_SAFE_FIELD_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
_PURPOSE_CONFIG_FIELDS: Final = frozenset({"fields", "required_fields"})
_POLICY_CONFIG_FIELDS: Final = frozenset({"allowed_fields", "denied_fields", "fields"})

_REASON_SELECTED = "purpose_and_policy_allowlisted"
_REASON_UNKNOWN_PURPOSE = "unknown_purpose_mapping"
_REASON_UNKNOWN_POLICY = "unknown_policy_profile"
_REASON_REQUIRED_MISSING = "required_fields_unavailable"
_REASON_REQUIRED_BLOCKED = "required_fields_not_permitted"
_REASON_NO_FIELDS = "no_permitted_fields"
_REASONS: Final = frozenset(
    {
        _REASON_SELECTED,
        _REASON_UNKNOWN_PURPOSE,
        _REASON_UNKNOWN_POLICY,
        _REASON_REQUIRED_MISSING,
        _REASON_REQUIRED_BLOCKED,
        _REASON_NO_FIELDS,
    }
)
_SELECTION_TOKEN: Final = object()

FieldInput: TypeAlias = Mapping[str, Any] | Iterable[str] | None
PurposeConfig: TypeAlias = "PurposeMapping | Mapping[str, Any] | Iterable[str]"
PolicyConfig: TypeAlias = "FieldPolicyProfile | Mapping[str, Any] | Iterable[str]"


def _snapshot_mapping(
    value: Mapping[Any, Any], *, field_name: str, max_items: int
) -> dict[str, Any]:
    """Copy a bounded mapping without exposing caller-controlled failures."""

    try:
        items = list(itertools.islice(value.items(), max_items + 1))
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise ValueError(f"{field_name} could not be read") from None
    if len(items) > max_items:
        raise ValueError(f"{field_name} exceeds the supported item limit")

    result: dict[str, Any] = {}
    for item in items:
        if type(item) not in {list, tuple} or len(item) != 2:
            raise ValueError(f"{field_name} contains an invalid entry")
        key, item_value = item
        if type(key) is not str:
            raise ValueError(f"{field_name} keys must be strings")
        if key in result:
            raise ValueError(f"{field_name} keys must be unique")
        result[key] = item_value
    return result


def _normalize_identifier(value: str, kind: str) -> str:
    """Normalize a bounded purpose or profile name without echoing input."""

    if type(value) is not str:
        raise TypeError(f"{kind} must be a string")
    stripped = value.strip()
    if not _SAFE_IDENTIFIER_RE.fullmatch(stripped):
        raise ValueError(f"{kind} must be a safe bounded identifier")
    return stripped.lower().replace("-", "_")


def _normalize_fields(
    value: Iterable[str] | None,
    *,
    field_name: str,
    max_items: int = MAX_FIELDS_PER_DECLARATION,
) -> tuple[str, ...]:
    """Return deterministic, bounded field names without exposing input."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must contain field names")
    try:
        iterator = iter(value)
    except TypeError:
        raise TypeError(f"{field_name} must contain field names") from None
    except Exception:  # noqa: BLE001 - iterables are caller-controlled protocols.
        raise ValueError(f"{field_name} could not be read") from None
    try:
        fields = list(itertools.islice(iterator, max_items + 1))
    except Exception:  # noqa: BLE001 - iterables are caller-controlled protocols.
        raise ValueError(f"{field_name} could not be read") from None
    if len(fields) > max_items:
        raise ValueError(f"{field_name} exceeds the supported item limit")

    normalized: set[str] = set()
    for raw_field_name in fields:
        if type(raw_field_name) is not str:
            raise ValueError(f"{field_name} must contain safe field identifiers")
        normalized_name = raw_field_name.strip()
        if len(
            normalized_name
        ) > _MAX_FIELD_NAME_LENGTH or not _SAFE_FIELD_RE.fullmatch(normalized_name):
            raise ValueError(f"{field_name} must contain safe field identifiers")
        if normalized_name in normalized:
            raise ValueError(f"{field_name} contains duplicate field identifiers")
        normalized.add(normalized_name)
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class PurposeMapping:
    """Bounded fields eligible and required for one declared purpose."""

    fields: tuple[str, ...]
    required_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fields = _normalize_fields(self.fields, field_name="purpose fields")
        required_fields = _normalize_fields(
            self.required_fields,
            field_name="required fields",
        )
        if not set(required_fields) <= set(fields):
            raise ValueError("required fields must be included in purpose fields")
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
    """Bounded allowlist and denylist for a structured-export profile."""

    name: str
    allowed_fields: tuple[str, ...] | None = None
    denied_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _normalize_identifier(self.name, "policy profile")
        allowed_fields = (
            None
            if self.allowed_fields is None
            else _normalize_fields(
                self.allowed_fields,
                field_name="allowed fields",
            )
        )
        denied_fields = _normalize_fields(
            self.denied_fields,
            field_name="denied fields",
        )
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
    """Validated, value-free metadata for one field-selection decision."""

    allowed: bool
    reason: str
    purpose: str | None
    policy_profile: str | None
    selected_fields: tuple[str, ...]
    omitted_fields: tuple[str, ...]
    required_fields: tuple[str, ...]
    available_field_count: int
    schema_version: int = MINIMUM_NECESSARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise ValueError("allowed must be a boolean")
        if type(self.reason) is not str or self.reason not in _REASONS:
            raise ValueError("reason must be a supported selection reason")
        if type(self.schema_version) is not int or (
            self.schema_version != MINIMUM_NECESSARY_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is unsupported")
        if type(self.available_field_count) is not int or not (
            0 <= self.available_field_count <= MAX_AVAILABLE_FIELDS
        ):
            raise ValueError("available_field_count is outside supported bounds")

        purpose = (
            None
            if self.purpose is None
            else _normalize_identifier(self.purpose, "purpose")
        )
        policy_profile = (
            None
            if self.policy_profile is None
            else _normalize_identifier(self.policy_profile, "policy profile")
        )
        selected_fields = _normalize_fields(
            self.selected_fields,
            field_name="selected fields",
        )
        omitted_fields = _normalize_fields(
            self.omitted_fields,
            field_name="omitted fields",
        )
        required_fields = _normalize_fields(
            self.required_fields,
            field_name="required fields",
        )

        expected_allowed = self.reason == _REASON_SELECTED
        if self.allowed is not expected_allowed:
            raise ValueError("selection outcome is inconsistent")
        if set(selected_fields) & set(omitted_fields):
            raise ValueError("selection field groups must be disjoint")
        if self.allowed:
            if purpose is None or policy_profile is None or not selected_fields:
                raise ValueError("allowed selection metadata is incomplete")
            if not set(required_fields) <= set(selected_fields):
                raise ValueError("allowed selection omits required fields")
        elif selected_fields:
            raise ValueError("denied selections cannot contain selected fields")

        if self.reason == _REASON_UNKNOWN_PURPOSE and any(
            (purpose, policy_profile, omitted_fields, required_fields)
        ):
            raise ValueError("unknown-purpose metadata is inconsistent")
        if self.reason == _REASON_UNKNOWN_POLICY and (
            purpose is None or policy_profile is not None
        ):
            raise ValueError("unknown-policy metadata is inconsistent")
        if self.reason in {
            _REASON_REQUIRED_MISSING,
            _REASON_REQUIRED_BLOCKED,
            _REASON_NO_FIELDS,
        } and (purpose is None or policy_profile is None):
            raise ValueError("denied selection metadata is incomplete")
        if self.reason in {_REASON_REQUIRED_MISSING, _REASON_REQUIRED_BLOCKED} and (
            not required_fields
        ):
            raise ValueError("required-field denial lacks required fields")

        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "policy_profile", policy_profile)
        object.__setattr__(self, "selected_fields", selected_fields)
        object.__setattr__(self, "omitted_fields", omitted_fields)
        object.__setattr__(self, "required_fields", required_fields)

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


@dataclass(frozen=True, init=False)
class FieldSelection:
    """Selector-created result that can project only approved field names."""

    selected_fields: tuple[str, ...]
    explanation: SelectionExplanation

    def __init__(
        self,
        selected_fields: tuple[str, ...],
        explanation: SelectionExplanation,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _SELECTION_TOKEN:
            raise TypeError("FieldSelection instances must be created by a selector")
        if type(explanation) is not SelectionExplanation:
            raise TypeError("explanation must be a SelectionExplanation")
        fields = _normalize_fields(
            selected_fields,
            field_name="selected fields",
        )
        if fields != explanation.selected_fields:
            raise ValueError("selection fields do not match the explanation")
        object.__setattr__(self, "selected_fields", fields)
        object.__setattr__(self, "explanation", explanation)

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
        """Return only approved fields without iterating over source values.

        If a required field is absent at projection time, fail closed instead
        of returning a partial export.
        """

        if not self.allowed:
            return {}
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        projected: dict[str, Any] = {}
        for field_name in self.selected_fields:
            try:
                projected[field_name] = record[field_name]
            except KeyError:
                if field_name in self.explanation.required_fields:
                    return {}
                continue
            except Exception:  # noqa: BLE001 - mapping access is caller-controlled.
                raise ValueError("record could not be projected safely") from None
        return projected

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free selection explanation."""

        return self.explanation.to_dict()

    def to_json(self) -> str:
        """Serialize the value-free selection explanation deterministically."""

        return self.explanation.to_json()


SelectionResult: TypeAlias = FieldSelection


class MinimumNecessarySelector:
    """Select fields using bounded caller-owned purpose and policy registries."""

    def __init__(
        self,
        purpose_mappings: Mapping[str, PurposeConfig],
        policy_profiles: Mapping[str, PolicyConfig],
    ) -> None:
        if not isinstance(purpose_mappings, Mapping):
            raise TypeError("purpose_mappings must be a mapping")
        if not isinstance(policy_profiles, Mapping):
            raise TypeError("policy_profiles must be a mapping")

        purpose_items = _snapshot_mapping(
            purpose_mappings,
            field_name="purpose mappings",
            max_items=MAX_PURPOSE_MAPPINGS,
        )
        profile_items = _snapshot_mapping(
            policy_profiles,
            field_name="policy profiles",
            max_items=MAX_POLICY_PROFILES,
        )

        normalized_purposes: dict[str, PurposeMapping] = {}
        for raw_name, raw_config in purpose_items.items():
            name = _normalize_identifier(raw_name, "purpose")
            if name in normalized_purposes:
                raise ValueError("purpose mappings contain duplicate names")
            normalized_purposes[name] = _coerce_purpose_mapping(raw_config)

        normalized_profiles: dict[str, FieldPolicyProfile] = {}
        for raw_name, raw_config in profile_items.items():
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
        """Select the minimum permitted field set for a declared request."""

        purpose_name = _normalize_identifier(purpose, "purpose")
        purpose_mapping = self._purpose_mappings.get(purpose_name)
        if purpose_mapping is None:
            return _denied_selection(
                reason=_REASON_UNKNOWN_PURPOSE,
                purpose=None,
                policy_profile=None,
            )

        profile_name, profile = self._resolve_profile(policy_profile)
        if profile is None:
            return _denied_selection(
                reason=_REASON_UNKNOWN_POLICY,
                purpose=purpose_name,
                policy_profile=None,
            )

        available = (
            purpose_mapping.fields
            if available_fields is None
            else _available_fields(available_fields)
        )
        purpose_fields = set(purpose_mapping.fields)
        available_set = set(available)
        eligible = purpose_fields & available_set
        if profile.allowed_fields is not None:
            eligible &= set(profile.allowed_fields)
        eligible -= set(profile.denied_fields)
        selected = tuple(sorted(eligible))
        omitted = tuple(sorted(purpose_fields - set(selected)))
        required = set(purpose_mapping.required_fields)

        if required - available_set:
            return _denied_selection(
                reason=_REASON_REQUIRED_MISSING,
                purpose=purpose_name,
                policy_profile=profile_name,
                omitted_fields=omitted,
                required_fields=purpose_mapping.required_fields,
                available_field_count=len(available),
            )
        if required - eligible:
            return _denied_selection(
                reason=_REASON_REQUIRED_BLOCKED,
                purpose=purpose_name,
                policy_profile=profile_name,
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
        return _selection(explanation)

    def _resolve_profile(
        self,
        policy_profile: str | FieldPolicyProfile,
    ) -> tuple[str | None, FieldPolicyProfile | None]:
        if type(policy_profile) is FieldPolicyProfile:
            return policy_profile.name, policy_profile
        profile_name = _normalize_identifier(
            cast(str, policy_profile), "policy profile"
        )
        return profile_name, self._policy_profiles.get(profile_name)


def _coerce_purpose_mapping(config: PurposeConfig) -> PurposeMapping:
    if type(config) is PurposeMapping:
        return config
    if isinstance(config, Mapping):
        values = _snapshot_mapping(
            config,
            field_name="purpose configuration",
            max_items=len(_PURPOSE_CONFIG_FIELDS),
        )
        if set(values) - _PURPOSE_CONFIG_FIELDS:
            raise ValueError("purpose configuration contains unsupported fields")
        if "fields" not in values:
            raise ValueError("purpose mappings must declare fields")
        return PurposeMapping(
            fields=values["fields"],
            required_fields=values.get("required_fields", ()),
        )
    return PurposeMapping(fields=cast(tuple[str, ...], config))


def _coerce_policy_profile(
    name: str,
    config: PolicyConfig,
) -> FieldPolicyProfile:
    if type(config) is FieldPolicyProfile:
        return FieldPolicyProfile(
            name=name,
            allowed_fields=config.allowed_fields,
            denied_fields=config.denied_fields,
        )
    if isinstance(config, Mapping):
        values = _snapshot_mapping(
            config,
            field_name="policy configuration",
            max_items=len(_POLICY_CONFIG_FIELDS),
        )
        if set(values) - _POLICY_CONFIG_FIELDS:
            raise ValueError("policy configuration contains unsupported fields")
        if "allowed_fields" in values and "fields" in values:
            raise ValueError(
                "policy configuration contains duplicate allowlist aliases"
            )
        allowed_fields = values.get("allowed_fields", values.get("fields"))
        return FieldPolicyProfile(
            name=name,
            allowed_fields=allowed_fields,
            denied_fields=values.get("denied_fields", ()),
        )
    return FieldPolicyProfile(
        name=name,
        allowed_fields=cast(tuple[str, ...], config),
    )


def _available_fields(available_fields: FieldInput) -> tuple[str, ...]:
    return _normalize_fields(
        available_fields,
        field_name="available fields",
        max_items=MAX_AVAILABLE_FIELDS,
    )


def _selection(explanation: SelectionExplanation) -> FieldSelection:
    return FieldSelection(
        explanation.selected_fields,
        explanation,
        _token=_SELECTION_TOKEN,
    )


def _denied_selection(
    *,
    reason: str,
    purpose: str | None,
    policy_profile: str | None,
    omitted_fields: tuple[str, ...] = (),
    required_fields: tuple[str, ...] = (),
    available_field_count: int = 0,
) -> FieldSelection:
    return _selection(
        SelectionExplanation(
            allowed=False,
            reason=reason,
            purpose=purpose,
            policy_profile=policy_profile,
            selected_fields=(),
            omitted_fields=omitted_fields,
            required_fields=required_fields,
            available_field_count=available_field_count,
        )
    )


def select_fields(
    available_fields: FieldInput,
    purpose: str,
    policy_profile: str | FieldPolicyProfile,
    *,
    purpose_mappings: Mapping[str, PurposeConfig],
    policy_profiles: Mapping[str, PolicyConfig],
) -> FieldSelection:
    """Select fields from caller-declared purpose and policy registries."""

    return MinimumNecessarySelector(purpose_mappings, policy_profiles).select(
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
    "MAX_AVAILABLE_FIELDS",
    "MAX_FIELDS_PER_DECLARATION",
    "MAX_POLICY_PROFILES",
    "MAX_PURPOSE_MAPPINGS",
    "FieldPolicyProfile",
    "FieldSelection",
    "MinimumNecessarySelector",
    "PurposeMapping",
    "SelectionExplanation",
    "SelectionResult",
    "select_fields",
    "select_minimum_necessary_fields",
]
