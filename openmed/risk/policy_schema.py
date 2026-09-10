"""Versioned, privacy-safe policy-as-data declarations.

The schema in this module describes policy intent.  It does not execute a
de-identification run, load a model, or resolve a remote policy document.  A
policy contains only bounded configuration values and references to operator-
managed key material; source text, mappings, and secrets are deliberately not
valid schema fields.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

from openmed.core.labels import (
    CANONICAL_LABELS,
    POLICY_LABELS,
    normalize_label,
    policy_label_for,
)
from openmed.core.schemas.span import ACTION_VALUES

CURRENT_POLICY_SCHEMA_VERSION: Final = 1
POLICY_SCHEMA_VERSION: Final = CURRENT_POLICY_SCHEMA_VERSION

DEFAULT_POLICY_NAME: Final = "default"
DEFAULT_JURISDICTION: Final = "UNSPECIFIED"
DEFAULT_RECALL_FLOOR: Final = 0.97
DEFAULT_DIRECT_IDENTIFIER_RECALL_FLOOR: Final = 0.99
DEFAULT_CRITICAL_RECALL_FLOOR: Final = 1.0
DEFAULT_ACTION: Final = "mask"
DEFAULT_SURROGATE_STRATEGY: Final = "none"
DEFAULT_AUDIT_RETENTION_DAYS: Final = 0
MAX_AUDIT_RETENTION_DAYS: Final = 36_500
MAX_POLICY_ACTIONS: Final = 512
MAX_POLICY_JSON_BYTES: Final = 1_048_576
MAX_POLICY_RECALL_OVERRIDES: Final = 512

_MAX_POLICY_OBJECT_ITEMS: Final = 64

SUPPORTED_ACTIONS: Final = frozenset(ACTION_VALUES)
SUPPORTED_SURROGATE_STRATEGIES: Final = frozenset(
    {
        "none",
        "random",
        "deterministic",
        "format_preserving",
        "date_shift",
        "vault",
    }
)

_ACTION_ALIASES: Final[Mapping[str, str]] = {
    "format-preserve": "format_preserve",
    "format_preserving": "format_preserve",
}
_SURROGATE_ALIASES: Final[Mapping[str, str]] = {
    "mask": "none",
    "replace": "random",
    "consistent_replace": "deterministic",
    "format-preserve": "format_preserving",
    "format_preserve": "format_preserving",
    "shift_dates": "date_shift",
    "hash": "deterministic",
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_JURISDICTION_CODE_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{1,31}$")


def _safe_text(value: Any, field_name: str, *, max_length: int = 128) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    text = value.strip()
    if len(text) > max_length or "\n" in text or "\r" in text:
        raise ValueError(f"{field_name} must be a bounded single-line string")
    return text


def _safe_identifier(value: Any, field_name: str) -> str:
    text = _safe_text(value, field_name)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a stable identifier")
    return text


def _bounded_probability(value: Any, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a numeric probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return result


def _strict_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _positive_or_zero_int(
    value: Any,
    field_name: str,
    *,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} exceeds the supported maximum")
    return value


def _snapshot_mapping(
    value: Any,
    field_name: str,
    *,
    maximum_items: int,
) -> dict[str, Any]:
    """Copy an external mapping without allowing its exceptions to leak values."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    try:
        items = list(itertools.islice(value.items(), maximum_items + 1))
    except Exception:
        raise ValueError(f"{field_name} must be a readable object") from None
    if len(items) > maximum_items:
        raise ValueError(f"{field_name} exceeds the supported item limit")

    snapshot: dict[str, Any] = {}
    for pair in items:
        if type(pair) not in {list, tuple} or len(pair) != 2:
            raise ValueError(f"{field_name} must be a readable object")
        key, item = pair
        if type(key) is not str:
            raise TypeError(f"{field_name} field names must be strings")
        if key in snapshot:
            raise ValueError(f"{field_name} contains duplicate field names")
        snapshot[key] = item
    return snapshot


def _aliased_value(
    value: Mapping[str, Any],
    names: tuple[str, ...],
    field_name: str,
    *,
    default: Any = None,
) -> Any:
    present = [name for name in names if name in value]
    if len(present) > 1:
        raise ValueError(f"{field_name} uses conflicting aliases")
    return value[present[0]] if present else default


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate policy JSON field")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite policy JSON number")


def _unknown_fields(
    value: Mapping[str, Any], allowed: set[str], field_name: str
) -> None:
    if set(value) - allowed:
        raise ValueError(f"{field_name} contains unsupported field(s)")


def _canonical_action(value: Any, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a supported action")
    action = value.strip().lower()
    action = _ACTION_ALIASES.get(action, action)
    if action not in SUPPORTED_ACTIONS:
        # Do not echo the configured value: callers may have accidentally put
        # an input value or other sensitive material in a policy field.
        allowed = ", ".join(sorted(SUPPORTED_ACTIONS))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return action


def _canonical_label(value: Any, field_name: str) -> str:
    label = _safe_text(value, field_name)
    normalized = normalize_label(label)
    candidate = label.upper().replace("-", "_")
    if normalized != "OTHER":
        return normalized
    if candidate in CANONICAL_LABELS or candidate in POLICY_LABELS:
        return candidate
    if not _IDENTIFIER_RE.fullmatch(candidate):
        raise ValueError(f"{field_name} must be a stable label identifier")
    return candidate


def _canonical_mapping_key(value: Any, field_name: str) -> str:
    return _canonical_label(value, field_name)


def _freeze_actions(
    value: Mapping[str, Any], field_name: str = "actions"
) -> Mapping[str, str]:
    actions: dict[str, str] = {}
    for raw_label, raw_action in value.items():
        label = _canonical_mapping_key(raw_label, f"{field_name} label")
        action = _canonical_action(raw_action, f"{field_name} value")
        if label in actions:
            raise ValueError(f"{field_name} contains duplicate labels")
        actions[label] = action
    if len(actions) > MAX_POLICY_ACTIONS:
        raise ValueError(f"{field_name} exceeds the supported item limit")
    return MappingProxyType(dict(sorted(actions.items())))


@dataclass(frozen=True, slots=True)
class Jurisdiction:
    """Bounded jurisdiction metadata with no legal-text or source-data field."""

    code: str = DEFAULT_JURISDICTION
    name: str = "unspecified"
    region: str | None = None

    def __post_init__(self) -> None:
        code = _safe_text(self.code, "jurisdiction.code").upper()
        if not _JURISDICTION_CODE_RE.fullmatch(code):
            raise ValueError("jurisdiction.code must be a stable jurisdiction code")
        name = _safe_text(self.name, "jurisdiction.name")
        region = (
            None
            if self.region is None
            else _safe_text(self.region, "jurisdiction.region")
        )
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "region", region)

    @classmethod
    def from_value(cls, value: Any) -> "Jurisdiction":
        """Parse a jurisdiction string or the version-one object form."""

        if type(value) is cls:
            return value
        if value is None:
            return cls()
        if type(value) is str:
            text = _safe_text(value, "jurisdiction")
            code = (
                text.upper()
                if _JURISDICTION_CODE_RE.fullmatch(text.upper())
                else "CUSTOM"
            )
            return cls(code=code, name=text)
        value = _snapshot_mapping(value, "jurisdiction", maximum_items=6)
        _unknown_fields(
            value,
            {"code", "country_code", "id", "name", "country", "region"},
            "jurisdiction",
        )
        code = _aliased_value(
            value,
            ("code", "country_code", "id"),
            "jurisdiction.code",
            default=DEFAULT_JURISDICTION,
        )
        name = _aliased_value(
            value,
            ("name", "country"),
            "jurisdiction.name",
            default=code,
        )
        return cls(code=code, name=name, region=value.get("region"))

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-compatible jurisdiction object."""

        return {"code": self.code, "name": self.name, "region": self.region}


@dataclass(frozen=True, slots=True)
class RecallFloors:
    """Recall thresholds for the default, direct-identifier, and critical paths."""

    default: float = DEFAULT_RECALL_FLOOR
    direct_identifier: float = DEFAULT_DIRECT_IDENTIFIER_RECALL_FLOOR
    critical: float = DEFAULT_CRITICAL_RECALL_FLOOR
    by_label: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "default",
            _bounded_probability(self.default, "recall_floors.default"),
        )
        object.__setattr__(
            self,
            "direct_identifier",
            _bounded_probability(
                self.direct_identifier,
                "recall_floors.direct_identifier",
            ),
        )
        object.__setattr__(
            self,
            "critical",
            _bounded_probability(self.critical, "recall_floors.critical"),
        )
        by_label = _snapshot_mapping(
            self.by_label,
            "recall_floors.by_label",
            maximum_items=MAX_POLICY_RECALL_OVERRIDES,
        )
        floors: dict[str, float] = {}
        for raw_label, value in by_label.items():
            label = _canonical_label(raw_label, "recall_floors.by_label label")
            if label in floors:
                raise ValueError("recall_floors.by_label contains duplicate labels")
            floors[label] = _bounded_probability(
                value,
                "recall_floors.by_label value",
            )
        object.__setattr__(
            self, "by_label", MappingProxyType(dict(sorted(floors.items())))
        )

    @classmethod
    def from_value(cls, value: Any) -> "RecallFloors":
        """Parse nested, flat-label, and legacy scalar recall configurations."""

        if type(value) is cls:
            return value
        if value is None:
            return cls()
        if type(value) in {int, float}:
            floor = _bounded_probability(value, "recall_floor")
            return cls(default=floor, direct_identifier=floor, critical=floor)
        structural = {
            "default",
            "overall",
            "recall_floor",
            "direct_identifier",
            "direct_identifiers",
            "critical",
            "critical_identifier",
            "by_label",
            "per_label",
        }
        value = _snapshot_mapping(
            value,
            "recall_floors",
            maximum_items=MAX_POLICY_RECALL_OVERRIDES + len(structural),
        )

        default = _aliased_value(
            value,
            ("default", "overall", "recall_floor"),
            "recall_floors.default",
            default=DEFAULT_RECALL_FLOOR,
        )
        direct = _aliased_value(
            value,
            ("direct_identifier", "direct_identifiers"),
            "recall_floors.direct_identifier",
            default=DEFAULT_DIRECT_IDENTIFIER_RECALL_FLOOR,
        )
        critical = _aliased_value(
            value,
            ("critical", "critical_identifier"),
            "recall_floors.critical",
            default=DEFAULT_CRITICAL_RECALL_FLOOR,
        )
        configured = _aliased_value(
            value,
            ("by_label", "per_label"),
            "recall_floors.by_label",
            default={},
        )
        if configured is None:
            configured = {}
        configured = _snapshot_mapping(
            configured,
            "recall_floors.by_label",
            maximum_items=MAX_POLICY_RECALL_OVERRIDES,
        )
        flat_labels = {
            key: item for key, item in value.items() if key not in structural
        }
        combined = dict(flat_labels)
        if set(combined) & set(configured):
            raise ValueError("recall_floors.by_label contains duplicate labels")
        combined.update(configured)
        if len(combined) > MAX_POLICY_RECALL_OVERRIDES:
            raise ValueError("recall_floors.by_label exceeds the supported item limit")
        return cls(
            default=default,
            direct_identifier=direct,
            critical=critical,
            by_label=combined,
        )

    def floor_for(self, label: str, *, category: str | None = None) -> float:
        """Return the most specific floor for a canonical or source label."""

        canonical = _canonical_label(label, "label")
        if canonical in self.by_label:
            return self.by_label[canonical]
        selected_category = category
        if selected_category is None:
            try:
                selected_category = policy_label_for(canonical)
            except (KeyError, TypeError, ValueError):
                selected_category = None
        if selected_category == "DIRECT_IDENTIFIER":
            return self.direct_identifier
        if selected_category == "CRITICAL":
            return self.critical
        return self.default

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible recall-floor data."""

        return {
            "default": self.default,
            "direct_identifier": self.direct_identifier,
            "critical": self.critical,
            "by_label": dict(self.by_label),
        }


def _canonical_surrogate_strategy(value: Any) -> str:
    if type(value) is not str:
        raise TypeError("surrogate_strategy.kind must be a supported strategy")
    strategy = value.strip().lower().replace("-", "_")
    strategy = _SURROGATE_ALIASES.get(strategy, strategy)
    if strategy not in SUPPORTED_SURROGATE_STRATEGIES:
        allowed = ", ".join(sorted(SUPPORTED_SURROGATE_STRATEGIES))
        raise ValueError(f"surrogate_strategy.kind must be one of: {allowed}")
    return strategy


@dataclass(frozen=True, slots=True)
class SurrogateStrategy:
    """Describe surrogate behavior without embedding keys, seeds, or mappings."""

    kind: str = DEFAULT_SURROGATE_STRATEGY
    consistent: bool = False
    reversible: bool = False
    key_ref: str | None = None

    def __post_init__(self) -> None:
        kind = _canonical_surrogate_strategy(self.kind)
        consistent = _strict_bool(self.consistent, "surrogate_strategy.consistent")
        reversible = _strict_bool(self.reversible, "surrogate_strategy.reversible")
        key_ref = (
            None
            if self.key_ref is None
            else _safe_identifier(self.key_ref, "surrogate_strategy.key_ref")
        )
        if kind == "none" and (consistent or reversible or key_ref is not None):
            raise ValueError(
                "surrogate_strategy.none cannot retain or reference surrogate state"
            )
        if reversible and key_ref is None:
            raise ValueError("reversible surrogate strategies require key_ref")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "consistent", consistent)
        object.__setattr__(self, "reversible", reversible)
        object.__setattr__(self, "key_ref", key_ref)

    @classmethod
    def from_value(cls, value: Any) -> "SurrogateStrategy":
        """Parse a strategy string or bounded strategy object."""

        if type(value) is cls:
            return value
        if value is None:
            return cls()
        if type(value) is str:
            kind = _canonical_surrogate_strategy(value)
            return cls(
                kind=kind,
                consistent=kind in {"deterministic", "format_preserving", "vault"},
            )
        value = _snapshot_mapping(
            value,
            "surrogate_strategy",
            maximum_items=10,
        )

        forbidden = {
            "secret",
            "key",
            "token",
            "password",
            "seed",
            "mapping",
            "source_values",
        }
        if any(key in forbidden for key in value):
            raise ValueError(
                "surrogate_strategy may contain only an operator key_ref, never key material or mappings"
            )
        _unknown_fields(
            value,
            {
                "kind",
                "type",
                "mode",
                "strategy",
                "consistent",
                "stable",
                "reversible",
                "reversible_id",
                "key_ref",
                "key_id",
            },
            "surrogate_strategy",
        )
        raw_kind = _aliased_value(
            value,
            ("kind", "type", "mode", "strategy"),
            "surrogate_strategy.kind",
            default=DEFAULT_SURROGATE_STRATEGY,
        )
        kind = _canonical_surrogate_strategy(raw_kind)
        default_consistent = kind in {"deterministic", "format_preserving", "vault"}
        consistent = _aliased_value(
            value,
            ("consistent", "stable"),
            "surrogate_strategy.consistent",
            default=default_consistent,
        )
        reversible = _aliased_value(
            value,
            ("reversible", "reversible_id"),
            "surrogate_strategy.reversible",
            default=False,
        )
        key_ref = _aliased_value(
            value,
            ("key_ref", "key_id"),
            "surrogate_strategy.key_ref",
        )
        return cls(
            kind=kind,
            consistent=consistent,
            reversible=reversible,
            key_ref=key_ref,
        )

    @property
    def strategy(self) -> str:
        """Compatibility alias for callers that use ``strategy`` terminology."""

        return self.kind

    def to_dict(self) -> dict[str, Any]:
        """Return a safe strategy object; no secret or source value is emitted."""

        return {
            "kind": self.kind,
            "consistent": self.consistent,
            "reversible": self.reversible,
            "key_ref": self.key_ref,
        }


@dataclass(frozen=True, slots=True)
class AuditRetention:
    """Retention controls for privacy-safe audit artifacts only."""

    enabled: bool = False
    retention_days: int = DEFAULT_AUDIT_RETENTION_DAYS
    include_text: bool = False
    include_mappings: bool = False

    def __post_init__(self) -> None:
        enabled = _strict_bool(self.enabled, "audit_retention.enabled")
        days = _positive_or_zero_int(
            self.retention_days,
            "audit_retention.retention_days",
            maximum=MAX_AUDIT_RETENTION_DAYS,
        )
        include_text = _strict_bool(self.include_text, "audit_retention.include_text")
        include_mappings = _strict_bool(
            self.include_mappings,
            "audit_retention.include_mappings",
        )
        if include_text or include_mappings:
            raise ValueError(
                "audit_retention can retain only privacy-safe hashes, offsets, and aggregates"
            )
        if enabled and days == 0:
            raise ValueError("enabled audit_retention requires retention_days")
        if not enabled and days != 0:
            raise ValueError("disabled audit_retention must use retention_days=0")
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "retention_days", days)
        object.__setattr__(self, "include_text", False)
        object.__setattr__(self, "include_mappings", False)

    @classmethod
    def from_value(cls, value: Any) -> "AuditRetention":
        """Parse retention days or the explicit version-one object form."""

        if type(value) is cls:
            return value
        if value is None:
            return cls()
        if type(value) is int:
            days = _positive_or_zero_int(
                value,
                "audit_retention.retention_days",
                maximum=MAX_AUDIT_RETENTION_DAYS,
            )
            return cls(enabled=days > 0, retention_days=days)
        value = _snapshot_mapping(
            value,
            "audit_retention",
            maximum_items=13,
        )
        _unknown_fields(
            value,
            {
                "enabled",
                "retention_days",
                "days",
                "include_text",
                "store_text",
                "include_mappings",
                "store_mappings",
                "store_surrogate_mappings",
                "raw_text",
                "source_text",
                "surrogate_mapping",
            },
            "audit_retention",
        )
        days = _aliased_value(
            value,
            ("retention_days", "days"),
            "audit_retention.retention_days",
            default=0,
        )
        enabled_value = value["enabled"] if "enabled" in value else days != 0
        return cls(
            enabled=enabled_value,
            retention_days=days,
            include_text=_aliased_value(
                value,
                ("include_text", "store_text", "raw_text", "source_text"),
                "audit_retention.include_text",
                default=False,
            ),
            include_mappings=_aliased_value(
                value,
                (
                    "include_mappings",
                    "store_mappings",
                    "store_surrogate_mappings",
                    "surrogate_mapping",
                ),
                "audit_retention.include_mappings",
                default=False,
            ),
        )

    @property
    def days(self) -> int:
        """Compatibility alias for the retention period."""

        return self.retention_days

    def to_dict(self) -> dict[str, Any]:
        """Return retention settings that cannot serialize raw audit material."""

        return {
            "enabled": self.enabled,
            "retention_days": self.retention_days,
            "include_text": False,
            "include_mappings": False,
        }


_TOP_LEVEL_FIELDS: Final = {
    "schema_version",
    "version",
    "name",
    "policy_name",
    "policy_id",
    "id",
    "jurisdiction",
    "recall_floors",
    "recall_floor",
    "label_recall_floors",
    "default_action",
    "actions",
    "policy_label_actions",
    "surrogate_strategy",
    "surrogate",
    "method",
    "consistent",
    "reversible_id",
    "keep_mapping",
    "audit_retention",
    "audit",
    # Existing policy-profile fields are accepted as compatibility input but
    # are intentionally not copied into this narrower risk contract.
    "posture",
    "threshold_profile",
    "default_action_bias",
    "arbitration_mode",
    "strict_no_leak",
    "safety_sweep_mandatory",
    "forced_cascade_tiers",
    "metadata",
}


def _merge_action_entries(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
    field_name: str,
) -> None:
    for key, item in incoming.items():
        if key in target:
            raise ValueError(f"{field_name} contains duplicate labels")
        target[key] = item


def _action_mapping_from_payload(value: Any) -> tuple[dict[str, Any], Any]:
    if value is None:
        return {}, None
    value = _snapshot_mapping(
        value,
        "actions",
        maximum_items=MAX_POLICY_ACTIONS + 3,
    )
    result: dict[str, Any] = {}
    by_label = value.get("by_label")
    if by_label is not None:
        by_label = _snapshot_mapping(
            by_label,
            "actions.by_label",
            maximum_items=MAX_POLICY_ACTIONS,
        )
        _merge_action_entries(result, by_label, "actions")
    for key, item in value.items():
        if key in {"by_label", "default", "default_action"}:
            continue
        _merge_action_entries(result, {key: item}, "actions")
    if len(result) > MAX_POLICY_ACTIONS:
        raise ValueError("actions exceeds the supported item limit")
    embedded_default = _aliased_value(
        value,
        ("default_action", "default"),
        "actions.default_action",
    )
    return result, embedded_default


def _legacy_surrogate(value: Mapping[str, Any]) -> Any:
    present = [
        name for name in ("surrogate_strategy", "surrogate", "method") if name in value
    ]
    if not present:
        return None
    selected = _aliased_value(
        value,
        ("surrogate_strategy", "surrogate", "method"),
        "surrogate_strategy",
    )
    if present[0] != "method":
        return selected
    method = selected
    if type(method) is not str:
        raise TypeError("method must be a supported de-identification method")
    kind = _SURROGATE_ALIASES.get(method.strip().lower().replace("-", "_"), "none")
    if method.strip().lower() not in {
        "mask",
        "replace",
        "format_preserve",
        "format-preserve",
        "shift_dates",
        "hash",
    }:
        # Let the strategy validator produce the bounded supported-values error.
        kind = method
    consistent = value.get(
        "consistent", kind in {"deterministic", "format_preserving", "vault"}
    )
    reversible = _aliased_value(
        value,
        ("reversible_id", "keep_mapping"),
        "surrogate_strategy.reversible",
        default=False,
    )
    return {
        "kind": kind,
        "consistent": consistent,
        "reversible": reversible,
        "key_ref": "operator-managed-key" if reversible else None,
    }


@dataclass(frozen=True, slots=True)
class PrivacyPolicy:
    """An immutable, version-one privacy policy declaration.

    The defaults preserve OpenMed's legacy local behavior: masking is the
    default action, no surrogate state is retained, and audit retention is
    disabled.  Recall defaults are explicit so a policy diff cannot silently
    inherit an evaluator or model default.
    """

    schema_version: int = CURRENT_POLICY_SCHEMA_VERSION
    name: str = DEFAULT_POLICY_NAME
    jurisdiction: Jurisdiction | Mapping[str, Any] | str | None = field(
        default_factory=Jurisdiction
    )
    recall_floors: RecallFloors | Mapping[str, Any] | float | None = field(
        default_factory=RecallFloors
    )
    default_action: str = DEFAULT_ACTION
    actions: Mapping[str, Any] = field(default_factory=dict)
    surrogate_strategy: SurrogateStrategy | Mapping[str, Any] | str | None = field(
        default_factory=SurrogateStrategy
    )
    audit_retention: AuditRetention | Mapping[str, Any] | int | None = field(
        default_factory=AuditRetention
    )

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != CURRENT_POLICY_SCHEMA_VERSION
        ):
            raise ValueError("policy schema_version is unsupported; expected version 1")
        object.__setattr__(self, "name", _safe_identifier(self.name, "name"))
        object.__setattr__(
            self,
            "jurisdiction",
            Jurisdiction.from_value(self.jurisdiction),
        )
        object.__setattr__(
            self,
            "recall_floors",
            RecallFloors.from_value(self.recall_floors),
        )

        actions, embedded_default = _action_mapping_from_payload(self.actions)
        resolved_default = (
            embedded_default if embedded_default is not None else self.default_action
        )
        resolved_default = _canonical_action(resolved_default, "default_action")
        object.__setattr__(self, "default_action", resolved_default)
        object.__setattr__(self, "actions", _freeze_actions(actions))
        object.__setattr__(
            self,
            "surrogate_strategy",
            SurrogateStrategy.from_value(self.surrogate_strategy),
        )
        object.__setattr__(
            self,
            "audit_retention",
            AuditRetention.from_value(self.audit_retention),
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | "PrivacyPolicy"
    ) -> "PrivacyPolicy":
        """Build a policy from version-one or supported legacy mapping data."""

        if type(value) is cls:
            return value
        value = _snapshot_mapping(
            value,
            "policy schema",
            maximum_items=_MAX_POLICY_OBJECT_ITEMS,
        )
        _unknown_fields(value, _TOP_LEVEL_FIELDS, "policy")

        schema_version = _aliased_value(
            value,
            ("schema_version", "version"),
            "policy schema_version",
            default=CURRENT_POLICY_SCHEMA_VERSION,
        )
        name = _aliased_value(
            value,
            ("name", "policy_name", "policy_id", "id"),
            "policy name",
            default=DEFAULT_POLICY_NAME,
        )
        recall_value = value.get("recall_floors")
        if "recall_floors" in value and (
            "recall_floor" in value or "label_recall_floors" in value
        ):
            raise ValueError("recall_floors uses conflicting aliases")
        if "recall_floors" not in value:
            if "label_recall_floors" in value:
                recall_value = {
                    "default": value.get("recall_floor", DEFAULT_RECALL_FLOOR),
                    "by_label": value["label_recall_floors"],
                }
            else:
                recall_value = value.get("recall_floor")

        actions, embedded_default = _action_mapping_from_payload(value.get("actions"))
        policy_label_actions = value.get("policy_label_actions")
        if policy_label_actions is not None:
            policy_label_actions = _snapshot_mapping(
                policy_label_actions,
                "policy_label_actions",
                maximum_items=MAX_POLICY_ACTIONS,
            )
            _merge_action_entries(actions, policy_label_actions, "actions")
        if len(actions) > MAX_POLICY_ACTIONS:
            raise ValueError("actions exceeds the supported item limit")
        if "default_action" in value and embedded_default is not None:
            raise ValueError("default_action uses conflicting aliases")
        default_action = (
            value["default_action"]
            if "default_action" in value
            else (embedded_default if embedded_default is not None else DEFAULT_ACTION)
        )

        surrogate = _legacy_surrogate(value)
        audit = _aliased_value(
            value,
            ("audit_retention", "audit"),
            "audit_retention",
        )
        if "audit" in value and audit is not None:
            audit = audit if isinstance(audit, Mapping) else {"enabled": audit}

        return cls(
            schema_version=schema_version,
            name=name,
            jurisdiction=value.get("jurisdiction"),
            recall_floors=recall_value,
            default_action=default_action,
            actions=actions,
            surrogate_strategy=surrogate,
            audit_retention=audit,
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "PrivacyPolicy":
        """Parse a JSON object without accessing the network."""

        if type(value) not in {str, bytes}:
            raise TypeError("policy JSON must be text or bytes")
        try:
            size = len(value.encode("utf-8") if type(value) is str else value)
        except UnicodeError:
            raise ValueError("invalid policy JSON") from None
        if size > MAX_POLICY_JSON_BYTES:
            raise ValueError("policy JSON exceeds the supported size limit")
        try:
            payload = json.loads(
                value,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise ValueError("invalid policy JSON") from None
        return cls.from_mapping(payload)

    def action_for(self, label: str) -> str:
        """Return the configured action for a label or the explicit default."""

        canonical = _canonical_label(label, "label")
        action = self.actions.get(canonical)
        if action is not None:
            return action
        try:
            policy_label = policy_label_for(canonical)
        except (KeyError, TypeError, ValueError):
            policy_label = None
        if policy_label is not None and policy_label in self.actions:
            return self.actions[policy_label]
        return self.default_action

    def recall_floor_for(self, label: str, *, category: str | None = None) -> float:
        """Return the configured recall floor for a label."""

        return cast(RecallFloors, self.recall_floors).floor_for(
            label,
            category=category,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, JSON-compatible policy data."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "jurisdiction": cast(Jurisdiction, self.jurisdiction).to_dict(),
            "recall_floors": cast(RecallFloors, self.recall_floors).to_dict(),
            "default_action": self.default_action,
            "actions": dict(self.actions),
            "surrogate_strategy": cast(
                SurrogateStrategy,
                self.surrogate_strategy,
            ).to_dict(),
            "audit_retention": cast(
                AuditRetention,
                self.audit_retention,
            ).to_dict(),
        }

    def canonical_json(self) -> str:
        """Return the canonical JSON representation used for fingerprints."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize policy data without adding any source or sensitive values."""

        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 digest of canonical policy data."""

        return (
            "sha256:"
            + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        )

    @property
    def fingerprint(self) -> str:
        """Compatibility alias for :attr:`digest`."""

        return self.digest


PolicySchema = PrivacyPolicy
PrivacyPolicySchema = PrivacyPolicy
PolicyDefinition = PrivacyPolicy


def default_policy_schema() -> PrivacyPolicy:
    """Return a fresh policy containing all backwards-compatible defaults."""

    return PrivacyPolicy()


def _read_policy_json(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_POLICY_JSON_BYTES + 1)
    except OSError:
        raise ValueError("could not read a valid local policy JSON document") from None
    if len(payload) > MAX_POLICY_JSON_BYTES:
        raise ValueError("policy JSON exceeds the supported size limit")
    return payload


def load_policy_schema(source: Any = None) -> PrivacyPolicy:
    """Load a policy from a mapping, JSON text, or a local JSON path.

    URLs are intentionally rejected.  Loading a policy must remain local and
    deterministic after the caller has supplied the configuration.
    """

    if source is None:
        return default_policy_schema()
    if type(source) is PrivacyPolicy:
        return source
    if isinstance(source, Mapping):
        return PrivacyPolicy.from_mapping(source)
    if type(source) is bytes:
        return PrivacyPolicy.from_json(source)
    if type(source) is not str and not isinstance(source, Path):
        raise TypeError("policy source must be a mapping, JSON text, or local path")

    if type(source) is str and source.lstrip().startswith(("{", "[")):
        return PrivacyPolicy.from_json(source)
    path_text = str(source)
    if "://" in path_text:
        raise ValueError("policy schema source must be local; URLs are unsupported")
    return PrivacyPolicy.from_json(_read_policy_json(Path(source)))


def validate_policy_schema(source: Any) -> None:
    """Validate policy data and raise without echoing configured values."""

    load_policy_schema(source)


def lint_policy_schema(source: Any) -> tuple[str, ...]:
    """Return privacy-safe validation diagnostics instead of raising."""

    try:
        validate_policy_schema(source)
    except Exception:
        return ("policy schema is invalid",)
    return ()


__all__ = [
    "AuditRetention",
    "CURRENT_POLICY_SCHEMA_VERSION",
    "DEFAULT_ACTION",
    "DEFAULT_AUDIT_RETENTION_DAYS",
    "DEFAULT_CRITICAL_RECALL_FLOOR",
    "DEFAULT_DIRECT_IDENTIFIER_RECALL_FLOOR",
    "DEFAULT_JURISDICTION",
    "DEFAULT_POLICY_NAME",
    "DEFAULT_RECALL_FLOOR",
    "DEFAULT_SURROGATE_STRATEGY",
    "Jurisdiction",
    "MAX_AUDIT_RETENTION_DAYS",
    "MAX_POLICY_ACTIONS",
    "MAX_POLICY_JSON_BYTES",
    "MAX_POLICY_RECALL_OVERRIDES",
    "POLICY_SCHEMA_VERSION",
    "PolicyDefinition",
    "PolicySchema",
    "PrivacyPolicy",
    "PrivacyPolicySchema",
    "RecallFloors",
    "SUPPORTED_ACTIONS",
    "SUPPORTED_SURROGATE_STRATEGIES",
    "SurrogateStrategy",
    "default_policy_schema",
    "lint_policy_schema",
    "load_policy_schema",
    "validate_policy_schema",
]
