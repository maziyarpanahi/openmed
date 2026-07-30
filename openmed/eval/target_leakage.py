"""Per-family Model-Steward target-leakage config and reader.

The Model Steward signs off on each manifest family's target leakage rate,
keyed by ``(family, tier, format)``. The committed, versioned config lives at
``gates/target_leakage.yaml`` and is the single source of truth that the
ReleaseGate harness reads to obtain the per-family leakage floor it must
enforce. This module loads and validates that config and exposes
:func:`get_targets` as the reader the harness consumes.

The config schema is intentionally small and versioned:

```
schema_version: 1
steward_owner: "model-stewardship-council"
defaults:
  target_leakage_rate: 0.005
  residual_leakage_ceiling: 0.005
  over_redaction_ceiling: 0.02
entries:
  directid::small::pytorch:
    family: DirectID
    tier: small
    format: pytorch
    target_leakage_rate: 0.0
    residual_leakage_ceiling: 0.005
    over_redaction_ceiling: 0.02
    steward_owner: "model-stewardship-council"
```

Each entry may omit any of the three ceilings; omitted values fall back first
to the config ``defaults`` block and then to the module-level defaults, so the
steward only records the numbers that deviate from the sane baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

TARGET_LEAKAGE_SCHEMA_VERSION = 1
TARGET_LEAKAGE_PATH = (
    Path(__file__).resolve().parents[2] / "gates" / "target_leakage.yaml"
)

DEFAULT_TARGET_LEAKAGE_RATE = 0.005
DEFAULT_RESIDUAL_LEAKAGE_CEILING = 0.005
DEFAULT_OVER_REDACTION_CEILING = 0.02

_CEILING_FIELDS = (
    "target_leakage_rate",
    "residual_leakage_ceiling",
    "over_redaction_ceiling",
)


class TargetLeakageConfigError(ValueError):
    """Raised when the target-leakage config is malformed."""


class _StrictLeakageLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    A duplicate key would let a later, laxer floor silently override an
    earlier stricter one (e.g. a signed ``0.0`` downgraded to ``0.9``), so the
    loader treats it as a malformed config rather than taking the last value.
    """


def _construct_mapping_no_duplicates(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise TargetLeakageConfigError(
                f"target-leakage config has a duplicate key: {key!r}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLeakageLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_duplicates,
)


class TargetLeakageMiss(LookupError):
    """Raised when a requested ``(family, tier, format)`` is unconfigured."""


@dataclass(frozen=True)
class LeakageTargets:
    """Signed per-family leakage floors for one ``(family, tier, format)``."""

    family: str
    tier: str | None
    format: str
    target_leakage_rate: float
    residual_leakage_ceiling: float
    over_redaction_ceiling: float
    steward_owner: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serialisable mapping of the leakage floors."""

        return {
            "family": self.family,
            "tier": self.tier,
            "format": self.format,
            "target_leakage_rate": float(self.target_leakage_rate),
            "residual_leakage_ceiling": float(self.residual_leakage_ceiling),
            "over_redaction_ceiling": float(self.over_redaction_ceiling),
            "steward_owner": self.steward_owner,
        }


def target_leakage_key(family: str, tier: str | None, format_name: str) -> str:
    """Return the stable config key for ``(family, tier, format)``."""

    return "::".join(
        (
            _normalise_key_part(family),
            _normalise_key_part(tier or "none"),
            _normalise_key_part(format_name),
        )
    )


def load_target_leakage_config(
    path: str | Path = TARGET_LEAKAGE_PATH,
) -> dict[str, Any]:
    """Load and validate the committed target-leakage config document."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            payload = yaml.load(handle, Loader=_StrictLeakageLoader)
        except TargetLeakageConfigError:
            raise
        except yaml.YAMLError as exc:
            raise TargetLeakageConfigError(
                f"target-leakage config is not valid YAML: {exc}"
            ) from exc
    if not isinstance(payload, Mapping):
        raise TargetLeakageConfigError("target-leakage config must be a mapping")
    validate_target_leakage_config(payload)
    return dict(payload)


def get_targets(
    family: str,
    tier: str | None,
    format_name: str,
    *,
    path: str | Path = TARGET_LEAKAGE_PATH,
    config: Mapping[str, Any] | None = None,
) -> LeakageTargets:
    """Return the signed leakage floors for ``(family, tier, format)``.

    Omitted per-entry ceilings inherit from the config ``defaults`` block and
    then the module-level defaults. A ``(family, tier, format)`` with no entry
    raises :class:`TargetLeakageMiss` naming the unconfigured key.
    """

    payload = dict(config) if config is not None else load_target_leakage_config(path)
    validate_target_leakage_config(payload)

    key = target_leakage_key(family, tier, format_name)
    entries = payload.get("entries") or {}
    entry = entries.get(key)
    if entry is None:
        raise TargetLeakageMiss(
            f"No Model-Steward target-leakage config for key: {key}. "
            "Add a signed entry to gates/target_leakage.yaml."
        )

    defaults = _resolve_defaults(payload)
    steward_owner = str(
        entry.get("steward_owner", payload.get("steward_owner", "unassigned"))
    )
    return LeakageTargets(
        family=str(entry["family"]),
        tier=None if entry.get("tier") is None else str(entry["tier"]),
        format=str(entry["format"]),
        target_leakage_rate=_pick_ceiling(entry, defaults, "target_leakage_rate"),
        residual_leakage_ceiling=_pick_ceiling(
            entry, defaults, "residual_leakage_ceiling"
        ),
        over_redaction_ceiling=_pick_ceiling(entry, defaults, "over_redaction_ceiling"),
        steward_owner=steward_owner,
    )


def validate_target_leakage_config(config: Mapping[str, Any]) -> None:
    """Validate the documented target-leakage config schema."""

    if not isinstance(config, Mapping):
        raise TargetLeakageConfigError("target-leakage config must be a mapping")
    if config.get("schema_version") != TARGET_LEAKAGE_SCHEMA_VERSION:
        raise TargetLeakageConfigError(
            f"target-leakage schema_version must be {TARGET_LEAKAGE_SCHEMA_VERSION}"
        )
    steward_owner = config.get("steward_owner")
    if not isinstance(steward_owner, str) or not steward_owner.strip():
        raise TargetLeakageConfigError(
            "target-leakage config requires a non-empty steward_owner"
        )

    defaults = config.get("defaults")
    if defaults is not None:
        if not isinstance(defaults, Mapping):
            raise TargetLeakageConfigError("target-leakage defaults must be a mapping")
        for field_name in _CEILING_FIELDS:
            if field_name in defaults:
                _validate_rate(defaults[field_name], f"defaults.{field_name}")

    entries = config.get("entries")
    if not isinstance(entries, Mapping):
        raise TargetLeakageConfigError("target-leakage entries must be a mapping")
    for key, entry in entries.items():
        _validate_entry(str(key), entry)


def _validate_entry(key: str, entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise TargetLeakageConfigError(f"entry {key!r} must be a mapping")

    for required in ("family", "format"):
        if not entry.get(required) or not isinstance(entry[required], str):
            raise TargetLeakageConfigError(
                f"entry {key!r} requires a non-empty {required}"
            )
    if entry.get("tier") is not None and not isinstance(entry["tier"], str):
        raise TargetLeakageConfigError(f"entry {key!r} tier must be a string or null")

    expected_key = target_leakage_key(
        entry["family"], entry.get("tier"), entry["format"]
    )
    if expected_key != key:
        raise TargetLeakageConfigError(
            f"entry {key!r} does not match its (family, tier, format) dimensions "
            f"(expected key {expected_key!r})"
        )

    for field_name in _CEILING_FIELDS:
        if field_name in entry:
            _validate_rate(entry[field_name], f"{key}.{field_name}")

    if "steward_owner" in entry and (
        not isinstance(entry["steward_owner"], str)
        or not entry["steward_owner"].strip()
    ):
        raise TargetLeakageConfigError(
            f"entry {key!r} steward_owner must be a non-empty string"
        )


def _validate_rate(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetLeakageConfigError(f"{field_name} must be a number in 0..1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise TargetLeakageConfigError(f"{field_name} must be between 0.0 and 1.0")
    return result


def _resolve_defaults(config: Mapping[str, Any]) -> dict[str, float]:
    module_defaults = {
        "target_leakage_rate": DEFAULT_TARGET_LEAKAGE_RATE,
        "residual_leakage_ceiling": DEFAULT_RESIDUAL_LEAKAGE_CEILING,
        "over_redaction_ceiling": DEFAULT_OVER_REDACTION_CEILING,
    }
    configured = config.get("defaults")
    if isinstance(configured, Mapping):
        for field_name in _CEILING_FIELDS:
            if field_name in configured:
                module_defaults[field_name] = float(configured[field_name])
    return module_defaults


def _pick_ceiling(
    entry: Mapping[str, Any],
    defaults: Mapping[str, float],
    field_name: str,
) -> float:
    if field_name in entry:
        return float(entry[field_name])
    return float(defaults[field_name])


def _normalise_key_part(value: str) -> str:
    return str(value).strip().lower().replace("_", "-") or "none"


__all__ = [
    "DEFAULT_OVER_REDACTION_CEILING",
    "DEFAULT_RESIDUAL_LEAKAGE_CEILING",
    "DEFAULT_TARGET_LEAKAGE_RATE",
    "LeakageTargets",
    "TARGET_LEAKAGE_PATH",
    "TARGET_LEAKAGE_SCHEMA_VERSION",
    "TargetLeakageConfigError",
    "TargetLeakageMiss",
    "get_targets",
    "load_target_leakage_config",
    "target_leakage_key",
    "validate_target_leakage_config",
]
