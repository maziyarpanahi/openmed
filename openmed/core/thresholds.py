"""Versioned per-label/language/policy threshold matrix."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from .labels import normalize_label
from .schemas.span import ACTION_VALUES

CURRENT_SCHEMA_VERSION = 1
DEFAULT_RESOURCE = "thresholds.json"
DEFAULT_POLICY_PROFILE = "balanced"
STRICT_NO_LEAK_PROFILE = "strict_no_leak"
WILDCARD_LANGUAGE = "*"


@dataclass(frozen=True)
class Threshold:
    keep_floor: float
    escalate_below: float
    action: str
    canonical_label: str
    language: str
    policy_profile: str
    source: str
    schema_version: int

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "keep_floor": self.keep_floor,
            "escalate_below": self.escalate_below,
            "action": self.action,
            "canonical_label": self.canonical_label,
            "language": self.language,
            "policy_profile": self.policy_profile,
            "source": self.source,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class RecallGuardResult:
    block: bool
    policy_profile: str
    recall_floor: float
    violations: tuple[dict[str, float | str], ...]

    @property
    def reason(self) -> str:
        if not self.block:
            return "ok"
        labels = ", ".join(str(item["canonical_label"]) for item in self.violations)
        return f"protected recall below floor for {labels}"


def load_thresholds(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the versioned thresholds matrix."""

    if path is None:
        resource = resources.files("openmed.core").joinpath(DEFAULT_RESOURCE)
        with resource.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in threshold file {path}: {exc}") from exc
    validate_threshold_matrix(payload)
    return payload


def validate_threshold_matrix(matrix: Mapping[str, Any]) -> None:
    schema_version = matrix.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("thresholds matrix requires positive integer schema_version")

    profiles = matrix.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("thresholds matrix requires profiles")

    for profile_name, profile in profiles.items():
        if not isinstance(profile, Mapping):
            raise ValueError(f"profile {profile_name!r} must be an object")
        _validate_recall_floor(profile.get("recall_floor"), profile_name)
        _validate_entry(profile.get("default"), f"{profile_name}.default")
        labels = profile.get("labels") or {}
        if not isinstance(labels, Mapping):
            raise ValueError(f"profile {profile_name!r} labels must be an object")
        for label, languages in labels.items():
            canonical_label = normalize_label(str(label))
            if not isinstance(languages, Mapping):
                raise ValueError(f"{profile_name}.{canonical_label} must be an object")
            for language, entry in languages.items():
                _validate_entry(entry, f"{profile_name}.{canonical_label}.{language}")


def lookup_threshold(
    canonical_label: str,
    language: str,
    policy_profile: str = DEFAULT_POLICY_PROFILE,
    *,
    matrix: Mapping[str, Any] | None = None,
) -> dict[str, float | str | int]:
    """Lookup a threshold entry.

    Fallback order is deterministic:
    exact (label, language, profile) -> wildcard language for the label ->
    profile default.
    """

    payload = matrix if matrix is not None else load_thresholds()
    validate_threshold_matrix(payload)
    profile_name = _resolve_profile_name(payload, policy_profile)
    profile = payload["profiles"][profile_name]
    label = normalize_label(canonical_label)
    lang = (language or WILDCARD_LANGUAGE).lower()
    labels = profile.get("labels") or {}
    language_entries = labels.get(label) or {}

    if lang in language_entries:
        entry = language_entries[lang]
        source = "exact"
        source_language = lang
    elif WILDCARD_LANGUAGE in language_entries:
        entry = language_entries[WILDCARD_LANGUAGE]
        source = "wildcard_language"
        source_language = WILDCARD_LANGUAGE
    else:
        entry = profile["default"]
        source = "profile_default"
        source_language = WILDCARD_LANGUAGE

    threshold = Threshold(
        keep_floor=float(entry["keep_floor"]),
        escalate_below=float(entry["escalate_below"]),
        action=str(entry["action"]),
        canonical_label=label,
        language=source_language,
        policy_profile=profile_name,
        source=source,
        schema_version=int(payload["schema_version"]),
    )
    return threshold.to_dict()


def profile_recall_floor(
    policy_profile: str = DEFAULT_POLICY_PROFILE,
    *,
    matrix: Mapping[str, Any] | None = None,
) -> float:
    payload = matrix if matrix is not None else load_thresholds()
    validate_threshold_matrix(payload)
    profile_name = _resolve_profile_name(payload, policy_profile)
    return float(payload["profiles"][profile_name]["recall_floor"])


def label_keep_floors(
    labels: Sequence[str],
    language: str,
    policy_profile: str,
    *,
    matrix: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    return {
        normalize_label(label): float(
            lookup_threshold(
                label,
                language,
                policy_profile,
                matrix=matrix,
            )["keep_floor"]
        )
        for label in labels
    }


def fit_thresholds(
    samples: Sequence[Mapping[str, Any]],
    *,
    policy_profile: str = DEFAULT_POLICY_PROFILE,
    base_matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a matrix copy with threshold entries fit from eval-style samples."""

    matrix = copy.deepcopy(dict(base_matrix or load_thresholds()))
    validate_threshold_matrix(matrix)
    profile_name = _resolve_profile_name(matrix, policy_profile)
    profile = matrix["profiles"][profile_name]
    labels = profile.setdefault("labels", {})

    grouped: dict[tuple[str, str], list[float]] = {}
    for sample in samples:
        if bool(sample.get("target", sample.get("is_true", True))) is False:
            continue
        label = normalize_label(str(sample["canonical_label"]))
        language = str(sample.get("language") or WILDCARD_LANGUAGE).lower()
        grouped.setdefault((label, language), []).append(float(sample["score"]))

    for (label, language), scores in grouped.items():
        if not scores:
            continue
        keep_floor = max(0.0, min(scores))
        entry = {
            "keep_floor": keep_floor,
            "escalate_below": min(1.0, keep_floor + 0.05),
            "action": lookup_threshold(
                label,
                language,
                profile_name,
                matrix=matrix,
            )["action"],
        }
        labels.setdefault(label, {})[language] = entry

    validate_threshold_matrix(matrix)
    return matrix


def update_thresholds(
    base_matrix: Mapping[str, Any],
    updates: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    bump_schema_version: bool = False,
) -> dict[str, Any]:
    """Apply explicit threshold updates and optionally bump schema_version."""

    matrix = copy.deepcopy(dict(base_matrix))
    validate_threshold_matrix(matrix)
    if bump_schema_version:
        matrix["schema_version"] = int(matrix["schema_version"]) + 1

    profiles = matrix["profiles"]
    for (label, language, profile_name), entry in updates.items():
        resolved_profile = _resolve_profile_name(matrix, profile_name)
        canonical_label = normalize_label(label)
        language_key = (language or WILDCARD_LANGUAGE).lower()
        profiles[resolved_profile].setdefault("labels", {}).setdefault(
            canonical_label,
            {},
        )[language_key] = {
            "keep_floor": float(entry["keep_floor"]),
            "escalate_below": float(entry["escalate_below"]),
            "action": str(entry["action"]),
        }

    validate_threshold_matrix(matrix)
    return matrix


def recall_floor_guard(
    old_recall: Mapping[str, float],
    new_recall: Mapping[str, float],
    *,
    policy_profile: str = STRICT_NO_LEAK_PROFILE,
    protected_labels: Sequence[str] | None = None,
    matrix: Mapping[str, Any] | None = None,
) -> RecallGuardResult:
    """Signal whether a threshold change would violate a profile recall floor."""

    floor = profile_recall_floor(policy_profile, matrix=matrix)
    labels = tuple(protected_labels or sorted(set(old_recall) | set(new_recall)))
    violations: list[dict[str, float | str]] = []

    for label in labels:
        canonical_label = normalize_label(label)
        before = float(old_recall.get(label, old_recall.get(canonical_label, 0.0)))
        after = float(new_recall.get(label, new_recall.get(canonical_label, 0.0)))
        if before >= floor and after < floor:
            violations.append(
                {
                    "canonical_label": canonical_label,
                    "old_recall": before,
                    "new_recall": after,
                    "recall_floor": floor,
                }
            )

    return RecallGuardResult(
        block=bool(violations),
        policy_profile=policy_profile,
        recall_floor=floor,
        violations=tuple(violations),
    )


def _resolve_profile_name(matrix: Mapping[str, Any], policy_profile: str) -> str:
    profiles = matrix["profiles"]
    if policy_profile in profiles:
        return policy_profile
    default_profile = str(
        matrix.get("default_policy_profile") or DEFAULT_POLICY_PROFILE
    )
    if default_profile in profiles:
        return default_profile
    raise KeyError(f"unknown policy profile {policy_profile!r}")


def _validate_recall_floor(value: Any, profile_name: str) -> None:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"profile {profile_name!r} recall_floor must be 0..1")


def _validate_entry(entry: Any, path: str) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{path} must be an object")
    for key in ("keep_floor", "escalate_below", "action"):
        if key not in entry:
            raise ValueError(f"{path} missing {key}")
    keep_floor = entry["keep_floor"]
    escalate_below = entry["escalate_below"]
    if not isinstance(keep_floor, (int, float)) or not 0.0 <= float(keep_floor) <= 1.0:
        raise ValueError(f"{path}.keep_floor must be 0..1")
    if (
        not isinstance(escalate_below, (int, float))
        or not 0.0 <= float(escalate_below) <= 1.0
    ):
        raise ValueError(f"{path}.escalate_below must be 0..1")
    if entry["action"] not in ACTION_VALUES:
        raise ValueError(f"{path}.action must be one of {ACTION_VALUES!r}")


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_POLICY_PROFILE",
    "RecallGuardResult",
    "STRICT_NO_LEAK_PROFILE",
    "Threshold",
    "fit_thresholds",
    "label_keep_floors",
    "load_thresholds",
    "lookup_threshold",
    "profile_recall_floor",
    "recall_floor_guard",
    "update_thresholds",
    "validate_threshold_matrix",
]
