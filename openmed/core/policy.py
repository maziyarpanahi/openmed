"""Versioned de-identification policy profile loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from .arbitration import MODE_BALANCED, MODE_HIGH_RECALL_UNION
from .detector_plugins import DetectorCapability, default_detector_capabilities
from .labels import CANONICAL_LABELS, POLICY_LABELS, normalize_label, policy_label_for
from .schemas.span import ACTION_VALUES

CURRENT_POLICY_SCHEMA_VERSION = 1


class PolicyName(str, Enum):
    """Canonical policy profile names accepted by the de-identification runtime."""

    HIPAA_SAFE_HARBOR = "hipaa_safe_harbor"
    HIPAA_EXPERT_REVIEW_ASSIST = "hipaa_expert_review_assist"
    GDPR_PSEUDONYMIZATION = "gdpr_pseudonymization"
    GDPR_ART9_HEALTH = "gdpr_art9_health"
    RESEARCH_LIMITED_DATASET = "research_limited_dataset"
    STRICT_NO_LEAK = "strict_no_leak"
    CLINICAL_MINIMAL_REDACTION = "clinical_minimal_redaction"
    CANADA_PIPEDA = "canada_pipeda"
    UK_ICO_ANONYMISATION = "uk_ico_anonymisation"
    AUSTRALIA_PRIVACY_ACT = "australia_privacy_act"
    CHINA_PIPL = "china_pipl"
    INDIA_DPDP_ACT = "india_dpdp_act"


CANONICAL_POLICY_NAMES = tuple(policy.value for policy in PolicyName)
POLICY_ALIASES: Mapping[str, str] = {
    "au_privacy": PolicyName.AUSTRALIA_PRIVACY_ACT.value,
    "gdpr": PolicyName.GDPR_PSEUDONYMIZATION.value,
    "gdpr_health": PolicyName.GDPR_ART9_HEALTH.value,
    "pipeda": PolicyName.CANADA_PIPEDA.value,
    "uk_ico": PolicyName.UK_ICO_ANONYMISATION.value,
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

    def derive(
        self,
        *,
        name: str | None = None,
        posture: str | None = None,
        default_action: str | None = None,
        default_action_bias: str | None = None,
        actions: Mapping[str, str] | None = None,
        policy_label_actions: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PolicyProfile":
        """Return a validated copy with selected profile fields replaced."""

        resolved_name = self.name if name is None else _non_empty_str(name, "name")
        resolved_posture = (
            self.posture if posture is None else _non_empty_str(posture, "posture")
        )
        resolved_default_action = (
            self.default_action
            if default_action is None
            else _action(default_action, "default_action")
        )
        resolved_default_action_bias = (
            self.default_action_bias
            if default_action_bias is None
            else _non_empty_str(default_action_bias, "default_action_bias")
        )
        resolved_actions = (
            dict(self.actions) if actions is None else _canonical_actions(actions)
        )
        resolved_policy_label_actions = (
            dict(self.policy_label_actions)
            if policy_label_actions is None
            else _policy_label_actions(policy_label_actions)
        )
        resolved_metadata = dict(self.metadata)
        if metadata is not None:
            resolved_metadata.update(dict(metadata))

        return replace(
            self,
            name=resolved_name,
            posture=resolved_posture,
            default_action=resolved_default_action,
            default_action_bias=resolved_default_action_bias,
            actions=resolved_actions,
            policy_label_actions=resolved_policy_label_actions,
            metadata=resolved_metadata,
        )


@dataclass(frozen=True)
class PolicyRequirement:
    """A canonical label that a policy requires the runtime to protect."""

    label: str
    policy_label: str
    minimum_action: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible requirement record."""

        return {
            "label": self.label,
            "policy_label": self.policy_label,
            "minimum_action": self.minimum_action,
        }


@dataclass(frozen=True)
class PolicyPlanEntry:
    """Executable detector/action assignment for one required label."""

    label: str
    policy_label: str
    detector_id: str
    detector_stage: str
    action: str
    required_action: str

    def __post_init__(self) -> None:
        label = _canonical_label(self.label, "label")
        policy_label = str(self.policy_label)
        if policy_label != policy_label_for(label):
            raise ValueError(
                f"policy_label for {label} must be {policy_label_for(label)!r}"
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "policy_label", policy_label)
        object.__setattr__(self, "action", _action(self.action, "action"))
        object.__setattr__(
            self,
            "required_action",
            _action(self.required_action, "required_action"),
        )
        if not str(self.detector_id).strip():
            raise ValueError("detector_id must be non-empty")
        if not str(self.detector_stage).strip():
            raise ValueError("detector_stage must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible plan entry."""

        return {
            "label": self.label,
            "policy_label": self.policy_label,
            "detector_id": self.detector_id,
            "detector_stage": self.detector_stage,
            "action": self.action,
            "required_action": self.required_action,
        }


@dataclass(frozen=True)
class PolicyActionPlan:
    """Compiled policy plan with no implicit label fall-through."""

    policy_name: str
    schema_version: int
    default_action: str
    allow_default_fallthrough: bool
    entries: tuple[PolicyPlanEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_name", _non_empty_str(self.policy_name, "policy_name")
        )
        object.__setattr__(
            self, "default_action", _action(self.default_action, "default_action")
        )
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(self.entries, key=lambda entry: entry.label)),
        )

    def entry_for(self, label: str) -> PolicyPlanEntry | None:
        """Return the plan entry for ``label`` if present."""

        canonical = normalize_label(label)
        for entry in self.entries:
            if entry.label == canonical:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible plan."""

        return {
            "policy_name": self.policy_name,
            "schema_version": self.schema_version,
            "default_action": self.default_action,
            "allow_default_fallthrough": self.allow_default_fallthrough,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def fingerprint(self) -> str:
        """Return the stable content hash for this plan."""

        return _stable_json_hash(self.to_dict())


@dataclass(frozen=True)
class PolicyCoverageWitness:
    """Machine-checkable witness for one required-label coverage claim."""

    label: str
    policy_label: str
    detector_id: str
    detector_stage: str
    action: str
    required_action: str
    detector_covers_label: bool
    action_meets_requirement: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible witness."""

        return {
            "label": self.label,
            "policy_label": self.policy_label,
            "detector_id": self.detector_id,
            "detector_stage": self.detector_stage,
            "action": self.action,
            "required_action": self.required_action,
            "detector_covers_label": self.detector_covers_label,
            "action_meets_requirement": self.action_meets_requirement,
        }


@dataclass(frozen=True)
class PolicyCoverageViolation:
    """Stable diagnostic emitted by policy compilation or verification."""

    code: str
    label: str
    message: str
    required_action: str | None = None
    observed_action: str | None = None
    detector_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic."""

        return {
            "code": self.code,
            "label": self.label,
            "message": self.message,
            "required_action": self.required_action,
            "observed_action": self.observed_action,
            "detector_id": self.detector_id,
        }


@dataclass(frozen=True)
class PolicyCoverageProof:
    """Independent coverage proof for a compiled policy plan."""

    policy_name: str
    plan_fingerprint: str
    required_labels: tuple[str, ...]
    witnesses: tuple[PolicyCoverageWitness, ...]
    uncovered_labels: tuple[str, ...] = ()
    fallthrough_labels: tuple[str, ...] = ()
    weak_actions: tuple[PolicyCoverageViolation, ...] = ()
    invalid_detectors: tuple[PolicyCoverageViolation, ...] = ()

    @property
    def covered_label_count(self) -> int:
        """Return the number of required labels with valid witnesses."""

        covered = {
            witness.label
            for witness in self.witnesses
            if witness.detector_covers_label and witness.action_meets_requirement
        }
        return len(covered)

    @property
    def coverage_percent(self) -> float:
        """Return required-label coverage as a percentage."""

        if not self.required_labels:
            return 100.0
        return 100.0 * self.covered_label_count / len(self.required_labels)

    @property
    def verified(self) -> bool:
        """Return whether the proof has no coverage or strength violations."""

        return not (
            self.uncovered_labels
            or self.fallthrough_labels
            or self.weak_actions
            or self.invalid_detectors
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible proof object."""

        return {
            "policy_name": self.policy_name,
            "plan_fingerprint": self.plan_fingerprint,
            "required_labels": list(self.required_labels),
            "covered_label_count": self.covered_label_count,
            "coverage_percent": self.coverage_percent,
            "verified": self.verified,
            "witnesses": [witness.to_dict() for witness in self.witnesses],
            "uncovered_labels": list(self.uncovered_labels),
            "fallthrough_labels": list(self.fallthrough_labels),
            "weak_actions": [violation.to_dict() for violation in self.weak_actions],
            "invalid_detectors": [
                violation.to_dict() for violation in self.invalid_detectors
            ],
        }


@dataclass(frozen=True)
class CompiledPolicy:
    """Compiler output containing the executable plan and verified proof."""

    policy_name: str
    plan: PolicyActionPlan
    proof: PolicyCoverageProof

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible compiler artifact."""

        return {
            "policy_name": self.policy_name,
            "plan": self.plan.to_dict(),
            "proof": self.proof.to_dict(),
        }


class PolicyCoverageError(ValueError):
    """Base class for policy coverage proof failures."""

    def __init__(
        self,
        message: str,
        *,
        uncovered_labels: Sequence[str] = (),
        fallthrough_labels: Sequence[str] = (),
        weak_actions: Sequence[PolicyCoverageViolation] = (),
        invalid_detectors: Sequence[PolicyCoverageViolation] = (),
        proof: PolicyCoverageProof | None = None,
    ) -> None:
        super().__init__(message)
        self.uncovered_labels = tuple(uncovered_labels)
        self.fallthrough_labels = tuple(fallthrough_labels)
        self.weak_actions = tuple(weak_actions)
        self.invalid_detectors = tuple(invalid_detectors)
        self.proof = proof


class PolicyCompilationError(PolicyCoverageError):
    """Raised when a policy cannot be lowered into a covered plan."""


class PolicyVerificationError(PolicyCoverageError):
    """Raised when an independently checked plan/proof is invalid."""


def canonical_policy_name(name: str | PolicyName) -> str:
    """Resolve aliases and validate a policy profile name."""

    if isinstance(name, PolicyName):
        return name.value
    if not isinstance(name, str):
        raise TypeError(
            "policy must be a profile name string or a PolicyProfile object, "
            f"got {type(name).__name__}"
        )
    normalized = name.strip().lower().replace("-", "_")
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


def policy_requirements(
    profile: str | PolicyName | PolicyProfile,
) -> tuple[PolicyRequirement, ...]:
    """Return labels whose configured policy action requires protection."""

    loaded = load_policy(profile)
    requirements: list[PolicyRequirement] = []
    for label in sorted(CANONICAL_LABELS):
        minimum_action = _minimum_required_action(loaded, label)
        if _action_strength(minimum_action) <= _action_strength("keep"):
            continue
        requirements.append(
            PolicyRequirement(
                label=label,
                policy_label=policy_label_for(label),
                minimum_action=minimum_action,
            )
        )
    return tuple(requirements)


def compile_policy(
    profile: str | PolicyName | PolicyProfile,
    *,
    detector_capabilities: Sequence[DetectorCapability] | None = None,
    allow_default_fallthrough: bool = False,
) -> CompiledPolicy:
    """Compile a policy into a detector/action plan and verified proof."""

    loaded = load_policy(profile)
    capabilities = (
        tuple(detector_capabilities)
        if detector_capabilities is not None
        else default_detector_capabilities()
    )
    capabilities_by_label = _capabilities_by_label(capabilities)
    requirements = policy_requirements(loaded)

    entries: list[PolicyPlanEntry] = []
    uncovered_labels: list[str] = []
    for requirement in requirements:
        candidates = capabilities_by_label.get(requirement.label, ())
        if not candidates:
            uncovered_labels.append(requirement.label)
            continue
        detector = candidates[0]
        entries.append(
            PolicyPlanEntry(
                label=requirement.label,
                policy_label=requirement.policy_label,
                detector_id=detector.detector_id,
                detector_stage=detector.stage,
                action=loaded.action_for(requirement.label),
                required_action=requirement.minimum_action,
            )
        )

    if uncovered_labels:
        raise PolicyCompilationError(
            _coverage_error_message(
                "policy compilation failed",
                loaded.name,
                uncovered_labels=tuple(uncovered_labels),
            ),
            uncovered_labels=tuple(sorted(uncovered_labels)),
        )

    plan = PolicyActionPlan(
        policy_name=loaded.name,
        schema_version=loaded.schema_version,
        default_action=loaded.default_action,
        allow_default_fallthrough=allow_default_fallthrough,
        entries=tuple(entries),
    )
    proof = verify_policy_plan(
        loaded,
        plan,
        detector_capabilities=capabilities,
        allow_default_fallthrough=allow_default_fallthrough,
    )
    return CompiledPolicy(policy_name=loaded.name, plan=plan, proof=proof)


def verify_policy_plan(
    profile: str | PolicyName | PolicyProfile,
    plan: PolicyActionPlan,
    *,
    detector_capabilities: Sequence[DetectorCapability] | None = None,
    allow_default_fallthrough: bool | None = None,
    raise_on_error: bool = True,
) -> PolicyCoverageProof:
    """Independently verify detector coverage and action strength for a plan."""

    loaded = load_policy(profile)
    capabilities = (
        tuple(detector_capabilities)
        if detector_capabilities is not None
        else default_detector_capabilities()
    )
    capabilities_by_id = {
        capability.detector_id: capability for capability in capabilities
    }
    capabilities_by_label = _capabilities_by_label(capabilities)
    requirements = policy_requirements(loaded)
    required_labels = tuple(requirement.label for requirement in requirements)
    entries_by_label = {entry.label: entry for entry in plan.entries}
    default_fallthrough_allowed = (
        plan.allow_default_fallthrough
        if allow_default_fallthrough is None
        else allow_default_fallthrough
    )

    witnesses: list[PolicyCoverageWitness] = []
    uncovered_labels: list[str] = []
    fallthrough_labels: list[str] = []
    weak_actions: list[PolicyCoverageViolation] = []
    invalid_detectors: list[PolicyCoverageViolation] = []

    for requirement in requirements:
        entry = entries_by_label.get(requirement.label)
        if entry is None:
            if not capabilities_by_label.get(requirement.label):
                uncovered_labels.append(requirement.label)
            elif not default_fallthrough_allowed:
                fallthrough_labels.append(requirement.label)
            continue

        capability = capabilities_by_id.get(entry.detector_id)
        detector_covers_label = bool(
            capability is not None and requirement.label in capability.covered_labels
        )
        action_meets_requirement = _action_meets_floor(
            entry.action,
            requirement.minimum_action,
        )
        witnesses.append(
            PolicyCoverageWitness(
                label=requirement.label,
                policy_label=requirement.policy_label,
                detector_id=entry.detector_id,
                detector_stage=entry.detector_stage,
                action=entry.action,
                required_action=requirement.minimum_action,
                detector_covers_label=detector_covers_label,
                action_meets_requirement=action_meets_requirement,
            )
        )

        if not detector_covers_label:
            invalid_detectors.append(
                PolicyCoverageViolation(
                    code="POLICY_DETECTOR_DOES_NOT_COVER_LABEL",
                    label=requirement.label,
                    detector_id=entry.detector_id,
                    message=(
                        f"detector {entry.detector_id!r} does not cover "
                        f"required label {requirement.label}"
                    ),
                )
            )
        if not action_meets_requirement:
            weak_actions.append(
                PolicyCoverageViolation(
                    code="POLICY_ACTION_TOO_WEAK",
                    label=requirement.label,
                    required_action=requirement.minimum_action,
                    observed_action=entry.action,
                    detector_id=entry.detector_id,
                    message=(
                        f"action {entry.action!r} is weaker than required "
                        f"{requirement.minimum_action!r} for {requirement.label}"
                    ),
                )
            )

    proof = PolicyCoverageProof(
        policy_name=loaded.name,
        plan_fingerprint=plan.fingerprint,
        required_labels=required_labels,
        witnesses=tuple(witnesses),
        uncovered_labels=tuple(sorted(uncovered_labels)),
        fallthrough_labels=tuple(sorted(fallthrough_labels)),
        weak_actions=tuple(weak_actions),
        invalid_detectors=tuple(invalid_detectors),
    )
    if raise_on_error and not proof.verified:
        raise PolicyVerificationError(
            _coverage_error_message(
                "policy verification failed",
                loaded.name,
                uncovered_labels=proof.uncovered_labels,
                fallthrough_labels=proof.fallthrough_labels,
                weak_labels=tuple(violation.label for violation in proof.weak_actions),
                invalid_detector_labels=tuple(
                    violation.label for violation in proof.invalid_detectors
                ),
            ),
            uncovered_labels=proof.uncovered_labels,
            fallthrough_labels=proof.fallthrough_labels,
            weak_actions=proof.weak_actions,
            invalid_detectors=proof.invalid_detectors,
            proof=proof,
        )
    return proof


def lint_policy(
    name: str | PolicyName | PolicyProfile | Mapping[str, Any],
) -> tuple[str, ...]:
    """Return schema validation errors for a policy profile."""

    if isinstance(name, PolicyProfile):
        return ()
    try:
        if isinstance(name, Mapping):
            _profile_from_mapping(name, source="<mapping>")
        else:
            load_policy(name)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return (str(exc),)
    return ()


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


def _canonical_label(value: str, field_name: str) -> str:
    label = str(value)
    canonical = normalize_label(label)
    if canonical != label:
        raise ValueError(f"{field_name} must use canonical labels, got {label!r}")
    return canonical


def _capabilities_by_label(
    capabilities: Sequence[DetectorCapability],
) -> dict[str, tuple[DetectorCapability, ...]]:
    by_label: dict[str, list[DetectorCapability]] = {
        label: [] for label in sorted(CANONICAL_LABELS)
    }
    for capability in capabilities:
        for label in capability.covered_labels:
            by_label[label].append(capability)
    return {
        label: tuple(
            sorted(
                label_capabilities,
                key=lambda capability: (
                    _detector_stage_order(capability.stage),
                    capability.detector_id,
                ),
            )
        )
        for label, label_capabilities in by_label.items()
    }


def _detector_stage_order(stage: str) -> int:
    return {
        "deterministic": 0,
        "fast_pii": 1,
        "clinical_phi": 2,
    }.get(stage, 99)


def _minimum_required_action(profile: PolicyProfile, label: str) -> str:
    canonical_label = normalize_label(label)
    explicit_action = profile.action_for(canonical_label)
    category_floor = profile.policy_label_actions.get(
        policy_label_for(canonical_label),
        profile.default_action,
    )
    return max((explicit_action, category_floor), key=_action_strength)


def _action_strength(action: str) -> int:
    from .redaction_strength import action_strength

    return action_strength(action)


def _action_meets_floor(action: str, floor: str) -> bool:
    from .redaction_strength import action_meets_floor

    return action_meets_floor(action, floor)


def _coverage_error_message(
    prefix: str,
    policy_name: str,
    *,
    uncovered_labels: Sequence[str] = (),
    fallthrough_labels: Sequence[str] = (),
    weak_labels: Sequence[str] = (),
    invalid_detector_labels: Sequence[str] = (),
) -> str:
    parts = [f"{prefix} for policy {policy_name!r}"]
    if uncovered_labels:
        parts.append(f"uncovered labels: {', '.join(sorted(uncovered_labels))}")
    if fallthrough_labels:
        parts.append(
            "labels would fall through to the default action: "
            + ", ".join(sorted(fallthrough_labels))
        )
    if weak_labels:
        parts.append(f"weak actions: {', '.join(sorted(weak_labels))}")
    if invalid_detector_labels:
        parts.append(
            "invalid detector witnesses: " + ", ".join(sorted(invalid_detector_labels))
        )
    return "; ".join(parts)


def _stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


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
    "CompiledPolicy",
    "PolicyActionPlan",
    "PolicyCompilationError",
    "PolicyCoverageError",
    "PolicyCoverageProof",
    "PolicyCoverageViolation",
    "PolicyCoverageWitness",
    "PolicyName",
    "PolicyPlanEntry",
    "PolicyProfile",
    "PolicyRequirement",
    "PolicyVerificationError",
    "canonical_policy_name",
    "compile_policy",
    "lint_policy",
    "list_policies",
    "load_policy",
    "policy_requirements",
    "verify_policy_plan",
]
