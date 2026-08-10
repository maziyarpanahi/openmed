"""Deterministic, value-free simulation of privacy-policy changes.

The simulation answers a narrow release-review question: for the same
synthetic resource-class counts and gate outcomes, how do two local policy
versions differ in action, affected count, and blocking outcome?  It does not
run a model, inspect source text, consume a privacy budget, or write an
artifact.  Matrix serialization contains only validated categories, counts,
gate statuses, and stable fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast

from openmed.core.policy import PolicyProfile, load_policy
from openmed.core.redaction_strength import action_strength
from openmed.core.schemas.span import ACTION_VALUES

POLICY_SIMULATION_SCHEMA_VERSION: Final = 1
"""Schema version for serialized policy-simulation matrices."""

POLICY_SIMULATION_MATRIX_SCHEMA_VERSION: Final = POLICY_SIMULATION_SCHEMA_VERSION
"""Descriptive alias for :data:`POLICY_SIMULATION_SCHEMA_VERSION`."""

POLICY_SIMULATION_ARTIFACT: Final = "openmed.risk.policy_simulation"
"""Stable artifact label for a serialized simulation matrix."""

ACTION_CHANGE_VALUES: Final = ("unchanged", "stronger", "weaker")
COUNT_CHANGE_VALUES: Final = ("unchanged", "increased", "decreased")
BLOCKING_CHANGE_VALUES: Final = ("unchanged", "blocked", "unblocked")
GATE_OUTCOME_VALUES: Final = ("not_required", "pass", "fail", "missing")

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/+@-]{0,127}$")
_ALLOWED_POLICY_FIELDS = frozenset(
    {
        "version",
        "name",
        "id",
        "policy_version",
        "schema_version",
        "actions",
        "rules",
        "policy_label_actions",
        "default_action",
        "blocking_gates",
        "required_gates",
        "gates",
        "strict_no_leak",
        "safety_sweep_mandatory",
        "metadata",
        "posture",
        "threshold_profile",
        "default_action_bias",
        "arbitration_mode",
        "keep_mapping",
        "reversible_id",
        "forced_cascade_tiers",
    }
)
_ALLOWED_SCENARIO_FIELDS = frozenset(
    {
        "scenario_id",
        "id",
        "name",
        "resource_class",
        "resource",
        "class",
        "count",
        "gate_outcomes",
        "gate_results",
        "gates",
    }
)


class PolicySimulationError(ValueError):
    """Base error for malformed or unsafe simulation inputs."""


class PolicySimulationSchemaError(PolicySimulationError):
    """Raised when an input does not satisfy the closed simulation schema."""


def _normalise_identifier(value: Any, *, field_name: str) -> str:
    if type(value) is not str:
        raise PolicySimulationSchemaError(f"{field_name} must be a safe identifier")
    normalised = unicodedata.normalize("NFC", value.strip())
    if not _IDENTIFIER_RE.fullmatch(normalised):
        raise PolicySimulationSchemaError(f"{field_name} must be a safe identifier")
    return normalised


def _normalise_action(value: Any, *, field_name: str = "action") -> str:
    if type(value) is not str or value not in ACTION_VALUES:
        raise PolicySimulationSchemaError(f"{field_name} is not a supported action")
    return value


def _normalise_count(value: Any, *, field_name: str = "count") -> int:
    if type(value) is not int or value < 0:
        raise PolicySimulationSchemaError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _normalise_identifier_sequence(
    value: Any,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if type(value) is str:
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise PolicySimulationSchemaError(f"{field_name} must be a sequence")

    normalised = {
        _normalise_identifier(item, field_name=f"{field_name} entry") for item in values
    }
    return tuple(sorted(normalised))


def _normalise_gate_requirements(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if not all(type(item) is bool for item in value.values()):
            raise PolicySimulationSchemaError(
                f"{field_name} mapping values must be boolean"
            )
        value = [key for key, enabled in value.items() if enabled]
    return _normalise_identifier_sequence(value, field_name=field_name)


def _normalise_gate_outcomes(value: Any) -> Mapping[str, bool]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise PolicySimulationSchemaError("gate outcomes must be a mapping")
    outcomes: dict[str, bool] = {}
    for gate, outcome in value.items():
        safe_gate = _normalise_identifier(gate, field_name="gate name")
        if type(outcome) is not bool:
            raise PolicySimulationSchemaError("gate outcomes must be boolean")
        outcomes[safe_gate] = outcome
    return MappingProxyType(dict(sorted(outcomes.items())))


def _mapping_copy(
    value: Mapping[str, str],
    *,
    field_name: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise PolicySimulationSchemaError(f"{field_name} must be a mapping")
    normalised: dict[str, str] = {}
    for key, action in value.items():
        safe_key = _normalise_identifier(key, field_name=f"{field_name} key")
        normalised[safe_key] = _normalise_action(
            action,
            field_name=f"{field_name} action",
        )
    return MappingProxyType(dict(sorted(normalised.items())))


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PolicySimulationSchemaError(
            "simulation values must be JSON-compatible"
        ) from exc


def _digest(value: Any) -> str:
    return (
        "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    )


def _change_kind(before: int, after: int) -> str:
    if after > before:
        return "increased"
    if after < before:
        return "decreased"
    return "unchanged"


def _action_change(before: str, after: str) -> str:
    if before == after:
        return "unchanged"
    if action_strength(after) > action_strength(before):
        return "stronger"
    return "weaker"


def _blocking_change(before: bool, after: bool) -> str:
    if before == after:
        return "unchanged"
    return "blocked" if after else "unblocked"


@dataclass(frozen=True, slots=True, repr=False)
class PolicyVersion:
    """Immutable action and gate requirements for one policy version.

    ``actions`` is keyed by a resource class such as ``PERSON`` or
    ``DIRECT_IDENTIFIER``.  ``blocking_gates`` names the scenario gate
    outcomes that must be present and true for this version to proceed.  The
    optional ``policy_label_actions`` field supports the policy-class map used
    by :class:`openmed.core.policy.PolicyProfile`.
    """

    version: str
    actions: Mapping[str, str] = dataclass_field(default_factory=dict)
    default_action: str = "keep"
    blocking_gates: tuple[str, ...] = ()
    policy_label_actions: Mapping[str, str] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "version",
            _normalise_identifier(self.version, field_name="policy version"),
        )
        object.__setattr__(
            self,
            "actions",
            _mapping_copy(self.actions, field_name="actions"),
        )
        object.__setattr__(
            self,
            "policy_label_actions",
            _mapping_copy(
                self.policy_label_actions,
                field_name="policy label actions",
            ),
        )
        object.__setattr__(
            self,
            "default_action",
            _normalise_action(self.default_action, field_name="default action"),
        )
        object.__setattr__(
            self,
            "blocking_gates",
            _normalise_identifier_sequence(
                self.blocking_gates,
                field_name="blocking gates",
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PolicyVersion":
        """Build a policy version from a closed, JSON-like mapping."""

        if not isinstance(payload, Mapping):
            raise PolicySimulationSchemaError("policy version must be a mapping")
        if set(payload) - _ALLOWED_POLICY_FIELDS:
            raise PolicySimulationSchemaError(
                "policy version contains unsupported fields"
            )

        version = _first_present(
            payload,
            ("version", "name", "id", "policy_version"),
            default="policy",
        )
        actions = payload.get("actions", payload.get("rules", {}))
        policy_label_actions = payload.get("policy_label_actions", {})
        gates = _first_present(
            payload,
            ("blocking_gates", "required_gates", "gates"),
            default=(),
        )
        return cls(
            version=version,
            actions=actions,
            default_action=payload.get("default_action", "keep"),
            blocking_gates=_normalise_gate_requirements(
                gates,
                field_name="blocking gates",
            ),
            policy_label_actions=policy_label_actions,
        )

    @classmethod
    def from_profile(cls, profile: PolicyProfile) -> "PolicyVersion":
        """Adapt a bundled :class:`PolicyProfile` without changing it."""

        if not isinstance(profile, PolicyProfile):
            raise PolicySimulationSchemaError("policy profile is incompatible")
        gates: list[str] = []
        if profile.safety_sweep_mandatory:
            gates.append("safety_sweep")
        if profile.strict_no_leak:
            gates.append("no_leak")
        return cls(
            version=profile.name,
            actions=profile.actions,
            default_action=profile.default_action,
            blocking_gates=tuple(gates),
            policy_label_actions=profile.policy_label_actions,
        )

    @property
    def name(self) -> str:
        """Return the version identifier under the common policy name alias."""

        return self.version

    def action_for(self, resource_class: str) -> str:
        """Return the configured action for one resource class."""

        safe_class = _normalise_identifier(
            resource_class,
            field_name="resource class",
        )
        action = _case_insensitive_lookup(self.actions, safe_class)
        if action is not None:
            return action
        action = _case_insensitive_lookup(self.policy_label_actions, safe_class)
        return self.default_action if action is None else action

    @property
    def fingerprint(self) -> str:
        """Return a stable fingerprint of the safe policy configuration."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the validated policy configuration."""

        return {
            "schema_version": POLICY_SIMULATION_SCHEMA_VERSION,
            "version": self.version,
            "actions": dict(self.actions),
            "policy_label_actions": dict(self.policy_label_actions),
            "default_action": self.default_action,
            "blocking_gates": list(self.blocking_gates),
        }

    def __repr__(self) -> str:
        return (
            f"PolicyVersion(version={self.version!r}, fingerprint={self.fingerprint!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class PolicyScenario:
    """One aggregate, synthetic resource-class scenario.

    The scenario deliberately stores a count and boolean gate outcomes only.
    ``scenario_id`` is retained only for an internal fingerprint and is never
    serialized or included in the object's representation.
    """

    resource_class: str
    count: int = 1
    gate_outcomes: Mapping[str, bool] = dataclass_field(default_factory=dict)
    scenario_id: str | None = dataclass_field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_class",
            _normalise_identifier(
                self.resource_class,
                field_name="resource class",
            ),
        )
        object.__setattr__(self, "count", _normalise_count(self.count))
        object.__setattr__(
            self, "gate_outcomes", _normalise_gate_outcomes(self.gate_outcomes)
        )
        if self.scenario_id is not None:
            object.__setattr__(
                self,
                "scenario_id",
                _normalise_identifier(self.scenario_id, field_name="scenario id"),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PolicyScenario":
        """Build a count-only scenario from a closed mapping."""

        if not isinstance(payload, Mapping):
            raise PolicySimulationSchemaError("scenario must be a mapping")
        if set(payload) - _ALLOWED_SCENARIO_FIELDS:
            raise PolicySimulationSchemaError("scenario contains unsupported fields")
        resource_class = _first_present(
            payload,
            ("resource_class", "resource", "class"),
            default=None,
        )
        if resource_class is None:
            raise PolicySimulationSchemaError("scenario requires a resource class")
        gate_outcomes = _first_present(
            payload,
            ("gate_outcomes", "gate_results", "gates"),
            default={},
        )
        scenario_id = _first_present(
            payload,
            ("scenario_id", "id", "name"),
            default=None,
        )
        return cls(
            resource_class=resource_class,
            count=payload.get("count", 1),
            gate_outcomes=gate_outcomes,
            scenario_id=scenario_id,
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable identifier without exposing an optional scenario id."""

        return _digest(
            {
                "resource_class": self.resource_class,
                "count": self.count,
                "gate_outcomes": dict(self.gate_outcomes),
                "scenario_id": self.scenario_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free scenario representation."""

        return {
            "resource_class": self.resource_class,
            "count": self.count,
            "gate_outcomes": dict(self.gate_outcomes),
            "scenario_fingerprint": self.fingerprint,
        }

    def __repr__(self) -> str:
        return (
            "PolicyScenario("
            f"resource_class={self.resource_class!r}, count={self.count}, "
            f"fingerprint={self.fingerprint!r})"
        )


@dataclass(frozen=True, slots=True)
class PolicySimulationRow:
    """Value-free comparison of one synthetic scenario under two policies."""

    scenario_index: int
    scenario_fingerprint: str
    resource_class: str
    count: int
    base_action: str
    candidate_action: str
    base_gate_outcome: str
    candidate_gate_outcome: str
    base_blocked: bool
    candidate_blocked: bool
    base_affected_count: int
    candidate_affected_count: int
    base_processed_count: int
    candidate_processed_count: int
    action_change: str
    count_change: str
    processed_count_change: str
    blocking_change: str

    @property
    def changed(self) -> bool:
        """Return whether this row changes in any release-relevant dimension."""

        return any(
            change != "unchanged"
            for change in (
                self.action_change,
                self.count_change,
                self.processed_count_change,
                self.blocking_change,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic row representation."""

        return {
            "scenario_index": self.scenario_index,
            "scenario_fingerprint": self.scenario_fingerprint,
            "resource_class": self.resource_class,
            "count": self.count,
            "base": {
                "action": self.base_action,
                "gate_outcome": self.base_gate_outcome,
                "blocked": self.base_blocked,
                "affected_count": self.base_affected_count,
                "processed_count": self.base_processed_count,
            },
            "candidate": {
                "action": self.candidate_action,
                "gate_outcome": self.candidate_gate_outcome,
                "blocked": self.candidate_blocked,
                "affected_count": self.candidate_affected_count,
                "processed_count": self.candidate_processed_count,
            },
            "changes": {
                "action": self.action_change,
                "count": self.count_change,
                "processed_count": self.processed_count_change,
                "blocking": self.blocking_change,
            },
        }


@dataclass(frozen=True, slots=True, repr=False)
class PolicySimulationMatrix:
    """Immutable, aggregate comparison of two policy versions."""

    base_policy: PolicyVersion
    candidate_policy: PolicyVersion
    rows: tuple[PolicySimulationRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_policy, PolicyVersion) or not isinstance(
            self.candidate_policy,
            PolicyVersion,
        ):
            raise PolicySimulationSchemaError("matrix policies are incompatible")
        rows = tuple(self.rows)
        if not all(isinstance(row, PolicySimulationRow) for row in rows):
            raise PolicySimulationSchemaError("matrix rows are incompatible")
        if tuple(row.scenario_index for row in rows) != tuple(range(len(rows))):
            raise PolicySimulationSchemaError("matrix row indexes must be contiguous")
        object.__setattr__(self, "rows", rows)

    @property
    def scenario_count(self) -> int:
        """Return the number of scenarios in the matrix."""

        return len(self.rows)

    @property
    def total_count(self) -> int:
        """Return the total synthetic count represented by the rows."""

        return sum(row.count for row in self.rows)

    def summary(self) -> dict[str, Any]:
        """Return aggregate action, count, and blocking changes."""

        base_actions = _count_values(row.base_action for row in self.rows)
        candidate_actions = _count_values(row.candidate_action for row in self.rows)
        base_action_counts = _weighted_action_counts(
            ((row.base_action, row.count) for row in self.rows)
        )
        candidate_action_counts = _weighted_action_counts(
            ((row.candidate_action, row.count) for row in self.rows)
        )
        action_deltas = {
            action: candidate_action_counts[action] - base_action_counts[action]
            for action in ACTION_VALUES
        }
        resource_class_counts = _weighted_identifier_counts(
            ((row.resource_class, row.count) for row in self.rows)
        )
        base_gate_counts = _count_values(row.base_gate_outcome for row in self.rows)
        candidate_gate_counts = _count_values(
            row.candidate_gate_outcome for row in self.rows
        )
        base_blocked_count = sum(row.count for row in self.rows if row.base_blocked)
        candidate_blocked_count = sum(
            row.count for row in self.rows if row.candidate_blocked
        )
        base_affected_count = sum(row.base_affected_count for row in self.rows)
        candidate_affected_count = sum(
            row.candidate_affected_count for row in self.rows
        )
        base_processed_count = sum(row.base_processed_count for row in self.rows)
        candidate_processed_count = sum(
            row.candidate_processed_count for row in self.rows
        )

        return {
            "scenario_count": self.scenario_count,
            "resource_class_count": len(resource_class_counts),
            "total_count": self.total_count,
            "resource_class_counts": resource_class_counts,
            "action_counts": {
                "base": base_action_counts,
                "candidate": candidate_action_counts,
                "delta": action_deltas,
            },
            "action_change_counts": _ordered_counts(
                _count_values(row.action_change for row in self.rows),
                ACTION_CHANGE_VALUES,
            ),
            "count_change": {
                "base_affected": base_affected_count,
                "candidate_affected": candidate_affected_count,
                "affected_delta": candidate_affected_count - base_affected_count,
                "affected": _change_kind(
                    base_affected_count,
                    candidate_affected_count,
                ),
                "base_processed": base_processed_count,
                "candidate_processed": candidate_processed_count,
                "processed_delta": candidate_processed_count - base_processed_count,
                "processed": _change_kind(
                    base_processed_count,
                    candidate_processed_count,
                ),
                "row_change_counts": _ordered_counts(
                    _count_values(row.count_change for row in self.rows),
                    COUNT_CHANGE_VALUES,
                ),
            },
            "gate_outcome_counts": {
                "base": _ordered_counts(base_gate_counts, GATE_OUTCOME_VALUES),
                "candidate": _ordered_counts(
                    candidate_gate_counts,
                    GATE_OUTCOME_VALUES,
                ),
            },
            "blocking": {
                "base_blocked_count": base_blocked_count,
                "candidate_blocked_count": candidate_blocked_count,
                "blocked_delta": candidate_blocked_count - base_blocked_count,
                "change": _change_kind(
                    base_blocked_count,
                    candidate_blocked_count,
                ),
                "change_counts": _ordered_counts(
                    _count_values(row.blocking_change for row in self.rows),
                    BLOCKING_CHANGE_VALUES,
                ),
            },
            "changed_scenario_count": sum(row.changed for row in self.rows),
            "changed_count": sum(row.count for row in self.rows if row.changed),
            "base_action_scenario_counts": base_actions,
            "candidate_action_scenario_counts": candidate_actions,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free, JSON-compatible matrix."""

        return {
            "schema_version": POLICY_SIMULATION_SCHEMA_VERSION,
            "artifact": POLICY_SIMULATION_ARTIFACT,
            "base_policy": {
                "version": self.base_policy.version,
                "fingerprint": self.base_policy.fingerprint,
            },
            "candidate_policy": {
                "version": self.candidate_policy.version,
                "fingerprint": self.candidate_policy.fingerprint,
            },
            "summary": self.summary(),
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the matrix deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    def to_markdown(self) -> str:
        """Render a deterministic review table without scenario values."""

        summary = self.summary()
        count_change = summary["count_change"]
        blocking = summary["blocking"]
        lines = [
            "# Privacy policy simulation matrix",
            "",
            f"Base policy: `{self.base_policy.version}` "
            f"({self.base_policy.fingerprint})",
            f"Candidate policy: `{self.candidate_policy.version}` "
            f"({self.candidate_policy.fingerprint})",
            "",
            "## Summary",
            "",
            f"- Synthetic scenarios: {summary['scenario_count']}",
            f"- Synthetic count: {summary['total_count']}",
            f"- Changed scenarios: {summary['changed_scenario_count']}",
            f"- Changed synthetic count: {summary['changed_count']}",
            f"- Affected count: {count_change['base_affected']} -> "
            f"{count_change['candidate_affected']} "
            f"({count_change['affected']})",
            f"- Blocked count: {blocking['base_blocked_count']} -> "
            f"{blocking['candidate_blocked_count']} ({blocking['change']})",
            "",
            "## Matrix",
            "",
            "| # | Resource class | Count | Base action | Candidate action | "
            "Action change | Base gate | Candidate gate | Blocking change | "
            "Count change |",
            "|---:|---|---:|---|---|---|---|---|---|---|",
        ]
        for row in self.rows:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(row.scenario_index + 1),
                        _markdown_cell(row.resource_class),
                        str(row.count),
                        row.base_action,
                        row.candidate_action,
                        row.action_change,
                        row.base_gate_outcome,
                        row.candidate_gate_outcome,
                        row.blocking_change,
                        row.count_change,
                    )
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "## Action counts",
                "",
                "| Action | Base | Candidate | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        action_counts = summary["action_counts"]
        for action in ACTION_VALUES:
            lines.append(
                f"| {action} | {action_counts['base'][action]} | "
                f"{action_counts['candidate'][action]} | "
                f"{action_counts['delta'][action]} |"
            )
        lines.extend(
            [
                "",
                "This matrix is synthetic review evidence, not a compliance "
                "certification or clinical decision guarantee.",
            ]
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            "PolicySimulationMatrix("
            f"base={self.base_policy.version!r}, "
            f"candidate={self.candidate_policy.version!r}, "
            f"scenarios={self.scenario_count})"
        )


@dataclass(frozen=True, slots=True)
class _PolicyEvaluation:
    action: str
    gate_outcome: str
    blocked: bool
    affected_count: int
    processed_count: int


def simulate_policy_matrix(
    base_policy: Any,
    candidate_policy: Any,
    scenarios: Sequence[PolicyScenario | Mapping[str, Any]],
) -> PolicySimulationMatrix:
    """Run count-only synthetic scenarios through two policy versions.

    Args:
        base_policy: A :class:`PolicyVersion`, bundled policy name/profile,
            local JSON path, or closed policy mapping.
        candidate_policy: The policy version to compare with ``base_policy``.
        scenarios: A sequence of :class:`PolicyScenario` values or mappings
            containing a resource class, non-negative count, and boolean gate
            outcomes.

    Returns:
        An immutable matrix with deterministic aggregate and row-level output.

    Raises:
        PolicySimulationError: If a policy or scenario is malformed.

    The function only evaluates in-memory counts and booleans.  It does not
    call a network, load a model, mutate an input mapping, or consume a budget.
    """

    base = _coerce_policy(base_policy)
    candidate = _coerce_policy(candidate_policy)
    scenario_values = _coerce_scenarios(scenarios)
    rows: list[PolicySimulationRow] = []
    for index, scenario in enumerate(scenario_values):
        base_result = _evaluate(base, scenario)
        candidate_result = _evaluate(candidate, scenario)
        rows.append(
            PolicySimulationRow(
                scenario_index=index,
                scenario_fingerprint=scenario.fingerprint,
                resource_class=scenario.resource_class,
                count=scenario.count,
                base_action=base_result.action,
                candidate_action=candidate_result.action,
                base_gate_outcome=base_result.gate_outcome,
                candidate_gate_outcome=candidate_result.gate_outcome,
                base_blocked=base_result.blocked,
                candidate_blocked=candidate_result.blocked,
                base_affected_count=base_result.affected_count,
                candidate_affected_count=candidate_result.affected_count,
                base_processed_count=base_result.processed_count,
                candidate_processed_count=candidate_result.processed_count,
                action_change=_action_change(
                    base_result.action,
                    candidate_result.action,
                ),
                count_change=_change_kind(
                    base_result.affected_count,
                    candidate_result.affected_count,
                ),
                processed_count_change=_change_kind(
                    base_result.processed_count,
                    candidate_result.processed_count,
                ),
                blocking_change=_blocking_change(
                    base_result.blocked,
                    candidate_result.blocked,
                ),
            )
        )
    return PolicySimulationMatrix(base, candidate, tuple(rows))


def build_policy_simulation_matrix(
    base_policy: Any,
    candidate_policy: Any,
    scenarios: Sequence[PolicyScenario | Mapping[str, Any]],
) -> PolicySimulationMatrix:
    """Alias for :func:`simulate_policy_matrix` for builder-style callers."""

    return simulate_policy_matrix(base_policy, candidate_policy, scenarios)


def run_policy_simulation(
    base_policy: Any,
    candidate_policy: Any,
    scenarios: Sequence[PolicyScenario | Mapping[str, Any]],
) -> PolicySimulationMatrix:
    """Alias for :func:`simulate_policy_matrix` for runner-style callers."""

    return simulate_policy_matrix(base_policy, candidate_policy, scenarios)


def render_policy_simulation_matrix(
    matrix: PolicySimulationMatrix,
    fmt: Literal["markdown", "md", "json", "dict", "text"] = "markdown",
) -> str | dict[str, Any]:
    """Render a simulation matrix as Markdown, JSON, or a safe dictionary."""

    if not isinstance(matrix, PolicySimulationMatrix):
        raise PolicySimulationSchemaError("matrix is incompatible")
    if fmt in {"markdown", "md", "text"}:
        return matrix.to_markdown()
    if fmt == "json":
        return matrix.to_json()
    if fmt == "dict":
        return matrix.to_dict()
    raise ValueError("format must be markdown, json, or dict")


def render_policy_matrix(
    matrix: PolicySimulationMatrix,
    fmt: Literal["markdown", "md", "json", "dict", "text"] = "markdown",
) -> str | dict[str, Any]:
    """Alias for :func:`render_policy_simulation_matrix`."""

    return render_policy_simulation_matrix(matrix, fmt=fmt)


def _coerce_policy(value: Any) -> PolicyVersion:
    if isinstance(value, PolicyVersion):
        return value
    if isinstance(value, PolicyProfile):
        return PolicyVersion.from_profile(value)
    if isinstance(value, Mapping):
        return PolicyVersion.from_mapping(value)
    if isinstance(value, Path):
        return _load_policy_path(value)
    if isinstance(value, str):
        path = Path(value)
        if path.exists():
            return _load_policy_path(path)
        try:
            return PolicyVersion.from_profile(load_policy(value))
        except (TypeError, ValueError, OSError) as exc:
            raise PolicySimulationSchemaError(
                "policy version could not be loaded"
            ) from exc
    try:
        return PolicyVersion.from_profile(load_policy(value))
    except (TypeError, ValueError, OSError) as exc:
        raise PolicySimulationSchemaError("policy version is incompatible") from exc


def _load_policy_path(path: Path) -> PolicyVersion:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PolicySimulationSchemaError("policy file could not be loaded") from exc
    if not isinstance(payload, Mapping):
        raise PolicySimulationSchemaError("policy file must contain a mapping")
    return PolicyVersion.from_mapping(cast(Mapping[str, Any], payload))


def _coerce_scenarios(
    scenarios: Sequence[PolicyScenario | Mapping[str, Any]],
) -> tuple[PolicyScenario, ...]:
    if isinstance(scenarios, (str, bytes, bytearray)) or not isinstance(
        scenarios,
        Sequence,
    ):
        raise PolicySimulationSchemaError("scenarios must be a sequence")
    values: list[PolicyScenario] = []
    for scenario in scenarios:
        if isinstance(scenario, PolicyScenario):
            values.append(scenario)
        elif isinstance(scenario, Mapping):
            values.append(PolicyScenario.from_mapping(scenario))
        else:
            raise PolicySimulationSchemaError("scenario is incompatible")
    return tuple(values)


def _evaluate(policy: PolicyVersion, scenario: PolicyScenario) -> _PolicyEvaluation:
    action = policy.action_for(scenario.resource_class)
    required = policy.blocking_gates
    if not required:
        gate_outcome = "not_required"
        blocked = False
    else:
        missing = [gate for gate in required if gate not in scenario.gate_outcomes]
        failed = [
            gate
            for gate in required
            if gate in scenario.gate_outcomes and not scenario.gate_outcomes[gate]
        ]
        if missing:
            gate_outcome = "missing"
            blocked = True
        elif failed:
            gate_outcome = "fail"
            blocked = True
        else:
            gate_outcome = "pass"
            blocked = False

    affected_count = scenario.count if action != "keep" else 0
    processed_count = 0 if blocked else scenario.count
    return _PolicyEvaluation(
        action=action,
        gate_outcome=gate_outcome,
        blocked=blocked,
        affected_count=affected_count,
        processed_count=processed_count,
    )


def _first_present(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: Any,
) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _case_insensitive_lookup(
    values: Mapping[str, str],
    key: str,
) -> str | None:
    folded = key.casefold()
    for candidate, value in values.items():
        if candidate.casefold() == folded:
            return value
    return None


def _count_values(values: Sequence[str] | Any) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _ordered_counts(values: Mapping[str, int], order: Sequence[str]) -> dict[str, int]:
    return {key: int(values.get(key, 0)) for key in order}


def _weighted_action_counts(values: Any) -> dict[str, int]:
    counts = Counter()
    for action, count in values:
        counts[action] += count
    return {action: counts[action] for action in ACTION_VALUES}


def _weighted_identifier_counts(values: Any) -> dict[str, int]:
    counts = Counter()
    for identifier, count in values:
        counts[identifier] += count
    return {key: counts[key] for key in sorted(counts)}


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


__all__ = [
    "ACTION_CHANGE_VALUES",
    "BLOCKING_CHANGE_VALUES",
    "COUNT_CHANGE_VALUES",
    "GATE_OUTCOME_VALUES",
    "POLICY_SIMULATION_ARTIFACT",
    "POLICY_SIMULATION_MATRIX_SCHEMA_VERSION",
    "POLICY_SIMULATION_SCHEMA_VERSION",
    "PolicyScenario",
    "PolicySimulationPolicy",
    "PolicySimulationError",
    "PolicySimulationMatrix",
    "PolicySimulationReport",
    "PolicySimulationRow",
    "PolicySimulationScenario",
    "PolicySimulationSchemaError",
    "PolicyVersion",
    "build_policy_simulation_matrix",
    "render_policy_matrix",
    "render_policy_simulation_matrix",
    "run_policy_simulation",
    "simulate_policy_matrix",
]


# Compatibility names keep the public vocabulary discoverable without creating
# separate mutable types or duplicating the simulation implementation.
PolicySimulationScenario = PolicyScenario
PolicySimulationPolicy = PolicyVersion
PolicySimulationReport = PolicySimulationMatrix
