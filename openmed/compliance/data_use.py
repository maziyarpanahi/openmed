"""Consent and data-use tag enforcement for local pipeline actions.

The types in this module intentionally carry policy metadata only. They never
accept document text, subject identifiers, or raw PHI, so denied decisions can
be reported safely before a pipeline processes its input.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar


class DataUseTag(str, Enum):
    """Canonical consent and permitted-use constraints attached to an input."""

    RESEARCH_ONLY = "research-only"
    NO_SECONDARY_USE = "no-secondary-use"
    NO_EXPORT = "no-export"
    NO_RETENTION = "no-retention"
    NO_SURROGATE_VAULT = "no-surrogate-vault"
    EMBARGOED = "embargoed"
    CONSENT_WITHDRAWN = "consent-withdrawn"


class DataUseAction(str, Enum):
    """Pipeline or downstream actions that a data-use policy can gate."""

    PROCESS = "process"
    RESEARCH = "research"
    CLINICAL_CARE = "clinical-care"
    SECONDARY_USE = "secondary-use"
    EXPORT = "export"
    FHIR_EXPORT = "fhir-export"
    RETENTION = "retention"
    RETAIN = "retention"
    SURROGATE_VAULT = "surrogate-vault"


_ALL_ACTIONS = frozenset(DataUseAction)
DEFAULT_DENIED_ACTIONS: Mapping[DataUseTag, frozenset[DataUseAction]] = (
    MappingProxyType(
        {
            DataUseTag.RESEARCH_ONLY: frozenset({DataUseAction.CLINICAL_CARE}),
            DataUseTag.NO_SECONDARY_USE: frozenset({DataUseAction.SECONDARY_USE}),
            DataUseTag.NO_EXPORT: frozenset(
                {DataUseAction.EXPORT, DataUseAction.FHIR_EXPORT}
            ),
            DataUseTag.NO_RETENTION: frozenset({DataUseAction.RETENTION}),
            DataUseTag.NO_SURROGATE_VAULT: frozenset({DataUseAction.SURROGATE_VAULT}),
            DataUseTag.EMBARGOED: frozenset(
                {
                    DataUseAction.SECONDARY_USE,
                    DataUseAction.EXPORT,
                    DataUseAction.FHIR_EXPORT,
                }
            ),
            DataUseTag.CONSENT_WITHDRAWN: _ALL_ACTIONS,
        }
    )
)


@dataclass(frozen=True)
class DataUseViolation:
    """One tag and attempted-action pair denied by a policy."""

    tag: DataUseTag
    attempted_action: DataUseAction

    @property
    def decision(self) -> str:
        """Return the canonical decision recorded for this violation."""

        return "deny"

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic JSON-compatible, PHI-free record."""

        return {
            "tag": self.tag.value,
            "attempted_action": self.attempted_action.value,
            "decision": self.decision,
        }


@dataclass(frozen=True)
class TagViolationReport:
    """PHI-safe report for all data-use violations in one decision."""

    tags: tuple[DataUseTag, ...]
    attempted_actions: tuple[DataUseAction, ...]
    violations: tuple[DataUseViolation, ...]

    def __post_init__(self) -> None:
        if not self.violations:
            raise ValueError("a tag-violation report requires at least one violation")

    @property
    def decision(self) -> str:
        """Return the fail-closed decision represented by this report."""

        return "deny"

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible, PHI-free report."""

        return {
            "tags": [tag.value for tag in self.tags],
            "attempted_actions": [action.value for action in self.attempted_actions],
            "decision": self.decision,
            "violations": [violation.to_dict() for violation in self.violations],
        }


@dataclass(frozen=True)
class DataUseEvaluation:
    """Result of evaluating tags against one or more attempted actions."""

    tags: tuple[DataUseTag, ...]
    attempted_actions: tuple[DataUseAction, ...]
    violations: tuple[DataUseViolation, ...] = ()

    @property
    def allowed(self) -> bool:
        """Return whether every attempted action is permitted."""

        return not self.violations

    @property
    def decision(self) -> str:
        """Return ``allow`` or ``deny`` for audit serialization."""

        return "allow" if self.allowed else "deny"

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible, PHI-free decision."""

        return {
            "tags": [tag.value for tag in self.tags],
            "attempted_actions": [action.value for action in self.attempted_actions],
            "decision": self.decision,
            "violations": [violation.to_dict() for violation in self.violations],
        }

    def violation_report(self) -> TagViolationReport | None:
        """Return the denial report, or ``None`` when the decision is allowed."""

        if self.allowed:
            return None
        return TagViolationReport(
            tags=self.tags,
            attempted_actions=self.attempted_actions,
            violations=self.violations,
        )


class DataUsePolicyViolation(PermissionError):
    """Raised before processing when a data-use policy denies an action."""

    def __init__(self, report: TagViolationReport) -> None:
        self.report = report
        pairs = ", ".join(
            f"{item.tag.value}:{item.attempted_action.value}"
            for item in report.violations
        )
        super().__init__(f"Data-use policy denied action(s): {pairs}")


# A descriptive alias for callers that prefer an error-oriented name.
DataUseViolationError = DataUsePolicyViolation


@dataclass(frozen=True)
class DataUsePolicy:
    """Map canonical tags to actions that must be denied.

    Unknown tags and actions are rejected rather than ignored. This makes
    misspelled or newer constraints fail closed instead of silently weakening
    enforcement.
    """

    denied_actions: Mapping[
        DataUseTag | str,
        Iterable[DataUseAction | str],
    ] = field(default_factory=lambda: DEFAULT_DENIED_ACTIONS)

    def __post_init__(self) -> None:
        normalized: dict[DataUseTag, frozenset[DataUseAction]] = {}
        for raw_tag, raw_actions in self.denied_actions.items():
            tag = _coerce_enum(raw_tag, DataUseTag, "data-use tag")
            normalized[tag] = frozenset(normalize_data_use_actions(raw_actions))
        object.__setattr__(self, "denied_actions", MappingProxyType(normalized))

    def evaluate(
        self,
        tags: DataUseTag | str | Iterable[DataUseTag | str],
        attempted_actions: DataUseAction
        | str
        | Iterable[DataUseAction | str] = DataUseAction.PROCESS,
    ) -> DataUseEvaluation:
        """Evaluate ``attempted_actions`` for inputs carrying ``tags``."""

        resolved_tags = normalize_data_use_tags(tags)
        resolved_actions = normalize_data_use_actions(attempted_actions)
        if not resolved_actions:
            raise ValueError("at least one attempted data-use action is required")

        violations = tuple(
            DataUseViolation(tag=tag, attempted_action=action)
            for tag in resolved_tags
            for action in resolved_actions
            if action in self.denied_actions.get(tag, frozenset())
        )
        return DataUseEvaluation(
            tags=resolved_tags,
            attempted_actions=resolved_actions,
            violations=violations,
        )

    def enforce(
        self,
        tags: DataUseTag | str | Iterable[DataUseTag | str],
        attempted_actions: DataUseAction
        | str
        | Iterable[DataUseAction | str] = DataUseAction.PROCESS,
    ) -> DataUseEvaluation:
        """Return an allowed evaluation or raise a PHI-safe denial error."""

        evaluation = self.evaluate(tags, attempted_actions)
        report = evaluation.violation_report()
        if report is not None:
            raise DataUsePolicyViolation(report)
        return evaluation


_EnumT = TypeVar("_EnumT", bound=Enum)


def _coerce_enum(value: Any, enum_type: type[_EnumT], field_name: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    canonical = value.strip().lower().replace("_", "-").replace(" ", "-")
    if enum_type is DataUseAction and canonical == "retain":
        canonical = DataUseAction.RETENTION.value
    try:
        return enum_type(canonical)
    except ValueError as exc:
        raise ValueError(
            f"unknown {field_name} {value!r}; refusing to process (fail closed)"
        ) from exc


def _as_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, Enum)):
        return (value,)
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError("data-use values must be a value or iterable") from exc


def normalize_data_use_tags(
    tags: DataUseTag | str | Iterable[DataUseTag | str],
) -> tuple[DataUseTag, ...]:
    """Return unique canonical tags in deterministic order."""

    resolved = {
        _coerce_enum(value, DataUseTag, "data-use tag") for value in _as_values(tags)
    }
    return tuple(sorted(resolved, key=lambda item: item.value))


def normalize_data_use_actions(
    actions: DataUseAction | str | Iterable[DataUseAction | str],
) -> tuple[DataUseAction, ...]:
    """Return unique canonical actions in deterministic order."""

    resolved = {
        _coerce_enum(value, DataUseAction, "data-use action")
        for value in _as_values(actions)
    }
    return tuple(sorted(resolved, key=lambda item: item.value))


DEFAULT_DATA_USE_POLICY = DataUsePolicy()


__all__ = [
    "DEFAULT_DATA_USE_POLICY",
    "DEFAULT_DENIED_ACTIONS",
    "DataUseAction",
    "DataUseEvaluation",
    "DataUsePolicy",
    "DataUsePolicyViolation",
    "DataUseTag",
    "DataUseViolation",
    "DataUseViolationError",
    "TagViolationReport",
    "normalize_data_use_actions",
    "normalize_data_use_tags",
]
