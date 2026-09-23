"""Frozen, provenance-bound evaluation for trial retrieval and criterion states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from openmed.clinical.journey import JourneySnapshot
from openmed.clinical.journey_contracts import canonical_digest
from openmed.clinical.trials import TrialStudyRecord
from openmed.clinical.trials.matcher import (
    JourneySignal,
    TrialCriterionState,
    TrialMatchPolicy,
    match_trial_candidates,
    matcher_provenance,
)

TRIAL_ELIGIBILITY_SUITE_VERSION: Final = "1.0.0"


@dataclass(frozen=True, slots=True)
class TrialEligibilityBenchmarkCase:
    """One synthetic or permitted frozen retrieval-and-state benchmark case."""

    case_id: str
    snapshot: JourneySnapshot
    signals: tuple[JourneySignal, ...]
    expected_study_ids: tuple[str, ...]
    expected_criterion_states: Mapping[str, TrialCriterionState]
    evaluated_at: str

    def __post_init__(self) -> None:
        if not self.case_id or not isinstance(self.snapshot, JourneySnapshot):
            raise ValueError("benchmark case identity and snapshot are required")
        if any(not isinstance(item, JourneySignal) for item in self.signals):
            raise TypeError("benchmark signals must contain JourneySignal")
        if not self.expected_study_ids:
            raise ValueError("benchmark case requires expected study identifiers")
        normalized = {
            key: value
            if isinstance(value, TrialCriterionState)
            else TrialCriterionState(value)
            for key, value in self.expected_criterion_states.items()
        }
        object.__setattr__(
            self,
            "expected_criterion_states",
            MappingProxyType(dict(sorted(normalized.items()))),
        )


@dataclass(frozen=True, slots=True)
class TrialEligibilityBenchmarkReport:
    """Aggregate retrieval and criterion-state metrics with exact provenance."""

    case_count: int
    retrieval_recall_at_k: float
    criterion_state_accuracy: float
    expected_study_count: int
    correct_criterion_state_count: int
    criterion_state_count: int
    fixture_digest: str
    policy_digest: str
    versions: Mapping[str, str]
    suite_version: str = TRIAL_ELIGIBILITY_SUITE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "case_count",
            "expected_study_count",
            "correct_criterion_state_count",
            "criterion_state_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("retrieval_recall_at_k", "criterion_state_accuracy"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("fixture_digest", "policy_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ValueError(f"{name} must be a normalized digest")
        object.__setattr__(
            self, "versions", MappingProxyType(dict(sorted(self.versions.items())))
        )

    def to_dict(self) -> dict[str, object]:
        """Return stable benchmark metrics and provenance."""

        return {
            "case_count": self.case_count,
            "correct_criterion_state_count": self.correct_criterion_state_count,
            "criterion_state_accuracy": self.criterion_state_accuracy,
            "criterion_state_count": self.criterion_state_count,
            "expected_study_count": self.expected_study_count,
            "fixture_digest": self.fixture_digest,
            "policy_digest": self.policy_digest,
            "retrieval_recall_at_k": self.retrieval_recall_at_k,
            "suite_version": self.suite_version,
            "versions": dict(self.versions),
        }


def run_trial_eligibility_benchmark(
    studies: Sequence[TrialStudyRecord],
    cases: Sequence[TrialEligibilityBenchmarkCase],
    *,
    limit: int = 20,
    policy: TrialMatchPolicy | None = None,
) -> TrialEligibilityBenchmarkReport:
    """Run deterministic retrieval and state scoring over frozen cases."""

    study_values = tuple(studies)
    case_values = tuple(cases)
    active_policy = policy or TrialMatchPolicy()
    retrieved_expected = 0
    expected_total = 0
    correct_states = 0
    state_total = 0
    fixture_payload: list[dict[str, object]] = []
    for case in case_values:
        results = match_trial_candidates(
            study_values,
            case.snapshot,
            case.signals,
            evaluated_at=case.evaluated_at,
            limit=limit,
            policy=active_policy,
        )
        retrieved_ids = {item.candidate.study_id for item in results}
        expected = set(case.expected_study_ids)
        retrieved_expected += len(retrieved_ids.intersection(expected))
        expected_total += len(expected)
        by_criterion = {
            item.criterion_id: item.state
            for result in results
            for item in result.criterion_results
        }
        for criterion_id, expected_state in case.expected_criterion_states.items():
            state_total += 1
            correct_states += by_criterion.get(criterion_id) is expected_state
        fixture_payload.append(
            {
                "case_id": case.case_id,
                "expected_criterion_states": {
                    key: value.value
                    for key, value in case.expected_criterion_states.items()
                },
                "expected_study_ids": list(case.expected_study_ids),
                "signal_digest": canonical_digest(
                    [
                        {
                            "concept": signal.concept,
                            "concept_kind": signal.concept_kind,
                            "conflict_ids": list(signal.conflict_ids),
                            "evidence_ids": list(signal.evidence_ids),
                            "fact_ids": list(signal.fact_ids),
                            "numeric_value": signal.numeric_value,
                            "observed_at": signal.observed_at,
                            "present": signal.present,
                            "snapshot_digest": signal.snapshot_digest,
                            "snapshot_id": signal.snapshot_id,
                            "unit": signal.unit,
                        }
                        for signal in case.signals
                    ]
                ),
                "snapshot": case.snapshot.to_dict(),
            }
        )
    versions = matcher_provenance()
    return TrialEligibilityBenchmarkReport(
        case_count=len(case_values),
        retrieval_recall_at_k=(
            retrieved_expected / expected_total if expected_total else 0.0
        ),
        criterion_state_accuracy=(correct_states / state_total if state_total else 0.0),
        expected_study_count=expected_total,
        correct_criterion_state_count=correct_states,
        criterion_state_count=state_total,
        fixture_digest=canonical_digest(
            {
                "cases": fixture_payload,
                "studies": [
                    {
                        "study_id": study.study_id,
                        "version_digest": study.version_digest,
                        "version_id": study.version_id,
                    }
                    for study in study_values
                ],
            }
        ),
        policy_digest=active_policy.digest,
        versions=versions,
    )


__all__ = [
    "TRIAL_ELIGIBILITY_SUITE_VERSION",
    "TrialEligibilityBenchmarkCase",
    "TrialEligibilityBenchmarkReport",
    "run_trial_eligibility_benchmark",
]
