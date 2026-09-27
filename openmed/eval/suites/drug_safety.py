"""Frozen evaluation for descriptive drug-safety signal calculations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from openmed.clinical.drug_safety import (
    DRUG_SAFETY_ADAPTER_VERSION,
    DRUG_SAFETY_METHOD_ID,
    DRUG_SAFETY_METHOD_VERSION,
    DrugSafetyDataset,
    SignalContingencyTable,
    SignalFilter,
    SignalPolicy,
    SignalState,
    compute_descriptive_signal,
)
from openmed.clinical.journey_contracts import canonical_digest

DRUG_SAFETY_SUITE_VERSION: Final = "1.0.0"


@dataclass(frozen=True, slots=True)
class DrugSafetyBenchmarkCase:
    """One frozen drug-event calculation with hand-checked expectations."""

    case_id: str
    drug: str
    event: str
    expected_state: SignalState
    expected_table: SignalContingencyTable
    expected_prr: float | None = None
    expected_ror: float | None = None
    signal_filter: SignalFilter = SignalFilter()

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.case_id, self.drug, self.event)
        ):
            raise ValueError("benchmark identity, drug, and event are required")
        if not isinstance(self.expected_table, SignalContingencyTable):
            raise TypeError("expected_table must be SignalContingencyTable")
        state = (
            self.expected_state
            if isinstance(self.expected_state, SignalState)
            else SignalState(self.expected_state)
        )
        object.__setattr__(self, "expected_state", state)
        for name in ("expected_prr", "expected_ror"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if state is SignalState.COMPUTED and (
            self.expected_prr is None or self.expected_ror is None
        ):
            raise ValueError("computed benchmark cases require expected ratios")
        if state is not SignalState.COMPUTED and (
            self.expected_prr is not None or self.expected_ror is not None
        ):
            raise ValueError("non-computed benchmark cases cannot expect ratios")


@dataclass(frozen=True, slots=True)
class DrugSafetyBenchmarkReport:
    """Exact-count and bounded-ratio metrics with input provenance."""

    case_count: int
    exact_table_count: int
    exact_state_count: int
    exact_table_accuracy: float
    exact_state_accuracy: float
    maximum_absolute_ratio_error: float
    fixture_digest: str
    dataset_digest: str
    policy_digest: str
    versions: Mapping[str, str]
    suite_version: str = DRUG_SAFETY_SUITE_VERSION

    def __post_init__(self) -> None:
        for name in ("case_count", "exact_table_count", "exact_state_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("exact_table_accuracy", "exact_state_accuracy"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.maximum_absolute_ratio_error < 0 or not math.isfinite(
            self.maximum_absolute_ratio_error
        ):
            raise ValueError("maximum_absolute_ratio_error must be finite")
        for name in ("fixture_digest", "dataset_digest", "policy_digest"):
            if not str(getattr(self, name)).startswith("sha256:"):
                raise ValueError(f"{name} must be a normalized digest")
        object.__setattr__(
            self, "versions", MappingProxyType(dict(sorted(self.versions.items())))
        )

    def to_dict(self) -> dict[str, object]:
        """Return stable benchmark metrics and provenance."""

        return {
            "case_count": self.case_count,
            "dataset_digest": self.dataset_digest,
            "exact_state_accuracy": self.exact_state_accuracy,
            "exact_state_count": self.exact_state_count,
            "exact_table_accuracy": self.exact_table_accuracy,
            "exact_table_count": self.exact_table_count,
            "fixture_digest": self.fixture_digest,
            "maximum_absolute_ratio_error": self.maximum_absolute_ratio_error,
            "policy_digest": self.policy_digest,
            "suite_version": self.suite_version,
            "versions": dict(self.versions),
        }


def run_drug_safety_benchmark(
    dataset: DrugSafetyDataset,
    cases: Sequence[DrugSafetyBenchmarkCase],
    *,
    policy: SignalPolicy | None = None,
) -> DrugSafetyBenchmarkReport:
    """Reproduce frozen tables and descriptive statistics exactly."""

    if not isinstance(dataset, DrugSafetyDataset):
        raise TypeError("dataset must be DrugSafetyDataset")
    case_values = tuple(cases)
    if any(not isinstance(item, DrugSafetyBenchmarkCase) for item in case_values):
        raise TypeError("cases must contain DrugSafetyBenchmarkCase")
    active_policy = policy or SignalPolicy()
    exact_tables = 0
    exact_states = 0
    maximum_error = 0.0
    fixture_payload: list[dict[str, object]] = []
    for case in case_values:
        signal = compute_descriptive_signal(
            dataset,
            drug=case.drug,
            event=case.event,
            signal_filter=case.signal_filter,
            policy=active_policy,
        )
        exact_tables += signal.table == case.expected_table
        exact_states += signal.state is case.expected_state
        if signal.state is SignalState.COMPUTED:
            assert signal.proportional_reporting_ratio is not None
            assert signal.reporting_odds_ratio is not None
            assert case.expected_prr is not None
            assert case.expected_ror is not None
            maximum_error = max(
                maximum_error,
                abs(signal.proportional_reporting_ratio - case.expected_prr),
                abs(signal.reporting_odds_ratio - case.expected_ror),
            )
        fixture_payload.append(
            {
                "case_id": case.case_id,
                "drug": case.drug,
                "event": case.event,
                "expected_prr": case.expected_prr,
                "expected_ror": case.expected_ror,
                "expected_state": case.expected_state.value,
                "expected_table": case.expected_table.to_dict(),
                "filter": case.signal_filter.to_dict(),
            }
        )
    count = len(case_values)
    return DrugSafetyBenchmarkReport(
        case_count=count,
        exact_table_count=exact_tables,
        exact_state_count=exact_states,
        exact_table_accuracy=exact_tables / count if count else 0.0,
        exact_state_accuracy=exact_states / count if count else 0.0,
        maximum_absolute_ratio_error=maximum_error,
        fixture_digest=canonical_digest(fixture_payload),
        dataset_digest=dataset.dataset_digest,
        policy_digest=active_policy.digest,
        versions={
            "adapter": DRUG_SAFETY_ADAPTER_VERSION,
            "method": f"{DRUG_SAFETY_METHOD_ID}/{DRUG_SAFETY_METHOD_VERSION}",
            "suite": DRUG_SAFETY_SUITE_VERSION,
        },
    )


__all__ = [
    "DRUG_SAFETY_SUITE_VERSION",
    "DrugSafetyBenchmarkCase",
    "DrugSafetyBenchmarkReport",
    "run_drug_safety_benchmark",
]
