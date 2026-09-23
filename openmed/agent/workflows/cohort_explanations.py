"""Privacy-safe workflow explanations for saved cohort membership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.structured.cohort.saved import (
    SAVED_COHORT_ADVISORY,
    CohortExecution,
    CohortMembership,
)
from openmed.structured.store import StoreResult, StoreState

COHORT_EXPLANATION_SCHEMA_VERSION: Final = "openmed.cohort.explanation.v1"


@dataclass(frozen=True, slots=True)
class CohortMembershipExplanation:
    """A value-free explanation with opaque patient and evidence references."""

    execution_id: str
    membership: CohortMembership
    membership_digest: str
    schema_version: str = COHORT_EXPLANATION_SCHEMA_VERSION
    advisory: str = SAVED_COHORT_ADVISORY

    def __post_init__(self) -> None:
        if self.schema_version != COHORT_EXPLANATION_SCHEMA_VERSION:
            raise ValueError("unsupported cohort explanation schema version")
        if not isinstance(self.membership, CohortMembership):
            raise TypeError("membership must be CohortMembership")
        expected = canonical_digest(self.membership.to_dict())
        if self.membership_digest != expected:
            raise ValueError("membership explanation digest differs")
        if self.advisory != SAVED_COHORT_ADVISORY:
            raise ValueError("cohort explanation advisory differs")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic explanation without source clinical values."""

        return {
            "advisory": self.advisory,
            "execution_id": self.execution_id,
            "membership": self.membership.to_dict(),
            "membership_digest": self.membership_digest,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return byte-stable explanation JSON."""

        return canonical_json(self.to_dict())


def explain_cohort_membership(
    execution: CohortExecution,
    patient_key: str,
) -> StoreResult[CohortMembershipExplanation]:
    """Explain one saved membership or return an explicit unknown outcome."""

    if not isinstance(execution, CohortExecution):
        return StoreResult.outcome(StoreState.FAILURE, "cohort_execution_invalid")
    try:
        membership = execution.membership_for(patient_key)
    except (TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "patient_key_invalid")
    if membership is None:
        return StoreResult.outcome(StoreState.UNKNOWN, "membership_not_found")
    explanation = CohortMembershipExplanation(
        execution_id=execution.manifest.execution_id or "",
        membership=membership,
        membership_digest=canonical_digest(membership.to_dict()),
    )
    return StoreResult.success(explanation)


__all__ = [
    "COHORT_EXPLANATION_SCHEMA_VERSION",
    "CohortMembershipExplanation",
    "explain_cohort_membership",
]
