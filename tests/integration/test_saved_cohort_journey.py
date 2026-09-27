"""Golden synthetic journey for saved cohort custody and explanations."""

from __future__ import annotations

from pathlib import Path

from openmed.agent.workflows import explain_cohort_membership
from openmed.structured.cohort import (
    CohortMembership,
    CohortSourceSnapshot,
    CriterionMembership,
    LocalSavedCohortStore,
    MembershipEvidence,
    MembershipState,
    PhenotypeDefinition,
    build_cohort_execution,
    save_cohort_definition,
)
from openmed.structured.store import StoreState

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cohort"
    / "phenotypes"
    / "diabetes_on_metformin.json"
)


def test_synthetic_snapshot_to_saved_membership_is_reproducible(
    tmp_path: Path,
) -> None:
    version = save_cohort_definition(PhenotypeDefinition.load(FIXTURE))
    membership = CohortMembership(
        patient_key="patient_goldenjourney001",
        state=MembershipState.MET,
        criteria=(
            CriterionMembership(
                criterion_id="has-diabetes",
                state=MembershipState.MET,
                evidence=(
                    MembershipEvidence(
                        evidence_id="evidence_goldencondition1",
                        fact_id="fact_goldencondition001",
                        time_window_id="window_goldenjourney001",
                    ),
                ),
            ),
            CriterionMembership(
                criterion_id="has-metformin",
                state=MembershipState.MET,
                evidence=(
                    MembershipEvidence(
                        evidence_id="evidence_goldenmedication",
                        fact_id="fact_goldenmedication01",
                        time_window_id="window_goldenjourney001",
                    ),
                ),
            ),
        ),
    )
    built = build_cohort_execution(
        version,
        source_snapshot=CohortSourceSnapshot(
            snapshot_id="snapshot_goldenjourney001",
            digest="sha256:" + "1" * 64,
            schema_version="journey-snapshot-v1",
            license_tags=("synthetic",),
        ),
        vocabulary_digest="sha256:" + "2" * 64,
        policy_digest="sha256:" + "3" * 64,
        evaluator_version="cohort-evaluator-1.0",
        memberships=(membership,),
    )
    assert built.state is StoreState.SUCCESS
    assert built.value is not None

    store = LocalSavedCohortStore(tmp_path / "saved-cohorts")
    assert store.put_definition(version).ok
    assert store.put_execution(built.value).ok
    rerun = store.rerun(
        built.value.manifest.execution_id or "",
        lambda context: (membership,),
    )
    explained = explain_cohort_membership(
        built.value,
        "patient_goldenjourney001",
    )

    assert rerun.state is StoreState.SUCCESS
    assert explained.state is StoreState.SUCCESS
    assert explained.value is not None
    assert explained.value.membership.eligible
    assert built.value.membership_digest == (
        "sha256:a47dcaee8fdd33c18463d81b309576026c5d54721ef4794c83925c1f28646fc9"
    )
    assert built.value.execution_digest == (
        "sha256:b2190591099483bd059137b6ed08421069e63ededc98dcf72fc6f73781bab15f"
    )
