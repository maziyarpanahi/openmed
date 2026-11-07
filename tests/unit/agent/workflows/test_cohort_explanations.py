"""Opaque, criterion-level saved-cohort explanation tests."""

from __future__ import annotations

from pathlib import Path

from openmed.agent.workflows import explain_cohort_membership
from openmed.structured.cohort import (
    CohortExecution,
    CohortMembership,
    CohortSourceSnapshot,
    CriterionMembership,
    MembershipEvidence,
    MembershipState,
    PhenotypeDefinition,
    build_cohort_execution,
    save_cohort_definition,
)
from openmed.structured.store import StoreState

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "cohort" / "phenotypes"


def _execution() -> CohortExecution:
    definition = PhenotypeDefinition.load(FIXTURES / "diabetes_on_metformin.json")
    membership = CohortMembership(
        patient_key="patient_aaaaaaaaaaaaaaaa",
        state=MembershipState.MET,
        criteria=(
            CriterionMembership(
                criterion_id="has-diabetes",
                state=MembershipState.MET,
                evidence=(
                    MembershipEvidence(
                        evidence_id="evidence_aaaaaaaaaaaaaaaa",
                        fact_id="fact_aaaaaaaaaaaaaaaa",
                        time_window_id="window_aaaaaaaaaaaaaaaa",
                    ),
                ),
            ),
            CriterionMembership(
                criterion_id="has-metformin",
                state=MembershipState.MET,
                evidence=(
                    MembershipEvidence(
                        evidence_id="evidence_bbbbbbbbbbbbbbbb",
                        fact_id="fact_bbbbbbbbbbbbbbbb",
                        time_window_id="window_bbbbbbbbbbbbbbbb",
                    ),
                ),
            ),
        ),
    )
    result = build_cohort_execution(
        save_cohort_definition(definition),
        source_snapshot=CohortSourceSnapshot(
            snapshot_id="snapshot_aaaaaaaaaaaaaaaa",
            digest="sha256:" + "a" * 64,
            schema_version="journey-synthetic-v1",
            license_tags=("synthetic",),
        ),
        vocabulary_digest="sha256:" + "b" * 64,
        policy_digest="sha256:" + "c" * 64,
        evaluator_version="cohort-evaluator-1.0",
        memberships=(membership,),
    )
    assert result.value is not None
    return result.value


def test_explanation_reuses_membership_evidence_and_contains_only_opaque_keys() -> None:
    result = explain_cohort_membership(
        _execution(),
        "patient_aaaaaaaaaaaaaaaa",
    )

    assert result.state is StoreState.SUCCESS
    assert result.value is not None
    payload = result.value.to_dict()
    assert payload["membership"]["state"] == "met"
    assert payload["membership"]["eligible"] is True
    assert payload["membership"]["criteria"][0]["evidence"][0] == {
        "evidence_id": "evidence_aaaaaaaaaaaaaaaa",
        "fact_id": "fact_aaaaaaaaaaaaaaaa",
        "role": "supporting",
        "time_window_id": "window_aaaaaaaaaaaaaaaa",
    }
    serialized = result.value.to_json()
    assert "source_text" not in serialized
    assert "raw_text" not in serialized


def test_missing_membership_is_an_explicit_unknown() -> None:
    result = explain_cohort_membership(
        _execution(),
        "patient_bbbbbbbbbbbbbbbb",
    )

    assert result.state is StoreState.UNKNOWN
    assert result.code == "membership_not_found"


def test_invalid_direct_identifier_is_rejected_without_echo() -> None:
    result = explain_cohort_membership(_execution(), "person@example.test")

    assert result.state is StoreState.FAILURE
    assert result.code == "patient_key_invalid"
