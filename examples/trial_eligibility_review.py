"""Run synthetic trial rule/assessment comparison offline.

Run from the repository root: python -m examples.trial_eligibility_review
Assessments are hand-authored fixtures, not model outputs or clinical evidence.
Agreement never authorizes enrollment or contact.
"""

import json
from dataclasses import replace

from openmed.agent.workflows import (
    CohortCriterion,
    CohortDefinition,
    CohortRecordEvidence,
    CriterionEvidence,
    CriterionKind,
    CriterionState,
    EligibilityCitation,
    EvidenceAssertion,
    ModelCriterionAssessment,
    TrialEligibilityReviewError,
    build_trial_eligibility_review_packet,
    explain_criterion_membership,
)


def run_example() -> dict:
    """Compare agreeing/conflicting fixtures and reject an undeclared criterion."""
    definition = CohortDefinition(
        "trial.synthetic",
        1,
        (CohortCriterion("trial.inclusion", CriterionKind.INCLUSION),),
    )
    record = CohortRecordEvidence(
        "sha256:" + "a" * 64,
        definition.definition_id,
        1,
        (
            CriterionEvidence(
                "trial.inclusion", EvidenceAssertion.MET, "sha256:" + "b" * 64
            ),
        ),
    )
    rule = explain_criterion_membership(record, definition)
    assessment = ModelCriterionAssessment(
        "trial.inclusion",
        CriterionState.MET,
        (EligibilityCitation("sha256:" + "c" * 64, 0, 12, "sha256:" + "b" * 64),),
        0.0,
        "sha256:" + "d" * 64,
    )
    agreed = build_trial_eligibility_review_packet(rule, (assessment,))
    conflict = build_trial_eligibility_review_packet(
        rule, (replace(assessment, state=CriterionState.NOT_MET),)
    )
    if agreed.requires_human_review or not conflict.requires_human_review:
        raise AssertionError("synthetic_review_routing_failed")
    if any(
        packet.authorizes_enrollment or packet.authorizes_contact
        for packet in (agreed, conflict)
    ):
        raise AssertionError("comparison_authorized_action")
    try:
        build_trial_eligibility_review_packet(
            rule, (replace(assessment, criterion_id="trial.undeclared"),)
        )
    except TrialEligibilityReviewError as error:
        if error.code != "unknown_criterion":
            raise
        failure_code = error.code
    else:
        raise AssertionError("undeclared_criterion_not_rejected")
    return {
        "workflow_id": "trial_eligibility_review",
        "passed": {"code": "agreement", "packet_digest": agreed.packet_digest},
        "review_required": {
            "packet_digest": conflict.packet_digest,
            "causes": [
                cause.value for item in conflict.comparisons for cause in item.causes
            ],
        },
        "fail_closed": {"code": failure_code},
    }


def main() -> dict:
    """Print deterministic JSON containing only digests, IDs and codes."""
    report = run_example()
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return report


if __name__ == "__main__":
    main()
