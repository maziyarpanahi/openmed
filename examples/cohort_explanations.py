"""Run synthetic criterion-level cohort explanations offline.

Run from the repository root: python -m examples.cohort_explanations
An eligible explanation never authorizes enrollment or contact.
"""

import json

from openmed.agent.workflows import (
    CohortCriterion,
    CohortDefinition,
    CohortExplanationError,
    CohortRecordEvidence,
    CriterionEvidence,
    CriterionKind,
    EvidenceAssertion,
    TimeWindowReference,
    explain_criterion_membership,
)


def run_example() -> dict:
    """Explain inclusion/exclusion evidence and reject a changed time window."""
    window = TimeWindowReference("window.synthetic", "sha256:" + "a" * 64)
    definition = CohortDefinition(
        "cohort.synthetic",
        1,
        (
            CohortCriterion("clinical.inclusion", CriterionKind.INCLUSION, window),
            CohortCriterion("clinical.exclusion", CriterionKind.EXCLUSION),
        ),
    )
    exclusion = CriterionEvidence(
        "clinical.exclusion", EvidenceAssertion.NOT_MET, "sha256:" + "b" * 64
    )
    inclusion = CriterionEvidence(
        "clinical.inclusion", EvidenceAssertion.MET, "sha256:" + "c" * 64, window
    )
    record = CohortRecordEvidence(
        "sha256:" + "d" * 64,
        definition.definition_id,
        definition.version,
        (inclusion, exclusion),
    )
    report = explain_criterion_membership(record, definition)
    if report.authorizes_enrollment or report.authorizes_contact:
        raise AssertionError("explanation_authorized_action")
    changed_window = TimeWindowReference(window.reference_id, "sha256:" + "e" * 64)
    mismatched = CohortRecordEvidence(
        record.record_digest,
        definition.definition_id,
        1,
        (
            CriterionEvidence(
                inclusion.criterion_id,
                inclusion.assertion,
                inclusion.evidence_digest,
                changed_window,
            ),
            exclusion,
        ),
    )
    try:
        explain_criterion_membership(mismatched, definition)
    except CohortExplanationError as error:
        if error.code != "time_window_mismatch":
            raise
        failure_code = error.code
    else:
        raise AssertionError("window_mismatch_not_rejected")
    return {
        "workflow_id": "cohort_explanations",
        "passed": {
            "code": report.membership_state.value,
            "report_digest": report.explanation_digest,
            "criteria": [
                {"criterion_id": item.criterion_id, "code": item.state.value}
                for item in report.criteria
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
