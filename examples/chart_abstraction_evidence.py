"""Run synthetic chart-abstraction finalization offline.

Run from the repository root: python -m examples.chart_abstraction_evidence
The receipt is evidence metadata, not authorization for clinical action.
"""

import json
from dataclasses import replace

from openmed.agent.workflows import (
    AbstractionEvidenceChain,
    AbstractionEvidenceError,
    ChartAbstractionEvidence,
    ReviewerState,
    SourceLocation,
    TransformationKind,
)


def run_example() -> dict:
    """Finalize an approved chain and reject the same chain pending review."""
    chain = AbstractionEvidenceChain(
        "registry.synthetic_field",
        (SourceLocation("sha256:" + "a" * 64, 0, 12),),
        "sha256:" + "b" * 64,
        TransformationKind.RULE,
        "sha256:" + "c" * 64,
        0.0,
        ReviewerState.APPROVED,
    )
    required = (chain.field_id,)
    receipt = ChartAbstractionEvidence((chain,)).finalize(required)
    pending = ChartAbstractionEvidence(
        (replace(chain, reviewer_state=ReviewerState.PENDING),)
    )
    try:
        pending.finalize(required)
    except AbstractionEvidenceError as error:
        if error.code != "evidence_not_finalizable" or error.report is None:
            raise
        failure = {
            "code": error.code,
            "report_digest": error.report.report_digest,
            "issues": [issue.to_dict() for issue in error.report.issues],
        }
    else:
        raise AssertionError("pending_review_not_rejected")
    return {
        "workflow_id": "chart_abstraction_evidence",
        "passed": {"code": "finalized", "report_digest": receipt.report_digest},
        "fail_closed": failure,
    }


def main() -> dict:
    """Print deterministic JSON containing only digests, IDs and codes."""
    report = run_example()
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return report


if __name__ == "__main__":
    main()
