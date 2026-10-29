"""Run synthetic prior-authorization completeness checks offline.

Run from the repository root: python -m examples.prior_auth_completeness
Structural completeness is not a coverage decision or clinical approval.
"""

import json

from openmed.agent.workflows import (
    PacketEvidence,
    PriorAuthCompletenessError,
    PriorAuthorizationPacket,
    PriorAuthRequirement,
    PriorAuthRequirementSchema,
    score_prior_authorization_packet,
)


def run_example() -> dict:
    """Build synthetic evidence and reject a mismatched schema version."""
    schema = PriorAuthRequirementSchema(
        "payer.synthetic", 1, (PriorAuthRequirement("clinical.requested_service"),)
    )
    evidence = (
        PacketEvidence(
            "clinical.requested_service", "sha256:" + "b" * 64, ("sha256:" + "c" * 64,)
        ),
    )
    packet = PriorAuthorizationPacket(
        "sha256:" + "a" * 64, schema.schema_id, 1, evidence
    )
    report = score_prior_authorization_packet(packet, schema)
    if report.requires_reviewer_action or report.completeness_score != 1.0:
        raise AssertionError("synthetic_completeness_failed")
    mismatched = PriorAuthorizationPacket(
        packet.packet_digest, schema.schema_id, 2, evidence
    )
    try:
        score_prior_authorization_packet(mismatched, schema)
    except PriorAuthCompletenessError as error:
        if error.code != "schema_version_mismatch":
            raise
        failure_code = error.code
    else:
        raise AssertionError("schema_mismatch_not_rejected")
    return {
        "workflow_id": "prior_auth_completeness",
        "passed": {"code": "complete", "report_digest": report.report_digest},
        "fail_closed": {"code": failure_code},
    }


def main() -> dict:
    """Print deterministic JSON containing only digests, IDs and codes."""
    report = run_example()
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return report


if __name__ == "__main__":
    main()
