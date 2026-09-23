# Reviewer handoff packets

`openmed.agent.ReviewerHandoffPacket` carries a strict, metadata-only request
from an agent workflow to a human reviewer. Every valid packet requires human
review. A packet never approves, authorizes, performs, or finalizes a clinical
action.

## Packet contract

The version 1 packet contains only:

- an opaque `RunId` and canonical `WorkflowId`;
- an abstention or review-required reason code from the stable workflow outcome
  vocabulary;
- one closed `RequestedDecision` value;
- ordered, content-free `ArtifactReference` evidence; and
- canonical whole-second UTC issue and expiry timestamps.

The requested decisions are `confirm_abstention`, `review_evidence`,
`resolve_evidence_conflict`, `assess_safety`, and `decide_next_step`. They state
the question for a human reviewer; they are not approval tokens.

```python
from datetime import UTC, datetime, timedelta

from openmed.agent import (
    ArtifactKind,
    ArtifactReference,
    RequestedDecision,
    ReviewerHandoffPacket,
    RunId,
    WorkflowId,
)

issued_at = datetime.now(UTC).replace(microsecond=0)
packet = ReviewerHandoffPacket(
    run_id=RunId.generate(),
    workflow_id=WorkflowId("workflow:org.example/document-review@1.0.0"),
    reason_code="conflicting_evidence",
    requested_decision=RequestedDecision.RESOLVE_EVIDENCE_CONFLICT,
    evidence_references=(
        ArtifactReference(
            artifact_id="art_" + "1" * 32,
            kind=ArtifactKind.EVIDENCE,
            schema_id="example.review.evidence.v1",
            sha256="a" * 64,
            byte_size=128,
        ),
    ),
    issued_at=issued_at,
    expires_at=issued_at + timedelta(minutes=30),
)

payload = packet.to_json()
```

Use `from_dict()` or `from_json()` at a trust boundary. Parsing rejects unknown
or duplicate fields, free-text payloads, malformed identifiers and digests,
duplicate evidence IDs, non-canonical timestamps, and packets whose expiry is
not after both their issue time and the current time. Evidence order is
preserved during deterministic round trips.

## Privacy and authority boundary

Packets may contain identifiers, hashes, timestamps, bounded categories, and
artifact byte sizes only. Do not add prompts, clinical notes, evidence text,
paths, URLs, credentials, patient identifiers, reviewer comments, or arbitrary
status strings. Validation errors contain only stable codes and public field
names, never rejected values.

Validation is local and performs no file, network, notification, authentication,
or clinical action. `requires_human_review` is always true and
`authorizes_clinical_action` is always false. A separate, explicitly governed
system must authenticate a reviewer and apply any later decision.

Run the focused offline tests with:

```text
.venv/bin/python -m pytest tests/unit/agent/test_reviewer_handoff.py -q
```
