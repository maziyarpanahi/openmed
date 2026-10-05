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

## Revisioned handoff storage

`openmed.agent.handoff_store.SQLiteHandoffStore` provides a local, durable
implementation of the narrow `HandoffStore` protocol. It reuses
`structured.store`'s `StoreResult`/`StoreState` outcomes and the clinical
`ReviewState.APPROVED`/`REJECTED` vocabulary. Those states record the caller's
decision; acceptance by this store never issues an approval token, authenticates
a reviewer, evaluates clinical evidence, or authorizes an action.

Each `publish()` commits an opaque action reference, a monotonically increasing
revision, a caller-supplied action SHA-256, and SHA-256 commitments to the exact
packet and its ordered evidence references. Only commitments and expiry are
persisted; the packet and source evidence are not stored. The action digest must
commit to the complete proposed action in the caller's existing action contract.
The adapter does not invent or normalize that contract.

Use `expected_revision=0` to create an action, and the last observed revision to
correct it. A correction appends a new revision even when all digests are
identical. Old leases cannot decide the corrected revision, and previous
decisions remain historical. Concurrent corrections return `revision_conflict`
rather than overwriting another writer's revision.

`acquire()` permits multiple reviewers to hold leases for the same revision.
The default lease is 300 seconds; the default ceiling is 900 seconds, configurable
from 1 through 86,400 seconds. Every lease is capped by packet expiry. Reviewer
references must use `rev_` and role references `role_`, each followed by exactly
32 lowercase hexadecimal digits. Action references use `act_` with the same
suffix shape. These must be opaque references, not encoded names, patient IDs,
credentials, or free-text comments. Caller authentication and role assignment
remain outside the adapter. Lease IDs are generated locally.

`decide()` compares the submitted lease against the persisted lease, including
all three commitments, expiry, reviewer references and revision. A SQLite
`BEGIN IMMEDIATE` transaction and unique action/revision constraint make one
decision win across independent connections, threads and local processes.
Decision receipts, leases and revisions are append-only, including SQL triggers
that reject updates and deletes through ordinary SQL. No network call or new
dependency is needed.

| Condition | Outcome |
| --- | --- |
| First exact, current, unexpired decision | `success`, with a receipt |
| Another decision already accepted for that revision | `conflict` / `already_decided` |
| Action was corrected | `conflict` / `superseded` |
| Packet expired | Explicit `expired` refusal |
| Lease expired at or before the transaction clock | `denied` / `expired` |
| Lease was altered or not issued by this database | `denied` / `invalid_lease` |
| Optimistic publish revision differs | `conflict` / `revision_conflict` |
| Action not present | Explicit `not_found` refusal |
| Packet issue time is in the future | `denied` / `not_yet_valid` |

Supersession takes precedence over packet or lease expiry, which takes precedence
over `already_decided` for decision submissions. Malformed input and unavailable
storage raise `HandoffStoreError` with controlled diagnostics. Failed operations
do not append receipts. There is no successful replay or automatic lease renewal.

Using the synthetic `packet` from the example above:

```python
from openmed.agent.handoff_store import SQLiteHandoffStore
from openmed.clinical.review_state_machine import ReviewState

store = SQLiteHandoffStore(":memory:")  # Use a protected local file for durability.
try:
    action_id = "act_" + "1" * 32
    binding = store.publish(
        action_id, "a" * 64, packet, expected_revision=0
    ).value
    first = store.acquire(
        action_id, expected_revision=binding.revision,
        reviewer_role_ref="role_" + "2" * 32,
        reviewer_ref="rev_" + "3" * 32,
    ).value
    second = store.acquire(
        action_id, expected_revision=binding.revision,
        reviewer_role_ref="role_" + "4" * 32,
        reviewer_ref="rev_" + "5" * 32,
    ).value
    assert store.decide(first, ReviewState.APPROVED).ok
    assert store.decide(second, ReviewState.REJECTED).code == "already_decided"
    assert store.receipts(action_id)[0].authorizes_clinical_action is False
finally:
    store.close()
```

Successful receipts survive connection and process restart when using a file.
`receipts()` returns historical decisions, including superseded or expired ones;
it is never an approval lookup. `current()` refuses expired packets and returns
the latest exact binding without implying an accepted decision. Any later
authorization must independently verify the exact current action/evidence,
reviewer authority, expiry and its own approval policy at execution time.

Use a dedicated caller-owned SQLite database on a filesystem with reliable local
SQLite locking. Protect it from untrusted writers; these records are not signed
attestations or a defense against an administrator changing the database. The
adapter uses full SQLite synchronization, an injected UTC wall clock and a
persisted clock watermark. Clock rollback fails closed with `clock_regressed`
until the clock catches up, including after restart, so expired leases cannot be
resurrected by a backward clock adjustment. There is no pruning or remote
replication. Database paths and SQLite exception text are absent from diagnostics.

This slice owns the existing Python handoff concurrency boundary. OpenMedKit's
on-device governance projection and parity fixtures are separately tracked in
[#3669](https://github.com/maziyarpanahi/openmed/issues/3669).

The synthetic process-race test demonstrates one accepted decision and one
controlled conflict; it supplies no clinical-validation or release-readiness
claim. Run the complete focused slice with:

```text
.venv/bin/python -m pytest tests/unit/agent/test_reviewer_handoff.py tests/unit/agent/test_handoff_store.py tests/integration/agent/test_handoff_store_concurrency.py -q
```
