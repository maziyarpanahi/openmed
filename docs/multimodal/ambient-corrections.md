# Ambient reviewer corrections

`openmed.multimodal.ambient_corrections` and OpenMedKit's
`AmbientCorrectionLedger` provide local, in-memory correction propagation over
caller-supplied finalized evidence and draft claim fixtures. They add no ASR,
text editor, note generation, clinical decision, EHR write, or network call.
This slice is independent of the pending transcript lineage, ambient assembly,
and speaker-role work in issues #2814, #2816, and #2817. It does not define or
claim integration with their future provider contracts.

## Evidence and authority boundary

Supply immutable evidence units with random opaque UUIDs, revision numbers,
and `sha256:` digests, and claims with **complete direct citation dependencies**.
An evidence unit must cover the context on which a claim depends, including
medication, negation, and speaker attribution. If several units change, include
all of them in one atomic correction. The ledger cannot discover omitted
citations, map edited text offsets, or prove that a changed digest represents a
correct human edit. The caller must enforce those responsibilities before
submitting an event. Stable evidence units avoid silently remapping offsets.

Protected transcript text, corrected text, speaker names, clinical note content,
credentials, file paths, and free-form notes stay outside the ledger. Use
caller-keyed digests of protected values to resist low-entropy dictionary attacks;
the API validates digest shape, not its provenance. Diagnostics use only the
closed `ConflictCode` / `AmbientReviewConflict` codes. Default graph, event,
representation, and reviewed exports contain references, digests, counters, and
controlled metadata, never source payloads. No telemetry or persistence is
performed. Handle the caller's protected content separately with local privacy
controls; metadata review does not de-identify a note.

Inject a trusted local authority callback `(reviewer_id, action) -> bool` in
Python or `(reviewerID, action) throws -> Bool` in Swift. Handles must be opaque
UUIDs, never account names. Grants are checked independently for `correct`,
`verify`, `approve`, and `export`. Python requires a literal `True` grant;
callback errors fail with `authority_required`, without their private messages.
This is an application trust boundary, not authentication: do not accept an
untrusted callback or let a client grant itself privileges.

## Revisions, invalidation, and replay

Every correction and review event has schema version 1, a unique event UUID,
an authority handle, the exact current draft/ledger revision, and the exact
current transcript revision. Initial transcript revision is caller-supplied
(default 0); initial ledger revision is 0. Each accepted event creates an
immutable child graph node. Corrections increment the transcript revision and
replace evidence using compare-and-set against the full prior evidence record.

Every claim citing any corrected unit keeps its claim UUID, receives updated
citations and a new claim revision, and loses **both** verification and approval.
Other citations and unrelated claim identities, revisions, verification links,
and approvals remain unchanged. Old evidence and review decisions remain in
historical nodes and the journal; they are never treated as current approval.
Re-verifying a claim also clears its previous approval.

The caller submits a separate `verify` event with a digest of **actual local
verification evidence**, then an `approve` event with explicit human confirmation.
Neither event mints a model result or clinical validation. The authority must
permit those actions; the verifier's correctness remains the caller's duty.
An approval without fresh verification fails with `verification_required`.

Duplicate event IDs, stale draft/transcript/claim revisions, and stale evidence
produce typed review conflicts. Unknown references and malformed metadata fail
closed. Multi-unit corrections validate completely before mutation. All calls
must be serialized by the application; this is a single-writer ledger. A stale
edit must be reassessed against the current node rather than blindly retried.

Replay supplies identical initial evidence, claims, transcript revision, and the
ordered `events` journal to `AmbientCorrectionLedger.replay`. It reconstructs
an identical graph and rechecks every authority grant. Replay with revoked or
missing authority fails closed. There is no persisted journal parser or transport
in this slice; applications retain typed immutable events locally.

## Python example

This metadata-only example uses synthetic opaque references and placeholder
digests. Applications obtain digests from protected local content and actual
verification results.

```python
from openmed.multimodal.ambient_corrections import (
    AmbientCorrectionLedger, CorrectionEvent, CorrectionKind,
    DraftClaim, EvidenceChange, EvidenceRevision,
)

reviewer = "00000000-0000-4000-8000-000000000900"
evidence = EvidenceRevision(
    "00000000-0000-4000-8000-000000000001", 0, "sha256:" + "a" * 64
)
claim = DraftClaim("00000000-0000-4000-8000-000000000002", (evidence,))
ledger = AmbientCorrectionLedger(
    (evidence,), (claim,), transcript_revision=17,
    authority=lambda handle, action: handle == reviewer and action == "correct",
)
ledger.apply(CorrectionEvent(
    event_id="00000000-0000-4000-8000-000000000500",
    reviewer_id=reviewer, base_revision=0, transcript_revision=17,
    kind=CorrectionKind.NEGATION,
    changes=(EvidenceChange(evidence, "sha256:" + "b" * 64),),
))
assert ledger.current.transcript_revision == 18
assert ledger.current.claims[0].approval_event is None
```

Swift uses the same semantics through `AmbientEvidenceRevision`,
`AmbientDraftClaim`, `AmbientEventBinding`, `AmbientCorrectionEvent`,
`AmbientClaimReviewEvent`, and `AmbientLedgerEvent`. Construct records with
throwing initializers, submit `.correction(...)` or `.review(...)` to `apply`,
and retain `graph` and `events` for local replay. Both surfaces reject invalid
metadata and retain the same parent, transcript, evidence, and claim revisions.

## Consequential output guard

`export_reviewed` / `exportReviewed` requires the **exact current ledger
revision**, verified and approved claims, export authority, and explicit reviewer
confirmation for that output. Historical nodes cannot be exported as currently
reviewed, even when all their claims were approved at the time. A previously
returned export is a historical snapshot; consumers must obtain a new guarded
export at the point of use rather than relying on a cached approval flag.

Every guarded output includes `non_diagnostic=true`, `reviewer_confirmed=true`,
and the notice: **For clinician review; not a diagnosis or treatment
recommendation.** Consumers must display the notice with the consequential
output and bind their local protected content to the returned exact revision.
No clinical action is auto-triggered. The ledger performs no model inference,
provider qualification, benchmark, clinical validation, or cloud fallback.

## Offline validation

Synthetic unit tests cover shared medication, negation and speaker dependencies,
unrelated citations, stale and atomic edits, authority failures, explicit
confirmation, historical export rejection, privacy-safe reports, and deterministic
replay. An offline integration test composes correction, authority, trusted
verification evidence, human review, export, and replay. Matching OpenMedKit
unit tests exercise the native boundary without providers or model assets.
