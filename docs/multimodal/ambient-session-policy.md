# Ambient consent and session lifetime

`AmbientSession` provides the same consent, capture-state and owned-memory contract in Python and OpenMedKit. Capture starts only after every declared participant has consented to the host's current policy. Pauses and interruptions are visible states; resuming requires an explicit host action. A stopped session cannot restart.

This is a local lifecycle component, not a recording-law assessment or a microphone implementation. The application acquires consent, supplies the complete participant set, presents capture state to users and stops its capture or inference producers whenever admission changes. The OS capture binding is a separate component. No cloud fallback, automatic note finalization or EHR submission is provided.

## Consent and reference lifetime

Use random opaque participant and session references, never names, patient identifiers or recording paths. The policy revision is the SHA-256 of the complete policy approved by the host. The library checks the identity's format and resource limits; it does not decide whether the host's policy or consent acquisition meets legal requirements.

Consent is bound to the current session, participant cohort and policy. Changing participants overwrites all owned audio, transcript, cache and draft buffers, invalidates their original references and requires fresh consent from every participant. Withdrawal is terminal and performs the same invalidation immediately. A new policy requires a new session. Serialized state is diagnostic only and cannot restore consent or reactivate references.

| Action | Required state | Result |
| --- | --- | --- |
| Consent | Awaiting consent | Ready after every participant consents |
| Start | Ready | Capturing |
| Pause | Capturing | Paused; new audio refused |
| Interrupt | Ready, capturing or paused | Interrupted; content use and new content refused |
| Resume | Paused or interrupted, with valid consent | Capturing |
| Stop | Ready, capturing, paused or interrupted | Stopped; audio refused, review work permitted |
| Change participants | Active, before stop | Old data destroyed; new consent required |
| Withdraw | Any active state | Terminal withdrawal and reference invalidation |
| Deadline or invalid clock | Any active state | Terminal expiry and reference invalidation |
| Delete | Any state | Owned data destroyed; external copies keep deletion incomplete |

Every session has one non-renewable monotonic deadline, including the time spent obtaining consent. Default timers enforce idle expiry while the process runs; host suspension or process death is outside that timer's guarantee. Lifecycle callbacks should stop producers and call `expire()` on wake before any further use. If automatic expiry is disabled for an embedded event loop, that loop must schedule expiry even while idle. Destruction on process death and persisted-file recovery require the existing [audio retention](streaming/audio-retention.md) component and host lifecycle handling.

## Owned buffers and deletion evidence

`retain` copies a bounded payload into owned mutable memory and returns an original-object capability. The caller remains responsible for clearing its input copy. Audio can be admitted only while capturing; transcript, cache and draft bytes can be admitted while capturing, paused or stopped. References from another session, reconstructed references and references from an invalidated cohort are refused.

`use` borrows a read-only Python memory view or a Swift raw buffer for a short synchronous local callback. The callback must not retain the view, copy or log its contents, perform network operations or block indefinitely. It runs under the session lock; this API cannot preempt arbitrary host code. A failed callback destroys pending owned data, and a callback that crosses the deadline cannot return its result through the session.

Withdrawal, deletion, participant changes, limits and expiry overwrite every owned buffer before releasing references. `deletion_verified` attests only to the declared inventory's owned-memory overwrite and absence of declared external copies. It is not proof of physical RAM erasure, discovery of unknown copies, erasure of caller inputs, a cleared OS microphone buffer or destruction of external model caches. Register any retained external copy with `declare_external_copy()` / `declareExternalCopy()`; the session then refuses to claim complete deletion. The application must inventory and clean external producers, caches and files separately. Existing audio-retention receipts cover that component's own enrolled buffers and files and must not be replaced by this session snapshot.

## Review exports remain drafts

Review export is disabled by default. With policy opt-in, stopped capture and valid unexpired consent, `export_for_review` / `exportForReview` copies a non-audio artifact for local human review. Every copy remains `draft_only` and `requires_review`. Audio export is refused. Exported bytes leave session ownership and are counted as unaccounted external copies; subsequent deletion remains incomplete even if the application later claims to have deleted those copies.

This lifecycle gate does not replace [PHI withholding](streaming/phi-withholding-buffer.md), [speaker-role review](streaming/ambient-speaker-roles.md), [transcript lineage](streaming/transcript-revisions.md) or [cited note review](streaming/ambient-note-evidence.md). Compose those components before consequential use. A generic draft byte buffer is not evidence that a note has passed their clinical or privacy gates.

## Python example

```python
import hashlib
import uuid

from openmed.multimodal.streaming.session import (
    AmbientArtifactKind,
    AmbientSession,
    AmbientSessionPolicy,
)

# Synthetic policy for this example; the host supplies its actual approved document.
policy_document = b'{"purpose":"synthetic local review","retain_seconds":300}'
participant = uuid.uuid4().hex
policy = AmbientSessionPolicy(revision=hashlib.sha256(policy_document).hexdigest())

with AmbientSession(uuid.uuid4().hex, (participant,), policy=policy) as session:
    session.consent(participant)
    session.transition("start")
    source = bytearray(b"synthetic draft")
    draft = session.retain(uuid.uuid4().hex, AmbientArtifactKind.DRAFT, source)
    source[:] = b"\0" * len(source)
    assert session.use(draft, len) == 15
    session.transition("interrupt")
    session.transition("resume")
    session.transition("stop")
    session.withdraw()
    assert session.snapshot()["deletion_verified"]
```

## OpenMedKit example

```swift
let policy = try AmbientSessionPolicy(revision: approvedPolicySHA256)
let session = try AmbientSession(
    sessionRef: opaqueSessionRef,
    participants: opaqueParticipantRefs,
    policy: policy)
defer { session.delete() }
for participant in opaqueParticipantRefs {
    try session.consent(participant)
}
try session.transition("start")
// Bind the application's capture permission to session.snapshot().state.
try session.transition("interrupt")
// Resume only after an explicit user action and rechecking host permissions.
try session.transition("resume")
session.withdraw()
assert(session.snapshot().deletionVerified)
```

## Shared validation

Python and Swift replay `tests/fixtures/multimodal/ambient_session_transitions.json` and compare complete serialized state after every accepted and refused action. The fixture contains only opaque references, kinds, actions and counts. Additional controls exercise actual owned-buffer overwrite, stale and foreign capabilities, original deadlines, malformed policies, retention limits, callback failure, external-copy accounting and draft-only export. No clinical audio, transcript content or model assets are required.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/streaming/test_session_policy.py -q
cd swift/OpenMedKit && swift test
```
