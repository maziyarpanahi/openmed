# Session-scoped speaker aliases

`SpeakerAliases` assigns stable session-local references to opaque acoustic clusters. It receives explicit revision lineage from a local diarization adapter; it does not receive audio, voice embeddings, names or identity claims, and it does not infer matches from similar labels. Python session behavior is defined here; OpenMedKit contract parity is tracked separately in #3679.

Aliases such as `speaker_0001` identify a cluster lineage within one session. They are not patient identities, cross-session identifiers or clinical roles. Always retain the `session_ref` alongside an alias. Allocate session and cluster references independently of patient identifiers; a hexadecimal encoding of an identifier is not an opaque reference.

## Stable revisions and explicit lineage

```python
from openmed.multimodal.streaming.speech import (
    ClinicalRole,
    LineageEvent,
    LineageKind,
    RoleReview,
    SpeakerAliases,
)

# Synthetic references; adapters supply stable opaque cluster references.
session_ref = "a" * 32
first, second, renamed = (f"{i:032x}" for i in (1, 2, 3))
aliases = SpeakerAliases(session_ref)
aliases.apply_revision(0, (first, second))
original_alias = aliases.alias_for(first)

aliases.apply_revision(
    1,
    (second, renamed),
    (LineageEvent(kind=LineageKind.RELABEL, parents=(first,), children=(renamed,)),),
)
assert aliases.alias_for(renamed) == original_alias

# Construct this only after the application obtains explicit reviewer approval.
aliases.confirm_role(RoleReview(
    session_ref=session_ref,
    alias=original_alias,
    role=ClinicalRole.PATIENT,
    revision=1,
    review_ref="b" * 32,
))
report = aliases.to_json()
aliases.close()
```

Each revision supplies the complete set of currently active cluster references. Reordering that set does not change aliases. A cluster that is temporarily absent can return with its existing alias. An unseen cluster without a lineage event receives a new alias, even if another cluster disappeared. Adapters must translate unstable diarization labels into opaque references and explicit lineage; reusing a raw label after its meaning changes is not evidence of continuity.

| Event | Alias behavior | Role behavior |
| --- | --- | --- |
| Relabel: one parent, one new child | Keep the parent alias and retire the old cluster reference. | Retain an existing explicit approval for that lineage. |
| Merge: multiple parents, one new child | Keep the earliest allocated alias; record every parent alias and cluster. | Clear affected approvals and require fresh reviewer confirmation. |
| Split: one parent, multiple new children | Retire the ambiguous parent alias; allocate new aliases in sorted cluster-reference order. | Children start without roles; do not copy the parent's approval. |

Historical transcript references keep their original aliases. The lineage report records the relationship; consumers must not silently replace all earlier references with a merge survivor or arbitrarily choose a split child. Ambiguous downstream evidence requires review.

## Reviewer confirmation and privacy

Roles are absent by default. Only `confirm_role(RoleReview(...))` attaches a role, and the review must name the exact current revision, session and active alias. The application is responsible for authenticating the reviewer and retaining the underlying approval separately. A role attestation is not identity verification, speaker-role inference or authorization for a clinical decision. Names and identity claims must remain in separately protected application records.

Reports contain the session reference, revision, active aliases, opaque cluster lineage and explicit role-review references. They contain no names, transcript text, acoustic vectors, audio, paths or arbitrary diagnostic messages. References remain potentially linkable metadata and require the caller's access and retention controls. The registry never logs or performs network or file access. `close()` discards its alias, lineage and role mappings and rejects further updates; it does not promise forensic erasure of Python memory or delete data held by an adapter.

## Bounds and failure behavior

The registry accepts up to 256 active clusters, 4,096 cluster references over a session, 4,096 lineage events and revision numbers up to 1,000,000. Revisions strictly increase. Each event has disjoint parents and children; parents must exist and children must be new active references. A parent can participate in only one event per revision. A child cannot also be a parent in the same revision. Retired cluster references cannot be reused.

Updates are atomic: malformed or stale input does not consume alias numbers, change roles or partially apply an earlier event. Reaching a bound fails closed instead of evicting lineage or resetting identifiers. Callers must serialize access to each instance. Errors use a fixed value-free `SpeakerAliasError` message. The JSON version is `openmed.speech.aliases.v1`; identical session inputs and events produce identical output regardless of tuple ordering.

## Validation

```bash
.venv/bin/python -m pytest tests/unit/multimodal/streaming/speech/test_speaker_aliases.py -q
```

Synthetic controls cover reordering, absence and return, explicit relabels, merges, splits, stale and cross-session reviews, transactional failure, session bounds, closed sessions and value-free diagnostics. No speaker-recognition accuracy or clinical qualification is claimed.
