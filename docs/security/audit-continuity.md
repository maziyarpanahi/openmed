# Audit continuity across rotation, retention, and restore

A [tamper-evident audit chain](tamper-evident-audit-log.md) proves that its
retained entries still link to each other. On its own it cannot prove that
nothing was silently removed when entries aged out, that a chain restored from a
backup is the chain that was backed up, or which signing key vouched for which
part of the history. `openmed.core.audit_continuity` adds a versioned
*continuity manifest* beside the chain that records those facts, and an offline
verifier that checks them and fails closed with a stable, value-free reason
code.

The manifest and verifier make no network calls and never hold audit payloads,
source text, identifiers, or key material.

## The continuity manifest

The manifest (`openmed.audit-continuity`, version 1) is a JSON document with an
opaque `chain_id`, the chain format and genesis hash it is bound to, the digest
and signature algorithms, and an ordered, hash-linked list of events. Every
event commits to the previous event digest, to the chain identifier, and to the
chain state when it was written: the next sequence number and the head hash.

| Event | Records | Signed by |
| --- | --- | --- |
| `genesis` | Opens signer epoch 0 with a key ID | the epoch 0 key |
| `anchor` | A periodic signed digest of the chain head | the active key |
| `rotation` | Closes the active epoch and opens the next one | the outgoing **and** incoming keys |
| `retention` | A pruning tombstone (below) | the active key |
| `restore` | A restore from backup; opens a restore epoch | the restore epoch key |

The header also carries a derived summary of the signer epochs. Each epoch has
a key ID, the event that opened it, and a sequence validity window
`[start_sequence, end_sequence)`; the active epoch has no end. Rotation windows
are contiguous, so every chain entry belongs to exactly one epoch and the
rotation event signs the boundary with both keys: there is no unsigned gap.

### Pruning tombstones

A `retention` event is a non-reversible tombstone for one pruned segment: the
first and last sequence, the record count, the hash the first pruned entry
linked to, the hash of the last pruned entry, a digest of the segment's entry
hashes, and the fingerprint of the approving retention policy (plus the
integrity digest of the approving retention report, when one is supplied).
It holds no entry content. Tombstones must be contiguous from sequence 0, and
the pruned chain must link to the last tombstone's boundary hash.

A pruned chain document uses `openmed.audit-chain` version 2, which adds
`base_sequence` and `base_hash`. Unpruned chains keep the version 1 layout.

## Use the store

`AuditContinuityStore` keeps the chain and the manifest in step. Key custody
stays with the caller, exactly as for
[audit-report signing key rotation](audit-key-rotation.md): the provider is a
callback or mapping that resolves a non-secret key ID.

```python
from openmed.core.audit_continuity import AuditContinuityStore
from openmed.risk.audit_retention import AuditRetentionPolicy, RetentionRule

store = AuditContinuityStore(
    "audit/chain-0001.json",
    key_provider=caller_owned_key_provider,
    anchor_interval=500,
)
store.initialize(key_id="audit-2026")

store.append(report)              # commit an AuditReport to the chain
store.anchor()                    # sign the current chain head
store.rotate("audit-2027")        # atomic rollover to a new signing key

policy = AuditRetentionPolicy(rules={"audit": RetentionRule.days(365)})
store.prune_through(last_expired_sequence, policy=policy)

report = store.verify()
assert report.valid, report.code
evidence = report.to_json()       # bounded, value-free release evidence
```

The manifest defaults to `<chain stem>.continuity.json` beside the chain.
`initialize` adopts an existing valid chain; the genesis event commits to its
current head.

`prune_through` requires an approving policy: a `policy_fingerprint`, an
`AuditRetentionPolicy`, or the `AuditRetentionReport` returned by
`scrub_audit_artifacts`. Supplying approvals that disagree fails with
`policy_mismatch`. Pruning is never skipped to keep the chain verifiable; the
tombstone is what keeps it verifiable.

### Concurrency and crash recovery

Every store operation holds an exclusive lock for its whole read-modify-write
cycle: a process-local lock plus an OS file lock on a permanent
`.<manifest>.lock` file beside the manifest. Each document is written to a
same-directory temporary file, flushed, fsynced, renamed into place, and the
directory is fsynced. A rotation therefore commits to the exact chain head at
the moment of rollover even when other threads or processes are appending, and
concurrent requests to rotate to the same key produce one transition: rotating
to the active key returns the event that opened its epoch.

Pruning is write-ahead. The signed tombstone is made durable before the chain
is rewritten. If the process stops in between, verification reports
`incomplete_transition`, and the next store operation or `store.recover()`
checks the still-present entries against the tombstone and completes the
pruning deterministically. If those entries no longer match, recovery refuses
with `tombstone_mismatch`. `recover()` also removes stale temporary files.

### Restores

After restoring the chain (and the manifest, when it was backed up) and before
writers resume, call the restore hook:

```python
from openmed.core.audit_continuity import record_restore_transition

record_restore_transition(
    "audit/chain-0001.json",
    key_provider=caller_owned_key_provider,
    backup_digest=recovery_point_manifest_digest,
)
```

Pruning recorded in the manifest is re-applied first, so a restore never brings
pruned entries back. The restore event opens a new signer epoch and commits to
the restored chain head. If only the chain was rolled back and the manifest is
the live one, chain-head commitments from the lost timeline are reported as
superseded rather than as failures; earlier commitments must still match, so a
restored chain that forked from the recorded history is refused.

`AuditContinuityRecoveryHook` adapts the same call to a recovery workflow that
calls `open_recovery_epoch(context)` with a context exposing `manifest_digest`
and `recovery_point_id`. Point it at the staged chain so the restore event is
part of the state that becomes live. Back up the continuity manifest together
with its chain.

## Verify offline

```python
from openmed.core.audit_chain import AuditChain
from openmed.core.audit_continuity import ContinuityManifest, verify_continuity

report = verify_continuity(
    AuditChain.load("audit/chain-0001.json"),
    ContinuityManifest.load("audit/chain-0001.continuity.json"),
    key_provider=verification_key_provider,
    trusted_anchors=retained_head_refs,
    allowed_policy_fingerprints=[policy.fingerprint],
)
```

The report contains only versions, counts, digests, the opaque chain ID, key
IDs, epoch windows, policy fingerprints, and a `code`. It is never partially
valid: `report.valid` is true only when every check passed. Chain entries
appended after the last signed event fail with `unanchored_tail` unless
`allow_unanchored_tail=True` is passed; they are then counted in
`unanchored_entry_count`.

| Code | Meaning |
| --- | --- |
| `valid` | Every check passed. |
| `malformed_manifest` | The manifest has unknown fields, wrong types, or invalid JSON. |
| `unsupported_version` | Unknown manifest format, version, or algorithm. |
| `chain_mismatch` | The manifest belongs to another chain or chain format. |
| `chain_invalid` | The audit chain itself does not verify. |
| `event_gap` | An event is missing, or the declared event count disagrees. |
| `event_duplicate` | An event index appears twice. |
| `event_reordered` | Events are out of order. |
| `fork` | An event does not link to its predecessor, history moved backwards without a restore, or a trusted anchor diverged. |
| `epoch_mismatch` | Epoch numbering, an epoch-opening event, or the epoch summary was altered. |
| `unsigned_epoch_transition` | A rotation or restore lacks a required signature. |
| `signature_invalid` | A signature does not verify. |
| `unknown_signer_key` | An event is signed by a key that never opened an epoch. |
| `retired_signer_key` | An event is signed by, or a new epoch reuses, a retired epoch key. |
| `signer_key_unavailable` | The provider cannot resolve a key that must be verified. |
| `anchor_mismatch` | A signed chain-head commitment no longer matches the chain, or an anchor was altered. |
| `anchor_missing` | A trusted anchor is no longer in the manifest. |
| `sequence_gap` | Chain records were deleted, or pruning is not explained by tombstones. |
| `tombstone_mismatch` | A tombstone was altered or disagrees with the chain or earlier anchors. |
| `incomplete_transition` | A recorded pruning has not been applied yet; run `recover()`. |
| `policy_mismatch` | A tombstone names a retention policy that is not allowed. |
| `unanchored_tail` | Chain entries were appended after the last signed event. |

### Trusted anchors and retiring keys

Export `manifest.head_ref()` (an event index and digest) into release or
compliance evidence. Passing those references back as `trusted_anchors`
detects a manifest that was truncated (`anchor_missing`) or rewritten (`fork`).
It also lets a retired key be removed from the verification provider once a
trusted reference covers its epochs: the event digests up to the newest matching
reference are authenticated by the hash links, so signatures at or before it
whose keys are no longer available are counted in `unverified_signature_count`
instead of failing. Without such a reference, a missing key fails with
`signer_key_unavailable`.

## Security boundary

Signatures are HMAC-SHA256 over domain-separated, per-purpose derived keys, so
a continuity signature can never be replayed as an audit-report signature or
the other way round. HMAC is symmetric: verification proves integrity to
holders of the provider, not public non-repudiation. The manifest header is not
signed on its own; retain head references outside the manifest when truncation
of the newest events matters. Choosing retention periods, key custody, and
backup storage remain caller responsibilities.

This continuity layer covers the de-identification audit chain in
`openmed.core.audit_chain`. The separate hash-chained log used by GDPR and Part
11 workflows in `openmed.compliance.audit_chain` is not covered yet.
