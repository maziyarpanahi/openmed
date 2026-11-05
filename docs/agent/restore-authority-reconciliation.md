# Restore authority reconciliation

A verified, schema-compatible backup proves that bytes were preserved. It does
not prove that the authority recorded in those bytes is still usable. A backup
taken before an approval token was spent still holds a nonce store with no claim
for that token. A snapshot taken before a revocation still records the grant as
active. `openmed.agent.approvals.restore_reconciliation` adds an explicit
anti-replay boundary between restore and admission. It reconciles the restored
generation against current, injected authority before any protected work
resumes.

This slice owns reconciliation only. Backup format, encryption, integrity
checks and identity services stay with their own components. The module reuses
the existing approval token, nonce store, runtime revocation and workflow
recovery contracts. It performs no network access.

## Inputs

`RestoredSnapshot` is a content-free view of one restored generation:

- `snapshot_digest`: opaque digest of the backup generation.
- `recorded_epoch`: recovery epoch recorded in the backup.
- `approvals`: `RestoredApproval` records for tokens outstanding at backup time.
  `RestoredApproval.from_token(token, clock_skew_seconds=...)` keeps only the
  token digest, nonce digest and verifier retention bound, never the bearer
  token. Pass the live verifier's skew so the burn covers its full acceptance
  window.
- `workflows`: `RestoredWorkflow` entries holding a validated checkpoint
  lineage, or a sealed `RetiredJournalRecord`, plus the `AuthorityBinding` the
  run recorded at preview. Use `authority=None` only for runs that use no
  runtime-revocable authority.

Everything in the snapshot is historical evidence. Nothing in it establishes
current authority.

`RestoreReconciler` takes the authoritative current state as injected ports:

| Port | Contract |
| --- | --- |
| `nonce_store` | The live `ApprovalNonceStore` shared with every verifier |
| `epochs` | A `RecoveryEpochStore` with monotonic high water, never restored from the same backup |
| `authority` | An `AuthorityRuntime` with the current status provider and independent generation store |
| `clock` | Trusted integer Unix clock |
| `observe` | Read-only effect status lookup by idempotency key for one restored tip |

`InMemoryRecoveryEpochStore` serves single-process use and tests. Production
restores need a durable epoch store kept outside every backup. A store restored
with the run state could rewind epochs.

## Reconciliation order

`reconcile(snapshot, observe=...)` runs these steps in a fixed order:

1. **Open a recovery epoch.** The new epoch is strictly above both the
   recorded epoch and the retained high water. `RecoveryEpoch` binds the epoch,
   its predecessor, the snapshot digest and the time into `epoch_digest`. If the
   store fails or rewinds, the call fails with `recovery_epoch_unavailable` or
   `epoch_not_monotonic`, and nothing is admitted.
2. **Burn pre-restore approvals.** Each unexpired snapshot nonce is claimed in
   the live nonce store, using the same digest and retention bound as
   `ApprovalTokenVerifier`. After this step the token raises
   `ApprovalReplayError`, even when the live store is the restored
   pre-consumption copy. The outcome is `pre_restore_nonce_burned`, or
   `pre_restore_nonce_already_claimed` when the authoritative store already
   held the claim. Expired approvals are `approval_expired`. A store failure
   holds the approval with `nonce_journal_unavailable`.
3. **Check current authority.** A workflow that recorded a binding is checked
   at the `resume` boundary with fresh status, before any sink read. Revoked,
   expired, changed-generation, rolled-back and mismatched authority stay
   unusable even when the snapshot recorded an earlier active state. The
   outcome is `fresh_review_required`. An unavailable, stale or
   store-unavailable status, or a missing runtime, keeps the workflow `held`.
4. **Reconcile effects.** The existing `recover_workflow` planner compares the
   restored lineage with live sink observations. Effects committed after the
   backup are reported as committed and are never repeated. Ambiguous or
   mismatched effects need `review_required`. A failed or malformed lookup
   keeps the workflow `held`.
5. **Invalidate pre-restore approval receipts.** A resumable plan whose
   pending retries need approval becomes `fresh_review_required` with
   `pre_restore_approval_invalidated`. The checkpoint receipt digest is kept as
   `historical_approval_receipt_digest`. It never authorizes a retry. Only
   plans without approval-gated retries return `resume` with idempotency keys.
   A missing or expired recorded receipt also needs `fresh_review_required`.

Completed, aborted and retired lineages skip authority and sink lookups,
because they admit no further effect.

## Epoch-bound approvals

Burning covers tokens the snapshot recorded. A token issued after the backup is
not in the snapshot. To cover it, issue and verify every approval against
`epoch_action_digest(action_digest, epoch)`. A token signed for an earlier epoch
then fails as an action mismatch and is consumed. Before admission, call
`require_current_epoch(epoch, epochs)`. It fails with `stale_recovery_epoch`
after a later restore.

```python
from openmed.agent.approvals import (
    ApprovalReplayError,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.agent.approvals.restore_reconciliation import (
    InMemoryRecoveryEpochStore,
    RestoredApproval,
    RestoredSnapshot,
    RestoreReconciler,
    epoch_action_digest,
    require_current_epoch,
)

key = b"synthetic-local-approval-key-32-bytes"
action = "sha256:" + "a" * 64
role = "role:org.example/clinical-reviewer@1.0.0"
signer = ApprovalTokenSigner(key, clock=lambda: 10)
token = signer.issue(action_digest=action, reviewer_role=role, expires_at=500)

# The live store was replaced by a pre-consumption backup copy.
restored_store = InMemoryApprovalNonceStore()
epochs = InMemoryRecoveryEpochStore()
snapshot = RestoredSnapshot(
    snapshot_digest="sha256:" + "b" * 64,
    recorded_epoch=0,
    approvals=(RestoredApproval.from_token(token),),
    workflows=(),
)
report = RestoreReconciler(restored_store, epochs, clock=lambda: 10).reconcile(
    snapshot, observe=lambda checkpoint: ()
)
report.require_complete()
assert report.approvals[0].reason.value == "pre_restore_nonce_burned"

verifier = ApprovalTokenVerifier(key, restored_store, clock=lambda: 10)
try:
    verifier.consume(token, action_digest=action, reviewer_role=role)
except ApprovalReplayError:
    pass
else:
    raise AssertionError("restored token must stay spent")

require_current_epoch(report.epoch, epochs)
bound = epoch_action_digest(action, report.epoch)
fresh = signer.issue(action_digest=bound, reviewer_role=role, expires_at=500)
verifier.consume(fresh, action_digest=bound, reviewer_role=role)
```

## Admission and evidence

`RestoreReconciliationReport.require_complete()` raises
`restore_reconciliation_incomplete` while any approval or workflow is held.
Hosts must not admit restored work until a later run completes. Reconciliation
is idempotent: rerunning it opens a newer epoch and never revives a burned
nonce.

Restore never erases committed effects. Committed effects recorded in the
snapshot stay in the outcome even when authority is denied or unavailable.
Journals and sinks are only read. Restore also never claims that review was
repeated. Its outcomes contain no approval-success code, and
`fresh_review_required` means a new human review is still needed. A resumable
plan still needs the normal read and dispatch checks at the next operation.

`to_json()` emits only the schema version, epoch record, controlled reason
codes, counts, opaque action identifiers, idempotency keys and digests. It never
includes bearer tokens, nonces, provider or store exception text, paths or
source payloads. Keep these linkable digests under audit-data access controls.

This slice adds no autonomous clinical action, cloud fallback, identity service
or backup format. It covers the Python contracts only; no matching Swift API is
introduced.

Focused offline validation:

```text
.venv/bin/python -m pytest tests/unit/agent/approvals/test_restore_reconciliation.py tests/integration/agent/test_restore_authority_reconciliation.py -q
```
