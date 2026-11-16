# Fail-closed required audit

`openmed.service.required_audit` defines what a service does when audit
evidence that is required for a state-changing operation cannot be recorded.
The rule: a declared operation never returns a fully evidenced success unless
its required audit records are durable and acknowledged. If an effect may
already have committed, the receipt stays reconcile-required with pending
evidence. The service never retries the effect blindly.

The module is local-first and adds no new audit backend. Required evidence goes
to the existing hash-chained [action ledger](../agent/action-ledger.md) through
`ActionLedgerAuditStore`. Reconciliation uses the same `ObservationState`
vocabulary as workflow effect recovery.

## Declare the operations that need required audit

```python
from openmed.agent.audit.action_ledger import ActionLedger
from openmed.agent.correlation import RunId
from openmed.service.required_audit import (
    ActionLedgerAuditStore,
    CommitBoundary,
    RequiredAuditCoordinator,
    RequiredAuditPolicy,
)

policy = RequiredAuditPolicy(
    {
        "fhir.write_back": CommitBoundary.INVOCATION,
        "omop.local_batch": CommitBoundary.ACKNOWLEDGED,
    },
    max_open_receipts=64,
)
store = ActionLedgerAuditStore(
    ActionLedger("private-state/required-audit"),
    run_id=RunId.generate(),
    operations=policy.operations,
)
coordinator = RequiredAuditCoordinator(policy, store)

receipt = coordinator.perform(
    "fhir.write_back",
    request_key=client_idempotency_token,
    operation_digest=payload_digest,
    effect=apply_write,  # called once with an opaque "idem_..." key
)
if not receipt.fully_evidenced:
    ...  # report receipt.outcome and receipt.reason; never claim success
```

The ledger directory is private state. Keep it on storage you back up together
with the effect sink. A deployment should reuse one `run_id` for the life of a
ledger directory, because the action ledger binds every entry to one run.

- Operations that are not declared are denied before anything is recorded.
- A declared operation has no optional mode. Without a required store, every
  declared operation is denied with `required_audit_unconfigured`.
- `request_key` is a caller-chosen opaque token of 16 to 128 characters from
  `[A-Za-z0-9_-]`. Only its digest goes into the receipt identifier. It must
  not encode a patient, encounter, or other identifier.
- `operation_digest` is a `sha256:` digest of the value-free payload. If the
  same request key comes back with a different digest, the request is denied
  with `idempotency_conflict`.

## Commit boundary

For each receipt, the coordinator appends these required records, in order:

| Stage | Ledger state | When |
|-------|--------------|------|
| `intent` | `proposed` | Admission, before anything else |
| `attempt` | `attempted` | Immediately before the effect is invoked |
| `commit` | `committed` | After the effect returns |
| `abort` | `rejected` | When the effect is known not to have committed |

The commit boundary is the effect invocation. A fault before the boundary
cannot have changed state, so the outcome is `denied`. After the boundary, the
outcome is `reconcile_required` unless the commit record is durable. The
declared `CommitBoundary` only says how much an adapter failure can be trusted:

- `INVOCATION`: an exception from the effect adapter means the outcome is
  unknown. The receipt becomes `reconcile_required` with
  `effect_outcome_unknown`.
- `ACKNOWLEDGED`: the adapter commits atomically when it returns, so an
  exception means nothing was committed. The coordinator records `abort` and
  returns `denied` with `effect_not_committed`.

| Fault | During `intent` or `attempt` | During `commit` |
|-------|------------------------------|-----------------|
| Storage full (`audit_storage_full`) | `denied` | `reconcile_required` |
| Rejected append (`audit_append_rejected`) | `denied` | `reconcile_required` |
| Interrupted acknowledgment (`audit_unavailable`) | `denied`, evidence pending | `reconcile_required` |

Stores raise `audit_storage_full` or `audit_append_rejected` only when the
record is known not to be durable. Any fault whose durability is unknown is
`audit_unavailable`. That includes a write that landed but whose
acknowledgment was lost, for example when the ledger directory sync fails after
the entry is linked. A store acknowledgment that is not a `sha256:` digest is
treated as unavailable.

The receipt's `evidence` field reports `complete`, `pending`, or `none`.
`fully_evidenced` is true only for a `completed` receipt whose `commit` record
is durable. In that case `evidence_digest` is the ledger entry digest of the
commit record.

## Reconcile without retrying the effect

`reconcile()` takes no effect adapter, so it cannot re-run an effect. It reads
the required store first:

- If a terminal record is already durable, for example after an interrupted
  acknowledgment, the receipt is returned as `reconciled_existing`. Nothing is
  appended and the observer is not queried.
- If only `intent` is durable, the effect was never invoked. The coordinator
  records `abort` without querying the observer.
- If `attempt` is durable, the read-only observer is queried by the opaque
  effect key. `committed` appends the missing `commit` record once. `absent`
  appends `abort`. `ambiguous`, an observer failure, or an invalid answer
  leaves the receipt `reconcile_required`.

The action ledger refuses a second terminal record, and a failed closing append
is followed by a re-read. Repeated or concurrent reconciliation therefore
completes the same receipt exactly once and returns the same `evidence_digest`
every time. A retry of `perform()` with the same request key never invokes the
effect again. It returns the recorded receipt or `reconcile_pending`.

After a restart, `open_receipts()` rebuilds open receipts from the required
store alone. `reconcile_open(observer=...)` closes them:

```python
for result in coordinator.reconcile_open(observer=effect_sink):
    print(result.receipt_id, result.outcome.value, result.reason.value)
```

Only reconcile a receipt after its original attempt has finished or its
process has stopped. An in-process claim prevents concurrent use of the same
receipt within one coordinator.

## Bounded open evidence

`max_open_receipts` (1 to 4096, default 64) limits how many receipts may have
open required evidence. When the limit is reached, new admissions are denied
with `reconcile_backlog_full` until reconciliation closes receipts. A store
that cannot be read denies admission with `audit_read_failed`. The same
happens when the ledger contains an operation the current policy no longer
declares.

## Diagnostics are not evidence

`diagnostics=` accepts an optional `AuditSink` such as
`openmed.compliance.audit_chain.HashChainAuditLog`. It receives one value-free
outcome event per call, on a best-effort basis. A diagnostics failure never
changes an outcome. A diagnostics record never upgrades evidence, and
reconciliation never reads diagnostics. A diagnostics sink cannot be installed
as the required store.

## Privacy

Receipts, ledger entries, diagnostics, and errors contain only fixed codes,
opaque `rcpt_`, `act_`, and `idem_` identifiers, and `sha256:` digests.
Exceptions raised by effect adapters and observers are discarded, never
copied. Error messages are fixed codes that never echo inputs.

## Validation

```console
.venv/bin/python -m pytest tests/unit/service/test_required_audit.py \
  tests/integration/service/test_required_audit_faults.py -q
```

The integration tests inject `ENOSPC` into ledger entry writes and an
interrupted directory sync after an entry is linked. They then rebuild the
coordinator and check that reconciliation completes the receipt exactly once.
