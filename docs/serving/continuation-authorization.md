# Continuation Authorization

Service authentication decides whether a request may be admitted. A queued
de-identification job or a long-lived NDJSON stream can keep running after the
consent, scope or policy revision it was admitted under has been revoked or has
expired. `openmed.service.continuation_auth` binds admitted work to an
authorization epoch and revalidates that epoch every time the work is about to
do something protected.

The module owns continuation authorization only. It does not authenticate
callers, define identity or consent policy, or persist the job queue. It is an
injectable component: the application supplies the authority provider, the
workers and the stream sources. It makes no network calls.

## Building blocks it reuses

- `AuthorityRuntime` from `openmed.agent.permissions.revocation` performs every
  status lookup. It combines the application's `AuthorityStatusProvider`, a
  monotonic `AuthorityGenerationStore` and a trusted clock. See
  [Runtime Authority Revocation](../agent/runtime-revocation.md) for the
  provider and generation-store contracts.
- `EffectRecord` from `openmed.agent.workflows.recovery` records external
  effects, the same record type the governed workflow service uses.
- `CancellationSignal` from `openmed.agent.workflows.scheduler` is the
  cooperative stop signal. Cancelling never undoes a committed effect.

## Admission pins an epoch

After the existing admission checks pass (authentication, route scopes and
request limits), derive the authority the work depends on and admit it:

```python
from openmed.service.continuation_auth import (
    ContinuationGuard,
    consent_authority,
    principal_authority,
)

guard = ContinuationGuard(runtime, ledger=durable_ledger)
contracts = (
    principal_authority(principal, policy_revision="policy:local@3", expires_at=exp),
    consent_authority(consent_digest, expires_at=consent_exp),
)
record = guard.admit(job_id, contracts, checkpoints=("document",))
job_store.save(job_id, {"continuation": record.to_dict()})
```

`principal_authority` commits to the authenticated subject, credential type,
key id, sorted scopes and policy revision. Narrowing scopes or moving to a new
policy revision gives a different contract, so the provider decides each one
explicitly. `consent_authority` references a consent or purpose decision by its
digest; the consent itself stays with its owner. Both return the existing
`AuthorityContract` type. Register them with your provider at admission.

`admit` calls `AuthorityRuntime.bind`, which pins the current generation of
every contract. If any contract is unknown, revoked or expired, admission
raises `ContinuationInterrupted` with an `admission` outcome.

The returned `ContinuationRecord` is the persisted epoch. Its fields are fixed:
a random admission id, the digest of the work id, the declared checkpoint names
and the pinned authority binding. The `epoch_digest` property is a content-free
commitment to all of them. A record never stores a verdict such as
"authorized". `ContinuationRecord.from_dict` rejects missing or extra fields, so
cached status or consent flags cannot be smuggled back in.

## Boundaries that recheck authority

Every boundary performs fresh provider lookups through the runtime. The
default status age the runtime accepts is zero seconds.

| Boundary | When it runs |
| --- | --- |
| `execution_start` | A queued item is picked up, or a stream starts, before the source is read |
| `resume` | A restarted or paused worker resumes persisted work |
| `checkpoint` | Each declared long-lived checkpoint; undeclared names are quarantined |
| `effect_dispatch` | Immediately before each external effect |
| `protected_output` | Immediately before each stream chunk or job result is released |

Each decision is a `ContinuationOutcome` with one of four dispositions:

- `continue`: authority is current and the work may cross the boundary.
- `completed`: the work finished and its outputs were released.
- `stopped`: authority was revoked or expired, or the work was cancelled.
- `quarantined`: an integrity or availability failure, such as a stale or
  unavailable status, a generation rollback, a record presented for a
  different work id, an undeclared checkpoint, a worker or source failure, or a
  ledger that cannot record the decision.

Any disposition other than `continue` closes the epoch in the injected
`ContinuationLedger`. A closed epoch reports `continuation_terminal` from then
on, even if the authority is later valid again. Running the work again requires
a new admission, which creates a new epoch.

## Queued jobs

`run_continuation` runs one unit of queued work under the guard:

```python
from openmed.service.continuation_auth import run_continuation

def worker(ctx):
    for ordinal, document in enumerate(documents):
        ctx.checkpoint("document")
        effect = plan_effect(ordinal, document)
        ctx.commit_effect(effect, lambda: sink.commit(effect.idempotency_key))
        ctx.emit(redact(document))

result = run_continuation(guard, record, job_id, worker, resumed=False)
job_store.save(job_id, {"continuation": record.to_dict(), "result": result.to_dict()})
```

- The start or resume boundary is checked before the worker is called, so
  revoking authority while a job is queued means it never runs.
- `ctx.checkpoint(name)` checks a declared checkpoint.
- `ctx.commit_effect` checks `effect_dispatch`, records the effect as pending
  and marks it committed only when the dispatch callback returns a `sha256:`
  commit evidence digest.
- `ctx.emit` buffers protected output. Buffered output is released only after
  a final `protected_output` check once the worker returns. If the work stops,
  the buffer is discarded and reported only as `withheld_output_count`.
- A worker exception becomes a `continuation_worker_failed` quarantine. The
  exception message and traceback are not kept.

## Committed effects are reported honestly

Stopping work never claims to undo what already happened. Every outcome
reports `committed_effect_count`, `pending_effect_count`, a digest over the
committed idempotency keys and evidence, a fixed `"compensation":
"not_attempted"` field, and a `reconciliation` state:

| Reconciliation | Meaning |
| --- | --- |
| `no_effects` | No effect was dispatched under this epoch |
| `committed_effects_retained` | Dispatched effects are committed and remain in effect |
| `pending_effects_unconfirmed` | At least one dispatch did not return commit evidence; reconcile it against the sink by idempotency key |

`ContinuationResult.effects` keeps the `EffectRecord` list. These are the same
records `openmed.agent.workflows.recovery` uses, so pending effects can be
reconciled against their sink by idempotency key with `EffectObservation`.

## Long-lived streams

`GuardedStream` wraps a synchronous or asynchronous source:

```python
from openmed.service.continuation_auth import GuardedStream, continuation_error_event

stream = GuardedStream(
    guard,
    record,
    stream_id,
    source,
    terminal=continuation_error_event,
)


async def body():
    async for item in stream:
        yield item
```

The stream checks `execution_start` before reading the source. It then pulls
each chunk and releases it only after a fresh `protected_output` check. A
denied chunk is discarded and counted as withheld; no later chunk is read. The
source is always closed. With `terminal=continuation_error_event` the stream
ends with an error event in the same shape as the streaming endpoint's error
envelope, carrying only the reason code, boundary, counts and epoch digest.
Without `terminal`, the stream raises `ContinuationInterrupted`. Consumer
disconnects, `asyncio` task cancellation and a set `CancellationSignal` are all
recorded as `continuation_cancelled`. The `outcome` attribute holds the
terminal decision.

## Restarts cannot revive stale consent

A restarted worker has only persisted metadata. That is never enough to
continue:

- The record holds pinned generations, not a status. Resuming always performs
  new provider lookups, so a revoked or expired consent stays revoked or
  expired.
- Editing the restored generations does not help. The runtime compares them
  exactly with current status, and the generation store rejects rollback and
  any revival of a revoked digest.
- The closed-epoch ledger is kept separate from job metadata. Resetting a job
  row to `queued` cannot reopen a stopped or quarantined epoch.

`InMemoryContinuationLedger` and `InMemoryAuthorityGenerationStore` only cover
a single process. For restart safety in production, inject a durable, atomic
ledger and generation store that are shared by every worker and stored
separately from the job queue. If both are restored to an older copy, local
rollback detection no longer works.

## Privacy

Outcomes, records, errors and stream error events contain only controlled
reason codes, counts, declared checkpoint names, `sha256:` digests and pinned
generations. Work ids, subjects, credentials, outputs and provider exception
text are never stored or echoed. Invalid arguments raise `ContinuationError`
with a fixed `invalid_continuation_*` code.

## Integration status

This module is not yet wired into `openmed/service/jobs.py` or the
`/pii/deidentify/stream` endpoint. Applications that enforce continuation
authorization must route queued execution through `run_continuation` and
protected streams through `GuardedStream`.
