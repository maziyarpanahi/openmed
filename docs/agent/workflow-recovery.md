# Durable workflow recovery

OpenMed recovery checkpoints let a caller restart a governed workflow without
blindly repeating local tool calls, FHIR writes, or staged OMOP commits. The
contract is local, deterministic, and content-free. It does not execute a tool,
contact a server, open a database, consume an approval token, or perform a
compensating clinical write.

## Safety boundary

Each effect records only an opaque run/action identifier, a governed tool ID,
an effect category, operation and commit-evidence digests, a deterministic
idempotency key, categorical state, and a compensation limit. Checkpoints do
not accept prompts, tool arguments, FHIR resources, OMOP rows, patient or
reviewer identifiers, URLs, credentials, paths, or free-text status.

The three effect categories share one recovery rule:

| Effect | Required adapter behavior | Compensation limit |
| --- | --- | --- |
| Local tool | Query the durable local effect store by idempotency key | `none` or proposal only |
| FHIR write | Query conditional-write or transaction evidence by the same key | Proposal only; never an automatic corrective write |
| Staged OMOP batch | Query the committed batch digest by the same key | Proposal only; execute rollback elsewhere after review |

An adapter must report `absent`, `committed`, or `ambiguous`. Recovery retries
only a proven-absent effect. A matching committed effect is recorded without
repeating it. Ambiguous, missing, conflicting, or changed evidence fails closed
to human review.

## Approval handling

Recovery never accepts a bearer approval token. The checkpoint may contain only
the digest of a value-free approval receipt, its exact approved action digest,
and its exclusive expiry. The approved action digest must equal the workflow
plan digest. A missing or expired receipt cannot authorize a pending retry.
Because the API has no token input, a consumed token cannot be replayed during
recovery; the caller resumes the same logical effect under its existing receipt
and idempotency key.

Applications should checkpoint the approval receipt before dispatch. If a
workflow needs a fresh review after expiry or ambiguity, start a newly approved
recovery lineage rather than changing receipt metadata in place.

## Durable append-only journal

`CheckpointJournal` stores one canonical JSON file per sequence. It writes a
private temporary file, flushes it, and atomically links the final sequence
name. On POSIX it also flushes the directory; Windows does not expose directory
`fsync` through Python, so power-loss durability of the new directory entry is
filesystem-dependent there. An identical append is idempotent; a conflicting
append is rejected. Every checkpoint binds the previous checkpoint digest,
workflow/run identity, plan digest, approval receipt, ordered effects, commit
evidence, and recovery evidence digest.

Loading validates the full chain. Gaps, changed identities, backward effect
state, modified commit evidence, terminal-state successors, symlinks, unknown
journal entries, malformed JSON, and digest mismatches raise a value-free
`RecoveryError`. Treat every such error as `review_required`; do not continue
from a partially trusted journal.

## Recovery sequence

```python
lineage = journal.load()
checkpoint = lineage[-1]

# Adapter code performs read-only lookups by each effect.idempotency_key.
observations = inspect_effect_sinks(checkpoint.effects)
decision = recover_workflow(lineage, observations, now=clock())

next_checkpoint = advance_checkpoint(checkpoint, decision)
journal.append(next_checkpoint)

if decision.disposition is RecoveryDisposition.RESUME:
    dispatch_only(decision.retry_idempotency_keys)
elif decision.disposition is RecoveryDisposition.REVIEW_REQUIRED:
    route_to_human_review(decision.reason)
```

Persist the `DISPATCHING` checkpoint before issuing retries. After interruption,
re-query every sink; do not infer commit from a transport response cached in
memory. `RecoveryDecision` JSON and its digest are deterministic for the same
checkpoint, observations, and caller-supplied time. `advance_checkpoint` binds
that evidence into the next append-only checkpoint.

Recovery phases are checkpoint boundaries only. They do not replace the shared
agent action lifecycle. Adapter integrations should map their action phase into
the nearest checkpoint boundary while preserving the lifecycle's own transition
validation.

## Open integration dependencies

The generic journal and reconciliation engine intentionally do not duplicate
the contracts tracked by #2766, #2767, #2768, #2771, #2773, #2774, #2775,
#2776, #2778, #2996, #2998, and #3085. Those contracts remain responsible for
ledger evidence, replay verification, single-use approval receipts, previews,
FHIR conditional/concurrency/compensation/subscription behavior, staged OMOP
batches and rollback manifests, action graphs, and action lifecycle phases.
