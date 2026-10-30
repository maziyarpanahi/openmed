# Federated accountant checkpoints

`openmed.training.federated_accountant_checkpoint` persists the state of a
composed federated privacy accountant so that a crash, a stale writer or an
older backup can never give back privacy budget that was already spent. It owns
persistence and recovery only: the accountant that composes the bound, the
durable store and the integrity key are all injected. It does not allocate site
reservations or generate noise.

The module is local-first and standard-library only. It makes no network call,
and its checkpoints and receipts contain only counts, aggregate bounds, opaque
digests and fixed codes.

## Lifecycle

1. `CheckpointedAccountant.genesis(...)` records an explicit fresh accountant
   in an empty store. It is the only way to start from zero spent budget.
2. `begin_round(round_digest, mechanism=..., parameters=...)` durably reserves
   and charges a round **before** its mechanism runs.
3. `commit_round(round_digest)` reconciles the round as committed after its
   output is released.
4. After a crash, `CheckpointedAccountant.restore(...)` recovers the latest
   checkpoint. A round that was reserved but not committed is still charged and
   blocks new rounds until `reconcile_round(round_digest, outcome=...)` closes
   it as `"committed"` or `"abandoned"`.

```python
import hashlib
import math

from openmed.training.federated_accountant_checkpoint import (
    AccountantCompositionParameters,
    CheckpointedAccountant,
    CheckpointIntegrityKey,
    InMemoryAccountantCheckpointStore,
)


class AdditiveAccountant:
    """Synthetic basic-composition accountant for this example only."""

    accountant_id = "additive-example"
    accountant_version = "1"

    def compose(self, mechanisms, parameters):
        return math.fsum(m.parameters["epsilon"] for m in mechanisms)


def round_digest(index: int) -> str:
    # Digest the caller's own round receipt; only the digest is recorded.
    return "sha256:" + hashlib.sha256(f"round-{index}".encode()).hexdigest()


store = InMemoryAccountantCheckpointStore()
parameters = AccountantCompositionParameters(
    composition="basic", delta=1e-5, max_epsilon=4.0
)
key = CheckpointIntegrityKey(key_id="example-key", secret=b"s" * 32)
session = CheckpointedAccountant.genesis(
    store=store,
    accountant=AdditiveAccountant(),
    parameters=parameters,
    integrity_key=key,
    genesis_id="example-genesis",
)
session.begin_round(
    round_digest(0), mechanism="sum_release", parameters={"epsilon": 0.5}
)
witness = session.commit_round(round_digest(0))
session.begin_round(
    round_digest(1), mechanism="sum_release", parameters={"epsilon": 0.5}
)
del session  # simulated crash: round 1 was reserved but never committed

restored = CheckpointedAccountant.restore(
    store=store,
    accountant=AdditiveAccountant(),
    parameters=parameters,
    integrity_key=key,
    witness=witness,
)
assert restored.uncertain_rounds == 1
assert restored.charged_epsilon == 1.0  # the uncertain round stays charged
restored.reconcile_round(round_digest(1), outcome="abandoned")
assert restored.charged_epsilon == 1.0  # reconciliation never refunds budget
```

## Conservative recovery

The accountant charges a round when it is reserved, so the bound recorded after
a crash already includes every round whose mechanism might have run.
Reconciliation never releases a charge: `"committed"` records that the output
was released, and `"abandoned"` records that it was not but keeps the charge,
because a partially executed mechanism cannot be proven unobserved. A retry
needs a new round reference and is charged again. As a result an interrupted
run that is restarted reaches the same bound as an uninterrupted run when the
round is reconciled as committed, and a strictly more conservative bound when it
is abandoned and retried.

`restore` never starts a fresh accountant. An empty store is refused with
`checkpoint_missing`, and `genesis` on a non-empty store is refused with
`genesis_conflict`.

## Checkpoint format

A checkpoint is canonical JSON (sorted keys, compact separators, ASCII, no
`NaN`) with schema version
`openmed.training.federated_accountant_checkpoint.v1`:

| Field | Meaning |
|---|---|
| `accountant` | Accountant identifier and version of the injected port. |
| `parameters` | Composition rule, `delta`, `max_epsilon` and Renyi `orders`. |
| `parameters_fingerprint` | SHA-256 of the accountant identity and parameters. |
| `generation` | Monotone generation; always equal to the number of events. |
| `events` | Append-only `genesis`, `reserve` and `reconcile` events. |
| `committed_boundary` | Committed and abandoned counts plus the last reconciled event and its chain digest. |
| `uncertain_rounds` | Round references reserved but not yet reconciled. |
| `bound` | Charged and committed epsilon recorded at this generation. |
| `head_digest` | Chain digest of the last event. |
| `integrity` | `hmac-sha256` MAC and the non-secret key identifier. |

Each event carries a chain digest over the previous digest and the event's
canonical JSON, starting from the genesis event that binds the accountant
identity and parameter fingerprint. A `reserve` event holds the round
reference, the mechanism kind and finite numeric mechanism parameters, such as
`sampling_rate`, `noise_multiplier` and `clipping_norm`.

### Why an HMAC

A plain digest can be recomputed by anyone who can write the store, so a
truncated or reordered history could simply be rehashed. The MAC is computed
under a `CheckpointIntegrityKey` injected by the coordinator and kept outside
the store, so the store cannot forge a checkpoint on its own. Use a secret of
at least 32 bytes from your key manager; the secret is hidden from `repr`.

## Restore checks

`restore` verifies, in order: canonical form and strict JSON, schema version,
HMAC, store generation, accountant identity, composition parameters, the hash
chain and event order, the generation and history length, and the committed
boundary. It then recomputes the bound from the full history with the injected
accountant instead of trusting the stored number.

Monotonicity is checked by the existing
[budget-migration verifier](../security/budget-migrations.md). The accountant
state is projected onto a two-entry budget ledger, `accountant.charged` and
`accountant.committed`, with the parameter fingerprint as policy fingerprint,
and `enforce_budget_migration` must accept:

- the previous ledger against the next ledger on every transition;
- the recorded bound against the recomputed bound on every restore, so an
  accountant that now reports less spend is refused while a more conservative
  one is accepted;
- the witness ledger against the restored ledger when a witness is supplied.

### Witness receipts

Every transition returns an `AccountantReceipt`. Keep the latest receipt
outside the store, for example in the release audit trail, and pass it to
`restore(..., witness=receipt)`. The store's own generation cannot detect a
backup restored wholesale or a wiped store with a new genesis; the witness can.
A restored history must be at least as long as the witnessed one
(`stale_checkpoint`) and must contain the witnessed head digest at the
witnessed generation (`history_diverged`).

## Concurrency fencing

Every write goes through `compare_and_swap(expected_generation=..., record=...)`
on the injected store, and each record's generation advances by exactly one. A
stale writer gets `generation_conflict` and its session must be restored. A
store error is reported as `store_failure`; because the write may or may not
have landed, the session is also poisoned (`restore_required`) and the restored
state decides whether the round was reserved.

`InMemoryAccountantCheckpointStore` is a thread-safe reference implementation
for tests and examples. A production store should provide the same atomic
compare-and-swap on durable storage.

## Reason codes

Errors are `AccountantCheckpointError` with a fixed `code` and `field`; the
message is only `"<code>: <field>"` and never echoes input values, store errors
or accountant errors.

| Code | Meaning |
|---|---|
| `checkpoint_missing` | The store is empty; no silent fresh start. |
| `genesis_conflict` | Genesis was requested on a store that already holds state. |
| `checkpoint_malformed` | The payload is not strict, canonical checkpoint JSON. |
| `schema_unsupported` | The checkpoint schema version is unknown. |
| `integrity_mismatch` | The MAC, algorithm or key identifier does not match. |
| `integrity_key_invalid` | The injected key is malformed or too short. |
| `generation_mismatch` | The payload generation differs from the store generation. |
| `history_truncated` | The history is shorter than the generation. |
| `history_out_of_order` | Event indices, genesis position or reconciliation order are invalid. |
| `history_chain_broken` | An event's chain digest does not match. |
| `boundary_mismatch` | The committed boundary or uncertain rounds disagree with the history. |
| `accountant_mismatch` | A different accountant identifier or version recorded this state. |
| `parameters_mismatch` | The composition parameters differ from the recorded ones. |
| `budget_regressed` | The budget-migration verifier refused a decrease in spent budget. |
| `budget_exceeded` | A reservation would exceed `max_epsilon`. |
| `stale_checkpoint` | The restored state is older than the witness. |
| `history_diverged` | The restored history does not contain the witnessed head. |
| `witness_invalid` | The witness is malformed or belongs to another accountant. |
| `uncertain_spend_unreconciled` | A reserved round must be reconciled before the next one. |
| `round_duplicate` | A round reference was already reserved. |
| `round_unknown` | No uncertain round matches the reference. |
| `outcome_invalid` | The reconciliation outcome is not `committed` or `abandoned`. |
| `generation_conflict` | A concurrent writer advanced the generation first. |
| `restore_required` | The session saw a conflict or store failure and must be restored. |
| `store_failure` | The injected store raised or returned an invalid result. |
| `accountant_failure` | The injected accountant raised or returned an invalid bound. |
| `mechanism_invalid` | A mechanism kind or parameter is invalid or identity-like. |
| `identifier_invalid` | A round reference, accountant or genesis identifier is invalid. |
| `parameters_invalid` | The composition parameters are invalid. |
| `history_limit_reached` | The history reached `MAX_ACCOUNTANT_EVENTS`. |

## Privacy limits

Round references must be `sha256:` digests of the caller's round receipt, so
site identities, seeds and per-site values stay with the caller. Mechanism
parameters are finite numbers, and names that suggest a seed, site, client,
participant, contribution or update are refused. Receipts omit round
references entirely. Checkpoints still reveal the number of rounds and their
mechanism parameters, so store them with the same access controls as other
privacy-accounting evidence.

The accountant port is expected to be deterministic and monotone. A
round-aware Renyi accountant can be adapted by mapping each
`AccountantMechanism` to its round type and returning the converted epsilon at
`parameters.delta`. This module does not certify a privacy guarantee, authorize
a release or make any clinical decision.
