# Federated round lifecycle

OpenMed models federated training coordination as an offline, metadata-only
state machine. The lifecycle contains no participant identifiers, local
metrics, model updates, or network behavior.

## States and transitions

| Current state | Allowed next states |
| --- | --- |
| `planned` | `preflight`, `aborted` |
| `preflight` | `collecting`, `aborted` |
| `collecting` | `aggregating`, `aborted` |
| `aggregating` | `evaluating`, `aborted` |
| `evaluating` | `held`, `promoted`, `aborted` |
| `held` | `evaluating`, `aborted` |
| `promoted` | None (terminal) |
| `aborted` | None (terminal) |

`held` represents a round waiting for review. It cannot move directly to
`promoted`; review resumes through `evaluating`, where the external quality and
privacy gates can make a new decision. This module validates the transition but
does not decide whether a gate passes.

## Usage

```python
from openmed.training.federated_round import (
    FederatedRoundLifecycle,
    FederatedRoundState,
)

round_lifecycle = FederatedRoundLifecycle()
round_lifecycle = round_lifecycle.transition_to(FederatedRoundState.PREFLIGHT)
round_lifecycle = round_lifecycle.transition_to(FederatedRoundState.COLLECTING)

payload = round_lifecycle.to_json()
restored = FederatedRoundLifecycle.from_json(payload)
assert restored == round_lifecycle
```

Lifecycle objects are immutable. Invalid skips, backward transitions, and
transitions from terminal states raise `FederatedRoundTransitionError` without
changing the existing value.

## Serialization contract

The serialized payload contains exactly two fields:

```json
{
  "schema_version": "openmed.training.federated_round.v1",
  "state": "collecting"
}
```

Deserialization rejects unknown fields, schema versions, and state values.
Errors are categorical and do not echo the rejected value, so accidentally
supplied participant metadata is not copied into routine diagnostics.

## Roster consistency before share release

Pairwise masking and dropout recovery assume the coordinator shows every
client the same enrolled and survivor sets. `FederatedRosterGuard` in
`openmed.training.federated_roster` lets each honest client refuse to
contribute or reveal recovery material when that assumption fails. Signers,
signature verifiers, operator attestations and share storage are injected
ports; the module holds no cryptography or key material.

A `FederatedRosterCommitment` hashes the sorted enrolled pseudonymous keys,
round digest, threshold and sorted survivor set under a versioned domain
separator. The threshold must exceed half of the enrolled keys, so two
disjoint survivor sets cannot both collect threshold co-signatures from
participants that sign once. One guard serves one round:

1. `authorize_submission(roster)` runs before any masked input is submitted.
   It refuses unless enough operator-attested peers are enrolled, excluding
   the local key and coordinator keys, and it pins the roster.
2. `cosign(commitment, signer)` signs one survivor set. A second set for the
   round, a changed roster, or a survivor set that drops the local client
   aborts the round.
3. `release_shares(commitment, cosignatures, requests, share_source)`
   requires threshold valid co-signatures from distinct, non-coordinator
   survivors on the co-signed set. Any validly signed conflicting set for the
   round aborts. Survivors may only receive self-mask shares and dropped peers
   pairwise-mask shares, and a peer never receives both kinds. The share port
   is called only after every check passes.

Equivocation, padding, double reveals and port failures abort the guard
permanently; a short co-signature bundle is refused without aborting. Errors
raise `FederatedRosterError` with a fixed message and a `reason` code such as
`survivor_set_equivocation`, `double_mask_reveal` or
`attested_identity_floor_not_met`. `guard.evidence()` returns only the state,
reason code, round and commitment digests and a released-share count. Keys,
signatures and shares are hidden from `repr` and never enter errors or
evidence. Drive the round lifecycle to `aborted` with `privacy_gate_failed`
when a guard aborts.
