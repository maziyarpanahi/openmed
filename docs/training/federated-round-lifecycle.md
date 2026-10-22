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

## Compromised site key holds

`openmed.training.federated_key_compromise` holds checkpoints derived from
updates signed by a site key that was later found to be compromised. Site
identity revocation refuses the key in future rounds; this module handles the
updates that were already accepted during the exposure window.

The registry accepts a signed incident record with an opaque key reference,
an exposure-window start, and a discovery time. It verifies the signature
through an injected verifier, then resolves impact through three injected
ports:

- the identity registry returns when revocation of the key took effect;
- the commitment store returns the key's accepted update commitments, with the
  coordinator acceptance time and the round that accepted each one;
- the lineage store returns checkpoint nodes with their producing round and
  parent checkpoints.

The exposure window runs from the exposure start to the later of discovery and
revocation, inclusive. When the key is not yet revoked the window stays open
and every later acceptance is held. The window is compared against the
coordinator acceptance time, because a compromised key can backdate its own
signed timestamps.

Rounds that accepted an affected update are held together with every
descendant checkpoint and the rounds that produced them. Promotion checks
consult the lineage port again, so a checkpoint trained from a held checkpoint
after the incident was recorded is also refused. When holds are active and the
lineage port does not know a checkpoint, or the port fails, promotion is
refused.

| Remediation | Meaning |
| --- | --- |
| `reaggregate_without_affected` | Every earliest affected round has clean parents and enough retained unaffected updates to meet its quorum. |
| `retrain_from_clean_ancestor` | An earliest affected round lacks quorum, retained updates, or clean parents, but a clean ancestor exists. |
| `unsupported` | An affected round is missing from the lineage or has no clean ancestor. |
| `none_required` | No accepted update fell inside the window. |

The incident reports the most severe class across its earliest affected
rounds. Descendants are always held; remediation only describes how a clean
replacement can be produced.

```python
from openmed.compliance.audit_chain import HashChainAuditLog
from openmed.training.federated_key_compromise import (
    KeyCompromiseHoldRegistry,
    promote_round,
)

registry = KeyCompromiseHoldRegistry(
    incident_verifier=incident_verifier,
    authority_verifier=operator_verifier,
    revocation=identity_registry,
    commitments=commitment_store,
    lineage=lineage_store,
    audit_sink=HashChainAuditLog(),
    state=persisted_state,
)
report = registry.apply_incident(signed_incident)
persist(registry.to_dict())

promote_round(
    lifecycle,
    checkpoint_ref=checkpoint_ref,
    round_ref=round_ref,
    registry=registry,
)  # raises KeyCompromiseHoldError with code "checkpoint_held" when held

registry.lift_hold(report.incident_digest, signed_operator_grant)
```

Lifting a hold requires a `HoldLiftGrant` bound to the incident digest and
signed by an operator whose authority the injected verifier accepts. A missing
grant fails with `operator_authority_required`; a forged grant or a grant for
another incident fails with `operator_authority_invalid`. When several
incidents hold the same checkpoint, it stays held until each incident is
lifted.

Re-applying a recorded incident, including after a restart from persisted
state, returns the same report without a second audit record. A lifted incident
is not re-held when it is replayed. An incident ID reused with a different
payload is refused with `incident_conflict`. Holds are committed before the
audit append, so an audit failure leaves the lineage held and the append is
retried on the next application; a lift is only applied after its audit
record is written.

Reports, audit payloads and persisted state contain only the schema version,
incident digest, status, window flag, held counts, a held-set digest, an
affected-commitment digest and the remediation class. They never contain the
key reference, site identity, signatures, operator references or update
commitments. Incident IDs must carry 32 to 64 random hex characters so the
incident digest cannot be linked back to a key reference by enumeration.
Errors carry a fixed message and a controlled `KeyCompromiseErrorCode`.
