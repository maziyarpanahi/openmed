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
