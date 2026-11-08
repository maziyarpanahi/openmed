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

## Offline round walkthrough

[`examples/federated_round_walkthrough.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/federated_round_walkthrough.py)
connects the existing metadata contracts with fixed synthetic inputs. From the
repository root, run either output mode with an installed OpenMed environment:

```sh
python -m examples.federated_round_walkthrough --format json
python -m examples.federated_round_walkthrough --format markdown
```

JSON contains the schedule, phases at its boundaries, lifecycle history, update
validation counts, schema groups, metric envelopes, and final status. Markdown
contains the final operator status rendered by the public status API. Both
outputs are deterministic, including their final newline; regression tests
compare them byte-for-byte with fixed synthetic golden fixtures.

The coordinator defines its parameter policy independently of incoming updates.
Five anonymous placeholder updates pass that policy despite differing update
digests and parameter order. A sixth replaces one permitted parameter name with
`adapter.unknown.weight` and is rejected. Only accepted updates enter
[schema fingerprint grouping](federated-schema-fingerprints.md), producing one
group of five; the rejected update contributes to neither that group nor status
participant counts. Output includes a controlled rejection code rather than the
rejected name or raw metadata.

One synthetic aggregate metric has four contributors and is suppressed at a
minimum group size of five. Its value, uncertainty bounds, and confidence level
are discarded before serialization. A second metric has five contributors and
is released with a coarse participant band. See the
[aggregate metrics contract](federated-metrics.md) for disclosure rules.

The lifecycle advances through `planned`, `preflight`, `collecting`,
`aggregating`, and `evaluating` to `held`, with `quality_review_required` as a
fixed synthetic review decision. All five accepted participants have completed
submission, so completion is `complete` while the round remains held. The
[schedule](federated-scheduling.md) uses explicit UTC timestamps and does not
automatically advance the lifecycle.

The walkthrough runs entirely offline using standard-library federated modules;
it requires no training frameworks, model downloads, tensors, or client
identifiers. [Update validation](federated-update-metadata.md) checks metadata
declarations against policy, not tensor contents, provenance, or the truth of
clipping claims. The `threshold_only` metrics demonstrate suppression and
provide no differential privacy guarantees. No model aggregation or automated
quality decision occurs, so the [status](federated-round-status.md) contains no
aggregate artifact digest references.

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
