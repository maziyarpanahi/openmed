# Federated update schema fingerprints

Coordinators can compare the declared schemas of anonymous dense adapter
updates without opening tensor values. `fingerprint_update_schema()` accepts an
already validated `FederatedUpdateMetadata` instance and returns a lowercase
`sha256:` reference. `same_update_schema()` fingerprints two validated metadata
instances and compares the resulting references with `hmac.compare_digest`.

```python
from openmed.training.federated_schema_fingerprint import (
    fingerprint_update_schema,
    same_update_schema,
)

# first and second were parsed with independently supplied coordinator policies.
reference = fingerprint_update_schema(first)
compatible = same_update_schema(first, second)
```

The fingerprint binds the update metadata schema version, model digest,
adapter format, and each parameter's name, shape, and dtype. Validated
parameter records have canonical name order. The function serializes these
fields as sorted-key compact JSON and hashes them with the domain prefix
`openmed.training.federated_schema_fingerprint.v1` followed by a NUL byte.
This prefix separates the digest from unrelated uses of SHA-256. A golden
vector in the focused tests pins the exact output bytes.

An individual update's content digest, clipping declaration, and derived
element count are excluded from schema identity. The coordinator must still
validate each update against an independently constructed policy; a matching
schema fingerprint says nothing about tensor contents, clipping truthfulness,
privacy, or safe aggregation. The current update contract accepts only the
`dense` adapter format, but the format field is included so a future validated
format cannot collide with it.

No client or site identifier, file path, endpoint, example, gradient, or tensor
value enters the canonical bytes. Invalid or unvalidated mappings are not
accepted by the fingerprint API.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_federated_schema_fingerprint.py -q
```

## Server optimizer state

After aggregation, a coordinator turns the aggregated adapter delta into the
next global adapter through a declared server optimizer in
`openmed.training.federated_server_optimizer`. Aggregation, coordinator
execution, and lineage storage are supplied by the caller; the module receives
an already-aggregated delta as a mapping from canonical parameter name to a
flat sequence of floats. The delta is the aggregated client adapter minus the
current global adapter, so each rule moves the global adapter in the delta
direction.

| Rule | Update |
| --- | --- |
| `sgd` | `x + lr * delta`; a learning rate of 1.0 adds the aggregate delta |
| `momentum` | `v = momentum * v + delta`, then `x + lr * v`, or `x + lr * (delta + momentum * v)` with `nesterov=True` |
| `adagrad` | `m = beta1 * m + (1 - beta1) * delta`, `v = v + delta^2` |
| `adam` | same `m`, `v = beta2 * v + (1 - beta2) * delta^2` |
| `yogi` | same `m`, `v = v - (1 - beta2) * delta^2 * sign(v - delta^2)` |

The adaptive rules follow Algorithm 2 of Reddi et al., "Adaptive Federated
Optimization" (ICLR 2021), without bias correction, and apply
`x + lr * (m / (sqrt(v) + tau))`. Defaults are momentum 0.9, beta1 0.9,
beta2 0.99, tau 1e-3, and an initial second moment of `tau * tau`.
Hyperparameters that do not apply to the declared rule must be left unset.

```python
from openmed.training.federated_server_optimizer import (
    ServerOptimizerConfig,
    ServerParameterSchema,
    apply_server_update,
    initial_server_optimizer_state,
    restore_server_optimizer_state,
)

config = ServerOptimizerConfig(rule="adam", learning_rate=0.1)
schema = ServerParameterSchema.from_lengths({"adapter.lora_A.weight": 3})
state = initial_server_optimizer_state(config, schema)
result = apply_server_update(
    state,
    {"adapter.lora_A.weight": [0.5, -0.25, 1.0]},
    {"adapter.lora_A.weight": [0.1, -0.2, 0.3]},
)
checkpoint = result.state.to_checkpoint_bytes()
resumed = restore_server_optimizer_state(
    checkpoint,
    config=ServerOptimizerConfig.from_hyperparameter_code(
        result.record.hyperparameter_code
    ),
    schema=schema,
    expected_step=1,
    expected_state_digest=result.record.state_digest,
)
assert resumed.state_digest == result.state.state_digest
```

### Declaration and determinism

`ServerOptimizerConfig.hyperparameter_code` is the canonical ASCII
declaration for a round contract, for example
`rule=sgd;learning_rate=0x1.0000000000000p-1`. Fields appear in a fixed
per-rule order and floats use exact `float.hex` encoding.
`from_hyperparameter_code()` accepts only that canonical form, and
`config_digest` is a domain-separated SHA-256 reference to it.

Numerics use only the standard library. Every operation is elementwise
IEEE-754 float64 arithmetic in a fixed order, so the same declaration, schema,
adapter, and deltas produce byte-identical adapters, state, checkpoints, and
records across runs, processes, and hash seeds. Schemas are bounded to 1,024
parameters and 2,097,152 total elements, and steps to `2**31 - 1`.
`ServerParameterSchema.from_update_metadata()` derives element counts from
validated update metadata and binds the update schema fingerprint above into
the parameter schema digest.

### Checkpoints and lineage

`to_checkpoint_bytes()` returns versioned, sorted-key compact ASCII JSON with
`schema_version` `openmed.training.federated_server_optimizer_state.v1`. Moment
vectors are base64 of big-endian float64 values, so restore is bit-exact. The
embedded `state_digest` is a SHA-256 over the canonical body with the domain
prefix `openmed.training.federated_server_optimizer.state.v1` followed by a
NUL byte.

`restore_server_optimizer_state()` checks, in order: payload size, strict
canonical JSON with no duplicate or extra fields, the state version, the
embedded integrity digest, the optional lineage-bound `expected_state_digest`,
then the rule, hyperparameter code, parameter schema digest, and step.
Failures raise `ServerOptimizerError` with a fixed reason code such as
`state_integrity_mismatch`, `state_lineage_mismatch`, `state_rule_mismatch`,
`state_hyperparameter_mismatch`, `state_schema_mismatch`, or
`state_step_mismatch`. The embedded digest detects corruption and edits that do
not recompute it; pass the state digest recorded in lineage evidence to also
reject a recomputed or substituted checkpoint. Restarting from a restored
checkpoint reproduces the same next global adapter as an uninterrupted run.

Each update returns a `ServerUpdateRecord` for lineage evidence. It contains
the rule, hyperparameter code, configuration and parameter schema digests, the
step, the previous and new state digests, and digests of the input adapter,
aggregate delta, and output adapter. `ServerOptimizerState.describe()` returns
the same codes and digests for a state without its vectors.

### Privacy accounting

A server update reads only the released aggregate, prior server state, and
public hyperparameters, so it is post-processing of the released aggregate and
adds no differential-privacy charge. Every record carries
`privacy_accounting="post_processing_no_additional_charge"`. This holds only
when the delta is the privacy-protected release itself and hyperparameters are
not tuned on private data; tuning or extra reads of client data need separate
accounting.

Server state, checkpoints, and records contain no client or site identifier,
per-client update, example, path, or endpoint. Checkpoints hold the parameter
schema, hyperparameter codes, digests, the step, and aggregate-derived moment
vectors; records hold only codes, digests, and the step. Error messages are a
fixed reason code plus, at most, a fixed hyperparameter field name.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_federated_server_optimizer.py -q
```
