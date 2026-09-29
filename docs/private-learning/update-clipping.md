# Deterministic client-update clipping

Bound dense adapter deltas to a trusted global or per-layer L2 norm before they
are submitted for aggregation. Clipping is a pure numeric transform: the same
policy and the same deltas always produce the same bytes, and the resulting
report describes what was bounded without carrying the submitted magnitudes.

## Coordinator policy

The coordinator supplies `FederatedClippingPolicy` independently of the update.
`global_norm_bound` applies to every layer that has no override, and
`per_layer_bounds` is the complete allowlist of layers that may be clipped under
a different bound. Do not build the policy from an incoming update: that would
let a submitter choose the bound that suits its own submission, and it would let
arbitrary free-text parameter names into the report.

```python
from openmed.training import (
    FederatedClippingPolicy,
    clip_federated_update,
    fingerprint_clipping_policy,
)

policy = FederatedClippingPolicy(
    global_norm_bound=2.5,
    per_layer_bounds=(("adapter.lora_B.weight", 0.6),),
)

result = clip_federated_update(
    {
        "adapter.lora_A.weight": [3.0, 4.0],
        "adapter.lora_B.weight": [0.3, 0.4],
    },
    policy=policy,
)

assert result.layer_deltas("adapter.lora_A.weight") == (1.5, 2.0)
assert result.layer_deltas("adapter.lora_B.weight") == (0.3, 0.4)
assert result.report.reason_codes == ("scaled_to_global_bound", "within_bound")
assert result.report.policy_digest == fingerprint_clipping_policy(policy)
```

`clip_federated_update()` accepts a mapping or a sequence of `(name, values)`
pairs and returns an immutable `ClippedFederatedUpdate`. Layers are ordered by
name in the result, so mapping order and pair order do not change the canonical
JSON. `to_mapping()` and `layer_deltas()` return fresh tuples; mutating them
cannot change the validated result.

A policy that declares a per-layer bound is a contract, not a hint. When the
update omits such a layer the call fails closed instead of silently falling back
to the global bound. Layers without an override are clipped to
`global_norm_bound`.

## Version and validation rules

The schema identifier is `openmed.training.federated.update_clipping.v1`.

| Field or limit | Contract |
| --- | --- |
| `global_norm_bound` | Finite positive number, at most `1e12`; strict float, no booleans or numeric strings |
| `per_layer_bounds` | Sequence of `{"layer": ..., "norm_bound": ...}` records, at most 1,024 entries, names unique |
| Layer names | Unique dotted ASCII identifiers, at most 256 characters, as in the update-metadata policy |
| Delta containers | Mapping or sequence of pairs; at most 1,024 layers, no duplicate names |
| Values | Sequence of finite numbers within `±1e12`; integers and floats are accepted, booleans and NaN/infinity are not |
| Elements | At most `2**24` per layer and `2**26` in total |
| Scaling | A layer above its bound is multiplied by `bound / norm`, where the norm is the L2 norm of the layer; a layer at or below its bound is returned unchanged |
| Zero layer | Reported as `zero_norm` and returned unchanged; no division by zero |
| Policy JSON | At most 1 MiB of UTF-8 text; duplicate object keys, `NaN`, and `Infinity` are rejected before decoding |
| Policy digest | `sha256:` followed by 64 lowercase hexadecimal digits, derived from the canonical policy JSON |

Any other field, an unknown schema version, a negative or zero bound, a
non-finite value, and any malformed container fail closed. Failures raise
`FederatedUpdateClippingError` with a fixed message; submitted layer names and
values never appear in the message.

`FederatedClippingPolicy.from_json()` should be used at a JSON boundary so
duplicate object keys are rejected before a generic decoder can silently
overwrite them. `FederatedClippingPolicy.from_dict()` and `to_dict()` accept and
return built-in dictionaries and lists containing JSON-style values.

## Deterministic output and privacy boundary

The clipping report is the only clipping artifact that leaves the client. For
each layer it contains the layer name, the applied bound, the element count,
whether the layer was scaled, and one of the frozen reason codes
`within_bound`, `zero_norm`, `scaled_to_global_bound`, or
`scaled_to_layer_bound`. Raw magnitudes, norms, and deltas are not included;
the report is intentionally content-free so it can be logged or attached to a
round manifest without carrying private model updates.

Clipping is a numeric transform only. It does not verify tensor bytes,
signatures, provenance, or the truth of any prior clipping declaration, and it
does not certify that coordinator-approved layer names are non-identifying.
Bounds are a coordinator decision: choose them from the aggregation contract,
not from received updates. The module performs no logging, filesystem access, or
network calls; the fingerprint is a local digest of the canonical policy JSON.

Layer names must come from the trusted policy allowlist. `clip_federated_update()`
rejects unknown names, and no name or value is echoed through exceptions.

See also [federated update metadata](../training/federated-update-metadata.md),
[federated aggregate metrics](../training/federated-metrics.md), and
[federated round lifecycle](../training/federated-round-lifecycle.md). Numerical
clipping is tracked in issue
[#2825](https://github.com/maziyarpanahi/openmed/issues/2825).
