# Federated update metadata

Validate the declared structure of a dense adapter update before aggregation
without opening tensors, computing gradients, or accepting client identity.
The envelope accepts only model and update SHA-256 references, an adapter
format, parameter names, shapes, dtypes, an exact total element count, a clipping
declaration, and the schema version.

## Coordinator policy

The coordinator supplies `FederatedUpdatePolicy` independently of the update.
Its parameter specifications are the complete allowlist of names, shapes, and
dtypes expected for the selected model. Do not construct this policy from
incoming update metadata: doing so would let a submitter authorize its own
parameters. This module does not infer model structure or read a model file.

```python
from openmed.training import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

# Synthetic digest references and adapter shapes for this example only.
model_digest = "sha256:" + "a" * 64
policy = FederatedUpdatePolicy(
    model_digest=model_digest,
    parameters=(
        FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
        FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
    ),
    require_clipped=True,
    max_total_elements=14,
)

payload = {
    "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    "model_digest": model_digest,
    "adapter_format": "dense",
    "parameters": [
        {"name": "adapter.lora_B.weight", "shape": [4, 2], "dtype": "float32"},
        {"name": "adapter.lora_A.weight", "shape": [2, 3], "dtype": "float32"},
    ],
    "total_elements": 14,  # (4 * 2) + (2 * 3)
    "update_digest": "sha256:" + "b" * 64,
    "clipped": True,
}

metadata = FederatedUpdateMetadata.from_dict(payload, policy=policy)
canonical_json = metadata.to_json()
assert FederatedUpdateMetadata.from_json(canonical_json, policy=policy) == metadata
```

Use `from_json()` at a JSON boundary so duplicate object keys are rejected
before they can be silently overwritten by a generic JSON decoder. `from_dict()`
accepts built-in dictionaries and lists containing JSON-style values. Direct
construction also requires a policy and enforces the same metadata invariants.

## Version and validation rules

The schema identifier is `openmed.training.federated_update_metadata.v1`.
Every envelope and parameter field shown above is required. Unknown fields
are rejected; there is no extensions or free-text field.

| Field or limit | Contract |
| --- | --- |
| `adapter_format` | Exactly `dense`; no sparse, packed, or container-specific formats |
| Model digest | `sha256:` followed by 64 lowercase hexadecimal digits; must match policy |
| Update digest | Same digest syntax; a declared content reference only |
| Parameter names | Unique dotted ASCII identifiers, at most 256 characters; numeric path components such as `layers.0.weight` are allowed |
| Parameter set | Exactly the coordinator's set, including each expected shape and dtype |
| Parameter count | Between 1 and 1,024 |
| Shape | Between 1 and 8 positive integer dimensions, each at most `2**31 - 1` |
| Dtype | `float16`, `bfloat16`, `float32`, or `float64`, matching that parameter's policy |
| Element arithmetic | Checked integer products and sum; at most `2**63 - 1` |
| Total elements | Must equal the sum of shape products and respect the policy budget, default 100,000,000 |
| `clipped` | Strict boolean; must be `true` unless trusted policy permits an unclipped declaration |
| JSON size | At most 1 MiB of UTF-8 text, including whitespace |

Booleans, strings, floating-point numbers, NaN, and infinity are not integer
dimensions or counts. Scalars (zero-rank shapes), zero-sized dimensions, unknown
parameters, missing parameters, duplicate parameters, unsupported versions, and
overflows fail closed. Reordering dimensions is a shape mismatch even when the
product remains the same.

## Deterministic output and privacy boundary

Validated metadata is immutable. Parameter records are sorted by name;
`to_json()` sorts object keys and emits indented JSON with a trailing newline.
Changing the input parameter or key order does not change the canonical bytes.
`to_dict()` returns fresh dictionaries and shape lists, so later caller mutations
cannot change the validated envelope.

Tensor arrays, gradients, examples, client IDs, site names, paths, endpoints,
and local metrics are rejected as unknown fields at every record level.
Parameter-name syntax excludes paths, URLs, whitespace, and arbitrary free
text. Known fields also reject invalid types instead of coercing them. Failures
raise `FederatedUpdateMetadataError` with fixed messages that omit submitted
keys and values; JSON parser excerpts are suppressed. The module performs no
logging, filesystem access, network calls, tensor allocation, or training.

The accepted metadata is still a declaration. Valid digest syntax and a matching
model reference do not verify tensor bytes, signatures, provenance, or the
truth of a clipping claim. This schema does not certify that digests or
coordinator-approved names are non-identifying. Those meanings depend on the
trusted caller's policy and handling. Do not log arbitrary rejected payloads.

Round manifest verification, actual tensor allowlisting, and numerical clipping
remain separate work in issues [#2822](https://github.com/maziyarpanahi/openmed/issues/2822),
[#2824](https://github.com/maziyarpanahi/openmed/issues/2824), and
[#2825](https://github.com/maziyarpanahi/openmed/issues/2825). This module has no
imports from those pending implementations. See also
[federated round lifecycle](federated-round-lifecycle.md) and
[federated round status](federated-round-status.md).
