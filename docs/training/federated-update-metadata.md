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
truth of a clipping claim. Use the canonical update encoding below to bind the
declared `update_digest` to exact payload bytes. This schema does not certify that digests or
coordinator-approved names are non-identifying. Those meanings depend on the
trusted caller's policy and handling. Do not log arbitrary rejected payloads.

## Canonical update encoding and digest binding

`openmed.training.federated_update_encoding` defines the only plaintext byte
layout whose SHA-256 digest may be declared as `update_digest`. Tensor bytes
are opaque to the module, so no tensor library is required.

### Byte layout, version 1

The encoding identifier is `openmed.training.federated_update_encoding.v1`.
All integers are unsigned and little-endian. There are no reserved, padding,
alignment, or metadata fields.

| Part | Bytes | Contract |
| --- | --- | --- |
| Magic | 4 | ASCII `OMFU` |
| Format version | 2 (`u16`) | Exactly `1` |
| Parameter count | 2 (`u16`) | Between 1 and 1,024 |
| Entries | variable | Exactly `parameter count` entries, then end of input |

Each entry is:

| Field | Bytes | Contract |
| --- | --- | --- |
| Name length | 2 (`u16`) | Between 1 and 256 |
| Name | name length | ASCII parameter name using the metadata name syntax |
| Dtype code | 1 (`u8`) | `1` float16, `2` bfloat16, `3` float32, `4` float64 |
| Rank | 1 (`u8`) | Between 1 and 8 |
| Dimensions | 4 (`u32`) per rank | Each between 1 and `2**31 - 1`; checked product at most `2**63 - 1` |
| Payload length | 8 (`u64`) | Exactly element count times item size (2, 2, 4, or 8) |
| Payload | payload length | Contiguous little-endian element bytes in row-major order |

Entries are strictly ascending by name. Because names are ASCII, string order
and byte order agree. The update digest is
`"sha256:" + sha256(b"openmed.training.federated_update_encoding.v1\0" + encoded)`,
which is domain-separated from the schema fingerprint and other OpenMed
digests.

### Encoding, decoding and admission

```python
import struct

from openmed.training.federated_schema_fingerprint import fingerprint_update_schema
from openmed.training.federated_update_encoding import (
    admit_federated_update,
    decode_federated_update,
    digest_federated_update,
    encode_federated_update,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

# Synthetic tensors: name -> (dtype, shape, little-endian bytes).
tensors = {
    "adapter.lora_B.weight": ("float32", (4, 2), struct.pack("<8f", *[0.0] * 8)),
    "adapter.lora_A.weight": ("float32", (2, 3), struct.pack("<6f", *[0.5] * 6)),
}
encoded = encode_federated_update(tensors)  # mapping order does not matter
assert [entry.name for entry in decode_federated_update(encoded)] == [
    "adapter.lora_A.weight",
    "adapter.lora_B.weight",
]

policy = FederatedUpdatePolicy(
    model_digest="sha256:" + "a" * 64,
    parameters=(
        FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
        FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
    ),
    max_total_elements=14,
)
metadata = FederatedUpdateMetadata(
    model_digest=policy.model_digest,
    adapter_format="dense",
    parameters=policy.parameters,
    total_elements=14,
    update_digest=digest_federated_update(encoded),
    clipped=True,
    policy=policy,
)

# The coordinator owns the expected fingerprint; never read it from a client.
expected_fingerprint = fingerprint_update_schema(metadata)
admission = admit_federated_update(
    metadata, encoded, expected_schema_fingerprint=expected_fingerprint
)
assert admission.to_dict()["parameter_count"] == 2
```

`admit_federated_update()` must succeed before an update reaches aggregation.
It runs its checks in a fixed order so the reported reason is stable: argument
types, the coordinator-owned schema fingerprint, strict canonical decoding,
parameter names, dtypes and shapes against the validated metadata, and finally
the declared `update_digest`. Digest and fingerprint comparisons are constant
time. A submitter that hashes its own non-canonical bytes is still rejected,
because decoding happens before the digest comparison. The returned
`FederatedUpdateAdmission` receipt carries only the encoding version, digests,
parameter count, total elements, and encoded size; it cannot be constructed
directly.

### Rejections and reason codes

Every failure raises `FederatedUpdateEncodingError` with a `reason_code` from
`FederatedUpdateEncodingReason` and the fixed message
`federated update encoding rejected: <reason_code>`. Messages, `repr` output,
and exception arguments never include tensor bytes, parameter names, declared
digests, or paths, and parser context is suppressed.

| Reason code | Typical cause |
| --- | --- |
| `invalid_argument` | Wrong argument type, malformed expected fingerprint, or invalid size bound |
| `encoded_size_exceeded` | Input or output larger than `max_encoded_bytes` (default 1 GiB) |
| `truncated_encoding` | Input ends inside the header, an entry, or a payload |
| `invalid_magic` | A different container or a prepended metadata header |
| `unsupported_version` | Any format version other than `1` |
| `invalid_parameter_count` | Zero or more than 1,024 parameters |
| `invalid_parameter_name` | Empty, over-long, non-ASCII, path-like, or padding bytes read as a name |
| `unsorted_parameters` | Entries not in ascending name order |
| `duplicate_parameter` | The same name encoded twice |
| `unsupported_dtype` | An unknown dtype code |
| `invalid_shape` | Rank 0 or above 8, zero or oversized dimension, or element overflow |
| `payload_length_mismatch` | Declared payload length not equal to the shape and dtype size, including alignment padding |
| `trailing_bytes` | Any byte after the final entry, such as an appended metadata section |
| `schema_fingerprint_mismatch` | Metadata schema differs from the coordinator's expected fingerprint |
| `parameter_mismatch` | Encoded names differ from the validated metadata |
| `dtype_mismatch` | Encoded dtype differs from the declared dtype |
| `shape_mismatch` | Encoded shape differs from the declared shape |
| `update_digest_mismatch` | Canonical bytes do not hash to the declared `update_digest` |

The decoder checks every declared length against the remaining input before
slicing, so header fields cannot trigger large allocations. Decoded entries
expose read-only payload views over an immutable snapshot of the input and
omit payload bytes from `repr`.

Canonical bytes make the digest a stable identity; they do not make tensor
values safe. Element values, including NaN payload bits, are not inspected
here. Value checks such as clipping and finite-value enforcement, parameter
allowlist policy, secure-aggregation codecs, and aggregate commitments remain
separate work.

Round manifest verification, actual tensor allowlisting, and numerical clipping
remain separate work in issues [#2822](https://github.com/maziyarpanahi/openmed/issues/2822),
[#2824](https://github.com/maziyarpanahi/openmed/issues/2824), and
[#2825](https://github.com/maziyarpanahi/openmed/issues/2825). This module has no
imports from those pending implementations. See also
[federated round lifecycle](federated-round-lifecycle.md) and
[federated round status](federated-round-status.md).
