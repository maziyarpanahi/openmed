# Clinical SLM memory preflight

`openmed.models.clinical_slm_memory` is a local admission check for a clinical
small language-model load. Run it after a local artifact has been prepared and
before a tokenizer, backend runtime, or weight tensor is constructed.

The check is deterministic and metadata-only. It does not download a model,
import an optional runtime, inspect live process memory, open weight bytes, or
fall back to a remote service. It is an engineering estimate, not a clinical
suitability test, a compliance certification, or a guarantee of runtime peak
memory.

## Inputs and calculation

The artifact can be a mapping, JSON document, local manifest directory, or a
regular local weight file. A mapping can use the compact `weights_bytes` form:

```python
artifact = {
    "weights_bytes": 350_000_000,
}
```

It can also use the component metadata produced by the local clinical SLM
artifact manifest. Only the sizes of components whose role is `weights` (or a
model/checkpoint equivalent) are used. Paths, identifiers, prompts, and other
metadata are not copied into the result.

The runtime profile supplies the device envelope and workload assumptions:

```python
from openmed.models.clinical_slm_memory import (
    ClinicalSLMRuntimeProfile,
    preflight_clinical_slm_memory,
)

profile = ClinicalSLMRuntimeProfile(
    name="edge-test",
    memory_budget_bytes=2_000_000_000,
    resident_memory_bytes=200_000_000,
    headroom_bytes=200_000_000,
    context_tokens=2_048,
    batch_size=1,
    cache_bytes_per_token=32_768,
    context_bytes_per_token=8_192,
    batch_bytes=1_000_000,
    runtime_overhead_bytes=50_000_000,
)

report = preflight_clinical_slm_memory(artifact, profile)
if not report.ready:
    raise RuntimeError("local clinical SLM memory preflight rejected the load")
```

The estimate is intentionally explicit:

| Component | Estimate |
| --- | ---: |
| Weights | declared weight bytes |
| Cache | `context_tokens × batch_size × cache_bytes_per_token` |
| Context | `context_tokens × batch_size × context_bytes_per_token` |
| Batch | `batch_size × batch_bytes` |
| Runtime overhead | configured `runtime_overhead_bytes` |
| Total | sum of the rows above |

The total must fit in the memory available before loading
(`memory_budget_bytes - resident_memory_bytes`) while leaving at least
`headroom_bytes`. Equality passes. A memory-budget failure and a headroom
failure are reported separately, in a stable order.

## Safe reports

`report.to_json()` contains the decision, fixed reason codes, aggregate byte
counts, bounded profile metadata, a component count, and a fingerprint derived
only from normalized artifact size metadata. It contains no artifact path,
model identifier, prompt, clinical text, credentials, or model contents.

Typical successful output has `status: "accept"` and `ready: true`. A resource
shortfall returns `status: "reject"`, `ready: false`, and one or both of:

* `memory_budget_exceeded`
* `headroom_insufficient`

Malformed artifact/profile metadata raises `ClinicalSLMMemoryError`. Its
message and `to_dict()` result contain only a static code and message, so a
caller can safely log the error category without logging its input.

## Operational boundary

This preflight must run before model loading, but it does not replace runtime
monitoring or a measured device benchmark. Keep the profile tied to the actual
backend, quantization, context policy, and batch policy used by the loader. If
the profile cannot describe those assumptions, reject the load and collect a
new local profile rather than treating an unknown value as safe.

The output is an assistive resource gate only. It does not make a diagnosis,
select treatment, or authorize autonomous clinical action. Clinical workflows
must retain their existing human-review and post-de-identification controls.
