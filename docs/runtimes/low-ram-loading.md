# Low-RAM Sharded Weight Loading

`StreamingWeightLoader` bounds transient weight-loading memory on edge devices.
It discovers logical layers deterministically, memory-maps only the current
layer group, and closes those mappings before advancing. A measured
incremental-RSS budget is enforced before and after every group; loading stops
with `RamBudgetExceeded` if the next group cannot fit.

## Supported local formats

The source must already exist on local storage. Repository IDs, URLs, and shard
paths outside the artifact directory are rejected.

- A standard `model.safetensors.index.json` with a `weight_map` and local
  `.safetensors` shards.
- One `.safetensors` file, or a directory of deterministically sorted
  `.safetensors` shards.
- An ONNX graph whose initializers use ONNX external data. The external tensor
  files may be one sidecar or multiple shards.

Inline ONNX initializers are rejected because parsing them would materialize
weights outside the streaming budget. Export those weights as external data
first.

## Stream layers into a runtime

Pass a required RAM budget in bytes or use `RamBudget.from_mib`:

```python
import numpy as np

from openmed.onnx import RamBudget, StreamingWeightLoader

loader = StreamingWeightLoader(
    "artifacts/model.safetensors.index.json",
    ram_budget=RamBudget.from_mib(192),
    layers_per_group=1,
)


def install_layer(group):
    # Replace this example dictionary with the target runtime's layer setter.
    # Copying here is intentional: mapped views expire when the callback ends.
    runtime_layers[group.name] = {
        name: np.array(weight, copy=True)
        for name, weight in group.tensors.items()
    }


runtime_layers = {}
report = loader.stream(install_layer)

print(report.groups_loaded)
print(report.peak_ram.peak_incremental_bytes)
```

Each `group.tensors` value is a read-only NumPy view over a local file mapping.
The loader releases the views, advises the operating system to discard mapped
pages where supported, and closes the mappings as soon as the callback returns.
Do not retain a mapped view. Copy a value needed later, or pass it to a runtime
API that copies the value during the callback. Retaining a view fails with
`BufferReleaseError` instead of silently defeating the memory bound.

Layer names are inferred from common `encoder.layer.N`, `layers.N`, `block.N`,
and `transformer.h.N` tensor paths. Embeddings load first, numbered layers use
numeric order, and classification heads load last. `layers_per_group` combines
consecutive logical layers when the device has more headroom.

## RAM budget semantics

The budget covers incremental process resident memory above a baseline sampled
when streaming starts. Before mapping a group, the loader reserves its full
tensor byte size against the remaining budget. It samples current RSS again
after mapping, after the consumer runs, and after release. The report records
the highest sampled RSS and its baseline-relative peak.

RSS measurement uses platform-native current-memory APIs on Linux, macOS, and
Windows. If current RSS cannot be measured, the loader raises
`RamProbeUnavailable` and reads no weights. This fail-closed behavior prevents a
low-RAM deployment from quietly continuing without an enforceable limit.

Use the smallest budget that covers the final runtime model plus one transient
layer group. If a single logical layer is larger than the budget, loading aborts
before that layer is mapped; increase the budget or export smaller layer shards.

## Determinism and parity

Shard index order does not affect loading order. Tensor and layer names are
natural-sorted, and the loader performs no network calls. The unit coverage
builds a synthetic token-classification note, runs it once with all weights and
once through streamed layer groups, and requires identical decoded spans:

```bash
.venv/bin/python -m pytest tests/unit/onnx/test_streaming_loader.py -q
```

The same tests measure a successful load below budget and verify that both a
projected oversized group and an observed RSS spike abort with a clear error.
