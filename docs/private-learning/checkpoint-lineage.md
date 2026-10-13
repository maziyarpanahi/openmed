# Federated checkpoint ancestry

`CheckpointLineageGraph` validates a bounded digest-addressed DAG describing
base models, declared round manifests, aggregates and released adapters. It
exports only a selected checkpoint and its transitive ancestors. Unrelated
branches, site identities, client metrics and source payloads are excluded.

```python
from openmed.training.federated.checkpoint_lineage import (
    CheckpointLineageGraph, CheckpointLineageNode,
)

base, round_ref, aggregate, adapter = ['sha256:' + c * 64 for c in 'abcd']
graph = CheckpointLineageGraph([
    CheckpointLineageNode(base, 'base_model'),
    CheckpointLineageNode(round_ref, 'round_manifest', (base,)),
    CheckpointLineageNode(aggregate, 'aggregate', (round_ref,)),
    CheckpointLineageNode(adapter, 'released_adapter', (aggregate,)),
])
assert len(graph.ancestry(adapter)) == 4
print(graph.to_json(adapter))
```

All references use `sha256:` followed by 64 lowercase hexadecimal characters,
matching the existing federated update-envelope digest spelling. Parents must
exist in the submitted graph. Base models have no parents; round manifests
reference base models or prior released adapters; aggregates reference round
manifests; released adapters reference aggregates. Multiple branches and shared
ancestors are allowed. Duplicate nodes or parents, self-references, cycles,
unresolved parents and invalid stage transitions reject with controlled errors.

The graph snapshots at most 10,000 nodes with at most 32 parents per node. Nodes
and ancestry tuples are immutable, and traversal is iterative rather than
recursion-limited. Output ordering and JSON bytes are independent of input order.
Only base-model or released-adapter nodes can be exported as checkpoints.

These are **declared metadata references**: the graph does not read artifact
files, verify digests or signatures, establish training quality, prove that a
private round occurred, or authorize a release. Callers must independently
verify source artifacts and the completeness of coordinator-issued lineage.
There is no automatic enrollment, training, cryptography, networking or clinical
claim. Do not put identities, paths, endpoints or free text into digest fields.
