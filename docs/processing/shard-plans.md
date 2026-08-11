# Offline file-shard plans

`openmed.processing.shard_plan` creates reproducible work partitions from
declared file metadata. The planner is intentionally metadata-only: it does
not resolve paths, inspect filesystem timestamps, calculate sizes, open files,
or make network calls.

## Build a plan

Provide each file's path and already-known byte size, together with hard limits
for each shard:

```python
from openmed.processing.shard_plan import FileDescriptor, plan_file_shards

plan = plan_file_shards(
    [
        FileDescriptor("batch/a.bin", size_bytes=8_000),
        FileDescriptor("batch/b.bin", size_bytes=4_000),
        FileDescriptor("batch/c.bin", size_bytes=3_000),
    ],
    max_bytes=10_000,
    max_files=2,
)
```

The planner orders descriptors by descending declared size and then by a
stable SHA-256 fingerprint of the normalized path. Each descriptor is placed
in the least-loaded shard that satisfies both limits; a new shard is created
when no existing shard can accept it. This makes the result independent of
input iteration order while keeping every shard within its declared bounds.
A descriptor larger than `max_bytes` is rejected, and normalized duplicate
paths are rejected before a plan is returned.

Path normalization is lexical and platform-independent. It does not call
`Path.resolve()`, `stat()`, or `open()`. The planner therefore works with
synthetic descriptors and can run before any worker receives file access.

## Safe serialization

`FileShardPlan.to_dict()` and `FileShardPlan.to_json()` are counts-only
representations. They contain the schema and algorithm versions, the limits,
aggregate file and byte counts, per-shard counts, and safe SHA-256 digests.
They do not contain original paths or per-file membership. Keep the returned
in-memory plan alongside the descriptors when a worker needs to map a file to
its shard; persist only the counts-only form in reports and audit artifacts.

```python
summary = plan.to_dict()
canonical_json = plan.to_json()
```

The plan is a partitioning aid, not a compliance certification or a clinical
decision. Callers remain responsible for validating the supplied metadata and
for enforcing their local access policy when processing the files.
