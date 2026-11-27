# Model Lifecycle Events

Model prefetch, verification, cache eviction and activation report what
happened to a model through one closed, value-free event vocabulary,
`openmed.model.lifecycle.v1`. An operator, the service or an OpenMedKit port
can subscribe to it by passing a sink. Nothing is recorded by default.

## Events

| Event | Emitted by | When |
|---|---|---|
| `requested` | `prefetch_model` | A snapshot was requested |
| `fetching` | `prefetch_model` | One file download starts (again on a forced re-fetch) |
| `verified` | `prefetch_model` | One file matched its Hub size and digest |
| `integrity_failed` | `prefetch_model` | One file did not match its Hub size or digest |
| `ready` | `prefetch_model` | Every file is verified, or offline mode served the cached snapshot |
| `activated` | `emit_model_activated` hook | A verified generation became active |
| `rolled_back` | `emit_model_rolled_back` hook | Activation returned to the previous generation |
| `evicted` | `ModelCachePolicy.apply_eviction` | One cache artifact was removed under the quota |
| `failed` | `prefetch_model` | The operation stopped; `reason` says why |

Every event has the same eight fields, in this order: `schema_version`,
`event`, `repo_id`, `revision_sha`, `file_count`, `byte_count`, `digest` and
`reason`. Unknown values are `null`. `repo_id` is a Hub repository id,
`revision_sha` a 40- or 64-hex commit hash, `digest` is `sha256:<hex>` or
`git-sha1:<hex>`, and `reason` is a controlled code. Events never carry a URL,
a token, a filename, a filesystem path or exception text, and the schema
rejects those shapes.

`fetching`, `verified` and `integrity_failed` describe one file, with its size
and its Hub digest. `ready` carries the file count and total bytes of a
download; on an offline cache hit the snapshot is not re-measured, so counts
stay `null`. `evicted` carries the artifact digest and the bytes freed, and
`repo_id` is `null` because the cache policy tracks digests, not
repositories.

| Reason | Event | Meaning |
|---|---|---|
| `digest_mismatch`, `size_mismatch`, `file_missing` | `integrity_failed` | How a file failed verification |
| `integrity_retry_exhausted` | `failed` | The file still failed after one forced re-fetch |
| `commit_unverifiable` | `failed` | The Hub reported no valid commit hash |
| `metadata_invalid_filename`, `metadata_missing_size`, `metadata_missing_digest`, `no_matching_files` | `failed` | Hub metadata cannot support verification |
| `corrupt_file_unremovable`, `revision_name_unsafe` | `failed` | A cache write was refused |
| `offline_cache_miss` | `failed` | Offline mode and nothing cached |
| `download_error` | `failed` | Any other download failure |
| `artifact_absent` | `evicted` | The artifact was already gone when eviction ran |

`DownloadIntegrityError` now also exposes the same code as its `reason`
attribute. Its message is unchanged.

## Subscribing

```python
from openmed.core.hf_hub import prefetch_model
from openmed.core.model_cache_policy import ModelCachePolicy
from openmed.core.model_lifecycle import RecordingModelLifecycleSink

sink = RecordingModelLifecycleSink()
prefetch_model("OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1", lifecycle_sink=sink)
policy = ModelCachePolicy("/var/cache/openmed", 2_000_000_000, lifecycle_sink=sink)

for event in sink.to_list():
    print(event["event"], event["byte_count"], event["reason"])
```

Any object with an `emit(event)` method is a sink. The default,
`NULL_MODEL_LIFECYCLE_SINK`, drops events without building them and makes no
network call. A sink that raises never changes the outcome of the download or
eviction that emitted the event, so do not use a sink for control flow.
`evicted` events are emitted after the cache manifest is written, outside the
policy lock.

Transactional activation calls `emit_model_activated(sink, repo_id=...,
revision_sha=..., digest=...)` after commit and `emit_model_rolled_back(sink,
..., reason=...)` after a rollback; wiring those hooks into
`openmed.service.model_activation` is separate work (#3575).

## Android OpenMedKit

`ModelDownloader(cache, client, logger, lifecycleSink, progressListener)` emits
the same event bodies as `ModelLifecycleEvent`, alongside per-file byte
progress through `ModelDownloadProgressListener`, which is not part of this
vocabulary. `ModelDownloaderTest` replays the prefetch sequences of the shared
vectors through a fake Hub. Two prefetch sequences do not apply to Android:

- Android checks a file without a published digest by its size, so it does
  not fail with `metadata_missing_digest`.
- `ModelDownloader` has no offline mode, so there is no `offline_cache_miss`.

The eviction and activation sequences come from other components. Android also
reports a catalog reproducibility-hash mismatch as `integrity_failed` for the
manifest (`file_count` 1, `byte_count` 0, the catalog digest), followed by
`failed` with `integrity_unverifiable`. See
[Android integration](../android-integration.md#lifecycle-events-and-byte-progress).

## Schema and shared vectors

`openmed/core/schemas/json/model_lifecycle_event.schema.json` describes one
event body. When the platform lifecycle event envelope is defined, an event is
carried unchanged as its data.

`tests/fixtures/parity/model_lifecycle_v1.json` holds synthetic, ordered event
sequences for the kits: a clean one- and two-file download, a pinned commit, a
re-fetch after corruption, corruption that persists, missing Hub digests, an
offline cache hit and miss, LRU eviction, eviction of an already-absent
artifact, and activation followed by rollback. Each sequence keeps the
scenario that produced it, and `tests/unit/core/test_model_lifecycle_events.py`
replays every one against an in-process fake Hub.
