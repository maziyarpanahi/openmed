# Local model-cache quotas

`openmed.core.model_cache_policy.ModelCachePolicy` provides a local-only
quota for model artifacts that an application explicitly owns. It does not
resolve model names, import a Hub client, download files, or make a mandatory
network request.

## Register owned artifacts

The policy uses a cache-local ownership manifest. Register a file or a complete
directory that the application owns; do not register a shared Hugging Face
cache root or a directory containing artifacts owned by another process.
Directory artifacts are measured recursively and must not contain symbolic
links. This prevents eviction from following a link to a shared or unrelated
location.

```python
from pathlib import Path

from openmed.core import ModelCachePolicy, sha256_path

cache_dir = Path("~/.cache/openmed").expanduser()
artifact = cache_dir / "models" / "synthetic-model"
policy = ModelCachePolicy(
    cache_dir,
    quota_bytes=8 * 1024**3,
    pinned_artifacts=[cache_dir / "models" / "baseline-model"],
)

expected = sha256_path(artifact)
entry = policy.register_artifact(
    artifact,
    expected_sha256=expected,
    last_accessed_ns=1_000,
)
```

The manifest stores cache-relative paths for ownership bookkeeping, while
public summaries, plans, results, and exceptions expose only a `path_hash`,
content checksum, byte totals, or counts. It is safe to serialize
`entry.to_dict()` in an operational report.

## Verify before reuse

Reuse must verify the recorded checksum first. A changed or missing artifact
raises `CacheIntegrityError` and does not update its LRU marker.

```python
local_path = policy.reuse_artifact(artifact, expected_sha256=expected)
```

The optional expected checksum must match both the local bytes and the checksum
recorded for the owned artifact. `verify_artifact_checksum()` can verify a
local file or directory before it is registered, without requiring ownership.

## Plan and apply deterministic eviction

Eviction is least-recently-used. Ties are broken by the canonical cache-
relative path, so the same manifest produces the same plan. Pinned artifacts
are never candidates. `additional_bytes` reserves space for an incoming
artifact that has not been registered yet.

```python
plan = policy.plan_eviction(additional_bytes=512 * 1024**2)
result = policy.apply_eviction(plan)

if not result.quota_satisfied:
    # Pinned, missing, changed, or otherwise unsafe entries can block cleanup.
    print(result.to_dict()["remaining_bytes"])
```

Applying a plan removes only artifacts that are still registered, unchanged,
inside the configured cache directory, and unpinned. Unregistered files are
left untouched. Use `dry_run=True` to inspect the candidate count without
deleting anything.
