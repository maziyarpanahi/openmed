# Bundled offline model bootstrap

OpenMed exposes one small PII model as an explicit, registry-backed offline
bundle. The package carries a versioned manifest containing the registry key,
exact model id, reproducibility checksum, and license. The bundle is opt-in;
importing OpenMed or constructing `ModelLoader` does not select it.

The model snapshot must already be present in the configured OpenMed cache (for
example, in an application or air-gap bundle built during deployment). The
bootstrap never downloads a missing snapshot. It resolves the model through the
ordinary registry, passes `local_files_only=True`, requires the exact cached
artifact-integrity sidecar for the pinned registry revision, and blocks sockets
for the complete load. Missing, stale, skipped, or tampered integrity evidence
fails closed with a content-free bootstrap error. An integrity-required load
rebuilds the model and tokenizer from that verified path instead of trusting a
model object cached earlier under a permissive policy.

The socket guard is process-wide. Overlapping guarded loads keep it active
until the final guarded scope exits, so one thread cannot reopen egress while
another offline load is still running. Other threads in the same process are
also prevented from opening sockets during that interval.

```python
from openmed.core import OpenMedConfig
from openmed.models.bundled import (
    get_bundled_model_manifest,
    load_bundled_model,
)

manifest = get_bundled_model_manifest()
print(manifest.version, manifest.model_id, manifest.license)

model = load_bundled_model(
    config=OpenMedConfig(
        cache_dir="/opt/openmed/model-cache",
        device="cpu",
    )
)
```

The returned value has the same shape as `ModelLoader.load_model`, so callers
can use the normal tokenizer/model pipeline construction. If the cache does
not contain the pinned snapshot and its integrity sidecar, prepare both through
the normal verified model-loading path on a connected build host before the
offline hand-off:

```python
from openmed.core import OpenMedConfig
from openmed.core.models import ModelLoader

config = OpenMedConfig(cache_dir="/opt/openmed/model-cache")
ModelLoader(config).load_model("pii_detection", require_integrity=True)
```

Transfer the complete cache, including its `integrity/` sidecars, into the
offline deployment. If preparation is not possible, surface the bootstrap
error to the operator; do not enable a remote fallback for clinical text.
An explicitly injected custom loader receives both `local_files_only=True` and
`require_integrity=True` and is responsible for honoring that contract.

This is a local inference bootstrap, not a medical device or a clinical
decision guarantee. Examples and tests use synthetic, non-PHI data only.
