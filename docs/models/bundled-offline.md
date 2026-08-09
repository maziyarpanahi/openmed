# Bundled offline model bootstrap

OpenMed exposes one small PII model as an explicit, registry-backed offline
bundle. The package carries a versioned manifest containing the registry key,
exact model id, reproducibility checksum, and license. The bundle is opt-in;
importing OpenMed or constructing `ModelLoader` does not select it.

The model snapshot must already be present in the configured OpenMed cache (for
example, in an application or air-gap bundle built during deployment). The
bootstrap never downloads a missing snapshot. It resolves the model through the
ordinary registry, passes `local_files_only=True`, verifies the registry
revision using the normal integrity path, and blocks sockets for the complete
load. A missing or tampered local snapshot fails closed.

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
not contain the pinned snapshot, prepare the deployment bundle or surface the
error to the operator; do not enable a remote fallback for clinical text.

This is a local inference bootstrap, not a medical device or a clinical
decision guarantee. Examples and tests use synthetic, non-PHI data only.
