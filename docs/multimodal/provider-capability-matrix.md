# Provider capability compatibility matrix

Operators can compare local provider declarations with validated asset-profile
requirements before loading a provider. This metadata adapter performs no
inference, media decoding, transformation, provider probing or automatic choice.
A compatible declaration is not proof that an installed runtime or model works,
nor clinical qualification.

```python
from openmed.multimodal.capability_matrix import (
    ModalityCapability, ModalityProfile, ProviderCapabilityEnvelope,
    capability_matrix, render_capability_matrix_json,
    render_capability_matrix_markdown,
)

providers = [ProviderCapabilityEnvelope(
    provider_id="local_audio", capability_version="1.0.0",
    capabilities=(ModalityCapability(
        modality="audio", exact_features=("mono", "pcm"),
        transformable_features=("stereo",),
    ),),
)]
profiles = [ModalityProfile("prepared_audio", "audio", ("mono", "pcm"))]
rows = capability_matrix(providers, profiles)
assert rows[0].outcome == "exact"
print(render_capability_matrix_json(rows))
print(render_capability_matrix_markdown(rows))
```

The schema version is `1` for profile and envelope declarations. Capability
versions are bounded numeric triples. Provider IDs, profile IDs and feature
codes contain lowercase letters, digits, underscores or hyphens, start with a
letter and are limited to 64 characters. Use opaque controlled codes only;
never use model paths, endpoint URLs, filenames, reviewer names or patient text.

This module introduces an explicit adapter vocabulary because provider runtime
classes do not expose one common capability envelope. Application-owned adapters
must derive feature codes from validated profiles and use the same vocabulary
as their provider declarations. The matrix does not infer these codes from
`AssetManifest`, audio headers or models, and does not alter those contracts.

| Outcome | Declared meaning | Reason |
| --- | --- | --- |
| `exact` | Every required feature is directly accepted for the modality | None |
| `transform_required` | At least one requirement is declared transformable; all others are directly accepted or transformable | `declared_transform_required` |
| `unsupported` | Modality is absent from an explicit capability inventory | `modality_unsupported` |
| `unsupported` | At least one requirement has no accepted or transformable declaration | `feature_unsupported` |
| `missing_capability` | Envelope exists but its capability inventory is `None` | `capability_missing` |

An empty capability tuple explicitly supports no modalities; it differs from
missing capability evidence. An unsupported feature takes precedence over a
transformable one. Exact and transformable feature inventories cannot overlap.
Declaring a transformation does not authorize or execute it: use an independent
validated transformation plan and rebuild the profile before inference.

Cross products accept at most 100 providers and 100 profiles. Each modality
allows 128 feature codes; supported modality codes are `image`, `pdf`, `dicom`,
`audio`, `waveform` and `video`. Inputs are normalized into immutable declarations;
duplicate provider IDs, profile IDs, modalities, features or rendered cells
reject. Iterables are consumed only up to their bound plus one.

JSON and Markdown contain provider IDs, capability versions, profile IDs,
outcomes and stable reasons only. Sorting is independent of declaration order.
Missing declarations never trigger network discovery or provider installation.
