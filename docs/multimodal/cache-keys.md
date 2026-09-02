# Content-free multimodal cache keys

`build_multimodal_cache_key()` derives a stable local cache key from processing
metadata without hashing source bytes itself or putting content in the key.
Callers provide a previously computed asset SHA-256 digest, media type,
provider version, model version, policy version, and any declared preprocessing
options.

```python
from openmed.multimodal.cache_key import build_multimodal_cache_key
from openmed.multimodal.digest import digest_asset

asset = digest_asset(b"synthetic image bytes")
key = build_multimodal_cache_key(
    asset_digest=asset,
    media_type="image/png",
    provider_version="doctr-1.0",
    model_version="vision-2.1",
    policy_version="redact-v3",
    preprocessing_options={
        "color_mode": "rgb",
        "resize_mode": "fit",
        "target_width": 512,
        "target_height": 512,
    },
)
```

The result has the form `openmed-multimodal-v1:<sha256>`. Mapping order does
not affect it. Changing the asset digest, media type, provider version, model
version, policy version, or a preprocessing option changes the key.

## Preprocessing options

Only the following options are accepted:

| Option | Allowed values |
| --- | --- |
| `channel_mode` | `mono`, `native`, `stereo` |
| `color_mode` | `grayscale`, `native`, `rgb`, `rgba` |
| `orientation_mode` | `normalize`, `preserve` |
| `resize_mode` | `fill`, `fit`, `none` |
| `clip_duration_seconds` | finite number greater than 0 and at most 86,400 |
| `frame_stride` | integer from 1 through 1,000,000 |
| `render_dpi` | integer from 1 through 9,600 |
| `sample_rate_hz` | integer from 1 through 768,000 |
| `target_height` | integer from 1 through 65,536 |
| `target_width` | integer from 1 through 65,536 |

Unknown names, arbitrary categorical strings, booleans in numeric fields,
nested values, non-finite numbers, and out-of-range numbers fail closed.

## Privacy boundary

Version fields accept only bounded metadata tokens. Paths, URLs, clinical text,
credentials, prompts, raw bytes, and free-form option values are rejected
without being echoed in errors. The returned key contains only a public schema
prefix and a SHA-256 digest.

The helper performs no file access, network access, media hashing, cache
storage, eviction, or encryption. Callers remain responsible for computing the
asset digest and declaring every preprocessing choice that affects their model.
