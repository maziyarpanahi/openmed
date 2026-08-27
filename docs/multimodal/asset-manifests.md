# Privacy-safe asset manifests

`openmed.multimodal.asset_manifest` defines a small manifest for preflight
steps that need to describe an input without reading or storing the input
itself.

The manifest is intentionally narrow. It records only:

- `version`
- `asset_id`
- `media_type`
- `sha256`
- `byte_size`
- optional bounded counts: `pages`, `width`, `height`, `frames`,
  `duration_seconds`

The validator rejects unknown fields, paths, URLs, and free-text fields such as
descriptions or source metadata. Validation errors name the failed field or rule
without echoing the supplied value.

```python
from openmed.multimodal.asset_manifest import AssetManifest

manifest = AssetManifest.from_dict(
    {
        "asset_id": "dicom-001",
        "media_type": "application/dicom",
        "sha256": "a" * 64,
        "byte_size": 4096,
        "frames": 12,
        "width": 512,
        "height": 512,
    }
)

payload = manifest.to_json()
```

`to_dict()` emits fields in a stable order and omits unset optional fields.
`to_json()` emits compact JSON with sorted keys so callers can compare manifest
payloads deterministically.

The manifest does not read assets, perform OCR or inference, retain embedded
metadata, or define model-specific tensor shapes.
