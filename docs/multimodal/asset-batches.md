# Privacy-safe asset batches

`openmed.multimodal.asset_batch` groups several
[asset manifests](asset-manifests.md) into one versioned batch so a clinical
packet with mixed media can be ordered, de-duplicated, and summarized before
any asset is opened.

A batch records only:

- `version`
- `batch_id`, an opaque identifier with the same format as `asset_id`
- `assets`, the ordered manifests
- derived totals: `asset_count`, `total_bytes`, `total_pages`,
  `total_frames`, `total_duration_seconds`

The validator rejects unknown fields, paths, URLs, inline manifests that fail
manifest validation, duplicate asset identifiers, repeated content digests,
non-canonical ordering, batches above the hard cap of 10,000 assets, aggregate
totals that overflow the manifest bounds, and declared totals that disagree
with the manifests. Errors and findings name a reason code, a fixed schema
field, or an asset position and never echo a supplied value.

## Canonical order

Assets are stored strictly ascending by `asset_id`. `AssetBatch.build()` sorts
its input; `from_dict()`, `from_json()`, and direct construction reject any
other order with `order_not_canonical`. Two batches that hold the same
manifests therefore always serialize to the same bytes.

```python
from openmed.multimodal.asset_batch import AssetBatch, validate_asset_batch
from openmed.multimodal.asset_manifest import AssetManifest

pdf = AssetManifest.from_dict(
    {
        "asset_id": "pdf-001",
        "media_type": "application/pdf",
        "sha256": "a" * 64,
        "byte_size": 2048,
        "pages": 4,
    }
)
audio = AssetManifest.from_dict(
    {
        "asset_id": "audio-001",
        "media_type": "audio/wav",
        "sha256": "b" * 64,
        "byte_size": 8192,
        "duration_seconds": 3.5,
    }
)

batch = AssetBatch.build("packet-001", [pdf, audio])
batch.total_pages  # 4
batch.total_duration_seconds  # 3.5
payload = batch.to_json()

assert validate_asset_batch(batch) == []
assert AssetBatch.from_json(payload) == batch
```

## Derived totals

Totals are computed from the manifests rather than stored, so a Python batch
can never hold inconsistent counts. `to_dict()` and `to_json()` always emit
every total, including zero values for modalities that are absent. When a
serialized payload declares totals they are optional but must match the
derived values; `aggregate_invalid` reports a wrong type or a non-finite
number and `aggregate_mismatch` reports a disagreement. Durations are summed
with `math.fsum` and compared with a `1e-9` tolerance.

## Findings

`validate_asset_batch()` returns a sorted, de-duplicated list of
`BatchFinding` records instead of raising, which lets preflight reports
collect every problem in one pass. Each finding holds a `reason_code`, an
optional `field_name` limited to the batch schema, and an optional zero-based
asset `position`.

| Reason code | Meaning |
| --- | --- |
| `invalid_batch` | The payload is not a readable mapping. |
| `unknown_field` | The payload contains a key outside the schema. |
| `missing_required` | `batch_id` or `assets` is absent. |
| `invalid_version` | `version` is not the supported integer. |
| `invalid_batch_id` | `batch_id` is not an opaque identifier. |
| `invalid_assets` | `assets` is not a sequence of manifests. |
| `invalid_asset` | The manifest at `position` failed manifest validation. |
| `duplicate_asset_id` | The manifest at `position` repeats an earlier `asset_id`. |
| `duplicate_sha256` | The manifest at `position` repeats an earlier digest. |
| `order_not_canonical` | The manifest at `position` is out of ascending order. |
| `batch_too_large` | The batch exceeds `max_assets` or the hard cap. |
| `aggregate_overflow` | A derived total exceeds the manifest bound. |
| `aggregate_invalid` | A declared total has the wrong type or is non-finite. |
| `aggregate_mismatch` | A declared total disagrees with the manifests. |
| `empty_batch` | The batch is empty and `allow_empty` is false. |

`max_assets` and `allow_empty` are caller policy. Both `validate_asset_batch()`
and the loaders accept them; `from_dict()` and `from_json()` raise
`AssetBatchError` when any finding is present.

The batch does not open assets, decide processing order from clinical
semantics, run inference, or de-duplicate or delete source files.
