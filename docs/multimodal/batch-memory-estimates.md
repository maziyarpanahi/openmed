# Multimodal Batch Memory Estimates

`openmed.multimodal.batch_memory` estimates the decoded memory of a multimodal
batch from validated [asset manifests](asset-manifests.md) before anything is
decoded, and compares it with an explicit memory budget. The result is one of
three plans: **accept** the batch as given, **split** it into ordered batches
that each fit, or **reject** it.

An estimate holds only under the documented assumptions of the policy that
produced it. It is not a guarantee that decoding or inference fits, and it
does not measure live memory, plan model weights, decode media, or schedule
inference.

## Inputs

`plan_batch_memory(assets, policy, *, budget_bytes, overhead_bytes)` takes:

- `assets`: a sequence of `AssetManifest` objects, planned in the order given.
- `policy`: a frozen `MemoryEstimationPolicy` holding the factors a manifest
  cannot carry.
- `budget_bytes`: the decoded-memory budget, an inclusive ceiling. Equal
  passes; zero rejects.
- `overhead_bytes`: a fixed overhead charged once per planned batch.

Both budget arguments are required, with no defaults and no built-in hardware
profiles. The admission ceilings in `MOBILE_V1` and `DESKTOP_V1` are not reused.

## Estimation policy

Every policy factor is either `None` or a bounded positive integer. None has a
default, and none is guessed: a factor the caller does not supply makes the
assets that need it unevaluable.

| Factor | Bound | Used for |
| --- | --- | --- |
| `image_bytes_per_pixel` | 1 to 1024 | images, per frame |
| `dicom_bytes_per_pixel` | 1 to 1024 | DICOM, per frame |
| `audio_sample_rate_hz` | 1 to 2^32 - 1 | audio waveforms |
| `audio_channels` | 1 to 1024 | audio waveforms |
| `audio_bytes_per_sample` | 1 to 1024 | audio waveforms, per channel |

Booleans, floats (including NaN and infinities), negative numbers, and values
outside these bounds are rejected with `BatchMemoryError`.

## Per-asset estimates

The modality comes from the manifest media type:

| Modality | Estimate | Manifest fields |
| --- | --- | --- |
| image | width x height x frames x `image_bytes_per_pixel` | `width`, `height`, `frames` |
| DICOM | width x height x frames x `dicom_bytes_per_pixel` | `width`, `height`, `frames` |
| audio | ceil(duration x rate) x channels x `audio_bytes_per_sample` | `duration_seconds` |
| PDF | never estimated | none |

The audio frame count is the exact rational ceiling of `duration_seconds`
times the sample rate, so the result does not depend on float rounding.

Image frame count must be supplied explicitly, even for a still image
(`frames=1`). GIF, WebP, APNG, and TIFF may contain multiple frames. Without
the count, a one-frame estimate would not bound a decoder that materializes
every frame, so the planner rejects with `insufficient_metadata`.

A PDF under the PDF manifest profile carries no raster geometry, and a page
count alone cannot stand in for it, so a PDF is always unevaluable.
`application/dicom+json` has no geometry profile and is unevaluable too.

## Outcomes

Each asset gets a content-free `AssetMemoryEstimate`: its position, modality,
estimated bytes, and, when it failed, a reason code.

| Reason code | Meaning | `estimated_bytes` |
| --- | --- | --- |
| `insufficient_metadata` | a manifest field or policy factor is missing; `field_name` names the first one | `null` |
| `estimate_saturated` | the estimate would pass the saturation ceiling | `null` |
| `budget_exceeded` | the asset plus one overhead does not fit the budget | the estimate |

If any asset has a reason code, the plan is `reject` and carries no batches.
Splitting cannot cure missing evidence, saturation, or a single asset that
does not fit on its own. Otherwise:

- `accept` when every asset fits in one batch with one overhead.
- `split` when it does not. Assets are packed greedily into contiguous
  batches in input order, and each batch is charged the overhead. This gives
  the fewest batches that preserve order, and every batch fits the budget.

## Arithmetic

All arithmetic is integer and platform-independent. Estimates, budgets,
overheads, and batch totals are held at or below `SATURATION_CEILING`,
2^63 - 1. A product that would exceed it saturates, and a saturated asset
always rejects the plan, so saturation never produces an accept.

## Example

```python
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.batch_memory import (
    MemoryEstimationPolicy,
    plan_batch_memory,
)

scan = AssetManifest(
    asset_id="scan-001",
    media_type="image/png",
    sha256="a" * 64,
    byte_size=4096,
    width=1024,
    height=1024,
    frames=1,
)
dictation = AssetManifest(
    asset_id="dictation-001",
    media_type="audio/wav",
    sha256="b" * 64,
    byte_size=8192,
    duration_seconds=30.0,
)
policy = MemoryEstimationPolicy(
    image_bytes_per_pixel=4,
    audio_sample_rate_hz=16_000,
    audio_channels=1,
    audio_bytes_per_sample=4,
)
plan = plan_batch_memory(
    [scan, dictation],
    policy,
    budget_bytes=5 * 1024**2,
    overhead_bytes=256 * 1024,
)
plan.outcome.value  # "split": 4 MiB + 1.83 MiB does not fit 5 MiB together
[batch.positions for batch in plan.batches]  # [(0,), (1,)]
```

`plan.to_json()` serializes with a fixed key order. It carries positions,
modalities, byte counts, and reason codes only, never asset identifiers,
digests, paths, or content.
