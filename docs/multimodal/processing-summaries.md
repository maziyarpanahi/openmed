# Multimodal Processing Summaries

At the end of a multimodal run, aggregate what happened to every asset into a
single, inspectable completion artifact with `summarize_processing_run()`.
The result, `ProcessingSummary`, contains only counts, byte/page/frame
totals, and opaque `asset_id`/SHA-256 values — never OCR text, transcripts,
DICOM tag values, file paths, or any other source content.

```python
from openmed.multimodal.abstention import AbstentionReason, AbstentionRecord, AbstentionStage
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.digest import AssetDigest
from openmed.multimodal.processing_summary import (
    AssetProcessingResult,
    ProcessingOutcome,
    render_processing_summary_markdown,
    summarize_processing_run,
)

results = [
    AssetProcessingResult(
        manifest=AssetManifest(
            asset_id="asset-001",
            media_type="image/png",
            sha256="a" * 64,
            byte_size=204800,
        ),
        outcome_code=ProcessingOutcome.SUCCESS,
        duration_seconds=0.42,
        input_digest=AssetDigest(sha256="a" * 64, byte_count=204800),
        output_digest=AssetDigest(sha256="b" * 64, byte_count=1024),
    ),
    AssetProcessingResult(
        manifest=AssetManifest(
            asset_id="asset-002",
            media_type="audio/wav",
            sha256="c" * 64,
            byte_size=512000,
        ),
        outcome_code=ProcessingOutcome.ABSTAINED,
        duration_seconds=0.05,
        input_digest=AssetDigest(sha256="c" * 64, byte_count=512000),
        abstention=AbstentionRecord(
            stage=AbstentionStage.INFERENCE,
            reason=AbstentionReason.PHI_UNCERTAINTY,
        ),
    ),
]

summary = summarize_processing_run(results)
print(summary.to_json())
print(render_processing_summary_markdown(summary))
```

## What this guarantees

- **Metadata-only.** Every field on `ProcessingSummary` is a count, a
  byte/page/frame total, a stable code, or an opaque `asset_id`/SHA-256
  value already validated as non-identifying by `AssetManifest` and
  `AssetDigest`.
- **Fail closed.** `summarize_processing_run()` validates every asset's
  `outcome_code` against the fixed `ProcessingOutcome` set before
  aggregating anything. An unrecognized code raises
  `ProcessingSummaryError` and no summary — partial or otherwise — is
  produced.
- **Deterministic.** `by_media_type`, `outcome_counts`,
  `abstention_counts`, and `asset_digests` are always sorted, so
  `to_dict()`/`to_json()` are identical no matter what order assets were
  processed in.

## What this does not do

Out of scope for this artifact: it does not store source content, generate
thumbnails, back a dashboard, or make any clinical-quality claim about the
run. It is a completion record, not an audit log of what a model saw.

## Schema

`PROCESSING_SUMMARY_SCHEMA_VERSION` is currently `1`. Adding, removing, or
reinterpreting a field on `ProcessingSummary` requires bumping this version.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | Contract version, currently `1`. |
| `total_assets` | int | Number of assets in the run. |
| `total_bytes` | int | Sum of `manifest.byte_size` across all assets. |
| `total_duration_seconds` | float | Sum of per-asset `duration_seconds`. |
| `by_media_type` | list | One entry per distinct `media_type`, sorted by that string: `media_type`, `count`, `total_bytes`, `total_pages`, `total_frames`. |
| `outcome_counts` | list | One entry per `ProcessingOutcome` present, in enum declaration order: `outcome`, `count`. |
| `abstention_counts` | list | One entry per distinct `(stage, reason)` pair present, sorted by `(stage, reason)`: `stage`, `reason`, `count`. |
| `asset_digests` | list | One entry per asset, sorted by `asset_id`: `asset_id`, `input_sha256`, `output_sha256` (omitted when there was no output). |
| `asset_count_with_output_digest` | int | Number of assets that produced an output digest. |

## Outcome codes

`ProcessingOutcome` is a fixed, closed set:

| Code | Meaning |
| --- | --- |
| `success` | The asset was processed to completion. |
| `abstained` | Processing stopped via an `AbstentionRecord`; see [Abstention Reasons](abstention-reasons.md). |
| `error` | Processing failed for a reason other than a documented abstention. |

Any `outcome_code` outside this set — on any asset in the run — causes
`summarize_processing_run()` to raise `ProcessingSummaryError` immediately.
There is no "unknown" bucket; the boundary fails closed rather than
silently guessing.

## Example output

```json
{"schema_version":1,"total_assets":2,"total_bytes":716800,"total_duration_seconds":0.47,"by_media_type":[{"media_type":"audio/wav","count":1,"total_bytes":512000,"total_pages":0,"total_frames":0},{"media_type":"image/png","count":1,"total_bytes":204800,"total_pages":0,"total_frames":0}],"outcome_counts":[{"outcome":"success","count":1},{"outcome":"abstained","count":1}],"abstention_counts":[{"stage":"inference","reason":"phi_uncertainty","count":1}],"asset_digests":[{"asset_id":"asset-001","input_sha256":"aaaa...","output_sha256":"bbbb..."},{"asset_id":"asset-002","input_sha256":"cccc..."}],"asset_count_with_output_digest":1}
```

```markdown
# Processing Summary

- Schema version: 1
- Total assets: 2
- Total bytes: 716800
- Total duration (seconds): 0.47
- Assets with an output digest: 1

## By Media Type

| Media Type | Count | Total Bytes | Total Pages | Total Frames |
| --- | --- | --- | --- | --- |
| audio/wav | 1 | 512000 | 0 | 0 |
| image/png | 1 | 204800 | 0 | 0 |

## Outcomes

| Outcome | Count |
| --- | --- |
| success | 1 |
| abstained | 1 |

## Abstentions

| Stage | Reason | Count |
| --- | --- | --- |
| inference | phi_uncertainty | 1 |

## Asset Digests

| Asset ID | Input SHA-256 | Output SHA-256 |
| --- | --- | --- |
| asset-001 | aaaa... | bbbb... |
| asset-002 | cccc... | |
```
