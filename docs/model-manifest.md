# Model Manifest

`models.jsonl` is the canonical OpenMed model catalog. Each line is one JSON
object keyed by `repo_id`, and the registry loads it without requiring network
access.

## Core fields

Every row must include:

| Field | Description |
|---|---|
| `repo_id` | Hugging Face model id, for example `OpenMed/...`. |
| `family` | Model family such as `NER`, `PII`, `Vision`, or `ZeroShot`. |
| `task` | Pipeline task, such as `token-classification`. |
| `languages` | List of BCP 47-style language codes. |
| `tier` | Published size tier, or `null` when unavailable. |
| `param_count` | Parameter count as an integer, or `null`. |
| `architecture` | Backbone architecture, or `null`. |
| `base_model` | Source model id when known, or `null`. |
| `formats` | Available artifact formats such as `pytorch`, `mlx-fp`, or `onnx`. |
| `canonical_labels` | Canonical OpenMed entity labels supported by the model. |
| `benchmark` | Legacy benchmark object or enriched suite list. |
| `arxiv` | Related arXiv id, or `null`. |
| `license` | Model license identifier, or `null` when not declared. |
| `reproducibility_hash` | Stable `sha256:<64 lower hex>` provenance hash. |
| `released` | Release date as `YYYY-MM-DD`, or `null`. |

## Enrichment fields

Benchmark and device measurements are optional. Existing rows without these
fields remain valid.

| Field | Shape | Description |
|---|---|---|
| `latency_ms` | object | Per-device latency map in milliseconds. Keys are device labels and values are non-negative numbers. |
| `peak_ram_mb` | object | Per-device peak RAM map in megabytes. Keys are device labels and values are non-negative numbers. |
| `recommended_tier` | string | One of `phone`, `laptop`, `workstation`, or `server`. |

The legacy benchmark shape remains valid:

```json
{"dataset":"openmed-golden-pii","micro_f1":0.9823,"recall":0.991,"leakage":0.0}
```

Enriched rows may replace it with a suite list:

```json
[
  {
    "suite": "shield",
    "dataset": "openmed-golden-pii",
    "micro_f1": 0.9823,
    "recall": 0.991,
    "leakage": 0.0
  }
]
```

Only aggregate metrics belong in the manifest. Do not store raw PHI, prompts,
documents, or examples in benchmark or device result files.

## Refresh and enrich

Regenerate the base manifest from the Hugging Face org:

```bash
python scripts/manifest/generate_manifest.py --output models.jsonl
```

Merge benchmark and latency results into a new manifest:

```bash
python scripts/manifest/enrich_manifest.py \
  --manifest models.jsonl \
  --results benchmark-results.json \
  --output models.enriched.jsonl
```

The results JSON can be a list, a top-level `results` list, or an object keyed
by `repo_id`:

```json
{
  "results": [
    {
      "repo_id": "OpenMed/OpenMed-PII-Fixture-Tiny-65M",
      "latency_ms": {"iphone_15_pro": 18.4, "m2_air": 7.0},
      "peak_ram_mb": {"iphone_15_pro": 512, "m2_air": 384},
      "recommended_tier": "phone",
      "benchmark": [
        {
          "suite": "shield",
          "dataset": "openmed-golden-pii",
          "micro_f1": 0.9823,
          "recall": 0.991,
          "leakage": 0.0
        }
      ]
    }
  ]
}
```

Rows whose `repo_id` has no measurement are written byte-identically to the
output. Rows with measurements are schema-validated after merging.
