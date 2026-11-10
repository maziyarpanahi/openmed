# Trust Status Contract

This document defines the committed contract for the trust status surface.
The rendered page at `docs/status/index.md` is produced by
`scripts/status/generate_status.py` from repository-owned inputs. The next
renderer slice is tracked in #375 and should consume this contract rather than
introducing a second status shape.

## Source Inputs

| Input | Current path | Role |
|---|---|---|
| Model manifest | `models.jsonl` | Canonical model family, tier, format, release, and manifest reproducibility data. |
| Last-green baseline store | `gates/baseline.json` | Release baseline per family, tier, and format. |
| Benchmark reports | `docs/benchmarks/*.report.json` | Latest harness metrics and run timestamp for each tested model. |
| Status renderer | `scripts/status/generate_status.py` | Joins the manifest, baseline store, and reports into status and leaderboard rows. |

The row key is `baseline_key(family, tier, format)` from
`openmed.core.baseline`. Key parts are lowercased, `_` is normalized to `-`,
and a missing tier is stored as `none`.

## Required Row Fields

Each status row is keyed by family, tier, and format. A future nightly JSON
bundle may denormalize the values below, but it must keep these names and
meanings stable:

| Field | Required | Source | Meaning |
|---|---:|---|---|
| `key` | yes | `baseline_key(...)` | Stable join key for baseline and report lookups. |
| `family` | yes | `models.jsonl` | Model family displayed on status and publication pages. |
| `tier` | yes | `models.jsonl` | Model tier, or `null` when the manifest has no tier. |
| `device` | yes | `BenchmarkReport.device` | Device or runtime used by the latest harness report. |
| `format` | yes | `models.jsonl` / report metadata | Artifact format that the row represents. |
| `model_count` | yes | manifest aggregate | Number of manifest rows in this family, tier, and format group. |
| `current_leakage` | yes | `metrics.leakage.overall` or `metrics.leakage_rate.overall` | Decimal leakage rate in the range `[0, 1]`; render as a percent. |
| `last_green_release` | yes | baseline `released` | Last release date that passed gates, formatted as `YYYY-MM-DD`. |
| `last_regression` | yes | baseline `metadata.last_regression` | Most recent known regression date, or `null`. |
| `last_rollback` | yes | baseline `metadata.last_rollback` | Rollback date for the latest known regression, or `null`. |
| `harness_freshness` | yes | `BenchmarkReport.generated_at` | Timestamp for the latest successful harness run, formatted as UTC ISO 8601. |
| `status` | yes | renderer | `green`, `amber`, or `red`. |
| `evidence` | yes | report suite / artifact path | Human-inspectable evidence for the row, such as `golden` or a report path. |
| `reproducibility_hash` | yes | manifest or baseline | Stable `sha256:<64 lower hex>` hash for the release or benchmark row. |

Missing report data currently renders as `n/a` and `amber`. The scheduled
nightly refresh should fail clearly when a required input is malformed or when
freshness is older than the accepted nightly window.

## Nightly Bundle Shape

The nightly job may write a bundle for downstream static rendering. It should be
deterministic JSON with sorted keys and no raw clinical text:

```json
{
  "schema_version": 1,
  "generated_at": "2026-06-14T00:00:00Z",
  "source_revision": "abc123",
  "inputs": {
    "baseline": "gates/baseline.json",
    "manifest": "models.jsonl",
    "reports": ["docs/benchmarks/golden.report.json"]
  },
  "rows": [
    {
      "current_leakage": 0.0,
      "device": "mlx-fp",
      "evidence": "golden",
      "family": "PII",
      "format": "mlx-fp",
      "harness_freshness": "2026-06-14T00:00:00Z",
      "key": "pii::small::mlx-fp",
      "last_green_release": "2026-04-14",
      "last_regression": null,
      "last_rollback": null,
      "model_count": 50,
      "reproducibility_hash": "sha256:4818bdef580eb406f5cc665cc0892aefab1fd90c8743822d843dd48f135bde34",
      "status": "green",
      "tier": "Small"
    }
  ]
}
```

Rows are sorted by `key`. Consumers must ignore unknown fields so future
children can add latency, RAM, confidence intervals, or device-tier metadata
without breaking the status page.

## Reproducibility Hash

Use `openmed.core.repro_hash.compute_reproducibility_hash` for benchmark and
release rows. The hash is:

```text
sha256(canonical_json({
  "base_model": base_model,
  "data_manifest": data_manifest,
  "git_sha": git_sha,
  "recipe": recipe
}))
```

Normalization is the implementation in `openmed/core/repro_hash.py`:

- mappings are sorted by string key;
- lists and tuples keep order;
- sets are sorted by representation;
- bytes are replaced by their SHA-256 digest;
- file paths are represented by path plus file digest;
- directory paths are represented by path plus a sorted list of file digests.

The stored value must use the form `sha256:<64 lower hex>`. The input
components are intentionally explicit:

- `recipe`: training, conversion, quantization, or evaluation recipe;
- `data_manifest`: dataset name, split, revision, filters, and fixture digest;
- `base_model`: source model identifier or immutable local model digest;
- `git_sha`: repository revision that produced the artifact or benchmark row.

## Relationship To Current Outputs

The current status and publication outputs already reuse the OM-021/#104
renderer:

- `docs/status/index.md` renders the status table fields from manifest,
  baseline, and `BenchmarkReport` inputs.
- `docs/leaderboard/index.md` renders the open benchmark rows from the same
  source inputs.
- `docs/benchmarks/golden.report.json` and `docs/benchmarks/golden.md` provide
  committed report evidence for the current golden harness row.

#375 should make the status rendering job read a nightly bundle with this shape
while preserving the current generated status columns.
