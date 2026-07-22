# Public Benchmark Leaderboard

The [public benchmark leaderboard](https://openmed.life/docs/eval/benchmark-leaderboard/)
is a static, versioned view of OpenMed's archived synthetic evaluation reports.
It is rebuilt without network access whenever the documentation site publishes
from `master` and whenever a `v*` release tag publishes the site.

The renderer ranks every row by leakage ascending and then recall descending.
F1 and recall remain visible as secondary metrics, but a higher F1 can never
outrank a report with lower leakage. Tabs separate benchmark suites, and model
family headings keep comparisons within the same model class legible.

## Public artifacts

The stable publication path is
`https://openmed.life/docs/eval/benchmark-leaderboard/`. Each render contains:

- `index.html`, the browsable leaderboard with per-suite tabs;
- `leaderboard.json`, the same rows in a versioned machine-readable schema;
- `reports/`, canonical JSON copies of the archived reports linked from every
  row.

Every row records the release tag, benchmark run date, and a
`sha256:<64 lower hex>` reproducibility hash. Older reports may obtain model
family and reproducibility metadata from the matching `models.jsonl` entry,
but missing ranking evidence fails the render instead of publishing an
unverifiable row.

## Source and refresh flow

Only committed `BenchmarkReport` JSON under `docs/benchmarks/` with
`metadata.synthetic` set to `true` is published. Auxiliary JSON that has no
BenchmarkReport fields is ignored, while malformed report candidates and
reports without the explicit synthetic marker fail closed. Non-synthetic,
private, DUA-restricted, or raw clinical data must never be added to this
archive.

To reproduce the release artifact locally:

```bash
.venv/bin/python -m openmed.eval.leaderboard \
  --reports-dir docs/benchmarks \
  --output-dir docs/eval/benchmark-leaderboard \
  --manifest models.jsonl \
  --release-tag vX.Y.Z
```

The renderer sorts input paths and JSON keys, derives no wall-clock timestamp,
and writes canonical report copies. Re-running it with the same inputs is
therefore byte-identical and does not require a network connection.

## Machine-readable contract

`leaderboard.json` uses schema version `1`. Its top-level `rows` array is the
canonical global ranking and declares the sort keys as `leakage:asc` and
`recall:desc`. The `suites` view repeats those rows under deterministic suite
and model-family groups for consumers that mirror the HTML navigation.

Each row includes:

| Field | Meaning |
|---|---|
| `suite` | Benchmark suite shown as a tab. |
| `model_family` | Manifest or report model family. |
| `model_name` / `device` | Evaluated model and runtime. |
| `leakage` | Primary lower-is-better ranking metric in `[0, 1]`. |
| `recall` / `f1` | Secondary higher-is-better metrics in `[0, 1]`. |
| `release_tag` | Release that published or originally stamped the report. |
| `run_date` | ISO `YYYY-MM-DD` date derived from the report timestamp. |
| `reproducibility_hash` | Validated SHA-256 reproducibility identifier. |
| `report_url` | Relative download URL for the underlying report JSON. |
