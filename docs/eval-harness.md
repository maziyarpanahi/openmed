# Eval Harness & Metrics

`run_benchmark` executes a model over a sequence of `BenchmarkFixture` objects and returns a
`BenchmarkReport` whose `metrics` dict contains the standard OM-018 metric bundle.

## Chinese clinical NER

The `chinese-clinical-ner` suite ships a tiny synthetic CMeEE-shaped fixture
for offline CI. It reports exact precision and recall per canonical label and
applies a zero-tolerance PHI-token leakage gate to injected synthetic
identifiers. Leakage findings contain hashes and offsets, never identifier
text.

The bundled Chinese PII default is a documented multilingual routing
placeholder, not a dedicated Chinese clinical NER checkpoint. CMeEE, CBLUE,
eHealth corpora, and related model weights are not redistributed: callers must
provision licensed assets outside the repository and pass an explicit local
path to `load_cmeee`. Missing paths and repository-internal real-data paths fail
with license-boundary guidance.

## Metric Bundle

| Metric | Path | Gating? | Description |
| --- | --- | --- | --- |
| Latency p50 | `latency.p50_ms` | No | Median steady-state fixture latency in ms. |
| Latency p95 | `latency.p95_ms` | No | 95th-percentile steady-state fixture latency in ms. |
| Latency count | `latency.count` | No | Number of steady-state fixtures (excludes cold start). |
| Cold-start latency | `latency.cold_start_ms` | No | Wall-clock latency of the first fixture call in ms. |
| Peak RSS | `resources.peak_rss_bytes` | No | Peak resident set size in bytes during the run. |

## Edge Metrics

### cold_start_ms

The harness records the wall-clock latency of the **first** fixture call separately. The default
runner keeps a shared model loader for the duration of the benchmark run, so that first call encloses
model and tokenizer loading plus the first forward pass. Later fixture calls reuse the warmed loader
and feed the steady-state latency summary. The value is surfaced at:

```
report.metrics['latency']['cold_start_ms']
```

It is **excluded** from the steady-state `p50_ms`, `p95_ms`, and `count` values.

!!! note "Reported, not gating"
    `cold_start_ms` does not participate in any release gate. It is an observability metric
    intended to track model-load overhead over time — not a pass/fail criterion.

```python
report = run_benchmark(fixtures, suite="my-suite", model_name="my-model", runner=runner)
cold_ms = report.metrics["latency"]["cold_start_ms"]
print(f"Cold-start latency: {cold_ms:.1f} ms")
```
