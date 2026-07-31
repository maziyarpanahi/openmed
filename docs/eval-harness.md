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

## Grounding accuracy gate

`openmed/eval/grounding_accuracy.py` measures whether the sparse candidate
generator maps a clinical mention to the **right** coded concept. It ships a
fully synthetic gold set under `openmed/eval/golden/grounding/` (one JSONL per
permissive system: RxNorm, LOINC, ICD-10-CM) and reports **top-1** and
**top-5** accuracy per system and per language, plus a not-groundable
**abstention rate**, in the standard benchmark report schema.

Run the suite locally and print the per-system table:

```
openmed gates grounding            # prints the table
openmed gates grounding --strict   # non-zero exit when a floor is breached
```

`GroundingGateConfig` in `openmed/eval/release_gates.py` holds the floors:
English gold is held to `top-1 >= 0.90` and `top-5 >= 0.97` per permissive
system; `zh`/`hi` gold is held to lower, explicitly **provisional** floors
(`top-1 >= 0.80`, `top-5 >= 0.90`) while multilingual grounding coverage
matures. `evaluate_grounding_accuracy_gate` emits one `GateCheck` per system,
and `build_grounding_gate_report` wraps them in a signed `GateReport`.

!!! warning "Fail closed"
    The release-gates workflow runs `python -m openmed.eval.release_gates
    --grounding` on every tag build, uploads the signed gate report artifact,
    and fails the job when any per-system floor is breached — a silently wrong
    crosswalk or ranking regression cannot ship.

```python
from openmed.eval.grounding_accuracy import evaluate_grounding_accuracy
from openmed.eval.release_gates import evaluate_grounding_accuracy_gate

report = evaluate_grounding_accuracy()
rxnorm_en = report.system("rxnorm").language("en")
print(rxnorm_en.top1_accuracy, rxnorm_en.top5_accuracy)
checks = evaluate_grounding_accuracy_gate(report)
```
