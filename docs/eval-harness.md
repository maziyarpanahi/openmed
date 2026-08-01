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

## Clinical PHI flagship certification

`build_clinical_privacy_release` certifies the named
`OpenMed/OpenMed-ClinicalPrivacy-tier0` checkpoint from four inputs: its
checkpoint manifest row, verified mode-C training provenance, a held-out
clinical-PHI `BenchmarkReport`, and the manifest-linked SHIELD comparison
report. It reuses the release harness's v1.6 G1a and G2 floors, requires
explicit category coverage plus a reported zero critical-leakage count and
zero residual leakage for G3, then signs a report containing exactly those
three gate verdicts.

The signed checks bind hashes and stable references for the checkpoint,
clinical-PHI dataset manifest, held-out report, and SHIELD report. SHIELD stays
comparison-only evidence and cannot be promoted to the high-recall gate. A
quarantined candidate yields a signed failure report but no model card or model
manifest entry.

```python
from openmed.eval.clinical_privacy_release import (
    build_clinical_privacy_release,
)

release = build_clinical_privacy_release(
    held_out_report,
    shield_report,
    checkpoint_manifest=checkpoint_row,
    training_provenance=training_provenance,
    checkpoint_manifest_ref="models.jsonl#clinical-privacy-tier0",
    held_out_report_ref="release-evidence/held-out-report.json",
    shield_report_ref="release-evidence/shield-report.json",
    release_date="2026-08-01",
    signing_key=release_signing_key,
)
paths = release.write("release-evidence/clinical-privacy-tier0")
```

The output directory contains `gate-report.json`, `README.md`,
`model-datasheet.json`, `model-manifest-entry.json`, and
`release-manifest.json`. These artifacts contain aggregate metrics, hashes,
and provenance references only—never source notes, fixture identifiers, or raw
PHI. The generated card marks the checkpoint as assistive de-identification,
not diagnosis, treatment recommendation, or an automatic clinical-decision
trigger.

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

## Cross-lingual grounding capstone gate

The standalone `cross_lingual_grounding` suite verifies the complete lexical
grounding path for English, Spanish, French, German, and Chinese mentions. It
uses the bundled synthetic/permissive alias fixture for ICD-10-CM, RxNorm, and
HPO, requires top-1 accuracy of at least `0.80` in every non-English language,
and requires the unchanged English baseline to remain at `1.00`.

`assert_cross_lingual_grounding_gate` evaluates those accuracy requirements
together with synthetic provenance and the restricted-corpus marker scan. Its
report contains aggregate counts and scores, not raw mentions; failure
diagnostics likewise contain only language codes, scores, and known
restricted-marker names. Language-aware linkers accept the source-language code
used by the suite and stamp it on every grounded candidate for downstream audit.

```python
from openmed.eval.suites import assert_cross_lingual_grounding_gate

report = assert_cross_lingual_grounding_gate()
print(report.per_language_top1)
```

The gate is deterministic and offline. The default grounding path stays
lexical and does not import an embedding dependency. Applications may opt in to
a separately provisioned local multilingual embedding backend for zero-alias
fallback; remote model identifiers are rejected rather than downloaded.

!!! warning "Synthetic/permissive data only"
    The committed fixture must remain synthetic or permissively licensed.
    UMLS, SNOMED CT, MIMIC, i2b2, n2c2, CPT, and other restricted or DUA-bound
    assets must stay user supplied and out of process; do not copy their text
    into alias tables, fixtures, logs, or gate diagnostics.
