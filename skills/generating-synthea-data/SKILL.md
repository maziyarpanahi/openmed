---
name: generating-synthea-data
description: "Generates synthetic but realistic patient records (FHIR R4 bundles, C-CDA documents, CSV) with MITRE Synthea for development, CI fixtures, demos, and leakage-gate test sets — zero real PHI. Use when you need safe, shareable test data for an OpenMed pipeline, reproducible fixtures for tests, or a held-out set for de-identification leakage gates, instead of touching real clinical data. Synthea output feeds the FHIR/C-CDA ingestion skills and openmed.eval. Trigger keywords: Synthea, synthetic data, fake patients, test fixtures, demo data, FHIR bundle generator, synthetic EHR, no PHI."
license: Apache-2.0
metadata:
  project: OpenMed
  category: data-ingestion
  pairs: adjacent
  version: "1.0"
---

# Generating Synthetic Patient Data with Synthea

You cannot develop, test, or demo a clinical NLP pipeline on real PHI without a
mountain of governance — and you shouldn't have to. **Synthea** (MITRE's
Synthetic Patient Population Simulator) generates statistically realistic,
fully synthetic patients: complete longitudinal records as FHIR R4 bundles,
C-CDA documents, and flat CSV, with **zero real-PHI risk**. Use it for OpenMed
dev fixtures, CI, demos, and — importantly — as **held-out test sets for
de-identification leakage gates**, where you need known-synthetic "PHI" to
measure recall.

## When to use

- Building or demoing an OpenMed ingestion pipeline (FHIR, C-CDA) and need
  shareable input that is safe to commit and pass around.
- Creating deterministic CI fixtures so tests don't depend on protected data.
- Producing a leakage-gate test corpus: synthetic notes with *known* fake
  identifiers, so you can score whether `openmed.deidentify` removed them all.
- Teaching/onboarding without a data-use agreement.

## Quick start

Synthea is a Java tool. Generate a small population in multiple formats:

```bash
# Requires Java 11+. Clone and build once.
git clone https://github.com/synthetichealth/synthea && cd synthea
./gradlew build -x test

# Generate 50 patients in Massachusetts as FHIR R4 + C-CDA + CSV.
./run_synthea -p 50 Massachusetts \
  --exporter.fhir.export=true \
  --exporter.ccda.export=true \
  --exporter.csv.export=true \
  --exporter.baseDirectory=./output

# Reproducible runs: fix the seed so fixtures are stable across CI.
./run_synthea -s 12345 -p 20 --exporter.baseDirectory=./fixtures
```

Output lands under `output/fhir/`, `output/ccda/`, and `output/csv/`. Feed the
FHIR bundles to `parsing-... ` skills, or hand narrative straight to OpenMed:

```python
import json, openmed

bundle = json.load(open("output/fhir/Patient_xyz.json"))
for entry in bundle.get("entry", []):
    res = entry.get("resource", {})
    div = (res.get("text") or {}).get("div", "")     # narrative XHTML
    if div.strip():
        deid = openmed.deidentify(div, method="replace", policy="hipaa_safe_harbor")
        result = openmed.analyze_text(deid.text, output_format="dict")
```

Synthea data is synthetic, so de-identifying it is *exercising the pipeline*,
not a privacy requirement — which is exactly what makes it a great test bed.

## Workflow

1. **Choose scale & geography.** `-p N` sets population; the state/location
   argument shapes demographics and addresses. Start small (10–50) for fixtures.
2. **Pick formats.** Enable FHIR (`exporter.fhir.export`), C-CDA
   (`exporter.ccda.export`), and/or CSV per your ingestion path. FHIR R4 is the
   default and pairs with `fetching-fhir-resources`; C-CDA pairs with
   `parsing-ccda-documents`.
3. **Pin a seed** (`-s`) for reproducible fixtures so test assertions are stable.
4. **Select modules** (optional). Synthea ships disease modules
   (`-m "diabetes*"` to filter); choose modules matching the entities your
   OpenMed pipeline targets.
5. **Use as a leakage-gate corpus.** Synthea emits known fake names, MRNs,
   addresses, and dates — inject/collect these as ground-truth PHI spans and
   score `openmed.deidentify` recall with `openmed.eval`
   (`evaluating-with-leakage-gates`). Because the "PHI" is synthetic and known,
   you can measure misses without exposing anyone.
6. **Commit fixtures** under your test tree (e.g. `tests/fixtures/synthea/`) —
   it is safe to version-control synthetic output.

## Hand-off to / from OpenMed

- **To OpenMed (as input):** Synthea FHIR/C-CDA narrative → `openmed.deidentify`
  → `openmed.analyze_text`, via the `fetching-fhir-resources` and
  `parsing-ccda-documents` skills.
- **To OpenMed eval:** use Synthea's known synthetic identifiers as ground truth
  for `openmed.eval` de-identification leakage gates — the daily-release thesis
  gates on leakage, not F1 alone, and synthetic data lets you build that test
  set without governance overhead.
- **Adjacent, not in-pipeline:** Synthea is a *source* of safe data; it does not
  call OpenMed and OpenMed does not call it. Keep it in dev/CI, never as a
  production data source.

## Edge cases & gotchas

- **Synthetic ≠ statistically perfect.** Synthea reproduces realistic disease
  progression and demographics but is not a substitute for real-world
  distribution validation; never report clinical model accuracy *only* on
  synthetic data.
- **Narrative is templated.** FHIR `text.div` narrative is generated from
  templates, so it is more regular than dictated notes. For NER robustness,
  supplement with varied real (de-identified) text where governance allows.
- **Determinism needs the seed.** Without `-s`, every run differs — CI fixtures
  will churn. Always pin the seed for committed fixtures.
- **Version drift.** Synthea modules and FHIR profile output change across
  releases; pin the Synthea version (git tag) alongside your fixtures.
- **Large populations are heavy.** `-p 100000` produces gigabytes; size to need.
- **Licensing.** Synthea and its generated output are permissively licensed
  (Apache-2.0), so output is safe to redistribute — unlike MIMIC/i2b2/n2c2,
  which require data-use agreements and must stay user-supplied.

## Standards & references

- Synthea project & docs: https://github.com/synthetichealth/synthea
- Synthea wiki (running, modules, exporters):
  https://github.com/synthetichealth/synthea/wiki
- Walonoski J, et al. "Synthea: An approach, method, and software mechanism for
  generating synthetic patients..." JAMIA 2018:
  https://doi.org/10.1093/jamia/ocx079
- FHIR R4: https://hl7.org/fhir/R4/
- C-CDA R2.1: https://www.hl7.org/ccdasearch/
