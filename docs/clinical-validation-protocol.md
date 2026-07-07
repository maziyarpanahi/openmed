# Clinical Validation Protocol

!!! warning "Not a medical device"
    This validation harness measures de-identification quality against
    **user-supplied labeled data**. It supports internal clinical-validation
    studies but **does not certify** a model for clinical use and is **not a
    medical device**. Results do not replace institutional review, regulatory
    clearance, or clinical judgment.

This document defines the OpenMed clinical-validation protocol
(`openmed-deid-clinical-validation-v1`) and how to run a reproducible study with
the scaffold in `openmed.clinical.validation`. The runner reuses the existing
`openmed.eval` scoring stack; it does not re-implement scoring.

## Scope and principles

- **Local-first.** The study runner performs no mandatory network calls and runs
  fully offline.
- **User-supplied validation data.** The dataset path is supplied through the
  study configuration. The repository ships **only a fully synthetic sample**
  (`tests/fixtures/clinical/validation_sample.jsonl`) for tests and
  documentation. No DUA-gated corpora (for example MIMIC, i2b2, n2c2) are
  bundled; such datasets are eval-only and must never be committed.
- **Leakage-first.** PHI leakage is a gating primary metric. A study that meets
  a detection-quality target but exceeds the leakage bound does **not** pass.
- **No raw PHI in reports.** Validation reports record offsets, hashes, counts,
  and rates only. The dataset is fingerprinted through a content-addressed
  manifest that never persists raw text.
- **Signed and reproducible.** Re-running on identical inputs yields identical
  provenance and repro hashes. Every report is HMAC-signed.

## Study design

A validation study evaluates one model against one labeled dataset.

1. **Inputs.** A `StudyConfig` names the user-supplied `dataset_path`, the
   `model_name` (or a custom runner), a stable `dataset_id`, and a
   `data_revision` (for example a git SHA or dataset version tag).
2. **Labeled data.** Each dataset record is a benchmark fixture with document
   `text`, `gold_spans` (character offsets and labels), a `language`, and
   optional `metadata` (used for subgroup axes, e.g. `group`). An optional
   leading `{"kind": "meta", ...}` row is skipped.
3. **Prediction.** The model runner produces predicted spans per document. The
   default runner uses the shared PII runtime; a custom deterministic runner can
   be supplied for offline or CI use.
4. **Scoring.** Predicted and gold spans are scored with `openmed.eval`:
   detection quality via `compute_exact_span_f1` and leakage via
   `compute_leakage_rate`; subgroup fairness via `fairness_report`.
5. **Acceptance.** Observed metrics are compared against documented thresholds
   deterministically, producing a per-metric pass/fail and an overall
   `accepted` flag (true only when every **primary** criterion passes).
6. **Report.** A signed, reproducible report is emitted as JSON and Markdown
   with provenance hashes.

## Metrics

### Primary metrics (gating)

| Metric | Source | Meaning |
|---|---|---|
| Leakage rate | `compute_leakage_rate` | Character-weighted fraction of gold PHI characters not covered by a prediction. Lower is better. |
| Recall | `compute_exact_span_f1` | Exact-span, label-aware detection recall over gold PHI spans. |

### Secondary metrics

| Metric | Source | Meaning |
|---|---|---|
| Precision | `compute_exact_span_f1` | Exact-span, label-aware precision. |
| F1 | `compute_exact_span_f1` | Harmonic mean of precision and recall. |
| Subgroup leakage disparity | `fairness_report` | Maximum minus minimum per-group leakage rate across any subgroup axis. |

Aggregate F1 alone is never sufficient: leakage and subgroup disparity are
reported and gated independently so a high-F1 model that leaks specific
identifier classes or underperforms on a subgroup is caught.

## Acceptance thresholds

Defaults are conservative scaffold values and may be overridden per study
through `threshold_overrides`. Direction indicates whether the observed value
must be at least (`lower_bound`) or at most (`upper_bound`) the threshold.

| Metric | Direction | Default | Primary |
|---|---|---:|---|
| `leakage_rate` | upper bound | 0.01 | yes |
| `recall` | lower bound | 0.95 | yes |
| `precision` | lower bound | 0.90 | no |
| `f1` | lower bound | 0.92 | no |
| `subgroup_leakage_disparity` | upper bound | 0.01 | no |

A study is `accepted` only when every **primary** criterion passes. Secondary
criteria are reported for review but do not by themselves block acceptance;
studies may promote them to primary through overrides.

## Subgroup breakdowns

The report breaks results down along the configured subgroup axes (default:
`group` and `language`). For each axis it reports per-group leakage rate,
recall, and gold span count, plus the axis-level leakage disparity and the
worst-performing group. Subgroup fairness reuses
`openmed.eval.fairness.fairness_report`, so subgroup scoring is identical to the
rest of the eval stack.

The synthetic sample exercises three demographic-surrogate groups (`adult`,
`pediatric`, `geriatric`) across two languages (`en`, `es`). Real studies
should choose subgroup axes that reflect the populations and identifier
distributions they must validate against.

## Provenance and reproducibility

Every report records:

- `openmed_version` — the package version used for the run.
- `dataset_id` and `data_revision` — caller-supplied identifiers for the
  labeled data.
- `dataset_manifest_hash` — a content-addressed hash of the labeled dataset
  produced by `build_training_data_manifest`. It fingerprints the documents and
  gold spans **without persisting raw text**.
- `eval_code_hash` — a hash of the eval harness and metric code, so a change in
  scoring logic changes the provenance.
- `acceptance_thresholds` — the exact thresholds the study was judged against.
- `repro_hash` — a `sha256` over the full, deterministic, PHI-free report
  payload.
- `signature` — an HMAC-SHA256 signature over the payload and repro hash.

Re-running the same model/runner over the same dataset with the same
configuration produces byte-identical JSON, an identical `repro_hash`, and an
identical signature. `ValidationReport.verify(key)` re-checks both the repro
hash and the signature and rejects any tampering.

## Running a study

```python
from openmed.clinical.validation import StudyConfig, run_validation_study

config = StudyConfig(
    dataset_path="/path/to/your/labeled_data.jsonl",  # user-supplied; never bundled
    model_name="OpenMed/your-deid-model",
    dataset_id="institution-deid-validation",
    data_revision="git:abc1234",
)

report = run_validation_study(config)
report.write_json("validation-report.json")
report.write_markdown("validation-report.md")

assert report.verify("your-signing-key")  # or the local default key
print("accepted:", report.accepted)
```

For offline or CI runs, pass a deterministic `runner` with the harness
`ModelRunner` signature `(fixture, model_name, device) -> iterable of spans`.

### Report shape

The JSON report contains: `study_id`, `model_name`, `device`, `protocol_id`,
`schema_version`, `fixture_count`, `metrics.overall`
(`recall`/`precision`/`f1`/`leakage_rate` and supporting counts), `subgroups`
(per-axis `per_group` breakdowns with `leakage_disparity` and `worst_group`),
`acceptance` (per-metric outcomes), `accepted`, `provenance`, `disclaimer`,
`repro_hash`, and `signature`. The Markdown report renders the same content as a
single reviewable page and always carries the medical-device disclaimer.

## Data handling requirements

- Keep labeled validation data outside the repository; reference it only by
  path in your study configuration.
- Do not commit any DUA-gated, license-gated, or otherwise restricted corpus.
  Only fully synthetic samples may be committed.
- Reports are PHI-free by construction; still store and share them under the
  same controls as any de-identification audit artifact.
