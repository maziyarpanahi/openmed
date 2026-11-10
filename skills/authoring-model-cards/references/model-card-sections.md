# Model Card Sections → OpenMed eval source

Maps each model-card section (Mitchell et al. 2019, with clinical extensions) to
the OpenMed eval artifact that populates it. Run the evals first
(`evaluating-with-leakage-gates`, `benchmarking-clinical-ner`,
`auditing-subgroup-fairness`), then fill the card from these objects.

| Card section | What it states | OpenMed source | Field / call |
|---|---|---|---|
| Model details | repo id, family, tier, format, params, license, milestone | `GateReport` | `.repo_id`, `.family`, `.tier`, `.format`, `.param_count` |
| Intended use | clinical task, language(s), deployment envelope | author + `models.jsonl` | manifest row (`task`, `languages`) |
| Out-of-scope use | not a device; no autonomous decisions; verify de-id | author | disclaimer block |
| Factors / subgroups | demographic surrogate groups evaluated | `FairnessReport` | `.per_group`, `.worst_group` |
| Metrics (NER) | entity-level precision / recall / F1 (strict + partial) | `BenchmarkReport.metrics` | `exact_span_f1`, `relaxed_span_f1`, `recall_slices.by_label` |
| Metrics (de-id) | residual leakage, critical leakage, per-label recall, gate decision | `GateReport` | `.residual_leakage_rate`, `.critical_leakage_count`, `.per_label_recall`, `.decision` |
| Quantization | INT8 / INT4 recall delta vs fp parent (G4) | `GateReport` | `.quant_recall_delta` |
| Performance / resources | p50/p95 latency, peak RAM vs tier budget (G5/G6) | `GateReport` | `.p50_ms`, `.p95_ms`, `.ram_mb` |
| Quantitative analysis | per-group leakage & recall, disparity | `FairnessReport` | `.leakage_disparity`, `.per_group[*].recall` |
| Limitations | dominant error patterns, label confusions | `ErrorAnalysisReport` | `.confusion_matrix`, `.false_negatives`, `.false_positives` |
| Caveats & recommendations | calibration assumptions, population shift, medical-device disclaimer | author + `CalibrationReport` | thresholds target leakage / min recall |
| Evidence / reproducibility | signed gate decision, eval-set hash | `GateReport` | `.decision`, `.eval_set_hash`, `.repro_hash`, `.signature` |

## Notes

- All listed artifacts are no-PHI: metrics are aggregates; error examples are
  offsets + `sha256:` hashes. Safe to paste into a public card.
- Keep `license`, `task`, and `languages` in the card front matter identical to
  the `models.jsonl` row — the release gate's `manifest_coherence` and
  `model_card` checks fail the build on drift.
- For de-identification models, lead the metrics section with **leakage and the
  gate decision**, then F1 — F1 is not the safety bar.
