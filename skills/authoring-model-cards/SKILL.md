---
name: authoring-model-cards
description: "Generate a model card for an OpenMed clinical NER or de-identification model documenting intended use, quantitative metrics, subgroup performance, limitations, and a medical-device disclaimer for clinical AI governance. Use when the user wants to write or update a model card, a README model section, or governance documentation, or to turn OpenMed eval outputs (release gate report, fairness_report, error_report) into the card's metrics and limitations sections. Trigger on \"model card\", \"intended use\", \"model documentation\", \"governance\", \"limitations section\", \"datasheet\", or \"FDA/ONC transparency\" for an OpenMed model."
license: Apache-2.0
metadata:
  project: OpenMed
  category: evaluation-quality
  pairs: after
  version: "1.0"
---

# Authoring Model Cards

A model card is the honest spec sheet for a model: what it's for, how well it
works, where it breaks, and who it might fail. For clinical models this is
governance-critical — an undocumented de-id model is one nobody can sign off on.
This skill fills a model card directly from OpenMed eval outputs so the numbers
are reproducible, not aspirational.

## When to use this skill

- You're publishing or updating an OpenMed model and need its card.
- You have eval artifacts (`GateReport`, `fairness_report`, `error_report`) and
  need to turn them into intended-use, metrics, and limitations sections.
- A clinical AI governance / model-risk review needs a transparency document.

Run the evals **first** (see `evaluating-with-leakage-gates`,
`benchmarking-clinical-ner`, `auditing-subgroup-fairness`); this skill documents
their results — it does not generate the numbers.

## Card sections (Mitchell et al., + clinical extensions)

See `references/model-card-sections.md` for the full section-to-source map. The
load-bearing sections for an OpenMed model:

- **Model details** — repo id, family, tier, format, params, milestone, license
  (Apache-2.0). Pull from the `GateReport` identity fields.
- **Intended use** — the clinical task and the deployment envelope.
- **Out-of-scope / misuse** — explicitly: not a medical device; not for autonomous
  clinical decisions; de-id is verified, not assumed.
- **Metrics** — entity-level P/R/F1 and, for de-id, residual leakage + per-label
  recall floors and the gate decision.
- **Quantitative analysis (subgroups)** — per-group leakage/recall from
  `fairness_report`, including which groups lack data.
- **Limitations** — error patterns from `error_report`; calibration assumptions.
- **Caveats & disclaimer** — the medical-device disclaimer.

## Quick start — fill the card from eval outputs

```python
from openmed.eval import (
    run_suite, ReleaseGate, fairness_report, error_report,
)

report = run_suite("eval/gold/test.json", suite="golden",
                   model_name="OpenMed/Privacy-PII-Detection", device="cpu",
                   metadata={"family": "PII", "tier": "base",
                             "policy": "hipaa_safe_harbor"})

gate = ReleaseGate(milestone="v1.6", policy="hipaa_safe_harbor").evaluate(report)
fair = fairness_report("OpenMed/Privacy-PII-Detection", "golden")
errs = error_report("OpenMed/Privacy-PII-Detection", "eval/gold/test.json")

card = {
    "model_details": {
        "repo_id": gate.repo_id, "family": gate.family, "tier": gate.tier,
        "format": gate.format, "license": "Apache-2.0",
    },
    "metrics": {
        "exact_span_f1": report.metrics["exact_span_f1"]["f1"],
        "residual_leakage_rate": gate.residual_leakage_rate,
        "critical_leakage_count": gate.critical_leakage_count,
        "per_label_recall": dict(gate.per_label_recall),
        "release_decision": gate.decision,            # RELEASABLE / QUARANTINED
    },
    "subgroup_analysis": fair.to_dict(),              # per-group leakage/recall
    "limitations": errs.to_dict()["confusion_matrix"],
}
# Render `card` into Markdown front matter + body (or the HF card template).
```

`error_report` and `fairness_report` carry no plaintext PHI (offsets + hashes),
so their output is safe to paste into a public card.

## Workflow

1. **Gather artifacts.** Gate report, fairness report, error report — all from a
   pinned model + synthetic eval set.
2. **Fill model details** from the `GateReport` identity fields so the card,
   `models.jsonl`, and the README cannot drift (the gate's `manifest_coherence`
   and `model_card` checks enforce this).
3. **Write intended use narrowly.** Name the clinical task, language(s), and the
   deployment envelope. Over-broad intended-use is the most common card failure.
4. **State out-of-scope and the disclaimer** plainly (see template below).
5. **Report metrics with their floors.** For de-id, lead with leakage and the
   gate decision, not F1.
6. **Report subgroups honestly**, including the documentation gap: if race/
   ethnicity isn't available, say so rather than implying parity.
7. **List limitations from real errors**, not boilerplate — cite the confusion
   matrix's worst cells.

### Disclaimer block (paste & adapt)

> This model assists clinical text processing and is **not a medical device**.
> It does not make autonomous clinical decisions. De-identification output must be
> independently verified before any data is shared; residual PHI risk is never
> zero. Validate on your own population before deployment.

## Hand-off to / from OpenMed

- **From** `evaluating-with-leakage-gates` (`GateReport`),
  `benchmarking-clinical-ner` (`error_report`), and `auditing-subgroup-fairness`
  (`fairness_report`): these are the card's evidence.
- **To** `building-with-openmed` / `models.jsonl`: keep card front matter
  (license, task, languages) coherent with the manifest — the gate checks it.
- **Pairs with** `gating-deid-leakage`: cite the green gate as the card's
  release evidence.

## Edge cases & gotchas

- **Don't claim numbers you can't reproduce.** Every metric in the card should
  trace to an eval artifact and a pinned eval-set hash.
- **Intended use ≠ capability.** Document the supported envelope; mark everything
  else out-of-scope.
- **Subgroup silence is a finding.** Omitting race because it wasn't collected is
  itself a limitation to state — don't let absence read as equity.
- **Card/manifest drift fails the gate.** License/task/language mismatches between
  the card and `models.jsonl` trip `manifest_coherence`.
- **No raw PHI examples.** Use the offset/hash examples from `error_report`; never
  paste real patient strings as "qualitative examples".
- **Quantized variants need their own line.** Report INT8/INT4 recall deltas
  (G4) per format; don't reuse the fp32 numbers.

## Standards & references

- Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019):
  https://arxiv.org/abs/1810.03993
- Hugging Face model card spec & template:
  https://huggingface.co/docs/hub/model-cards
- Sendak et al., *Presenting machine learning model information to clinical end
  users* (clinical "model facts" label): https://doi.org/10.1038/s41746-020-0253-3
- FDA, *Clinical Decision Support Software* guidance:
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- Section-to-source map: `references/model-card-sections.md`.
