---
name: auditing-subgroup-fairness
description: "Audit an OpenMed NER or de-identification model for performance disparities across demographic subgroups (sex, age band, race/ethnicity when available) using openmed.eval.fairness_report. Use when the user wants per-subgroup recall and leakage, wants to check whether de-identification under-protects a group, wants to surface a documentation gap where subgroup data is missing, or needs equalized-odds-style disparity numbers for a clinical model. Trigger on \"fairness\", \"subgroup\", \"bias audit\", \"disparity\", \"equalized odds\", \"under-protected group\", \"per-group recall\", or \"STANDING Together\" for an OpenMed model."
license: Apache-2.0
metadata:
  project: OpenMed
  category: evaluation-quality
  pairs: adjacent
  version: "1.0"
---

# Auditing Subgroup Fairness

An aggregate pass can hide a group the model fails. For de-identification that
failure has a name: **under-protection** — PHI that leaks more often for one
demographic group than another. `openmed.eval.fairness_report` slices leakage and
recall by gold-span group so disparities surface before deployment, not after a
breach.

## When to use this skill

- You want per-subgroup recall and leakage for a de-id or NER model.
- You suspect (or must rule out) that one group is under-protected.
- You need disparity numbers for a clinical AI governance review.
- You need to document *which* subgroups you couldn't evaluate (the data gap).

## What it measures

For each surrogate group `fairness_report` returns:

- **leakage_rate** — fraction of that group's gold PHI characters left exposed
  (the de-id harm metric).
- **recall** — fraction of that group's gold spans detected.
- **leakage_disparity** — `max - min` leakage across groups (the gap to close).
- **worst_group** / **worst_group_leakage** — the most-failed group.

Group membership comes from a `group` tag in each gold span's `metadata` (keys
`group`, `demographic_group`, or `surrogate_group`); ungrouped spans fall into
`unspecified`.

## Quick start

```python
from openmed.eval import fairness_report

# Gold fixtures must tag spans with a surrogate group, e.g.
#   {"start": 4, "end": 12, "label": "PERSON", "metadata": {"group": "female"}}
fair = fairness_report(
    "OpenMed/Privacy-PII-Detection",
    "golden",                 # named suite, or pass a list of fixtures
    device="cpu",
)

print("leakage disparity:", fair.leakage_disparity)
print("worst group      :", fair.worst_group, fair.worst_group_leakage)
for group, m in sorted(fair.per_group.items()):
    print(f"  {group:14s} recall={m.recall:.3f}  leakage={m.leakage_rate:.4f}")

# Under-protection alarm: any group leaking more than the rest.
LEAKAGE_GAP_LIMIT = 0.0       # leakage-first: ideally zero leakage everywhere
assert fair.leakage_disparity <= LEAKAGE_GAP_LIMIT or fair.worst_group_leakage == 0
```

`FairnessReport.to_dict()` is JSON-ready and PHI-free — drop it straight into a
model card.

## Workflow

1. **Tag the gold corpus by group.** Add a synthetic `group` to each PHI span's
   `metadata` (sex, age band, race/ethnicity surrogate). Use synthetic surrogates,
   not real protected attributes (see `building-gold-corpus`).
2. **Run `fairness_report`** on the model + suite.
3. **Read leakage first, recall second.** For de-id, a group with higher leakage
   is under-protected — that is the headline finding.
4. **Compute the disparity** (`leakage_disparity`) and locate `worst_group`.
   Equalized-odds framing: equal true-positive (recall) *and* equal leakage
   across groups.
5. **Document the gap.** If race/ethnicity surrogates are absent, report that the
   audit could not cover them — most clinical NLP studies omit race entirely, so
   silence is the default failure mode, not equity.
6. **Feed it forward.** Put per-group numbers and the gap into the model card and
   the governance review.

## Hand-off to / from OpenMed

- **From** `building-gold-corpus`: supplies group-tagged synthetic fixtures.
- **From** `evaluating-with-leakage-gates`: an aggregate `RELEASABLE` decision
  should be paired with this audit — overall pass, subgroup fail is exactly the
  trap this catches.
- **To** `authoring-model-cards`: `FairnessReport.to_dict()` fills the
  quantitative-analysis / subgroup section.
- **Pairs with** `benchmarking-clinical-ner`: same run, different slice (label vs
  group).

## Edge cases & gotchas

- **Under-protection is the de-id harm; lead with leakage.** A group with equal
  recall but higher leakage is still failed.
- **The race documentation gap is the norm.** Most clinical NLP corpora don't
  record race/ethnicity, so most fairness audits silently can't measure it.
  Report the absence explicitly — don't let missing data read as parity.
- **`unspecified` is not a real group.** A pile of spans in `unspecified` means
  your gold isn't tagged; fix the corpus before trusting the disparity.
- **Small groups give noisy rates.** Report span counts (`span_count`,
  `total_chars`) alongside rates; a 1-of-2 leak isn't a 50% population rate.
- **Synthetic surrogates only.** Never store real protected attributes in eval
  fixtures; use fabricated group labels for slicing.
- **Disparity ≈ 0 with high leakage everywhere is not "fair".** Equal failure is
  still failure — check absolute leakage, not just the gap.

## Standards & references

- STANDING Together — reporting standards for health-dataset diversity &
  documentation: https://www.datadiversity.org/
- Hardt, Price, Srebro, *Equality of Opportunity in Supervised Learning*
  (equalized odds): https://arxiv.org/abs/1610.02413
- Chen et al., *Ethical ML in Health Care* (subgroup performance in clinical NLP):
  https://doi.org/10.1146/annurev-biodatasci-092820-114757
- OpenMed source of truth: `openmed/eval/fairness.py`
  (`fairness_report`, `FairnessReport`, `FairnessGroupMetrics`).
