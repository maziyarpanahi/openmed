---
name: benchmarking-clinical-ner
description: "Score an OpenMed clinical or biomedical NER model against a user-supplied gold corpus with entity-level precision, recall, and F1, then break errors down per label. Use when the user wants a seqeval-style scorecard, strict vs partial (relaxed) span matching, a per-label confusion matrix, false-negative / false-positive examples, or to debug why a model misses entities. Trigger on \"evaluate NER\", \"entity-level F1\", \"seqeval\", \"precision recall F1\", \"confusion matrix\", \"error analysis\", \"strict vs partial match\", or \"score against gold\" in an OpenMed context. The gold corpus is user-supplied; OpenMed bundles no i2b2/n2c2/MIMIC data."
license: Apache-2.0
metadata:
  project: OpenMed
  category: evaluation-quality
  pairs: adjacent
  version: "1.0"
---

# Benchmarking Clinical NER

This skill produces an honest entity-level scorecard for an OpenMed NER model:
precision / recall / F1 plus a per-label error breakdown. It scores **spans**,
not tokens, because clinical entities are multi-token ("type 2 diabetes
mellitus") and token-level accuracy hides boundary errors. Reported numbers are
**entity-level** in the seqeval tradition (CoNLL-2000 / SemEval-2013 families).

## When to use this skill

- You have a gold-annotated clinical corpus and an OpenMed NER model to score.
- You want strict (exact-boundary) and partial (relaxed-overlap) span F1.
- You need per-label numbers, not one aggregate — DRUG recall ≠ DISEASE recall.
- You need to *explain* the errors: what was missed, what was spurious, what was
  mislabeled.

For PHI de-id specifically, gate on leakage with `evaluating-with-leakage-gates`
instead of (or in addition to) F1.

## Match modes

| Mode | Counts a hit when… | Use for |
| --- | --- | --- |
| **Strict / exact** | predicted span boundaries **and** label match gold exactly | release scoring, boundary-sensitive tasks |
| **Partial / relaxed** | predicted span **overlaps** gold with the right label | recall-oriented triage, tokenizer-mismatch tolerance |

OpenMed exposes both: `compute_exact_span_f1` (strict) and
`compute_relaxed_span_f1` (partial), with the full bundle in
`compute_metrics_bundle`.

## Quick start

Run a model over a user-supplied gold fixtures file and print a scorecard:

```python
from openmed.eval import run_suite, error_report

# Fixtures: JSON list of {"id", "text", "gold_spans": [{start, end, label}, ...]}
report = run_suite(
    "eval/gold/clinical_ner.json",        # YOUR gold corpus, not bundled
    suite="golden",
    model_name="OpenMed/Disease-Detection",
    device="cpu",
)

m = report.metrics
print("exact F1 :", m["exact_span_f1"]["f1"])      # strict
print("relaxed F1:", m["relaxed_span_f1"]["f1"])    # partial
print("recall by label:", m["recall_slices"]["by_label"])

# Per-label confusion matrix + capped, no-PHI error examples.
errors = error_report(
    "OpenMed/Disease-Detection",
    "eval/gold/clinical_ner.json",
    suite_name="clinical_ner",
    example_cap=5,
)
print(errors.to_markdown())                 # confusion matrix + FN/FP tables
errors.write_json("eval/out/error_analysis.json")
```

Need just the metrics on spans you already have? Call the metric functions
directly:

```python
from openmed.eval import compute_exact_span_f1, compute_relaxed_span_f1

strict = compute_exact_span_f1(gold_spans, predicted_spans)
partial = compute_relaxed_span_f1(gold_spans, predicted_spans)
```

## Workflow

1. **Align the corpus to OpenMed fixtures.** Convert CoNLL/BIO or BRAT
   standoff into the fixture shape: `text` + `gold_spans` of
   `{start, end, label}` character offsets. (CoNLL → offsets; BRAT `.ann` is
   already character offsets.)
2. **Normalize labels** to OpenMed's canonical set so DRUG/MEDICATION variants
   don't count as label confusion. Mislabeled-but-overlapping spans show up in
   the confusion matrix, not as misses.
3. **Run** `run_suite` / `run_benchmark` to get a `BenchmarkReport`.
4. **Read both F1s.** A large strict↓ / relaxed↑ gap means boundary errors, not
   detection failures — often tokenizer or whitespace issues.
5. **Run `error_report`** for the per-label confusion matrix and capped
   examples. `MISSED` = false negatives (recall problem); `SPURIOUS` = false
   positives (precision problem); off-diagonal = label confusion.
6. **Triage per label.** Fix the worst-recall label first; in clinical NER a few
   labels usually dominate the error budget.

## Hand-off to / from OpenMed

- **From** `extracting-clinical-entities` (`openmed.analyze_text`): the model and
  predictions you score here come from the NER pipeline.
- **To** `evaluating-with-leakage-gates`: for de-id models, F1 is necessary but
  not sufficient — pass the same fixtures through the release gates.
- **To** `authoring-model-cards`: drop `error_report` confusion matrices and
  per-label F1 straight into the model card's quantitative-analysis section.
- **Pairs with** `building-gold-corpus` (supplies the fixtures) and
  `auditing-subgroup-fairness` (slices the same run by demographic group).

## Edge cases & gotchas

- **Token F1 lies; report span F1.** Always use the span metrics
  (`compute_exact_span_f1` / `compute_relaxed_span_f1`), not token accuracy.
- **Overlapping/nested gold spans** need a documented matching rule. OpenMed's
  matcher picks the best single overlapping prediction per gold span; nested
  schemes (e.g. DISEASE inside ANATOMY) should be flattened or scored per layer.
- **Class imbalance hides failures.** A macro view per label surfaces a rare-but-
  critical entity (e.g. ALLERGY) that micro-F1 buries.
- **Error examples are no-PHI by design.** `ErrorSpanExample` stores offsets,
  context windows, and `sha256:` text hashes — never plaintext. Keep it that way.
- **Gold quality caps your ceiling.** If inter-annotator agreement is low, a
  "low-F1" model may be right and the gold wrong. Spot-check disagreements before
  blaming the model.
- **No restricted corpora in the repo.** i2b2/n2c2/MIMIC are DUA-gated: load them
  from the user's licensed copy at eval time; never commit them.

## Standards & references

- seqeval (entity-level sequence-labeling metrics):
  https://github.com/chakki-works/seqeval
- SemEval-2013 Task 9.1 strict/partial/exact/type evaluation scheme:
  https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
- CoNLL-2003 NER shared task (entity-level F1 convention):
  https://aclanthology.org/W03-0419/
- BRAT standoff annotation format:
  https://brat.nlplab.org/standoff.html
- OpenMed eval source of truth: `openmed/eval/metrics.py`,
  `openmed/eval/error_analysis.py`, `openmed/eval/harness.py`.
