# Chart-abstraction field scoring

`openmed.eval.suites.chart_abstraction` compares private abstraction outputs
with gold fields using the existing Python abstraction-evidence contract. It
performs no file, model, database or network access. This Python evaluation
slice does not change the Swift runtime or add sealed orchestration or
clinician adjudication.

```python
from openmed.eval.suites.chart_abstraction import (
    ChartAbstractionPrediction,
    run_chart_abstraction_benchmark,
    synthetic_chart_abstraction_gold,
)

gold = synthetic_chart_abstraction_gold()
# Explicitly synthetic perfect-value baseline, without source evidence.
outputs = [
    ChartAbstractionPrediction(row.case_id, row.field_id, row.value)
    for row in gold
]
report = run_chart_abstraction_benchmark(gold, outputs)
assert report.overall.exact_agreement.success_count == 8
assert report.overall.evidence_support.success_count == 0
```

The bundled gold set has twelve invented fields across three cases and four
types: text, categorical, number and boolean. Eight fields are answerable;
four have no gold answer. These fixtures establish scorer behavior only and
provide no clinical, model-performance or release evidence.

## Joining and normalization

Gold and output rows join on private `case_id` and developer-authored
`field_id`. Only field IDs are reported; never use patient values as schema
identifiers. Duplicate gold/output keys, unknown output keys, inconsistent
types for the same field ID, malformed scalar values and nonfinite numbers
are rejected with controlled error codes that do not echo input.

`None` means an unanswerable gold field or an explicit output abstention.
An absent output is missing, including on an unanswerable field; it never
earns correct-abstention credit. Empty strings are answers.

Exact agreement requires equal scalar types and values. Normalized text and
categorical agreement use Unicode NFKC, case folding and whitespace collapse.
Numbers use finite decimal equality, including decimal strings, without
guessing units or tolerances. Booleans require actual booleans; `1` is not
`True`. No clinical synonyms, terminology assets or learned normalizers are
included. A valid scalar of the wrong field type counts as disagreement.

## Counts, denominators and intervals

Reports contain overall, per-field and per-type slices. Types are closed
labels. Each slice includes field, answerable, unanswerable, missing, answered
and abstained counts, plus these binomial metrics:

| Metric | Success/event count | Denominator |
| --- | --- | --- |
| `exact_agreement` | Exactly correct answers | All answerable gold fields |
| `normalized_agreement` | Correct after normalization | All answerable gold fields |
| `abstention_correctness` | Answering an answerable field or explicitly abstaining on an unanswerable one | All gold fields |
| `answerable_abstention` | Abstaining despite an available gold answer | Answerable gold fields |
| `unanswerable_answer` | Answering despite no gold answer | Unanswerable gold fields |
| `evidence_support` | Answer with a matching, fact-bound clinical-source chain | Submitted non-abstained answers |

The two error metrics remain separate. Missing outputs reduce agreement and
abstention correctness but do not inflate either explicit-abstention error
count. Wrong answers can have a correct abstention decision. Correct answers
can lack source support. Compute rates as `success_count / total_count` when
the denominator is positive.

Every metric supplies a deterministic two-sided 95% Wilson interval; zero
trials produce `interval: null`, not evidence of a zero error rate. Intervals
are descriptive field-level binomial intervals, not patient-cluster-adjusted
estimates or release thresholds. Canonical JSON is invariant to input ordering.
Reports include no clinical values, case IDs, source payloads, source offsets,
digests, paths, credentials or per-case decisions.

## Evidence trust boundary

An answer is supported only if its `evidence_chain` has the same field ID,
contains a clinical-record source span, and its `normalized_fact_digest`
matches the producer-declared digest on the submitted prediction. Missing,
empty, generated-only, different-field and different-fact chains count as
unsupported, even for correct answers. Mixed clinical/generated evidence can
count as supported. Abstentions create no evidence-support trials.

Use the evidence producer's existing normalized-fact encoding. The scorer
does not define an alternate fact format or rehash scalar values: binding
the submitted value to that producer digest is the trusted local adapter's
responsibility. Source presence measures mechanical provenance coverage; it
does not verify the source's clinical meaning or the digest-to-value assertion.
Pending/rejected reviewer states do not change this metric. Approval remains
the separate abstraction finalization gate, and support is not adjudication.

Keep values, case mappings and evidence chains inside the protected local
evaluation boundary. Input dataclass representations omit values, case IDs
and chains; only the aggregate report is suitable for value-free diagnostics.
No autonomous clinical action, release approval or cloud fallback is added.
