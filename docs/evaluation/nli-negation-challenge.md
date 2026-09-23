# Synthetic clinical NLI negation challenge

`openmed.eval.nli_negation_challenge` provides a small, deterministic challenge
for a clinical natural-language-inference (NLI) predictor. It targets a failure
mode that aggregate NLI accuracy can hide: predicting `entailment` for a
negated or otherwise non-entailing hypothesis.

The challenge is backend-neutral. It does not load a model, download weights,
contact a service, or emit telemetry. Supply predictions from an already
installed local model, or pass labels computed by a local evaluation job.

## Covered patterns

The built-in corpus contains synthetic, explicitly PHI-free pairs for four
patterns:

| Pattern | What it checks |
|---|---|
| `simple` | A direct negation and its affirmative/negative hypotheses. |
| `nested` | A construction such as “no indication that ... not present”, which does not entail that the finding is present. |
| `double` | A double negative such as “not absent”, including both entailment and contradiction targets. |
| `section_scoped` | Negation in a named clinical section without leaking that assertion into another section. |

The cases are returned by `default_nli_negation_cases()`. Their premise and
hypothesis strings are synthetic challenge inputs. They are not extracted from
clinical records or restricted datasets.

## Local evaluation

The predictor contract accepts either `(premise, hypothesis)` or one
`NliNegationCase` and returns an NLI label. Precomputed labels may be supplied
as an iterable in case order or as a mapping keyed by the case IDs. The IDs are
join keys only and are omitted from reports.

```python
from openmed.eval.nli_negation_challenge import (
    assert_false_entailment_gate,
    default_nli_negation_cases,
    run_nli_negation_challenge,
)

cases = default_nli_negation_cases()

# Replace this mapping with labels returned by an already-installed local NLI
# model. This line is only a deterministic smoke example.
predictions = {case.case_id: case.gold_label for case in cases}
report = run_nli_negation_challenge(
    predictions=predictions,
    max_false_entailment_rate=0.0,
)
assert_false_entailment_gate(report)
```

`None` or an explicit `abstention` is counted as incorrect for aggregate
accuracy, but it is not counted as a false entailment. Unknown labels and
ambiguous fixture rows fail closed with value-free errors.

## Separate safety signal

The report keeps the two signals separate:

- `aggregate_accuracy` is correct labels divided by all challenge cases.
  Abstentions remain in the denominator.
- `false_entailment_rate` is predictions of `entailment` divided by the
  contradiction/neutral target cases.
- `report.gate.false_entailment_gate_passed` evaluates the dedicated safety
  ceiling. The default ceiling is zero.
- `minimum_aggregate_accuracy` can be supplied when an accuracy floor is also
  required. Omitting it measures accuracy without making it a gate.

Use `assert_false_entailment_gate(report)` when the safety decision must remain
independent of aggregate accuracy. Use `assert_nli_negation_gate(report)` when
both configured checks must pass. A model can therefore have acceptable
aggregate accuracy while failing the false-entailment gate; that is an intended
blocking outcome.

## Privacy and review boundary

JSON and Markdown reports contain counts, controlled pattern names, gate values,
and SHA-256 content digests only. They do not contain premises, hypotheses, case
IDs, predictor output, or predictor exception details. Keep any custom cases
explicitly synthetic and PHI-free, and treat the source strings as local input
to the predictor.

This challenge is evaluation evidence, not a compliance certification or an
autonomous clinical decision guarantee. A qualified human must review model
selection, thresholds, false-entailment failures, and the intended clinical use
before relying on any result.
