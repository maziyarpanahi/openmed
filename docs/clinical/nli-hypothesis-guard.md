# Clinical NLI hypothesis-complexity guard

`guard_hypothesis()` applies a deterministic pre-inference contract to a
clinical NLI hypothesis. A bounded atomic hypothesis is returned with status
`ready`. An over-complex hypothesis is withheld and returned as
`segmentation_required` with value-free reason codes.

```python
from openmed.clinical.nli_hypothesis_guard import guard_hypothesis

result = guard_hypothesis("The finding is absent.")
if result.inference_allowed:
    hypothesis = result.hypothesis
else:
    reasons = result.reasons
```

The default contract allows at most 240 characters, two clauses, and one
assertion. Each limit is inclusive and configurable per call.

## Deterministic measurement

- Character count is the Python string length.
- Clause count includes non-empty sentence or semicolon segments, subordinate
  links such as `because` and `whereas`, and coordinated predicates recognized
  by a fixed expression.
- Assertion count includes non-empty sentence or semicolon segments plus
  coordinated predicates with an explicit subject or a fixed clinical-style
  predicate cue. A shared predicate over a list, such as “shows fever and
  cough,” remains one assertion.
- Decimal points between digits are not sentence boundaries.

The rules intentionally avoid a model, tokenizer download, or network call.
They are a conservative input contract, not a linguistic parser. Callers may
set tighter limits but should segment rejected text into independently cited
claims rather than raising limits to pass ambiguous input.

## Privacy boundary

For `segmentation_required`, the result discards the hypothesis and contains
only counts and reason codes. For a ready result, hypothesis text is excluded
from `repr` and `to_dict()`. Validation errors are fixed categories and never
echo submitted text or values. Do not log the ready result's `hypothesis`
property.

This guard does not perform NLI, guarantee that a claim is atomic, or make a
clinical decision.
