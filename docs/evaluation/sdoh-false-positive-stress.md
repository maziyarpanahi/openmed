# SDOH false-positive stress gate

For recall, status and trigger-offset scoring, see the
[SDOH extraction benchmark](sdoh-extraction.md).

`openmed.eval.sdoh_false_positive_stress` measures patient-level SDOH false
positives on repository-authored synthetic hard negatives. The matrix covers
five Social History categories and seven patterns per category: screening,
education, unanswered boilerplate, third-party language, out-of-section text,
negation, and historical mentions. It never loads a restricted corpus or calls
a remote model.

The default predictor applies OpenMed's local section scope and experiencer
filter. Non-assertive screening, education, and unanswered template clauses
are excluded before a finding enters the patient-level view. Third-party
findings remain available to the existing experiencer review layer but are
not counted as patient findings. Negated and historical use remains
noncurrent. Double-negated unemployment stays `unknown` for review rather
than becoming an automatic positive finding.

```python
from openmed.eval.sdoh_false_positive_stress import (
    assert_sdoh_stress_gate,
    run_sdoh_false_positive_stress,
)

report = run_sdoh_false_positive_stress()
assert_sdoh_stress_gate(report)
print(report.to_dict())
```

Every category has its own false-positive rate and ceiling. The default
ceiling is zero for each category; a caller may supply an explicit per-category
ceiling for evaluation. The count of attempted automated eligibility actions
must always remain zero and cannot be relaxed. The report contains only
controlled category names, counts, rates, gate results, and the action count.
It contains no source text, finding values, identifiers, or spans.

This synthetic gate covers the included patterns; it is not a clinical
decision guarantee or an estimate on a real population. Permissioned corpora
must remain eval-only, user supplied, and outside the repository. Keep review
and eligibility decisions with qualified humans.
