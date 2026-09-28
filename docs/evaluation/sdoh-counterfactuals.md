# Synthetic SDOH counterfactual checks

`openmed.eval.sdoh_counterfactuals` checks whether changing non-causal age
and pronoun context changes SDOH labels or confidence. It uses the local,
deterministic social-history generator and keeps each pair's determinant
evidence identical. The rule-based section detector scopes extraction to the
Social History section.

```python
from openmed.eval.sdoh_counterfactuals import (
    generate_sdoh_counterfactual_pairs,
    require_sdoh_counterfactual_invariance,
)

pairs = generate_sdoh_counterfactual_pairs(30, seed=17)
report = require_sdoh_counterfactual_invariance(pairs)
print(report.to_dict())
```

The report contains pair counts, a pair-level invariance rate, and category
counts for label or confidence mismatches. It contains no source text,
finding values, offsets, or identifiers. Unknown extractor categories are
reported as `other`. An extractor error is replaced with a fixed message so
an exception cannot print text from a synthetic or user-supplied input.

The generator uses only repository-authored synthetic examples. Real SHAC data
remains DUA-gated, user supplied, and evaluation-only; it is never bundled,
loaded by this check, or used for training. The result is a regression gate,
not a fairness certification or an autonomous clinical decision.
