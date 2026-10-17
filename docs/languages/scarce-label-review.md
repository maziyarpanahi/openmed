# Budgeted native label review

`openmed.training.languages.scarce_label_queue.select_native_review` ranks local
metadata candidates by caller-declared label scarcity, then model disagreement.
Equally scored candidates first prefer error signatures with fewer selected
candidates, then diversity clusters with fewer selected candidates, then
SHA-256 digest order. Selection is deterministic across input permutations.

Supply `ReviewCandidate` records with `candidate_digest`, language and label
codes, finite `[0, 1]` scarcity and disagreement, a `diversity_digest`, and a separate `error_signature_digest`. Signature digests
represent controlled local error categories; they are not source snippets.
References must be SHA-256 digests; source text and institutional identifiers
are not accepted. Hashing is not an anonymization guarantee: callers must use
appropriate local reference handling and must not hash patient identifiers for
export. Keep candidate-to-source resolution local.

Pass independent `language_budgets`, `label_budgets`, and `total_budget`.
All are non-negative integer caps. Missing categories have zero budget. A row
must satisfy every cap; no constraint is relaxed to fill the total budget.
Language/region aliases are normalized with the existing clinical language
registry, so `HI`, `hi-IN`, and `hi_IN` share the `hi` budget. Duplicate normalized
language budget keys and automatic language hints are rejected. Label codes
must come from a caller-controlled vocabulary; syntax checks alone do not make
arbitrary labels PHI-safe.

Duplicate candidate references, malformed categories, and invalid scores fail
before selection. `ReviewSelection.to_dict()` contains only digests, category
counts, and total counts.

The queue uses a lazy heap and accepts at most 10,000 candidates, 256 budget
categories per mapping, and 100,000 stale-rank updates. Pathological tie groups
can still require quadratic work; exceeding the work cap raises an error instead
of returning a partially selected report. Reduce the batch or refine the ranking
evidence before retrying. Count mappings are immutable snapshots.

This is a greedy review queue, not an optimal assignment solver. Scarcity is
supplied by the caller rather than inferred clinical importance. Diversity breaks
ties and never outranks scarcity or disagreement. Selection does not accept
pseudo-labels, change models, send data to reviewers, or certify a language. Native
review and adjudication remain separate. The existing active-learning queue and
its persistence contract are unchanged.

Run the offline acceptance tests with:

```sh
.venv/bin/python -m pytest tests/unit/training/languages/test_scarce_label_queue.py -q
```
