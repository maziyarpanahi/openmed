# Last-green multilingual regression gate

`cross_lingual_regression_gate` compares every expected language/label baseline
with a proposed checkpoint using aggregate paired outcomes from the same pinned
cohort. Supply baseline/candidate checkpoint digests and an explicit expected
cell-to-cohort map. Missing cells, unexpected cells, mismatched cohorts and small
samples fail qualification rather than disappearing into aggregate F1.

Each `PairedTransferCell` carries success totals and baseline-only/candidate-only
success counts. The constructor checks their joint-count consistency. Statistical
classification uses an exact two-sided paired discordance binomial test with
Bonferroni correction. Exact summation is bounded to 4,096 discordant pairs;
larger unbalanced cases report an explicit computation-budget limitation and
remain statistically inconclusive. Balanced pairs have exact p-value 1 without
summation. Critical decline policy still applies. Positive, negative and
inconclusive cells stay separate.

An independently baseline-policy-declared critical cell whose point recall decline exceeds the pinned
tolerance fails release independently of statistical significance. Target gains
cannot offset it. This conservative policy does not call an inconclusive decline
statistically established. `require_cross_lingual_release` enforces the result by
raising a controlled error when qualification fails. Call it from a checkpoint
promotion boundary; existing release jobs are not silently rewired by this module.

Reports contain controlled codes, counts, deltas, digests and statistical values,
never source records or identifiers. Labels and critical designations must come
from an approved local policy, and baseline/cohort provenance must be verified
outside this API. Digests do not prove that a baseline is last-green. The paired
sampling assumptions and multiplicity family must match the actual evaluation.
No model execution, clinical certification or publication occurs.

The required baseline map binds every `(language, label)` to `(cohort_digest, critical)` from trusted last-green policy. Candidate critical flags must match; a downgraded flag fails qualification. The baseline map is bounded to 1–10,000 cells before iteration.
