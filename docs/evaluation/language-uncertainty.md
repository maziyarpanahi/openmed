# Evidence for small language slices

`language_uncertainty_report` accepts registered-language and controlled-label
slices with local aggregate successes/trials for each declared stratum. It reports
Bonferroni-adjusted Wilson intervals weighted by observed stratum proportions,
not a pooled estimate that hides a tiny critical subgroup. Empty strata remain
empty and force `insufficient_evidence`.

The minimum-detectable-effect summary uses a normal approximation for two equal,
independent Bernoulli comparison arms, with conservative variance 0.25 per arm.
Pin confidence, power, desired effect and minimum stratum size. Every stratum must
meet both sample and effect-size conditions before `ranking_allowed` becomes
true. Small perfect samples retain uncertainty and never imply language support.
MDE is a planning approximation, not a retrospective claim of achieved power.

Confidence bounds are descriptive; sampling independence and representative
strata must be justified outside this helper. Observed weights are not population
weights. No model runs or network calls occur. Outputs contain only controlled
language/label/stratum codes, aggregate counts and statistical summaries; codes
must come from an approved vocabulary, not source identifiers. The helper cannot
certify privacy, clinical performance or a model release.
