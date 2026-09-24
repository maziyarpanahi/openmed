# Deterministic clinical NLI cascade

`openmed.clinical.nli_cascade` is the execution boundary between deterministic
structured checks and a caller-supplied local NLI model. It evaluates stages in
this fixed order:

1. assertion;
2. temporality;
3. experiencer;
4. numeric evidence;
5. local model inference.

Each structured component supplies a normalized `NliRuleStatus`. The
`from_precheck()` adapter recognizes OpenMed's controlled `compatible`,
`contradiction`/`incompatible`, `review_required`, and `not_applicable`
statuses. This keeps assertion, temporal, experiencer, and numeric parsing in
their owning modules instead of duplicating those APIs in the cascade.

```python
from openmed.clinical.nli_cascade import (
    NliCascadePair,
    NliRuleStatus,
    evaluate_nli_pair,
)

pair = NliCascadePair(
    pair_id="pair-001",
    premise="synthetic de-identified premise",
    hypothesis="synthetic de-identified hypothesis",
    assertion=NliRuleStatus.COMPATIBLE,
    temporality=NliRuleStatus.UNRESOLVED,
    experiencer=NliRuleStatus.COMPATIBLE,
    numeric=NliRuleStatus.CONFLICT,
)

result = evaluate_nli_pair(pair, local_model)
assert result.model_invoked is False
assert result.deciding_stage.value == "numeric"
```

The first exact conflict decides `contradiction`; the model is not invoked.
Compatible, unresolved, and non-applicable structured outcomes leave the
semantic relation unresolved and allow the local model to decide. Model output
must use the canonical four-state vocabulary: `entailment`, `contradiction`,
`neutral`, or `abstention`.

The cascade performs no model discovery and no network request. The supplied
callable must already be a resolved local backend. Source text is exposed only
through the explicit model-call boundary and is excluded from results,
representations, and audit reports. Pair identifiers must be opaque and
non-PHI; audit reports replace them with domain-separated fingerprints. Model
exceptions are converted to a value-free error.

The cascade is an assistive verification layer. Its output requires qualified
human review and must not autonomously drive diagnosis or treatment.
