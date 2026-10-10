# Bounded clinical NLI batch planning

`openmed.clinical.nli_batch_plan` builds deterministic local-inference batches
without inspecting premise or hypothesis text. Callers provide an opaque pair
identifier and a token estimate produced by their selected local tokenizer.
The planner performs no model loading, filesystem access, telemetry, or network
request.

```python
from openmed.clinical.nli_batch_plan import (
    NliPairCost,
    NliRuntimeProfile,
    plan_nli_batches,
)

profile = NliRuntimeProfile(
    max_tokens_per_batch=512,
    max_pairs_per_batch=8,
    max_total_tokens=1_024,
)
plan = plan_nli_batches(
    [
        NliPairCost("pair-001", 180),
        NliPairCost("pair-002", 420),
        NliPairCost("pair-003", 700),
    ],
    profile,
)
```

The planner preserves caller order and starts a new batch before either the
per-batch token ceiling or pair ceiling would be exceeded. A pair larger than
the per-batch token ceiling is returned in `deferred_pair_ids`. Optional total
token and pair budgets are hard invocation limits; pairs that do not fit are
also deferred while later smaller pairs remain eligible.

Pair identifiers are returned because the caller must be able to retry or
route deferred work. They must therefore be opaque, non-PHI identifiers. The
ordinary representations and `to_audit_dict()` omit identifiers; audit output
contains only counts, token estimates, and domain-separated SHA-256
fingerprints. Exceptions never echo caller values.

Planning is a resource-control mechanism, not a clinical inference or safety
decision. Deferred pairs require an explicit caller policy such as a smaller
local profile, a later local run, or qualified review.
