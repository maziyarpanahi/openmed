# Medication-status contradiction precheck

`medication_status_contradiction_precheck()` compares explicit medication
status and event-time evidence before a clinical NLI model runs. It is a
deterministic local consistency check, not a medication recommendation engine.

```python
from openmed.clinical.nli_medication_status import (
    medication_status_contradiction_precheck,
)

result = medication_status_contradiction_precheck(
    {
        "medication_key": "synthetic-medication",
        "status": "continued",
        "event_time": "2026-01-02",
    },
    {
        "medication_key": "synthetic-medication",
        "status": "stopped",
        "event_time": "2026-01-02",
    },
)

assert result.contradiction
assert result.inference_allowed is False
```

## Status and time contract

The controlled statuses are `started`, `continued`, `held`, `changed`,
`stopped`, `historical`, and `uncertain`. Common explicit aliases are
normalized. Missing status becomes `uncertain`; it is never defaulted to an
active or recommended state.

Absolute event times use the existing medication reconciliation timestamp
contract. Relative expressions must first be normalized by a timeline layer
with an explicit anchor.

The result status is one of:

- `compatible`: controlled statuses agree without a time conflict.
- `contradiction`: explicit active and inactive states conflict at the same
  normalized event time.
- `review_required`: a state is uncertain or changed, a status transition has
  different times, or the times needed to order a disagreement are missing.
- `not_applicable`: both records identify different medications.

Ordered transitions are escalated even when they appear plausible. The
precheck does not infer which regimen is current, preferred, or recommended.

## Privacy and scope

Reports contain controlled status labels, a coarse time relation, source
offsets, and domain-separated SHA-256 fingerprints. They do not contain raw
medication identities, raw event times, source text, or arbitrary status
strings. Errors are value-free. The implementation performs no network call,
uses no wall clock, and loads no terminology data.

The result is an assistive consistency signal for human review. It is not a
prescription, medication order, treatment recommendation, or autonomous
clinical decision.
