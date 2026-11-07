# Privacy release-gate aggregation

`openmed.compliance.privacy_gate` combines local risk, policy, evidence, and
dependency gate results into one deterministic release record. It is a
technical release aid, not a compliance certification or clinical decision
guarantee.

## Create a decision

Gate producers should provide a stable identifier, one explicit state, and an
aggregate count. A producer can also provide a SHA-256 fingerprint over its
private finding set:

```python
from openmed.compliance import PrivacyGateResult, aggregate_privacy_gates

record = aggregate_privacy_gates(
    [
        PrivacyGateResult("risk", "blocking", finding_count=1),
        PrivacyGateResult("policy", "warning", finding_count=2),
        PrivacyGateResult("dependency", "waived", waiver_code="reviewed"),
    ]
)
```

The states are deliberately explicit:

| Gate state | Meaning |
|---|---|
| `blocking` | A finding prevents release. |
| `warning` | Release needs review but is not blocked by this gate. |
| `waived` | The gate is acknowledged as waived and remains visible in the record. |

Aggregate precedence is `blocked` when any gate is blocking, `warning` when
there are warnings but no blockers, and `released` when only waived gates
remain. Waived counts are never silently discarded.

## Safe record format

`record.to_dict()` and `record.to_json()` contain the decision, counts for all
three states, sorted gate summaries, per-gate fingerprints, and an aggregate
record fingerprint. They do not contain findings, free-text reasons, source
values, identifiers, or arbitrary metadata. `from_findings` can be used at a
local boundary when a producer has a sized collection; only its length is
retained.

The implementation uses the local Python runtime and stable hashing only. It
performs no network calls and has no telemetry or remote service dependency.
