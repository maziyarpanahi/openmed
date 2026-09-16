# Agent outcome reason codes

`openmed.agent.outcomes` gives agent workflows a small, closed vocabulary for
what happened. The payload is metadata only: a class, a reason code, and a
schema version. It does not store prompts, tool arguments, tool outputs,
evidence text, paths, or credentials.

The public API distinguishes:

- `success` with `completed`
- `abstained` with `insufficient_evidence`, `out_of_scope`, or `low_confidence`
- `review_required` with `conflicting_evidence`, `safety_review`, or `human_gate`
- `policy_denied` with `consent_required`, `purpose_mismatch`, or `phi_policy`
- `failed` with `tool_error`, `timeout`, or `invalid_input`

Unknown classes, unknown reason codes, class/reason mismatches, extra fields,
and free-text reasons fail closed. Exception messages name the field or a
stable error code and do not echo the submitted value.

```python
from openmed.agent import WorkflowOutcome

outcome = WorkflowOutcome.from_dict(
    {
        "outcome_class": "abstained",
        "reason_code": "insufficient_evidence",
    }
)

payload = outcome.to_json()
```

`to_dict()` emits fields in a stable order. `to_json()` emits compact JSON
with sorted keys so identical inputs serialize byte-for-byte identically.

This module does not execute tools, persist run logs, or decide clinical
actions.
