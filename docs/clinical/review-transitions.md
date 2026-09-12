# Guarded review-state transitions

`openmed.clinical.review_transitions` provides a deterministic state machine for
human review of an assistive clinical output. It validates workflow metadata; it
does not make a clinical decision, certify an output, or replace qualified
clinical judgment.

## Default state graph

The default policy supports these transitions:

| Current state | Allowed next state(s) |
| --- | --- |
| `queued` | `in_review`, `expired` |
| `in_review` | `approved`, `rejected`, `expired` |
| `approved` | `reopened` |
| `rejected` | `reopened` |
| `expired` | `reopened` |
| `reopened` | `in_review` |

Reopening is intentionally not a shortcut to approval or rejection. A reopened
result must enter review again before reaching a terminal outcome.

## Validating a history

Every transition requires two opaque values:

- `event_id`, normally produced by `make_opaque_event_id`; and
- `provenance_fingerprint`, normally produced by
  `compute_provenance_fingerprint` from safe metadata such as an artifact
  digest, schema version, or policy identifier.

The machine stores only state values, sequence numbers, opaque identifiers,
fingerprints, and optional machine-readable reason codes. Reviewer identities,
case contents, timestamps, and free-text notes are not accepted or copied into
reports.

```python
from openmed.clinical import (
    ReviewState,
    ReviewStateMachine,
    compute_provenance_fingerprint,
    make_opaque_event_id,
)

machine = ReviewStateMachine()
provenance = compute_provenance_fingerprint(
    {"artifact": "synthetic-review", "schema": "v1"}
)

machine.transition(
    ReviewState.IN_REVIEW,
    make_opaque_event_id("synthetic-event-1"),
    provenance,
)
machine.transition(
    ReviewState.APPROVED,
    make_opaque_event_id("synthetic-event-2"),
    provenance,
)

report = machine.report().to_dict()
assert report["current_state"] == "approved"
```

Invalid edges, repeated event identifiers, malformed fingerprints, and missing
provenance raise `ReviewTransitionValidationError`. The exception exposes only
a stable diagnostic code and known state values, so it is safe to include in a
structured local diagnostic.

## Injecting policy rules

Use `ReviewTransitionPolicy` to narrow the graph, require reason codes for
selected states, or inject a predicate over the PHI-free
`ReviewTransitionRequest`:

```python
from openmed.clinical import (
    ReviewPolicyRule,
    ReviewState,
    ReviewStateMachine,
    ReviewTransitionPolicy,
)

policy = ReviewTransitionPolicy(
    policy_id="synthetic-supervised-review",
    required_reason_states=(ReviewState.REOPENED,),
    rules=(
        ReviewPolicyRule(
            "reopen_reason",
            lambda request: request.reason_code == "corrected_artifact",
        ),
    ),
)
machine = ReviewStateMachine(policy=policy)
```

Rules receive no case payload or reviewer identity. Keep rule codes and reason
codes machine-readable and free of sensitive values. All fingerprints and
policy descriptions are stable for the same input, and the module performs no
network I/O.
