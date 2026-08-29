# Agent Timing Metadata

Agent run timing metadata is caller-supplied and monotonic. The timing records
accept integer nanosecond boundaries, compute exact integer durations, and never
read a wall clock.

Use `RunTiming` for the overall run and `ActionTiming` for individual steps:

```python
from openmed.agent.timing import ActionTiming, AgentRunTiming, RunTiming

timing = AgentRunTiming(
    run=RunTiming(start_ns=0, end_ns=250, correlation_id="run-01"),
    actions=(
        ActionTiming(action_id="retrieve", start_ns=10, end_ns=80),
        ActionTiming(action_id="redact", start_ns=80, end_ns=120),
    ),
    allow_action_overlaps=False,
)

payload = timing.to_dict()
```

The serialized payload contains only relative nanosecond boundaries, exact
durations, action identifiers, parent action identifiers when present, and
optional opaque correlation identifiers. It does not include wall-clock
timestamps, event payloads, local paths, or PHI.

Validation fails closed for negative, reversed, boolean, non-integer, and
over-maximum boundaries. When `allow_action_overlaps=False`, adjacent sequential
actions are allowed but overlapping action intervals are rejected. Nested actions
must reference an existing parent action and stay inside the parent interval.
Error messages name only fields, not submitted values.
