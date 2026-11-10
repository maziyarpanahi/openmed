# Agent Event Sequences

`validate_event_sequence` checks that an append-only run log can be replayed.
It is an explicit helper, not an event store, a repair tool, or a consensus
protocol across writers.

```python
from openmed.agent.event_sequence import EventReference, validate_event_sequence

references = [
    EventReference(run_id="run_0123456789abcdef", event_id="ev-000000", sequence_number=0),
    EventReference(run_id="run_0123456789abcdef", event_id="ev-000001", sequence_number=1),
]
report = validate_event_sequence(
    "run_0123456789abcdef", references, terminal_sequence_number=1
)
print(report.is_valid, report.reason_codes)
```

## What is validated

References are consumed in the order supplied, which is the order a reader
would replay them in. Each reference carries a bounded opaque `run_id`, a
bounded opaque `event_id`, and a non-negative integer `sequence_number`.
Identifiers match `^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$`, so the
canonical `RunId`/`ActionId` strings from
[Agent Event Correlation](event-correlation.md) are accepted unchanged.

Structural problems fail closed with `EventSequenceError`: non-integer or
boolean sequence numbers, negative or out-of-range numbers, malformed
identifiers, non-reference items, a non-iterable container, a terminal
sequence below the expected start, and sequences longer than
`MAX_EVENT_SEQUENCE_LENGTH`. Booleans are rejected because `type(value) is
int` is checked rather than `isinstance`.

Sequence problems are reported as findings so one pass shows every defect:

| Reason code | Meaning |
| --- | --- |
| `empty_sequence` | No references were supplied and `allow_empty` is false. |
| `start_sequence_mismatch` | The lowest sequence number is not `expected_start`. |
| `sequence_gap` | A number is missing; the finding names the first missing number of that gap. |
| `duplicate_sequence_number` | A number repeats; each repeat after the first is reported. |
| `duplicate_event_id` | An opaque event identifier repeats. |
| `out_of_order` | A reference does not increase on the previous one. |
| `cross_run_reference` | A reference names a different run. |
| `post_terminal_event` | A reference follows the declared terminal sequence number. |
| `terminal_event_missing` | The declared terminal sequence number never appears. |
| `findings_truncated` | More than `MAX_SEQUENCE_FINDINGS` findings were produced. |

A declared terminal that is far beyond the last event is reported once as
`terminal_event_missing` rather than as a run of gap findings.

## Determinism and privacy

Findings are ordered by sequence number, then reason code, then opaque event
identifier, so the same input always produces the same report and the same
`to_json()` bytes. `to_dict()` preserves declared field order and `to_json()`
sorts keys for byte-identical payloads.

Findings, reports and exceptions contain sequence numbers, opaque identifiers,
counts and stable codes only. Event payloads, prompts, tool arguments,
outputs, and clinical text are never accepted by this module, so they cannot
appear in output. `EventSequenceError` exposes a stable `.code` and an optional
`.field_name`; the rejected value is never echoed.

Repairing gaps, persisting events, and deciding event semantics are out of
scope. A valid sequence is an integrity statement about ordering, not a
clinical or security approval.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/agent/test_event_sequence.py -q
```

Fixtures are synthetic. Tests cover empty-when-allowed, single-event and
multi-event runs, every reason code, finding order, serialization stability,
and boolean/negative/oversized rejection.
