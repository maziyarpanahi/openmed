# Agent action execution phases

`openmed.agent.action_phases` defines a closed, metadata-only lifecycle for
one action. It validates one edge at a time without changing state or running
the action. Outcomes describe what happened; phases describe where an action
is in its lifecycle. Existing correlation and timing records remain separate.

## Transition contract

`ACTION_PHASE_TRANSITIONS` is a read-only mapping from each `ActionPhase` to
its allowed targets. Each target's boolean says whether external review is
required. Both levels are read-only; omitted edges always fail.

| Current phase | Ordinary targets (`False` in the table) | Reviewed resume target (`True` in the table) |
| --- | --- | --- |
| `queued` | `preflight`, `aborted` | None |
| `preflight` | `ready`, `waiting-review`, `aborted` | None |
| `ready` | `running`, `aborted` | None |
| `running` | `waiting-review`, `completed`, `aborted` | None |
| `waiting-review` | `aborted` | `preflight` |
| `completed` | None | None |
| `aborted` | None | None |

The ordinary completion path is
`queued -> preflight -> ready -> running -> completed`. Preflight may require
review before readiness, and running may pause for review. Every nonterminal
phase permits aborting without review. All self-transitions, skipped steps,
and backward edges fail except the explicit reviewed return to preflight.
`completed` and `aborted` are terminal, including transitions to themselves.

## Reviewed resume

`waiting-review` is resumable, but its only resume edge is to `preflight`,
and `validate_action_transition` requires the exact boolean `reviewed=True`
for that edge. Omitting it, passing `False`, or passing truthy substitutes such
as `1` or `"true"` fails. Even with `reviewed=True`, transitions directly to
`ready`, `running`, or `completed` fail. Review does not waive preflight.

The calling adapter supplies `reviewed=True` only after its external review
process has permitted resumption. This is an explicit declaration at the
lifecycle boundary, not proof of authorization: the validator does not verify
review evidence, decide approval, or issue tokens. It also does not infer
approval from a `WorkflowOutcome` or a span-level review feedback record.
The caller owns authoritative state and must validate every edge before
updating it. A pure pairwise validator cannot verify a caller's state history
or prevent a caller from falsely declaring review complete.

Returning to preflight allows current constraints to be checked again before
the action becomes ready. A running action held for review follows this same
path; it cannot directly complete after review. How work continues after
preflight is outside this contract, and no retry or replay is scheduled here.

```python
from openmed.agent import (
    ActionPhase,
    is_resumable_phase,
    is_terminal_phase,
    validate_action_transition,
)

validate_action_transition(ActionPhase.QUEUED, ActionPhase.PREFLIGHT)
validate_action_transition("preflight", "waiting-review")
assert is_resumable_phase("waiting-review")

# Only after external review has permitted resumption:
validate_action_transition("waiting-review", "preflight", reviewed=True)
validate_action_transition("preflight", "ready")
validate_action_transition("ready", "running")
validate_action_transition("running", "completed")
assert is_terminal_phase("completed")
```

`is_resumable_phase` identifies a phase with a reviewed resume edge, not every
nonterminal phase. Only `waiting-review` is resumable in this sense.

## Deterministic validation

The validator returns `None` on success and raises `ActionPhaseError`
otherwise. APIs accept `ActionPhase` members or exact canonical built-in
strings. They reject unknown strings, case/whitespace variants, other types,
and string subclasses without coercion. Use these APIs for untrusted values;
Python's direct enum constructor has its own ordinary `ValueError` behavior.

| Error code | Field | Meaning |
| --- | --- | --- |
| `unknown_phase` | `current`, `target`, or `phase` | Unknown phase or unsupported type |
| `invalid_reviewed` | `reviewed` | Review declaration is not an exact boolean |
| `invalid_transition` | None | Edge is absent, including any post-terminal edge |
| `review_required` | `reviewed` | Reviewed resume requested without completed review |

Validation checks current phase, target phase, review argument type, edge
membership, and then the edge's review requirement, in that order. Setting
`reviewed=True` has no effect on ordinary edges or forbidden edges. Error
messages and attributes contain only fixed codes and field names, never
submitted payloads; enum parsing errors are not retained in exception chains.

No action payload, clock, random source, callback, network, file I/O, model,
tool execution, approval decision, token issuance, deadline orchestration,
or retry scheduling is involved in validation.
