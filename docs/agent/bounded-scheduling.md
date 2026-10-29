# Bounded Agent Scheduling

`ActionScheduler` schedules independent actions from a validated
[identifier-only action graph](action-graphs.md) in one Python event loop.
It uses an application-owned `GuardedDispatcher` for every invocation. The
scheduler receives no tool arguments, model output, protected inputs, credentials
or paths. It does not grant authority or consume approvals.

## Public contract

```python
from openmed.agent.action_graph import ActionNode
from openmed.agent.correlation import RunId
from openmed.agent.workflows.scheduler import (
    ActionScheduler,
    ScheduledAction,
    SchedulerLimits,
)

# The host supplies guarded_dispatcher and atomic_save_checkpoint.
# Both fetches are classified read-only by trusted application policy.
actions = [
    ScheduledAction(ActionNode("left", "tool:openmed.agent/fetch"), read_only=True),
    ScheduledAction(ActionNode("right", "tool:openmed.agent/fetch"), read_only=True),
    ScheduledAction(
        ActionNode("join", "tool:openmed.agent/review", ("left", "right")),
    ),
]
scheduler = ActionScheduler(
    actions,
    SchedulerLimits(
        max_concurrency=2,
        max_actions=3,
        max_inflight_units=2,
        max_total_units=3,
        max_elapsed_ns=30_000_000_000,
        parallel_reads=True,
    ),
    guarded_dispatcher,
    atomic_save_checkpoint,
)
# In the host's async function:
# result = await scheduler.run(RunId.generate())
```

The injected dispatcher implements
`async dispatch(run_id, node, cancellation) -> DispatchOutcome`. Payloads and
input bindings remain in its protected store. It must verify authority, tool
identity and required approval at the invocation boundary, and check the
cancellation signal immediately before sensitive reads or effects. A valid graph
or `read_only=True` never constitutes approval. The host must classify tools
from trusted contracts rather than accepting a model's assertion of read-only
behavior.

`COMMITTED` means the dispatcher's protected input/output state is committed and
available for descendants; an effect additionally needs authentic sink commit
evidence. `FAILED` stops admission. `REVIEW_REQUIRED`, an exception, a cancelled
invocation or an unexpected return type means uncertain work requiring review.
The scheduler records only these controlled outcomes and never exception text.
The guarded dispatcher must be safe for concurrent reads. This protocol is an
injection boundary, not a replacement for the guarded workflow dispatcher.

## Admission and effect ordering

Ready actions have all dependencies in `COMMITTED` state. Among eligible actions,
the scheduler examines ascending action IDs. Input order and dependency-list
order do not affect the plan digest. Completion timing can affect when new work
becomes eligible; the scheduler collects all already-finished peers before
admitting descendants, so a peer's review verdict blocks further admission.

Parallel reads require `parallel_reads=True`; the default policy serializes all
work. Only explicitly read-only actions can overlap, up to `max_concurrency`.
Effects are the default classification and run exclusively against both reads
and effects. A ready effect forms an admission barrier until existing reads
drain. This conservative policy serializes potential conflicts without guessing
resource-level effect independence.

Each action reserves a positive `resource_units` estimate. The host defines the
unit and supplies a worst-case estimate. In-flight reservations may not exceed
`max_inflight_units`; when a large ready read does not fit alongside existing
reads, a smaller independent read may proceed. An action that cannot fit even
alone stops admission with `budget_exhausted`.

Admissions also charge `max_actions` and `max_total_units`. Cumulative charges
are never refunded, including failed, interrupted or write-ahead admissions.
Exhausting these budgets lets already admitted work finish, then stops additional
admission. A non-negative injected monotonic nanosecond clock checks
`max_elapsed_ns` before admission, after checkpoint saves and while waiting.
Invalid or regressing clocks fail closed. Snapshots store elapsed durations,
never absolute clock readings; restart adds the saved duration to a fresh clock.
Time while the process is stopped is excluded.

These are scheduling reservations, not a general resource meter. Actual memory,
artifact storage and internal tool-call limits remain the guarded runtime's
responsibility. There is no model inference, network call or new dependency.

## Cancellation and checkpoint safety

Pass a `CancellationSignal` to `run` and call `cancel()` on the same event loop.
Cancellation or an elapsed-budget breach stops pending actions. Review and
failures also stop independent pending work, not only descendants. Running tasks
receive the stop signal and an asyncio cancellation request and are drained.
Already committed results remain committed; uncertain interrupted work becomes
`review_required`. Cancellation cannot reverse an effect or guarantee a hard
elapsed-time limit when an executor suppresses cancellation or blocks the loop.

Cancelling the scheduler's asyncio task also drains its child tasks and saves a
stopped checkpoint before propagating `CancelledError`. The application must
allow cleanup to finish; process termination and repeated task cancellation may
interrupt persistence. Recovery then uses the last durable snapshot and treats
in-flight admissions as uncertain.

`save_checkpoint` must synchronously persist each immutable snapshot atomically
and durably in a trusted host store. It runs before dispatch and after completion,
before descendants start. A persistence exception stops admission with
`checkpoint_failed`; the returned snapshot alone is not evidence of durability.
Snapshot JSON contains opaque run/action IDs, states, the plan digest, numeric
utilization and a controlled stop reason. Stores must protect integrity and
freshness: the plan digest is a binding, not a signature or rollback defense.

Restore with `SchedulerCheckpoint.from_dict` and pass `checkpoint=` to `run`.
Restoration verifies run/plan identity, complete action coverage, dependency
commit ordering and cumulative charges. A checkpoint containing in-flight,
failed or review-required work cannot dispatch again automatically. Stopped
checkpoints remain stopped, and completed actions are never repeated. A trusted
partial snapshot containing only committed and pending actions can resume with
the original identities and budgets. Reconcile uncertain effects through
[workflow recovery](workflow-recovery.md) and host review before preparing any
resumable state; this scheduler neither retries nor invents commit evidence.
The host must also prevent concurrent schedulers for the same run in its store.

## Scope and validation

This slice targets the existing Python action-graph and workflow contracts.
OpenMedKit has no corresponding agent graph/guarded-dispatch surface; on-device
Apple Foundation Models behavior is unchanged. The scheduler adds no autonomous
clinical action, distributed orchestration, cloud fallback or publication rights.

Synthetic offline tests inject executors, stores and clocks, hold independent
branches at explicit barriers, verify exclusive effects, exercise negative
controls and restore every saved boundary:

```bash
.venv/bin/python -m pytest tests/unit/agent/workflows/test_scheduler.py tests/integration/agent/test_scheduler_restart.py -q
make format
make lint
make format-check
.venv/bin/python -m pytest tests/ -q
make docs-build
```
