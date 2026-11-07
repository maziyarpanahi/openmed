# Per-run agent resource budgets

`RunResourceBudget` enforces declared limits for one local run at safe
checkpoints. An integrator creates a fresh meter at run start and calls
`reserve()` **before** scheduling each step, tool invocation, or artifact
write. A stopped report means the caller must stop scheduling work and retain
its last safe checkpoint for review or recovery. The meter does not execute
tools, write checkpoints, interrupt an in-flight call, or grant a retry.

```python
import time

from openmed.agent.runtime.resource_budget import (
    ResourceBudgetLimits,
    RunResourceBudget,
)

meter = RunResourceBudget(
    ResourceBudgetLimits(
        max_steps=20,
        max_tool_calls=10,
        max_wall_time_ns=30_000_000_000,
        max_memory_estimate_bytes=256 * 1024 * 1024,
        max_artifact_bytes=1024 * 1024,
    ),
    started_ns=time.monotonic_ns(),
)

# At an existing safe checkpoint, before executing the next action:
report = meter.reserve(
    now_ns=time.monotonic_ns(),
    steps=1,
    tool_calls=1,
    memory_estimate_bytes=32 * 1024 * 1024,
)
if report.stopped:
    # Keep the last safe checkpoint and stop this run.
    handle_budget_stop(report.to_dict())
else:
    execute_next_action()
```

All five limits are required. Count model invocations against the tool-call
limit when they are scheduled by the agent. Steps, tool calls, and redacted
artifact bytes accumulate. Memory is the highest estimated peak supplied for
an action. The wall-time limit uses caller-supplied monotonic nanoseconds and
is checked even with zero proposed work. Exact limits are allowed; a proposed
amount above a limit stops the run without charging that proposed work. Checks
are ordered:
wall time, steps, tool calls, memory estimate, then artifact storage. A clock
rollback also stops the run. A stopped meter cannot be reused.

`ResourceBudgetReport.to_dict()` and `to_json()` emit only numeric limits and
utilization, a stopped flag, and a closed reason code. Neither accepts a
prompt, clinical payload, patient identifier, tool arguments, artifact path,
or free-text checkpoint name. Exceptions contain fixed codes and field names
without rejected values. The module performs no network call, logging, or
telemetry. Count **only redacted** artifact bytes, and reserve the maximum
expected size before writing. Integrators must supply conservative memory and
storage estimates and recheck after work with a zero-work call for elapsed
time; this cooperative meter cannot enforce use inside an unbounded tool.

The separate action-phase issue
[#2998](https://github.com/maziyarpanahi/openmed/issues/2998) defines lifecycle
transitions. Until that contract lands, a budget stop is a scheduling gate and
content-free report; it does not invent a phase transition. Existing durable
workflow recovery may retain the last safe checkpoint, subject to its own
rules for external effects.

Run the focused offline tests:

```text
.venv/bin/python -m pytest tests/unit/agent/runtime/test_resource_budget.py -q
```
