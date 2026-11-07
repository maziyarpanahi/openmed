# Escalation circuit breakers

`EscalationCircuitBreaker` counts failed attempts within one run. It stops at
the first per-class or total threshold and produces a metadata-only reviewer
handoff. It does not execute retries, authorize a new run, or make a clinical
decision. The caller must check `halted` before scheduling another action and
stop the run when `record_failure()` returns a handoff.

## Failure limits

The closed classes are `uncertainty`, `permission_denial`, and `tool_failure`.
The default limits are three uncertainty failures, two permission denials,
three tool failures, or six failures across all classes. A failure counts for
both its class and the total. The attempt that reaches a limit trips the
circuit. Counts do not reset within a run, and a tripped circuit cannot record
more attempts. `CircuitBreakerPolicy` accepts positive integer limits up to 32;
no history can exceed 32 digests.

```python
from datetime import datetime, timezone

from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.agent.runtime.circuit_breaker import (
    EscalationCircuitBreaker,
    FailureClass,
)

breaker = EscalationCircuitBreaker(
    RunId.generate(),
    WorkflowId("workflow:org.example/review@1.0.0"),
)

# Supply a SHA-256 digest of non-sensitive, canonical step metadata. Never
# hash raw clinical text, identifiers, credentials, prompts, or tool arguments.
step_digest = "a" * 64
handoff = breaker.record_failure(
    FailureClass.UNCERTAINTY,
    step_digest,
    now=datetime.now(timezone.utc).replace(microsecond=0),
)
if handoff is not None:
    stop_scheduling_this_run()
    send_to_local_reviewer(handoff.to_dict())
```

The clock is supplied by the caller so identical inputs produce identical
results without network access. The handoff wraps a standard
`ReviewerHandoffPacket` with an opaque run ID, canonical workflow ID, stable
reason code, requested decision, and a 30-minute expiry. It also carries
ordered SHA-256 step digests and closed recovery actions. The trip class maps
to `low_confidence` and `review_evidence`, `human_gate` and
`decide_next_step`, or `safety_review` and `assess_safety`, respectively.
Recovery choices ask a human to review evidence, recheck permissions, review
tool failure, or abort the run; they grant no approval or retry authority.
The reviewer packet contains no raw step content or evidence references.

Only canonical lowercase SHA-256 digests are accepted. Callers must calculate
them from non-sensitive step metadata outside this module; a digest of a
low-entropy patient value is not safe merely because it is hashed. Do not put
clinical text, patient identifiers, arguments, paths, URLs, credentials, or
prompts in the handoff or in the input used to derive a digest. Exceptions
contain fixed codes and field names without rejected values. The module does
not log, send telemetry, or make network calls.

The action-phase dependency [#2998](https://github.com/maziyarpanahi/openmed/issues/2998)
defines lifecycle transitions separately. Integrators should use that contract
to transition a halted run to review or abort once it lands; this circuit
breaker does not invent phase transitions.

Run the focused offline test suite with:

```text
.venv/bin/python -m pytest tests/unit/agent/runtime/test_circuit_breaker.py -q
```
