# Guarded workflow dispatch

`openmed.agent.workflows.dispatch.GuardedDispatchAdapter` is a callable step
adapter for the existing `openmed.mcp.workflow.WorkflowRunner`. It connects the
signed grant, purpose ticket, minimum-data planner, single-use approval, action
outcome and recovery contracts on the actual executor boundary. It introduces no
second workflow engine, network transport, ledger implementation or UI.

This slice scopes execution to the existing Python dispatcher. OpenMedKit does
not gain a dispatcher or a cloud fallback. The existing on-device review and
receipt surfaces remain separate; this adapter grants no autonomous clinical
action or release authority.

## Host configuration and authority

The trusted host constructs a `DispatchBinding` for one opaque run/action and
one pinned registered `ToolSpec`. It supplies exact capability scope, purpose,
record-selector digests, projected data classes and reviewer role. These values
must come from reviewed host policy and the host's selected record source, not
from an agent's workflow declaration. The host must ensure that source records
match the configured selectors; selector syntax alone cannot prove provenance.

A separate `DispatchAuthority` carries the presented grant, ticket, requests,
projection and approval. It stays outside the pipeline's `inputs`. Each request
must equal the trusted binding. The adapter verifies the signed grant and ticket
for the active run, derives the projection from the pinned schema, and compares
it with the presented projection. The projected classes must exactly match the
binding. Arguments must satisfy the registered schema, and every actual nested
field must appear in the planner's field paths. Extra fields are rejected rather
than silently stripped. The existing executor resolves handle bindings first;
only its resulting process-local JSON argument snapshot is approved and invoked.

`adapter.action_digest(arguments)` is the approval-preview binding. The digest
covers the exact JSON arguments, registered contract/version, tool identity,
workflow/run/action identifiers, resource/action/policy scope, purpose,
projection, record selectors, approval requirement and reviewer role. A signed
approval for different arguments fails even when all other authority is valid.
The standard `ApprovalTokenVerifier` consumes the nonce once; a mismatched signed
presentation requires fresh human approval. Approval is required by default and
cannot be disabled for a state-changing `ToolSpec`.

## Default-off effect admission

State-changing tools also require an explicitly injected `EffectAdmissionCheck`
and the exact admission generation captured at preview. Omitted configuration
uses the disabled `EffectAdmissionController`; reads remain available while
effects are disabled or stopped. The generation is part of the approval action
digest, so stop and re-enable cannot reuse an earlier approved preview. A new
generation requires fresh review, grants, tickets and approval.

The adapter rechecks admission before reservation, after approval storage and
again after the durable dispatch append immediately before invocation. Production
hosts use operator-managed durable admission and its independent rollback anchor.
A stop cannot recall an effect whose invocation has already begun; the final
check narrows that boundary without claiming cross-provider atomic execution.

## Injected protocols

- `DispatchToolProvider.get(name)` resolves the registered specification.
  `invoke(spec, arguments, effect=...)` invokes the **pinned** implementation
  once, using the recorded idempotency key. It keeps payloads and tool outputs
  private. A provider must not silently substitute another version or transport.
- `DispatchApprovalProvider.consume_authorization(...)` uses the existing
  `ApprovalTokenVerifier` signature and returns protected local
  `ApprovalAuthorization` after signature, action, role, time and nonce checks.
  Its public `receipt` exposes only codes and digests. Serialized receipts and
  dictionaries never supply role or validity authority; only the trusted local
  verifier may create the execution context. The checkpoint records the receipt
  digest and verified expiry, without retaining the bearer token or key.
- `DispatchEffectStore.claim(checkpoint)` atomically reserves the **run/action**,
  including attempts with changed argument digests, and persists the initial
  content-free intent. Existing reservations return `False`. Failed or
  interrupted claims must remain fenced pending review. `append(checkpoint)`
  persists validated successors. `observe(effect)` queries the actual sink by
  idempotency key and returns an existing `EffectObservation`; it must use
  `AMBIGUOUS` when commit state cannot be established. A tool return value alone
  is not commit evidence.

These protocols permit offline tests without depending on pending tool-adapter
or action-ledger PRs. Production integrations must provide their atomicity,
key custody, provenance and durability guarantees. This adapter does not claim
distributed exactly-once execution or implement automatic compensation.

## Existing executor integration

```python
from openmed.mcp.workflow import WorkflowRunner, WorkflowStateStore

# adapter is a host-configured GuardedDispatchAdapter with injected providers.
runner = WorkflowRunner(
    store=WorkflowStateStore(),
    executors={registered_spec.name: adapter},
)
result = runner.run({
    "steps": [{
        "id": "governed-step",
        "tool": registered_spec.name,
        "inputs": resolved_arguments,
        "max_attempts": 3,
    }],
})
```

The host must expose guarded executors for governed tools. Existing unrelated
executors retain their behavior; installing this adapter does not automatically
guard every MCP tool. The callable returns content-free evidence only. The
runner's usual egress de-identification still applies to successful outputs;
`adapter.dispatch(arguments)` returns the typed `DispatchResult` directly.

A non-successful dispatch raises `GuardedDispatchError` at the callable boundary.
The runner stops after one attempt even if the step requested retries, records
the safe result in `trace[].dispatch`, and does not invoke downstream steps.
Successful runner resume reuses the existing cached step result without another
tool invocation. Reservations also prevent repeat invocation across independent
adapter instances and concurrent deliveries. A changed action needs a new opaque
action identity and fresh review, not a retry of an uncertain reservation.

## Cancellation and uncertain commit state

The adapter records intent before consuming approval or invoking the tool, then
records approval and dispatch boundaries. It checks cooperative cancellation and
rechecks expiry and registered identity after approval and durable dispatch
storage. Once invocation starts, an arbitrary tool cannot be recalled. Providers
should propagate cancellation using `asyncio.CancelledError` or the injected
cooperative probe; the adapter makes no assumption that cancellation undid an
effect.

Cancellation produces an aborted action with a typed `RecoveryDecision` reason
`workflow_aborted` and disposition `review_required`. Tool exceptions, absent or
ambiguous sink observations, inconsistent commit evidence, duplicate reservation,
or failure to durably record completion produce `waiting_review` and
`ambiguous_effect`. No retry or compensation is executed. The original intent
stays reserved. Successful dispatch requires a matching committed sink observation
and a persisted completed checkpoint.

`DispatchResult.to_dict()` contains existing `WorkflowOutcome`, action phase,
checkpoint and recovery contracts. All diagnostics contain controlled codes,
opaque identifiers and digests; no argument values, tool outputs, provider
exception text, credentials or private source paths are emitted. If diagnostic
storage fails, returned recovery evidence describes the proposed safe outcome;
it does not attest that the final checkpoint was persisted. The stored lineage
is authoritative for later recovery.

## Offline validation

```bash
.venv/bin/python -m pytest tests/unit/agent/workflows/test_dispatch.py tests/integration/agent/test_guarded_dispatch.py -q
```

Synthetic fixtures independently mutate grant signatures/scope/expiry, ticket
run/purpose/classes/selectors/tool action, projection, registered identity and
approval signature/role/expiry/arguments. They also exercise cancellation at each
boundary, concurrent deliveries, durable-storage failure, uncertain sink state,
and private-value negative controls on the real workflow runner. No model,
restricted dataset, external service or benchmark claim is involved.
