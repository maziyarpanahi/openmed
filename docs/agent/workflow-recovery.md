# Durable workflow recovery

OpenMed recovery checkpoints let a caller restart a governed workflow without
blindly repeating local tool calls, FHIR writes, or staged OMOP commits. The
contract is local, deterministic, and content-free. It does not execute a tool,
contact a server, open a database, consume an approval token, or perform a
compensating clinical write.

## Safety boundary

Each effect records only an opaque run/action identifier, a governed tool ID,
an effect category, operation and commit-evidence digests, a deterministic
idempotency key, categorical state, and a compensation limit. Checkpoints do
not accept prompts, tool arguments, FHIR resources, OMOP rows, patient or
reviewer identifiers, URLs, credentials, paths, or free-text status.

The three effect categories share one recovery rule:

| Effect | Required adapter behavior | Compensation limit |
| --- | --- | --- |
| Local tool | Query the durable local effect store by idempotency key | `none` or proposal only |
| FHIR write | Query conditional-write or transaction evidence by the same key | Proposal only; never an automatic corrective write |
| Staged OMOP batch | Query the committed batch digest by the same key | Proposal only; execute rollback elsewhere after review |

An adapter must report `absent`, `committed`, or `ambiguous`. Recovery retries
only a proven-absent effect. A matching committed effect is recorded without
repeating it. Ambiguous, missing, conflicting, or changed evidence fails closed
to human review.

## Approval handling

Recovery never accepts a bearer approval token. The checkpoint may contain only
the digest of a value-free approval receipt, its exact approved action digest,
and its exclusive expiry. The approved action digest must equal the workflow
plan digest. A missing or expired receipt cannot authorize a pending retry.
Because the API has no token input, a consumed token cannot be replayed during
recovery; the caller resumes the same logical effect under its existing receipt
and idempotency key.

Applications should checkpoint the approval receipt before dispatch. If a
workflow needs a fresh review after expiry or ambiguity, start a newly approved
recovery lineage rather than changing receipt metadata in place.

## Durable append-only journal

`CheckpointJournal` stores one canonical JSON file per sequence. It writes a
private temporary file, flushes it, and atomically links the final sequence
name. On POSIX it also flushes the directory; Windows does not expose directory
`fsync` through Python, so power-loss durability of the new directory entry is
filesystem-dependent there. An identical append is idempotent; a conflicting
append is rejected. Every checkpoint binds the previous checkpoint digest,
workflow/run identity, plan digest, approval receipt, ordered effects, commit
evidence, and recovery evidence digest.

Loading validates the full chain. Gaps, changed identities, backward effect
state, modified commit evidence, terminal-state successors, symlinks, unknown
journal entries, malformed JSON, and digest mismatches raise a value-free
`RecoveryError`. Treat every such error as `review_required`; do not continue
from a partially trusted journal.

## Recovery sequence

```python
lineage = journal.load()
checkpoint = lineage[-1]

# Adapter code performs read-only lookups by each effect.idempotency_key.
observations = inspect_effect_sinks(checkpoint.effects)
decision = recover_workflow(lineage, observations, now=clock())

next_checkpoint = advance_checkpoint(checkpoint, decision)
journal.append(next_checkpoint)

if decision.disposition is RecoveryDisposition.RESUME:
    dispatch_only(decision.retry_idempotency_keys)
elif decision.disposition is RecoveryDisposition.REVIEW_REQUIRED:
    route_to_human_review(decision.reason)
```

Persist the `DISPATCHING` checkpoint before issuing retries. After interruption,
re-query every sink; do not infer commit from a transport response cached in
memory. `RecoveryDecision` JSON and its digest are deterministic for the same
checkpoint, observations, and caller-supplied time. `advance_checkpoint` binds
that evidence into the next append-only checkpoint.

Recovery phases are checkpoint boundaries only. They do not replace the shared
agent action lifecycle. Adapter integrations should map their action phase into
the nearest checkpoint boundary while preserving the lifecycle's own transition
validation.

## Governed workflow CLI

The console entry and the Typer application expose the same guarded command
boundary. `plan` and `preview` are aliases for read-only plan inspection;
`inspect` reads run status. `submit-review`, `cancel` and `resume` are explicit
mutating requests to an injected service. No command approves an action or
enables a clinical effect adapter by default.

```bash
openmed agents workflow --help
openmed agents workflow preview --request request.json
openmed agents workflow inspect --request request.json
openmed agents workflow submit-review --request request.json
openmed agents workflow cancel --request request.json
openmed agents workflow resume --request request.json --receipt receipt.json
```

Inputs are caller-owned local regular files. On POSIX they must belong to the
current user and have no group/other permissions, such as mode `0600`. Final
symlinks and FIFOs are refused, reads are bounded to 64 KiB, and changing files,
duplicate JSON keys, non-finite values, deep documents and unknown fields are
refused. Platforms without `O_NOFOLLOW` return a controlled protection failure.
Do not pass tokens, credentials, identities or clinical payloads as command-line
arguments. Global configuration overrides are not accepted on this surface.

A request has exactly these fields:

```json
{
  "schema_version": "openmed.cli.workflow_request.v1",
  "run_id": "run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "workflow_id": "workflow:org.example/synthetic@1.0.0",
  "action_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "expected_state_digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
}
```

Use the exact action/preview digest and current state digest from the caller's
service. `expected_state_digest` may be `null` for read-only commands; mutating
commands require it. The request refers to clinical inputs already held by the
service, rather than embedding values, FHIR resources or OMOP rows.

`--receipt` accepts only the existing metadata-only `ApprovalReceipt` schema.
A receipt file is not authority by itself. Resume checks exact action binding,
receipt consumption/expiry, the current snapshot and a typed verification result
from trusted durable service custody. The verification must bind the same action,
receipt digest and state digest. Expiry is checked again after verification.
Changing a receipt's role, token digest or expiry changes its receipt digest and
cannot establish verification. No bearer approval token is accepted or consumed.

The service must atomically compare the expected state/action and reverify receipt
custody and expiry at dispatch, including policy roles and replay/idempotency
rules. CLI inspection alone cannot close a race between inspection and dispatch.
The injected service owns authority enforcement, effect execution and recovery;
these commands do not duplicate the workflow executor, FHIR or OMOP adapters.

Every operational response is one value-free JSON document with
`schema_version="openmed.cli.governed_workflow.v1"`, including refusals.
`--json` is accepted for consistency; JSON is also the default. Responses contain
opaque run identifiers, trusted developer workflow names, digests, categorical
phase/status and bounded effect counts. Receipt roles, reviewer identities,
tokens, source values and private paths are omitted. Adapter output on standard
output/error is suppressed, but adapters remain trusted code and must implement
their own value-free logging; this is not a callback sandbox.

| Exit | Meaning |
| --- | --- |
| 0 | Ready preview or acknowledged completed resume |
| 1 | Service failure or invalid service acknowledgement |
| 2 | Invalid, malformed, unreadable or unprotected input |
| 3 | Denied authority or an unverified receipt |
| 4 | Review required, requested review, future receipt or expired receipt |
| 5 | Acknowledged cancellation |
| 6 | Action/state/receipt conflict or terminal resume |
| 7 | No configured adapter or an unsupported adapter operation |

A successful review request still exits `4`, because the action needs human
review. An acknowledged cancellation exits `5` with `ok=true`. If a service
fails or returns a malformed acknowledgement after a mutation, its commit state
may be unknown: reconcile through the existing recovery protocol. The CLI never
automatically retries, rolls back or compensates an effect.

### Offline service injection

Applications inject a trusted local service through
`openmed.cli.main(argv, governance_service=service)`,
`build_app(governance_service=service)` for Typer, or
`run_governed_workflow_cli(argv, service=service, clock=clock)`.
The protocol is `WorkflowCLIGovernanceService`; the clock is an injected Unix
second function, never a command-line override. No service is dynamically loaded
from request JSON or environment variables.

This synthetic example creates a private temporary metadata request and invokes
the actual CLI boundary offline. The preview service executes no effects:

```python
from openmed.agent.action_phases import ActionPhase
from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from openmed.cli.governed_workflows import (
    WorkflowCLIRequest, WorkflowCLIStatus, WorkflowCLIView,
    run_governed_workflow_cli,
)

request = WorkflowCLIRequest(
    RunId("run_" + "a" * 32),
    WorkflowId("workflow:org.example/synthetic@1.0.0"),
    "sha256:" + "a" * 64,
)

class PreviewOnly:
    def preview(self, request):
        return WorkflowCLIView(
            request.run_id, request.workflow_id, request.action_digest,
            "sha256:" + "b" * 64,
            ActionPhase.READY, WorkflowCLIStatus.READY,
            proposed_effect_count=1,
        )

view = PreviewOnly().preview(request)
assert view.status is WorkflowCLIStatus.READY
assert view.committed_effect_count == 0

with TemporaryDirectory() as directory:
    path = Path(directory) / "request.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(request.to_dict(), stream)
    assert run_governed_workflow_cli(
        ["workflow", "preview", "--request", str(path)],
        service=PreviewOnly(),
    ) == 0
```

With protected request files, the same adapter can be supplied to the console
entry for preview. Its unsupported operations return exit `7`; the example
does not provide approval, execution or recovery capabilities. Without injection,
valid operational CLI requests also return `adapter_unavailable` with exit `7`.

## Open integration dependencies

The generic journal and reconciliation engine intentionally do not duplicate
the contracts tracked by #2766, #2767, #2768, #2771, #2773, #2774, #2775,
#2776, #2778, #2996, #2998, and #3085. Those contracts remain responsible for
ledger evidence, replay verification, single-use approval receipts, previews,
FHIR conditional/concurrency/compensation/subscription behavior, staged OMOP
batches and rollback manifests, action graphs, and action lifecycle phases.
