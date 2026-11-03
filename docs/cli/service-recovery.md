# Service Recovery CLI

`openmed service-recovery` is the local operator interface for authorized
service recovery. It inspects the active service state, presents an exact
recovery plan, verifies that plan against live state, and applies one
supported recovery action only after a signed, single-use approval that is
bound to that plan.

The command owns only the adapter and the plan presentation. Backup storage,
model activation, migration engines, and job replay stay inside the component
services that a deployment injects. Nothing is enabled by default: the
`openmed` console entry point registers the help and completion surface, and
every command reports `adapter_unavailable` (exit 7) until a deployment wrapper
calls `run_service_recovery_cli` with its own services.

This page describes recovery mechanics only. It adds no autonomous clinical
action, cloud fallback, or publication authorization.

## Commands

| Command | Writes | Purpose |
| --- | --- | --- |
| `inspect` | None | Report the active generation, state digest, and the recovery points each component offers. |
| `plan --action A --recovery-point REF [--output FILE]` | Only the new `--output` file | Build the exact plan and the digest an approval must be issued for. |
| `verify --plan FILE [--approval FILE]` | None | Re-check the plan, live state, recovery point, and approval binding. The approval is never consumed. |
| `apply ACTION --plan FILE --approval FILE` | Component state | Consume the approval once and apply the exact plan under compare-and-set custody. |

The supported actions are `restore-backup`, `rollback-model`, and
`replay-jobs`. Each maps to one injected component service.

`inspect`, `plan`, and `verify` call only the read-only `inspect`,
`recovery_points`, and `propose` methods. They never call `apply` and never
claim an approval nonce. `plan --output` creates a new file with mode `0600`
using an exclusive create; an existing file is never replaced
(`output_exists`, exit 2). Without `--output`, `plan` writes nothing.

## Exit codes

Every command writes exactly one compact JSON document to stdout with
`schema_version` `openmed.cli.service_recovery.v1`, `ok`, `command`, and
`status`, plus either `data` or a fixed `error` code. Successful help output is
plain text.

| Exit | Status | Meaning |
| --- | --- | --- |
| 0 | `ready` / `completed` | Read-only command succeeded, or the plan was fully applied. |
| 1 | `failed` | A service failed or returned an invalid result. |
| 2 | `invalid` | Invalid, unprotected, or edited input, or an existing `--output` file. |
| 3 | `denied` | The approval signature, role, or nonce was rejected. |
| 4 | `held` | The component or approval cannot proceed yet (for example, writes are active or the approval expired). |
| 6 | `conflict` | State changed, the plan is stale, the action differs, or the recovery point is unknown or incompatible. |
| 7 | `unavailable` | No services, component, or approval verifier is injected. |
| 8 | `partial` | The component stopped after applying some steps. Nothing is retried or compensated automatically. |

## Approval binding

A plan records the action, the active state generation, the state digest, the
recovery point reference and digest, and the step count. `plan_digest` commits
to every field. `approval_digest` commits to the plan digest, action, active
generation, and state digest under the domain label
`openmed.cli.service_recovery.approval.v1`.

The operator's application issues an
[`ApprovalToken`](../agent/human-approval-tokens.md) for `approval_digest` with
`ApprovalTokenSigner` and a caller-provided HMAC key. `apply` checks, in order,
before anything changes:

1. The plan and approval files are regular files owned by the current user,
   are not symlinks, and are not readable or writable by group or others.
2. The plan digests match its fields, and the plan action matches the `apply`
   subcommand.
3. The active generation and state digest still equal the planned values
   (`state_changed`).
4. The component still offers the same compatible recovery point with the same
   digest and step count (`recovery_point_incompatible`, `plan_stale`).
5. The component is not holding recovery (`recovery_held`).
6. The approval token is bound to this exact `approval_digest`
   (`approval_stale`) and is unexpired (`approval_expired`).
7. The injected `ApprovalTokenVerifier` verifies the signature and role and
   atomically claims the nonce (`approval_denied`, `approval_replayed`).
8. The state is read once more; any change rejects with `state_changed`.

Only then does the component's `apply` run. The component must compare the
plan's generation and state digest with live state atomically and report
`conflict` without writes when they differ. The CLI rejects a reported
completion that does not advance the generation by exactly one or does not
change the state digest, so no command silently overwrites active state.

A consumed approval cannot be reused. After a held, conflicting, or partial
outcome, run `inspect` and `plan` again and obtain a fresh approval for the new
plan.

## Embedding in a deployment

Deployments construct `ServiceRecoveryServices` from their own component
services and call `run_service_recovery_cli`. The example below is a complete,
offline program with synthetic in-memory services. Real components replace the
`SyntheticState` and `SyntheticBackups` classes.

```python
import json
import os
import tempfile
import time
from io import StringIO
from pathlib import Path

from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.cli.service_recovery import (
    RecoveryAction,
    RecoveryCLIStatus,
    RecoveryOutcome,
    RecoveryPoint,
    RecoveryProposal,
    RecoveryStateSnapshot,
    ServiceRecoveryServices,
    run_service_recovery_cli,
)

KEY = b"synthetic-example-approval-key-32b"
ROLE = "role:org.example/recovery-operator@1.0.0"
POINT = RecoveryPoint("rp_" + "1" * 32, "sha256:" + "d" * 64, True)


class SyntheticState:
    def __init__(self):
        self.snapshot = RecoveryStateSnapshot(3, "sha256:" + "b" * 64)

    def inspect(self):
        return self.snapshot


class SyntheticBackups:
    def __init__(self, state):
        self.state = state

    def recovery_points(self, snapshot):
        return (POINT,)

    def propose(self, snapshot, recovery_point_ref):
        return RecoveryProposal(POINT, step_count=2)

    def apply(self, plan, receipt):
        current = self.state.snapshot
        if (current.active_generation, current.state_digest) != (
            plan.active_generation,
            plan.state_digest,
        ):
            return RecoveryOutcome(
                plan.plan_digest,
                RecoveryCLIStatus.CONFLICT,
                current.active_generation,
                current.state_digest,
                0,
            )
        self.state.snapshot = RecoveryStateSnapshot(
            plan.active_generation + 1, "sha256:" + "c" * 64
        )
        return RecoveryOutcome(
            plan.plan_digest,
            RecoveryCLIStatus.COMPLETED,
            self.state.snapshot.active_generation,
            self.state.snapshot.state_digest,
            plan.step_count,
        )


def run(argv, services):
    stream = StringIO()
    code = run_service_recovery_cli(argv, services=services, stdout=stream)
    return code, json.loads(stream.getvalue())


state = SyntheticState()
services = ServiceRecoveryServices(
    state=state,
    components={RecoveryAction.RESTORE_BACKUP: SyntheticBackups(state)},
    approval_verifier=ApprovalTokenVerifier(KEY, InMemoryApprovalNonceStore()),
    reviewer_role=ROLE,
)

with tempfile.TemporaryDirectory() as directory:
    plan_path = Path(directory) / "plan.json"
    code, result = run(
        [
            "plan",
            "--action",
            "restore-backup",
            "--recovery-point",
            POINT.reference,
            "--output",
            str(plan_path),
        ],
        services,
    )
    if code != 0:
        raise SystemExit(code)

    # The reviewing application signs the exact approval digest.
    token = ApprovalTokenSigner(KEY).issue(
        action_digest=result["data"]["approval_digest"],
        reviewer_role=ROLE,
        expires_at=int(time.time()) + 600,
    )
    approval_path = Path(directory) / "approval.json"
    descriptor = os.open(approval_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token.to_json())

    files = ["--plan", str(plan_path), "--approval", str(approval_path)]
    code, result = run(["verify", *files], services)
    print("verify:", code, result["status"])
    code, result = run(["apply", "restore-backup", *files], services)
    print("apply:", code, result["status"], result["data"]["active_generation"])
    code, result = run(["apply", "restore-backup", *files], services)
    print("repeat apply:", code, result["error"]["code"])
```

The program prints `verify: 0 ready`, `apply: 0 completed 4`, and
`repeat apply: 6 state_changed`: the second `apply` is rejected because the
state generation advanced, so the old plan and approval no longer match.

## Privacy and failure behavior

Output contains only controlled status and reason codes, counts, generations,
opaque `rp_` references, and digests. It never contains file paths, approval
signatures, key material, exception text, or component output; component
stdout and stderr are discarded while a service method runs. Invalid arguments
produce `input_invalid` without echoing the argument. Error messages are the
fixed string `Service recovery command refused.`

`verify` checks the approval binding and expiry but cannot check the signature
without consuming the single-use nonce. The signature is verified, and the
nonce claimed, only by `apply`.
