# Effect admission and emergency stop

Agent effects require an explicit enable. `EffectAdmissionController()` has no
configuration and is **disabled**. A configured controller fails **stopped** if
its ledger or independent high-water anchor is missing, unreadable, corrupt,
unsigned, or inconsistent. This Python controller/CLI slice uses an injected
check; it does not depend on pending FHIR/OMOP dispatch implementations (#3661,
#3662, #3663) or the per-run circuit breaker (#2769).

## Operator state and persistence

`SQLiteAdmissionStore(ledger, anchor, key)` uses only the standard library and
local storage. Provision it explicitly with `initialize()`; existing files are
never overwritten. The key must contain at least 32 secret bytes. Key custody and
filesystem access are the operator authorization boundary: role codes are
assertions from that trusted control plane, not authentication credentials.
Protect both database files and the key with operator-only access. New databases
use mode `0600`; keep their parent directories protected too.

Keep the anchor on independently protected storage **outside ledger restore
operations**. Each transaction atomically appends an HMAC-SHA256 signed receipt
and advances the anchor using SQLite rollback journals and `synchronous=EXTRA`
for both databases. This includes directory synchronization when a DELETE-mode
journal is removed, following [SQLite's durability requirements](https://www.sqlite.org/pragma.html#pragma_synchronous).
The filesystem must provide reliable local locking and synchronization.
Restoring either database alone fails closed, including after restart. Restoring
both databases and the signing key together defeats local rollback detection;
this threat requires a trusted external storage/restore boundary. Losing or
corrupting state requires operator investigation and restoration of matching
trusted evidence, not an automatic reset or a `resume --initialize` bypass.

Receipts contain only a schema code, canonical workflow identifier (or `global`),
state/role/reason codes, integer time, monotonically increasing generation,
previous receipt digest, and signature digest. They contain no actor identity,
clinical payload, endpoint, source path, credential, or free-text reason. Clock
values do not determine generation order. CLI output is limited to scope,
reason code, generation and receipt digest. An untrusted status preserves the
highest known generation, using the surviving anchor when available; zero means
no trustworthy generation is known. No transition is permitted in that state.

Global disabled allows individual workflow enables; an enable admits only its
named workflow. Global enabled admits workflows unless a workflow has been
explicitly stopped. A global stop overrides all workflows and clears their prior
enables. A workflow cannot enable itself while global state is stopped. `resume`
is a **fresh recorded enable**, not restoration of an earlier run's authority.
Resuming global scope explicitly enables all workflows. An effect already in
progress cannot be recalled; the stop takes effect at the next check boundary.

## CLI

Use explicit ledger, independent anchor and protected key-file paths. Paths and
key bytes never appear in command results. Initially:

```sh
openmed agents status
# {"generation": 0, "reason_code": "admission_disabled", ...}

openmed agents resume --initialize --scope workflow:org.example/synthetic-fhir \
  --state "$ADMISSION_LEDGER" --anchor "$ADMISSION_ANCHOR" \
  --key-file "$ADMISSION_KEY_FILE"

openmed agents stop --state "$ADMISSION_LEDGER" --anchor "$ADMISSION_ANCHOR" \
  --key-file "$ADMISSION_KEY_FILE"

openmed agents status --scope workflow:org.example/synthetic-fhir \
  --state "$ADMISSION_LEDGER" --anchor "$ADMISSION_ANCHOR" \
  --key-file "$ADMISSION_KEY_FILE"
# reason_code: admission_stopped

openmed agents resume --state "$ADMISSION_LEDGER" --anchor "$ADMISSION_ANCHOR" \
  --key-file "$ADMISSION_KEY_FILE"
```

`--initialize` applies only to new stores. Missing configured state fails closed.
The role vocabulary is `operator` or `incident_commander`; stop defaults to the
latter and resume to the former. Status/rejected commands emit controlled JSON;
untrusted state exits with status 1. The production console script also accepts
`--json`, using the standard CLI envelope around the same content-free data.
The Typer frontend uses the same validation and controller. No command starts
agents, performs clinical
writes, loads models, contacts a server, or changes approval/grant policy.

## Adapter and recovery boundary

Import from `openmed.agent.admission`. Capture the current generation when
previewing an effect, then inject `EffectAdmissionCheck` immediately before
**every** dispatch, retry, and run resume. Read-only tools, previews, recovery
planning and review requests do not need effect admission. A changed generation
requires fresh preview and grant/approval evaluation, including after a stop and
fresh enable. Persist that generation with the host application's run metadata.

```python
from openmed.agent.admission import dispatch_with_admission

preview_generation = admission.require_admitted(workflow_id).generation
# Produce the content-free preview and obtain the exact action approval.
result, receipt = dispatch_with_approval_token(
    token,
    action_digest=action_digest,
    reviewer_role=reviewer_role,
    verifier=verifier,
    dispatch=lambda: dispatch_with_admission(
        effect_adapter,
        workflow_id=workflow_id,
        admission=admission,
        generation=preview_generation,
    ),
)
```

The admission wrapper belongs **inside** the approval callback: a valid token
cannot bypass a stop between preview and dispatch. Tokens consumed before a
rejected dispatch require new approval. `dispatch_with_admission` defaults to a
disabled controller when no check is injected. Existing approval verification
and recovery planning helpers remain independent; they are not effect adapters
and do not themselves provide admission. For a recovery plan returning `resume`,
call `admission.require_admitted(checkpoint.workflow_id, generation=saved_generation)`
before resuming, then check again at each adapter boundary.

Focused synthetic tests cover default-off FHIR/OMOP effects, scope isolation,
valid approval after stop, stopped restart/recovery, stale generations, missing
and restored state, tampering, transactional interruptions, concurrent operators,
and value-free CLI errors. No model or clinical efficacy claim is made.
