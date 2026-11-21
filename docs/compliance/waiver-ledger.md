# Privacy Waiver Lifecycle Ledger

`openmed.compliance.WaiverLedger` records the lifecycle of a privacy waiver as
an append-only sequence of local events. It is an evidence helper, not a
compliance certification, legal approval, or clinical decision mechanism.

## Safe record shape

Each event contains only:

- a sequence number;
- one controlled event type: `create`, `approve`, `supersede`, `revoke`, or
  `expire`;
- an opaque `waiver_id`;
- an opaque `policy_id`;
- the resulting state; and
- an optional opaque replacement waiver reference for `supersede`.

There are deliberately no identity, finding-text, reason, source-document, or
timestamp fields. Use references or hashes managed by the surrounding
governance system when additional evidence is needed. Do not put raw personal
data or finding text in an identifier.

## Lifecycle

```text
create  -> pending -> approve -> active
                                  |  |  |
                           supersede  |  expire
                                  |   |  |
                             superseded revoke
                                      |  |
                                   revoked expired
```

Only a pending waiver can be approved, and only an active waiver can be
superseded, revoked, or explicitly expired. A policy reference is required at
creation and must remain unchanged for later events. Invalid transitions fail
without appending a record.

## Local deterministic usage

```python
from openmed.compliance import WaiverLedger

ledger = WaiverLedger()
ledger.create("wvr_001", "pol_privacy_001")
ledger.approve("wvr_001")
ledger.expire("wvr_001")

print(ledger.render_active_state_counts())
# {"active":0,"expired":1,"pending":0,"revoked":0,"superseded":0}
```

The ledger performs no network calls and does not infer expiration from the
wall clock. Expiration is an explicit event so the same synthetic event
sequence produces the same state counts and JSON representation every time.

`to_json()` and `write_json()` expose the same controlled fields for local
audit storage. The aggregate renderer reports counts only; it never includes
waiver or policy identifiers.

This helper does not decide whether a waiver should be approved, whether a
policy exception is lawful, or whether a deployment is clinically safe.

## Corrective actions

Track a release finding with
`openmed.compliance.corrective_action.CorrectiveActionLedger`. Open a
`CorrectiveAction` with a CAPA identifier, source, severity, affected
capabilities and versions. Advance through investigating, action planned,
implemented, verified and closed. Verification requires an after-evidence
digest. Corrections, preventive actions and root causes use the codes in
`corrective_action.schema.json`.

Use `ledger.to_dict()` to export the events and their core audit chain. Loading
the snapshot with `CorrectiveActionLedger.from_dict()` verifies the chain and
replays the transitions. A local note is omitted from both records and chain
material.

Run the following commands against a local ledger directory. Each subcommand
accepts `--json` for the shared CLI result envelope.

```sh
openmed compliance capa new --directory /var/lib/openmed/capa --id CAPA-2026-0001 --source incident --severity high --capability deidentify --version 4.3.0
openmed compliance capa transition --directory /var/lib/openmed/capa --id CAPA-2026-0001 --state investigating
openmed compliance capa list --directory /var/lib/openmed/capa
openmed compliance capa verify --directory /var/lib/openmed/capa
```

A rejected transition exits with status 2. A ledger that fails verification
exits with status 1. Use `--allow-repo` for synthetic fixtures stored inside a
checkout.
