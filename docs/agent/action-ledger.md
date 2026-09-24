# Agent Action Ledger

`ActionLedger` records metadata-only transitions for proposed side effects in a
caller-owned local directory. Each sequence gets a private, immutable JSON file.
The next entry includes the preceding entry's SHA-256 digest. `load()` verifies
the complete chain, the run and action identity, metadata continuity, and the
state path before `record()` can append. Gaps, edits, conflicting writers, and
unknown files fail closed with stable, value-free error codes.

```python
from openmed.agent.audit.action_ledger import ActionLedger, ActionState
from openmed.agent.correlation import ActionId, RunId

ledger = ActionLedger("/private/operator-owned/action-ledger")
ledger.record(
    run_id=RunId.generate(),
    action_id=ActionId.generate(),
    state=ActionState.PROPOSED,
    actor_role="role:openmed.local/operator",
    grant_digest="sha256:" + "a" * 64,
    tool_digest="sha256:" + "b" * 64,
    resource_refs=("sha256:" + "c" * 64,),
)
evidence = ledger.export_evidence()
```

Subsequent records for the same action keep the grant, tool, and resource
commitments fixed. Allowed paths are `proposed → approved → attempted →
committed`, `proposed → attempted → committed` when external policy requires no
approval, and rejection from proposed, approved, or attempted. Terminal states
cannot advance. The role code may change as an action passes between operator
and reviewer; it must never contain a person's identity.

The caller must supply opaque run and action IDs, a governed role code, and
SHA-256 commitments to the grant, tool contract, and resource references.
Resource commitments should be keyed hashes of canonical references to avoid
guessing low-entropy clinical identifiers. Never pass raw FHIR IDs, URLs,
patient names, tool arguments/results, approval bearer tokens, or clinical text.
The API accepts no payload field. `export_evidence()` includes only verified
entries and the terminal digest. The output can be anchored in a separately
trusted store to detect replacement or truncation of the entire local chain;
a hash chain alone does not authenticate its writer.

Record a proposal before dispatch and the attempted state before invoking a
side-effecting adapter. Record committed only after the adapter supplies
independent success evidence; this module does not execute tools, verify grants
or approvals, reconcile ambiguous outcomes, or infer commit from a timeout.
The execution-phase contract in #2998 and adapter integrations remain separate.

Run the offline test with:

```bash
.venv/bin/python -m pytest tests/unit/agent/audit/test_action_ledger.py -q
```
