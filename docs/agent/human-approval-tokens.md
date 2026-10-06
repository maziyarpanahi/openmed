# Single-use human approval tokens

`openmed.agent.approvals` provides a local, fail-closed approval gate for
high-impact actions. A token is signed over exactly five claims:

- the SHA-256 digest of the action or reviewed side-effect preview;
- a canonical reviewer policy role;
- an exclusive Unix expiry timestamp;
- a random 128-bit nonce; and
- the token schema version.

Changing any signed claim invalidates the signature. The verifier atomically
claims the nonce before it compares the expected action and role. A successful
verification, changed action, or wrong role therefore makes that token
unusable. Expired, malformed, or incorrectly signed tokens fail without
dispatching the action.

## Issue and consume a token

The application must authenticate the human reviewer and enforce its own role
policy before calling `ApprovalTokenSigner.issue()`. This module binds the role
that the application supplies; it does not authenticate a person, make a
clinical judgment, or decide which actions require approval.

```python
from openmed.agent.approvals import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
    dispatch_with_approval_token,
)

key = b"replace-with-32-or-more-local-key-bytes"
action_digest = "sha256:" + "a" * 64
reviewer_role = "role:org.example/clinical-reviewer@1.0.0"

token = ApprovalTokenSigner(key).issue(
    action_digest=action_digest,
    reviewer_role=reviewer_role,
    expires_at=2_000_000_000,
)

verifier = ApprovalTokenVerifier(key, InMemoryApprovalNonceStore())
result, receipt = dispatch_with_approval_token(
    token,
    action_digest=action_digest,
    reviewer_role=reviewer_role,
    verifier=verifier,
    dispatch=lambda: "local result",
    now=1_999_999_999,
)
```

The callback has no token-layer arguments, which keeps the action payload out
of approval diagnostics. The token is consumed before the callback starts and
remains consumed if the callback fails. Persist the returned receipt as needed
before reporting the action as approved or committed.

## Exact-action and role binding

Pass the digest of the exact canonical action representation that the reviewer
saw. When a governed side-effect preview is available, use its digest directly.
Do not place action payloads, clinical values, record identifiers, reviewer
identities, credentials, paths, or free text in `reviewer_role` or other token
fields. A reviewer role uses this developer-authored form:

```text
role:<reverse-domain>/<local-name>[@<version>]
```

The token layer accepts a digest rather than raw action content, so it cannot
leak that content through exceptions or object representations. Avoid making a
plain digest from a single low-entropy sensitive identifier; bind the complete
canonical action or a separately governed preview artifact instead.

## Replay protection and deployment scope

Every verifier requires an `ApprovalNonceStore`. Its `claim()` operation must
be atomic for every process that can execute the protected action.
`InMemoryApprovalNonceStore` is thread-safe but process-local, so it is suitable
for one-process local workflows and tests. Multi-process or restarted services
must inject an application-owned durable local store with the same atomic
claim semantics. Store only the nonce digest supplied to `claim()`, not the
serialized bearer token.

The HMAC key is also application-owned and stays local. OpenMed performs no
network request, key lookup, telemetry, notification, or persistence. HMAC is
a shared-key primitive: any component that can verify with the key can also
issue a token, so keep signing in the component that authenticates reviewers.

## Value-free receipts and errors

Successful consumption returns `ApprovalReceipt`. It contains only the action
digest, reviewer role, a digest of the complete signed token, consumption and
expiry timestamps, and a schema version. It omits the nonce, signature, action
payload, reviewer identity, and clinical values. Receipts are verification
evidence, not proof that the callback committed successfully and not evidence
that a clinical decision was correct.

Failures expose stable codes and fixed field names only:

- `invalid_signature` for changed or incorrectly keyed tokens;
- `expired` at or after the exclusive expiry;
- `replayed` after a nonce has been claimed;
- `action_mismatch` for a material action change; and
- `reviewer_role_mismatch` for the wrong policy role.

Never log serialized approval tokens because they are bearer credentials. Safe
diagnostics may record the failure code and field name. Verification and tests
are deterministic when `now` and the nonce are supplied explicitly.

Run the focused offline tests with:

```text
.venv/bin/python -m pytest tests/unit/agent/approvals/test_tokens.py -q
```

## Distinct-role approval quorums

`ApprovalQuorumEvaluator` evaluates existing, successfully consumed
`ApprovalReceipt` objects against an `ApprovalQuorumPolicy` keyed by a controlled
`action_class`. This Python receipt-policy slice leaves token issuance, handoff
leases and dispatcher wiring unchanged; it introduces no Swift token verifier
or autonomous clinical action.

```python
from openmed.agent.approvals import (
    ApprovalQuorumEvaluator,
    ApprovalQuorumPolicy,
    SQLiteApprovalQuorumStore,
)

clinician = "role:org.example/clinician@1.0.0"
pharmacist = "role:org.example/pharmacist@1.0.0"
requester = "role:org.example/trainee@1.0.0"
policy = ApprovalQuorumPolicy(
    action_class="high-impact-write",
    required_count=2,
    allowed_reviewer_roles=(clinician, pharmacist),
    distinct_roles=True,
    excluded_requester_roles=(requester,),
)
evaluator = ApprovalQuorumEvaluator([policy])
decision = evaluator.evaluate(
    action_class="high-impact-write",
    action_digest="sha256:" + "a" * 64,
    requester_role=requester,
    receipts=(),  # Supply trusted receipts from successful token consumption.
    now=20,
)
assert decision.approved_count == 0
assert not decision.satisfied
```

Classes are developer-authored lower-case labels such as `high-impact-write`;
unknown classes and duplicate policies fail closed. Counts must be positive.
Policies can require distinct roles or count multiple unique receipts from one
role. The actual requester role is **always** excluded; the policy's
`excluded_requester_roles` additionally excludes designated requester-category
roles even when they differ from the current requester. Impossible distinct-role
policies are rejected at construction. An otherwise feasible policy may remain
unsatisfied when the actual requester is one of its eligible roles.

Evaluation excludes different action digests, disallowed roles, requester roles,
future consumption timestamps and receipts at or beyond their exclusive expiry.
Repeated token digests within a supplied set exclude **all** copies, including
copies with conflicting roles. `replayed_receipt_digests` accepts token digests
already used elsewhere. A stateless evaluator cannot detect cross-call replay;
use durable collection for incremental approvals. Receipt schema validation does
not authenticate receipt provenance: never accept caller-edited receipt JSON as
verified evidence. Roles do not identify people; hosts must authenticate reviewers
and enforce person independence before issuing tokens.

Decisions serialize only `action_digest`, `policy_digest`, `requester_role`,
`reviewer_roles`, `receipt_digests`, `required_count` and `approved_count`.
`receipt_digests` are the existing receipts' token commitments, not bearer tokens.
`satisfied` is a derived property, not an extra serialized field. Decisions contain
no action class, source values, timestamps, reviewer identities or free text.
The policy digest commits to all policy rules; role ordering does not change it.

### Durable partial approvals

Create `SQLiteApprovalQuorumStore` with an application-owned, protected, dedicated
local database path. Call `collect` with the evaluator arguments above plus a
`progress_digest`: a stable opaque commitment to one logical action slot. Use
separate slots for independent actions and reuse the slot when editing that
action. Do not derive slots from lone sensitive identifiers.

```python
# The application supplies the protected path and new verified receipts.
def collect_partial_approvals(database_path, verified_receipts):
    return SQLiteApprovalQuorumStore(database_path).collect(
        progress_digest="sha256:" + "c" * 64,
        evaluator=evaluator,
        action_class="high-impact-write",
        action_digest="sha256:" + "a" * 64,
        requester_role=requester,
        receipts=verified_receipts,
        now=20,
    )
```

Collection atomically preserves partial approvals across independent connections
and restarts. A changed action digest, policy digest or requester role resets
that slot's receipts. Global token-digest replay tombstones survive these resets,
so reverting to an old action does not restore its approvals. Re-submission adds
nothing; a previously collected receipt may still count as existing progress.
Every submitted token digest is claimed, including ineligible submissions.
Multiple eligible receipts for one role are retained for expiry renewal but count
once under a distinct-role policy. Every read rechecks expiry; a database-wide
clock watermark rejects rollback so expired approvals cannot revive.

SQLite transactions serialize collection and reset operations. Callers must
supply the authoritative current action and serialize edits to the same logical
slot; a stale caller must not restore an older action binding. This store does not
replace revisioned handoff leases. Its decision is advisory and can be read
repeatedly: hosts remain responsible for single-use effect authorization,
revalidation at dispatch, reviewer authentication, database integrity and access
control. Token issuance and existing single-token dispatch behavior are unchanged.
No network calls, model outputs or restricted assets are added.

Storage contains only validated receipt metadata, roles, digests and timestamps;
errors use controlled codes such as `store_unavailable` and `clock_rollback` and
omit the private path. Replay tombstones are retained without automatic pruning;
hosts own database lifecycle and storage capacity. Never log token credentials or
source action payloads. Malformed receipt data and storage failures fail closed.

Run the synthetic offline policy and persistence controls with:

```text
.venv/bin/python -m pytest tests/unit/agent/approvals/test_quorum.py tests/integration/agent/test_approval_quorum.py -q
```
