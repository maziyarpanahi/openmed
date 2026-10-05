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
can inject `SQLiteApprovalNonceStore` using the same application-owned private
local database file for every consumer:

```python
from openmed.agent.approvals import SQLiteApprovalNonceStore

store = SQLiteApprovalNonceStore("approval-nonces.sqlite3", timeout=5)
verifier = ApprovalTokenVerifier(key, store)
```

Provision the parent directory privately before constructing the store; do not
use a network filesystem or allow other users to replace the database. The
store creates its file with mode `0600` and enforces owner-only permissions on
POSIX. On Windows, provision an owner-only directory ACL. Each claim opens a
connection, takes an exclusive transaction, purges entries whose integer expiry
is at or before `now`, and inserts the nonce digest under a unique key. The
rollback journal and `synchronous=EXTRA` make successful commits durable within
SQLite's local filesystem guarantees. No connections need closing by callers.
Only nonce digests and integer expiries are stored; tokens, signatures, keys,
action digests, and reviewer roles are never persisted.

Reopening the same database preserves unexpired claims. Lock timeouts,
corruption (including empty or truncated existing files), missing files during
claims, invalid schemas, and unknown schema versions raise
`ApprovalNonceStoreError` with controlled codes. Never recover by deleting or
resetting a database while approvals remain valid: that discards replay
protection. Keep the database through every outstanding token's expiry and
deny dispatch when storage fails. The trusted clock must be consistent across
consumers; moving it forward may purge entries that another consumer still
considers unexpired.

### MCP consent receipt adapter

`ConsentReceiptVerifier` keeps its existing process-local default. To protect
receipts across MCP processes and restarts, inject the same claim protocol:

```python
from openmed.mcp.consent_receipts import ConsentReceiptVerifier

consent_verifier = ConsentReceiptVerifier(
    key_provider, consumption_store=store,
)
```

The adapter hashes a domain-separated receipt ID and rounds fractional expiry
up to integer seconds (current time is rounded down). This keeps a receipt
claimed throughout its exclusive lifetime. Binding failures do not consume
receipts, preserving the existing behavior. Store failures raise
`ConsentReceiptStoreError`, also an `ApprovalNonceStoreError`, and
`verify_result()` returns `nonce_store_unavailable`; the policy denies dispatch.
`is_consumed()` and `consumed_receipt_ids` remain local diagnostic snapshots of
this verifier's successful consumption, not queries of the shared database.
Issuance, token formats, and signing keys are unchanged.

The HMAC key is also application-owned and stays local. OpenMed performs no
network request, key lookup, telemetry, or notification. Persistence is opt-in
through the injected store. HMAC is a shared-key primitive: any component that
can verify with the key can also
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
.venv/bin/python -m pytest tests/unit/agent/approvals/ tests/unit/mcp/test_consent_receipts.py tests/integration/test_durable_approval_nonces.py -q
```
