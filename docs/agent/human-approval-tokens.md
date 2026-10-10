# Single-use human approval tokens

`openmed.agent.approvals` provides a local, fail-closed approval gate for
high-impact actions. A token is signed over exactly seven claims:

- the SHA-256 digest of the action or reviewed side-effect preview;
- a canonical reviewer policy role;
- an exclusive Unix expiry timestamp;
- a random 128-bit nonce;
- a non-secret signing-key identifier;
- a Unix issuance timestamp; and
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
# Synthetic injected time for a fully offline, reproducible example.
now = 1_999_999_900
action_digest = "sha256:" + "a" * 64
reviewer_role = "role:org.example/clinical-reviewer@1.0.0"

token = ApprovalTokenSigner(key, clock=lambda: now).issue(
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
    now=now,
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
protection. Keep the database through every outstanding token's expiry plus
the shared verifier skew allowance, and deny dispatch when storage fails. The trusted clock must be consistent across
consumers; moving it forward may purge entries that another consumer still
considers unexpired. All consumers sharing a store must use the same trusted
clock and skew policy.

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
The durable store does not alter issuance, token formats, or signing keys.

The HMAC key is also application-owned and stays local. OpenMed performs no
network request, telemetry, or notification. Key lookup is delegated only to
the application's injected local provider; persistence is opt-in through the
injected store. HMAC is a shared-key primitive: any component that can verify
with the key can also
issue a token, so keep signing in the component that authenticates reviewers.

## Key rotation and bounded validity

Issuance defaults to `openmed.agent.approval_token.v2`. `key_id` and `issued_at`
are part of the canonical HMAC payload: changing either invalidates the
signature. Keys are never derived from identifiers. `key_id` is a
**developer-authored, non-secret label**, matching `[a-z][a-z0-9._-]{0,127}`;
do not use key material, credentials, patient identifiers or private paths.
Passing raw bytes is shorthand for the single key identifier `default`.

Inject an `ApprovalKeyProvider` with `get_key(key_id) -> bytes | None`, or use
`MappingApprovalKeyProvider` over an application-owned mapping. Keep both
current and retiring entries during rotation. Sign with the selected `key_id`;
verification resolves each token's signed identifier. Removing the retiring
entry immediately produces `unknown_key`, even for an otherwise valid token.
A provider may return `None` or raise `KeyError` for unknown keys. Other provider
failures are reduced to `key_provider_unavailable`, without source details.
HMAC keys must be at least 32 bytes; providers and clocks must remain local.

Both signer and verifier accept `max_lifetime_seconds` (default 900, configurable
from 1 through 86,400) and `clock_skew_seconds` (default 0, configurable from 0
through 300). Supply matching policy at both ends; a stricter verifier fails
closed. The injected integer clock defaults to local Unix time. `issued_at`
defaults to that clock but can be supplied explicitly. Policy checks enforce:

- `expires_at > issued_at`;
- `expires_at - issued_at <= max_lifetime_seconds` (skew never enlarges this ceiling);
- `issued_at <= now + clock_skew_seconds`; and
- `now < expires_at + clock_skew_seconds` (exclusive upper edge).

Issuance checks these bounds before generating a nonce or signing. Verification
checks the signature and bounds before claiming the nonce. Nonce retention uses
expiry **plus skew**, so a token consumed before expiry cannot be replayed
during the tolerance window. The store protocol is unchanged. Token expiry
plus skew must fit the supported signed 64-bit timestamp range.

## Explicit v1 migration

v1 verification is disabled by default (`legacy_token_disabled`). For a short,
application-controlled migration, configure `ApprovalTokenVerifier(...,
allow_v1=True, legacy_key_id="retiring")` with the local provider. The default
legacy key label is `default`. This flag verifies the original five-claim HMAC
format, without adding unsigned v2 fields or guessing a key from the signature.
`ApprovalToken.from_dict` and `from_json` also require `allow_v1=True` to parse
v1 directly; direct token objects cannot bypass the verifier flag.

v1 has neither issuance time nor key identifier. Its historical total lifetime
and not-before time cannot be proven. Compatibility therefore restricts its
**remaining** validity (`expires_at - now <= max_lifetime_seconds`), applies the
same exclusive expiry/skew bound and nonce retention, and selects only the
explicit legacy key. Use v2 for full lifetime protection and disable compatibility
after outstanding approvals have been renewed. Issuance never creates v1 tokens.

## Value-free receipts and errors

Successful consumption returns `ApprovalReceipt` using
`openmed.agent.approval_receipt.v2`. Its exact fields are `schema_version`,
`code` (`approved`), `action_digest`, and `token_digest` (the SHA-256 digest of
the complete signed token). This codes-and-digests-only receipt replaces the
v1 receipt: reviewer roles and consumption/expiry timestamps are no longer
receipt fields. It omits key identifiers, issuance time, nonce, signature,
action payload, reviewer identity, and clinical values. Receipts are verification
evidence, not proof that the callback committed successfully and not evidence
that a clinical decision was correct.

Failures expose stable codes and fixed field names only:

- `invalid_signature` for changed or incorrectly keyed tokens;
- `expired` at or after expiry plus configured skew;
- `not_yet_valid` when issuance is beyond the allowed skew;
- `lifetime_exceeded` for an excessive signed lifetime;
- `unknown_key` when the selected key has been removed or is unknown;
- `key_provider_unavailable` when the injected provider fails;
- `legacy_token_disabled` for v1 tokens without explicit compatibility;
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
