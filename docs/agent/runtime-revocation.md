# Runtime clinical-agent authority revocation

Static signature, expiry and scope checks do not establish whether a grant is
still active. `openmed.agent.permissions.revocation` adds an injected local
status provider and an independent monotonic generation store. The adapters in
`openmed.agent.permissions.runtime_authority` compose this check with the existing
capability grant, purpose ticket, delegation, single-use approval and recovery
contracts. They perform no network access and add no identity service.

## Provider and generation contracts

`AuthorityContract` identifies a complete artifact by kind, SHA-256 digest and
exclusive expiry. Use `grant_authority`, `ticket_authority` and
`delegation_authorities` to derive these references from existing contracts.
Ticket references commit to every existing ticket field, including its run,
purpose, projection, keyed record selectors, tool actions and expiry. Static
verification is still required; hashing an artifact does not authorize it.

The trusted application registers these references with its injected
`AuthorityStatusProvider`. `get_status(kind, digest)` must return an exact
`AuthorityStatus` or `None` for unknown authority. Each revocation advances an
integer generation and permanently tombstones that artifact digest. Renewed
authority needs a new artifact, not resurrection of a revoked digest. Provider
observation times must use the runtime clock domain and describe when authority
was actually observed; returning a cached record cannot refresh its timestamp.

Every boundary performs new lookups. Unknown, unavailable, malformed, mismatched,
future-dated or stale status fails closed. The default acceptable status age is
zero seconds; an application may explicitly configure a bounded age. This is a
freshness policy, not a guarantee about an untrusted provider. A provider must
authenticate and serialize its own authority changes; these protocols cannot
detect a provider lying about current state.

`AuthorityGenerationStore.observe(status)` must atomically retain high water and
tombstones across all consumers. It rejects decreasing generations, state
changes at an unchanged generation, and any revival of a revoked digest. Store
failures also fail closed. `InMemoryAuthorityGenerationStore` supports a single
process. Production restart recovery requires a durable injected store retained
independently of run checkpoints, or an equivalent trusted authoritative storage
boundary. Restoring both authority state and high water to an older copy defeats
local rollback detection. No production persistence or identity backend is
provided by this slice.

## Preview, read and effect boundaries

After verifying static contracts, capture all authority used by the run:

```python
grant_verifier.verify(manifest, grant_request)
ticket_verifier.verify(ticket, ticket_request)
delegation_verifier.verify_chain(chain)

contracts = (
    grant_authority(manifest),
    ticket_authority(ticket),
    *delegation_authorities(chain),
)
binding = runtime.bind(contracts)
```

`runtime` is an `AuthorityRuntime` constructed with the application's provider,
generation store and trusted clock. `bind` queries active status at preview and
pins each generation. It never grants authority by itself.

Use `read_with_revocable_ticket` before a sensitive read. Use
`dispatch_with_revocable_grant` and `dispatch_with_revocable_delegation` before
effects. Each adapter checks the supplied artifact against the preview binding,
repeats existing static verification and then checks fresh status immediately
before invoking its callback. The delegation adapter requires every ancestor in
the verified root-to-leaf chain; a leaf-only binding fails closed. Revocation of
any bound authority invalidates the whole run boundary, including descendants.

These adapters can be nested when an operation needs several contracts. Carry
the same complete binding through every adapter; do not substitute a freshly
captured binding at execution. Static-only APIs remain available for compatibility;
hosts enforcing revocation must route every protected operation through runtime
adapters. An unguarded callback or legacy static-only call is outside this policy.

## Outstanding approvals

At issuance or preview, create `ApprovalAuthorityBinding.create(token, binding)`.
Persist it with the protected reviewed action state. It contains the token digest
and the exact preview authority generations, never the bearer token.

`dispatch_with_revocable_approval` requires that exact token, verifies and consumes
it using the existing `ApprovalTokenVerifier`, and rechecks authority after nonce
consumption. Revocation during consumption prevents the effect and leaves the
token consumed. The adapter retains verified local approval validity and
checks its exclusive expiry again after all status/store callbacks at the
same final clock boundary. Public receipts remain codes and digests and cannot
supply execution authority. An active status with a changed generation also invalidates the
original approval, requiring a new preview and fresh human approval. A different
token cannot reuse the saved binding. Nest static grant, ticket and delegation
adapters in the effect callback as appropriate; approval alone never establishes
tool or data scope.

This conservative policy invalidates outstanding approvals on every bound
generation change. It does not change the existing token wire format, key
rotation, nonce-store contract or cached-output consent invalidation.

## Recovery and diagnostics

Retain `binding.to_dict()` alongside protected run state and restore it with
`AuthorityBinding.from_dict`. The application must authenticate this state and
its association with the run, action and approval. Bindings are content-free
checkpoint records, not signed credentials; a digest is not an integrity check
against an attacker able to rewrite the record and its digest.

Call `recover_with_revocable_authority` before reconciliation. It checks current
authority before consuming checkpoint or observation iterables, then delegates
to the existing read-only recovery planner. Lazy observation lookups therefore
run only after the check. Eager sink reads need their own read adapter. Recovery
cannot override the live runtime clock with historical reconciliation time, renew
the binding, restore high water from checkpoint data or authorize a later effect.
A valid resume plan still needs fresh read and dispatch checks at the next
operation. Existing idempotency and committed-effect rules remain unchanged.

`AuthorityDeniedError.receipt.to_dict()` contains only a controlled schema,
boundary, reason code, authority count and binding digest. Reasons distinguish
`authority_revoked`, `authority_expired`, `authority_generation_changed`,
`authority_status_unavailable`, `authority_status_stale` and
`authority_status_rollback`. Provider and store exception text is suppressed.
Keep these linkable digests under audit-data access controls; never log artifact
contents, signing keys, record selectors, source payloads or private paths.

The final status check is the local admission boundary. Effects already admitted
or in flight cannot be recalled. Atomic enforcement across an external effect
sink and concurrent revocation requires the application's trusted dispatcher
and provider to share a transaction or lease boundary. This slice does not claim
distributed atomic revocation, autonomous clinical action or release readiness.
It adapts existing Python authority contracts; no matching Swift authority
contracts or Apple Foundation Models execution path is introduced.

Focused offline validation:

```text
.venv/bin/python -m pytest tests/unit/agent/permissions/test_revocation.py tests/unit/agent/permissions/test_runtime_authority.py tests/integration/agent/test_authority_revocation.py -q
```
