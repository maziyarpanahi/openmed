# MCP consent receipts

OpenMed MCP scopes and server-side authorization remain the source of
authority. A consent receipt adds durable, single-use proof that a client
approved one state-changing action. It does not replace OAuth scopes, identity
verification, or server-side authorization.

## What a receipt contains

`openmed.mcp.consent_receipts.ConsentReceipt` binds:

- a client, MCP tool, resource, and required scope;
- an `allow` or `deny` decision;
- issuance and expiry timestamps;
- a SHA-256 digest of the canonical JSON tool arguments; and
- a receipt identifier, key identifier, and HMAC-SHA256 signature.

The serialized receipt contains no tool arguments, bearer values, clinical
text, or exception messages. The argument digest uses sorted keys, compact
JSON separators, and rejects non-finite numbers. Keep resource and client
identifiers policy-safe; the receipt mechanism does not turn an arbitrary
application string into a de-identified identifier.

## Issuing and verifying a receipt

Key custody is injected. The example uses synthetic values and an in-memory
provider only; production applications should resolve keys from their own
vault or key service without placing key material in receipts or logs.

```python
from openmed.mcp.consent_receipts import (
    ConsentReceiptIssuer,
    ConsentReceiptVerifier,
    MappingConsentKeyProvider,
)

keys = MappingConsentKeyProvider({"local": "synthetic-signing-key"})
clock = lambda: 1_000.0  # inject a trusted UTC epoch clock in production
arguments = {"model_name": "synthetic-model", "all_models": False}

issuer = ConsentReceiptIssuer(keys, key_id="local", clock=clock)
receipt = issuer.issue(
    client="synthetic-client",
    tool="openmed_unload_model",
    resource="openmed://mcp",
    scope="mcp:state-changing",
    arguments=arguments,
    ttl_seconds=60,
)

verifier = ConsentReceiptVerifier(keys, clock=clock)
verifier.verify(
    receipt,
    "synthetic-client",
    "openmed_unload_model",
    "openmed://mcp",
    "mcp:state-changing",
    arguments,
)
```

Verification consumes a valid receipt atomically. A second verification,
expired receipt, changed tool, changed resource, changed scope, changed
client, or changed argument set fails closed. Failed binding checks do not
consume a receipt that was never authorized for that request.

Call `verifier.verify_result(...)` when an adapter needs a non-throwing result.
It returns an immutable result with `verified`, a stable `code`, and the verified
receipt only on success. Codes are `verified`, `missing_receipt`, `expired`,
`not_yet_valid`, `invalid_signature`, `binding_mismatch`, `key_unavailable`,
`decision_denied`, `replay`, or `invalid_receipt`. Results contain no exception
message, signature, key material, request argument, or clinical text.

## Optional MCP policy hook

Pass a `ConsentReceiptPolicy` to `create_mcp_server` to require receipts for
tools whose registry annotation has `readOnlyHint=False`:

```python
from openmed.mcp.consent_receipts import ConsentReceiptPolicy
from openmed.mcp.server import create_mcp_server

policy = ConsentReceiptPolicy(
    verifier=verifier,
    client="synthetic-client",
    resource="openmed://mcp",
    scope="mcp:state-changing",
    policy_version="gateway-policy-v1",
)
server = create_mcp_server(consent_policy=policy)
```

The server adds an optional `consent_receipt` transport field only to
state-changing tool schemas when this hook is enabled. The field is removed
before the original handler receives its arguments. Read-only tools keep their
existing schemas and remain callable without a receipt, including local stdio
servers created without a policy hook. A missing receipt is rejected before a
state-changing handler runs.

Policy audit records contain only receipt and policy identifiers, the argument
digest, the decision/outcome, and a policy version. Applications may provide
an `audit_sink` to retain these safe fields. Do not log receipt JSON, request
arguments, bearer values, or clinical content.

Consent is an authorization input, not a clinical decision. OpenMed does not
provide a consent UI, identity provider, or clinical approval workflow.
