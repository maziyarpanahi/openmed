# SMART token custody

`SmartTokenCustody` keeps SMART access and optional refresh credentials in
process memory. A trusted application component stores a credential and gives
the resulting random opaque handle to an agent. The agent never receives the
custody object or its trusted sender. When a trusted dispatcher receives an
approved action, it asks custody to validate the exact FHIR audience, expiry,
and required SMART v2 scopes before the bound sender receives an Authorization
header. The call has no built-in network request or token refresh.

```python
from openmed.interop.fhir.smart_custody import SmartTokenCustody

# Bind a trusted local transport; do not expose it to the agent or tool trace.
custody = SmartTokenCustody(send_authorized_request)
handle = custody.store(
    access_token=access_token,
    refresh_token=refresh_token,
    audience="https://fhir.example.test/r4",
    expires_at=token_expiry,
    scopes=("patient/Observation.rs",),
)
# After separate grant, approval, and FHIR write checks:
evidence = custody.dispatch(
    handle,
    audience="https://fhir.example.test/r4",
    required_scopes=("patient/Observation.r",),
)
ledger_scope_evidence = evidence.to_dict()
custody.revoke(handle)
```

`send_authorized_request(audience, authorization_header)` is a trusted callback
bound when custody is created. It should reject redirects, avoid logging
headers or response bodies, and keep request and response payloads outside
agent-visible traces. Its return value is discarded. Dispatch returns only
normalized required scope names; it does not return tokens, handles, URLs,
headers, refresh secrets, or clinical data. Failures use fixed reason codes.
The value-free scope evidence can be attached to an action ledger once that
ledger's entry contract is available (#2766); this module does not invent or
write ledger entries.

Run the separate [SMART scope audit](smart-scope-audit.md) before granting an
agent a handle. Custody checks whether a token covers the requested scopes;
it does not decide whether the grant is least privilege, authorize an agent
action, or approve a clinical write. The scope audit tracked in #3084 supplies
that pre-run review. Expired tokens fail closed. Refresh tokens are retained
only in memory and are never exchanged automatically; the trusted application
must register a new credential and revoke the old handle after refresh.

In-process Python objects are not a security boundary against arbitrary code
running in the same interpreter. Keep custody and the sender in trusted code;
give agent components only opaque handles. Python strings cannot be reliably
zeroized, so use process isolation for a stronger secret boundary.
