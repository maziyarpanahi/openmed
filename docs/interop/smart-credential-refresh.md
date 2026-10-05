# SMART credential refresh

`openmed.interop.fhir.smart_refresh.SmartCredentialRefresher` is an opt-in,
Python server-side credential acquisition and refresh boundary. It installs no
HTTP transport, makes no default network calls and adds no interactive app launch
or on-device OAuth client. It does not change the existing bulk-ingestion runner.

The host supplies a token-endpoint callable, an epoch-seconds clock and a custody
adapter. For backend-services client-credentials grants, supply the existing
`SMARTBackendConfig`; the refresher reuses `build_client_assertion` with the
injected clock. An optional assertion builder supports a trusted signer or
synthetic tests. The transport owns endpoint configuration, HTTP status handling,
client authentication, timeouts and decoding of OAuth success/error objects.
It must not log requests, responses, credentials, endpoints or client identifiers.

## Custody contract

`CredentialCustody.transaction(handle)` yields a `CredentialSlot` with `read`,
`replace`, `revoke` and a terminal `revoked` flag. `SmartCredential` is a protected
record containing the access token, optional refresh token, epoch expiry and
granted scopes. Its representation hides all fields; never serialize the record
into agent state, evidence, logs or public results.

The adapter must provide exclusion across **all** refresh, dispatch and revocation
clients for the same handle, including separate refresher instances. Keep the
transaction locked through transport completion, validation, atomic replacement
and the trusted dispatch callback. A failed operation must not roll back
revocation on transaction exit. Replacement updates all credential fields in one
operation, and cannot resurrect a revoked handle. Revocation erases both secrets.
Use a bounded transport and sender so they cannot indefinitely hold this boundary.

This protocol consumes application-owned custody; it does not duplicate the
storage implementation in pending PR #3443 (#2772). That implementation needs an
adapter providing these additional atomic transaction operations before it can
be wired to this refresher. No integration with an unmerged API is claimed.

The host binds `sender` at refresher construction to a reviewed target. Handle
holders cannot supply a different sender at dispatch time. Preserve audience
binding, grants, approval and other FHIR write controls in the adapter/application;
refresh alone does not authorize clinical effects. The sender receives a bearer
header inside the trusted boundary. Its return value is discarded and exceptions
produce `dispatch_failed`, with no automatic effect retry.

## Grants and validation

- `acquire(handle)` performs a backend-services grant into a caller-created empty
  handle. It refuses existing or revoked handles.
- `ensure(handle)` refreshes an existing credential when its expiry is at or
  before `clock() + refresh_margin` (default 60 seconds). It never implicitly
  acquires into an unknown/empty handle.
- With a refresh token, use a refresh-token grant requesting only the currently
  granted scopes. A returned refresh token atomically supersedes the old secret;
  omission retains the current secret. Without a refresh token, a configured
  backend-services signer can reacquire using client credentials.
- Responses must include a syntactically valid bearer access token,
  case-insensitive `Bearer` token type and integer `expires_in` between 1 and
  `max_lifetime` (default and maximum 86,400 seconds). Boolean, fractional and
  string lifetimes are rejected. Expiry is anchored at request start, and must
  leave more than the configured refresh margin after response arrival, preventing
  repeated successful refreshes with unusably short lifetimes.
- A missing `scope` means the grant's requested scopes; an explicit empty scope
  grants nothing. Malformed scope strings and unexpected scope escalation fail
  closed. SMART v2 resource operations are compared as operation sets using the
  existing scope parser. Other OAuth scopes require exact matching: this slice
  adds no SMART v1, wildcard or granular-scope implication rules.
- `invalid_grant`, malformed responses, unsupported token types, invalid
  lifetimes, transport/signer failures and invalid clocks revoke the handle.
  There is no automatic retry. Reauthorization requires a fresh handle.

## Value-free reports and dispatch

Reports contain only `code`, controlled `findings` and `dropped_scope_count`.
The count measures missing SMART v2 operations plus missing exact-match OAuth
scopes against the original request; neither scope values nor credential values
appear in a report. `usable` describes the credential at the check instant; it
is not a reusable dispatch authorization. Dispatch always rechecks under custody.

For example, a synthetic grant dropping update permission and offline access
produces:

```json
{
  "code": "refreshed",
  "findings": ["scope_narrowed"],
  "dropped_scope_count": 2
}
```

`dispatch(handle, required_scopes="system/SyntheticObservation.u")` then returns
`insufficient_scope` and never invokes the sender. A still-granted create scope
can dispatch. `scope_narrowed` continues to appear on subsequent reports while
the credential remains narrower than the original request.

Custody failures return `custody_unavailable` and locally suppress subsequent
attempts for that handle. A broken adapter that cannot persist revocation cannot
guarantee erasure or exclusion for other clients; such an adapter is unsuitable
for production. No live EHR, live authorization-server compatibility or release
readiness claim follows from the synthetic offline tests.

## Offline validation

```bash
.venv/bin/python -m pytest tests/unit/interop/fhir/test_smart_refresh.py tests/integration/test_smart_credential_refresh.py tests/unit/service/test_smart_backend.py -q
```

Fixtures use synthetic credentials, clocks, custody and a configured mock token
endpoint. Negative controls cover dropped scopes, rotated secrets, failed grants,
malformed responses, sensitive exception payloads and concurrent clients.
