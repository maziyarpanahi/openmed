# SMART scope audit examples

OpenMed includes an offline SMART-on-FHIR scope comparison example for local
workflow planning. It compares the resource scopes a synthetic workflow says it
needs with the scopes declared for that workflow and reports missing or
excessive operations.

The example is deliberately local-only:

- it does not implement OAuth;
- it does not contact a FHIR server;
- it does not include endpoints, tokens, launch context values, or patient data;
- it uses synthetic resource names such as `SyntheticObservation`.

## Run the example

```bash
python examples/smart_scope_audit.py
```

The report is deterministic JSON with patient, user, and system context cases:

- a passing read-only patient workflow;
- a patient workflow missing a read scope;
- a user write workflow that over-claims search operations;
- a system read workflow with both missing and excessive scopes.

Each audit lists:

- `required_scopes`: normalized workflow needs;
- `declared_scopes`: normalized declared scopes;
- `missing_scopes`: required operations absent from the declaration;
- `excessive_scopes`: declared operations not needed by the workflow.

## Example shape

```json
{
  "workflow_id": "patient-read-missing",
  "status": "missing",
  "missing_scopes": [
    {
      "scope": "patient/SyntheticCondition.r",
      "context": "patient",
      "resource_type": "SyntheticCondition",
      "operations": [
        {
          "code": "r",
          "name": "read"
        }
      ]
    }
  ]
}
```

This helper is for deterministic local comparison only. It is not a production
permissions recommendation, an OAuth client, or a substitute for deployment
review.

## Configured SMART credential refresh

`openmed.interop.fhir.SMARTCredentialRefresher` acquires or renews credentials
only through a caller-injected token transport and clock. It supports
`refresh_token` and backend-services `client_credentials` grants. Backend grants
reuse `openmed.service.smart_backend.build_client_assertion` with the injected
clock and an exactly matching `SMARTBackendConfig` endpoint/client binding.
Importing the refresh helpers requires no service extra; signing through the
backend-services builder uses the existing optional service installation.

Configure an HTTPS endpoint, client identity, original requested scope ceiling,
refresh margin and lifetime cap in `SMARTRefreshConfig`. The default lifetime
cap is 300 seconds; a caller may explicitly choose a cap up to 86400 seconds.
The injected transport sends one form-encoded POST, enforces verified TLS and
timeouts, bounds response reads to 64 KiB, and refuses redirects. It returns a
`SMARTTokenResponse` containing HTTP status and raw bytes. No default HTTP
transport, endpoint discovery, interactive launch or retry scheduler is added.
Refresh clients without a private-key JWT supply their configured client ID;
any other confidential-client authentication belongs in the trusted transport.

The small `SMARTCredentialCustody` protocol is independent of any particular
store. Its adapter must implement four atomic operations:

| Operation | Required behavior |
| --- | --- |
| `reserve(handle)` | Exclusively quarantine the credential for every borrower and return its opaque lease, generation, binding digest and private snapshot. |
| `replace(lease, credential)` | Compare the exact lease/version and atomically replace access token, rotated refresh token, expiry and actual scopes together. |
| `release(lease)` | End the reservation while retaining an unchanged healthy credential. The completed lease cannot be reused. |
| `revoke(handle)` | Permanently tombstone the handle, clear secrets and invalidate all active leases and later commits. |

Create handles as `cred_` plus 32 lowercase hex characters and leases as `lease_`
plus 32 lowercase hex characters. Initialize their binding with
`config.binding_digest`; backend acquisition can start with no credential.
Failed or uncertain custody operations must retain quarantine, including after
a lost commit acknowledgement. A lease timeout must never restore the old
secret. A failed revocation acknowledgement reports `custody_unavailable` with
`revocation_confirmed=false`; this is not evidence that the store cleared its
secret. Custody implementations must enforce the protocol rather than relying
on Python types as a security boundary.

Call `ensure_ready(handle, required_scopes=(...))` before dispatch. It refreshes
when remaining lifetime reaches the configured margin, makes one token request,
and never automatically retries a failed refresh. Actual grants control scope
checks. A narrower response may still permit reads while an update request
returns `insufficient_scope` and the value-free `scope_narrowed` finding. OAuth
refresh requests cannot regain permissions dropped from the previous grant.
Rotated refresh tokens replace the old secret atomically; an omitted rotation
retains the existing secret. A missing response scope is accepted only for an
unchanged OAuth refresh grant; backend-services responses require explicit scope.
Opaque refresh tokens and client IDs preserve visible ASCII, including spaces;
they are form values rather than bearer-header values. Bearer access tokens use
the stricter header-safe token syntax. Control characters are rejected.

Bearer type matching is case-insensitive. Lifetime values must be bounded
integers, excluding booleans, and elapsed transport time consumes the lifetime.
Duplicate JSON keys, malformed UTF-8/JSON, oversized/deep responses, scope
expansion, invalid token values and a response already inside the refresh margin
fail closed. `invalid_grant`, invalid responses and uncertain exchanges revoke
the handle; there is no fallback to the previous access token. A backwards or
invalid clock also revokes it.

`smart_scopes_cover` compares SMART v1 read/write and v2 CRUDS permission unions,
resource wildcards and identical query restrictions. It does not interpret FHIR
search predicates: different filters cannot substitute for one another. Scope
values and filters remain private. `audit_granted_smart_scopes` reports only
counts and narrowing/expansion flags. The original synthetic planning helper
above retains its concrete-resource-only behavior and detailed synthetic output.

Refresh reports contain closed codes, counts and flags, with no token, assertion,
endpoint, client identity or scope/filter values. Secret-bearing DTOs suppress
their representations and must never be serialized with generic dataclass
helpers, persisted to evidence, printed or logged. Injected transports, assertion
builders and custody adapters are trusted code responsible for their own logging.
The built-in bulk reader suppresses HTTPX/HTTPcore token-exchange logs in that
task's context while retaining unrelated request logging.

Credential readiness carries no clinical-action approval. A governed writer must
separately validate its exact approved action and recheck current custody,
expiry and required scope at the actual effect boundary. The offline integration
tests use a counts-only simulated dispatch with sockets forbidden; they provide
engineering evidence, not live EHR or clinical validation.

The existing bulk-export reader now validates all required SMART token response
fields, rejects narrower grants insufficient for its configured request, and
checks expiry before export, each poll and each new file download. An expired
token stops the run; the legacy reader does not implicitly renew credentials.

This behavior follows the published
[SMART Backend Services STU 2.2 token response contract](https://hl7.org/fhir/smart-app-launch/STU2.2/backend-services.html),
[SMART scope syntax](https://hl7.org/fhir/smart-app-launch/STU2.2/scopes-and-launch-context.html),
and [OAuth 2.0 refresh rules](https://www.rfc-editor.org/rfc/rfc6749.html#section-6).
