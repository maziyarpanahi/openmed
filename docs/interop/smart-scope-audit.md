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
- a user v1 write workflow that also grants delete and over-claims search;
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

## Shared scope grammar

`openmed.interop.smart_scope_grammar` owns parsing and comparison. The Python
scope audit and the current preflight use it directly. Token custody normalizes
private scopes through the same strict adapter and exports only digest-bearing
evidence. OAuth, credential storage and discovery remain outside this change.
No Swift scope-audit API is introduced.

Clinical resource scopes accept patient, user and system contexts, exact resource
names and `*` resource wildcards. Operations normalize in `cruds` order:

| SMART v1 operation | Normalized v2 operations | Meaning |
| --- | --- | --- |
| `read` | `rs` | read and search |
| `write` | `cud` | create, update **and delete** |
| `*` | `cruds` | all five operations |

V2 operation subsets and granular query constraints are supported. Identity
scopes (`openid`, `profile`, `fhirUser`), launch scopes (`launch`,
`launch/patient`, `launch/encounter`) and session scopes (`online_access`,
`offline_access`) match by name and never satisfy clinical permissions. Both
iterables of individual names and OAuth space-delimited scope strings work:

```python
from openmed.interop.smart_scope_audit import audit_smart_scopes

report = audit_smart_scopes(
    workflow_id="synthetic-review",
    required_scopes="openid fhirUser launch/patient offline_access patient/Observation.rs",
    declared_scopes="openid fhirUser launch/patient offline_access patient/Observation.read",
)
assert report.status == "pass"
```

### Conservative coverage

- `patient/*.r` covers `patient/Observation.r`, but is excessive when only that
  specific resource is needed. Specific resources never cover a wildcard need.
- `patient/Observation.rs?category=urn:synthetic|laboratory` **cannot** satisfy
  `patient/Observation.rs`: the grant covers only constrained Observations.
- An unconstrained `patient/Observation.rs` covers the constrained requirement,
  but is reported excessive for that narrower need.
- Constrained scopes match only identical normalized query pairs. Parameter
  order and percent encoding normalize; repeated parameters are preserved.
  Different constraints are not treated as equivalent or as subsets. This
  helper does not execute or infer FHIR search semantics, combine disjoint
  category grants into unrestricted access, or validate server support.
- A v1 `write` grant against a `cu` requirement reports excessive `d` (delete).

### Invalid input and private query values

`parse_smart_scope` now returns either a `SmartScope` or `SmartScopeFinding`.
`normalize_smart_scope` returns a canonical string or the same finding type.
Callers must check the result type rather than relying on a parsing exception:

```python
from openmed.interop.smart_scope_grammar import SmartScopeFinding, parse_smart_scope

parsed = parse_smart_scope("unsupported-synthetic-scope")
assert isinstance(parsed, SmartScopeFinding)
assert parsed.to_dict() == {"reason_code": "unrecognized_scope"}
```

Unknown or malformed entries never confer permissions. The audit returns
`status="invalid"` whenever either input has findings, even if all recognized
clinical needs are covered. Findings contain only `reason_code` (`malformed_scope`
or `unrecognized_scope`), `source` (`required` or `declared`) and zero-based
`index`. For string inputs the index refers to the space-delimited token; for
iterables it refers to the entry. Invalid input text is never echoed. Valid-only
reports retain the existing report keys; invalid reports add `findings`.

A parsed scope's `value` and `constraints` are **private in-memory inputs** and
can contain PHI. Do not log or persist them. `to_dict()`, audit reports and
`evidence_value` replace the complete query (keys and values) with a deterministic
SHA-256 constraint digest. Object representations omit constraints. Digests are
comparison evidence, not anonymization or authorization credentials; low-entropy
values may be guessable. Workflow identifiers must be caller-supplied opaque IDs.

Preflight adapters can use `compare_smart_scopes(required_scopes=...,
declared_scopes=...)` from the shared module to obtain parsed scope sets, missing
and excessive permissions and parse findings, then render their own controlled
reason codes. Every finding must stop or require review; the comparison is an
offline least-privilege check, not authority to execute a clinical operation.

## Pre-run least-privilege check

For workflows that declare launch context or resource wildcards, use the
preflight API. Valid-only `audit_smart_scopes` reports retain their existing
keys. The preflight accepts names or OAuth space-delimited sets;
it does not inspect tokens, endpoints, patient identifiers, or clinical data.

```python
from openmed.interop.smart_scope_audit import audit_smart_scope_preflight

preflight = audit_smart_scope_preflight(
    required_scopes=("patient/Observation.rs", "launch/patient"),
    requested_scopes=("patient/*.r", "launch"),
)
if not preflight.is_least_privilege:
    print(preflight.to_dict())  # Route findings to operator review.
```

The preflight uses the same v1/v2, granular, identity, launch and session grammar.
Findings have stable reason codes:
`missing_scope`, `excessive_scope`, and `overbroad_resource`. A wildcard can
cover a specific resource's required operation while still being reported as
overbroad. Unknown formats and unsupported custom launch contexts produce
value-free `findings`; `is_least_privilege` is false whenever any parse finding
is present. The strict `parse_smart_scope_preflight` adapter still raises a
controlled `ValueError` on malformed intake for existing custody callers. Its
normalized `name` is private and excluded from object representations. Custody
retains granular constraints for permission checks but replaces the complete
query with a constraint digest in `ScopeEvidence`. Any finding requires
operator review; this helper never authorizes a clinical action.

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
task's context while retaining unrelated request logging. Validation failures
raise fresh, fixed errors without retaining private decoder, iterator, endpoint
or callback exception context. Only exact local validation exceptions with a
closed stored code influence the refresh report; callback diagnostic getters
and free-text codes are never used.

Credential readiness carries no clinical-action approval. A governed writer must
separately validate its exact approved action and recheck current custody,
expiry and required scope at the actual effect boundary. The offline integration
tests use a counts-only simulated dispatch with sockets forbidden; they provide
engineering evidence, not live EHR or clinical validation.

The existing bulk-export reader now validates all required SMART token response
fields, rejects narrower grants insufficient for its configured request, and
checks expiry before export, each poll and each new file download. An expired
token stops the run; the legacy reader does not implicitly renew credentials.
Token responses stream into a bounded 64 KiB buffer; an oversized response stops
the read and closes its stream before export or file retrieval can begin.

This behavior follows the published
[SMART Backend Services STU 2.2 token response contract](https://hl7.org/fhir/smart-app-launch/STU2.2/backend-services.html),
[SMART scope syntax](https://hl7.org/fhir/smart-app-launch/STU2.2/scopes-and-launch-context.html),
and [OAuth 2.0 refresh rules](https://www.rfc-editor.org/rfc/rfc6749.html#section-6).
