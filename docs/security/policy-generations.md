# Service Security-policy Generations

`openmed.service.policy_generations` applies runtime changes to the REST
service's non-secret security policy as versioned, fully validated
*generations*. A candidate is either activated atomically as the next
generation or rejected while the prior generation stays active. In-flight work
is pinned to one generation, so a request never combines old and new limits,
scopes, or egress rules.

The module is a standalone, opt-in primitive: it does not change how the
service reads its environment at startup, and existing deployments behave
exactly as before until an operator wires a manager in.

## Covered settings

A candidate document must declare every field below. Partial documents are
reported as incomplete; they are never merged with the active generation or
with defaults.

| Section | Field | Startup source | Validator reused |
| --- | --- | --- | --- |
| `limits` | `max_request_bytes` | `OPENMED_SERVICE_MAX_REQUEST_BYTES` | `OperationalLimits` safe range |
| `limits` | `max_text_length` | `OPENMED_SERVICE_MAX_TEXT_LENGTH` | `parse_max_text_length` (strict) |
| `limits` | `rate_limit_rps`, `rate_limit_burst` | `OPENMED_SERVICE_RATE_LIMIT_RPS`, `_BURST` | runtime throttle parsers |
| `limits` | `max_concurrency` | `OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY` | runtime throttle parser |
| `limits` | `concurrency_wait_seconds` | `OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS` | runtime throttle parser |
| `limits` | `throttle_key` | `OPENMED_SERVICE_THROTTLE_KEY` | `parse_service_throttle_key` |
| `authorization` | `enabled`, `deny_by_default` | `OPENMED_SERVICE_AUTH_ENABLED`, `_DENY_BY_DEFAULT` | strict booleans |
| `authorization` | `route_scopes` | `OPENMED_SERVICE_AUTH_ROUTE_SCOPES` | `parse_route_scopes` (declared routes only) |
| `authorization` | `failure_rate_limit_rps`, `_burst`, `_key` | `OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_*` | auth parsers |
| `egress` | `offline` | `OPENMED_OFFLINE` | strict boolean |
| `egress` | `require_tls` | none (defaults to `true`) | strict boolean |
| `egress` | `allowed_purposes` | presence of a privacy-gateway or OTLP endpoint | closed purpose vocabulary |

The egress purpose vocabulary is `fhir_bulk`, `openhim`, `privacy_gateway`,
`remote_inference`, `smart_backend`, `trace_export`, and `webhook_callback`.
`EgressPolicy.allows(purpose, tls=...)` denies when offline, when the purpose
is not allowed or unknown, and when TLS is required but absent.

Out of scope: TLS and mTLS material, API keys, JWKS, JWT issuer, audience and
lifetime settings (verifier rotation), endpoints and headers, CORS origins and
trusted hosts, and the service profile itself. Changing `profile` at runtime is
rejected with `profile_change_unsupported`.

## Bootstrap and reload

```python
from openmed.service.policy_generations import (
    PolicyGenerationManager,
    policy_document_from_environment,
)

# Pass os.environ (the default) in a deployment; this mapping is synthetic.
environ = {
    "OPENMED_PROFILE": "prod",
    "OPENMED_SERVICE_AUTH_ENABLED": "true",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "8",
}
manager = PolicyGenerationManager(policy_document_from_environment(environ))

candidate = manager.active.settings.to_document()
candidate["limits"]["max_request_bytes"] = 2 * 1024 * 1024
result = manager.reload(candidate)
print(result.outcome, result.active_generation)  # activated 2
```

`policy_document_from_environment()` reads only the variables listed above,
through the same parsers the service uses at startup. Credentials, verifier
keys, endpoint values, headers and certificate paths are never copied into the
document; a configured endpoint only contributes its egress purpose. An
unparsable variable raises `PolicyGenerationError("environment_invalid")`
without echoing the value.

`reload()` is `propose()` followed by `activate()` under one lock. Calling
them separately lets an operator inspect a candidate first; activation then
fails with `candidate_stale` if another generation was activated in between.
A candidate whose effective policy equals the active one returns `unchanged`
and does not create a generation.

## Validation as one unit

Every candidate is validated as a whole before activation:

1. Structure: schema version, every section and field present, no unsupported
   keys.
2. Field parsing with the existing startup validators listed above. Unlike the
   startup text-length parser, a reload never silently substitutes a default.
3. Production posture for the `prod` profile: authentication enabled,
   deny-by-default kept, a failed-authentication rate limit, at least one
   admission budget (concurrency cap or rate limit), TLS required for egress,
   and non-empty scopes on every route that declares scopes by default.
4. Injected self-checks. Pass `self_checks=[...]` to plug in a deployment
   self-check; each returns snake_case finding codes, and any finding blocks
   activation. A check that raises reports `self_check_failed`; a free-text
   finding is replaced by `self_check_finding_invalid`.

A rejected candidate never changes the active generation, so a failed reload
cannot weaken readiness. `manager.readiness()` stays `ready: true` and reports
the last outcome and its finding codes.

## Pinning in-flight work

```python
body = b'{"text": "synthetic note"}'

with manager.pin() as pin:
    within_limit = pin.settings.limits.allows_request_bytes(len(body))
    may_forward = pin.settings.egress.allows("privacy_gateway", tls=True)
    print(pin.generation.number, within_limit, may_forward)  # 2 True False
```

Read every decision through `pin.settings`; the pin only changes binding inside
`checkpoint()`, and then as a whole. `PolicyGenerationMiddleware` pins every
HTTP and WebSocket connection, stores the pin in `scope["state"]` under
`openmed_policy_pin`, and releases it after the application returns, including
after the final chunk of a streaming response. `readiness()["inflight"]`
reports pinned work per generation, which supports draining.

## Emergency restrictive changes

`manager.reload(document, emergency=True)` is the tighten-only fast path. The
candidate passes the same full validation and must also be no more permissive
than the active generation in every dimension: lower or equal byte, length,
rate, burst, concurrency and wait limits (a disabled gate counts as
unlimited), authentication and deny-by-default not relaxed, route scopes as
supersets, egress purposes as subsets, offline and TLS requirements not
relaxed, and identical throttle keying. Otherwise it is rejected with
`emergency_not_restrictive`.

Standard reloads affect new work only. Emergency reloads also reach pinned work
at its next declared checkpoint, where `pin.checkpoint()` returns:

| Action | Code | When | What the work does |
| --- | --- | --- | --- |
| `continue` | `policy_pinned` | No emergency since the work was pinned, or the pinned generation is already at least as strict as the active one | Keep using the pinned generation |
| `rebound` | `policy_rebound` | The active generation is no more permissive than the pinned one | Continue under the active generation as a whole |
| `halt` | `policy_continuation_denied` | Neither generation is uniformly stricter | Stop or checkpoint durably; the decision is sticky |

Rebinding never grants pinned work more than it was admitted with. Long-running
work should call `checkpoint()` before each batch, chunk, retry, or outbound
call.

## Diagnostics

Reload results, readiness, history, generation summaries, and every `repr`
contain only generation numbers, digests (`sha256:` over the canonical
effective policy), injected-clock timestamps, counts, fixed finding codes, and
field names from the fixed schema. They never contain environment values,
endpoints, secrets, scope strings, route templates, or other policy text.
Unknown keys are reported as `field_unsupported` without naming the key.

Finding codes: `document_invalid`, `schema_version_unsupported`,
`field_missing`, `field_unsupported`, `field_invalid`, `profile_invalid`,
`route_scopes_invalid`, `egress_purpose_unsupported`, `auth_disabled`,
`deny_by_default_disabled`, `auth_failure_limit_disabled`,
`route_scope_missing`, `admission_unbounded`, `egress_tls_not_required`,
`profile_change_unsupported`, `emergency_not_restrictive`, `candidate_stale`,
`candidate_invalid`, `self_check_failed`, and `self_check_finding_invalid`.

## Validation

```bash
python -m pytest tests/unit/service/test_policy_generations.py -q
```

The tests use synthetic documents, an injected clock, and an in-process ASGI
client. They cover rejected and incomplete candidates, the production posture
rules, stale candidates, emergency continuation outcomes, a multi-threaded
mixed-generation check, and sentinel leak scans.
