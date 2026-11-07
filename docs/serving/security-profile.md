# Service Security Profile

The REST service runs under one of two versioned security profiles, `dev` and
`prod`. Under `prod`, a read-only self-check runs at startup and the service
fails closed when a required control fails: it never becomes ready, skips model
preload and the OpenHIM mediator, and answers every non-probe route with
`503 security_profile_invalid`. Startup, `/readyz`, the CLI preflight and the
deployment images all run the same implementation,
`openmed.service.security_profile`.

The profile is a configuration check, not a certification. It does not
provision certificates or secrets and makes no network calls.

## Which profile applies

`resolve_security_profile()` is the only resolver; service features call it
instead of reading `OPENMED_PROFILE` themselves. It applies the first rule that
matches:

| Setting | Security profile |
|---|---|
| `OPENMED_SERVICE_SECURITY_PROFILE=dev` or `prod` | that profile (any other value is a startup error) |
| `OPENMED_PROFILE=dev`, `test` or `fast` | `dev` |
| `OPENMED_PROFILE` set to anything else (`prod`, `low_resource`, custom names) | `prod` |
| nothing set | `prod` |

An unset or unrecognised configuration profile selects `prod`, so a deployment
must ask for development behavior explicitly. `OPENMED_PROFILE` still chooses
the configuration preset (timeouts, log level, backend); use
`OPENMED_SERVICE_SECURITY_PROFILE` when the preset and the security posture
differ, for example `OPENMED_PROFILE=low_resource` on a development laptop.

The `dev` profile never blocks. It runs the same checks and reports every
finding as advisory, so existing local workflows keep working:

```bash
OPENMED_PROFILE=dev python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

## Preflight

Run the check before starting or deploying the service. It builds the app from
the current environment exactly as the server does, but never starts it:

```bash
openmed service preflight
openmed service preflight --json
openmed service preflight --profile prod --host 0.0.0.0 --no-tls
openmed service preflight --grpc-address '[::]:50051'
```

| Exit code | Meaning |
|---|---|
| `0` | The profile is valid. Advisory findings may be present (`status: warn`). |
| `1` | A required control failed (`status: fail`). |
| `2` | The configuration could not be read, or the `service` extra is missing. |

`--host` and `--tls/--no-tls` describe the listener the service will bind.
Without them the preflight uses `OPENMED_SERVICE_LISTEN_HOST` and
`OPENMED_SERVICE_LISTEN_TLS`, which the reference launcher
(`python -m openmed.service.logging`) exports from its `--host` and
`--ssl-certfile` arguments. Custom ASGI launchers can set those two variables
themselves.

## Readiness

`/readyz` returns the controlled failure until every required control passes:

```json
{
  "error": {
    "code": "security_profile_invalid",
    "message": "Service security profile checks failed; ...",
    "details": {
      "profile": "prod",
      "profile_version": 1,
      "blocking_checks": ["auth.enabled"],
      "report_digest": "sha256:..."
    }
  }
}
```

`/livez`, `/health`, `/metrics`, `/docs` and `/openapi.json` keep answering so
orchestrators can tell a closed service from a dead one. The service logs one
line, `security_profile_invalid profile=prod version=1 blocking=... digest=...`,
when it closes.

## Report

The JSON report (`schema_version: openmed.service.security_profile_report.v1`)
contains the profile name, version, identifier, source and a digest of the
required-check set; an overall `status` (`pass`, `warn` or `fail`) and `ready`
flag; counts; the blocking check ids; and one entry per check with its `id`,
`control`, `level`, `status`, `reason` and, for missing controls, the
`tracking_issue`. A `digest` over the canonical report body identifies the exact
result; record it in release evidence. The preflight digest equals the
`report_digest` that `/readyz` exposes for the same environment and listener.

Reports never contain secrets, key ids, token claims, hosts, origins,
connection strings, paths, request data or clinical values. Check evaluators
return closed status values and reason codes; an evaluator that raises or
returns free text is recorded as `fail` with reason `check_error`.

## Checks in profile v1

A check reports `pass`, `fail`, `not_implemented` or `not_applicable`. Under
`prod`, a `required` check blocks on `fail` or `not_implemented`; an `advisory`
check is reported but never blocks.

`not_implemented` means this build has no implementation of the control while
the feature it guards is enabled. When the feature is disabled the check
reports `not_applicable`. A missing control therefore blocks a production
deployment only when that deployment turns its feature on: for example, setting
`OPENMED_JOURNEY_POSTGRES_DSN` makes `journey.principal_binding` and
`journey.encryption_at_rest` block until their controls ship.

Some surfaces cannot be switched off on this build: GraphQL is always mounted
with GET queries and the GraphiQL IDE, FastAPI serves its API docs, and job and
FHIR routes accept caller-supplied outbound URLs. Their checks are advisory in
v1 so a production deployment stays attainable; the profile version that ships
each control makes its check required. Listener TLS, admission limits and the
served-model allowlist are also advisory in v1, because platforms commonly
terminate TLS and enforce budgets in front of the service.

| Check | Level in `prod` v1 | Passes when | Tracking |
|---|---|---|---|
| `listener.transport` | advisory | listener is loopback, terminates TLS, or sits behind the mTLS proxy | |
| `auth.enabled` | required | `OPENMED_SERVICE_AUTH_ENABLED=true` | |
| `auth.credentials` | required | API-key hashes, a JWKS or mTLS is configured | |
| `auth.jwt_claims` | required | with a JWKS: issuer, audience and a max lifetime of at most 24 hours | |
| `auth.deny_by_default` | required | `OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT` is not disabled | |
| `auth.route_policy` | required | every mounted route has a declared policy | |
| `http.cors` | required | CORS is off, or every origin is an exact `https://` origin | |
| `http.trusted_hosts` | required | the trusted-host allowlist is non-empty and has no `*` | |
| `http.api_docs` | advisory | `/docs`, `/redoc` and `/openapi.json` are not served | #3911 |
| `limits.request_body` | required | global and model-route body limits are installed | |
| `limits.admission` | advisory | a rate or concurrency limit is configured | |
| `models.served_allowlist` | advisory | `OPENMED_SERVICE_SERVED_MODELS` is explicit | |
| `graphql.transport` | advisory | POST-only GraphQL without the IDE | #3544 |
| `graphql.limits` | advisory | GraphQL depth, alias and node budgets | #3545 |
| `journey.principal_binding` | required | Journey access is bound to the authenticated principal | #3549 |
| `journey.encryption_at_rest` | required | Journey storage is encrypted | #3550 |
| `journey.store_tls` | required | the Journey DSN uses `sslmode=verify-full` | |
| `egress.configured_adapters` | required | an egress policy covers the privacy gateway, OTLP and OpenHIM | #3581 |
| `egress.request_targets` | advisory | an egress policy covers webhook and FHIR URLs | #3581 |
| `grpc.transport` | required | with `--grpc-address`: loopback, or TLS for remote binds | #3546 |
| `grpc.auth` | required | with `--grpc-address`: gRPC authentication | #3547 |

## Deployments

The Docker images, the Compose bundle and the Helm chart start the service
through the reference launcher and probe `/readyz`, so they run the same
check. Their images default to `OPENMED_PROFILE=prod`; a deployment becomes
ready only after it configures authentication. With Helm, pass the settings
through `extraEnv`, reading the API-key hashes or JWKS from a Secret:

```yaml
extraEnv:
  - name: OPENMED_SERVICE_AUTH_ENABLED
    value: "true"
  - name: OPENMED_SERVICE_AUTH_API_KEYS
    valueFrom:
      secretKeyRef:
        name: openmed-service-auth
        key: api-keys
```

Deployments that enable Journey storage (the Compose bundle, or Helm with
`journey.enabled`) cannot pass `prod` v1 until Journey principal binding and
encryption at rest ship (#3549, #3550). Use `OPENMED_PROFILE=dev` to evaluate
them locally.

The [deployment posture lane](../deploy/security-posture.md) feeds the
environment the chart and the Compose bundle render into these same checks
before deployment, and adds deployment-only rules for security contexts,
resource limits, readiness wiring, telemetry, development overrides and
worker inheritance. `deploy/posture/helm/production.yaml` is a validated
production overlay for the chart.

## Adding a check

Features register their controls with `register_security_check()`, or replace
a built-in `not_implemented` stub with `replace=True`:

```python
from openmed.service.security_profile import (
    CheckLevel,
    CheckOutcome,
    CheckStatus,
    SecurityCheck,
    SecurityControl,
    register_security_check,
)


def check_example(context):
    if not context.env("OPENMED_EXAMPLE_ENABLED"):
        return CheckOutcome(CheckStatus.NOT_APPLICABLE, "example_disabled")
    return CheckOutcome(CheckStatus.PASS, "example_bounded")


register_security_check(
    SecurityCheck(
        "example.control", SecurityControl.LIMITS, CheckLevel.REQUIRED, check_example
    )
)
```

Bump the profile version when a change alters which checks are required or
what they accept.
