# Fair Request Admission

OpenMed can bound how much model-backed work each caller holds at once, so one
busy principal cannot use up the service for everyone else. Fair admission
counts in-flight and waiting requests at three levels:

- one global ceiling for the whole service runtime,
- one ceiling per authenticated principal, and
- optionally, one ceiling per namespace (a group of principals).

REST and GraphQL draw on the same counters, so a caller cannot get more than
its share by switching protocol. Fair admission is off until you set a
concurrency ceiling. It is in-process: each replica keeps its own counters.

## Configuration

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `OPENMED_SERVICE_ADMISSION_MAX_CONCURRENCY` | `0` | Global in-flight ceiling. `0` leaves the global level unlimited. |
| `OPENMED_SERVICE_ADMISSION_MAX_QUEUE` | `0` | Global ceiling on waiting requests. |
| `OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_CONCURRENCY` | `0` | In-flight ceiling for each principal. |
| `OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_QUEUE` | `0` | Waiting ceiling for each principal. |
| `OPENMED_SERVICE_ADMISSION_NAMESPACE_MAX_CONCURRENCY` | `0` | In-flight ceiling for each namespace. |
| `OPENMED_SERVICE_ADMISSION_NAMESPACE_MAX_QUEUE` | `0` | Waiting ceiling for each namespace. |
| `OPENMED_SERVICE_ADMISSION_NAMESPACE_CLAIM` | unset | JWT claim that names a principal's namespace. Unset turns namespace ceilings off. |
| `OPENMED_SERVICE_ADMISSION_QUEUE_TIMEOUT_SECONDS` | `5` | Longest a request waits before it is rejected. |
| `OPENMED_SERVICE_ADMISSION_RETRY_AFTER_SECONDS` | `1` | Constant `Retry-After` hint sent with every rejection. |

A level with a concurrency ceiling of `0` is unlimited and its queue setting is
ignored. A level with a ceiling uses its queue setting as the number of
requests that may wait there; `0` means a request that cannot start is
rejected at once. A request may wait only when it fits the queue of every
level that has a ceiling. With the defaults, nothing ever waits.

Example for one replica that serves several API clients:

```bash
OPENMED_SERVICE_ADMISSION_MAX_CONCURRENCY=8 \
OPENMED_SERVICE_ADMISSION_MAX_QUEUE=32 \
OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_CONCURRENCY=4 \
OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_QUEUE=8 \
python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

Keep the principal ceiling below the global ceiling so a busy principal always
leaves room for others, and keep the principal queue well below the global
queue so one principal cannot fill it.

## Principals and namespaces

The principal is the authenticated subject that
[REST authentication](authentication.md) or [mutual TLS](mtls.md) attaches to
the request: the API key's configured principal, the JWT `sub` (or
`client_id`) claim, or the certificate principal. Requests without an
authenticated principal, including every request when authentication is off,
share one anonymous bucket.

When `OPENMED_SERVICE_ADMISSION_NAMESPACE_CLAIM` names a claim (for example
`tenant`), JWT principals whose token carries that claim as a string of at most
128 characters are also counted against that namespace's ceilings. Principals
without the claim, API-key principals and anonymous callers are not
namespaced.

Principal and namespace values are reduced to process-local keyed digests
before they are stored. They never appear in error bodies, headers, metrics or
logs.

## What is admitted

- REST routes whose declared route policy is `model` or `heavy`, such as
  `/analyze`, `/pii/extract`, `/pii/deidentify`, `/brief`, `/v1/decisions`,
  `/omop/load` and `/jobs` submissions. Health, readiness, metrics and other
  control routes are never admitted.
- GraphQL root fields `analyze`, `deidentify` and `journeyResources`. Each
  selected field, aliases included, takes one unit when it executes, so
  batching fields into one document cannot exceed a principal's share.
  `entityTypes` and introspection are free.

An admitted unit is held until the response and any model work the request
started have settled. A request whose HTTP wait times out or is cancelled
keeps its unit while its model work still runs. Cancellation, timeout and
worker failure release each unit exactly once. Streaming endpoints hold their
unit until the stream response starts, the same as the throttle's concurrency
gate.

## Scheduling

Each principal has its own first-in, first-out queue. When capacity frees, it
goes to the next principal in a round-robin rotation whose oldest waiting
request fits its own principal and namespace ceilings. A principal with
waiting work is served within `k - 1` grants to other principals, where `k`
is the number of principals waiting, unless its own ceiling holds it back.
The queue timeout bounds every wait.

A new request from a principal with nothing waiting starts at once when every
level has room. It cannot overtake requests that are waiting for shared
capacity, because freed capacity is handed to the rotation first.

## Rejections

| Status | Error code | `details.reason` | Meaning |
| --- | --- | --- | --- |
| `429` | `admission_limited` | `principal_limit` | The principal is over its own concurrency or queue share. |
| `429` | `admission_limited` | `namespace_limit` | The principal's namespace is over its share. |
| `503` | `admission_saturated` | `global_limit` | The caller is within its share, but the service's shared capacity or queue is full. |

A request that times out while waiting gets the status of the level that held
it back. Every rejection has a `Retry-After` header and `details` of the form
`{"reason": "...", "retry_after_seconds": 1.0}`. The retry hint is the
configured constant; it never reflects current load, and no response reveals
queue depths or other callers' activity.

A GraphQL document with a rejected field returns the same status and
`Retry-After` header with a single error and no partial data:

```json
{
  "data": null,
  "errors": [
    {
      "message": "Request admission limit reached; retry later.",
      "extensions": {"code": "OPENMED_ADMISSION_REJECTED", "reason": "principal_limit"}
    }
  ]
}
```

Clients should honor `Retry-After` and retry with bounded exponential backoff
and jitter.

## Other limits

Fair admission sits between authentication and the route handler, inside the
[throttle](../rest-service.md): authentication runs first, then the throttle's
rate limit and process-wide concurrency gate, then fair admission. The
throttle's concurrency gate is not principal-aware; leave it unset or size it
at or above `OPENMED_SERVICE_ADMISSION_MAX_CONCURRENCY` when you use fair
admission. [Serving backpressure](backpressure.md) for dynamic batching, model
residency limits and async job submission bounds are separate controls and
still apply.

## gRPC

The admission controller is transport-neutral. A gRPC server interceptor can
share the REST and GraphQL budget by calling
`runtime.admission_controller.admit_blocking(ticket)` with
`AdmissionTicket.for_principal(principal, transport="grpc", ...)` and mapping a
rejection to its `grpc_status` (`RESOURCE_EXHAUSTED` or `UNAVAILABLE`) with
`grpc_trailing_metadata()`. The bundled [gRPC server](grpc.md) does not call it
yet, because it does not authenticate callers.

## Metrics

With `OPENMED_SERVICE_METRICS_ENABLED=true`, `/metrics` exports only aggregate
values:

- `openmed_service_principal_admission_inflight`: admitted units held.
- `openmed_service_principal_admission_queued`: requests waiting.
- `openmed_service_principal_admission_active_principals`: principals with
  admitted or waiting requests.
- `openmed_service_principal_admission_rejected_total{reason,transport}`:
  rejections by static reason (`principal_limit`, `namespace_limit`,
  `global_limit`) and transport (`rest`, `graphql`, `grpc`).
