# Service Egress Policy

Every outbound connection that the OpenMed service opens goes through one
fail-closed policy in `openmed.service.egress`. The policy authorizes the
operation before the adapter serializes a request body or opens a transport.
An operation with no decision does not run.

The policy covers webhook callbacks, KServe remote inference, SMART backend
services and FHIR Bulk Data, the OpenHIM mediator, the privacy gateway, the
OTLP trace exporter, and MCP OAuth metadata discovery. It runs inside the
process. It does not replace an OS-level or network-level egress control.

## Modes

| Mode | Default for | Behaviour |
| --- | --- | --- |
| `deny` | none (opt-in) | Refuses every outbound operation. |
| `enforce` | `OPENMED_PROFILE=prod`, unset, or any other profile | Allows only operations that match a rule. |
| `permissive` | `OPENMED_PROFILE=dev` and `test` | Also allows HTTPS or gRPCS to public addresses without a rule. |

Set the mode explicitly with `OPENMED_EGRESS_MODE=deny|enforce|permissive`.
`OPENMED_OFFLINE` (or a policy built with `local_only=True`) overrides every
mode and rule. Offline mode is checked again on every attempt, redirect, and
resolution, so switching it on also stops retries that are already running.

## Rules

Rules come from `OPENMED_EGRESS_RULES` (a JSON array) and from the JSON file
named by `OPENMED_EGRESS_RULES_FILE` (an array or `{"rules": [...]}`). When
both are set, the rules from both are combined. An invalid rule stops service
startup with a reason code. The error never repeats the rule's values.

```json
[
  {"purposes": ["webhook"], "destination": "https://hooks.example.org"},
  {
    "purposes": ["smart_token", "fhir_bulk"],
    "destination": "https://fhir.example.org",
    "allow_credentials": true
  },
  {
    "purposes": ["openhim"],
    "destination": "http://openhim-core:8080",
    "cidrs": ["10.20.0.0/16"],
    "allow_credentials": true
  }
]
```

| Field | Meaning |
| --- | --- |
| `purposes` | Required. One or more of `webhook`, `remote_inference`, `smart_token`, `fhir_bulk`, `openhim`, `privacy_gateway`, `trace_export`, `connector_metadata`. |
| `destination` | Required. One exact origin: scheme, host, and optional port. Paths, queries, wildcards, and user info are rejected. Schemes are `http` and `https`, plus `grpc` and `grpcs` for remote inference. |
| `cidrs` | Networks the resolved addresses must fall within. These networks may be private or loopback. Without `cidrs`, a hostname must resolve to public addresses only, and an IP-literal destination allows exactly that address. |
| `allow_credentials` | Allows bearer tokens, basic auth, client assertions, and credential headers to reach this destination. Webhook HMAC signatures are not counted as credentials. |
| `allow_redirects` | Allows a redirect back to this origin. Each hop is authorized again. |
| `max_request_bytes` | Request-body budget for this rule. The default is 16 MiB. |
| `max_timeout_seconds` | Highest per-request deadline allowed for this rule. The default is 300 seconds. |

A rule covers only the purposes it lists. A webhook rule does not authorize
privacy-gateway traffic to the same host. Plain `http` and `grpc` need a rule
that names that scheme.

## Destination checks

The policy resolves the host right before the connection opens. It rejects
loopback, private, link-local, multicast, reserved, unspecified, and cloud
metadata addresses unless a rule's `cidrs` include them. A cloud metadata
address must be listed as a single-address network, for example
`169.254.169.254/32`. If any resolved address is refused, the whole connection
is refused.

The HTTP clients built by the policy connect only to the addresses that passed
the check, while TLS still verifies the configured host name. A DNS answer that
changes between the check and the connection cannot move the socket. These
clients never follow redirects. They also ignore proxy and `.netrc`
environment settings. gRPC channels for remote inference dial the checked
addresses and keep the original name for TLS and `:authority`.

Each logical operation is bound to its destinations and to the first
credential class it used. A retry, redirect, or later attempt cannot reach a
new destination or switch to a different credential class.

## Adapter coverage

| Adapter | Purposes | Decision point | Socket pinning |
| --- | --- | --- | --- |
| `webhooks` | `webhook` | Job submission, then every delivery attempt before the payload is serialized | Pinned client |
| `remote_inference` | `remote_inference` | Transport creation and every inference call; tensor bytes are checked against the budget | Pinned client; gRPC dials checked addresses |
| `smart_backend` (incl. FHIR Bulk) | `smart_token`, `fhir_bulk` | Job submission, then the token, kickoff, status, and file requests | Pinned client |
| `openhim` | `openhim` | Every authentication, registration, and heartbeat request | Pinned client |
| `privacy_gateway` | `privacy_gateway` | Every forwarded prompt, before serialization, with a byte budget | Pinned client |
| `tracing` | `trace_export` | Exporter creation and every export batch | Resolution checked before each batch; the OTLP exporter owns the socket |
| `mcp_metadata` | `connector_metadata` | Every metadata fetch and every redirect hop | Pinned connection |

`tests/unit/service/test_egress_adapter_coverage.py` scans the service source
for outbound call sites. It fails when a new site does not use the egress guard
and is not listed as reviewed. Caller-side SDKs (`openmed.service.client`,
`openmed.service.workflow_client`) and the inbound gRPC server are reviewed
exemptions.

## Denials and receipts

A refusal raises `EgressDenied`, which is a `PermissionError` with a stable
`reason_code`. REST routes that receive a refused destination, such as a job
webhook URL or SMART endpoints, return HTTP 403:

```json
{"error": {"code": "egress_denied", "message": "...", "details": {"reason": "no_matching_rule"}}}
```

When a webhook delivery is refused, it stops with `webhook_delivery.error`
set to `egress_denied` and is not retried. When privacy-gateway forwarding is
refused, the request fails with `privacy_gateway_transport_error` and reason
`egress_denied`.

Every decision produces a value-free receipt. It holds the policy version and
digest, mode, stage, decision, reason code, adapter, purpose, data and
credential classes, an opaque destination digest, the destination class, the
rule identifier, the operation identifier, the attempt number, the request
bytes, and the timing. Receipts are logged on the `openmed.service.egress`
logger: denials at `WARNING` and allowed decisions at `DEBUG`. URLs, hosts,
queries, headers, credentials, and payloads are never recorded.
`EgressPolicy.to_safe_dict()` returns a posture summary for deployment checks.
The summary holds the mode, digest, and rule counts per purpose.

## Migrating

The policy is deny-by-default for the production profile. To keep a webhook,
remote-inference endpoint, SMART server, OpenHIM core, privacy-gateway
endpoint, OTLP collector, or MCP authorization server working, add a rule for
its origin and purpose. Add `allow_credentials` where the adapter sends
credentials. Add `cidrs` where the destination resolves to private addresses.
Webhook URLs and SMART endpoints that no rule allows are rejected when the job
is submitted.
