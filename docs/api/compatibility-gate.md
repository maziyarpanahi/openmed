# Service API compatibility gate

OpenMed keeps the REST contract in [`openapi.json`](./openapi.json). The
compatibility gate compares that checked-in document with the live FastAPI
schema without starting a server or making a network request.

```bash
python -m openmed.service.api_compatibility --json
```

The gate checks the client-visible parts of the contract:

- route and HTTP-method pairs;
- required request-body fields, including nested object fields; and
- stable machine-readable error categories.

Removed operations, newly required fields, newly required request bodies, and
removed error categories are breaking changes. New routes, optional fields, and
new error categories are reported as additions but do not fail the gate. The
report is deterministic and contains only route identifiers, JSON-schema paths,
and stable error codes. It never includes request examples, response examples,
field values, credentials, or model payloads.

The service error envelope currently exposes these stable categories:

`auth_rate_limited`, `authentication_required`, `backpressure`, `bad_request`,
`budget_exceeded`, `capability_error`, `circuit_breaker_open`,
`configuration_error`, `forbidden`, `grounding_invalid_request`,
`inference_error`, `input_error`, `internal_error`, `invalid_credentials`,
`missing_extra`, `model_load_error`, `not_ready`,
`offline_snapshot_unavailable`, `openmed_error`, `policy_error`,
`privacy_gateway_blocked`, `privacy_gateway_error`,
`privacy_gateway_not_configured`, `privacy_gateway_reidentification_error`,
`privacy_gateway_transport_error`, `rate_limited`,
`restricted_terminology_unconfigured`, `service_busy`, `snapshot_invalid`,
`timeout`, and `validation_error`. The taxonomy subset and HTTP/MCP mapping are
documented in [Structured public errors](errors.md).

For an in-process check, callers can use
`openmed.service.api_compatibility.check_api_compatibility()` and inspect the
returned `CompatibilityReport`, or call `assert_api_compatibility()` to fail
closed with `APICompatibilityError` when a breaking change is present. Both
APIs are local-only and deterministic.

The gate is a release-safety check, not a compliance certification or a
clinical decision guarantee.
