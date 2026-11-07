# Service protocol parity

The OpenMed service answers the same analysis and de-identification requests
over REST, GraphQL and gRPC. [`protocol-map.json`](./protocol-map.json) is the
versioned, machine-readable contract that says which operations are
equivalent, which fields correspond, and how every controlled outcome is
reported on each transport. Every entry is an identifier, status, code, field
path or fixed explanation; the document never contains request content.

## What the map declares

- **Operations.** `analyze`, `extract`, `deidentify` and `stream_deidentify`,
  each with its REST route, GraphQL query field and gRPC method, and the scopes
  every transport requires. A transport that deliberately lacks an operation
  says so with a reason: GraphQL has no extraction field and no streaming
  operation, and neither is added only for symmetry.
- **Fields.** Request fields map one to one (`model_name`, `modelName`,
  `model_name`). Response fields name the path on each transport and how a
  conformance check compares them: offsets and counts as integers, labels and
  actions as codes, text as a SHA-256 digest, keyed span hashes as opaque
  values.
- **Codes.** For each transport, every error code with its status and
  canonical category. REST codes that only routes outside the shared
  operations emit (bulk data, grounding, the privacy gateway and proxy) are
  listed under `rest_only_codes`.
- **Outcomes.** Each controlled outcome, such as `authentication_required`,
  `insufficient_scope`, `rate_limited`, `text_too_long`, `model_not_served`,
  `deadline_exceeded`, `cancelled` or `internal_error`, has one
  transport-neutral category and the REST status and code, GraphQL
  `extensions.code`, and gRPC status with its `openmed-reason` trailer that
  report it. GraphQL requests refused by HTTP middleware before execution use
  the REST envelope, so those GraphQL variants name a `rest_code`. An in-band
  REST NDJSON error event after a stream has started is a `stream_event`
  variant.

The usual shape of each category is below; the outcome entries and the known
divergences give the exact variants.

| Category | REST | GraphQL | gRPC |
| --- | --- | --- | --- |
| `unauthenticated` | 401 | 401, REST envelope | `UNAUTHENTICATED` |
| `permission_denied` | 403 `forbidden` | 200 `OPENMED_FORBIDDEN` | `PERMISSION_DENIED` |
| `invalid_argument` | 400 or 422 | 200, coded or uncoded error | `INVALID_ARGUMENT` |
| `resource_exhausted` | 413, 429 or 503 | REST envelope | `RESOURCE_EXHAUSTED` |
| `deadline_exceeded` | 504 `timeout` | 200 `OPENMED_RESOLVER_ERROR` | `DEADLINE_EXCEEDED` |
| `cancelled` | no response | no response | `CANCELLED` |
| `unavailable` | 503 | REST envelope | not yet reported |
| `internal` | 500 `internal_error` | 200 `OPENMED_RESOLVER_ERROR` | `INTERNAL` |

## Known divergences

Where transports genuinely differ today, the map records the difference with
an owning follow-up instead of changing a contract here:

- `rest_limits_reported_as_validation`: REST reports over-limit text and
  stream chunk sizes as `422 validation_error`.
- `rest_stream_errors_collapse_to_internal`: once a REST stream has started,
  capacity, budget, circuit-breaker and capability failures arrive as
  in-band `internal_error` events.
- `graphql_limits_and_deadlines_uncoded`: GraphQL reports text limits and
  timeouts as `OPENMED_RESOLVER_ERROR` (owned by #3545).
- `graphql_resolver_error_masks_category`: GraphQL collapses validation,
  capacity and availability failures inside a resolver.
- `grpc_unavailable_reported_as_internal`: gRPC reports warm-pool, budget,
  circuit-breaker and capability failures as `internal_error`.
- `graphql_code_casing`: GraphQL mixes `OPENMED_*` codes with the lower-case
  `model_not_served`.

## Drift check

```bash
python -m openmed.service.protocol_map --json
```

The check reads the codes each transport can emit from local source, the gRPC
failure table and methods, the GraphQL query fields, the route scopes, and the
request fields of [`openapi.json`](./openapi.json),
[`graphql-schema.graphql`](./graphql-schema.graphql) and the protobuf
descriptors. It fails, with a report of identifiers, codes and field paths
only, when:

- a transport emits a code the map does not list, or the map lists a code no
  transport emits;
- a gRPC reason's status or category differs from the failure table;
- an outcome variant reports a different category than its outcome without a
  declared divergence, or a declared divergence no longer diverges;
- an operation, scope, GraphQL argument, gRPC method or request field changes
  on one transport and not in the map.

`tests/unit/service/test_protocol_map.py` runs the check in CI. A change that
adds or renames a service error code, extension code, gRPC reason, field or
operation must update `protocol-map.json` in the same pull request and bump
`contract_version`.

## Conformance suite

`tests/integration/test_service_protocol_conformance.py` serves all three
transports in-process over one runtime against the frozen synthetic fixtures
in `tests/fixtures/service/protocol_conformance_v1.json`. It checks that
successful operations normalize to identical spans, offsets, labels and
actions and reach the core with identical arguments; that authentication,
scope, limit, validation, unknown-model, deadline, cancellation, availability
and internal failures are the declared outcomes; that deadlines and
cancellation release warm-pool leases, permits and tasks; and that synthetic
sentinel values never reach logs, traces, metrics, error bodies, gRPC details
or trailers, or the value-free conformance records.

The map and the suite are release-safety checks, not a compliance
certification or a clinical guarantee.
