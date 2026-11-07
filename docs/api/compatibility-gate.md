# Service API compatibility gate

OpenMed keeps the REST contract in [`openapi.json`](./openapi.json), the
GraphQL schema baseline in
[`baselines/graphql-schema.graphql`](./baselines/graphql-schema.graphql), and
the gRPC descriptor baseline in
[`baselines/openmed-proto-descriptor.json`](./baselines/openmed-proto-descriptor.json).
The compatibility gate compares each committed contract with the live surface
without starting a server or making a network request.

```bash
python -m openmed.service.api_compatibility --json
```

The command checks REST, GraphQL and gRPC by default. Pass `--rest-only` to run
only the OpenAPI comparison.

## REST

The gate checks the client-visible parts of the REST contract:

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

## GraphQL

The live Strawberry schema is rendered to SDL and parsed with `graphql-core`
alongside the baseline. Built-in scalars, directives and introspection types
are ignored.

| Classification | Changes |
| --- | --- |
| Breaking | removed type, field, argument, input field, enum value, union member, implemented interface, directive, directive argument or directive location; changed type kind; changed field, argument or input-field type; output field that became nullable; argument or input field that became non-null; argument or input field that became required (non-null without a default); new required argument, input field or directive argument; removed `repeatable` |
| Additive | new type, field, optional argument, optional input field, enum value, union member, interface, directive, directive location or optional directive argument |
| Compatible | output field that became non-null; argument or input field that became nullable; changed default value |

The exact SDL snapshot at [`graphql-schema.graphql`](./graphql-schema.graphql)
still tracks every change byte-for-byte; the baseline only moves when a change
is accepted.

## gRPC

`openmed/service/proto/openmed.proto` is compiled in-process with
`grpcio-tools` to a `FileDescriptorSet` and reduced to messages, fields, enums,
services, RPC signatures and reservations. Comments, options, default values
and JSON names are not part of the contract.

| Classification | Changes |
| --- | --- |
| Breaking | removed message, enum, service, RPC, field or enum value; renumbered field or enum value; renamed field; changed field type, cardinality, `optional` presence or oneof membership; changed RPC request type, response type or streaming mode; removed reserved number or name; field or enum value that reuses a previously reserved number or name |
| Additive | new message, enum, service, RPC, field, enum value, reserved range or reserved name |

## Report

Each finding has a stable `change` code. REST findings carry `route`,
`schema_path` and `error_category`. GraphQL and gRPC findings carry `surface`,
`type_name` (a GraphQL type, `@directive`, or a fully qualified protobuf
message, enum or service), `member` (field, enum value, union member,
interface, directive location or RPC), `argument`, and the protobuf
`field_number`. Schema values, descriptions, defaults and comments are never
reported.

The JSON report keeps the REST `summary`, `added`, `breaking` and `compatible`
keys. When GraphQL or gRPC is checked it also has a `surfaces` object with the
same keys per surface plus `status`, and the summary adds `total_added`,
`total_breaking` and `skipped_surfaces`.

| Exit code | Meaning |
| --- | --- |
| `0` | No breaking change, and every requested surface was checked. |
| `1` | At least one breaking change. |
| `2` | A contract or baseline is invalid. |
| `3` | A surface was `skipped` because its local toolchain is missing. |

A surface is `skipped` with a controlled `reason` (`graphql_core_unavailable`,
`graphql_runtime_unavailable`, `grpc_tools_unavailable` or
`protobuf_unavailable`) instead of passing silently. Install the `dev` extra
to check every surface, or pass `--allow-skipped` to accept a skipped surface
deliberately.

## Updating the baselines

The baselines represent the accepted GraphQL and gRPC contracts. After an
additive change ships, or when a breaking change has been approved, accept the
current schemas with:

```bash
python -m openmed.service.api_compatibility --write-baselines
```

The writer refuses to overwrite a baseline when the current schema breaks it.
Pass `--allow-breaking` together with `--write-baselines` only for an approved
break, so the removal stays visible in the baseline diff under review.

## Python API

For an in-process check, callers can use
`openmed.service.api_compatibility.check_api_compatibility()` and inspect the
returned `CompatibilityReport`, or call `assert_api_compatibility()` to fail
closed with `APICompatibilityError` when a breaking change is present. Pass
`surfaces=SCHEMA_SURFACES` to include GraphQL and gRPC; a skipped surface also
fails `assert_api_compatibility()` unless `allow_skipped=True`.
`compare_graphql_schemas()` and `compare_proto_contracts()` classify any
baseline and candidate pair directly. All of these APIs are local-only and
deterministic.

The gate is a release-safety check, not a compliance certification or a
clinical decision guarantee.
