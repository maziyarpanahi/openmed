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

`admission_limited`, `admission_saturated`, `auth_rate_limited`,
`authentication_required`, `backpressure`, `bad_request`,
`budget_exceeded`, `capability_error`, `circuit_breaker_open`,
`client_certificate_revoked`, `configuration_error`, `crl_invalid`, `crl_stale`,
`egress_denied`, `forbidden`, `grounding_invalid_request`,
`idempotency_capacity_exhausted`, `idempotency_key_conflict`,
`idempotency_key_in_progress`, `idempotency_key_invalid`,
`incompatible_state`, `inference_error`, `input_error`, `internal_error`,
`invalid_credentials`, `missing_extra`, `model_load_error`, `not_ready`,
`offline_snapshot_unavailable`, `openmed_error`, `peer_contract_malformed`,
`peer_incompatible`, `policy_error`,
`privacy_gateway_blocked`, `privacy_gateway_error`,
`privacy_gateway_not_configured`, `privacy_gateway_reidentification_error`,
`privacy_gateway_transport_error`, `rate_limited`, `request_body_timeout`,
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

## Packaged JSON Schemas across releases

The checks above compare the code with baselines committed in the same tree,
so a schema property can be deleted and `schema-fingerprints.json` regenerated
in one commit without any of them failing. `scripts/release/schema_release_diff.py`
compares every packaged `*.schema.json` file with the last release instead:

```bash
python scripts/release/schema_release_diff.py
python scripts/release/schema_release_diff.py --base-ref v3.3.0 --json report.json
```

The base defaults to the newest `v*` tag reachable from `HEAD`, found with
`git describe --tags --abbrev=0 --match 'v[0-9]*' HEAD`, and the candidate
defaults to the working tree (`--head-ref` accepts any Git ref). The script
reads both sides straight from Git and never imports `openmed`, so it runs
offline. A shallow clone without tags fails with exit code 2 and asks for the
history; release workflows check out with `fetch-depth: 0`.

Each schema is classified by file name:

| Status | Meaning |
| --- | --- |
| `added` | Present only in the candidate. |
| `removed` | Present at the release and missing from the candidate. |
| `unchanged` | Byte-for-byte the same canonical document. |
| `compatible` | Changed without narrowing and without a version change. |
| `version_bump` | A declared version changed: top-level `schema_version`, the record `schema_version` `const` or `enum`, or the version in `$id`. |
| `breaking` | Narrowed without a version change. |

A change is narrowing when a record the older schema described may no longer
validate, or a field a reader relied on may be gone. That covers removed
properties or definitions, newly required fields, narrowed types, removed enum
values, changed constants, closed `additionalProperties`, new `allOf`
constraints, removed `anyOf` or `oneOf` branches, and tightened bounds,
patterns or formats. Each finding carries a JSON Pointer into the schema.

`breaking` and `removed` schemas fail the check (exit code 1) unless
`scripts/release/schema_break_allowlist.json` has a reviewed entry for them:

```json
{
  "schema_version": 1,
  "entries": [
    {
      "schema": "decision_result",
      "reason": "Why the break is acceptable and who reviewed it.",
      "target_version": "4.1.0"
    }
  ]
}
```

Every entry needs a schema name, a non-empty `reason` and a semantic
`target_version` naming the OpenMed release that ships the break. Allowed
breaks stay in the report with their reason, and entries that no longer match
a break are listed as unused so they can be removed. The
`Model Release Gates` workflow runs the comparison on every manual release
run.

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
