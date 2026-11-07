# GraphQL service

OpenMed exposes a read-only GraphQL endpoint at `POST /graphql`. It uses the
same service configuration, model loader, warm pool, timeout, retry, and circuit
breaker as the REST endpoints. Install the service extra before starting the
application:

```bash
pip install "openmed[service]"
python -m openmed.service.logging --host 127.0.0.1 --port 8000
```

The schema has four root queries:

- `analyze` runs clinical entity analysis.
- `deidentify` returns redacted text, entities, canonical spans, the selected
  policy profile, and aggregate risk facets.
- `entityTypes` lists canonical labels and their policy categories without
  loading a model.
- `journeyResources` provides policy-filtered Journey metadata.

There are no mutations or subscriptions. When REST authentication is enabled,
GraphQL uses the same principal and authorizes the entire document before
running any resolver: `analyze` requires `analyze:write`, `deidentify` requires
`pii:write`, and `journeyResources` requires `journey:read`. `entityTypes` has
no additional field scope. Aliases, fragment spreads, and mixed-field requests
cannot bypass this check. A denied document returns `data: null` and the
value-free `OPENMED_FORBIDDEN` extension code, without partial results.
A root field missing its explicit scope declaration also fails closed, even
for wildcard principals; standard schema introspection keeps its existing
separate policy. Authentication-disabled local development behavior is unchanged.

## Select only the fields you need

This synthetic example requests only the label and start offset for each span.
The source span text and all other response fields are omitted:

```graphql
query Analyze($input: AnalyzeInput!) {
  analyze(input: $input) {
    spans {
      label
      start
    }
  }
}
```

```json
{
  "input": {
    "text": "Patient Jane Example reports nausea."
  }
}
```

A de-identification query can combine redacted output, policy configuration,
and privacy-safe aggregate risk facets in the same request:

```graphql
query Deidentify($input: DeidentifyInput!) {
  deidentify(input: $input) {
    deidentifiedText
    spans {
      label
      start
      action
    }
    policy {
      name
      defaultAction
    }
    risk {
      leakageRate
      reidentificationRate
      minimumK
    }
  }
}
```

Resolver failures return a generic `OPENMED_RESOLVER_ERROR`. OpenMed suppresses
exception-derived GraphQL messages and the GraphQL execution logger so raw
input text is not copied into resolver errors or logs.

## Introspection and SDL export

Standard GraphQL introspection is enabled. A browser can open `/graphql` for
the GraphiQL interface, and code generators can consume the committed schema
at `docs/api/graphql-schema.graphql`.

Regenerate that artifact after changing the schema:

```bash
.venv/bin/python scripts/export_graphql_schema.py
```

The unit test compares the committed SDL byte-for-byte with the live Strawberry
schema so drift fails CI.

## Document, variable, and execution budgets

Every GraphQL request is checked against a versioned `GraphQLLimits` contract
(`openmed/service/graphql_limits.py`) before any resolver runs, so an oversized
or adversarial document never loads a model or reads Journey resources. The
checks run in a fixed order: document bytes and supplied variables before
parsing, tokens and bracket nesting before the recursive parser, and the
structural budgets before the standard GraphQL validation rules. Fragment
spreads are expanded with multiplicity, iteratively and cycle-safely, so
fragment cycles, alias amplification, and fragment "bombs" fail the same way
every time.

| Budget (`limit`) | Default | Environment variable |
| --- | --- | --- |
| `document_bytes` (UTF-8) | 131072 | `OPENMED_SERVICE_GRAPHQL_MAX_DOCUMENT_BYTES` |
| `tokens` (comments excluded) | 2000 | `OPENMED_SERVICE_GRAPHQL_MAX_TOKENS` |
| `nesting` (`{`, `[`, `(`) | 32 | `OPENMED_SERVICE_GRAPHQL_MAX_NESTING` |
| `nodes` (syntax-tree nodes) | 5000 | `OPENMED_SERVICE_GRAPHQL_MAX_NODES` |
| `operations` | 8 | `OPENMED_SERVICE_GRAPHQL_MAX_OPERATIONS` |
| `fragments` | 64 | `OPENMED_SERVICE_GRAPHQL_MAX_FRAGMENTS` |
| `fragment_cycles` | always 0 | not configurable |
| `variables` (supplied or declared) | 32 | `OPENMED_SERVICE_GRAPHQL_MAX_VARIABLES` |
| `variable_bytes` (compact JSON) | 8388608 | `OPENMED_SERVICE_GRAPHQL_MAX_VARIABLE_BYTES` |
| `variable_depth` | 16 | `OPENMED_SERVICE_GRAPHQL_MAX_VARIABLE_DEPTH` |
| `depth` (fields, after fragments) | 20 | `OPENMED_SERVICE_GRAPHQL_MAX_DEPTH` |
| `aliases` (after fragments) | 16 | `OPENMED_SERVICE_GRAPHQL_MAX_ALIASES` |
| `selected_fields` (complexity) | 1000 | `OPENMED_SERVICE_GRAPHQL_MAX_SELECTED_FIELDS` |
| `deadline` (seconds) | 300 | `OPENMED_SERVICE_GRAPHQL_DEADLINE_SECONDS` |

The defaults admit the full standard introspection query used by schema
exporters and client code generators. An override that is not a number or is
outside its supported range stops the service at startup; the error names the
variable but never echoes its value. Send clinical text as a variable rather
than as an inline string literal: variables have a larger byte budget than the
document itself, and the HTTP body limit still applies to the whole request.

A rejected operation returns HTTP 200, `data: null`, and one error whose
extensions contain only the stable code, the limit category, the configured
ceiling, and the aggregate count that crossed it. Scans that stop early report
the running count at the point of crossing, and expanded counts saturate at one
billion.
The document, variable values, operation names, and field names are never
echoed:

```json
{
  "data": null,
  "errors": [
    {
      "message": "The GraphQL request exceeds a configured limit.",
      "extensions": {
        "code": "OPENMED_GRAPHQL_LIMIT_EXCEEDED",
        "limit": "aliases",
        "maximum": 16,
        "observed": 17
      }
    }
  ]
}
```

The deadline covers body decoding, parsing, validation, resolver execution, and
result cleanup. When it expires the operation is cancelled and the response
carries `OPENMED_GRAPHQL_TIMEOUT` with `limit: "deadline"` and the configured
`maximum`. Cancelled resolvers release their admission permits immediately.
Model work that is already running in a worker thread cannot be interrupted, so
it keeps its concurrency permit, in-flight count, and warm-pool lease until the
worker returns, and then releases all of them.

Batched operation arrays are not supported. A JSON array body is rejected with
HTTP 400 and `OPENMED_GRAPHQL_BATCH_UNSUPPORTED` before any operation is parsed.

When the metrics endpoint is enabled, rejections are counted in
`openmed_service_graphql_rejection_total`, labelled only by `reason` (one of
the limit categories above, plus `batch`).
