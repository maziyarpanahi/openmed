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
