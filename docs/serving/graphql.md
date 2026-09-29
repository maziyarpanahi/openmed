# GraphQL service

OpenMed exposes a read-only GraphQL endpoint at `POST /graphql`. It uses the
same service configuration, model loader, warm pool, timeout, retry, and circuit
breaker as the REST endpoints. Install the service extra before starting the
application:

```bash
pip install "openmed[service]"
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8000
```

The schema has three root queries:

- `analyze` runs clinical entity analysis.
- `deidentify` returns redacted text, entities, canonical spans, the selected
  policy profile, and aggregate risk facets.
- `entityTypes` lists canonical labels and their policy categories without
  loading a model.

There are no mutations or subscriptions. Authentication is not part of this
endpoint yet, so place it behind a trusted application boundary until GraphQL
auth middleware is available.

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

## Transport

Operations run over `POST` only. Queries sent as `GET /graphql?query=...` are
rejected with `400` before any resolver or introspection work runs, so a URL
cannot be used to ship query text to the service. Clients that previously used
`GET` must send the same document as a JSON `POST` body; the schema, variables,
and response shape are unchanged.

The interactive GraphiQL IDE is disabled by default and is only served when the
service starts with the explicit development profile (`OPENMED_PROFILE=dev`).
Operations issued from that IDE still use `POST`.

Every `/graphql` response carries `Cache-Control: no-store`,
`Pragma: no-cache`, and `Referrer-Policy: no-referrer` so intermediate caches
and referrer headers cannot retain query text or schema details.

## Introspection and SDL export

Standard GraphQL introspection is enabled over `POST`. The GraphiQL interface
is available at `/graphql` only under the development profile, and code
generators can consume the committed schema at `docs/api/graphql-schema.graphql`.

Regenerate that artifact after changing the schema:

```bash
.venv/bin/python scripts/export_graphql_schema.py
```

The unit test compares the committed SDL byte-for-byte with the live Strawberry
schema so drift fails CI.
