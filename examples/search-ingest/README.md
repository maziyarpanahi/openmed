# Search-ingest inline redaction sidecar

This example places an OpenMed HTTP processor in front of an
Elasticsearch/OpenSearch-style indexing path. The processor receives the full
ingest-document envelope, redacts only configured string fields in `_source`,
and returns the same envelope for the indexing client or gateway to forward.

All processing stays local after the selected OpenMed model is downloaded. Do
not enable request-body logging in the sidecar, reverse proxy, or indexing
gateway because ingest documents can contain PHI.

## Start the processor

Install the service extra and bind to loopback for local development:

```bash
uv pip install --upgrade "openmed[service]"
uvicorn openmed.integrations.search_ingest_processor:app \
  --host 127.0.0.1 --port 8090
```

Use authenticated TLS between the indexing gateway and sidecar in any remote
deployment. The example service intentionally does not add a second auth model;
deploy it on a private network boundary controlled by the gateway.

## Configure a pipeline

[`pipeline.json`](pipeline.json) is a vendor-neutral gateway manifest. It:

1. registers the `openmed-inline-redaction` HTTP processor;
2. configures the nested `_source` fields and policy profile for that pipeline;
3. routes `clinical-documents-*` index actions through the processor.

Load or translate this manifest in the indexing client, proxy, or ingest gateway
that owns the route. Elasticsearch and OpenSearch run their built-in ingest
processors inside the cluster and do not provide one portable arbitrary-HTTP
processor definition. Therefore, do not submit this manifest directly to
`_ingest/pipeline`; doing so would require a cluster-specific plugin, which is
outside this example's scope.

Field paths are relative to `_source`. Explicit envelope paths such as
`_source.clinical.note` are also supported. A missing path, a non-string value,
or an empty string is left unchanged. Each request supplies its pipeline policy,
so multiple routes can use different OpenMed profiles with the same sidecar.

## Request contract

The pipeline sends the document envelope plus its configured fields and policy:

```bash
curl --fail-with-body http://127.0.0.1:8090/process \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "document": {
    "_index": "clinical-documents-2026",
    "_id": "synthetic-001",
    "_source": {
      "clinical": {"note": "Synthetic patient Jane Roe called 555-0100."},
      "patient": {"summary": "Synthetic follow-up note."},
      "status": "open"
    },
    "_ingest": {"timestamp": "2026-07-19T00:00:00Z"}
  },
  "fields": ["clinical.note", "patient.summary"],
  "policy": "hipaa_safe_harbor"
}
JSON
```

The response is the modified document itself, not a second wrapper. `_index`,
`_id`, `_routing`, `_ingest`, and all non-target `_source` values are preserved.
