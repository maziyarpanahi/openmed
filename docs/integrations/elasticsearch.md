# Elasticsearch ingest redaction

`openmed.interop.elasticsearch` builds a native Elasticsearch `redact` ingest
processor for an explicit list of fields. It is dependency-free: creating a
configuration or serializing a pipeline does not import the Elasticsearch
client, open a connection, or make a network request.

## Build a pipeline

Select literal field paths and provide Grok patterns appropriate for the
deployment. A mapping allows each field to use its own patterns:

```python
import json

from openmed.interop.elasticsearch import ElasticsearchRedactionProcessor

processor = ElasticsearchRedactionProcessor(
    fields={
        "message": ("%{EMAILADDRESS:email}",),
        "clinical.note": ("%{IP:client_ip}",),
    }
)

pipeline_body = processor.to_ingest_pipeline()
print(json.dumps(pipeline_body, indent=2, sort_keys=True))
```

The resulting body contains one native `redact` processor per selected field.
Submit it through the deployment’s existing Elasticsearch control plane when
that operation is approved; this adapter never submits it for the caller.
`to_json()` is deterministic and is suitable for a checked-in pipeline
manifest.

The default patterns cover email addresses, IP addresses, and URIs. Use
field-specific patterns for other identifiers and validate them against the
Grok patterns installed in the target cluster. This helper is a configuration
boundary, not a compliance certification or a clinical decision system.

## Static field boundary

Field selection is intentionally fail-closed. Wildcards, template expressions,
array selectors, and empty path segments are rejected. A local processing call
also rejects a selected mapping, list, or other non-string value rather than
walking dynamic data or coercing it to text. Missing, null, and empty selected
fields are skipped when `ignore_missing=True` (the default); unselected fields
are preserved. Configuration field and pattern collections are bounded, as are
pipeline names, markers, and JSON indentation.

For local execution, supply a caller-owned redactor. This keeps model loading,
offline policy, and network policy outside the adapter:

```python
result = processor.process(
    {"message": "synthetic input"},
    redactor=lambda text: text.replace("synthetic input", "[REDACTED]"),
)
redacted_document = result.document
```

The native Elasticsearch pipeline performs the actual server-side redaction;
the injected callable is only for applications that redact before indexing or
for offline tests. Local input documents must contain bounded, JSON-compatible
mappings, lists, tuples, scalar values, and string keys. Cyclic or oversized
documents fail with a stable error before the redactor is called.

## Diagnostics

`result.diagnostics.to_dict()` returns counts only:

```python
{
    "documents_processed": 1,
    "fields_configured": 1,
    "fields_processed": 1,
    "fields_redacted": 1,
    "fields_skipped": 0,
    "spans_redacted": 1,
    "dynamic_fields_rejected": 0,
}
```

The diagnostic summary and result representation do not include source text,
document identifiers, field values, exception details from the redactor, or
mappings. The redacted document remains available explicitly through
`result.document`. Keep request and response-body logging disabled at the
Elasticsearch gateway when ingest documents may contain sensitive text.
