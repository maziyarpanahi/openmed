# OpenSearch ingest redaction processor

OpenMed provides a small, local-first adapter for a Python-side OpenSearch
ingest bridge. It does not import `opensearch-py`, create a client, or make a
network request. The processor accepts one document mapping and returns a
redacted copy, so a bridge can assign the returned mapping to its ingest
document.

## Configure selected fields

Only explicitly selected fields are processed. Dotted names address nested
objects, and list or tuple fields may contain multiple text values. Missing
fields raise a value-free `OpenSearchRedactionError` by default; set
`ignore_missing=True` when a field is optional.

```python
from openmed.interop.opensearch import OpenSearchRedactionProcessor

processor = OpenSearchRedactionProcessor(
    fields=("message", "clinical.note"),
    policy="hipaa_safe_harbor",
)

document = {
    "message": "Synthetic Person called synthetic-555-0100",
    "clinical": {"note": "Synthetic Person has a follow-up visit."},
    "index_tag": "keep unchanged",
}
redacted_document = processor.execute(document)
```

`execute` is the ingest bridge entry point; `process` is an equivalent name.
The input mapping and all unselected fields remain unchanged. Policy names are
validated against OpenMed's local policy registry when the processor is
constructed.

## Diagnostics and local testing

Use `process_with_report` when a bridge needs processor diagnostics. The
report contains the selected field names, policy, and aggregate counts only;
it never contains source text, redacted text, mappings, or exception details.

```python
redacted_document, report = processor.process_with_report(document)
print(report.to_dict())
```

The default OpenMed deidentifier uses a cache-only configuration, so a missing
model fails locally instead of triggering a download. For a preloaded local
model or an offline test, pass a `deidentifier` callable. It receives `text`,
`policy`, `method`, and any explicit `deidentify_kwargs`, and returns either a
redacted string or an object with a string `deidentified_text` attribute/key.
Redaction failures are converted to the stable message `redaction failed`;
source values are not copied into logs, exceptions, or reports.
