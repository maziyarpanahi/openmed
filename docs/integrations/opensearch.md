# OpenSearch ingest redaction processor

OpenMed provides a small, local-first adapter for a Python-side OpenSearch
ingest bridge. It does not import `opensearch-py`, create a client, or make a
network request. The processor accepts one document mapping and returns a
redacted copy, so a bridge can assign the returned mapping to its ingest
document.

## Configure selected fields

Only explicitly selected fields are processed. Dotted names address nested
objects, and a list or tuple field may contain flat text or `None` values.
Missing fields raise a value-free `OpenSearchRedactionError` by default; set
`ignore_missing=True` when a field is optional. Documents must use bounded,
JSON-compatible mappings, lists, tuples, scalar values, and string keys; cyclic
or oversized inputs fail with a stable error before redaction starts. The
document boundary accepts at most 10,000 container items, 32 levels, 32 MiB of
UTF-8 key and string payload, and signed 64-bit finite numeric values.

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

The default OpenMed deidentifier enforces a cache-only configuration, disables
mapping and audit retention, and keeps the deterministic safety sweep enabled,
so a missing model fails locally instead of triggering a download. For a
preloaded local model or an offline test, pass a `deidentifier` callable. It
receives `text`, `policy`, `method`, and any explicit `deidentify_kwargs`, and
returns either a redacted string or an object with a string
`deidentified_text` attribute/key. Redaction failures are converted to the
stable message `redaction failed`; source values are not copied into logs,
exceptions, or reports. Deidentifier options are detached data-only snapshots:
up to 64 top-level options, 4,096 nested values, 16 levels, 1 MiB per string or
byte value, and 4 MiB in aggregate. Supported option values are null, booleans,
signed 64-bit integers, finite floats, strings, bytes, lists, tuples, and
string-keyed dictionaries. Callables and other live objects are rejected.
