# In-flow record redaction for dataflow tools

`openmed.integrations.dataflow_tool_processor` provides a small callable for
visual dataflow tools and script processors. It accepts one record and its
flow-file attributes, redacts only configured fields through OpenMed's batch
API, and returns exactly one record plus one attribute mapping.

The adapter keeps a model loader in a module-level cache. A long-lived worker
therefore loads a model pipeline once and reuses it across records. Cache keys
include the model and OpenMed configuration, so differently configured flows
do not share a loader.

## Call it from an embedded Python processor

Import the callable at script-module scope so the host keeps the module and its
pipeline cache alive between record callbacks:

```python
from openmed.integrations.dataflow_tool_processor import script_processor


def transform(record, attributes):
    return script_processor(
        record,
        attributes,
        fields=("note", "patient.contact"),
        policy="hipaa_safe_harbor",
        method="mask",
    )
```

For this synthetic input:

```python
record = {
    "record_id": "synthetic-001",
    "note": "Patient Jane Roe called 555-0100.",
    "patient": {"contact": "jane.roe@example.org"},
    "facility": "example-clinic",
}
attributes = {"route": "clinical", "mime.type": "application/json"}

redacted_record, output_attributes = transform(record, attributes)
```

the non-target fields and incoming attributes are preserved. OpenMed adds only
string-encoded counts suitable for flow-file attribute systems:

```json
{
  "openmed.redaction.record_count": "1",
  "openmed.redaction.field_count": "2",
  "openmed.redaction.entity_count": "3"
}
```

The adapter never adds original values, replacement values, entity surfaces,
or field names to attributes. Existing attributes are passed through as-is, so
the flow itself remains responsible for preventing PHI from being placed in
incoming attributes.

Configured fields may be top-level names or dotted mapping paths. Missing,
null, empty, and non-string targets pass through unchanged. A batch failure,
invalid item, or cardinality mismatch fails closed with a diagnostic that does
not include record content.

## Run the persistent JSON-lines entrypoint

Hosts that exchange data over stdin/stdout can keep one Python worker alive and
send one envelope per line. Each envelope must contain a `record` object and
may contain an `attributes` object:

```json
{"record":{"record_id":"synthetic-001","note":"Patient Jane Roe"},"attributes":{"route":"clinical"}}
```

Start the worker after installing OpenMed and caching the selected PII model:

```bash
OPENMED_OFFLINE=1 python -m openmed.integrations.dataflow_tool_processor \
  --field note \
  --field patient.contact \
  --policy hipaa_safe_harbor
```

The worker writes one compact JSON envelope for every non-blank input envelope,
in the same order. Keep stderr on a protected operational channel; errors
contain only a line number and a PHI-free failure category.

## NiFi-style wiring

For a CPython-capable scripted record processor, use the `script_processor`
callback above and map its returned record and attributes to the outgoing
FlowFile. Keep one worker/module instance per processor task to retain the
pipeline cache.

Apache NiFi's `ExecuteScript` Python engine is Jython and cannot load CPython
packages. For NiFi, use
[`ExecuteStreamCommand`](https://nifi.apache.org/components/org.apache.nifi.processors.standard.ExecuteStreamCommand/)
or a long-lived sidecar that runs the module entrypoint. Configure the command
to receive stdin, route zero-status output to `output stream`, and route
`nonzero status` to a protected failure path. If `ExecuteStreamCommand` starts
a fresh process for each FlowFile, batch multiple record envelopes in that
FlowFile or use a resident sidecar so the model is not reloaded per record.

Do not place raw records in command arguments, environment variables,
attributes, provenance annotations, or processor logs. Pass record content
only through stdin or the in-process callable, run with `OPENMED_OFFLINE=1`
after model provisioning, and validate the output relationship before it
reaches an external sink.

This integration is a script adapter only. It does not provide a compiled NiFi
NAR or configure a flow controller.
