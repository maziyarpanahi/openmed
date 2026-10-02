# Agent governance JSON Schemas

`openmed.agent.schemas` exports four Draft 2020-12 JSON Schemas for the
serialized, metadata-only records used by external adapters. The catalog
covers `WorkflowOutcome`, `ActionCorrelation`, `AgentRunTiming`, and
`RunSummary`. It uses their existing public field names and version values.

```python
import json

from openmed.agent.schemas import (
    build_agent_schema,
    build_agent_schema_catalog,
    list_agent_schema_names,
    render_agent_schema,
)

names = list_agent_schema_names()
# ('correlation', 'outcome', 'run_summary', 'timing')
schema = build_agent_schema("run_summary")
schema_json = render_agent_schema("run_summary")
assert json.loads(schema_json) == schema
catalog = build_agent_schema_catalog()
```

Each call creates independent JSON-compatible mappings; rendering uses sorted
keys, ASCII escaping, and compact separators for byte-stable output. Unknown
schema names raise a value-free `ValueError`. The exporter imports no schema
validator, contacts no registry, and performs no I/O. Validation can be done
offline with a caller-supplied Draft 2020-12 validator:

```python
from jsonschema import Draft202012Validator
from openmed.agent import RunSummary

summary = RunSummary.from_events([])
Draft202012Validator.check_schema(schema)
Draft202012Validator(schema).validate(summary.to_dict())
```

Schemas carry distinct versioned `$id` values and only fragment-local `$ref`
links into their own `$defs`. A consumer can register the four documents by
their `$id` values in a local registry; no network retrieval is needed.

The outcome schema binds every `OutcomeClass` to its permitted reason codes.
Correlation IDs use the fixed `run_` and `act_` prefixes with 128-bit lowercase
hex tokens. Run summaries require the exact Python schema version, the complete
closed outcome-count vocabulary, bounded counts, bounded arrays and safe
identifier/digest patterns. Timing describes `AgentRunTiming.to_dict()`:
`run` and `actions` with nonnegative monotonic nanoseconds and optional bounded
opaque identifiers. Timing currently has no version field in its serialized
payload; its catalog `$id` is versioned independently.

The schemas reject extra properties at every object boundary, including
free-text clinical content, prompts, paths, tool arguments, and outputs.
JSON Schema validation checks the serialized shape. Python constructors retain
additional invariants: sorted summary arrays, aggregate count limits, timing
interval arithmetic and parent references, self-parent rejection, and duplicate
JSON-key rejection. Validate with those constructors when these semantic checks
are required. Do not log a validator's raw error object for untrusted input;
it may include the rejected value. No schemas for raw events or content-bearing
records are exported.
