# Minimum-data projections for agent tools

`openmed.agent.tools.plan_data_projection` determines which declared tool
fields may be materialized for one clinical workflow. It consumes only a
reviewed input schema, a versioned workflow purpose, and the active access
ticket's granted data classes. It never accepts record contents, invokes a
tool, or performs a network request.

Plan the projection before loading a record or constructing tool arguments.
Only after planning succeeds should trusted local code materialize the returned
field paths.

## Declare field data classes

Start with a schema that passes the minimum-data contract linter. Every
required property also declares one canonical `x-openmed-data-class`:

```python
from openmed.agent.tools import plan_data_projection

purpose = "purpose:org.example/care-summary@1.0.0"
clinical = "data:org.example/clinical-text@1.0.0"
schema = {
    "type": "object",
    "x-openmed-purpose": purpose,
    "properties": {
        "summary": {
            "type": "string",
            "x-openmed-purpose": purpose,
            "x-openmed-minimum-data": "required",
            "x-openmed-data-class": clinical,
        }
    },
    "required": ["summary"],
    "additionalProperties": False,
}

plan = plan_data_projection(
    schema,
    workflow_purpose=purpose,
    granted_data_classes=(clinical,),
)
assert plan.field_paths == ("/summary",)
```

Pass `AccessTicket.permitted_data_classes` only after verifying the ticket for
the active run, purpose, selectors, and tool action. A broader grant does not
broaden the plan: unused granted classes are omitted. If any declared field
class is absent from the grant, the complete tool projection is denied before
materialization.

Nested object paths use JSON Pointer notation. Array item paths include `*`,
for example `/documents/*/text`. Object and array container properties remain
in the plan because they are declared required fields with their own data-class
boundary. Field names must use stable developer-authored identifiers; do not
put patient, clinician, tenant, encounter, or record identifiers in schema
property names.

## Value-free rationale

An approved plan includes a canonical rationale entry for every declared
field. Each entry contains only:

- `include` or `deny`;
- a stable reason code;
- a developer-authored schema path and projected field path; and
- a canonical data-class identifier.

`DataProjectionDeniedError` exposes the same safe rationale when a field class
is not granted. A purpose mismatch produces only the fixed
`purpose_mismatch` reason and the root annotation path; it does not copy either
purpose into the exception or report. Malformed schemas and governance
identifiers fail with fixed validation codes.

Plans and rationales sort and deduplicate their metadata and provide canonical
JSON serialization. Schema descriptions, examples, defaults, runtime values,
and record contents are never retained or rendered. Paths and data classes are
still governance metadata and should be retained only as long as policy
requires.

## Boundary of the planner

The planner enforces declared purpose and data-class scope. It does not decide
clinical necessity, discover PHI, verify an access ticket, fetch records,
validate materialized values against JSON Schema, redact arguments, or prove
that a tool implementation follows its contract. Use it after tool-contract
lint and access-ticket verification, and before local materialization and
grant-aware argument classification.

Run the focused offline tests with:

```text
.venv/bin/python -m pytest tests/unit/agent/tools/test_data_projection.py -q
```
