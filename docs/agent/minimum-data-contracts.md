# Minimum-data agent tool contracts

`openmed.agent.tool_contract_lint` checks a clinical agent tool's JSON input
schema before registration or review. It is deterministic, local-only, and
value-free: findings contain only a fixed severity, a stable reason code, and a
JSON Pointer path into the schema.

## Annotate an input contract

The root object declares one versioned, canonical purpose identifier with
`x-openmed-purpose`. Every property, including nested properties, repeats that
purpose and declares one `x-openmed-minimum-data` disposition:

- `required`: the field is a necessary, directly supplied input;
- `optional`: the tool accepts the field but does not require it;
- `derived`: the tool could derive the field instead of receiving it.

Only `required` properties that also appear in their containing object's JSON
Schema `required` array pass. Optional and derived inputs expand the accepted
data surface and are blocking findings. A field declared for a different
purpose is also blocking.

```python
from openmed.agent.tool_contract_lint import lint_tool_contract

schema = {
    "type": "object",
    "x-openmed-purpose": "purpose:org.example/care-summary@1.0.0",
    "properties": {
        "question": {
            "type": "string",
            "x-openmed-purpose": "purpose:org.example/care-summary@1.0.0",
            "x-openmed-minimum-data": "required",
        }
    },
    "required": ["question"],
    "additionalProperties": False,
}

report = lint_tool_contract(schema)
assert report.passed
```

For arrays, annotate the array property. Object properties nested below its
`items` schema have their own annotations. Keep input objects closed with
`additionalProperties: false` so runtime schema validation cannot admit fields
outside the reviewed contract.

## Stable findings

All current findings have severity `error` and fail review.

| Reason code | Meaning |
| --- | --- |
| `invalid_schema` | The inspected schema structure is malformed. |
| `missing_purpose` | The root or a field lacks `x-openmed-purpose`. |
| `invalid_purpose` | A purpose is not a versioned canonical purpose ID. |
| `overbroad_for_purpose` | A field's purpose differs from the root purpose. |
| `missing_minimum_data` | A field lacks `x-openmed-minimum-data`. |
| `invalid_minimum_data` | A minimum-data disposition is outside the closed vocabulary. |
| `optional_input` | A field is optional by annotation or JSON Schema structure. |
| `derived_input` | A field is marked as derivable and should not be supplied. |
| `open_input_object` | An object does not set `additionalProperties` to `false`. |
| `unsupported_schema_keyword` | A composition or dynamic-property keyword could hide inputs from this linter. |

Reports are sorted by schema path and reason code, deduplicate identical
findings, and serialize to byte-stable compact JSON:

```json
{"findings":[],"passed":true,"schema_version":"openmed.agent.tool_contract_lint.v1"}
```

The linter never includes annotation values, examples, defaults, descriptions,
or clinical values in a finding, exception, representation, or report. Schema
paths can include developer-authored property names, so property names must not
contain patient, clinician, tenant, or record identifiers.

## Boundary of the check

This check makes a declared minimum-data contract reviewable; it does not
decide whether a field is clinically necessary, authorize access, redact
runtime values, or prove that a tool implementation follows its schema. A
reviewer remains responsible for confirming that each `required` declaration
is appropriate for the stated purpose. Combine the reviewed schema with
purpose-bound access tickets, capability grants, runtime argument
classification, and the later projection layer where applicable.

The first contract version intentionally accepts bounded direct `properties`
trees and array `items` only. `$ref`, tuple schemas, composition keywords,
conditional subschemas, pattern properties, and unevaluated properties fail
closed instead of being partially inspected.
