# FHIR Interop Helpers

OpenMed exposes small FHIR R4 helpers for producing resources that downstream
FHIR servers and clients already understand. These helpers are local and
mechanical: they shape data you provide, but they do not call external
validators or network services.

## OperationOutcome

Use `to_operation_outcome()` when an exporter, operation wrapper, or validation
pass needs to report errors, warnings, or informational notes in a FHIR-native
shape.

```python
from openmed.clinical.exporters.fhir import (
    OperationOutcomeIssue,
    to_operation_outcome,
)

outcome = to_operation_outcome(
    [
        OperationOutcomeIssue(
            severity="error",
            code="required",
            diagnostics="Patient.name is required.",
            expression="Patient.name",
        ),
        {
            "severity": "warning",
            "code": "code-invalid",
            "diagnostics": "Unknown LOINC code.",
            "expression": "Observation.code.coding[0]",
        },
    ]
)
```

The returned resource is an R4 `OperationOutcome`:

```python
{
    "resourceType": "OperationOutcome",
    "issue": [
        {
            "severity": "error",
            "code": "required",
            "diagnostics": "Patient.name is required.",
            "expression": ["Patient.name"],
        },
        {
            "severity": "warning",
            "code": "code-invalid",
            "diagnostics": "Unknown LOINC code.",
            "expression": ["Observation.code.coding[0]"],
        },
    ],
}
```

Each issue must use FHIR R4 issue-severity values: `fatal`, `error`, `warning`,
or `information`. Issue codes must come from the R4 issue-type value set, such
as `invalid`, `structure`, `required`, `value`, `invariant`, `processing`,
`business-rule`, `exception`, or `informational`.

When there are no findings, `to_operation_outcome([])` returns a valid all-ok
resource with one informational issue:

```python
{
    "resourceType": "OperationOutcome",
    "issue": [
        {
            "severity": "information",
            "code": "informational",
            "diagnostics": "No issues detected.",
        }
    ],
}
```

For compatibility with older or ad-hoc result objects, `from_validation_result()`
accepts duck-typed shapes such as:

```python
from openmed.clinical.exporters.fhir import from_validation_result

result = {
    "errors": [
        {
            "message": "Malformed Patient resource.",
            "path": "Patient",
        }
    ],
    "warnings": ["Bundle.entry[0] has an unsupported profile."],
}

outcome = from_validation_result(result)
```

`from_validation_result()` is an adapter only. It does not implement structural
validation, US Core conformance checks, or the FHIR `$de-identify` operation.
Those producers should emit issue-like objects and pass them through this shared
builder.

## Privacy Boundary

FHIR `OperationOutcome.issue.diagnostics` is human-readable and may be logged by
servers, clients, gateways, or observability tools. Do not put raw PHI or direct
identifiers in diagnostics. Prefer `expression` paths, offsets, hashes,
provenance identifiers, and risk scores when reporting where a problem occurred.

OpenMed emits R4 `issue.expression` for element paths. It accepts legacy
`location` as input for adapter compatibility, but it never emits
`issue.location` because that field is deprecated in FHIR R4.

## Bundles

Use `to_bundle()` to assemble standalone FHIR resources into a deterministic R4
`Bundle`.

```python
from openmed.clinical.exporters.fhir import to_bundle

bundle = to_bundle(
    [
        {
            "resourceType": "Observation",
            "id": "obs1",
            "status": "final",
            "code": {"text": "Glucose"},
        },
        {
            "resourceType": "DiagnosticReport",
            "id": "report1",
            "status": "final",
            "result": [{"reference": "Observation/obs1"}],
        },
    ],
    doc_id="note-123",
)
```

The helper assigns stable `urn:uuid` `fullUrl` values and rewrites internal
references that point to resources present in the bundle. It does not synthesize
missing resources and does not validate external FHIR profiles.
