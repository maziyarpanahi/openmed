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

For an opt-in, offline check of profiles declared in `meta.profile`, including
post-de-identification comparison, see
[WHO SMART Guidelines Profile Checks](./fhir-smart-guidelines.md).

## Base R4 Structural Validation

Use `validate_resource()` or `validate_bundle()` before handing an OpenMed
export to a FHIR server. Both functions run entirely offline against a bundled,
minimal table of base FHIR R4 (4.0.1) cardinalities, datatypes, and small fixed
required bindings:

```python
from openmed.clinical.exporters.fhir import validate_bundle, validate_resource

resource_result = validate_resource(
    {
        "resourceType": "Observation",
        "status": "final",
        "code": {"text": "synthetic measurement"},
    }
)
assert resource_result.is_valid

bundle_result = validate_bundle(
    {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "synthetic measurement"},
                }
            }
        ],
    }
)
assert bundle_result.errors[0].location == "Bundle.entry[0].resource.status"
```

`ValidationResult.errors` and `.warnings` contain immutable
`ValidationFinding` objects with `severity`, `location`, `message`, and a FHIR
issue `code`. Messages describe structure only and never quote resource values.
Results also expose `.issues`, so `from_validation_result(result)` can render a
standard R4 `OperationOutcome`.

The bundled subset covers the resources OpenMed emits: `Condition`,
`Observation`, `MedicationStatement`, `Procedure`, `DiagnosticReport`,
`AllergyIntolerance`, `Immunization`, and `Encounter`. A different resource type
produces a `not-supported` warning rather than a false conformance claim. The
constraint table contains only OpenMed's compact derivation of CC0-licensed base
R4 structure and fixed code-system metadata; it does not include clinical
terminology content, proprietary profiles, or implementation-guide packages.

This base validator is intentionally distinct from `check_bundle()`. The latter
loads caller-supplied `StructureDefinition` and `ValueSet` resources to check
declared implementation-guide profiles. Neither checker contacts a terminology
server or replaces the complete HL7 validator for invariants, extensions, and
full profile conformance.
