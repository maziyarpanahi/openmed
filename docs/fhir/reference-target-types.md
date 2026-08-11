# FHIR reference target-type checks

OpenMed provides a small, deterministic check for a common export error: a
FHIR reference can have a valid shape while naming a resource type that the
containing field does not permit. The check is local and opt-in. It accepts a
FHIR resource, a Bundle, or an ordered sequence of resources and returns a
FHIR `OperationOutcome`.

```python
from openmed.clinical.exporters.fhir import check_reference_targets

outcome = check_reference_targets(
    [
        {"resourceType": "Patient", "id": "subject-a"},
        {
            "resourceType": "Observation",
            "id": "observation-a",
            "subject": {"reference": "Patient/subject-a"},
        },
    ],
    fhir_version="R4",
)
```

The supported R4 and R5 field maps are exposed as
`FHIR_R4_REFERENCE_TARGETS`, `FHIR_R5_REFERENCE_TARGETS`, and
`REFERENCE_TARGET_ALLOWLISTS`. Only fields in those explicit maps are
inspected. This keeps unknown profiles and unsupported fields non-blocking.
For R5, the checker recognizes `MedicationUsage` and the R5 choice-element
spellings used by this map.

The checker resolves only local `ResourceType/id` references, untyped local IDs
when they identify exactly one resource, and `urn:uuid` references whose
`fullUrl` is present in the supplied Bundle. It does not dereference HTTP URLs,
contact a server, expand terminology, or inspect logical references that use an
`identifier` instead of `reference`. Contained references are reported as an
informational `not-supported` finding so they remain visible without being
treated as an external lookup.

Findings use structural `OperationOutcome.issue.expression` paths and generic
diagnostics:

- `not-found` — no local target was supplied;
- `multiple-matches` — an untyped reference matched more than one local target;
- `structure` — the resolved target type is not allowed for the field;
- `not-supported` — the reference is contained and is outside this local check.

Diagnostics never echo reference strings, resource IDs, identifiers, or other
resource values. An empty result is represented by the standard informational
`No issues detected.` OperationOutcome issue. This helper is an export safety
check, not a complete FHIR validator or a clinical decision guarantee.
