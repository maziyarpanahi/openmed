# SMART scope audit examples

OpenMed includes an offline SMART-on-FHIR scope comparison example for local
workflow planning. It compares the resource scopes a synthetic workflow says it
needs with the scopes declared for that workflow and reports missing or
excessive operations.

The example is deliberately local-only:

- it does not implement OAuth;
- it does not contact a FHIR server;
- it does not include endpoints, tokens, launch context values, or patient data;
- it uses synthetic resource names such as `SyntheticObservation`.

## Run the example

```bash
python examples/smart_scope_audit.py
```

The report is deterministic JSON with patient, user, and system context cases:

- a passing read-only patient workflow;
- a patient workflow missing a read scope;
- a user write workflow that over-claims search operations;
- a system read workflow with both missing and excessive scopes.

Each audit lists:

- `required_scopes`: normalized workflow needs;
- `declared_scopes`: normalized declared scopes;
- `missing_scopes`: required operations absent from the declaration;
- `excessive_scopes`: declared operations not needed by the workflow.

## Example shape

```json
{
  "workflow_id": "patient-read-missing",
  "status": "missing",
  "missing_scopes": [
    {
      "scope": "patient/SyntheticCondition.r",
      "context": "patient",
      "resource_type": "SyntheticCondition",
      "operations": [
        {
          "code": "r",
          "name": "read"
        }
      ]
    }
  ]
}
```

This helper is for deterministic local comparison only. It is not a production
permissions recommendation, an OAuth client, or a substitute for deployment
review.

## Pre-run least-privilege check

For workflows that declare launch context or resource wildcards, use the
additive preflight API. The existing `audit_smart_scopes` report above remains
unchanged for exact-resource examples. The preflight accepts only scope names;
it does not inspect tokens, endpoints, patient identifiers, or clinical data.

```python
from openmed.interop.smart_scope_audit import audit_smart_scope_preflight

preflight = audit_smart_scope_preflight(
    required_scopes=("patient/Observation.rs", "launch/patient"),
    requested_scopes=("patient/*.r", "launch"),
)
if not preflight.is_least_privilege:
    print(preflight.to_dict())  # Route findings to operator review.
```

The preflight normalizes `patient`, `user`, and `system` scopes with SMART v2
`c`, `r`, `u`, `d`, and `s` operations. It also recognizes `launch`,
`launch/patient`, and `launch/encounter`. Findings have stable reason codes:
`missing_scope`, `excessive_scope`, and `overbroad_resource`. A wildcard can
cover a specific resource's required operation while still being reported as
overbroad. Unknown formats, SMART v1 operation words, and custom launch
contexts are rejected without echoing the supplied value. Any finding requires
operator review; this helper never authorizes a clinical action.
