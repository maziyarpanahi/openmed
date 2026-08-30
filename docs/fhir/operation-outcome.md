# Safe FHIR OperationOutcome reports

`openmed.interop.fhir.operation_outcome` turns local validator findings into a
FHIR R4 `OperationOutcome` without copying source diagnostics into the report.
It is a reporting boundary, not a complete FHIR validator: it does not load
models, contact terminology services, or make network calls.

```python
from openmed.interop.fhir.operation_outcome import (
    ValidationFinding,
    build_operation_outcome,
    render_counts,
    render_json,
)

findings = [
    ValidationFinding(
        category="structural",
        path="Patient.name",
        severity="error",
        diagnostics="A source-specific message that must not be echoed.",
    ),
    ValidationFinding(
        category="policy",
        path="Patient.identifier[0].value",
        severity="warning",
        diagnostics="A protected value failed a local policy.",
    ),
]

outcome = build_operation_outcome(findings)
json_text = render_json(findings)
counts_text = render_counts(findings)
```

The default category mapping is:

| Finding category | FHIR `issue.code` |
| --- | --- |
| `structural` | `structure` |
| `policy` | `business-rule` |
| `security` / `privacy` | `security` |
| `terminology` | `code-invalid` |
| `required` | `required` |

An explicit FHIR issue code may be supplied when a finding is more specific.
Expressions are sorted and literal values in predicates or resource IDs are
replaced with `[REDACTED]`. Diagnostics are always replaced with a
category-level message such as `Structural validation failed; details
redacted.`; the supplied diagnostic is never rendered or included in a report
exception.

`render_json()` sorts JSON keys and findings deterministically. Use
`render_counts()` for logs or dashboards that need only `total`, category,
code, and severity counts. The counts-only output contains no paths or
diagnostic text.

This report can identify a failed path and rule family, but it is not a
compliance certification or a clinical decision. Keep validation itself local
and pass only synthetic data to tests and examples.
