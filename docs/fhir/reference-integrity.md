# FHIR Bundle reference integrity

`openmed.interop.fhir.reference_integrity.reference_integrity_report` performs
an offline structural check for a transformed FHIR Bundle. It supports the
shared Bundle JSON shape used by FHIR R4 and R5 and does not load a validator,
contact a terminology server, or make any network call.

```python
from openmed.interop.fhir.reference_integrity import reference_integrity_report

report = reference_integrity_report(bundle, fhir_version="R4")
if not report.valid:
    print(report.to_json(indent=2))
```

The checker resolves:

- relative `ResourceType/id` references;
- exact references to an entry `fullUrl`;
- absolute URLs whose trailing path identifies an in-Bundle resource; and
- local `#contained-id` references within the containing top-level resource.

The report is counts-only. Findings contain a fixed code, a count, and stable
FHIRPath-style structural paths such as
`Bundle.entry[1].resource.subject.reference`. Raw full URLs, resource ids,
reference strings, and resource payloads are never copied into the report or
included in exceptions. Duplicate counts represent occurrences after the
first occurrence in each duplicate group. `valid` is false for any finding.

The checker is a structural integrity aid, not a complete FHIR validator or a
clinical or compliance certification.
