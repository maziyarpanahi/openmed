# Structured access review

`openmed.risk.review_structured_access` compares the fields a workflow
declares for `read` and `export` access with a resource schema and an optional
deny policy. It is a local review aid for integration configuration, not a
compliance certification or a clinical decision guarantee.

```python
from openmed.risk import render_access_review, review_structured_access

report = review_structured_access(
    {
        "triage": {
            "read": {"patient_id", "age"},
            "export": {"diagnosis"},
        }
    },
    {
        "properties": {
            "patient_id": {},
            "age": {},
            "diagnosis": {},
            "notes": {},
        }
    },
    denied_fields={"export": {"diagnosis"}},
)

print(report.to_json())
print(render_access_review(report))
```

The report identifies three classes of review finding for each access mode:

- **Missing** — a declared field is absent from the resource schema.
- **Excessive** — a schema field is not declared for that workflow and mode.
- **Denied** — a declared field is covered by the global or mode-specific deny
  policy.

Schema mapping values are ignored. This keeps examples, defaults, descriptions,
and record-like values out of the report. The report contains only validated
structural field and workflow names, decisions, and counts; invalid names are
rejected without being echoed in exceptions. Inputs are normalized and sorted,
so equivalent declarations produce deterministic JSON and Markdown. The
implementation performs no mandatory network call and does not inspect records.

Review results should be treated as configuration evidence. A complete result
means that no missing, excessive, or denied finding remains; it does not grant
access, prove that an integration enforces the declaration, or certify a
deployment.
