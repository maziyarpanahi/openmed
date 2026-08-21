# FHIR SDC QuestionnaireResponse privacy projection

`openmed.interop.fhir.sdc_privacy` provides a local, deterministic projection
for FHIR SDC `QuestionnaireResponse` resources. It is a field projection, not
a de-identification or clinical-safety certification step: an explicit policy
decides which answer `value[x]` fields may remain in the returned copy.

## Explicit paths

Policy paths are fully indexed, FHIRPath-like locations. The resource prefix is
optional, but every repeated `item` and `answer` element must have an integer
index:

```text
QuestionnaireResponse.item[0].answer[0].valueString
QuestionnaireResponse.item[0].item[1].answer[0].valueCoding
```

Unindexed paths, wildcards, and the generic `value[x]` selector are rejected as
ambiguous. This keeps a policy from silently applying to the wrong repeated
answer. A policy path that is not present in the supplied response is also
rejected instead of being ignored.

## Allow-list projection

The default is fail-closed: only paths explicitly marked `allow` remain. Item
objects, `linkId` values, nested evidence items, and list ordering are retained.
An answer whose values are all dropped is removed from its parent `answer` list;
the parent item remains, with an empty `answer` list when that field was
present.

```python
from openmed.interop.fhir.sdc_privacy import (
    project_questionnaire_response_with_summary,
)

policy = {
    "fields": {
        "item[0].answer[0].valueString": "allow",
        "item[1].answer[0].valueBoolean": "allow",
    }
}

projected, summary = project_questionnaire_response_with_summary(
    questionnaire_response,
    policy,
)
```

The input is never mutated. `summary.to_dict()` contains only counts, a
boolean change flag, and canonical paths such as
`QuestionnaireResponse.item[0].answer[0].valueString`; it never includes an
answer value. A deny-list can be expressed with an explicit default:

```python
policy = {
    "default": "allow",
    "deny": ["item[0].answer[0].valueString"],
}
```

No network call, model, or optional dependency is required by this module.
Malformed response structure and ambiguous policy paths fail closed with
schema/path errors that do not echo response payloads.
